from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

for module_name in list(sys.modules):
    if module_name == "verl" or module_name.startswith("verl."):
        sys.modules.pop(module_name, None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verl"))

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _compute_scenecot_group_zscore_credit,
    _expand_visual_obs_span,
    _remove_scenecot_image_range_from_mm_inputs,
    _replace_image_runs_in_visual_span,
    _scenecot_image_block_token_counts,
    _scenecot_image_row_range_for_token_span,
)


class _FakeTokenizer:
    pad_token_id = 0
    vocab_size = 1000
    all_special_ids = [0, 998, 999]

    def encode(self, text: str, add_special_tokens: bool = False):
        mapping = {
            "<obs> no observation available </obs>": [70, 71],
            "<answer>": [40],
            "</answer>": [41],
            "gt": [50],
        }
        return mapping[text]


class _FakeProcessor:
    image_token_id = 999
    video_token_id = 998

    def get_rope_index(self, *, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None):
        seq_len = input_ids.shape[-1]
        base = torch.arange(seq_len, dtype=torch.long).view(1, seq_len)
        content = input_ids.to(torch.long)
        if mm_token_type_ids is not None:
            content = content + mm_token_type_ids.to(torch.long) * 1000
        return torch.stack((base + content, base + content * 2, base + content * 3), dim=0), None


class _FakeGemma3Processor:
    image_token = "<image_soft_token>"
    image_token_id = 999


def _trainer() -> RayPPOTrainer:
    trainer = object.__new__(RayPPOTrainer)
    trainer.processor = _FakeProcessor()
    trainer.tokenizer = _FakeTokenizer()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=1.0))
    )
    return trainer


def _visual_batch(mm_inputs: dict) -> DataProto:
    prompt = [101, _FakeProcessor.image_token_id, _FakeProcessor.image_token_id, 102]
    response_valid = [
        10,
        _FakeProcessor.image_token_id,
        _FakeProcessor.image_token_id,
        _FakeProcessor.image_token_id,
        20,
        30,
        60,
        40,
        61,
        41,
    ]
    response_width = 12
    responses = torch.tensor([response_valid + [0] * (response_width - len(response_valid))], dtype=torch.long)
    prompts = torch.tensor([prompt], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1] * len(prompt) + [1] * len(response_valid) + [0] * (response_width - len(response_valid))],
        dtype=torch.long,
    )

    tool_extra_fields = np.empty(1, dtype=object)
    tool_extra_fields[0] = {
        "scenecot_actions": [{"mode": "highlight"}],
        "scenecot_coac": {
            "tool_called": True,
            "qwen3_counterfactual_supported": True,
            "mode": "highlight",
            "mode_tag_end_pos": 1,
            # The agent loop records the text payload span; visual COAC expands
            # this backward to include the preceding image placeholder tokens.
            "obs_span": (4, 5),
            "obs_context_end_pos": 6,
            "gt_answer_text": "gt",
        },
    }
    multi_modal_inputs = np.empty(1, dtype=object)
    multi_modal_inputs[0] = mm_inputs
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensors={
            "tool_extra_fields": tool_extra_fields,
            "multi_modal_inputs": multi_modal_inputs,
        },
    )


def _mm_inputs_two_images() -> dict:
    return {
        "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]], dtype=torch.long),
        "pixel_values": torch.arange(10, dtype=torch.float32).view(5, 2),
        "images_seqlens": torch.tensor([2, 3], dtype=torch.long),
        # Presence marker: position id code must recompute token types from the
        # new input ids, not reuse a stale sequence-shaped tensor.
        "mm_token_type_ids": torch.ones(1, dtype=torch.long),
    }


def test_visual_coac_masks_observation_image_and_keeps_prompt_image(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_VISUAL_ENABLE", "1")
    trainer = _trainer()
    batch = _visual_batch(_mm_inputs_two_images())

    coac_batch, pairs, metrics, skip_reasons, error_reasons = RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)

    assert pairs == [(0, 0)]
    assert skip_reasons.tolist() == ["built"]
    assert error_reasons.tolist() == [""]
    assert metrics["coac/visual_build_attempted"] == 1
    assert metrics["coac/visual_build_valid"] == 1
    assert metrics["coac/visual_build_failed"] == 0
    assert metrics["coac/visual_build_valid_rate"] == pytest.approx(1.0)

    observed_response = coac_batch.batch["responses"][0].tolist()
    masked_response = coac_batch.batch["responses"][1].tolist()
    assert observed_response.count(_FakeProcessor.image_token_id) == 3
    assert _FakeProcessor.image_token_id not in masked_response
    assert 70 in masked_response and 71 in masked_response
    assert 20 not in masked_response
    assert 30 in masked_response

    mm_rows = coac_batch.non_tensor_batch["multi_modal_inputs"]
    observed_mm = mm_rows[0]
    masked_mm = mm_rows[1]
    assert observed_mm["image_grid_thw"].shape[0] == 2
    assert masked_mm["image_grid_thw"].tolist() == [[1, 1, 2]]
    assert masked_mm["pixel_values"].tolist() == torch.arange(4, dtype=torch.float32).view(2, 2).tolist()
    assert masked_mm["images_seqlens"].tolist() == [2]

    independent_position_ids = RayPPOTrainer._compute_scenecot_position_ids(
        trainer,
        coac_batch.batch["input_ids"][1:2].clone(),
        coac_batch.batch["attention_mask"][1:2].clone(),
        [masked_mm],
    )
    assert torch.equal(coac_batch.batch["position_ids"][1:2], independent_position_ids)


def test_visual_coac_supports_gemma3_native_pixel_values_without_image_grid(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_VISUAL_ENABLE", "1")
    trainer = _trainer()
    trainer.processor = _FakeGemma3Processor()
    original_valid_ids = [
        101,
        _FakeGemma3Processor.image_token_id,
        _FakeGemma3Processor.image_token_id,
        102,
        10,
        _FakeGemma3Processor.image_token_id,
        _FakeGemma3Processor.image_token_id,
        _FakeGemma3Processor.image_token_id,
        20,
        30,
        60,
        40,
        61,
        41,
    ]
    pixel_values = torch.arange(24, dtype=torch.float32).view(2, 3, 2, 2)
    batch = _visual_batch(
        {
            "pixel_values": pixel_values,
            "token_type_ids": torch.tensor(
                [[int(token_id == _FakeGemma3Processor.image_token_id) for token_id in original_valid_ids]],
                dtype=torch.long,
            ),
        }
    )

    coac_batch, pairs, metrics, skip_reasons, error_reasons = (
        RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)
    )

    assert pairs == [(0, 0)]
    assert skip_reasons.tolist() == ["built"]
    assert error_reasons.tolist() == [""]
    assert metrics["coac/visual_build_valid"] == 1
    observed_mm, masked_mm = coac_batch.non_tensor_batch["multi_modal_inputs"]
    assert torch.equal(observed_mm["pixel_values"], pixel_values)
    assert torch.equal(masked_mm["pixel_values"], pixel_values[:1])
    for row_idx, candidate_mm in enumerate((observed_mm, masked_mm)):
        valid_ids = coac_batch.batch["input_ids"][row_idx][
            coac_batch.batch["attention_mask"][row_idx].bool()
        ]
        assert torch.equal(
            candidate_mm["token_type_ids"],
            valid_ids.eq(_FakeGemma3Processor.image_token_id).to(torch.long).unsqueeze(0),
        )


def test_visual_coac_random_tokens_remove_image_features_and_match_span_length(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_VISUAL_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_COUNTERFACTUAL_MODE", "random_tokens")
    trainer = _trainer()
    trainer.global_steps = 9
    batch = _visual_batch(_mm_inputs_two_images())

    coac_batch, pairs, metrics, skip_reasons, error_reasons = (
        RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)
    )

    assert pairs == [(0, 0)]
    assert skip_reasons.tolist() == ["built"]
    assert error_reasons.tolist() == [""]
    observed = coac_batch.batch["responses"][0].tolist()
    counterfactual = coac_batch.batch["responses"][1].tolist()
    assert _FakeProcessor.image_token_id not in counterfactual
    assert sum(token != 0 for token in observed) == sum(token != 0 for token in counterfactual)
    assert metrics["coac/random_token_count"] == 4
    counterfactual_mm = coac_batch.non_tensor_batch["multi_modal_inputs"][1]
    assert counterfactual_mm["image_grid_thw"].tolist() == [[1, 1, 2]]


def test_visual_coac_build_failure_falls_back_without_batch_crash(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_VISUAL_ENABLE", "1")
    trainer = _trainer()
    mm_inputs = _mm_inputs_two_images()
    mm_inputs["image_grid_thw"] = torch.tensor([[1, 1, 2]], dtype=torch.long)
    mm_inputs["pixel_values"] = torch.arange(4, dtype=torch.float32).view(2, 2)
    mm_inputs["images_seqlens"] = torch.tensor([2], dtype=torch.long)
    batch = _visual_batch(mm_inputs)

    coac_batch, pairs, metrics, skip_reasons, error_reasons = RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)

    assert coac_batch is None
    assert pairs == []
    assert skip_reasons.tolist() == ["visual_build_failed"]
    assert error_reasons.tolist() == ["visual_image_token_span_not_aligned:2:3:[2]"]
    assert metrics["coac/visual_build_attempted"] == 1
    assert metrics["coac/visual_build_valid"] == 0
    assert metrics["coac/visual_build_failed"] == 1
    assert metrics["coac/visual_build_valid_rate"] == 0.0


def test_group_zscore_membership_includes_mixed_visual_and_text_valid_rows():
    deltas = np.array([0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5], dtype=np.float32)
    valid = np.ones(8, dtype=bool)
    groups = np.array(["prompt"] * 8, dtype=object)

    _, group_sizes, group_stds, r_tool = _compute_scenecot_group_zscore_credit(deltas, valid, groups)

    assert group_sizes.tolist() == [8] * 8
    assert all(value > 0 for value in group_stds)
    assert all(left > right for left, right in zip(r_tool, r_tool[1:]))


def test_visual_span_expansion_and_image_row_alignment_use_merge_size():
    vision_start = 151652
    vision_end = 151653
    image_pad = 151655
    response = [10, 11, vision_start] + [image_pad] * 920 + [vision_end, 27, 29218, 30405, 8768, 10171, 12]

    span = _expand_visual_obs_span(
        response,
        mode_tag_end_pos=0,
        obs_span=(924, 929),
        obs_context_end_pos=len(response),
        image_token_id=image_pad,
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
    )

    assert span == (2, 929)
    image_start, image_count = _scenecot_image_row_range_for_token_span(
        {"image_grid_thw": torch.tensor([[1, 46, 80]], dtype=torch.long)},
        image_token_offset=0,
        image_token_count=920,
        merge_size=2,
    )
    assert (image_start, image_count) == (0, 1)


def test_visual_cached_embeds_are_sliced_by_cache_feature_count_not_expanded_pad_count():
    vision_start = 151652
    vision_end = 151653
    image_pad = 151655
    response = [10, vision_start] + [image_pad] * 750 + [vision_end, 11]
    counts = _scenecot_image_block_token_counts(
        response,
        image_token_id=image_pad,
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
    )
    assert counts == [750]

    resized, resized_span, resized_context_end = _replace_image_runs_in_visual_span(
        response,
        span=(1, len(response) - 1),
        image_token_id=image_pad,
        target_image_token_counts=[120],
        obs_context_end_pos=len(response),
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
    )
    assert resized.count(image_pad) == 120
    assert resized_span == (1, 123)
    assert resized_context_end == 124

    mm_inputs = {
        # Deliberately mismatched with the expanded pad count, mirroring cached
        # highlight observations from the diagnostic dump: the response can
        # contain hundreds of image_pad tokens while the cached Qwen3-VL
        # embeddings already live in pooled feature-token space.
        "image_grid_thw": torch.tensor([[1, 18, 28]], dtype=torch.long),
        "image_embeds": torch.arange(120 * 2, dtype=torch.float32).view(120, 2),
        "deepstack_image_embeds": [
            torch.arange(120 * 3, dtype=torch.float32).view(120, 3),
        ],
        "images_seqlens": torch.tensor([504], dtype=torch.long),
    }
    masked = _remove_scenecot_image_range_from_mm_inputs(
        mm_inputs,
        image_start=0,
        image_count=1,
        image_token_counts=counts,
    )

    assert "image_grid_thw" not in masked
    assert "image_embeds" not in masked
    assert "deepstack_image_embeds" not in masked
    assert "images_seqlens" not in masked
