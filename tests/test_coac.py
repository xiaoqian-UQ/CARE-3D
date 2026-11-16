from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

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
    _build_scenecot_random_token_ids,
    _build_scenecot_coac_response_variants,
    _check_scenecot_step1_format_reward_sentinel,
    _compute_scenecot_group_zscore_credit,
    _compute_train_scenecot_reward_metrics,
    _drop_stale_scenecot_reward_extra_for_recompute,
    _remove_scenecot_image_range_from_mm_inputs,
)

from scenecot_verl.rewards.scenecot_reward import compute_coac_tool_reward


def test_coac_teacher_forcing_excludes_reasoning_tokens():
    mode_tag = [10, 11]
    tool_prefix = [12, 13]
    observation = [20, 21, 22]
    instruction_and_template = [23, 24, 25, 26]
    reasoning = [30, 31, 32]
    answer_open = [40]
    gt_answer = [50, 51]
    answer_close = [41]
    generated_answer = [60, 61]
    placeholder = [70, 71]

    response_valid = (
        mode_tag
        + tool_prefix
        + observation
        + instruction_and_template
        + reasoning
        + answer_open
        + generated_answer
        + answer_close
    )
    mode_tag_end_pos = len(mode_tag)
    obs_span = (
        mode_tag_end_pos + len(tool_prefix),
        mode_tag_end_pos + len(tool_prefix) + len(observation),
    )
    obs_context_end_pos = obs_span[1] + len(instruction_and_template)

    observed, masked = _build_scenecot_coac_response_variants(
        response_valid,
        mode_tag_end_pos=mode_tag_end_pos,
        obs_span=obs_span,
        obs_context_end_pos=obs_context_end_pos,
        answer_open_ids=answer_open,
        gt_ids=gt_answer,
        close_ids=answer_close,
        placeholder_ids=placeholder,
    )

    assert observed == mode_tag + tool_prefix + observation + instruction_and_template + answer_open + gt_answer + answer_close
    assert masked == mode_tag + tool_prefix + placeholder + instruction_and_template + answer_open + gt_answer + answer_close
    assert not any(token in observed for token in reasoning)
    assert not any(token in masked for token in reasoning)
    assert not any(token in masked for token in observation)
    observed_answer_pos = len(mode_tag) + len(tool_prefix) + len(observation) + len(instruction_and_template)
    masked_answer_pos = len(mode_tag) + len(tool_prefix) + len(placeholder) + len(instruction_and_template)
    assert observed[observed_answer_pos : observed_answer_pos + len(answer_open)] == answer_open
    assert masked[masked_answer_pos : masked_answer_pos + len(answer_open)] == answer_open


def test_coac_masked_sequence_keeps_instruction_but_removes_observation_payload():
    token_text = {
        1: "<list_obj_prob>",
        2: "<tool_response>\n",
        3: "<obj_prob> chair 0.90 chair 0.80 </obj_prob>",
        4: "\n<scenecot_instruction>Use the observation.</scenecot_instruction>\n",
        5: "</tool_response>\nassistant\n",
        6: "<reasoning>chair chair chair</reasoning>",
        7: "<answer>",
        8: "gt",
        9: "</answer>",
        10: "<obs> no observation available </obs>",
        11: "generated",
    }
    response_valid = [1, 2, 3, 4, 5, 6, 7, 11, 9]

    _, masked = _build_scenecot_coac_response_variants(
        response_valid,
        mode_tag_end_pos=1,
        obs_span=(2, 3),
        obs_context_end_pos=5,
        answer_open_ids=[7],
        gt_ids=[8],
        close_ids=[9],
        placeholder_ids=[10],
    )

    masked_text = "".join(token_text[token] for token in masked)
    assert "<scenecot_instruction>Use the observation.</scenecot_instruction>" in masked_text
    assert "<obj_prob>" not in masked_text
    assert "<reasoning>" not in masked_text
    assert masked_text.endswith("<answer>gt</answer>")


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


def _coac_test_trainer() -> RayPPOTrainer:
    trainer = object.__new__(RayPPOTrainer)
    trainer.processor = _FakeProcessor()
    trainer.tokenizer = _FakeTokenizer()
    trainer.global_steps = 7
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=1.0))
    )
    return trainer


def _text_coac_batch(*, coac_metadata: dict) -> DataProto:
    prompt = [101, 102]
    response_valid = [10, 11, 20, 21, 22, 30, 60, 40, 61, 41]
    response_width = 18
    tool_extra_fields = np.empty(1, dtype=object)
    tool_extra_fields[0] = {
        "scenecot_actions": [{"mode": "prob"}],
        "scenecot_coac": {
            "tool_called": True,
            "qwen3_counterfactual_supported": True,
            "mode": "prob",
            "mode_tag_end_pos": 1,
            "obs_span": (2, 5),
            "obs_context_end_pos": 6,
            "gt_answer_text": "gt",
            **coac_metadata,
        },
    }
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([prompt], dtype=torch.long),
            "responses": torch.tensor(
                [response_valid + [0] * (response_width - len(response_valid))],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1] * len(prompt) + [1] * len(response_valid) + [0] * (response_width - len(response_valid))],
                dtype=torch.long,
            ),
        },
        non_tensors={
            "tool_extra_fields": tool_extra_fields,
            "uid": np.array(["sample-a"], dtype=object),
            "rollout_n": np.array([3], dtype=object),
        },
    )


def test_coac_random_tokens_are_deterministic_length_matched_and_qwen3_only(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_COUNTERFACTUAL_MODE", "random_tokens")
    monkeypatch.setenv("SCENECOT_COAC_RANDOM_SEED", "123")
    trainer = _coac_test_trainer()
    batch = _text_coac_batch(coac_metadata={})

    coac_batch, pairs, metrics, skip_reasons, error_reasons = (
        RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)
    )

    assert pairs == [(0, 0)]
    assert skip_reasons.tolist() == ["built"]
    assert error_reasons.tolist() == [""]
    expected_random = _build_scenecot_random_token_ids(
        trainer.tokenizer,
        length=3,
        seed_parts=(123, 7, "sample-a", 3, 0),
        forbidden_token_ids={
            0,
            40,
            41,
            70,
            71,
            _FakeProcessor.image_token_id,
            _FakeProcessor.video_token_id,
        },
    )
    observed = coac_batch.batch["responses"][0].tolist()
    counterfactual = coac_batch.batch["responses"][1].tolist()
    assert counterfactual[2:5] == expected_random
    assert sum(token != 0 for token in observed) == sum(token != 0 for token in counterfactual)
    assert metrics["coac/counterfactual_random_tokens"] == 1
    assert metrics["coac/random_token_count"] == 3


def test_coac_random_tool_uses_agent_supplied_alternate_qwen3_context(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_COUNTERFACTUAL_MODE", "random_tool")
    trainer = _coac_test_trainer()
    batch = _text_coac_batch(
        coac_metadata={
            "random_tool_mode": "loc_rect",
            "random_tool_prefix_ids": [90, 91],
            "random_tool_response_ids": [92, 93, 94],
        }
    )

    coac_batch, pairs, metrics, skip_reasons, error_reasons = (
        RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)
    )

    assert pairs == [(0, 0)]
    assert skip_reasons.tolist() == ["built"]
    assert error_reasons.tolist() == [""]
    counterfactual = coac_batch.batch["responses"][1].tolist()
    assert counterfactual[:8] == [90, 91, 92, 93, 94, 40, 50, 41]
    assert metrics["coac/counterfactual_random_tool"] == 1
    assert metrics["coac/random_tool_count"] == 1
    assert metrics["coac/random_tool_mode/loc_rect/count"] == 1


def test_coac_random_tool_supports_cached_highlight_counterfactual(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_COUNTERFACTUAL_MODE", "random_tool")
    monkeypatch.setenv("SCENECOT_COAC_VISUAL_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_VISUAL_CACHE_DIR", "/fake/cache")
    trainer = _coac_test_trainer()

    class _FakeVisualCache:
        def load_images(self, paths):
            assert paths == ["/images/random-highlight.jpg"]
            return {
                "image_embeds": torch.ones((2, 4), dtype=torch.float32),
                "deepstack_image_embeds": [torch.ones((2, 3), dtype=torch.float32)],
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
                "images_seqlens": torch.tensor([2], dtype=torch.long),
            }

    trainer._scenecot_coac_visual_cache = _FakeVisualCache()
    trainer._scenecot_coac_visual_cache_root = "/fake/cache"
    batch = _text_coac_batch(
        coac_metadata={
            "random_tool_mode": "highlight",
            "random_tool_prefix_ids": [90],
            "random_tool_response_ids": [999, 999, 20, 30],
            "random_tool_payload_span": (2, 3),
            "random_tool_image_path": "/images/random-highlight.jpg",
        }
    )

    coac_batch, pairs, metrics, skip_reasons, error_reasons = (
        RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)
    )

    assert pairs == [(0, 0)]
    assert skip_reasons.tolist() == ["built"]
    assert error_reasons.tolist() == [""]
    assert coac_batch.batch["responses"][1].tolist().count(999) == 2
    random_tool_mm = coac_batch.non_tensor_batch["multi_modal_inputs"][1]
    assert random_tool_mm["image_grid_thw"].tolist() == [[1, 1, 2]]
    assert metrics["coac/random_tool_mode/highlight/count"] == 1


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


class _FakeLlavaProcessor:
    image_token_id = 999


class _FakeGemma3Processor:
    image_token = "<image_soft_token>"
    image_token_id = 999


def test_coac_rebuilds_gemma3_token_type_ids_for_each_candidate():
    trainer = _coac_test_trainer()
    trainer.processor = _FakeGemma3Processor()
    batch = _text_coac_batch(coac_metadata={})
    batch.batch["prompts"][0, 0] = _FakeGemma3Processor.image_token_id
    mm_inputs = np.empty(1, dtype=object)
    mm_inputs[0] = {
        # Deliberately stale: the original rollout length differs from both
        # teacher-forced COAC candidates.
        "token_type_ids": torch.ones((1, 99), dtype=torch.long),
    }
    batch.non_tensor_batch["multi_modal_inputs"] = mm_inputs

    coac_batch, pairs, _, _, _ = RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)

    assert pairs == [(0, 0)]
    for row_idx in range(2):
        valid_ids = coac_batch.batch["input_ids"][row_idx][
            coac_batch.batch["attention_mask"][row_idx].bool()
        ]
        candidate_mm = coac_batch.non_tensor_batch["multi_modal_inputs"][row_idx]
        assert torch.equal(
            candidate_mm["token_type_ids"],
            valid_ids.eq(_FakeGemma3Processor.image_token_id).to(torch.long).unsqueeze(0),
        )


def test_visual_coac_does_not_slice_sequence_shaped_gemma3_token_types():
    token_types = torch.tensor([[0, 1, 1, 0]], dtype=torch.long)

    trimmed = _remove_scenecot_image_range_from_mm_inputs(
        {
            "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
            "token_type_ids": token_types,
        },
        image_start=0,
        image_count=1,
        image_token_counts=[2],
    )

    assert torch.equal(trimmed["token_type_ids"], token_types)


def test_coac_llava_position_ids_use_standard_one_dimensional_rope():
    trainer = object.__new__(RayPPOTrainer)
    trainer.processor = _FakeLlavaProcessor()
    input_ids = torch.tensor(
        [
            [0, 101, _FakeLlavaProcessor.image_token_id, 102],
            [201, 202, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    position_ids = RayPPOTrainer._compute_scenecot_position_ids(
        trainer,
        input_ids,
        attention_mask,
        [{}, {}],
    )

    assert position_ids.shape == attention_mask.shape
    assert torch.equal(
        position_ids,
        torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    )


def test_coac_masked_position_ids_are_recomputed_for_masked_sequence():
    trainer = object.__new__(RayPPOTrainer)
    trainer.processor = _FakeProcessor()

    prompt = [101, 102]
    observed_response = [10, 11, 20, 21, 40, 50, 41]
    masked_response = [10, 11, 70, 71, 72, 40, 50, 41]
    width = len(prompt) + len(masked_response)

    def padded_input(response: list[int]) -> tuple[list[int], list[int]]:
        valid = prompt + response
        return valid + [0] * (width - len(valid)), [1] * len(valid) + [0] * (width - len(valid))

    observed_ids, observed_attn = padded_input(observed_response)
    masked_ids, masked_attn = padded_input(masked_response)
    input_ids = torch.tensor([observed_ids, masked_ids], dtype=torch.long)
    attention_mask = torch.tensor([observed_attn, masked_attn], dtype=torch.long)
    mm_inputs = [{"mm_token_type_ids": torch.ones(1)}, {"mm_token_type_ids": torch.ones(1)}]

    batched_position_ids = RayPPOTrainer._compute_scenecot_position_ids(
        trainer,
        input_ids,
        attention_mask,
        mm_inputs,
    )
    independent_masked_position_ids = RayPPOTrainer._compute_scenecot_position_ids(
        trainer,
        input_ids[1:2].clone(),
        attention_mask[1:2].clone(),
        [mm_inputs[1]],
    )

    assert torch.equal(batched_position_ids[1:2], independent_masked_position_ids)


def test_coac_skips_visual_placeholder_observation_span():
    trainer = object.__new__(RayPPOTrainer)
    trainer.processor = _FakeProcessor()
    trainer.tokenizer = _FakeTokenizer()

    prompt_width = 2
    response_width = 8
    mode_tag = [10, 11]
    observation_with_image = [_FakeProcessor.image_token_id, 20]
    response_valid = mode_tag + observation_with_image + [30, 40, 60, 41]
    responses = torch.tensor([response_valid + [0] * (response_width - len(response_valid))], dtype=torch.long)
    prompts = torch.tensor([[101, 102]], dtype=torch.long)
    attention_mask = torch.tensor([[1] * prompt_width + [1] * len(response_valid) + [0] * (response_width - len(response_valid))])

    tool_extra_fields = np.empty(1, dtype=object)
    tool_extra_fields[0] = {
        "scenecot_coac": {
            "tool_called": True,
            "mode_tag_end_pos": len(mode_tag),
            "obs_span": (len(mode_tag), len(mode_tag) + len(observation_with_image)),
            "gt_answer_text": "gt",
        }
    }
    batch = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensors={"tool_extra_fields": tool_extra_fields},
    )

    coac_batch, pairs, metrics, skip_reasons, error_reasons = RayPPOTrainer._build_scenecot_coac_batch(trainer, batch)

    assert coac_batch is None
    assert pairs == []
    assert metrics["coac/build_visual_obs_skipped"] == 1
    assert skip_reasons.tolist() == ["visual_obs_mode"]
    assert error_reasons.tolist() == ["visual_observation_span"]


def test_train_metrics_include_coac_fallback_credit_rate():
    metrics = _compute_train_scenecot_reward_metrics(
        {
            "coac/fallback_credit_rate": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            "coac_fixed_fallback_applied": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            "coac_valid": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "coac_r_tool": np.array([0.5, 0.8, 0.5], dtype=np.float32),
            "coac_tool_credit_source": np.array(["fixed_fallback", "coac_delta", "fixed_fallback"], dtype=object),
        }
    )

    assert metrics["coac/fallback_credit_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["coac/fixed_fallback_applied_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["coac/valid_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["coac/r_tool_mean"] == pytest.approx(0.6)
    assert metrics["train/scenecot/coac_tool_credit_source/coac_delta/count"] == 1
    assert metrics["train/scenecot/coac_tool_credit_source/fixed_fallback/count"] == 2


def test_train_metrics_include_answer_judge_rates_and_statuses():
    metrics = _compute_train_scenecot_reward_metrics(
        {
            "answer_judge_called": np.array([1, 1, 0, 0], dtype=np.float32),
            "answer_judge_fallback": np.array([0, 1, 0, 0], dtype=np.float32),
            "answer_judge_score_1_to_5": np.array([5, np.nan, np.nan, np.nan], dtype=np.float32),
            "answer_judge_status": np.array(
                ["gemini", "fallback_rule_on_error", "em", "protocol_invalid"],
                dtype=object,
            ),
            "answer_judge_label": np.array(["correct", None, None, None], dtype=object),
        }
    )

    assert metrics["train/scenecot/answer_judge_called/mean"] == pytest.approx(0.5)
    assert metrics["train/scenecot/answer_judge_fallback/mean"] == pytest.approx(0.25)
    assert metrics["train/scenecot/answer_judge_score_1_to_5/mean"] == pytest.approx(5.0)
    assert metrics["train/scenecot/answer_judge_status/gemini/rate"] == pytest.approx(0.25)
    assert metrics["train/scenecot/answer_judge_status/fallback_rule_on_error/rate"] == pytest.approx(0.25)
    assert metrics["train/scenecot/answer_judge_label/correct/count"] == 1


def test_drop_stale_reward_extra_keeps_coac_recompute_inputs():
    batch = SimpleNamespace(
        non_tensor_batch={
            "coac_delta": np.array([0.2, 0.3], dtype=np.float32),
            "coac_valid": np.array([True, False]),
            "coac_obs_text": np.array(["red chair", "blue table"], dtype=object),
            "coac_r_tool_override": np.array([0.7, 0.5], dtype=np.float32),
            "coac_r_tool": np.array([0.5, 0.5], dtype=np.float32),
            "coac/fallback_credit_rate": np.array([1.0, 1.0], dtype=np.float32),
            "coac_copy_overlap": np.array([0.0, 0.0], dtype=np.float32),
            "format_reward": np.array([1.0, 1.0], dtype=np.float32),
        },
        meta_info={
            "reward_extra_keys": [
                "coac_delta",
                "coac_valid",
                "coac_obs_text",
                "coac_r_tool_override",
                "coac_r_tool",
                "coac/fallback_credit_rate",
                "coac_copy_overlap",
                "format_reward",
            ]
        },
    )

    _drop_stale_scenecot_reward_extra_for_recompute(batch)

    assert set(batch.non_tensor_batch) == {
        "coac_delta",
        "coac_valid",
        "coac_obs_text",
        "coac_r_tool_override",
    }
    assert batch.meta_info["reward_extra_keys"] == [
        "coac_delta",
        "coac_valid",
        "coac_obs_text",
        "coac_r_tool_override",
    ]


def test_train_metrics_include_coac_copy_overlap_sentinel():
    metrics = _compute_train_scenecot_reward_metrics(
        {
            "coac_copy_overlap": np.array([0.0, 0.5, 1.0], dtype=np.float32),
            "score": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        }
    )

    assert metrics["coac/copy_overlap_mean"] == pytest.approx(0.5)
    assert metrics["coac/overlap_reward_corr"] == pytest.approx(1.0)


def test_step1_format_reward_sentinel_warns_on_low_format_by_default(monkeypatch):
    monkeypatch.setenv("SCENECOT_STEP1_FORMAT_REWARD_MIN", "0.15")

    metrics = {"train/scenecot/format_reward/mean": 0.10}
    _check_scenecot_step1_format_reward_sentinel(metrics, global_steps=1)
    assert metrics["diagnostic/step1_format_reward_sentinel_tripped"] == 1.0

    metrics = {"train/scenecot/format_reward/mean": 0.30}
    _check_scenecot_step1_format_reward_sentinel(metrics, global_steps=1)
    assert metrics["diagnostic/step1_format_reward_sentinel_tripped"] == 0.0

    metrics = {"train/scenecot/format_reward/mean": 0.0}
    _check_scenecot_step1_format_reward_sentinel(metrics, global_steps=2)
    assert "diagnostic/step1_format_reward_sentinel_tripped" not in metrics


def test_step1_format_reward_sentinel_can_be_fatal(monkeypatch):
    monkeypatch.setenv("SCENECOT_STEP1_FORMAT_REWARD_MIN", "0.15")
    monkeypatch.setenv("SCENECOT_STEP1_FORMAT_REWARD_FATAL", "1")

    with pytest.raises(RuntimeError, match="Step-1 format reward sentinel tripped"):
        _check_scenecot_step1_format_reward_sentinel(
            {"train/scenecot/format_reward/mean": 0.10},
            global_steps=1,
        )


def test_coac_group_zscore_credit_is_monotonic_and_not_saturated():
    deltas = np.array([6.0, 3.0, 0.5, 0.3, 0.1, 0.0, -0.5, -1.0], dtype=np.float32)
    valid = np.ones_like(deltas, dtype=bool)
    groups = np.array(["prompt"] * len(deltas), dtype=object)

    norms, group_sizes, group_stds, r_tool = _compute_scenecot_group_zscore_credit(deltas, valid, groups)

    assert group_sizes.tolist() == [8] * 8
    assert all(value > 0 for value in group_stds)
    assert all(left > right for left, right in zip(r_tool, r_tool[1:]))
    assert 0.0 < float(r_tool[-1]) < float(r_tool[0]) < 1.0
    assert all(left > right for left, right in zip(norms, norms[1:]))


def test_coac_group_zscore_zero_std_returns_neutral_credit():
    deltas = np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    valid = np.ones_like(deltas, dtype=bool)
    groups = np.array(["prompt"] * len(deltas), dtype=object)

    norms, group_sizes, group_stds, r_tool = _compute_scenecot_group_zscore_credit(deltas, valid, groups)

    assert norms.tolist() == [0.0] * 4
    assert group_sizes.tolist() == [4] * 4
    assert group_stds.tolist() == [0.0] * 4
    assert r_tool.tolist() == pytest.approx([0.5] * 4)


def test_coac_group_zscore_singleton_returns_neutral_and_fallback_stays_fixed(monkeypatch):
    deltas = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    valid = np.array([True, False, False], dtype=bool)
    groups = np.array(["prompt"] * len(deltas), dtype=object)

    _, group_sizes, group_stds, r_tool = _compute_scenecot_group_zscore_credit(deltas, valid, groups)

    assert group_sizes.tolist() == [1, 0, 0]
    assert group_stds.tolist() == [0.0, 0.0, 0.0]
    assert r_tool.tolist() == pytest.approx([0.5, 0.0, 0.0])

    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_GROUP_ZSCORE", "1")
    reward, metrics = compute_coac_tool_reward(
        extra_info={"coac_valid": False, "coac_delta": 0.0},
        gt_answers=["chair"],
        tool_called=True,
        full_protocol_valid=True,
        catastrophic_invalid=False,
    )

    assert reward == pytest.approx(0.5)
    assert metrics["coac_tool_credit_source"] == "fixed_fallback"


def test_coac_group_zscore_disabled_preserves_clip_tau_behavior(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_GROUP_ZSCORE", "0")
    monkeypatch.setenv("SCENECOT_COAC_TAU", "0.5")

    reward, metrics = compute_coac_tool_reward(
        extra_info={
            "coac_valid": True,
            "coac_delta": 0.25,
            "coac_delta_group_norm": 10.0,
            "coac_group_size": 8,
            "coac_group_std": 1.0,
            "coac_r_tool_override": 0.99,
            "coac_obs_text": "table",
        },
        gt_answers=["chair"],
        tool_called=True,
        full_protocol_valid=True,
        catastrophic_invalid=False,
    )

    assert reward == pytest.approx(0.5)
    assert metrics["coac_r_tool"] == pytest.approx(0.5)
    assert metrics["coac_group_zscore_enabled"] == 0
