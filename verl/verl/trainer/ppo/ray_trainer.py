# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import hashlib
import logging
import math
import os
import random
import uuid
from collections import Counter, defaultdict
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from scenecot_verl.training.llava_format_aux import (
    LLAVA_FORMAT_AUX_ANSWER_MAX_TOKENS_ENV,
    LLAVA_FORMAT_AUX_COEF_ENV,
    LLAVA_FORMAT_AUX_ENABLE_ENV,
    LLAVA_FORMAT_AUX_REASONING_MAX_TOKENS_ENV,
    LLAVA_FORMAT_TAGS,
    build_llava_canonical_format_target,
    build_llava_format_aux_mask,
    is_llava_processor,
    resolve_llava_format_token_ids,
)
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
    ref_policy_in_actor,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.tokenizer import get_processor_token_id
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


logger = logging.getLogger(__name__)

VISUAL_CUE_METRIC_LABELS = {
    "<list_obj_prob>": "list_obj_prob",
    "<list_obj_loc_prob>": "list_obj_loc_prob",
    "<list_obj_loc_plr_prob>": "list_obj_loc_plr_prob",
    "<highlight_obj>": "highlight_obj",
}
COAC_MASK_PLACEHOLDER = "<obs> no observation available </obs>"
COAC_ANSWER_OPEN_TAG = "<answer>"
COAC_ANSWER_CLOSE_TAG = "</answer>"
COAC_VISUAL_ENABLE_ENV = "SCENECOT_COAC_VISUAL_ENABLE"
COAC_VISUAL_MODE = "highlight"
COAC_COVERED_MODES = {"prob", "loc_rect", "loc_polar"}
COAC_GROUP_ZSCORE_ENV = "SCENECOT_COAC_GROUP_ZSCORE"
COAC_FALLBACK_CREDIT_ENV = "SCENECOT_COAC_FALLBACK_CREDIT"
COAC_COUNTERFACTUAL_MODE_ENV = "SCENECOT_COAC_COUNTERFACTUAL_MODE"
COAC_RANDOM_SEED_ENV = "SCENECOT_COAC_RANDOM_SEED"
COAC_GROUP_ZSCORE_EPS = 1e-6
SCENECOT_STEP1_FORMAT_REWARD_MIN_ENV = "SCENECOT_STEP1_FORMAT_REWARD_MIN"
SCENECOT_STEP1_FORMAT_REWARD_FATAL_ENV = "SCENECOT_STEP1_FORMAT_REWARD_FATAL"
COAC_REWARD_RECOMPUTE_INPUT_KEYS = {
    "coac_delta",
    "coac_delta_group_norm",
    "coac_group_size",
    "coac_group_std",
    "coac_r_tool_override",
    "coac_valid",
    "coac_obs_text",
    "coac_skip_reason",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _scenecot_coac_counterfactual_mode() -> str:
    value = os.getenv(COAC_COUNTERFACTUAL_MODE_ENV, "").strip().lower()
    aliases = {"": "mask", "masked": "mask", "masked_token": "mask"}
    value = aliases.get(value, value)
    if value not in {"mask", "random_tokens", "random_tool"}:
        raise ValueError(
            f"{COAC_COUNTERFACTUAL_MODE_ENV} must be one of "
            "'mask', 'random_tokens', or 'random_tool'."
        )
    return value


def _stable_scenecot_random(seed_parts: tuple[Any, ...]) -> random.Random:
    payload = "\x1f".join(str(part) for part in seed_parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))


def _build_scenecot_random_token_ids(
    tokenizer: Any,
    *,
    length: int,
    seed_parts: tuple[Any, ...],
    forbidden_token_ids: set[int] | None = None,
) -> list[int]:
    """Sample deterministic ordinary vocabulary ids for a COAC intervention."""
    if length <= 0:
        return []
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    forbidden = set(int(token_id) for token_id in (forbidden_token_ids or set()))
    forbidden.update(int(token_id) for token_id in (getattr(tokenizer, "all_special_ids", None) or []))
    if vocab_size <= 0 or len(forbidden) >= vocab_size:
        raise ValueError("Tokenizer has no usable ordinary tokens for COAC random_tokens.")
    rng = _stable_scenecot_random(seed_parts)
    sampled: list[int] = []
    while len(sampled) < length:
        token_id = rng.randrange(vocab_size)
        if token_id not in forbidden:
            sampled.append(token_id)
    return sampled


def _first_ground_truth_text(ground_truth: Any) -> str:
    if isinstance(ground_truth, str):
        return ground_truth
    if isinstance(ground_truth, (list, tuple)):
        return str(ground_truth[0]) if ground_truth else ""
    if isinstance(ground_truth, dict):
        for key in ("answer", "answers", "gt_answer", "label"):
            if key not in ground_truth:
                continue
            value = ground_truth[key]
            if isinstance(value, (list, tuple)):
                return str(value[0]) if value else ""
            if value is not None:
                return str(value)
    return "" if ground_truth is None else str(ground_truth)


def _build_scenecot_coac_response_variants(
    response_valid: list[int],
    *,
    mode_tag_end_pos: int,
    obs_span: tuple[int, int],
    obs_context_end_pos: int | None = None,
    answer_open_ids: list[int],
    gt_ids: list[int],
    close_ids: list[int],
    placeholder_ids: list[int],
) -> tuple[list[int], list[int]]:
    obs_start, obs_end = obs_span
    context_end = obs_end if obs_context_end_pos is None else int(obs_context_end_pos)
    prefix = response_valid[:mode_tag_end_pos]
    observed_context = response_valid[mode_tag_end_pos:context_end]
    masked_context = response_valid[mode_tag_end_pos:obs_start] + placeholder_ids + response_valid[obs_end:context_end]
    observed_response = prefix + observed_context + answer_open_ids + gt_ids + close_ids
    masked_response = prefix + masked_context + answer_open_ids + gt_ids + close_ids
    return observed_response, masked_response


def _span_contains_any_token_id(token_ids: list[int], span: tuple[int, int], target_token_ids: set[int]) -> bool:
    if not target_token_ids:
        return False
    start, end = span
    return any(token_id in target_token_ids for token_id in token_ids[start:end])


def _count_token_id(token_ids: list[int], token_id: int | None) -> int:
    if token_id is None:
        return 0
    return sum(int(token == token_id) for token in token_ids)


def _count_tensor_token_id(token_ids: torch.Tensor, token_id: int | None) -> int:
    if token_id is None:
        return 0
    return int((token_ids == token_id).sum().item())


def _copy_scenecot_mm_inputs(mm_inputs: dict[str, Any] | None) -> dict[str, Any] | None:
    if mm_inputs is None:
        return None
    copied = {}
    for key, value in mm_inputs.items():
        if isinstance(value, torch.Tensor):
            copied[key] = value.clone()
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def _append_scenecot_cached_image_mm_inputs(
    base_mm_inputs: dict[str, Any] | None,
    image_mm_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Append one cached Qwen3-VL image to existing prompt image features."""
    result = _copy_scenecot_mm_inputs(base_mm_inputs) or {}
    if "pixel_values" in result:
        raise ValueError("random_tool_highlight_requires_cached_base_images")
    for key in ("image_embeds", "image_grid_thw", "images_seqlens"):
        incoming = image_mm_inputs.get(key)
        if not isinstance(incoming, torch.Tensor):
            raise ValueError(f"random_tool_highlight_missing_{key}")
        existing = result.get(key)
        if existing is None:
            result[key] = incoming.clone()
        elif isinstance(existing, torch.Tensor):
            result[key] = torch.cat((existing, incoming), dim=0)
        else:
            raise ValueError(f"random_tool_highlight_invalid_{key}")

    incoming_deepstack = image_mm_inputs.get("deepstack_image_embeds")
    existing_deepstack = result.get("deepstack_image_embeds")
    if not isinstance(incoming_deepstack, list):
        raise ValueError("random_tool_highlight_missing_deepstack_image_embeds")
    if existing_deepstack is None:
        result["deepstack_image_embeds"] = [tensor.clone() for tensor in incoming_deepstack]
    elif isinstance(existing_deepstack, list) and len(existing_deepstack) == len(incoming_deepstack):
        result["deepstack_image_embeds"] = [
            torch.cat((existing, incoming), dim=0)
            for existing, incoming in zip(existing_deepstack, incoming_deepstack, strict=True)
        ]
    else:
        raise ValueError("random_tool_highlight_deepstack_layer_mismatch")
    # This is only a presence marker. Position-id construction regenerates the
    # sequence-shaped token types from the final candidate input ids.
    result.pop("mm_token_type_ids", None)
    return result


def _remove_tensor_range(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if start < 0 or end < start or end > int(value.shape[0]):
        raise ValueError(f"invalid tensor slice removal range [{start}, {end}) for shape {tuple(value.shape)}")
    if start == 0 and end == int(value.shape[0]):
        return value[:0].clone()
    parts = []
    if start > 0:
        parts.append(value[:start])
    if end < int(value.shape[0]):
        parts.append(value[end:])
    if not parts:
        return value[:0].clone()
    return torch.cat(parts, dim=0)


def _scenecot_image_feature_counts(mm_inputs: dict[str, Any] | None, merge_size: int = 1) -> list[int]:
    image_grid = (mm_inputs or {}).get("image_grid_thw")
    if not isinstance(image_grid, torch.Tensor):
        # Gemma 3 has no Qwen-style image_grid_thw.  Its sequence-shaped
        # token_type_ids mark each contiguous block of 256 native soft image
        # tokens, while pixel_values stays image-indexed on dimension 0.
        token_type_ids = (mm_inputs or {}).get("token_type_ids")
        if not isinstance(token_type_ids, torch.Tensor):
            return []
        image_mask = token_type_ids.reshape(-1).bool().tolist()
        counts: list[int] = []
        index = 0
        while index < len(image_mask):
            if not image_mask[index]:
                index += 1
                continue
            run_end = index + 1
            while run_end < len(image_mask) and image_mask[run_end]:
                run_end += 1
            counts.append(run_end - index)
            index = run_end
        return counts
    merge_size = max(1, int(merge_size or 1))
    image_grid_long = image_grid.to(torch.long)
    patch_counts = image_grid_long[:, 0] * image_grid_long[:, 1] * image_grid_long[:, 2]
    token_counts = patch_counts // (merge_size * merge_size)
    return [int(value) for value in token_counts.tolist()]


def _scenecot_cached_image_feature_counts(
    mm_inputs: dict[str, Any] | None,
    *,
    merge_size: int = 1,
) -> list[int]:
    """Infer per-image feature counts for cached Qwen-VL image embeddings."""
    mm_inputs = mm_inputs or {}
    image_grid = mm_inputs.get("image_grid_thw")
    image_embeds = mm_inputs.get("image_embeds")
    if not isinstance(image_grid, torch.Tensor) or not isinstance(image_embeds, torch.Tensor):
        return _scenecot_image_feature_counts(mm_inputs, merge_size=merge_size)
    num_images = int(image_grid.shape[0])
    total_features = int(image_embeds.shape[0])
    if num_images <= 0:
        return []
    if num_images == 1:
        return [total_features]

    images_seqlens = mm_inputs.get("images_seqlens")
    if isinstance(images_seqlens, torch.Tensor) and int(images_seqlens.numel()) == num_images:
        counts = [int(value) for value in images_seqlens.to(torch.long).tolist()]
        if sum(counts) == total_features:
            return counts

    grid_counts = _scenecot_image_feature_counts(mm_inputs, merge_size=merge_size)
    if len(grid_counts) == num_images and sum(grid_counts) == total_features:
        return grid_counts

    raise ValueError(f"cached_image_feature_counts_ambiguous:{total_features}:{num_images}")


def _replace_image_runs_in_visual_span(
    token_ids: list[int],
    *,
    span: tuple[int, int],
    image_token_id: int | None,
    target_image_token_counts: list[int],
    obs_context_end_pos: int,
    vision_start_token_id: int | None = None,
    vision_end_token_id: int | None = None,
) -> tuple[list[int], tuple[int, int], int]:
    """Resize visual-observation image token blocks to match cached feature counts."""
    if image_token_id is None or not target_image_token_counts:
        return token_ids, span, obs_context_end_pos

    span_start, span_end = span
    span_tokens = token_ids[span_start:span_end]
    targets = list(target_image_token_counts)
    target_index = 0
    resized: list[int] = []
    index = 0

    while index < len(span_tokens):
        if (
            vision_start_token_id is not None
            and vision_end_token_id is not None
            and span_tokens[index] == vision_start_token_id
        ):
            end = index + 1
            while end < len(span_tokens) and span_tokens[end] != vision_end_token_id:
                end += 1
            block_end = end + 1 if end < len(span_tokens) else len(span_tokens)
            block = span_tokens[index:block_end]
            image_count = _count_token_id(block, image_token_id)
            if image_count and target_index < len(targets):
                target_count = max(0, int(targets[target_index]))
                target_index += 1
                new_block: list[int] = []
                replaced = False
                block_pos = 0
                while block_pos < len(block):
                    if block[block_pos] != image_token_id:
                        new_block.append(block[block_pos])
                        block_pos += 1
                        continue
                    run_end = block_pos
                    while run_end < len(block) and block[run_end] == image_token_id:
                        run_end += 1
                    if not replaced:
                        new_block.extend([image_token_id] * target_count)
                        replaced = True
                    block_pos = run_end
                resized.extend(new_block)
            else:
                resized.extend(block)
            index = block_end
            continue

        if span_tokens[index] == image_token_id:
            run_end = index
            while run_end < len(span_tokens) and span_tokens[run_end] == image_token_id:
                run_end += 1
            if target_index < len(targets):
                target_count = max(0, int(targets[target_index]))
                target_index += 1
                resized.extend([image_token_id] * target_count)
            else:
                resized.extend(span_tokens[index:run_end])
            index = run_end
            continue

        resized.append(span_tokens[index])
        index += 1

    if target_index != len(targets):
        raise ValueError(f"visual_image_block_count_mismatch:{target_index}:{len(targets)}")

    new_tokens = token_ids[:span_start] + resized + token_ids[span_end:]
    delta = len(resized) - (span_end - span_start)
    return new_tokens, (span_start, span_end + delta), obs_context_end_pos + delta


def _remove_scenecot_image_range_from_mm_inputs(
    mm_inputs: dict[str, Any] | None,
    *,
    image_start: int,
    image_count: int,
    image_token_counts: list[int] | None = None,
    merge_size: int = 1,
) -> dict[str, Any] | None:
    """Remove image-indexed multimodal features for masked visual COAC."""
    if mm_inputs is None:
        raise ValueError("missing_multi_modal_inputs")
    if image_count <= 0:
        raise ValueError("missing_visual_observation_image")

    result = _copy_scenecot_mm_inputs(mm_inputs) or {}
    image_grid = result.get("image_grid_thw")
    gemma_native_images = image_grid is None and isinstance(result.get("token_type_ids"), torch.Tensor)
    if gemma_native_images:
        feature_counts = _scenecot_image_feature_counts(result, merge_size=merge_size)
        pixel_values = result.get("pixel_values")
        if not feature_counts or not isinstance(pixel_values, torch.Tensor):
            raise ValueError("missing_gemma3_image_inputs")
        num_images = len(feature_counts)
        if pixel_values.ndim < 1 or int(pixel_values.shape[0]) != num_images:
            raise ValueError(
                f"gemma3_pixel_values_shape_mismatch:{tuple(pixel_values.shape)}:{num_images}"
            )
        patch_counts = frame_counts = [1] * num_images
    else:
        if not isinstance(image_grid, torch.Tensor):
            raise ValueError("missing_image_grid_thw")
        if image_grid.ndim != 2 or image_grid.shape[1] != 3:
            raise ValueError(f"invalid_image_grid_thw_shape:{tuple(image_grid.shape)}")
        num_images = int(image_grid.shape[0])
        image_grid_long = image_grid.to(torch.long)
        patch_counts = (image_grid_long[:, 0] * image_grid_long[:, 1] * image_grid_long[:, 2]).tolist()
        frame_counts = image_grid_long[:, 0].tolist()
    image_end = image_start + image_count
    if image_start < 0 or image_end > num_images:
        raise ValueError(f"visual_image_index_out_of_range:{image_start}:{image_count}:{num_images}")

    patch_start = int(sum(patch_counts[:image_start]))
    patch_end = int(sum(patch_counts[:image_end]))
    frame_start = int(sum(frame_counts[:image_start]))
    frame_end = int(sum(frame_counts[:image_end]))
    feature_start = feature_end = None
    if not gemma_native_images:
        feature_counts = None
    if not gemma_native_images and result.get("image_embeds") is not None:
        feature_counts = _scenecot_cached_image_feature_counts(result, merge_size=merge_size)
    elif not gemma_native_images and image_token_counts is not None:
        feature_counts = image_token_counts
    if feature_counts is not None:
        if len(feature_counts) != num_images:
            raise ValueError(f"image_feature_counts_shape_mismatch:{len(feature_counts)}:{num_images}")
        feature_start = int(sum(feature_counts[:image_start]))
        feature_end = int(sum(feature_counts[:image_end]))

    if not gemma_native_images:
        result["image_grid_thw"] = _remove_tensor_range(image_grid, image_start, image_end)

    image_embeds = result.get("image_embeds")
    if image_embeds is not None:
        if not isinstance(image_embeds, torch.Tensor):
            raise ValueError("invalid_image_embeds_type")
        if feature_start is None or feature_end is None:
            raise ValueError("missing_image_token_counts_for_image_embeds")
        if int(image_embeds.shape[0]) < feature_end:
            raise ValueError(f"image_embeds_too_short:{int(image_embeds.shape[0])}:{feature_end}")
        result["image_embeds"] = _remove_tensor_range(image_embeds, feature_start, feature_end)

    deepstack = result.get("deepstack_image_embeds")
    if deepstack is not None:
        if not isinstance(deepstack, list):
            raise ValueError("invalid_deepstack_image_embeds_type")
        if feature_start is None or feature_end is None:
            raise ValueError("missing_image_token_counts_for_deepstack")
        trimmed_deepstack = []
        for tensor in deepstack:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("invalid_deepstack_image_embed_type")
            if int(tensor.shape[0]) < feature_end:
                raise ValueError(f"deepstack_image_embeds_too_short:{int(tensor.shape[0])}:{feature_end}")
            trimmed_deepstack.append(_remove_tensor_range(tensor, feature_start, feature_end))
        result["deepstack_image_embeds"] = trimmed_deepstack

    pixel_values = result.get("pixel_values")
    if pixel_values is not None:
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("invalid_pixel_values_type")
        if gemma_native_images:
            result["pixel_values"] = _remove_tensor_range(pixel_values, image_start, image_end)
        elif int(pixel_values.shape[0]) < patch_end:
            raise ValueError(f"pixel_values_too_short:{int(pixel_values.shape[0])}:{patch_end}")
        else:
            result["pixel_values"] = _remove_tensor_range(pixel_values, patch_start, patch_end)

    images_seqlens = result.get("images_seqlens")
    if images_seqlens is not None:
        if not isinstance(images_seqlens, torch.Tensor):
            raise ValueError("invalid_images_seqlens_type")
        if int(images_seqlens.shape[0]) == int(sum(frame_counts)):
            result["images_seqlens"] = _remove_tensor_range(images_seqlens, frame_start, frame_end)
        elif int(images_seqlens.shape[0]) == num_images:
            result["images_seqlens"] = _remove_tensor_range(images_seqlens, image_start, image_end)
        else:
            raise ValueError(f"images_seqlens_shape_mismatch:{int(images_seqlens.shape[0])}:{num_images}")

    for key, value in list(result.items()):
        if key in {
            "image_grid_thw",
            "image_embeds",
            "deepstack_image_embeds",
            "pixel_values",
            "images_seqlens",
            "mm_token_type_ids",
            # Gemma 3's token_type_ids are sequence-shaped, not
            # image-indexed.  They are rebuilt from each final COAC
            # candidate below after the response has been rewritten.
            "token_type_ids",
        }:
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == num_images:
            result[key] = _remove_tensor_range(value, image_start, image_end)

    no_images_remain = (
        int(result["pixel_values"].shape[0]) == 0
        if gemma_native_images and isinstance(result.get("pixel_values"), torch.Tensor)
        else int(result["image_grid_thw"].shape[0]) == 0
    )
    if no_images_remain:
        result.pop("image_grid_thw", None)
        result.pop("image_embeds", None)
        result.pop("deepstack_image_embeds", None)
        result.pop("pixel_values", None)
        result.pop("images_seqlens", None)
    return result


def _scenecot_total_image_feature_count(mm_inputs: dict[str, Any] | None, merge_size: int = 1) -> int:
    image_embeds = (mm_inputs or {}).get("image_embeds")
    if isinstance(image_embeds, torch.Tensor):
        return int(image_embeds.shape[0])
    return int(sum(_scenecot_image_feature_counts(mm_inputs, merge_size=merge_size)))


def _scenecot_image_block_token_counts(
    token_ids: list[int],
    *,
    image_token_id: int | None,
    vision_start_token_id: int | None = None,
    vision_end_token_id: int | None = None,
) -> list[int]:
    if image_token_id is None:
        return []
    if vision_start_token_id is None or vision_end_token_id is None:
        count = _count_token_id(token_ids, image_token_id)
        return [count] if count else []

    counts = []
    index = 0
    while index < len(token_ids):
        if token_ids[index] != vision_start_token_id:
            index += 1
            continue
        end = index + 1
        while end < len(token_ids) and token_ids[end] != vision_end_token_id:
            end += 1
        if end < len(token_ids):
            end += 1
        count = _count_token_id(token_ids[index:end], image_token_id)
        if count:
            counts.append(count)
        index = max(end, index + 1)
    return counts


def _scenecot_image_row_range_for_token_span(
    mm_inputs: dict[str, Any] | None,
    *,
    image_token_offset: int,
    image_token_count: int,
    merge_size: int = 1,
) -> tuple[int, int]:
    counts = _scenecot_image_feature_counts(mm_inputs, merge_size=merge_size)
    if image_token_count <= 0:
        raise ValueError("missing_visual_observation_image")
    start_row = None
    end_row = None
    cursor = 0
    for row_idx, count in enumerate(counts):
        if start_row is None and cursor == image_token_offset:
            start_row = row_idx
        cursor += count
        if cursor == image_token_offset + image_token_count:
            end_row = row_idx + 1
            break
    if start_row is None or end_row is None:
        raise ValueError(
            f"visual_image_token_span_not_aligned:{image_token_offset}:{image_token_count}:{counts}"
        )
    return start_row, end_row - start_row


def _expand_visual_obs_span(
    response_valid: list[int],
    *,
    mode_tag_end_pos: int,
    obs_span: tuple[int, int],
    obs_context_end_pos: int,
    image_token_id: int | None,
    vision_start_token_id: int | None = None,
    vision_end_token_id: int | None = None,
) -> tuple[int, int]:
    """Include image placeholder token(s) that belong to a highlight observation."""
    if image_token_id is None:
        raise ValueError("missing_image_token_id")
    obs_start, obs_end = obs_span
    context_start = max(0, mode_tag_end_pos)
    context_end = min(len(response_valid), obs_context_end_pos)
    image_positions = [
        index for index in range(context_start, context_end) if response_valid[index] == image_token_id
    ]
    if not image_positions:
        raise ValueError("missing_visual_observation_image")
    visual_start = image_positions[0]
    visual_end = image_positions[-1] + 1
    if vision_start_token_id is not None:
        for index in range(image_positions[0] - 1, context_start - 1, -1):
            if response_valid[index] == vision_start_token_id:
                visual_start = index
                break
    if vision_end_token_id is not None:
        for index in range(image_positions[-1] + 1, context_end):
            if response_valid[index] == vision_end_token_id:
                visual_end = index + 1
                break
    return min(obs_start, visual_start), max(obs_end, visual_end)


def _validate_scenecot_image_count(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_inputs: dict[str, Any] | None,
    image_token_id: int | None,
    merge_size: int = 1,
) -> None:
    if image_token_id is None:
        return
    valid_ids = input_ids[attention_mask.bool()]
    placeholder_count = _count_tensor_token_id(valid_ids, image_token_id)
    image_feature_count = _scenecot_total_image_feature_count(mm_inputs, merge_size=merge_size)
    if placeholder_count != image_feature_count:
        raise ValueError(f"image_placeholder_count_mismatch:{placeholder_count}:{image_feature_count}")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _compute_scenecot_group_zscore_credit(
    delta_values: np.ndarray,
    valid_values: np.ndarray,
    group_ids: Any,
    *,
    fallback_credit: float = 0.5,
    eps: float = COAC_GROUP_ZSCORE_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample z-score, group size/std, and pre-copy r_tool."""
    deltas = np.asarray(delta_values, dtype=np.float32)
    valid = np.asarray(valid_values, dtype=bool)
    group_array = np.asarray(group_ids, dtype=object) if group_ids is not None else np.arange(len(deltas), dtype=object)
    fallback = float(np.clip(fallback_credit, 0.0, 1.0))

    norm_values = np.zeros(len(deltas), dtype=np.float32)
    group_sizes = np.zeros(len(deltas), dtype=np.int32)
    group_stds = np.zeros(len(deltas), dtype=np.float32)
    r_tool_values = np.zeros(len(deltas), dtype=np.float32)

    groups: dict[Any, list[int]] = defaultdict(list)
    for index, is_valid in enumerate(valid):
        if is_valid:
            groups[group_array[index]].append(index)

    for indices in groups.values():
        values = deltas[indices].astype(np.float64)
        group_size = len(indices)
        group_std = float(np.std(values)) if group_size else 0.0
        if group_size >= 2 and group_std > eps:
            group_mean = float(np.mean(values))
            for index in indices:
                z_value = float((float(deltas[index]) - group_mean) / group_std)
                norm_values[index] = z_value
                r_tool_values[index] = _sigmoid(z_value)
        else:
            r_tool_values[indices] = fallback
        group_sizes[indices] = group_size
        group_stds[indices] = group_std

    return norm_values, group_sizes, group_stds, r_tool_values


def _metric_label(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, np.generic):
        value = value.item()
    text = str(value).strip()
    if not text or text.lower() == "none":
        return "none"
    if text in VISUAL_CUE_METRIC_LABELS:
        return VISUAL_CUE_METRIC_LABELS[text]
    label = "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")
    return label or "unknown"


def _row_non_tensor(batch: DataProto, key: str, index: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _scenecot_tool_extra_for_row(batch: DataProto, index: int) -> dict[str, Any]:
    tool_extra = _row_non_tensor(batch, "tool_extra_fields", index)
    if isinstance(tool_extra, dict):
        return tool_extra

    # Agent-loop postprocessing flattens extra_fields into top-level
    # non-tensor keys before the trainer sees the batch.  Rebuild the small
    # SceneCOT metadata dict COAC expects from those flattened fields.
    rebuilt: dict[str, Any] = {}
    for key in (
        "scenecot_coac",
        "scenecot_actions",
        "scenecot_metrics",
        "rollout_step",
        "rollout_n",
        "sample_index",
    ):
        value = _row_non_tensor(batch, key, index)
        if value is not None:
            rebuilt[key] = value
    return rebuilt


def _add_validation_count_metrics(
    metric_dict: dict[str, float],
    data_sources,
    values,
    key: str,
) -> None:
    if values is None or len(values) != len(data_sources):
        return

    source_to_values = defaultdict(list)
    for data_source, value in zip(data_sources, values, strict=False):
        source_to_values[str(data_source)].append(value)

    for data_source, source_values in source_to_values.items():
        total = len(source_values)
        if total == 0:
            continue
        counts = Counter(_metric_label(value) for value in source_values)
        metric_dict[f"val-aux/{data_source}/{key}/total"] = total
        for label, count in sorted(counts.items()):
            metric_dict[f"val-aux/{data_source}/{key}/{label}/count"] = count
            metric_dict[f"val-aux/{data_source}/{key}/{label}/rate"] = count / total


def _add_train_count_metrics(
    metric_dict: dict[str, float],
    values,
    key: str,
) -> None:
    if values is None:
        return

    value_list = list(values)
    total = len(value_list)
    if total == 0:
        return

    counts = Counter(_metric_label(value) for value in value_list)
    metric_dict[f"train/scenecot/{key}/total"] = total
    for label, count in sorted(counts.items()):
        metric_dict[f"train/scenecot/{key}/{label}/count"] = count
        metric_dict[f"train/scenecot/{key}/{label}/rate"] = count / total


def _scenecot_mode_label(value: Any) -> str:
    label = _metric_label(value)
    return {
        "list_obj_prob": "prob",
        "list_obj_loc_prob": "loc_rect",
        "list_obj_loc_plr_prob": "loc_polar",
        "highlight_obj": "highlight",
    }.get(label, label)


def _add_train_format_valid_by_mode_metrics(metric_dict: dict[str, float], non_tensor_batch: dict) -> None:
    mode_values = non_tensor_batch.get("first_visual_cue")
    if mode_values is None:
        return

    valid_values = non_tensor_batch.get("full_protocol_valid")
    if valid_values is None:
        valid_values = non_tensor_batch.get("format_reward")
    if valid_values is None:
        return

    modes = np.asarray([_scenecot_mode_label(value) for value in mode_values], dtype=object)
    valid = np.asarray(valid_values, dtype=np.float32)
    if modes.shape[0] != valid.shape[0]:
        return

    finite = np.isfinite(valid)
    if not finite.any():
        return
    valid_binary = (valid > 0.0).astype(np.float32)

    for mode in sorted(set(modes[finite].tolist())):
        mode_mask = finite & (modes == mode)
        if not mode_mask.any():
            continue
        metric_dict[f"train/scenecot/format_valid_rate/{mode}"] = float(np.mean(valid_binary[mode_mask]))
        metric_dict[f"train/scenecot/format_valid_total/{mode}"] = int(np.sum(mode_mask))

    forced_modes = non_tensor_batch.get("gcs_forced_mode")
    if forced_modes is None:
        return
    forced_modes = np.asarray(forced_modes, dtype=object)
    if forced_modes.shape[0] != modes.shape[0]:
        return
    forced_mask = np.array([bool(str(mode or "").strip()) for mode in forced_modes], dtype=bool)
    for stream, stream_mask in (("free", ~forced_mask), ("forced", forced_mask)):
        stream_finite = finite & stream_mask
        if not stream_finite.any():
            continue
        for mode in sorted(set(modes[stream_finite].tolist())):
            mode_mask = stream_finite & (modes == mode)
            if not mode_mask.any():
                continue
            metric_dict[f"train/scenecot/format_valid_rate/{stream}/{mode}"] = float(
                np.mean(valid_binary[mode_mask])
            )
            metric_dict[f"train/scenecot/format_valid_total/{stream}/{mode}"] = int(np.sum(mode_mask))


def _finite_pearson_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(mask)) < 2:
        return None
    left = left[mask]
    right = right[mask]
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denom = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denom <= 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denom)


def _compute_train_scenecot_reward_metrics(non_tensor_batch: dict) -> dict[str, float]:
    metric_keys = (
        "answer_reward",
        "answer_judge_called",
        "answer_judge_fallback",
        "answer_judge_score_1_to_5",
        "format_reward",
        "format_progress_reward",
        "format_progress_first_turn",
        "format_progress_reasoning",
        "format_progress_answer_body",
        "format_progress_answer_close",
        "format_has_answer_reward",
        "format_has_reasoning_reward",
        "format_has_tool_call_reward",
        "penalty",
        "final_answer_autoclosed_count",
        "final_answer_autoclose_eligible_count",
        "format_minor_violation",
        "agent_reward_progress",
        "progress_switch_step",
        "w_format",
        "w_answer",
        "w_tool",
        "catastrophic_reward_value",
    )
    metrics = {}
    for key in metric_keys:
        if key not in non_tensor_batch:
            continue
        values = np.asarray(non_tensor_batch[key], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        metrics[f"train/scenecot/{key}/mean"] = float(np.mean(values))

    if "coac/fallback_credit_rate" in non_tensor_batch:
        values = np.asarray(non_tensor_batch["coac/fallback_credit_rate"], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            metrics["coac/fallback_credit_rate"] = float(np.mean(values))

    if "coac_fixed_fallback_applied" in non_tensor_batch:
        values = np.asarray(non_tensor_batch["coac_fixed_fallback_applied"], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            metrics["coac/fixed_fallback_applied_rate"] = float(np.mean(values))

    if "coac_valid" in non_tensor_batch:
        values = np.asarray(non_tensor_batch["coac_valid"], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            metrics["coac/valid_rate"] = float(np.mean(values))

    if "coac_r_tool" in non_tensor_batch:
        values = np.asarray(non_tensor_batch["coac_r_tool"], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            metrics["coac/r_tool_mean"] = float(np.mean(values))

    if "coac_copy_overlap" in non_tensor_batch:
        overlap_values = np.asarray(non_tensor_batch["coac_copy_overlap"], dtype=np.float32)
        finite_overlap = overlap_values[np.isfinite(overlap_values)]
        if finite_overlap.size:
            metrics["coac/copy_overlap_mean"] = float(np.mean(finite_overlap))
        reward_values = non_tensor_batch.get("score", non_tensor_batch.get("reward"))
        if reward_values is not None:
            corr = _finite_pearson_corr(overlap_values, np.asarray(reward_values, dtype=np.float32))
            if corr is not None:
                metrics["coac/overlap_reward_corr"] = corr

    if "gcs_forced_mode" in non_tensor_batch and "full_protocol_valid" in non_tensor_batch:
        forced_modes = np.asarray(non_tensor_batch["gcs_forced_mode"], dtype=object)
        protocol_valid = np.asarray(non_tensor_batch["full_protocol_valid"], dtype=np.float32)
        forced_mask = np.array([bool(str(mode or "").strip()) for mode in forced_modes], dtype=bool)
        finite_mask = forced_mask & np.isfinite(protocol_valid)
        if finite_mask.any():
            invalid = 1.0 - (protocol_valid[finite_mask] > 0.0).astype(np.float32)
            metrics["gcs/forced_invalid_rate"] = float(np.mean(invalid))
            for mode in ("prob", "loc_rect", "loc_polar", "highlight"):
                mode_mask = finite_mask & (forced_modes == mode)
                if mode_mask.any():
                    mode_invalid = 1.0 - (protocol_valid[mode_mask] > 0.0).astype(np.float32)
                    metrics[f"gcs/forced_invalid_rate/{mode}"] = float(np.mean(mode_invalid))

    _add_train_count_metrics(metrics, non_tensor_batch.get("first_visual_cue"), "first_visual_cue")
    _add_train_count_metrics(metrics, non_tensor_batch.get("coac_tool_credit_source"), "coac_tool_credit_source")
    _add_train_count_metrics(metrics, non_tensor_batch.get("answer_judge_status"), "answer_judge_status")
    _add_train_count_metrics(metrics, non_tensor_batch.get("answer_judge_label"), "answer_judge_label")
    _add_train_format_valid_by_mode_metrics(metrics, non_tensor_batch)
    return metrics


def _compute_scenecot_grpo_group_metrics(batch: DataProto) -> dict[str, float]:
    """Measure how often outcome rewards provide within-prompt GRPO signal."""
    if "uid" not in batch.non_tensor_batch or "token_level_scores" not in batch.batch:
        return {}

    group_ids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    scores = batch.batch["token_level_scores"].detach().float().sum(dim=-1).cpu().numpy()
    if len(group_ids) != len(scores):
        return {}

    grouped_scores: dict[Any, list[float]] = defaultdict(list)
    for group_id, score in zip(group_ids, scores, strict=False):
        if np.isfinite(score):
            grouped_scores[group_id].append(float(score))
    if not grouped_scores:
        return {}

    group_stds = np.asarray(
        [np.std(values, dtype=np.float64) for values in grouped_scores.values()],
        dtype=np.float64,
    )
    unique_counts = np.asarray(
        [len(set(values)) for values in grouped_scores.values()],
        dtype=np.float64,
    )
    zero_std = group_stds <= COAC_GROUP_ZSCORE_EPS
    return {
        "train/scenecot/grpo_group/reward_std_mean": float(np.mean(group_stds)),
        "train/scenecot/grpo_group/reward_std_min": float(np.min(group_stds)),
        "train/scenecot/grpo_group/reward_std_max": float(np.max(group_stds)),
        "train/scenecot/grpo_group/zero_std_rate": float(np.mean(zero_std)),
        "train/scenecot/grpo_group/unique_reward_count_mean": float(np.mean(unique_counts)),
        "train/scenecot/grpo_group/total": float(len(group_stds)),
    }


def _check_scenecot_step1_format_reward_sentinel(metrics: dict[str, Any], global_steps: int) -> None:
    threshold = _env_float(SCENECOT_STEP1_FORMAT_REWARD_MIN_ENV, -1.0)
    if global_steps != 1 or threshold < 0.0:
        return

    key = "train/scenecot/format_reward/mean"
    value = metrics.get(key)
    if value is None:
        message = (
            f"{SCENECOT_STEP1_FORMAT_REWARD_MIN_ENV} is set, but {key} is missing. "
            "Cannot verify that the SFT LoRA adapter loaded."
        )
        metrics["diagnostic/step1_format_reward_sentinel_tripped"] = 1.0
        metrics["diagnostic/step1_format_reward_threshold"] = threshold
        if _env_bool(SCENECOT_STEP1_FORMAT_REWARD_FATAL_ENV, False):
            raise RuntimeError(message)
        logger.warning(message)
        return
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} is not numeric: {value!r}") from exc
    if value_float < threshold:
        message = (
            f"Step-1 format reward sentinel tripped: {key}={value_float:.6f} < {threshold:.6f}. "
            "This is consistent with the actor starting from the bare base model instead of the SFT LoRA adapter."
        )
        metrics["diagnostic/step1_format_reward_sentinel_tripped"] = 1.0
        metrics["diagnostic/step1_format_reward_threshold"] = threshold
        if _env_bool(SCENECOT_STEP1_FORMAT_REWARD_FATAL_ENV, False):
            raise RuntimeError(message)
        logger.warning(message)
    else:
        metrics["diagnostic/step1_format_reward_sentinel_tripped"] = 0.0
        metrics["diagnostic/step1_format_reward_threshold"] = threshold


def _drop_stale_scenecot_reward_extra_for_recompute(batch: DataProto) -> None:
    """Clear rollout-time reward diagnostics before COAC-aware reward recompute."""
    reward_extra_keys = list(batch.meta_info.get("reward_extra_keys", []))
    if not reward_extra_keys:
        return

    kept_reward_extra_keys = [
        key for key in reward_extra_keys if key in COAC_REWARD_RECOMPUTE_INPUT_KEYS and key in batch.non_tensor_batch
    ]
    stale_reward_extra_keys = [
        key for key in reward_extra_keys if key not in COAC_REWARD_RECOMPUTE_INPUT_KEYS and key in batch.non_tensor_batch
    ]
    for key in stale_reward_extra_keys:
        batch.non_tensor_batch.pop(key, None)
    batch.meta_info["reward_extra_keys"] = kept_reward_extra_keys


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = ref_policy_in_actor(config)

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(outputs)
        base_data = {
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=str))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _compute_scenecot_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: list[dict[str, Any] | None],
    ) -> torch.Tensor:
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)
        if not callable(getattr(self.processor, "get_rope_index", None)):
            # Qwen-VL processors expose get_rope_index() for multimodal
            # mRoPE. LLaVA uses the underlying Llama model's standard 1-D
            # rotary positions even when image embeddings replace <image>
            # placeholders, so its LlavaProcessor intentionally has no such
            # method. Keep COAC candidates consistent with the normal LLaVA
            # agent-loop path instead of calling the Qwen-only API.
            return compute_position_id_with_mask(attention_mask)

        rows = []
        for row_idx in range(input_ids.shape[0]):
            mm_inputs = multi_modal_inputs[row_idx] or {}
            multi_modal_kwargs = {
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
            }
            if mm_inputs.get("mm_token_type_ids") is not None:
                mm_token_type_ids = torch.zeros_like(input_ids[row_idx : row_idx + 1])
                image_token_id = get_processor_token_id(self.processor, "image")
                video_token_id = get_processor_token_id(self.processor, "video")
                if image_token_id is not None:
                    mm_token_type_ids[input_ids[row_idx : row_idx + 1] == image_token_id] = 1
                if video_token_id is not None:
                    mm_token_type_ids[input_ids[row_idx : row_idx + 1] == video_token_id] = 2
                multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids
            vision_position_ids, _ = self.processor.get_rope_index(
                input_ids=input_ids[row_idx : row_idx + 1],
                attention_mask=attention_mask[row_idx : row_idx + 1],
                **multi_modal_kwargs,
            )
            vision_position_ids = vision_position_ids.transpose(0, 1)

            valid_mask = attention_mask[row_idx].bool()
            text_position_ids = torch.ones((1, input_ids.shape[1]), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            text_position_ids = text_position_ids.unsqueeze(0)
            rows.append(torch.cat((text_position_ids, vision_position_ids), dim=1))
        return torch.cat(rows, dim=0)

    def _build_scenecot_coac_batch(
        self, batch: DataProto
    ) -> tuple[DataProto | None, list[tuple[int, int]], dict[str, Any], np.ndarray, np.ndarray]:
        prompt_width = batch.batch["prompts"].shape[1]
        response_width = batch.batch["responses"].shape[1]
        prompts = batch.batch["prompts"].cpu()
        responses = batch.batch["responses"].cpu()
        attention_mask = batch.batch["attention_mask"].cpu()
        response_attention = attention_mask[:, prompt_width:]
        placeholder_ids = self.tokenizer.encode(COAC_MASK_PLACEHOLDER, add_special_tokens=False)
        answer_open_ids = self.tokenizer.encode(COAC_ANSWER_OPEN_TAG, add_special_tokens=False)
        close_ids = self.tokenizer.encode(COAC_ANSWER_CLOSE_TAG, add_special_tokens=False)
        counterfactual_mode = _scenecot_coac_counterfactual_mode()
        random_seed = int(os.getenv(COAC_RANDOM_SEED_ENV, "42") or 42)
        visual_coac_enabled = _env_bool(COAC_VISUAL_ENABLE_ENV, False)
        image_token_id = get_processor_token_id(self.processor, "image")
        video_token_id = get_processor_token_id(self.processor, "video")
        image_processor = getattr(self.processor, "image_processor", None)
        image_merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
        try:
            vision_start_ids = self.tokenizer.encode("<|vision_start|>", add_special_tokens=False)
        except Exception:
            vision_start_ids = []
        try:
            vision_end_ids = self.tokenizer.encode("<|vision_end|>", add_special_tokens=False)
        except Exception:
            vision_end_ids = []
        vision_start_token_id = vision_start_ids[0] if len(vision_start_ids) == 1 else None
        vision_end_token_id = vision_end_ids[0] if len(vision_end_ids) == 1 else None

        rows: list[dict[str, Any]] = []
        sample_pairs: list[tuple[int, int]] = []
        skip_reasons = np.empty(len(batch), dtype=object)
        skip_reasons[:] = ""
        error_reasons = np.empty(len(batch), dtype=object)
        error_reasons[:] = ""
        skipped = 0
        visual_obs_skipped = 0
        visual_build_attempted = 0
        visual_build_valid = 0
        visual_build_failed = 0
        errors: list[str] = []
        mm_inputs_source = batch.non_tensor_batch.get("multi_modal_inputs")
        mm_token_ids = {
            token_id
            for token_id in (image_token_id, video_token_id)
            if token_id is not None
        }
        random_token_forbidden_ids = set(mm_token_ids)
        random_token_forbidden_ids.update(placeholder_ids)
        random_token_forbidden_ids.update(answer_open_ids)
        random_token_forbidden_ids.update(close_ids)
        for structural_text in (
            "<think_grd>",
            "</think_grd>",
            "<reasoning>",
            "</reasoning>",
            "<list_obj_prob>",
            "<list_obj_loc_prob>",
            "<list_obj_loc_plr_prob>",
            "<highlight_obj>",
            "<|vision_start|>",
            "<|vision_end|>",
        ):
            try:
                random_token_forbidden_ids.update(
                    self.tokenizer.encode(structural_text, add_special_tokens=False)
                )
            except Exception:
                pass
        random_token_count = 0
        random_tool_count = 0
        random_tool_mode_counts = Counter()

        for sample_idx in range(len(batch)):
            tool_extra = _scenecot_tool_extra_for_row(batch, sample_idx)
            coac_meta = tool_extra.get("scenecot_coac", {}) if isinstance(tool_extra, dict) else {}
            if not isinstance(coac_meta, dict) or not coac_meta.get("tool_called"):
                skipped += 1
                skip_reasons[sample_idx] = "not_tool_called"
                continue
            if (
                counterfactual_mode != "mask"
                and not coac_meta.get("qwen3_counterfactual_supported", False)
            ):
                skipped += 1
                skip_reasons[sample_idx] = "counterfactual_not_qwen3"
                error_reasons[sample_idx] = "counterfactual_not_qwen3"
                continue
            actions = tool_extra.get("scenecot_actions", []) if isinstance(tool_extra, dict) else []
            first_action = actions[0] if isinstance(actions, list) and actions else {}
            mode = first_action.get("mode") if isinstance(first_action, dict) else coac_meta.get("mode")
            is_visual_mode = mode == COAC_VISUAL_MODE
            if is_visual_mode and not visual_coac_enabled:
                skipped += 1
                visual_obs_skipped += 1
                skip_reasons[sample_idx] = "visual_obs_mode"
                continue
            if mode and mode not in COAC_COVERED_MODES and not is_visual_mode:
                skipped += 1
                skip_reasons[sample_idx] = "not_covered_mode"
                continue
            obs_span = coac_meta.get("obs_span")
            obs_context_end_pos = coac_meta.get("obs_context_end_pos")
            mode_tag_end_pos = coac_meta.get("mode_tag_end_pos")
            gt_answer_text = coac_meta.get("gt_answer_text") or _first_ground_truth_text(
                batch.non_tensor_batch.get("reward_model", np.array([{}], dtype=object))[sample_idx].get("ground_truth", None)
            )
            if not isinstance(obs_span, (list, tuple)) or len(obs_span) != 2:
                skipped += 1
                skip_reasons[sample_idx] = "missing_obs_span"
                error_reasons[sample_idx] = "missing_obs_span"
                continue
            if not isinstance(mode_tag_end_pos, (int, np.integer)):
                skipped += 1
                skip_reasons[sample_idx] = "missing_mode_tag_end_pos"
                error_reasons[sample_idx] = "missing_mode_tag_end_pos"
                continue
            if not str(gt_answer_text).strip():
                skipped += 1
                skip_reasons[sample_idx] = "missing_gt_answer"
                error_reasons[sample_idx] = "missing_gt_answer"
                continue
            obs_start, obs_end = int(obs_span[0]), int(obs_span[1])
            obs_context_end_pos = obs_end if obs_context_end_pos is None else int(obs_context_end_pos)
            mode_tag_end_pos = int(mode_tag_end_pos)
            valid_response_len = int(response_attention[sample_idx].sum().item())
            if (
                mode_tag_end_pos < 0
                or obs_start < mode_tag_end_pos
                or obs_end <= obs_start
                or obs_context_end_pos < obs_end
                or obs_end > valid_response_len
                or obs_context_end_pos > valid_response_len
            ):
                skipped += 1
                errors.append("invalid_span")
                skip_reasons[sample_idx] = "invalid_span"
                error_reasons[sample_idx] = "invalid_span"
                continue

            gt_ids = self.tokenizer.encode(str(gt_answer_text), add_special_tokens=False)
            if not answer_open_ids or not gt_ids:
                skipped += 1
                errors.append("missing_gt_answer")
                skip_reasons[sample_idx] = "missing_gt_answer"
                error_reasons[sample_idx] = "missing_gt_answer"
                continue

            response_valid = responses[sample_idx, :valid_response_len].tolist()
            base_mm_inputs = mm_inputs_source[sample_idx] if mm_inputs_source is not None else None
            coac_obs_span = (obs_start, obs_end)
            observed_mm_inputs = _copy_scenecot_mm_inputs(base_mm_inputs)
            counterfactual_mm_inputs = _copy_scenecot_mm_inputs(base_mm_inputs)
            if is_visual_mode:
                visual_build_attempted += 1
                try:
                    coac_obs_span = _expand_visual_obs_span(
                        response_valid,
                        mode_tag_end_pos=mode_tag_end_pos,
                        obs_span=(obs_start, obs_end),
                        obs_context_end_pos=obs_context_end_pos,
                        image_token_id=image_token_id,
                        vision_start_token_id=vision_start_token_id,
                        vision_end_token_id=vision_end_token_id,
                    )
                    prompt_valid_ids = prompts[sample_idx][attention_mask[sample_idx, :prompt_width].bool()].tolist()
                    image_token_count = _count_token_id(
                        response_valid[coac_obs_span[0] : coac_obs_span[1]],
                        image_token_id,
                    )
                    image_token_counts = None
                    if vision_start_token_id is not None and vision_end_token_id is not None:
                        image_token_counts = _scenecot_image_block_token_counts(
                            prompt_valid_ids + response_valid,
                            image_token_id=image_token_id,
                            vision_start_token_id=vision_start_token_id,
                            vision_end_token_id=vision_end_token_id,
                        )
                        image_start = len(
                            _scenecot_image_block_token_counts(
                                prompt_valid_ids + response_valid[: coac_obs_span[0]],
                                image_token_id=image_token_id,
                                vision_start_token_id=vision_start_token_id,
                                vision_end_token_id=vision_end_token_id,
                            )
                        )
                        image_count = len(
                            _scenecot_image_block_token_counts(
                                response_valid[coac_obs_span[0] : coac_obs_span[1]],
                                image_token_id=image_token_id,
                                vision_start_token_id=vision_start_token_id,
                                vision_end_token_id=vision_end_token_id,
                            )
                        )
                        if image_count <= 0 or image_token_count <= 0:
                            raise ValueError("missing_visual_observation_image")
                    else:
                        image_token_offset = _count_token_id(prompt_valid_ids, image_token_id) + _count_token_id(
                            response_valid[: coac_obs_span[0]],
                            image_token_id,
                        )
                        image_start, image_count = _scenecot_image_row_range_for_token_span(
                            base_mm_inputs,
                            image_token_offset=image_token_offset,
                            image_token_count=image_token_count,
                            merge_size=image_merge_size,
                        )
                    target_image_token_counts = _scenecot_cached_image_feature_counts(
                        base_mm_inputs,
                        merge_size=image_merge_size,
                    )[image_start : image_start + image_count]
                    response_valid, coac_obs_span, obs_context_end_pos = _replace_image_runs_in_visual_span(
                        response_valid,
                        span=coac_obs_span,
                        image_token_id=image_token_id,
                        target_image_token_counts=target_image_token_counts,
                        obs_context_end_pos=obs_context_end_pos,
                        vision_start_token_id=vision_start_token_id,
                        vision_end_token_id=vision_end_token_id,
                    )
                    counterfactual_mm_inputs = _remove_scenecot_image_range_from_mm_inputs(
                        base_mm_inputs,
                        image_start=image_start,
                        image_count=image_count,
                        image_token_counts=image_token_counts,
                        merge_size=image_merge_size,
                    )
                except Exception as exc:
                    skipped += 1
                    visual_build_failed += 1
                    errors.append(f"visual_build_failed:{exc}")
                    skip_reasons[sample_idx] = "visual_build_failed"
                    error_reasons[sample_idx] = str(exc)
                    continue
            elif _span_contains_any_token_id(response_valid, (obs_start, obs_end), mm_token_ids):
                skipped += 1
                visual_obs_skipped += 1
                errors.append("visual_observation_span")
                skip_reasons[sample_idx] = "visual_obs_mode"
                error_reasons[sample_idx] = "visual_observation_span"
                continue
            replacement_ids = placeholder_ids
            counterfactual_visual_mode = False
            if counterfactual_mode == "random_tokens":
                uid_values = batch.non_tensor_batch.get("uid")
                uid = uid_values[sample_idx] if uid_values is not None else sample_idx
                rollout_values = batch.non_tensor_batch.get("rollout_n")
                rollout_n = rollout_values[sample_idx] if rollout_values is not None else None
                replacement_ids = _build_scenecot_random_token_ids(
                    self.tokenizer,
                    length=coac_obs_span[1] - coac_obs_span[0],
                    seed_parts=(
                        random_seed,
                        getattr(self, "global_steps", 0),
                        uid,
                        rollout_n,
                        sample_idx,
                    ),
                    forbidden_token_ids=random_token_forbidden_ids,
                )
                random_token_count += len(replacement_ids)

            observed_response, counterfactual_response = _build_scenecot_coac_response_variants(
                response_valid,
                mode_tag_end_pos=mode_tag_end_pos,
                obs_span=coac_obs_span,
                obs_context_end_pos=obs_context_end_pos,
                answer_open_ids=answer_open_ids,
                gt_ids=gt_ids,
                close_ids=close_ids,
                placeholder_ids=replacement_ids,
            )
            if counterfactual_mode == "random_tool":
                random_tool_prefix_ids = coac_meta.get("random_tool_prefix_ids")
                random_tool_response_ids = coac_meta.get("random_tool_response_ids")
                random_tool_mode = coac_meta.get("random_tool_mode")
                if (
                    not isinstance(random_tool_prefix_ids, (list, tuple))
                    or not random_tool_prefix_ids
                    or not isinstance(random_tool_response_ids, (list, tuple))
                    or not random_tool_response_ids
                    or (
                        random_tool_mode not in COAC_COVERED_MODES
                        and random_tool_mode != COAC_VISUAL_MODE
                    )
                ):
                    skipped += 1
                    errors.append("missing_random_tool_counterfactual")
                    skip_reasons[sample_idx] = "missing_random_tool_counterfactual"
                    error_reasons[sample_idx] = (
                        coac_meta.get("random_tool_error")
                        or "missing_random_tool_counterfactual"
                    )
                    continue
                random_tool_prefix_ids = [int(token_id) for token_id in random_tool_prefix_ids]
                random_tool_response_ids = [int(token_id) for token_id in random_tool_response_ids]
                counterfactual_context = random_tool_prefix_ids + random_tool_response_ids
                if random_tool_mode == COAC_VISUAL_MODE:
                    counterfactual_visual_mode = True
                    if not visual_coac_enabled:
                        skipped += 1
                        skip_reasons[sample_idx] = "visual_obs_mode"
                        error_reasons[sample_idx] = "random_tool_visual_disabled"
                        continue
                    try:
                        payload_span = coac_meta.get("random_tool_payload_span")
                        image_path = coac_meta.get("random_tool_image_path")
                        if not isinstance(payload_span, (list, tuple)) or len(payload_span) != 2:
                            raise ValueError("missing_random_tool_payload_span")
                        if not str(image_path or "").strip():
                            raise ValueError("missing_random_tool_image_path")
                        cache_root = os.getenv("SCENECOT_VISUAL_CACHE_DIR", "").strip()
                        if not cache_root:
                            raise ValueError("random_tool_highlight_requires_visual_cache")
                        from scenecot_verl.agent.visual_cache import Qwen3VLVisualCache

                        resolved_cache_root = os.path.abspath(os.path.expanduser(cache_root))
                        visual_cache = getattr(self, "_scenecot_coac_visual_cache", None)
                        visual_cache_root = getattr(self, "_scenecot_coac_visual_cache_root", None)
                        if visual_cache is None or visual_cache_root != resolved_cache_root:
                            visual_cache = Qwen3VLVisualCache(
                                resolved_cache_root,
                                max_shards=int(os.getenv("SCENECOT_VISUAL_CACHE_MAX_SHARDS", "2")),
                            )
                            self._scenecot_coac_visual_cache = visual_cache
                            self._scenecot_coac_visual_cache_root = resolved_cache_root
                        random_image_mm_inputs = visual_cache.load_images([str(image_path)])
                        prefixed_payload_span = (
                            len(random_tool_prefix_ids) + int(payload_span[0]),
                            len(random_tool_prefix_ids) + int(payload_span[1]),
                        )
                        expanded_span = _expand_visual_obs_span(
                            counterfactual_context,
                            mode_tag_end_pos=len(random_tool_prefix_ids),
                            obs_span=prefixed_payload_span,
                            obs_context_end_pos=len(counterfactual_context),
                            image_token_id=image_token_id,
                            vision_start_token_id=vision_start_token_id,
                            vision_end_token_id=vision_end_token_id,
                        )
                        target_counts = _scenecot_cached_image_feature_counts(
                            random_image_mm_inputs,
                            merge_size=image_merge_size,
                        )
                        counterfactual_context, _, _ = _replace_image_runs_in_visual_span(
                            counterfactual_context,
                            span=expanded_span,
                            image_token_id=image_token_id,
                            target_image_token_counts=target_counts,
                            obs_context_end_pos=len(counterfactual_context),
                            vision_start_token_id=vision_start_token_id,
                            vision_end_token_id=vision_end_token_id,
                        )
                        counterfactual_mm_inputs = _append_scenecot_cached_image_mm_inputs(
                            counterfactual_mm_inputs,
                            random_image_mm_inputs,
                        )
                        if not is_visual_mode:
                            visual_build_attempted += 1
                    except Exception as exc:
                        skipped += 1
                        visual_build_failed += 1
                        errors.append(f"random_tool_visual_build_failed:{exc}")
                        skip_reasons[sample_idx] = "visual_build_failed"
                        error_reasons[sample_idx] = str(exc)
                        continue
                counterfactual_response = counterfactual_context + answer_open_ids + gt_ids + close_ids
                random_tool_count += 1
                random_tool_mode_counts[str(random_tool_mode)] += 1

            if len(observed_response) > response_width or len(counterfactual_response) > response_width:
                skipped += 1
                errors.append("too_long")
                if is_visual_mode or counterfactual_visual_mode:
                    visual_build_failed += 1
                    skip_reasons[sample_idx] = "visual_build_failed"
                    error_reasons[sample_idx] = "too_long"
                else:
                    skip_reasons[sample_idx] = "too_long"
                    error_reasons[sample_idx] = "too_long"
                continue

            sample_pair_start = len(rows)
            candidate_rows = []
            try:
                for candidate_response, candidate_mm_inputs in (
                    (observed_response, observed_mm_inputs),
                    (counterfactual_response, counterfactual_mm_inputs),
                ):
                    response_ids = torch.full((response_width,), self.tokenizer.pad_token_id or 0, dtype=torch.long)
                    response_ids[: len(candidate_response)] = torch.tensor(candidate_response, dtype=torch.long)
                    response_attn = torch.zeros((response_width,), dtype=torch.long)
                    response_attn[: len(candidate_response)] = 1
                    loss_mask = torch.zeros((response_width,), dtype=torch.long)
                    answer_start = len(candidate_response) - len(close_ids) - len(gt_ids)
                    loss_mask[answer_start : answer_start + len(gt_ids)] = 1
                    prompt_attn = attention_mask[sample_idx, :prompt_width]
                    input_ids = torch.cat([prompts[sample_idx], response_ids], dim=0)
                    attn = torch.cat([prompt_attn, response_attn], dim=0)
                    if (
                        candidate_mm_inputs is not None
                        and "token_type_ids" in candidate_mm_inputs
                        and getattr(self.processor, "image_token", None) == "<image_soft_token>"
                    ):
                        if image_token_id is None:
                            raise ValueError("Gemma3Processor did not expose an image token id")
                        # COAC replaces/truncates the sampled response and
                        # teacher-forces the ground-truth answer.  The
                        # sequence-shaped mask produced for the original
                        # rollout is therefore stale.  Gemma 3 marks exactly
                        # its native soft image tokens with type 1, so derive
                        # the candidate mask from the final unpadded ids.
                        candidate_mm_inputs = _copy_scenecot_mm_inputs(candidate_mm_inputs)
                        valid_candidate_ids = input_ids[attn.bool()]
                        candidate_mm_inputs["token_type_ids"] = (
                            valid_candidate_ids.eq(image_token_id).to(dtype=torch.long).unsqueeze(0)
                        )
                    if is_visual_mode or counterfactual_visual_mode:
                        _validate_scenecot_image_count(
                            input_ids=input_ids,
                            attention_mask=attn,
                            mm_inputs=candidate_mm_inputs,
                            image_token_id=image_token_id,
                            merge_size=image_merge_size,
                        )
                    candidate_rows.append(
                        {
                            "prompts": prompts[sample_idx],
                            "responses": response_ids,
                            "response_mask": loss_mask,
                            "attention_mask": attn,
                            "input_ids": input_ids,
                            "multi_modal_inputs": candidate_mm_inputs,
                        }
                    )
                candidate_input_ids = torch.stack([row["input_ids"] for row in candidate_rows], dim=0)
                candidate_attn = torch.stack([row["attention_mask"] for row in candidate_rows], dim=0)
                candidate_mm_inputs = [row["multi_modal_inputs"] for row in candidate_rows]
                candidate_position_ids = self._compute_scenecot_position_ids(
                    candidate_input_ids,
                    candidate_attn,
                    candidate_mm_inputs,
                )
            except Exception as exc:
                if not (is_visual_mode or counterfactual_visual_mode):
                    raise
                skipped += 1
                visual_build_failed += 1
                errors.append(f"visual_position_failed:{exc}")
                skip_reasons[sample_idx] = "visual_build_failed"
                error_reasons[sample_idx] = str(exc)
                continue
            for row_idx, row in enumerate(candidate_rows):
                row["position_ids"] = candidate_position_ids[row_idx]
            if is_visual_mode or counterfactual_visual_mode:
                visual_build_valid += 1
            skip_reasons[sample_idx] = "built"
            rows.extend(candidate_rows)
            sample_pairs.append((sample_idx, sample_pair_start))

        if not rows:
            visual_build_valid_rate = (
                visual_build_valid / visual_build_attempted if visual_build_attempted else 0.0
            )
            empty_metrics = {
                "coac/build_skipped": skipped,
                "coac/build_valid": 0,
                "coac/build_error_count": len(errors),
                "coac/build_visual_obs_skipped": visual_obs_skipped,
                "coac/visual_build_attempted": visual_build_attempted,
                "coac/visual_build_valid": visual_build_valid,
                "coac/visual_build_failed": visual_build_failed,
                "coac/visual_build_valid_rate": visual_build_valid_rate,
                "coac/counterfactual_mask": int(counterfactual_mode == "mask"),
                "coac/counterfactual_random_tokens": int(counterfactual_mode == "random_tokens"),
                "coac/counterfactual_random_tool": int(counterfactual_mode == "random_tool"),
                "coac/random_token_count": random_token_count,
                "coac/random_tool_count": random_tool_count,
            }
            for mode_name, count in sorted(random_tool_mode_counts.items()):
                empty_metrics[f"coac/random_tool_mode/{mode_name}/count"] = count
            return None, [], empty_metrics, skip_reasons, error_reasons

        input_ids = torch.stack([row["input_ids"] for row in rows], dim=0)
        attn = torch.stack([row["attention_mask"] for row in rows], dim=0)
        mm_inputs = [row["multi_modal_inputs"] for row in rows]
        position_ids = torch.stack([row["position_ids"] for row in rows], dim=0)
        non_tensors = {}
        if any(item is not None for item in mm_inputs):
            temp_arr = np.empty(len(mm_inputs), dtype=object)
            temp_arr[:] = mm_inputs
            non_tensors["multi_modal_inputs"] = temp_arr
        coac_batch = DataProto.from_dict(
            tensors={
                "prompts": torch.stack([row["prompts"] for row in rows], dim=0),
                "responses": torch.stack([row["responses"] for row in rows], dim=0),
                "response_mask": torch.stack([row["response_mask"] for row in rows], dim=0),
                "input_ids": input_ids,
                "attention_mask": attn,
                "position_ids": position_ids,
                "temperature": torch.full(
                    (len(rows),),
                    float(getattr(self.config.actor_rollout_ref.rollout, "temperature", 1.0)),
                    dtype=torch.float32,
                ),
            },
            non_tensors=non_tensors,
        )
        visual_build_valid_rate = (
            visual_build_valid / visual_build_attempted if visual_build_attempted else 0.0
        )
        build_metrics = {
            "coac/build_skipped": skipped,
            "coac/build_valid": len(sample_pairs),
            "coac/build_error_count": len(errors),
            "coac/build_visual_obs_skipped": visual_obs_skipped,
            "coac/visual_build_attempted": visual_build_attempted,
            "coac/visual_build_valid": visual_build_valid,
            "coac/visual_build_failed": visual_build_failed,
            "coac/visual_build_valid_rate": visual_build_valid_rate,
            "coac/counterfactual_mask": int(counterfactual_mode == "mask"),
            "coac/counterfactual_random_tokens": int(counterfactual_mode == "random_tokens"),
            "coac/counterfactual_random_tool": int(counterfactual_mode == "random_tool"),
            "coac/random_token_count": random_token_count,
            "coac/random_tool_count": random_tool_count,
        }
        for mode_name, count in sorted(random_tool_mode_counts.items()):
            build_metrics[f"coac/random_tool_mode/{mode_name}/count"] = count
        return coac_batch, sample_pairs, build_metrics, skip_reasons, error_reasons

    def _compute_scenecot_coac(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        if not _env_bool("SCENECOT_COAC_ENABLE", False):
            return

        delta_values = np.zeros(len(batch), dtype=np.float32)
        valid_values = np.zeros(len(batch), dtype=bool)
        group_norm_values = np.zeros(len(batch), dtype=np.float32)
        group_size_values = np.zeros(len(batch), dtype=np.int32)
        group_std_values = np.zeros(len(batch), dtype=np.float32)
        r_tool_override_values = np.zeros(len(batch), dtype=np.float32)
        obs_text_values = np.empty(len(batch), dtype=object)
        obs_text_values[:] = ""
        for idx in range(len(batch)):
            tool_extra = _scenecot_tool_extra_for_row(batch, idx)
            coac_meta = tool_extra.get("scenecot_coac", {}) if isinstance(tool_extra, dict) else {}
            if isinstance(coac_meta, dict):
                obs_text_values[idx] = coac_meta.get("obs_text", "") or ""

        coac_batch, sample_pairs, build_metrics, skip_reasons, error_reasons = self._build_scenecot_coac_batch(batch)
        metrics.update(build_metrics)
        error_counts = Counter(_metric_label(value) for value in error_reasons if value)
        for label, count in sorted(error_counts.items()):
            metrics[f"coac/build_error_reason/{label}/count"] = count
        if coac_batch is None:
            batch.non_tensor_batch["coac_delta"] = delta_values
            batch.non_tensor_batch["coac_valid"] = valid_values
            batch.non_tensor_batch["coac_delta_group_norm"] = group_norm_values
            batch.non_tensor_batch["coac_group_size"] = group_size_values
            batch.non_tensor_batch["coac_group_std"] = group_std_values
            batch.non_tensor_batch["coac_r_tool_override"] = r_tool_override_values
            batch.non_tensor_batch["coac_obs_text"] = obs_text_values
            batch.non_tensor_batch["coac_skip_reason"] = skip_reasons
            batch.non_tensor_batch["coac_error_reason"] = error_reasons
            return

        rollout_config = self.config.actor_rollout_ref.rollout
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        log_prob_micro_batch_size = 1
        if not rollout_config.get("log_prob_use_dynamic_bsz", False):
            log_prob_micro_batch_size = int(rollout_config.get("log_prob_micro_batch_size_per_gpu") or 1)
        coac_size_divisor = max(1, dp_size * log_prob_micro_batch_size)
        coac_batch_for_logprob, coac_pad_size = pad_dataproto_to_divisor(coac_batch, coac_size_divisor)
        if coac_pad_size:
            metrics["coac/logprob_pad_size"] = float(coac_pad_size)
            metrics["coac/logprob_size_divisor"] = float(coac_size_divisor)

        batch_td = coac_batch_for_logprob.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
        # COAC is a counterfactual under the current actor policy.  Do not call
        # ref_policy_wg/compute_ref_log_prob here; ref may share weights with
        # actor but disables LoRA, which would measure the wrong policy.
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if coac_pad_size:
            log_probs = log_probs[:-coac_pad_size]
        loss_mask = coac_batch.batch["response_mask"].to(log_probs.device).float()
        token_counts = loss_mask.sum(dim=-1).clamp_min(1.0)
        sequence_scores = (log_probs.float() * loss_mask).sum(dim=-1) / token_counts

        for sample_idx, pair_start in sample_pairs:
            observed = float(sequence_scores[pair_start].item())
            masked = float(sequence_scores[pair_start + 1].item())
            delta_values[sample_idx] = observed - masked
            valid_values[sample_idx] = np.isfinite(delta_values[sample_idx])

        if _env_bool(COAC_GROUP_ZSCORE_ENV, False):
            fallback_credit = _env_float(COAC_FALLBACK_CREDIT_ENV, 0.5)
            group_ids = batch.non_tensor_batch.get("uid")
            group_norm_values, group_size_values, group_std_values, r_tool_override_values = (
                _compute_scenecot_group_zscore_credit(
                    delta_values,
                    valid_values,
                    group_ids,
                    fallback_credit=fallback_credit,
                )
            )

        batch.non_tensor_batch["coac_delta"] = delta_values
        batch.non_tensor_batch["coac_valid"] = valid_values
        batch.non_tensor_batch["coac_delta_group_norm"] = group_norm_values
        batch.non_tensor_batch["coac_group_size"] = group_size_values
        batch.non_tensor_batch["coac_group_std"] = group_std_values
        batch.non_tensor_batch["coac_r_tool_override"] = r_tool_override_values
        batch.non_tensor_batch["coac_obs_text"] = obs_text_values
        batch.non_tensor_batch["coac_skip_reason"] = skip_reasons
        batch.non_tensor_batch["coac_error_reason"] = error_reasons
        finite_delta = delta_values[valid_values]
        if finite_delta.size:
            metrics["coac/delta_mean"] = float(np.mean(finite_delta))
            metrics["coac/delta_p50"] = float(np.percentile(finite_delta, 50))
            metrics["coac/delta_p90"] = float(np.percentile(finite_delta, 90))
        finite_r_tool = r_tool_override_values[valid_values]
        if _env_bool(COAC_GROUP_ZSCORE_ENV, False) and finite_r_tool.size:
            metrics["coac/group_zscore_enabled"] = 1.0
            metrics["coac/group_r_tool_mean"] = float(np.mean(finite_r_tool))
            metrics["coac/group_r_tool_p50"] = float(np.percentile(finite_r_tool, 50))

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key == "reward":
                    continue
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        _add_validation_count_metrics(
            metric_dict,
            data_sources,
            reward_extra_infos_dict.get("first_visual_cue"),
            "first_visual_cue",
        )
        _add_validation_count_metrics(
            metric_dict,
            data_sources,
            reward_extra_infos_dict.get("expected_first_visual_cue"),
            "expected_first_visual_cue",
        )

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        rollout_standalone = self.config.actor_rollout_ref.rollout.get("standalone", False)
        if rollout_standalone:
            if self.config.actor_rollout_ref.rollout.nnodes <= 0:
                raise ValueError(
                    "actor_rollout_ref.rollout.nnodes must be > 0 when "
                    "actor_rollout_ref.rollout.standalone=True"
                )
            if self.config.actor_rollout_ref.rollout.n_gpus_per_node <= 0:
                raise ValueError(
                    "actor_rollout_ref.rollout.n_gpus_per_node must be > 0 when "
                    "actor_rollout_ref.rollout.standalone=True"
                )

        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            worker_group=None if rollout_standalone else self.actor_rollout_wg,
            rollout_resource_pool=None if rollout_standalone else actor_rollout_resource_pool,
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _prepare_scenecot_llava_format_aux(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        """Pair PPO rows with canonical LLaVA XML teacher-forcing rows."""
        if not _env_bool(LLAVA_FORMAT_AUX_ENABLE_ENV, False):
            return batch, {}
        if not is_llava_processor(self.processor):
            processor_name = type(self.processor).__name__ if self.processor is not None else None
            raise RuntimeError(
                f"{LLAVA_FORMAT_AUX_ENABLE_ENV}=1 is only supported by LLaVA; "
                f"active processor is {processor_name!r}."
            )

        coefficient = _env_float(LLAVA_FORMAT_AUX_COEF_ENV, 1.0)
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(f"{LLAVA_FORMAT_AUX_COEF_ENV} must be finite and > 0, got {coefficient!r}.")

        reasoning_max_tokens = _env_int(LLAVA_FORMAT_AUX_REASONING_MAX_TOKENS_ENV, 48)
        answer_max_tokens = _env_int(LLAVA_FORMAT_AUX_ANSWER_MAX_TOKENS_ENV, 16)
        if reasoning_max_tokens <= 0:
            raise ValueError(
                f"{LLAVA_FORMAT_AUX_REASONING_MAX_TOKENS_ENV} must be > 0, got {reasoning_max_tokens!r}."
            )
        if answer_max_tokens <= 0:
            raise ValueError(
                f"{LLAVA_FORMAT_AUX_ANSWER_MAX_TOKENS_ENV} must be > 0, got {answer_max_tokens!r}."
            )

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        attention_mask = batch.batch["attention_mask"]
        prompt_width = int(prompts.shape[1])
        response_width = int(responses.shape[1])
        valid_response_mask = attention_mask[:, -response_width:].to(torch.bool)
        token_ids = resolve_llava_format_token_ids(self.tokenizer)

        obs_context_end_positions: list[int | None] = []
        autoclose_counts: list[int] = []
        for row_index in range(len(batch)):
            tool_extra = _scenecot_tool_extra_for_row(batch, row_index)
            coac_meta = tool_extra.get("scenecot_coac")
            obs_context_end = coac_meta.get("obs_context_end_pos") if isinstance(coac_meta, dict) else None
            try:
                obs_context_end_positions.append(None if obs_context_end is None else int(obs_context_end))
            except (TypeError, ValueError):
                obs_context_end_positions.append(None)

            autoclose_value = _row_non_tensor(batch, "final_answer_autoclosed_count", row_index, 0)
            try:
                autoclose_counts.append(int(autoclose_value or 0))
            except (TypeError, ValueError):
                autoclose_counts.append(0)

        # Retain sampled-prefix diagnostics so a collapsing CE loss cannot be
        # mistaken for an increase in actual tag-generation frequency.
        sampled_format_mask = build_llava_format_aux_mask(
            responses=responses,
            actor_response_mask=response_mask,
            valid_response_mask=valid_response_mask,
            obs_context_end_positions=obs_context_end_positions,
            autoclose_counts=autoclose_counts,
            format_token_ids=token_ids,
        )

        pad_token_id = int(self.tokenizer.pad_token_id or 0)
        fallback_body_ids = self.tokenizer.encode("unknown", add_special_tokens=False)
        ignored_token_ids = {
            int(token_id)
            for token_id in (getattr(self.tokenizer, "all_special_ids", None) or [])
            if token_id is not None
        }

        canonical_responses = responses.clone()
        canonical_attention_mask = attention_mask.clone()
        canonical_input_ids = batch.batch["input_ids"].clone()
        canonical_response_mask = torch.zeros_like(response_mask)
        canonical_format_mask = torch.zeros_like(responses, dtype=torch.bool)
        built_rows = 0

        for row_index in range(len(batch)):
            valid_response_len = int(valid_response_mask[row_index].sum().item())
            canonical_target = build_llava_canonical_format_target(
                response_valid=responses[row_index, :valid_response_len].tolist(),
                actor_mask_valid=response_mask[row_index, :valid_response_len].tolist(),
                obs_context_end_pos=obs_context_end_positions[row_index],
                response_width=response_width,
                format_token_ids=token_ids,
                fallback_body_ids=fallback_body_ids,
                ignored_token_ids=ignored_token_ids,
                reasoning_max_tokens=reasoning_max_tokens,
                answer_max_tokens=answer_max_tokens,
            )
            if canonical_target is None:
                continue
            canonical_valid, target_mask_valid = canonical_target
            canonical_len = len(canonical_valid)

            canonical_responses[row_index].fill_(pad_token_id)
            canonical_responses[row_index, :canonical_len] = torch.tensor(
                canonical_valid,
                dtype=responses.dtype,
                device=responses.device,
            )
            canonical_format_mask[row_index, :canonical_len] = torch.tensor(
                target_mask_valid,
                dtype=torch.bool,
                device=responses.device,
            )
            canonical_attention_mask[row_index, prompt_width:].zero_()
            canonical_attention_mask[row_index, prompt_width : prompt_width + canonical_len] = 1
            canonical_input_ids[row_index] = torch.cat(
                (prompts[row_index], canonical_responses[row_index]),
                dim=0,
            )
            built_rows += 1

        metrics = {
            "train/scenecot/format_aux/enabled": 1.0,
            "train/scenecot/format_aux/coefficient": coefficient,
            "train/scenecot/format_aux/canonical_sequence_count": float(built_rows),
            "train/scenecot/format_aux/sequence_rate": float(built_rows / len(batch)) if len(batch) else 0.0,
            "train/scenecot/format_aux/token_count": float(canonical_format_mask.sum().item()),
            "train/scenecot/format_aux/reasoning_max_tokens": float(reasoning_max_tokens),
            "train/scenecot/format_aux/answer_max_tokens": float(answer_max_tokens),
        }
        tag_labels = {
            "<reasoning>": "reasoning_open",
            "</reasoning>": "reasoning_close",
            "<answer>": "answer_open",
            "</answer>": "answer_close",
        }
        for tag in LLAVA_FORMAT_TAGS:
            sampled_tag_count = float((sampled_format_mask & (responses == token_ids[tag])).sum().item())
            target_tag_count = float((canonical_format_mask & (canonical_responses == token_ids[tag])).sum().item())
            label = tag_labels[tag]
            metrics[f"train/scenecot/format_aux/{label}_count"] = target_tag_count
            metrics[f"train/scenecot/format_aux/sampled_{label}_count"] = sampled_tag_count
            metrics[f"train/scenecot/format_aux/repaired_{label}_count"] = max(
                0.0,
                target_tag_count - sampled_tag_count,
            )

        if built_rows == 0:
            logger.warning(
                "%s=1, but no canonical LLaVA format targets could be built for this actor batch.",
                LLAVA_FORMAT_AUX_ENABLE_ENV,
            )
            return batch, metrics

        mm_inputs_source = batch.non_tensor_batch.get("multi_modal_inputs")
        multi_modal_inputs = [
            mm_inputs_source[row_index] if mm_inputs_source is not None else None
            for row_index in range(len(batch))
        ]
        canonical_position_ids = self._compute_scenecot_position_ids(
            canonical_input_ids,
            canonical_attention_mask,
            multi_modal_inputs,
        )

        original_tensors = {key: value for key, value in batch.batch.items()}
        original_tensors["scenecot_format_aux_mask"] = torch.zeros_like(responses, dtype=torch.bool)

        canonical_tensors = {key: value.clone() for key, value in batch.batch.items()}
        canonical_tensors.update(
            {
                "responses": canonical_responses,
                "response_mask": canonical_response_mask,
                "attention_mask": canonical_attention_mask,
                "input_ids": canonical_input_ids,
                "position_ids": canonical_position_ids,
                "scenecot_format_aux_mask": canonical_format_mask,
            }
        )
        # Synthetic rows must never enter PPO, KL, entropy, value, or rollout
        # correction objectives. Their only nonzero loss mask is the four-token
        # SceneCOT auxiliary mask above.
        for key in (
            "advantages",
            "returns",
            "old_log_probs",
            "ref_log_prob",
            "rollout_log_probs",
            "rollout_is_weights",
            "token_level_scores",
            "token_level_rewards",
            "values",
        ):
            value = canonical_tensors.get(key)
            if isinstance(value, torch.Tensor):
                canonical_tensors[key] = torch.zeros_like(value)

        original_batch = DataProto.from_dict(
            tensors=original_tensors,
            non_tensors={key: value for key, value in batch.non_tensor_batch.items()},
            meta_info={},
        )
        canonical_batch = DataProto.from_dict(
            tensors=canonical_tensors,
            non_tensors={key: value.copy() for key, value in batch.non_tensor_batch.items()},
            meta_info={},
        )
        actor_batch = DataProto.concat([original_batch, canonical_batch])
        pair_indices = torch.stack(
            (
                torch.arange(len(batch), dtype=torch.long),
                torch.arange(len(batch), 2 * len(batch), dtype=torch.long),
            ),
            dim=1,
        ).reshape(-1)
        actor_batch.reorder(pair_indices)
        actor_batch.meta_info = dict(batch.meta_info)
        actor_batch.meta_info["scenecot_format_aux_paired"] = True
        actor_batch.meta_info["global_token_num"] = torch.sum(
            actor_batch.batch["attention_mask"],
            dim=-1,
        ).tolist()
        if "images_seqlens" in actor_batch.meta_info:
            original_images_seqlens = list(actor_batch.meta_info.get("images_seqlens") or [])
            actor_batch.meta_info["images_seqlens"] = original_images_seqlens + original_images_seqlens

        metrics["train/scenecot/format_aux/actor_batch_multiplier"] = 2.0
        return actor_batch, metrics

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        paired_format_aux = bool(batch.meta_info.get("scenecot_format_aux_paired", False))
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        scenecot_format_aux_coef = (
            _env_float(LLAVA_FORMAT_AUX_COEF_ENV, 1.0)
            if "scenecot_format_aux_mask" in batch_td
            else 0.0
        )
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        actor_mini_batch_size = ppo_mini_batch_size * (2 if paired_format_aux else 1)
        if len(batch_td) % actor_mini_batch_size != 0:
            raise ValueError(
                f"actor batch size {len(batch_td)} must be divisible by mini batch size "
                f"{actor_mini_batch_size} (paired_format_aux={paired_format_aux})."
            )
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=actor_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            # Canonical rows are interleaved with their PPO source rows. Keep
            # each optimizer mini-batch exactly balanced between the two.
            dataloader_kwargs={"shuffle": False if paired_format_aux else shuffle},
            compute_loss=True,
            scenecot_format_aux_coef=scenecot_format_aux_coef,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("coac", timing_raw, color="orange"):
                        self._compute_scenecot_coac(batch, metrics)
                        if _env_bool("SCENECOT_COAC_ENABLE", False) and "rm_scores" in batch.batch.keys():
                            _drop_stale_scenecot_reward_extra_for_recompute(batch)
                            batch.pop(batch_keys=["rm_scores"])

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if (
                            (_env_bool("SCENECOT_COAC_ENABLE", False) or self.use_rm)
                            and "rm_scores" not in batch.batch.keys()
                        ):
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    diagnostic_rollout_only = _env_bool("SCENECOT_DIAGNOSTIC_ROLLOUT_ONLY", False)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if diagnostic_rollout_only:
                        # Diagnostic runs only need rollout + reward/COAC dumps.  Avoid
                        # PPO logprob/update forwards so multimodal rollouts with appended
                        # tool observations cannot fail this lightweight inspection path.
                        batch.batch["old_log_probs"] = torch.zeros_like(
                            batch.batch["responses"], dtype=torch.float32
                        )
                        metrics["diagnostic/rollout_only"] = 1.0
                    elif bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy and diagnostic_rollout_only:
                        batch.batch["ref_log_prob"] = torch.zeros_like(
                            batch.batch["responses"], dtype=torch.float32
                        )
                    elif self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if diagnostic_rollout_only:
                        metrics["diagnostic/skipped_actor_update"] = 1.0
                        # Generation always sleeps rollout replicas before
                        # actor-side scoring. Diagnostic runs skip actor update,
                        # so explicitly wake the rollout engine for the next
                        # batch without changing training weights.
                        with marked_timer("diagnostic_update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)
                    elif self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # update actor
                        actor_batch, format_aux_metrics = self._prepare_scenecot_llava_format_aux(batch)
                        metrics.update(format_aux_metrics)
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(actor_batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(_compute_train_scenecot_reward_metrics(batch.non_tensor_batch))
                metrics.update(_compute_scenecot_grpo_group_metrics(batch))
                _check_scenecot_step1_format_reward_sentinel(metrics, self.global_steps)
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
