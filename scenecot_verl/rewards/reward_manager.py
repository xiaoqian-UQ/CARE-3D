from __future__ import annotations

from collections import defaultdict
import json
import os
from typing import Any

import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register as register_async
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager as AsyncNaiveRewardManager
from verl.workers.reward_manager import register as register_legacy

from scenecot_verl.rewards.scenecot_reward import _merge_scenecot_reward_extra_info


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain_value(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _as_metrics_dict(value: Any) -> dict[str, Any]:
    value = _plain_value(value)
    return value if isinstance(value, dict) else {}


def _valid_response_values(data_item: Any, key: str) -> list[Any]:
    try:
        response_values = data_item.batch[key]
        attention_values = data_item.batch["attention_mask"]
    except Exception:
        return []
    response_values = _plain_value(response_values)
    attention_values = _plain_value(attention_values)
    if (
        not isinstance(response_values, list)
        or not isinstance(attention_values, list)
        or not response_values
        or not attention_values
    ):
        return []
    response_width = len(response_values)
    response_attention = attention_values[-response_width:]
    return [value for value, keep in zip(response_values, response_attention, strict=False) if int(keep or 0)]


def _actor_generated_response_text(data_item: Any, tokenizer, valid_response_ids: torch.Tensor) -> str:
    try:
        response_mask = data_item.batch["response_mask"][: valid_response_ids.shape[-1]].to(torch.bool)
    except Exception:
        return ""
    if response_mask.numel() != valid_response_ids.shape[-1] or not bool(response_mask.any().item()):
        return ""
    actor_ids = valid_response_ids[response_mask]
    if actor_ids.numel() == 0:
        return ""
    return tokenizer.decode(actor_ids, skip_special_tokens=True)


def _scenecot_tool_extra_fields(data_item: Any) -> dict[str, Any]:
    tool_extra_fields = _as_metrics_dict(data_item.non_tensor_batch.get("tool_extra_fields", None))
    if tool_extra_fields:
        return tool_extra_fields

    rebuilt: dict[str, Any] = {}
    for key in (
        "scenecot_coac",
        "scenecot_actions",
        "scenecot_metrics",
        "rollout_step",
        "rollout_n",
        "sample_index",
    ):
        value = data_item.non_tensor_batch.get(key, None)
        if value is not None:
            rebuilt[key] = _plain_value(value)
    return rebuilt


def _rollout_dump_path() -> str | None:
    dump_dir = os.getenv("SCENECOT_ROLLOUT_DUMP_DIR", "").strip()
    if not dump_dir:
        return None
    os.makedirs(dump_dir, exist_ok=True)
    return os.path.join(dump_dir, "rollouts_scored.jsonl")


def _write_scored_rollout_dump(data_item: Any, response_str: str, reward_extra_info: dict[str, Any]) -> None:
    path = _rollout_dump_path()
    if path is None:
        return
    extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
    tool_extra_fields = _scenecot_tool_extra_fields(data_item)
    coac_meta = _as_metrics_dict(tool_extra_fields.get("scenecot_coac"))
    scenecot_metrics = _as_metrics_dict(tool_extra_fields.get("scenecot_metrics"))
    actions = tool_extra_fields.get("scenecot_actions") if isinstance(tool_extra_fields, dict) else None
    first_action = actions[0] if isinstance(actions, list) and actions else {}
    forced_mode = scenecot_metrics.get("gcs_forced_mode")
    is_coverage = int(bool(scenecot_metrics.get("gcs_forced_count") or forced_mode))
    coverage_fallback = int(bool(scenecot_metrics.get("gcs_coverage_fallback_count")))
    public_response_text = tool_extra_fields.get("response_text")
    if not isinstance(public_response_text, str) or not public_response_text.strip():
        public_response_text = response_str
    full_protocol_valid = reward_extra_info.get("full_protocol_valid")
    coac_skip_reason = data_item.non_tensor_batch.get("coac_skip_reason", reward_extra_info.get("coac_skip_reason"))
    if (
        reward_extra_info.get("coac_enabled")
        and not int(full_protocol_valid or 0)
        and coac_skip_reason in (None, "", "built")
    ):
        coac_skip_reason = "protocol_invalid"
    payload = {
        "step": tool_extra_fields.get("rollout_step"),
        "sample_index": tool_extra_fields.get("sample_index"),
        "rollout_n": tool_extra_fields.get("rollout_n"),
        "uid": data_item.non_tensor_batch.get("uid", None),
        "scene_id": extra_info.get("scene_id") if isinstance(extra_info, dict) else None,
        "question_id": extra_info.get("question_id") if isinstance(extra_info, dict) else None,
        "question_type": (extra_info.get("question_type") or extra_info.get("qa_type")) if isinstance(extra_info, dict) else None,
        "dataset_name": extra_info.get("dataset_name") if isinstance(extra_info, dict) else None,
        "mode": first_action.get("mode") if isinstance(first_action, dict) else None,
        "is_coverage": is_coverage,
        "forced_mode": forced_mode,
        "coverage_fallback": coverage_fallback,
        "mode_raw": first_action.get("trigger") if isinstance(first_action, dict) else None,
        "tool_tag": first_action.get("trigger") if isinstance(first_action, dict) else None,
        "think_grd_query": first_action.get("query") if isinstance(first_action, dict) else None,
        "tool_called": coac_meta.get("tool_called"),
        "obs_span": coac_meta.get("obs_span"),
        "obs_context_end_pos": coac_meta.get("obs_context_end_pos"),
        "mode_tag_end_pos": coac_meta.get("mode_tag_end_pos"),
        "answer_tag_pos": coac_meta.get("answer_tag_pos"),
        "gt_answer_text": coac_meta.get("gt_answer_text"),
        "observation": coac_meta.get("obs_text"),
        "answer": reward_extra_info.get("qwen25_response_text", public_response_text),
        "gt_answer": data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None),
        "score": reward_extra_info.get("score"),
        "reward": reward_extra_info.get("reward"),
        "acc": reward_extra_info.get("acc"),
        "reward_before_clip": reward_extra_info.get("reward_before_clip"),
        "reward_final_before_clip": reward_extra_info.get("reward/final_reward_before_clip"),
        "reward_final_after_clip": reward_extra_info.get("reward/final_reward_after_clip"),
        "format_reward": reward_extra_info.get("format_reward"),
        "format_gate_tier": reward_extra_info.get("format_gate_tier"),
        "format_autoclose_reward": reward_extra_info.get("format_autoclose_reward"),
        "format_minor_violation": reward_extra_info.get("format_minor_violation"),
        "final_answer_autoclosed_count": reward_extra_info.get("final_answer_autoclosed_count"),
        "final_answer_autoclose_eligible_count": reward_extra_info.get(
            "final_answer_autoclose_eligible_count"
        ),
        "answer_reward": reward_extra_info.get("answer_reward"),
        "answer_reward_gated": reward_extra_info.get("answer_reward_gated"),
        "answer_autoclose_scoring_enabled": reward_extra_info.get(
            "answer_autoclose_scoring_enabled"
        ),
        "answer_reward_raw": reward_extra_info.get("answer_reward_raw"),
        "answer_rule_decision": reward_extra_info.get("answer_rule_decision"),
        "answer_exact": reward_extra_info.get("answer_exact"),
        "answer_token_f1": reward_extra_info.get("answer_token_f1"),
        "pred_answer": reward_extra_info.get("pred_answer"),
        "pred_answer_raw": reward_extra_info.get("pred_answer_raw"),
        "format_valid": reward_extra_info.get("full_protocol_valid"),
        "full_protocol_valid": reward_extra_info.get("full_protocol_valid"),
        "structural_protocol_valid": reward_extra_info.get("structural_protocol_valid"),
        "catastrophic_invalid": reward_extra_info.get("catastrophic_invalid"),
        "catastrophic_reward_value": reward_extra_info.get("catastrophic_reward_value"),
        "invalid_reason": (
            reward_extra_info.get("catastrophic_invalid_reason")
            or reward_extra_info.get("hard_invalid_reason")
            or reward_extra_info.get("final_turn_invalid_reason")
            or reward_extra_info.get("first_turn_invalid_reason")
            or (
                "final_answer_autoclosed"
                if reward_extra_info.get("format_gate_tier") == "minor_violation"
                else None
            )
            or (None if reward_extra_info.get("full_protocol_valid") else "unknown_protocol_invalid")
        ),
        "first_turn_invalid_reason": reward_extra_info.get("first_turn_invalid_reason"),
        "final_turn_invalid_reason": reward_extra_info.get("final_turn_invalid_reason"),
        "hard_invalid_reason": reward_extra_info.get("hard_invalid_reason"),
        "catastrophic_invalid_reason": reward_extra_info.get("catastrophic_invalid_reason"),
        "first_generation_finish_reason": scenecot_metrics.get("first_generation_finish_reason"),
        "final_generation_finish_reason": scenecot_metrics.get("final_generation_finish_reason"),
        "first_generation_native_finish_reason": scenecot_metrics.get("first_generation_native_finish_reason"),
        "final_generation_native_finish_reason": scenecot_metrics.get("final_generation_native_finish_reason"),
        "coac_enabled": reward_extra_info.get("coac_enabled"),
        "coac_delta": reward_extra_info.get("coac_delta"),
        "coac_delta_group_norm": reward_extra_info.get("coac_delta_group_norm"),
        "coac_group_size": reward_extra_info.get("coac_group_size"),
        "coac_group_std": reward_extra_info.get("coac_group_std"),
        "coac_tau": reward_extra_info.get("coac_tau"),
        "coac_r_tool": reward_extra_info.get("coac_r_tool"),
        "coac_r_tool_override": reward_extra_info.get("coac_r_tool_override"),
        "coac_fixed_r_tool": reward_extra_info.get("coac_fixed_r_tool"),
        "coac_fallback_credit": reward_extra_info.get("coac_fallback_credit"),
        "coac_tool_credit_source": reward_extra_info.get("coac_tool_credit_source"),
        "coac_fixed_fallback_applied": reward_extra_info.get("coac_fixed_fallback_applied"),
        "coac/fallback_credit_rate": reward_extra_info.get("coac/fallback_credit_rate"),
        "coac_valid": reward_extra_info.get("coac_valid"),
        "coac_copy_overlap": reward_extra_info.get("coac_copy_overlap"),
        "coac_copy_thresh": reward_extra_info.get("coac_copy_thresh"),
        "coac_copy_discount": reward_extra_info.get("coac_copy_discount"),
        "coac_copy_discount_enabled": reward_extra_info.get("coac_copy_discount_enabled"),
        "coac_copy_discount_applied": reward_extra_info.get("coac_copy_discount_applied"),
        "reward/w_format": reward_extra_info.get("reward/w_format"),
        "reward/w_answer": reward_extra_info.get("reward/w_answer"),
        "reward/w_tool": reward_extra_info.get("reward/w_tool"),
        "w_format": reward_extra_info.get("w_format"),
        "w_answer": reward_extra_info.get("w_answer"),
        "w_tool": reward_extra_info.get("w_tool"),
        "penalty": reward_extra_info.get("penalty"),
        "penalty_total": reward_extra_info.get("penalty_total"),
        "coac_skip_reason": coac_skip_reason,
        "coac_error_reason": (
            data_item.non_tensor_batch.get("coac_error_reason")
            or reward_extra_info.get("coac_error_reason")
        ),
        "coac_covered_modes": ["prob", "loc_rect", "loc_polar"],
        "response_ids": _valid_response_values(data_item, "responses"),
        "response_mask": _valid_response_values(data_item, "response_mask"),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_plain_value(payload), ensure_ascii=False) + "\n")


def _extra_info_with_tool_metrics(data_item: Any) -> dict[str, Any]:
    extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
    return _merge_scenecot_reward_extra_info(extra_info, data_item.non_tensor_batch)


@register_legacy("scenecot_naive")
@register_async("scenecot_naive")
class SceneCOTNaiveRewardManager(AsyncNaiveRewardManager):
    """Naive reward manager that exposes SceneCOT rollout tool metrics to the reward fn."""

    def __init__(
        self,
        config=None,
        tokenizer=None,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        num_examine: int | None = None,
        reward_fn_key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
        self.num_examine = 0 if num_examine is None else num_examine
        self.reward_fn_key = reward_fn_key or "data_source"

    @staticmethod
    def _extract_reward_from_rm_scores(
        data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any] | None:
        if "rm_scores" not in data.batch.keys():
            return None

        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        return data.batch["rm_scores"]

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        data = data[-1:]
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = _extra_info_with_tool_metrics(data_item)
        tool_extra_fields = _as_metrics_dict(data_item.non_tensor_batch.get("tool_extra_fields", None))
        if tool_extra_fields:
            extra_info.update(tool_extra_fields)
            scenecot_metrics = _as_metrics_dict(tool_extra_fields.get("scenecot_metrics", None))
            if scenecot_metrics:
                extra_info["scenecot_tool_metrics"] = scenecot_metrics

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores
        extra_info["scenecot_actor_generated_text"] = _actor_generated_response_text(
            data_item,
            self.tokenizer,
            valid_response_ids,
        )

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            reward = result["score"]
            reward_extra_info.update(result)
        else:
            reward = result
            reward_extra_info["acc"] = reward
        _write_scored_rollout_dump(data_item, response_str, reward_extra_info)

        # Async reward results are unioned back into the rollout batch.  Keep
        # dump payloads complete, but do not return keys that already exist in
        # the rollout non-tensor batch; those may legitimately carry different
        # rollout-side values and would make DataProto.union treat the merge as
        # a conflicting overwrite.
        existing_keys = set(data_item.non_tensor_batch.keys())
        reward_extra_info_for_batch = {
            key: value for key, value in reward_extra_info.items() if key not in existing_keys
        }
        return {"reward_score": reward, "reward_extra_info": reward_extra_info_for_batch}

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = _extra_info_with_tool_metrics(data_item)
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["scenecot_actor_generated_text"] = _actor_generated_response_text(
                data_item,
                self.tokenizer,
                valid_response_ids,
            )

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                _write_scored_rollout_dump(data_item, response_str, score)
            else:
                reward = score
                _write_scored_rollout_dump(data_item, response_str, {"score": reward, "acc": reward})

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
