from __future__ import annotations

import ast
import json
import math
import os
import re
import string
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

from scenecot_verl.rewards.answer_judge_client import JudgeOutputTruncatedError, request_judge

from scenecot_verl.protocol import (
    extract_final_assistant_turn,
    max_repeated_ngram_count,
    max_repeated_sentence_count,
    repeated_punct_penalty,
)


ANSWER_RE = re.compile(r"<answer\s*>(.*?)(?:</answer>|$)", flags=re.DOTALL | re.IGNORECASE)
COMPLETE_ANSWER_RE = re.compile(r"<answer\s*>(.*?)</answer\s*>", flags=re.DOTALL | re.IGNORECASE)
REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", flags=re.DOTALL | re.IGNORECASE)
THINK_GRD_RE = re.compile(r"<think_grd>(.*?)</think_grd>", flags=re.DOTALL | re.IGNORECASE)
TOOL_CALL_TOKENS = (
    "<list_obj_prob>",
    "<list_obj_loc_prob>",
    "<list_obj_loc_plr_prob>",
    "<highlight_obj>",
)
TOOL_TAG_NAMES = tuple(token.strip("<>") for token in TOOL_CALL_TOKENS)
TOOL_TAG_NAME_PATTERN = "|".join(re.escape(name) for name in TOOL_TAG_NAMES)
TOOL_OPEN_RE = re.compile(rf"<({TOOL_TAG_NAME_PATTERN})\s*>", flags=re.IGNORECASE)
TOOL_CLOSE_RE = re.compile(rf"</({TOOL_TAG_NAME_PATTERN})\s*>", flags=re.IGNORECASE)
REASONING_OPEN_RE = re.compile(r"<reasoning\s*>", flags=re.IGNORECASE)
REASONING_CLOSE_RE = re.compile(r"</reasoning\s*>", flags=re.IGNORECASE)
ANSWER_OPEN_TAG_RE = re.compile(r"<answer\s*>", flags=re.IGNORECASE)
ANSWER_CLOSE_TAG_RE = re.compile(r"</answer\s*>", flags=re.IGNORECASE)
HARMLESS_NON_ASCII_CHARS = frozenset("“”‘’–—…°，。！？；：、（）《》【】")
ROLE_MARKER_CONTAMINATION_RE = re.compile(
    r"(?:^|\n|>)\s*(?:assistant|user|tool|system)\s*:",
    flags=re.IGNORECASE,
)
IM_START_ROLE_CONTAMINATION_RE = re.compile(
    r"<\|im_start\|>\s*(?:assistant|user|tool|system)",
    flags=re.IGNORECASE,
)
TOOL_RESPONSE_CONTAMINATION_RE = re.compile(
    r"</?tool_response\b|tool_response\s*:",
    flags=re.IGNORECASE,
)
SCENECOT_INSTRUCTION_CONTAMINATION_RE = re.compile(
    r"</?scenecot_instruction\b|scenecot_instruction",
    flags=re.IGNORECASE,
)
OBSERVATION_OR_FINAL_MARKERS = (
    "<tool_response",
    "</tool_response>",
    "<|im_start|>tool",
    "<|im_start|>user",
    "\ntool\n",
    "\nuser\n",
    "\nuser:",
    "\nassistant\n",
    "\nassistant:",
    "<|im_start|>assistant\n",
    "<reasoning>",
    "<answer>",
)
TOOL_TAG_PLACEHOLDER = "<tool_tag>"
THINK_GRD_PLACEHOLDERS = {
    "observation query",
    "query for the evidence",
    "evidence target",
}
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
NUMBER_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
DECIMAL_POINT_PLACEHOLDER = "DECIMALPOINTTOKEN"
ANSWER_PLACEHOLDERS = {"...", "…"}
REPEATED_DOT_RE = re.compile(r"\.{4,}")
FINAL_LENGTH_PENALTY_ENV = "SCENECOT_FINAL_LENGTH_PENALTY"
FINAL_REASONING_SOFT_TOKEN_LIMIT_ENV = "SCENECOT_FINAL_REASONING_SOFT_TOKEN_LIMIT"
FINAL_REASONING_HARD_TOKEN_LIMIT_ENV = "SCENECOT_FINAL_REASONING_HARD_TOKEN_LIMIT"
FINAL_REASONING_MAX_PENALTY_ENV = "SCENECOT_FINAL_REASONING_MAX_PENALTY"
FINAL_ANSWER_SOFT_TOKEN_LIMIT_ENV = "SCENECOT_FINAL_ANSWER_SOFT_TOKEN_LIMIT"
FINAL_ANSWER_HARD_TOKEN_LIMIT_ENV = "SCENECOT_FINAL_ANSWER_HARD_TOKEN_LIMIT"
FINAL_ANSWER_MAX_PENALTY_ENV = "SCENECOT_FINAL_ANSWER_MAX_PENALTY"
TOO_LONG_TOKENIZER_PATH_ENV = "SCENECOT_TOO_LONG_TOKENIZER_PATH"
TOO_LONG_TOKENIZER_LOCAL_FILES_ONLY_ENV = "SCENECOT_TOO_LONG_TOKENIZER_LOCAL_FILES_ONLY"
MAX_REPEAT_PUNCT_RUN_ENV = "SCENECOT_MAX_REPEAT_PUNCT_RUN"
REPEATED_PUNCT_MAX_PENALTY_ENV = "SCENECOT_REPEATED_PUNCT_MAX_PENALTY"
REPEATED_SENTENCE_THRESHOLD_ENV = "SCENECOT_REPEATED_SENTENCE_THRESHOLD"
SEVERE_REPEATED_PUNCT_RUN_ENV = "SCENECOT_SEVERE_REPEATED_PUNCT_RUN"
REWARD_MAX_ENV = "SCENECOT_REWARD_MAX"
CATASTROPHIC_REWARD_ENV = "SCENECOT_CATASTROPHIC_REWARD"
AGENT_REWARD_ENABLED_ENV = "SCENECOT_ENABLE_AGENT_REWARD"
AGENT_REWARD_PROGRESS_SWITCH_STEP_ENV = "SCENECOT_PROGRESS_SWITCH_STEP"
AGENT_REWARD_PROGRESS_SWITCH_ENABLED_ENV = "SCENECOT_ENABLE_PROGRESS_SWITCH"
FORCE_AGENT_REWARD_PROGRESS_ENV = "SCENECOT_FORCE_AGENT_REWARD_PROGRESS"
GLOBAL_STEP_ENV = "SCENECOT_GLOBAL_STEP"
AGENT_FORMAT_W_START_ENV = "SCENECOT_FORMAT_W_START"
AGENT_FORMAT_W_END_ENV = "SCENECOT_FORMAT_W_END"
AGENT_ANSWER_W_START_ENV = "SCENECOT_ANSWER_W_START"
AGENT_ANSWER_W_END_ENV = "SCENECOT_ANSWER_W_END"
COAC_ENABLE_ENV = "SCENECOT_COAC_ENABLE"
COAC_TAU_ENV = "SCENECOT_COAC_TAU"
COAC_COPY_THRESH_ENV = "SCENECOT_COAC_COPY_THRESH"
COAC_COPY_DISCOUNT_ENV = "SCENECOT_COAC_COPY_DISCOUNT"
COAC_COPY_DISCOUNT_ENABLE_ENV = "SCENECOT_COAC_COPY_DISCOUNT_ENABLE"
COAC_GROUP_ZSCORE_ENV = "SCENECOT_COAC_GROUP_ZSCORE"
COAC_FALLBACK_CREDIT_ENV = "SCENECOT_COAC_FALLBACK_CREDIT"
TOOL_WEIGHT_OVERRIDE_ENV = "SCENECOT_TOOL_WEIGHT_OVERRIDE"
AUTOCLOSE_FORMAT_REWARD_ENV = "SCENECOT_AUTOCLOSE_FORMAT_REWARD"
SCORE_AUTOCLOSED_ANSWER_ENV = "SCENECOT_SCORE_AUTOCLOSED_ANSWER"
AUTOCLOSE_FORMAT_REWARD = 0.3
PROGRESS_FORMAT_REWARD_ENABLE_ENV = "SCENECOT_PROGRESS_FORMAT_REWARD_ENABLE"
PROGRESS_FORMAT_FIRST_TURN_W_ENV = "SCENECOT_PROGRESS_FORMAT_FIRST_TURN_W"
PROGRESS_FORMAT_REASONING_W_ENV = "SCENECOT_PROGRESS_FORMAT_REASONING_W"
PROGRESS_FORMAT_ANSWER_BODY_W_ENV = "SCENECOT_PROGRESS_FORMAT_ANSWER_BODY_W"
PROGRESS_FORMAT_ANSWER_CLOSE_W_ENV = "SCENECOT_PROGRESS_FORMAT_ANSWER_CLOSE_W"

ANSWER_JUDGE_BACKEND_ENV = "SCENECOT_ANSWER_JUDGE_BACKEND"
ANSWER_JUDGE_MODEL_ENV = "SCENECOT_ANSWER_JUDGE_MODEL"
ANSWER_JUDGE_TIMEOUT_ENV = "SCENECOT_ANSWER_JUDGE_TIMEOUT"
ANSWER_JUDGE_MIN_F1_ENV = "SCENECOT_ANSWER_JUDGE_MIN_F1"
ANSWER_JUDGE_MAX_F1_ENV = "SCENECOT_ANSWER_JUDGE_MAX_F1"
ANSWER_JUDGE_PROMPT_STYLE_ENV = "SCENECOT_ANSWER_JUDGE_PROMPT_STYLE"
ANSWER_JUDGE_RUBRIC_PATH_ENV = "SCENECOT_ANSWER_JUDGE_RUBRIC_PATH"
ANSWER_JUDGE_CACHE_SIZE = 50000
ANSWER_JUDGE_STYLE_EQUIVALENCE = "equivalence"
ANSWER_JUDGE_STYLE_MSQA_1TO5 = "msqa_1to5"
ANSWER_JUDGE_SUPPORTED_STYLES = {ANSWER_JUDGE_STYLE_EQUIVALENCE, ANSWER_JUDGE_STYLE_MSQA_1TO5}
DEFAULT_ANSWER_JUDGE_PROMPT_STYLE = ANSWER_JUDGE_STYLE_MSQA_1TO5
DEFAULT_ANSWER_JUDGE_RUBRIC_PATH = "assets/eval/MSQA_eval_prompt.json"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


MAX_REPEAT_PUNCT_RUN = int(_env_float(MAX_REPEAT_PUNCT_RUN_ENV, 4.0))
REPEATED_PUNCT_MAX_PENALTY = _env_float(REPEATED_PUNCT_MAX_PENALTY_ENV, 1.0)
REPEATED_SENTENCE_THRESHOLD = int(_env_float(REPEATED_SENTENCE_THRESHOLD_ENV, 3.0))
DEFAULT_REWARD_MAX = 2.2
REWARD_MAX = DEFAULT_REWARD_MAX


def _reward_max() -> float:
    return _env_float(REWARD_MAX_ENV, DEFAULT_REWARD_MAX)


@lru_cache(maxsize=1)
def _load_too_long_tokenizer() -> Any:
    tokenizer_path = (
        os.getenv(TOO_LONG_TOKENIZER_PATH_ENV, "").strip()
        or os.getenv("QWEN_MODEL_PATH", "").strip()
    )
    if not tokenizer_path:
        raise RuntimeError(f"{TOO_LONG_TOKENIZER_PATH_ENV} or QWEN_MODEL_PATH must be set")

    raw_local_only = os.getenv(TOO_LONG_TOKENIZER_LOCAL_FILES_ONLY_ENV, "1").strip().lower()
    local_files_only = raw_local_only not in {"0", "false", "no", "off"}

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def _fallback_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", str(text or ""), flags=re.UNICODE))


def _count_too_long_tokens(text: str) -> tuple[int, str]:
    try:
        tokenizer = _load_too_long_tokenizer()
        return len(tokenizer.encode(str(text or ""), add_special_tokens=False)), "hf_tokenizer"
    except Exception as exc:
        return _fallback_token_count(text), f"regex_fallback:{type(exc).__name__}"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _to_list(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, (str, bytes, dict)) and hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, (list, tuple)):
            return list(converted)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return parsed
    return [value]


def _as_bool(value: Any, *, default: bool) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _catastrophic_reward() -> float:
    """Return the configurable hard-invalid reward, preserving -1 by default."""
    return _clip(_env_float(CATASTROPHIC_REWARD_ENV, -1.0), -1.0, 0.0)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _answer_judge_backend() -> str:
    return os.getenv(ANSWER_JUDGE_BACKEND_ENV, "rule").strip().lower()


def _answer_judge_prompt_style() -> str:
    raw = os.getenv(ANSWER_JUDGE_PROMPT_STYLE_ENV, DEFAULT_ANSWER_JUDGE_PROMPT_STYLE)
    style = raw.strip().lower().replace("-", "_")
    if style in {"msqa", "msqa_eval", "msqa_1_to_5", "score_1_to_5", "1to5", "1_to_5"}:
        return ANSWER_JUDGE_STYLE_MSQA_1TO5
    if style in {"semantic_equivalence", "answer_equivalence", "0_0_5_1", "0_5_1"}:
        return ANSWER_JUDGE_STYLE_EQUIVALENCE
    return style


def _answer_judge_rubric_path() -> str:
    return os.getenv(ANSWER_JUDGE_RUBRIC_PATH_ENV, DEFAULT_ANSWER_JUDGE_RUBRIC_PATH).strip()


def _extract_question_from_extra_info(extra_info: dict | None) -> str:
    if not isinstance(extra_info, dict):
        return ""

    for key in ("question", "raw_question", "problem", "query", "prompt", "instruction"):
        value = extra_info.get(key)
        if not _is_missing(value):
            return str(value).strip()

    raw_prompt = extra_info.get("raw_prompt") or extra_info.get("messages")
    if isinstance(raw_prompt, str):
        return raw_prompt.strip()
    if isinstance(raw_prompt, (list, tuple)):
        parts: list[str] = []
        for item in raw_prompt:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, (list, tuple)):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            parts.append(part["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part.strip())

    return ""


def strip_answer_tags(text: str) -> str:
    text = str(text or "").strip()
    matches = [match.group(1).strip() for match in ANSWER_RE.finditer(text)]
    matches = [match for match in matches if match]
    for match in reversed(matches):
        if not _is_placeholder_answer(match):
            return match
    return matches[-1] if matches else text


def _is_placeholder_answer(text: Any) -> bool:
    return str(text or "").strip() in ANSWER_PLACEHOLDERS


def _select_scoring_text(solution_str: str, extra_info: dict | None) -> tuple[str, str]:
    if isinstance(extra_info, dict):
        actor_response = extra_info.get("response_text")
        if not _is_missing(actor_response):
            return str(actor_response), "extra_info.response_text"
    return str(solution_str or ""), "solution_str"


def _plain_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_metric_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_metric_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_metric_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain_metric_value(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _merge_scenecot_reward_extra_info(
    extra_info: Any,
    non_tensor_batch: Any,
) -> dict[str, Any]:
    """Normalize rollout metadata for both initial and trainer-side reward calls."""
    merged = dict(extra_info) if isinstance(extra_info, dict) else {}
    batch_fields = non_tensor_batch if isinstance(non_tensor_batch, dict) else {}
    tool_extra_fields = _plain_metric_value(batch_fields.get("tool_extra_fields"))
    if not isinstance(tool_extra_fields, dict):
        tool_extra_fields = {}

    # AgentLoop._postprocess flattens extra_fields before COAC removes the
    # rollout-time rm_scores and asks the trainer to recompute reward.
    response_text = batch_fields.get("response_text")
    if response_text is None:
        response_text = tool_extra_fields.get("response_text")
    response_text = _plain_metric_value(response_text)
    if isinstance(response_text, str) and response_text.strip():
        merged["response_text"] = response_text

    scenecot_metrics = _plain_metric_value(batch_fields.get("scenecot_metrics"))
    if not isinstance(scenecot_metrics, dict) or not scenecot_metrics:
        scenecot_metrics = _plain_metric_value(tool_extra_fields.get("scenecot_metrics"))
    if isinstance(scenecot_metrics, dict) and scenecot_metrics:
        merged["scenecot_tool_metrics"] = scenecot_metrics

    rollout_step = batch_fields.get("rollout_step")
    if rollout_step is None:
        rollout_step = tool_extra_fields.get("rollout_step")
    rollout_step = _plain_metric_value(rollout_step)
    if not _is_missing(rollout_step):
        merged["rollout_step"] = rollout_step
        if _is_missing(merged.get("global_step")):
            merged["global_step"] = rollout_step

    for key in (
        "coac_delta",
        "coac_delta_group_norm",
        "coac_group_size",
        "coac_group_std",
        "coac_r_tool",
        "coac_r_tool_override",
        "coac_valid",
        "coac_obs_text",
        "coac_copy_overlap",
        "coac_copy_discount_applied",
        "coac_skip_reason",
        "coac_error_reason",
    ):
        if key in batch_fields:
            merged[key] = _plain_metric_value(batch_fields[key])
    return merged


def _to_optional_bool(value: Any) -> bool | None:
    value = _plain_metric_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _metric_lookup(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return _plain_metric_value(metrics[key])
    return None


def _has_gt_label_from_extra_info(extra_info: dict | None) -> bool | None:
    if not isinstance(extra_info, dict) or "gt_objects" not in extra_info:
        return None
    gt_objects = _plain_metric_value(extra_info.get("gt_objects"))
    if gt_objects is None:
        return False
    if isinstance(gt_objects, (list, tuple)):
        return bool(gt_objects)
    return bool(str(gt_objects).strip())


def _grounding_passthrough_metrics(extra_info: dict | None) -> dict[str, Any]:
    raw_metrics = {}
    if isinstance(extra_info, dict):
        maybe_metrics = extra_info.get("scenecot_tool_metrics")
        if isinstance(maybe_metrics, dict):
            raw_metrics = maybe_metrics

    has_gt_label = _to_optional_bool(
        _metric_lookup(raw_metrics, "has_gt_label", "scenecot/has_gt_label")
    )
    if has_gt_label is None:
        has_gt_label = _has_gt_label_from_extra_info(extra_info)

    return {
        "has_gt_label": has_gt_label,
        "grd_stats_valid": _to_optional_bool(
            _metric_lookup(raw_metrics, "grd_stats_valid", "scenecot/grd_stats_valid")
        ),
        "grd_hit1": _metric_lookup(
            raw_metrics,
            "grd_hit1",
            "scenecot/grd_hit1",
            "ground" + "_hit1",
            "scenecot/ground" + "_hit1",
        ),
        "grd_hit5": _metric_lookup(
            raw_metrics,
            "grd_hit5",
            "scenecot/grd_hit5",
            "ground" + "_hit5",
            "scenecot/ground" + "_hit5",
        ),
        "grd_hit10": _metric_lookup(
            raw_metrics,
            "grd_hit10",
            "scenecot/grd_hit10",
            "ground" + "_hit10",
            "scenecot/ground" + "_hit10",
        ),
        "grd_hit15": _metric_lookup(raw_metrics, "grd_hit15", "scenecot/grd_hit15"),
        "grd_rank": _metric_lookup(raw_metrics, "grd_rank", "scenecot/grd_rank"),
        "grd_prob": _metric_lookup(raw_metrics, "grd_prob", "scenecot/grd_prob"),
        "grd_topk_ids": _metric_lookup(raw_metrics, "grd_topk_ids", "scenecot/grd_topk_ids"),
        "grd_topk_probs": _metric_lookup(raw_metrics, "grd_topk_probs", "scenecot/grd_topk_probs"),
    }


def _coerce_metric_int(value: Any, default: int = 0) -> int:
    value = _plain_metric_value(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _tool_execution_metrics(extra_info: dict | None) -> dict[str, Any]:
    raw_metrics = {}
    if isinstance(extra_info, dict):
        maybe_metrics = extra_info.get("scenecot_tool_metrics")
        if isinstance(maybe_metrics, dict):
            raw_metrics = maybe_metrics
    has_rollout_tool_metrics = any(
        key in raw_metrics
        for key in (
            "tool_call_count",
            "scenecot/tool_call_count",
            "scenecot/tool_call",
            "tool_error_count",
            "scenecot/tool_error_count",
            "scenecot/tool_error",
            "cache_miss_count",
            "scenecot/cache_miss_count",
            "scenecot/cache_miss",
        )
    )
    tool_call_count = _coerce_metric_int(
        _metric_lookup(raw_metrics, "tool_call_count", "scenecot/tool_call_count", "scenecot/tool_call"),
        default=0,
    )
    tool_error_count = _coerce_metric_int(
        _metric_lookup(raw_metrics, "tool_error_count", "scenecot/tool_error_count", "scenecot/tool_error"),
        default=0,
    )
    cache_miss_count = _coerce_metric_int(
        _metric_lookup(raw_metrics, "cache_miss_count", "scenecot/cache_miss_count", "scenecot/cache_miss"),
        default=0,
    )
    final_answer_autoclosed_count = _coerce_metric_int(
        _metric_lookup(
            raw_metrics,
            "final_answer_autoclosed_count",
            "scenecot/final_answer_autoclosed_count",
        ),
        default=0,
    )
    final_answer_autoclose_eligible_count = _coerce_metric_int(
        _metric_lookup(
            raw_metrics,
            "final_answer_autoclose_eligible_count",
            "scenecot/final_answer_autoclose_eligible_count",
        ),
        default=0,
    )
    return {
        "rollout_has_tool_metrics": int(has_rollout_tool_metrics),
        "rollout_tool_call_count": tool_call_count,
        "rollout_tool_error_count": tool_error_count,
        "rollout_cache_miss_count": cache_miss_count,
        "final_answer_autoclosed_count": final_answer_autoclosed_count,
        "final_answer_autoclose_eligible_count": final_answer_autoclose_eligible_count,
    }


def _gcs_passthrough_metrics(extra_info: dict | None) -> dict[str, Any]:
    raw_metrics = {}
    if isinstance(extra_info, dict):
        maybe_metrics = extra_info.get("scenecot_tool_metrics")
        if isinstance(maybe_metrics, dict):
            raw_metrics = maybe_metrics
    forced_mode = _metric_lookup(raw_metrics, "gcs_forced_mode", "scenecot/gcs_forced_mode")
    forced_mode = "" if forced_mode is None else str(forced_mode)
    forced_count = _coerce_metric_int(
        _metric_lookup(raw_metrics, "gcs_forced_count", "scenecot/gcs_forced_count"),
        default=1 if forced_mode else 0,
    )
    coverage_fallback_count = _coerce_metric_int(
        _metric_lookup(raw_metrics, "gcs_coverage_fallback_count", "scenecot/gcs_coverage_fallback_count"),
        default=0,
    )
    return {
        "gcs_forced_mode": forced_mode,
        "gcs_forced_count": int(bool(forced_count or forced_mode)),
        "gcs_coverage_fallback_count": coverage_fallback_count,
    }


def _optional_float(value: Any, default: float = 0.0) -> float:
    value = _plain_metric_value(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


COAC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "in",
        "is",
        "it",
        "near",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def _content_words(text: str) -> list[str]:
    words = [word.lower() for word in WORD_RE.findall(str(text or ""))]
    return [word for word in words if word not in COAC_STOPWORDS]


def copy_overlap_ratio(gt_answer: Any, observation_text: Any) -> float:
    answer_words = _content_words(str(gt_answer or ""))
    if not answer_words:
        return 0.0
    observation_words = set(_content_words(str(observation_text or "")))
    if not observation_words:
        return 0.0
    return sum(1 for word in answer_words if word in observation_words) / len(answer_words)


def _coac_tool_weight() -> float:
    raw = os.getenv(TOOL_WEIGHT_OVERRIDE_ENV, "").strip()
    if raw:
        return _env_float(TOOL_WEIGHT_OVERRIDE_ENV, 0.2)
    return 0.2


def _coac_fallback_credit() -> float:
    return _clip(_env_float(COAC_FALLBACK_CREDIT_ENV, 0.5), 0.0, 1.0)


def compute_fixed_tool_reward(
    *,
    tool_called: bool,
    full_protocol_valid: bool,
    catastrophic_invalid: bool,
) -> float:
    if not tool_called or not full_protocol_valid or catastrophic_invalid:
        return 0.0
    return _coac_fallback_credit()


def compute_coac_tool_reward(
    *,
    extra_info: dict | None,
    gt_answers: list[str],
    tool_called: bool,
    full_protocol_valid: bool,
    catastrophic_invalid: bool,
) -> tuple[float, dict[str, Any]]:
    enabled = _env_bool(COAC_ENABLE_ENV, False)
    tau = max(_env_float(COAC_TAU_ENV, 0.5), 1e-8)
    copy_thresh = _env_float(COAC_COPY_THRESH_ENV, 0.6)
    copy_discount = _env_float(COAC_COPY_DISCOUNT_ENV, 0.5)
    copy_discount_enabled = _env_bool(COAC_COPY_DISCOUNT_ENABLE_ENV, False)
    group_zscore_enabled = _env_bool(COAC_GROUP_ZSCORE_ENV, False)
    delta = _optional_float((extra_info or {}).get("coac_delta") if isinstance(extra_info, dict) else None, 0.0)
    delta_group_norm = _optional_float(
        (extra_info or {}).get("coac_delta_group_norm") if isinstance(extra_info, dict) else None,
        0.0,
    )
    group_size = int(
        _optional_float((extra_info or {}).get("coac_group_size") if isinstance(extra_info, dict) else None, 0.0)
    )
    group_std = _optional_float((extra_info or {}).get("coac_group_std") if isinstance(extra_info, dict) else None, 0.0)
    group_r_tool_override = _optional_float(
        (extra_info or {}).get("coac_r_tool_override") if isinstance(extra_info, dict) else None,
        None,
    )
    coac_valid = _to_optional_bool((extra_info or {}).get("coac_valid") if isinstance(extra_info, dict) else None)
    obs_text = (extra_info or {}).get("coac_obs_text", "") if isinstance(extra_info, dict) else ""
    primary_gt = gt_answers[0] if gt_answers else ""
    overlap = copy_overlap_ratio(primary_gt, obs_text)
    copy_applied = bool(copy_discount_enabled and overlap >= copy_thresh)
    fixed_tool_reward = compute_fixed_tool_reward(
        tool_called=tool_called,
        full_protocol_valid=full_protocol_valid,
        catastrophic_invalid=catastrophic_invalid,
    )

    if not enabled:
        r_tool = 0.0
        credit_source = "disabled"
    elif fixed_tool_reward <= 0.0:
        r_tool = 0.0
        credit_source = "ineligible"
    elif coac_valid is not True:
        r_tool = fixed_tool_reward
        credit_source = "fixed_fallback"
    else:
        if group_zscore_enabled and group_r_tool_override is not None:
            r_tool = _clip(group_r_tool_override, 0.0, 1.0)
        else:
            r_tool = _clip(delta / tau, 0.0, 1.0)
        if copy_applied:
            r_tool *= copy_discount
        credit_source = "coac_delta"

    return r_tool, {
        "coac_enabled": int(enabled),
        "coac_delta": delta,
        "coac_tau": tau,
        "coac_r_tool": r_tool,
        "coac_delta_group_norm": delta_group_norm,
        "coac_group_size": group_size,
        "coac_group_std": group_std,
        "coac_r_tool_override": group_r_tool_override if group_r_tool_override is not None else 0.0,
        "coac_fixed_r_tool": fixed_tool_reward,
        "coac_fallback_credit": _coac_fallback_credit(),
        "coac_tool_credit_source": credit_source,
        "coac_fixed_fallback_applied": int(enabled and coac_valid is not True and fixed_tool_reward > 0.0),
        "coac/fallback_credit_rate": float(enabled and coac_valid is not True and fixed_tool_reward > 0.0),
        "coac_valid": int(bool(coac_valid)) if coac_valid is not None else 0,
        "coac_copy_overlap": overlap,
        "coac_copy_thresh": copy_thresh,
        "coac_copy_discount": copy_discount,
        "coac_copy_discount_enabled": int(copy_discount_enabled),
        "coac_copy_discount_applied": int(copy_applied and enabled and coac_valid is True and r_tool > 0.0),
        "coac_group_zscore_enabled": int(group_zscore_enabled),
        "reward/tool_weight": _coac_tool_weight() if enabled else 0.0,
    }


def normalize_for_em(text: str) -> str:
    return strip_answer_tags(text).strip()


def normalize_number_commas(text: str) -> str:
    return NUMBER_COMMA_RE.sub("", str(text or ""))


_NUMBER_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
_NUMBER_WORDS = set(_NUMBER_UNITS) | set(_NUMBER_TENS) | set(_NUMBER_SCALES)


def _parse_number_words(words: list[str]) -> int | None:
    if not words or not any(word in _NUMBER_WORDS for word in words):
        return None

    total = 0
    current = 0
    for word in words:
        if word == "and":
            continue
        if word in _NUMBER_UNITS:
            current += _NUMBER_UNITS[word]
        elif word in _NUMBER_TENS:
            current += _NUMBER_TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word in _NUMBER_SCALES:
            total += max(current, 1) * _NUMBER_SCALES[word]
            current = 0
        else:
            return None
    return total + current


def _normalize_number_words(text: str) -> str:
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    tokens = re.findall(r"[a-z]+|[^a-z]+", text)
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.isalpha() or token not in _NUMBER_WORDS:
            output.append(token)
            index += 1
            continue

        words = [token]
        cursor = index + 1
        while cursor < len(tokens):
            separator = tokens[cursor]
            next_token = tokens[cursor + 1] if cursor + 1 < len(tokens) else ""
            if (
                separator.isspace()
                and next_token.isalpha()
                and (next_token in _NUMBER_WORDS or next_token == "and")
            ):
                words.append(next_token)
                cursor += 2
                continue
            break

        number = _parse_number_words(words)
        if number is None:
            output.append(token)
            index += 1
        else:
            output.append(str(number))
            index = cursor
    return "".join(output)


def normalize_for_re_em(text: str) -> str:
    text = normalize_for_em(text).lower()
    text = normalize_number_commas(text)
    text = _normalize_number_words(text)
    text = re.sub(r"(?<=\d)\.(?=\d)", DECIMAL_POINT_PLACEHOLDER, text)
    punctuation = string.punctuation.replace(".", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = text.replace(".", " ")
    text = text.replace(DECIMAL_POINT_PLACEHOLDER, ".")
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens).strip()


def tokenize_for_f1(text: str) -> list[str]:
    return WORD_RE.findall(normalize_for_re_em(text))


def em_match(pred_answer: str, gt_answer: str) -> bool:
    return normalize_for_em(pred_answer) == normalize_for_em(gt_answer)


def re_em_match(pred_answer: str, gt_answer: str) -> bool:
    pred_norm = normalize_for_re_em(pred_answer)
    gt_norm = normalize_for_re_em(gt_answer)
    return bool(pred_norm) and pred_norm == gt_norm


def has_complete_answer(text: str) -> bool:
    final_turn = extract_final_assistant_turn(str(text or ""))
    matches = [match.group(1).strip() for match in COMPLETE_ANSWER_RE.finditer(final_turn)]
    return any(match and not _is_placeholder_answer(match) for match in matches)


def has_complete_reasoning(text: str) -> bool:
    return REASONING_RE.search(extract_final_assistant_turn(str(text or ""))) is not None


def count_tool_call_tokens(text: str) -> int:
    response = str(text or "")
    return sum(response.count(token) for token in TOOL_CALL_TOKENS)


def first_tool_call_token(text: str) -> str | None:
    response = str(text or "")
    first_pos: int | None = None
    first_token: str | None = None
    for token in TOOL_CALL_TOKENS:
        pos = response.find(token)
        if pos < 0:
            continue
        if first_pos is None or pos < first_pos:
            first_pos = pos
            first_token = token
    return first_token


def _normalize_placeholder_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def has_tool_tag_placeholder(text: str) -> bool:
    return TOOL_TAG_PLACEHOLDER in str(text or "")


def has_bad_think_grd_placeholder(text: str) -> bool:
    response = str(text or "")
    for match in THINK_GRD_RE.finditer(response):
        query = _normalize_placeholder_text(match.group(1))
        if query in THINK_GRD_PLACEHOLDERS:
            return True
    return False


def compute_placeholder_format_metrics(text: str) -> dict:
    bad_tool_tag = has_tool_tag_placeholder(text)
    bad_think_grd = has_bad_think_grd_placeholder(text)
    return {
        "format_placeholder_violation": int(bad_tool_tag or bad_think_grd),
        "format_bad_tool_tag_placeholder": int(bad_tool_tag),
        "format_bad_think_grd_placeholder": int(bad_think_grd),
    }


def has_exactly_one_tool_call(text: str) -> bool:
    placeholder_metrics = compute_placeholder_format_metrics(text)
    return count_tool_call_tokens(text) == 1 and not bool(placeholder_metrics["format_placeholder_violation"])



def _first_marker_position(text: str, *, start: int = 0) -> int | None:
    positions = [pos for marker in OBSERVATION_OR_FINAL_MARKERS if (pos := text.find(marker, start)) >= 0]
    return min(positions) if positions else None


def _tool_request_end(text: str, tool_match: re.Match[str]) -> int:
    tool_name = re.escape(tool_match.group(1))
    close_re = re.compile(rf"</{tool_name}\s*>", flags=re.IGNORECASE)
    marker_pos = _first_marker_position(text, start=tool_match.end())
    close_match = close_re.search(text, tool_match.end())
    if close_match is not None and (marker_pos is None or close_match.start() < marker_pos):
        return close_match.end()
    return tool_match.end()


def _extract_first_turn_protocol_span(text: str) -> str:
    text = str(text or "")
    tool_match = TOOL_OPEN_RE.search(text)
    if tool_match is not None:
        return text[: _tool_request_end(text, tool_match)]
    marker_pos = _first_marker_position(text)
    return text[:marker_pos] if marker_pos is not None else text


def _has_final_signal(text: str) -> bool:
    text = str(text or "")
    if REASONING_OPEN_RE.search(text) is not None or ANSWER_OPEN_TAG_RE.search(text) is not None:
        return True
    return any(marker in text for marker in ("\nassistant\n", "\nassistant:", "<|im_start|>assistant\n"))


def _answer_body_is_valid(answer_text: str | None) -> bool:
    if answer_text is None:
        return False
    stripped = str(answer_text).strip()
    return bool(stripped) and not _is_placeholder_answer(stripped) and bool(normalize_for_re_em(stripped))


def _normalized_sentences_for_repetition(text: str) -> list[str]:
    normalized_text = re.sub(r"<[^>]+>", " ", str(text or ""))
    parts = re.split(r"(?<=[.!?。！？])\s*", normalized_text)
    sentences: list[str] = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip().lower()
        if len(sentence) >= 10:
            sentences.append(sentence)
    return sentences


def _max_repeated_sentence_run(text: str) -> int:
    sentences = _normalized_sentences_for_repetition(text)
    if not sentences:
        return 0
    best_run = 1
    current_run = 1
    previous = sentences[0]
    for sentence in sentences[1:]:
        if sentence == previous:
            current_run += 1
        else:
            best_run = max(best_run, current_run)
            current_run = 1
            previous = sentence
    return max(best_run, current_run)


def _token_count_for_repetition(text: str) -> int:
    return len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", str(text or "")).lower(), flags=re.UNICODE))


def _repetition_metrics_for_text(text: str) -> dict[str, Any]:
    threshold = _repeated_sentence_threshold()
    _, same_sentence_count = max_repeated_sentence_count(text)
    sentence_run = _max_repeated_sentence_run(text)
    repeated_ngram, ngram_count = max_repeated_ngram_count(text)
    ngram_size = len(repeated_ngram.split()) if repeated_ngram else 0
    token_count = _token_count_for_repetition(text)
    ngram_ratio = 0.0 if not token_count else min(1.0, (ngram_count * ngram_size) / token_count)
    sentence_loop = max(same_sentence_count, sentence_run) >= threshold
    ngram_loop = ngram_count >= threshold
    return {
        "repeated_sentence_loop": int(sentence_loop),
        "repeated_ngram_loop": int(ngram_loop),
        "max_repeated_sentence_run": int(sentence_run),
        "max_same_sentence_count": int(same_sentence_count),
        "repeated_ngram_count": int(ngram_count),
        "repeated_ngram_ratio": float(ngram_ratio),
    }


def _combine_repetition_metrics(metrics_by_scope: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics_by_scope:
        return {
            "repeated_sentence_loop": 0,
            "repeated_ngram_loop": 0,
            "max_repeated_sentence_run": 0,
            "max_same_sentence_count": 0,
            "repeated_ngram_count": 0,
            "repeated_ngram_ratio": 0.0,
        }
    return {
        "repeated_sentence_loop": int(any(item.get("repeated_sentence_loop") for item in metrics_by_scope)),
        "repeated_ngram_loop": int(any(item.get("repeated_ngram_loop") for item in metrics_by_scope)),
        "max_repeated_sentence_run": max(int(item.get("max_repeated_sentence_run", 0) or 0) for item in metrics_by_scope),
        "max_same_sentence_count": max(int(item.get("max_same_sentence_count", 0) or 0) for item in metrics_by_scope),
        "repeated_ngram_count": max(int(item.get("repeated_ngram_count", 0) or 0) for item in metrics_by_scope),
        "repeated_ngram_ratio": max(float(item.get("repeated_ngram_ratio", 0.0) or 0.0) for item in metrics_by_scope),
    }


def _extract_reasoning_spans_for_repetition(text: str) -> list[str]:
    response = str(text or "")
    spans: list[str] = []
    for open_match in REASONING_OPEN_RE.finditer(response):
        close_match = REASONING_CLOSE_RE.search(response, open_match.end())
        end = close_match.start() if close_match is not None else len(response)
        span = response[open_match.end() : end].strip()
        if span:
            spans.append(span)
    return spans


def _max_consecutive_matching_chars(text: str, predicate) -> int:
    best = 0
    current = 0
    for char in str(text or ""):
        if predicate(char):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _is_suspicious_non_ascii(char: str) -> bool:
    if ord(char) <= 127 or char in HARMLESS_NON_ASCII_CHARS or char.isspace():
        return False
    category = unicodedata.category(char)
    if category.startswith("P") or category.startswith("S"):
        return False
    return True


def compute_non_ascii_garbage_metrics(text: str) -> dict[str, Any]:
    content = str(text or "")
    nonspace_chars = [char for char in content if not char.isspace()]
    non_ascii_chars = [char for char in nonspace_chars if ord(char) > 127]
    suspicious_chars = [char for char in non_ascii_chars if _is_suspicious_non_ascii(char)]
    non_ascii_ratio = 0.0 if not nonspace_chars else len(non_ascii_chars) / len(nonspace_chars)
    suspicious_ratio = 0.0 if not nonspace_chars else len(suspicious_chars) / len(nonspace_chars)
    max_consecutive_non_ascii = _max_consecutive_matching_chars(content, lambda char: ord(char) > 127)
    max_consecutive_suspicious = _max_consecutive_matching_chars(content, _is_suspicious_non_ascii)
    max_same_suspicious = Counter(suspicious_chars).most_common(1)[0][1] if suspicious_chars else 0
    rare_unicode_garbage = bool(suspicious_chars and (max_consecutive_suspicious >= 4 or max_same_suspicious >= 6))
    non_ascii_garbage = bool(suspicious_ratio > 0.20 or max_consecutive_suspicious >= 8 or rare_unicode_garbage)
    return {
        "non_ascii_garbage": int(non_ascii_garbage),
        "non_ascii_ratio": float(non_ascii_ratio),
        "max_consecutive_non_ascii": int(max_consecutive_non_ascii),
        "rare_unicode_garbage": int(rare_unicode_garbage),
    }


def compute_contamination_metrics(text: str) -> dict[str, Any]:
    content = str(text or "")
    role_marker = ROLE_MARKER_CONTAMINATION_RE.search(content) is not None
    im_start_role = IM_START_ROLE_CONTAMINATION_RE.search(content) is not None
    tool_response = TOOL_RESPONSE_CONTAMINATION_RE.search(content) is not None
    scenecot_instruction = SCENECOT_INSTRUCTION_CONTAMINATION_RE.search(content) is not None
    contamination_reason = None
    if tool_response:
        contamination_reason = "tool_response"
    elif scenecot_instruction:
        contamination_reason = "scenecot_instruction"
    elif role_marker or im_start_role:
        contamination_reason = "role_marker"
    return {
        "final_turn_contamination": int(contamination_reason is not None),
        "contamination_reason": contamination_reason,
        "role_marker_contamination": int(role_marker or im_start_role),
        "tool_response_contamination": int(tool_response),
        "scenecot_instruction_contamination": int(scenecot_instruction),
    }


def compute_first_turn_protocol_metrics(scoring_text: str) -> dict[str, Any]:
    first_turn = _extract_first_turn_protocol_span(scoring_text).strip()
    think_matches = list(THINK_GRD_RE.finditer(first_turn))
    tool_matches = list(TOOL_OPEN_RE.finditer(first_turn))
    tool_close_matches = list(TOOL_CLOSE_RE.finditer(first_turn))
    has_think_grd = len(think_matches) > 0
    has_tool_call = len(tool_matches) > 0
    reason: str | None = None

    if ANSWER_OPEN_TAG_RE.search(first_turn) is not None or ANSWER_CLOSE_TAG_RE.search(first_turn) is not None:
        reason = "first_turn_contains_answer"
    elif len(think_matches) != 1:
        reason = "first_turn_think_grd_count"
    elif not think_matches[0].group(1).strip():
        reason = "first_turn_empty_think_grd"
    elif len(tool_matches) != 1:
        reason = "first_turn_tool_call_count"
    elif len(tool_close_matches) > 1:
        reason = "first_turn_tool_close_count"
    else:
        think_match = think_matches[0]
        tool_match = tool_matches[0]
        tool_end = _tool_request_end(first_turn, tool_match)
        if tool_match.start() < think_match.end():
            reason = "first_turn_tool_before_think_grd_close"
        elif first_turn[: think_match.start()].strip():
            reason = "first_turn_prefix_text"
        elif first_turn[think_match.end() : tool_match.start()].strip():
            reason = "first_turn_text_between_think_grd_and_tool"
        elif first_turn[tool_end:].strip():
            reason = "first_turn_suffix_text"
        elif len(tool_close_matches) == 1:
            close_match = tool_close_matches[0]
            if close_match.group(1).lower() != tool_match.group(1).lower():
                reason = "first_turn_tool_close_mismatch"
            elif close_match.start() < tool_match.end():
                reason = "first_turn_tool_close_before_open"

    return {
        "first_turn_valid": int(reason is None),
        "first_turn_invalid_reason": reason,
        "has_think_grd": int(has_think_grd),
        "has_tool_call": int(has_tool_call),
        "first_turn_tool_call_count": len(tool_matches),
        "first_turn_think_grd_count": len(think_matches),
    }


def compute_final_turn_protocol_metrics(scoring_text: str) -> dict[str, Any]:
    text = str(scoring_text or "")
    final_signal = _has_final_signal(text)
    final_turn = extract_final_assistant_turn(text).strip() if final_signal else ""
    reasoning_matches = list(REASONING_RE.finditer(final_turn))
    answer_matches = list(COMPLETE_ANSWER_RE.finditer(final_turn))
    loose_answer_matches = list(ANSWER_RE.finditer(final_turn))
    reasoning_open_count = len(REASONING_OPEN_RE.findall(final_turn))
    reasoning_close_count = len(REASONING_CLOSE_RE.findall(final_turn))
    answer_open_count = len(ANSWER_OPEN_TAG_RE.findall(final_turn))
    answer_close_count = len(ANSWER_CLOSE_TAG_RE.findall(final_turn))
    final_tool_call_count = count_tool_call_tokens(final_turn) if final_signal else 0
    parsed_answer = answer_matches[0].group(1).strip() if len(answer_matches) == 1 else None
    parsed_answer_from_unclosed = 0
    if (
        parsed_answer is None
        and len(answer_matches) == 0
        and answer_open_count == 1
        and answer_close_count == 0
        and len(loose_answer_matches) == 1
    ):
        parsed_answer = loose_answer_matches[0].group(1).strip()
        parsed_answer_from_unclosed = 1
    answer_valid = _answer_body_is_valid(parsed_answer)
    unclosed_answer_extractable = bool(parsed_answer_from_unclosed and answer_valid)
    reason: str | None = None

    if not final_signal:
        reason = "missing_final_turn"
    elif final_tool_call_count:
        reason = "final_turn_tool_tag"
    elif len(reasoning_matches) != 1 or reasoning_open_count != 1 or reasoning_close_count != 1:
        reason = "final_turn_reasoning_count"
    elif len(answer_matches) != 1 or answer_open_count != 1 or answer_close_count != 1:
        reason = "final_turn_answer_count"
    elif reasoning_matches[0].start() > answer_matches[0].start():
        reason = "final_turn_answer_before_reasoning"
    elif not answer_valid:
        reason = "empty_final_answer"

    non_ascii_metrics = compute_non_ascii_garbage_metrics(final_turn)
    contamination_metrics = compute_contamination_metrics(final_turn)
    return {
        "final_turn_valid": int(reason is None),
        "final_turn_invalid_reason": reason,
        "has_reasoning": int(len(reasoning_matches) > 0),
        "has_answer": int(answer_valid),
        "answer_parsed_from_answer_tag": int(parsed_answer is not None and answer_valid),
        "answer_parsed_from_unclosed_answer_tag": int(unclosed_answer_extractable),
        "final_answer_unclosed_extractable": int(unclosed_answer_extractable),
        "parsed_answer": parsed_answer,
        "final_tool_call_count": final_tool_call_count,
        "final_reasoning_count": len(reasoning_matches),
        "final_reasoning_open_count": reasoning_open_count,
        "final_reasoning_close_count": reasoning_close_count,
        "final_answer_count": len(answer_matches),
        "final_answer_open_count": answer_open_count,
        "final_answer_close_count": answer_close_count,
        "final_non_ascii_detected": int(non_ascii_metrics["non_ascii_ratio"] > 0.0),
        "final_contamination_detected": contamination_metrics["final_turn_contamination"],
        "final_empty_answer_detected": int(answer_open_count > 0 and len(answer_matches) == 1 and not answer_valid),
        **non_ascii_metrics,
        **contamination_metrics,
    }


def _severe_repeated_punct_threshold() -> int:
    default = max(12, int(MAX_REPEAT_PUNCT_RUN) * 3)
    return max(1, int(_env_float(SEVERE_REPEATED_PUNCT_RUN_ENV, float(default))))


def compute_catastrophic_invalid_metrics(protocol_metrics: dict[str, Any], penalty_metrics: dict[str, Any]) -> dict[str, Any]:
    reason: str | None = None
    if int(penalty_metrics.get("penalty_dot_only", 0) or 0):
        reason = "dot_only_output"
    elif int(protocol_metrics.get("repeated_sentence_loop", 0) or 0) or int(
        protocol_metrics.get("repeated_ngram_loop", 0) or 0
    ):
        reason = "repeated_sentence_loop"
    elif int(penalty_metrics.get("penalty_repeated_punct_max_run", 0) or 0) >= _severe_repeated_punct_threshold():
        reason = "severe_repeated_punctuation"
    elif int(protocol_metrics.get("non_ascii_garbage", 0) or 0):
        reason = "non_ascii_garbage"
    elif int(protocol_metrics.get("final_turn_contamination", 0) or 0):
        reason = "final_turn_contamination"
    elif int(protocol_metrics.get("final_tool_call_count", 0) or 0):
        reason = "final_turn_tool_tag"
    elif int(protocol_metrics.get("final_empty_answer_detected", 0) or 0):
        reason = "empty_final_answer"

    return {
        "catastrophic_invalid": int(reason is not None),
        "hard_invalid_reason": reason,
        "catastrophic_invalid_reason": reason,
    }



def extract_answer(text: str) -> str | None:
    final_turn = extract_final_assistant_turn(str(text or ""))
    matches = [match.group(1).strip() for match in ANSWER_RE.finditer(final_turn)]
    matches = [match for match in matches if match]
    for match in reversed(matches):
        if not _is_placeholder_answer(match):
            return match
    return matches[-1] if matches else None


def normalize_answer(x: Any) -> str:
    if _is_missing(x):
        return ""
    text = normalize_for_re_em(str(x))
    yes_values = {"yes", "yeah", "yep", "true"}
    no_values = {"no", "nope", "false"}
    if text in yes_values:
        return "yes"
    if text in no_values:
        return "no"
    return text


def normalize_ground_truths(ground_truth: Any, extra_info: dict | None = None) -> list[str]:
    values = _to_list(ground_truth)
    if not values and extra_info:
        values = _to_list(extra_info.get("gt_answer"))
    normalized: list[str] = []
    for value in values:
        item = normalize_for_em(str(value))
        if item:
            normalized.append(item)
    return normalized


def _tokens(text: str) -> list[str]:
    return tokenize_for_f1(text)


def _token_f1(pred: str, gt: str) -> float:
    pred_tokens = _tokens(pred)
    gt_tokens = _tokens(gt)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _explicit_yes_no(text: str) -> str | None:
    tokens = tokenize_for_f1(text)
    has_yes = "yes" in tokens
    has_no = "no" in tokens
    if has_yes and has_no:
        return "ambiguous"
    if has_yes:
        return "yes"
    if has_no:
        return "no"
    return None


def _canonical_gt_answers_json(gt_answers: list[str]) -> str:
    return json.dumps([str(answer) for answer in gt_answers], ensure_ascii=False, sort_keys=True)


def _resolve_rubric_path(path_text: str) -> Path:
    path = Path(path_text or DEFAULT_ANSWER_JUDGE_RUBRIC_PATH).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


@lru_cache(maxsize=16)
def _load_msqa_eval_rubric(path_text: str) -> str:
    path = _resolve_rubric_path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        parts = [
            str(item.get("content", "")).strip()
            for item in payload
            if isinstance(item, dict) and item.get("content")
        ]
        rubric = "\n\n".join(part for part in parts if part)
    elif isinstance(payload, dict):
        rubric = str(payload.get("content", "")).strip()
    else:
        rubric = str(payload).strip()
    if not rubric:
        raise ValueError(f"empty MSQA eval rubric: {path}")
    rubric = re.sub(r"\n?Output only the score\.\s*$", "", rubric.strip(), flags=re.IGNORECASE)
    return rubric.strip()


def _build_gemini_equivalence_answer_judge_prompt(
    question: str,
    pred_answer: str,
    gt_answers: list[str],
) -> str:
    question_text = question.strip() or "(question unavailable)"
    gt_json = json.dumps(gt_answers, ensure_ascii=False)
    return f"""You are a strict answer-equivalence judge for 3D visual question answering.
Judge only whether the predicted final answer is semantically equivalent to one of the ground-truth answers.
Ignore wording, articles, punctuation, capitalization, and harmless number formatting differences.
Do not judge the reasoning steps, tool use, visual grounding, or response formatting.

Return JSON only with this schema:
{{"score": 0.0|0.5|1.0, "label": "incorrect|partially_correct|equivalent", "reason": "short reason"}}

Scoring rubric:
- 1.0: the predicted answer is equivalent to a ground-truth answer.
- 0.5: the predicted answer is partially correct but incomplete, overspecific, underspecific, or mildly ambiguous.
- 0.0: the predicted answer is wrong, contradicts the ground truth, or is too vague to be accepted.

Question:
{question_text}

Ground-truth answers JSON:
{gt_json}

Predicted answer:
{pred_answer.strip()}
"""


def _build_gemini_msqa_1to5_answer_judge_prompt(
    question: str,
    pred_answer: str,
    gt_answers: list[str],
    rubric_path: str,
) -> str:
    question_text = question.strip() or "(question unavailable)"
    gt_json = json.dumps(gt_answers, ensure_ascii=False)
    rubric = _load_msqa_eval_rubric(rubric_path)
    return f"""{rubric}

Additional instructions for this reward judge:
- Judge only the predicted final answer, not the reasoning steps, visual tool calls, or response formatting.
- Use the ground-truth answers as the source of truth. If multiple ground-truth answers are provided, score against the best matching one.
- Return JSON only with this schema:
{{"score_1_to_5": 1|2|3|4|5, "reason": "short reason"}}
- Do not include markdown, code fences, or extra text outside the JSON object.

Question:
{question_text}

Ground-truth answers JSON:
{gt_json}

Predicted answer:
{pred_answer.strip()}
"""


def _build_gemini_answer_judge_prompt(
    prompt_style: str,
    question: str,
    pred_answer: str,
    gt_answers: list[str],
    rubric_path: str,
) -> str:
    if prompt_style == ANSWER_JUDGE_STYLE_MSQA_1TO5:
        return _build_gemini_msqa_1to5_answer_judge_prompt(
            question,
            pred_answer,
            gt_answers,
            rubric_path,
        )
    if prompt_style == ANSWER_JUDGE_STYLE_EQUIVALENCE:
        return _build_gemini_equivalence_answer_judge_prompt(question, pred_answer, gt_answers)
    raise ValueError(f"unsupported answer judge prompt style: {prompt_style}")


def _parse_gemini_judge_text(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("empty Gemini response text")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if isinstance(parsed, dict):
        return parsed
    return {"score": parsed, "score_1_to_5": parsed}


def _gemini_response_schema(prompt_style: str) -> dict[str, Any]:
    if prompt_style == ANSWER_JUDGE_STYLE_MSQA_1TO5:
        return {
            "type": "object",
            "properties": {
                "score_1_to_5": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["score_1_to_5"],
        }
    if prompt_style == ANSWER_JUDGE_STYLE_EQUIVALENCE:
        return {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "label": {
                    "type": "string",
                    "enum": ["incorrect", "partially_correct", "equivalent"],
                },
                "reason": {"type": "string"},
            },
            "required": ["score", "label"],
        }
    raise ValueError(f"unsupported answer judge prompt style: {prompt_style}")


def _coerce_judge_score(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip()
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("non-finite Gemini score")
    score = _clip(score, 0.0, 1.0)
    allowed = (0.0, 0.5, 1.0)
    return min(allowed, key=lambda candidate: abs(candidate - score))


def _coerce_judge_score_1_to_5(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip()
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("non-finite Gemini 1-to-5 score")
    return int(round(_clip(score, 1.0, 5.0)))


def _judge_reward_from_score_1_to_5(score: int) -> float:
    return (float(score) - 1.0) / 4.0


def _label_from_score_1_to_5(score: int) -> str:
    return {
        1: "incorrect",
        2: "mostly_incorrect",
        3: "partially_correct",
        4: "mostly_correct",
        5: "correct",
    }.get(score, "unknown")


@lru_cache(maxsize=ANSWER_JUDGE_CACHE_SIZE)
def _gemini_answer_judge_cached(
    model: str,
    prompt_style: str,
    rubric_path: str,
    question: str,
    pred_answer: str,
    gt_answers_json: str,
) -> tuple[float, dict]:
    api_key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"{GEMINI_API_KEY_ENV} is not set")

    gt_answers = json.loads(gt_answers_json)
    prompt = _build_gemini_answer_judge_prompt(
        prompt_style,
        question,
        pred_answer,
        gt_answers,
        rubric_path,
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 128,
            "responseMimeType": "application/json",
            "responseSchema": _gemini_response_schema(prompt_style),
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        GEMINI_API_ENDPOINT.format(model=model),
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    timeout = max(_env_float(ANSWER_JUDGE_TIMEOUT_ENV, 8.0), 0.1)
    with request.urlopen(req, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    candidates = response_payload.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response has no candidates")
    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    judge_payload = _parse_gemini_judge_text("\n".join(text_parts))
    return _answer_judge_payload_to_reward(judge_payload, prompt_style, rubric_path)


def _answer_judge_payload_to_reward(judge_payload: dict, prompt_style: str, rubric_path: str) -> tuple[float, dict]:
    """Identical score-to-reward mapping for every judge backend."""
    reason = str(judge_payload.get("reason", "")).strip()
    if prompt_style == ANSWER_JUDGE_STYLE_MSQA_1TO5:
        raw_score = judge_payload.get("score_1_to_5", judge_payload.get("score"))
        score_1_to_5 = _coerce_judge_score_1_to_5(raw_score)
        reward = _judge_reward_from_score_1_to_5(score_1_to_5)
        return reward, {
            "answer_judge_label": _label_from_score_1_to_5(score_1_to_5),
            "answer_judge_reason": reason[:300],
            "answer_judge_raw_score": raw_score,
            "answer_judge_score_1_to_5": score_1_to_5,
            "answer_judge_reward_mapping": "(score_1_to_5 - 1) / 4",
            "answer_judge_rubric_path": str(_resolve_rubric_path(rubric_path)),
        }

    score = _coerce_judge_score(judge_payload.get("score"))
    label = str(judge_payload.get("label", "")).strip() or "unknown"
    return score, {
        "answer_judge_label": label,
        "answer_judge_reason": reason[:300],
        "answer_judge_raw_score": judge_payload.get("score"),
        "answer_judge_score_1_to_5": None,
        "answer_judge_reward_mapping": None,
        "answer_judge_rubric_path": None,
    }


@lru_cache(maxsize=ANSWER_JUDGE_CACHE_SIZE)
def _http_answer_judge_cached(
    backend: str, model: str, prompt_style: str, rubric_path: str,
    question: str, pred_answer: str, gt_answers_json: str,
    base_url: str, max_tokens: int, reasoning_effort: str,
) -> tuple[float, dict]:
    prompt = _build_gemini_answer_judge_prompt(
        prompt_style, question, pred_answer, json.loads(gt_answers_json), rubric_path,
    )
    payload = request_judge(
        backend, model, prompt, prompt_style, base_url=base_url,
        max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        timeout=max(_env_float(ANSWER_JUDGE_TIMEOUT_ENV, 30.0), 0.1),
        retries=max(0, int(os.getenv("SCENECOT_ANSWER_JUDGE_RETRIES", "2"))),
    )
    return _answer_judge_payload_to_reward(payload, prompt_style, rubric_path)


def _answer_judge_model() -> str:
    default = {"openai": "gpt-5.6-luna", "local_qwen": "Qwen/Qwen3-VL-4B-Instruct"}.get(
        _answer_judge_backend(), "gemini-3.1-flash-lite",
    )
    return os.getenv(ANSWER_JUDGE_MODEL_ENV, default)


def _should_call_answer_judge(metrics: dict) -> tuple[bool, str]:
    backend = _answer_judge_backend()
    if backend in {"", "none", "off", "rule", "disabled"}:
        return False, "disabled"
    if backend not in {"gemini", "openai", "local_qwen"}:
        return False, f"unsupported_backend:{backend}"
    prompt_style = _answer_judge_prompt_style()
    if prompt_style not in ANSWER_JUDGE_SUPPORTED_STYLES:
        return False, f"unsupported_prompt_style:{prompt_style}"
    key_env = {"gemini": GEMINI_API_KEY_ENV, "openai": "OPENAI_API_KEY"}.get(backend)
    if key_env and not os.getenv(key_env, "").strip():
        return False, "missing_api_key"
    if metrics.get("answer_rule_direct"):
        return False, str(metrics.get("answer_rule_decision") or "direct_rule")

    token_f1 = float(metrics.get("answer_token_f1", 0.0) or 0.0)
    default_min_f1 = 0.0 if prompt_style == ANSWER_JUDGE_STYLE_MSQA_1TO5 else 0.05
    min_f1 = _env_float(ANSWER_JUDGE_MIN_F1_ENV, default_min_f1)
    default_max_f1 = 1.0 if prompt_style == ANSWER_JUDGE_STYLE_MSQA_1TO5 else 0.95
    max_f1 = _env_float(ANSWER_JUDGE_MAX_F1_ENV, default_max_f1)
    if token_f1 < min_f1:
        return False, "below_min_f1"
    if token_f1 > max_f1:
        return False, "above_max_f1"
    return True, "ambiguous_rule_score"


def maybe_apply_answer_judge(
    rule_reward: float,
    metrics: dict,
    *,
    question: str,
    pred_answer: str,
    gt_answers: list[str],
) -> tuple[float, dict]:
    backend = _answer_judge_backend()
    prompt_style = _answer_judge_prompt_style()
    rubric_path = _answer_judge_rubric_path()
    metrics.update(
        {
            "answer_judge_backend": backend,
            "answer_judge_prompt_style": prompt_style,
            "answer_judge_called": 0,
            "answer_judge_model": _answer_judge_model(),
            "answer_judge_fallback": 0,
            "answer_judge_status": "rule",
            "answer_judge_error": None,
            "answer_judge_label": None,
            "answer_judge_reason": None,
            "answer_judge_raw_score": None,
            "answer_judge_score_1_to_5": None,
            "answer_judge_reward_mapping": None,
            "answer_judge_rubric_path": rubric_path if prompt_style == ANSWER_JUDGE_STYLE_MSQA_1TO5 else None,
        }
    )

    should_call, reason = _should_call_answer_judge(metrics)
    if not should_call:
        if os.getenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "fallback") == "raise" and (
            reason == "missing_api_key" or reason.startswith("unsupported_")
        ):
            raise RuntimeError(f"Answer judge configuration error: {backend}: {reason}")
        metrics["answer_judge_status"] = reason
        return rule_reward, metrics

    model = metrics["answer_judge_model"]
    try:
        judge_args = (model, prompt_style, rubric_path, question or "", pred_answer or "",
                      _canonical_gt_answers_json(gt_answers))
        if backend == "gemini":
            judged_reward, judge_metrics = _gemini_answer_judge_cached(*judge_args)
        else:
            default_url = "https://api.openai.com/v1" if backend == "openai" else "http://127.0.0.1:8000/v1"
            judged_reward, judge_metrics = _http_answer_judge_cached(
                backend, *judge_args,
                os.getenv("SCENECOT_ANSWER_JUDGE_BASE_URL", default_url),
                int(os.getenv("SCENECOT_ANSWER_JUDGE_MAX_TOKENS", "128")),
                os.getenv("SCENECOT_ANSWER_JUDGE_REASONING_EFFORT", "none"),
            )
    except (TimeoutError, OSError, ValueError, RuntimeError, error.URLError, json.JSONDecodeError) as exc:
        truncated = isinstance(exc, JudgeOutputTruncatedError)
        allow_truncation_fallback = (
            backend in {"local_qwen", "openai"} and truncated
            and os.getenv("SCENECOT_ANSWER_JUDGE_ON_TRUNCATION", "raise") == "fallback"
        )
        if os.getenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "fallback") == "raise" and not allow_truncation_fallback:
            raise RuntimeError(f"{backend} answer judge failed: {exc}") from exc
        metrics.update(
            {
                "answer_judge_called": 1,
                "answer_judge_fallback": 1,
                "answer_judge_status": "fallback_rule_on_truncation" if truncated else "fallback_rule_on_error",
                "answer_judge_error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
        )
        return rule_reward, metrics

    metrics.update(judge_metrics)
    metrics.update(
        {
            "answer_reward_raw": float(judged_reward),
            "answer_judge_called": 1,
            "answer_judge_status": backend,
        }
    )
    return float(judged_reward), metrics


def compute_answer_reward_raw(
    pred_answer: str | None,
    gt_answers: list[str],
    dataset_name: str | None,
    *,
    has_answer: bool,
    extra_info: dict | None = None,
) -> tuple[float, dict]:
    dataset = (dataset_name or "").lower()
    if dataset == "gqa3d":
        dataset_policy = "gqa3d_strict"
    elif dataset == "msqa":
        dataset_policy = "msqa_soft"
    else:
        dataset_policy = "default_strict"

    metrics = {
        "answer_reward_raw": 0.0,
        "answer_rule_reward": 0.0,
        "answer_rule_direct": 1,
        "answer_rule_decision": "missing_answer_or_ground_truth",
        "answer_em": 0,
        "answer_re_em": 0,
        "answer_exact": 0,
        "answer_token_f1": 0.0,
        "answer_dataset_policy": dataset_policy,
        "answer_short_gt": 0,
        "answer_best_gt_index": None,
        "answer_best_gt": None,
        "answer_contains_bonus_applied": 0,
        "pred_answer": pred_answer,
        "pred_answer_em_normalized": normalize_for_em(pred_answer or ""),
        "pred_answer_re_em_normalized": normalize_for_re_em(pred_answer or ""),
        "gt_answers_em_normalized": [normalize_for_em(gt) for gt in gt_answers],
        "gt_answers_re_em_normalized": [normalize_for_re_em(gt) for gt in gt_answers],
        "num_gt_answers": len(gt_answers),
        "missing_ground_truth": 0 if gt_answers else 1,
        "has_answer": bool(has_answer),
    }
    question = _extract_question_from_extra_info(extra_info)

    if not has_answer or pred_answer is None or not gt_answers:
        return maybe_apply_answer_judge(
            0.0,
            metrics,
            question=question,
            pred_answer=pred_answer or "",
            gt_answers=gt_answers,
        )

    pred_em_norm = normalize_for_em(pred_answer)
    pred_re_em_norm = normalize_for_re_em(pred_answer)
    metrics["pred_answer"] = pred_em_norm
    if not pred_em_norm:
        metrics["answer_rule_decision"] = "empty_pred_answer"
        return maybe_apply_answer_judge(
            0.0,
            metrics,
            question=question,
            pred_answer=pred_answer or "",
            gt_answers=gt_answers,
        )
    if _is_placeholder_answer(pred_em_norm):
        metrics["answer_rule_decision"] = "placeholder_pred_answer"
        return maybe_apply_answer_judge(
            0.0,
            metrics,
            question=question,
            pred_answer="",
            gt_answers=gt_answers,
        )

    best = {
        "reward": 0.0,
        "gt_index": None,
        "gt": None,
        "token_f1": 0.0,
        "short_gt": 0,
        "em": 0,
        "re_em": 0,
        "direct": 0,
        "decision": "token_f1",
    }

    for gt_index, gt in enumerate(gt_answers):
        gt_em_norm = normalize_for_em(gt)
        gt_re_em_norm = normalize_for_re_em(gt)
        gt_tokens = tokenize_for_f1(gt)
        short_gt = int(len(gt_tokens) <= 2)
        token_f1 = _token_f1(pred_answer, gt)

        candidate_reward = 0.0
        candidate_em = 0
        candidate_re_em = 0
        candidate_direct = 0
        candidate_decision = "token_f1"

        if em_match(pred_answer, gt):
            candidate_reward = 1.0
            candidate_em = 1
            candidate_re_em = int(bool(pred_re_em_norm) and pred_re_em_norm == gt_re_em_norm)
            candidate_direct = 1
            candidate_decision = "em"
        elif re_em_match(pred_answer, gt):
            candidate_reward = 1.0
            candidate_re_em = 1
            candidate_direct = 1
            candidate_decision = "re_em"
        else:
            gt_yes_no = gt_re_em_norm if gt_re_em_norm in {"yes", "no"} else None
            if gt_yes_no:
                pred_yes_no = _explicit_yes_no(pred_answer)
                if pred_yes_no in {"yes", "no"}:
                    candidate_reward = 1.0 if pred_yes_no == gt_yes_no else 0.0
                    candidate_direct = 1
                    candidate_decision = "explicit_yes_no"
                elif pred_yes_no == "ambiguous":
                    candidate_reward = 0.0
                    candidate_direct = 1
                    candidate_decision = "ambiguous_yes_no"
                else:
                    candidate_reward = 0.0
                    candidate_decision = "missing_yes_no"
            elif dataset == "gqa3d":
                cap = 0.3 if short_gt else 0.6
                candidate_reward = min(token_f1, cap)
            elif dataset == "msqa":
                candidate_reward = token_f1
            else:
                candidate_reward = min(token_f1, 0.5)

        if (
            best["gt_index"] is None
            or candidate_reward > best["reward"]
            or (
                candidate_reward == best["reward"]
                and token_f1 > float(best["token_f1"])
            )
        ):
            best.update(
                {
                    "reward": candidate_reward,
                    "gt_index": gt_index,
                    "gt": gt_em_norm,
                    "token_f1": token_f1,
                    "short_gt": short_gt,
                    "em": candidate_em,
                    "re_em": candidate_re_em,
                    "direct": candidate_direct,
                    "decision": candidate_decision,
                }
            )

    rule_reward = float(best["reward"])
    metrics["answer_reward_raw"] = rule_reward
    metrics["answer_rule_reward"] = rule_reward
    metrics["answer_rule_direct"] = int(best["direct"])
    metrics["answer_rule_decision"] = str(best["decision"])
    metrics["answer_em"] = int(best["em"])
    metrics["answer_re_em"] = int(best["re_em"])
    metrics["answer_exact"] = int(best["em"])
    metrics["answer_token_f1"] = float(best["token_f1"])
    metrics["answer_short_gt"] = int(best["short_gt"])
    metrics["answer_best_gt_index"] = best["gt_index"]
    metrics["answer_best_gt"] = best["gt"]

    return maybe_apply_answer_judge(
        rule_reward,
        metrics,
        question=question,
        pred_answer=pred_answer,
        gt_answers=gt_answers,
    )


def compute_format_reward(
    full_protocol_valid: bool,
    *,
    catastrophic_invalid: bool = False,
    protocol_metrics: dict[str, Any] | None = None,
) -> tuple[float, dict]:
    protocol_metrics = protocol_metrics or {}
    autoclose_reward = _clip(
        _env_float(AUTOCLOSE_FORMAT_REWARD_ENV, AUTOCLOSE_FORMAT_REWARD),
        0.0,
        1.0,
    )
    minor_violation = bool(
        protocol_metrics.get("structural_protocol_valid")
        and protocol_metrics.get("final_answer_autoclosed_count")
        and not catastrophic_invalid
    )
    progress_enabled = _env_bool(PROGRESS_FORMAT_REWARD_ENABLE_ENV, False)
    first_turn_component = 0.0
    reasoning_component = 0.0
    answer_body_component = 0.0
    answer_close_component = 0.0
    if progress_enabled and not catastrophic_invalid:
        first_turn_ready = bool(protocol_metrics.get("first_turn_valid"))
        reasoning_ready = bool(
            first_turn_ready
            and int(protocol_metrics.get("final_reasoning_count", 0) or 0) == 1
            and int(protocol_metrics.get("final_reasoning_open_count", 0) or 0) == 1
            and int(protocol_metrics.get("final_reasoning_close_count", 0) or 0) == 1
        )
        answer_body_ready = bool(
            reasoning_ready
            and int(protocol_metrics.get("final_answer_open_count", 0) or 0) == 1
            and protocol_metrics.get("has_answer")
        )
        answer_close_ready = bool(
            answer_body_ready
            and not minor_violation
            and int(protocol_metrics.get("final_answer_count", 0) or 0) == 1
            and int(protocol_metrics.get("final_answer_close_count", 0) or 0) == 1
        )
        if first_turn_ready:
            first_turn_component = max(
                0.0,
                _env_float(PROGRESS_FORMAT_FIRST_TURN_W_ENV, 0.25),
            )
        if reasoning_ready:
            reasoning_component = max(
                0.0,
                _env_float(PROGRESS_FORMAT_REASONING_W_ENV, 0.25),
            )
        if answer_body_ready:
            answer_body_component = max(
                0.0,
                _env_float(PROGRESS_FORMAT_ANSWER_BODY_W_ENV, 0.20),
            )
        if answer_close_ready:
            answer_close_component = max(
                0.0,
                _env_float(PROGRESS_FORMAT_ANSWER_CLOSE_W_ENV, 0.30),
            )
    progress_reward = _clip(
        first_turn_component
        + reasoning_component
        + answer_body_component
        + answer_close_component,
        0.0,
        1.0,
    )
    if full_protocol_valid:
        reward = 1.0
        tier = "valid"
    elif minor_violation:
        # A parser-inserted close tag receives no close-tag component, while
        # the actor-generated prefix can still receive dense progress credit.
        reward = max(autoclose_reward, progress_reward)
        tier = "minor_violation"
    elif catastrophic_invalid:
        reward = 0.0
        tier = "catastrophic"
    elif progress_reward > 0.0:
        reward = progress_reward
        tier = "progress"
    else:
        reward = 0.0
        tier = "invalid"
    return reward, {
        "format_reward": reward,
        "format_gate": reward,
        "format_gate_tier": tier,
        "format_autoclose_reward": autoclose_reward,
        "format_progress_enabled": int(progress_enabled),
        "format_progress_reward": progress_reward,
        "format_progress_first_turn": first_turn_component,
        "format_progress_reasoning": reasoning_component,
        "format_progress_answer_body": answer_body_component,
        "format_progress_answer_close": answer_close_component,
    }


def compute_answer_reward_gated(answer_reward_raw: float) -> tuple[float, dict]:
    return answer_reward_raw, {
        "answer_reward_gated": answer_reward_raw,
        "answer_reward": answer_reward_raw,
        "answer_gate_type": "answer",
        "answer_gate_cap": None,
    }

def _agent_reward_enabled() -> bool:
    return _env_bool(AGENT_REWARD_ENABLED_ENV, True)


def _linear_anneal(start: float, end: float, p: float) -> float:
    p = _clip(p, 0.0, 1.0)
    return float(start) + (float(end) - float(start)) * p


def _coerce_optional_step(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        step = float(str(value).strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return step if math.isfinite(step) else None


def _mapping_value(mapping: Any, key: str) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key)
    return None


def _progress_switch_step() -> int:
    return max(0, int(_env_float(AGENT_REWARD_PROGRESS_SWITCH_STEP_ENV, 200.0)))


def _get_agent_reward_progress(
    metadata: Any = None,
    extra_info: Any = None,
    sample: Any = None,
    global_step: Any = None,
) -> float:
    forced = os.getenv(FORCE_AGENT_REWARD_PROGRESS_ENV, "").strip()
    if forced:
        try:
            return _clip(float(forced), 0.0, 1.0)
        except ValueError:
            return 1.0

    if not _env_bool(AGENT_REWARD_PROGRESS_SWITCH_ENABLED_ENV, True):
        return 1.0

    step = _coerce_optional_step(global_step)
    if step is None:
        for container, key in (
            (metadata, "global_step"),
            (metadata, "step"),
            (metadata, "rollout_step"),
            (extra_info, "global_step"),
            (extra_info, "step"),
            (extra_info, "rollout_step"),
            (sample, "global_step"),
            (sample, "step"),
            (sample, "rollout_step"),
        ):
            step = _coerce_optional_step(_mapping_value(container, key))
            if step is not None:
                break
    if step is None:
        step = _coerce_optional_step(os.getenv(GLOBAL_STEP_ENV))

    if step is None:
        return 1.0
    return 0.0 if step < _progress_switch_step() else 1.0


def _agent_reward_weights(progress: float) -> tuple[float, float]:
    format_w = _linear_anneal(
        _env_float(AGENT_FORMAT_W_START_ENV, 1.20),
        _env_float(AGENT_FORMAT_W_END_ENV, 0.80),
        progress,
    )
    answer_w = _linear_anneal(
        _env_float(AGENT_ANSWER_W_START_ENV, 0.80),
        _env_float(AGENT_ANSWER_W_END_ENV, 1.20),
        progress,
    )
    return format_w, answer_w


def _extract_first_tool_tag(text: str) -> str | None:
    content = str(text or "")
    first_match: tuple[int, str] | None = None
    bare_re = re.compile(rf"(?<![\w/])<?({TOOL_TAG_NAME_PATTERN})\s*>?(?!\w)", flags=re.IGNORECASE)
    for match in bare_re.finditer(content):
        name = match.group(1).lower()
        if name not in TOOL_TAG_NAMES:
            continue
        candidate = (match.start(), name)
        if first_match is None or candidate[0] < first_match[0]:
            first_match = candidate
    return first_match[1] if first_match is not None else None


def _repeated_sentence_threshold() -> int:
    return max(1, int(_env_float(REPEATED_SENTENCE_THRESHOLD_ENV, float(REPEATED_SENTENCE_THRESHOLD))))


def compute_repeated_sentence_penalty(scoring_text: str) -> dict:
    response_text = str(scoring_text or "")
    metrics_by_scope: list[dict[str, Any]] = []

    full_metrics = _repetition_metrics_for_text(response_text)
    metrics_by_scope.append(full_metrics)

    final_turn_exists = _has_final_signal(response_text)
    final_metrics = None
    if final_turn_exists:
        final_metrics = _repetition_metrics_for_text(extract_final_assistant_turn(response_text))
        metrics_by_scope.append(final_metrics)

    reasoning_spans = _extract_reasoning_spans_for_repetition(response_text)
    for span in reasoning_spans:
        metrics_by_scope.append(_repetition_metrics_for_text(span))

    combined = _combine_repetition_metrics(metrics_by_scope)
    repeated_loop_detected = bool(combined["repeated_sentence_loop"] or combined["repeated_ngram_loop"])
    repeated_loop_count = max(
        int(combined["max_same_sentence_count"]),
        int(combined["max_repeated_sentence_run"]),
        int(combined["repeated_ngram_count"]),
    )
    final_repeated_loop_detected = bool(
        final_metrics
        and (final_metrics.get("repeated_sentence_loop") or final_metrics.get("repeated_ngram_loop"))
    )
    return {
        "penalty_repeated_sentence": 1.0 if repeated_loop_detected else 0.0,
        "penalty_repeated_sentence_count": int(repeated_loop_detected),
        "penalty_repeated_sentence_max_count": int(repeated_loop_count),
        "final_repeated_loop_detected": int(final_repeated_loop_detected),
        "repetition_checked_full_response": 1,
        "repetition_checked_final_turn": int(final_turn_exists),
        "repetition_checked_reasoning_span": int(bool(reasoning_spans)),
        **combined,
    }


def _penalty_scoring_text(scoring_text: str, extra_info: dict | None) -> tuple[str, str]:
    if isinstance(extra_info, dict):
        actor_text = extra_info.get("scenecot_actor_generated_text")
        if actor_text is not None and str(actor_text).strip():
            return str(actor_text), "response_mask_actor_tokens"
    return str(scoring_text or ""), "full_response"


def _final_length_penalty_enabled() -> bool:
    return _env_bool(FINAL_LENGTH_PENALTY_ENV, True)


def _env_int_runtime(name: str, default: int) -> int:
    return max(0, int(_env_float(name, float(default))))


def linear_over_limit_penalty(length: int, soft: int, hard: int, max_penalty: float) -> float:
    if length <= soft:
        return 0.0
    if hard <= soft:
        return float(max_penalty)
    ratio = min(1.0, max(0.0, (int(length) - int(soft)) / (int(hard) - int(soft))))
    return float(max_penalty) * ratio


def _empty_final_length_penalty_metrics(status: str = "skipped") -> dict:
    return {
        "penalty_final_length": 0.0,
        "penalty_final_reasoning_length": 0.0,
        "penalty_final_answer_length": 0.0,
        "final_reasoning_token_length": 0,
        "final_answer_token_length": 0,
        "final_reasoning_soft_token_limit": _env_int_runtime(FINAL_REASONING_SOFT_TOKEN_LIMIT_ENV, 128),
        "final_reasoning_hard_token_limit": _env_int_runtime(FINAL_REASONING_HARD_TOKEN_LIMIT_ENV, 256),
        "final_answer_soft_token_limit": _env_int_runtime(FINAL_ANSWER_SOFT_TOKEN_LIMIT_ENV, 40),
        "final_answer_hard_token_limit": _env_int_runtime(FINAL_ANSWER_HARD_TOKEN_LIMIT_ENV, 80),
        "final_length_tokenizer_status": status,
    }


def _extract_complete_final_bodies(scoring_text: str) -> tuple[str, str] | None:
    final_turn = extract_final_assistant_turn(str(scoring_text or ""))
    reasoning_match = REASONING_RE.search(final_turn)
    answer_match = re.search(r"<answer\s*>(.*?)</answer\s*>", final_turn, flags=re.DOTALL | re.IGNORECASE)
    if reasoning_match is None or answer_match is None:
        return None
    return reasoning_match.group(1).strip(), answer_match.group(1).strip()


def compute_final_length_penalty(scoring_text: str) -> tuple[float, dict]:
    if not _final_length_penalty_enabled():
        return 0.0, _empty_final_length_penalty_metrics("disabled")

    bodies = _extract_complete_final_bodies(scoring_text)
    if bodies is None:
        return 0.0, _empty_final_length_penalty_metrics("missing_complete_final_xml")

    reasoning_body, answer_body = bodies
    reasoning_len, reasoning_status = _count_too_long_tokens(reasoning_body)
    answer_len, answer_status = _count_too_long_tokens(answer_body)
    reasoning_soft = _env_int_runtime(FINAL_REASONING_SOFT_TOKEN_LIMIT_ENV, 128)
    reasoning_hard = max(reasoning_soft + 1, _env_int_runtime(FINAL_REASONING_HARD_TOKEN_LIMIT_ENV, 256))
    answer_soft = _env_int_runtime(FINAL_ANSWER_SOFT_TOKEN_LIMIT_ENV, 40)
    answer_hard = max(answer_soft + 1, _env_int_runtime(FINAL_ANSWER_HARD_TOKEN_LIMIT_ENV, 80))
    reasoning_max_penalty = _env_float(FINAL_REASONING_MAX_PENALTY_ENV, 0.10)
    answer_max_penalty = _env_float(FINAL_ANSWER_MAX_PENALTY_ENV, 0.05)

    reasoning_penalty = linear_over_limit_penalty(
        reasoning_len,
        reasoning_soft,
        reasoning_hard,
        reasoning_max_penalty,
    )
    answer_penalty = linear_over_limit_penalty(
        answer_len,
        answer_soft,
        answer_hard,
        answer_max_penalty,
    )
    total = reasoning_penalty + answer_penalty
    status = reasoning_status if reasoning_status == answer_status else f"reasoning:{reasoning_status};answer:{answer_status}"
    return total, {
        "penalty_final_length": total,
        "penalty_final_reasoning_length": reasoning_penalty,
        "penalty_final_answer_length": answer_penalty,
        "final_reasoning_token_length": int(reasoning_len),
        "final_answer_token_length": int(answer_len),
        "final_reasoning_soft_token_limit": reasoning_soft,
        "final_reasoning_hard_token_limit": reasoning_hard,
        "final_answer_soft_token_limit": answer_soft,
        "final_answer_hard_token_limit": answer_hard,
        "final_length_tokenizer_status": status,
    }


def compute_penalty(solution_str: str, *, penalty_text: str | None = None, penalty_text_source: str = "full_response") -> tuple[float, dict]:
    text = str(solution_str if penalty_text is None else penalty_text or "")
    penalty_repeated_dot, repeated_punct_max_run, penalty_dot_only = repeated_punct_penalty(
        text,
        min_run=MAX_REPEAT_PUNCT_RUN,
        max_penalty=REPEATED_PUNCT_MAX_PENALTY,
    )
    repeated_dot_count = int(bool(penalty_repeated_dot))
    return penalty_repeated_dot, {
        "penalty": penalty_repeated_dot,
        "penalty_repeated_dot": penalty_repeated_dot,
        "penalty_repeated_dot_count": repeated_dot_count,
        "penalty_repeated_punct_max_run": repeated_punct_max_run,
        "penalty_dot_only": penalty_dot_only,
        "penalty_scope": penalty_text_source,
    }


def score_scenecot_solution(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> tuple[float, dict]:
    metadata = kwargs.get("metadata")
    sample = kwargs.get("sample")
    global_step = kwargs.get("global_step")
    extra_info = extra_info or {}
    dataset_name = extra_info.get("dataset_name")
    scoring_text, scoring_text_source = _select_scoring_text(solution_str, extra_info)
    grounding_metrics = _grounding_passthrough_metrics(extra_info)
    tool_execution_metrics = _tool_execution_metrics(extra_info)
    gcs_metrics = _gcs_passthrough_metrics(extra_info)

    first_protocol_metrics = compute_first_turn_protocol_metrics(scoring_text)
    final_protocol_metrics = compute_final_turn_protocol_metrics(scoring_text)
    protocol_metrics = {**first_protocol_metrics, **final_protocol_metrics}

    penalty_text, penalty_text_source = _penalty_scoring_text(scoring_text, extra_info)
    penalty, penalty_metrics = compute_penalty(
        scoring_text,
        penalty_text=penalty_text,
        penalty_text_source=penalty_text_source,
    )
    repeated_sentence_metrics = compute_repeated_sentence_penalty(penalty_text)
    repeated_sentence_metrics["repetition_scope"] = penalty_text_source
    repeated_sentence_penalty = float(repeated_sentence_metrics.get("penalty_repeated_sentence", 0.0) or 0.0)
    if repeated_sentence_penalty:
        penalty += repeated_sentence_penalty
        penalty_metrics["penalty"] = float(penalty_metrics["penalty"]) + repeated_sentence_penalty
    final_length_penalty, final_length_metrics = compute_final_length_penalty(penalty_text)
    if final_length_penalty:
        penalty += final_length_penalty
        penalty_metrics["penalty"] = float(penalty_metrics["penalty"]) + float(final_length_penalty)
    penalty_metrics["penalty_total"] = float(penalty)

    protocol_metrics.update(repeated_sentence_metrics)
    catastrophic_metrics = compute_catastrophic_invalid_metrics(protocol_metrics, penalty_metrics)
    catastrophic_invalid = bool(catastrophic_metrics["catastrophic_invalid"])
    final_answer_autoclosed_count = int(tool_execution_metrics["final_answer_autoclosed_count"])
    structural_protocol_valid = bool(protocol_metrics["first_turn_valid"] and protocol_metrics["final_turn_valid"])
    full_protocol_valid = (
        structural_protocol_valid
        and not catastrophic_invalid
        and final_answer_autoclosed_count == 0
    )
    protocol_metrics["structural_protocol_valid"] = int(structural_protocol_valid)
    protocol_metrics["full_protocol_valid"] = int(full_protocol_valid)
    protocol_metrics["final_answer_autoclosed_count"] = final_answer_autoclosed_count
    protocol_metrics["final_answer_autoclose_eligible_count"] = int(
        tool_execution_metrics["final_answer_autoclose_eligible_count"]
    )
    protocol_metrics["format_minor_violation"] = int(
        structural_protocol_valid and not catastrophic_invalid and final_answer_autoclosed_count > 0
    )

    parsed_answer = protocol_metrics.get("parsed_answer")
    gt_answers = normalize_ground_truths(ground_truth, extra_info)
    format_reward, format_metrics = compute_format_reward(
        full_protocol_valid,
        catastrophic_invalid=catastrophic_invalid,
        protocol_metrics=protocol_metrics,
    )
    score_autoclosed_answer = _env_bool(SCORE_AUTOCLOSED_ANSWER_ENV, True)
    # Progress-format credit teaches partial protocol compliance, but must not
    # unlock correctness reward. Only a genuinely valid protocol, or the
    # explicitly supported parser-only autoclose tier, may pass this gate.
    answer_format_eligible = full_protocol_valid or (
        score_autoclosed_answer
        and format_metrics.get("format_gate_tier") == "minor_violation"
        and float(format_metrics.get("format_autoclose_reward", 0.0) or 0.0) > 0.0
    )
    should_score_answer = (
        answer_format_eligible
        and not catastrophic_invalid
        and isinstance(parsed_answer, str)
        and bool(parsed_answer.strip())
    )
    answer_reward_raw, answer_metrics = compute_answer_reward_raw(
        parsed_answer if isinstance(parsed_answer, str) else None,
        gt_answers,
        dataset_name,
        has_answer=should_score_answer,
        extra_info=extra_info,
    )
    if not should_score_answer:
        answer_reward_raw = 0.0
        answer_metrics["answer_reward_raw"] = 0.0
        answer_metrics["answer_rule_reward"] = 0.0
        answer_metrics["answer_rule_direct"] = 1
        answer_metrics["answer_rule_decision"] = "protocol_invalid" if not catastrophic_invalid else "catastrophic_invalid"
        answer_metrics["answer_judge_called"] = 0
        answer_metrics["answer_judge_status"] = answer_metrics["answer_rule_decision"]

    answer_reward_gated, answer_gate_metrics = compute_answer_reward_gated(answer_reward_raw)
    answer_gate_metrics["answer_autoclose_scoring_enabled"] = int(score_autoclosed_answer)

    first_turn_tool_tag = _extract_first_tool_tag(_extract_first_turn_protocol_span(scoring_text))
    progress = _get_agent_reward_progress(
        metadata=metadata,
        extra_info=extra_info,
        sample=sample,
        global_step=global_step,
    )
    agent_reward_enabled = _agent_reward_enabled()
    tool_call_count = count_tool_call_tokens(scoring_text)
    first_visual_cue = first_tool_call_token(scoring_text)
    placeholder_format_metrics = compute_placeholder_format_metrics(scoring_text)
    coac_tool_reward, coac_metrics = compute_coac_tool_reward(
        extra_info=extra_info,
        gt_answers=gt_answers,
        tool_called=tool_call_count > 0,
        full_protocol_valid=full_protocol_valid,
        catastrophic_invalid=catastrophic_invalid,
    )
    tool_w = float(coac_metrics["reward/tool_weight"])
    if agent_reward_enabled:
        format_w, answer_w = _agent_reward_weights(progress)
    else:
        format_w, answer_w = 1.0, 1.0

    catastrophic_reward_value = _catastrophic_reward()
    if catastrophic_invalid:
        reward_before_clip = catastrophic_reward_value
        reward = catastrophic_reward_value
    elif format_reward > 0.0:
        # Fixed-credit fallback for modes COAC does not cover: coac_tool_reward may be a
        # fixed r_tool when COAC skips a mode such as highlight.
        reward_before_clip = (
            float(format_w) * float(format_reward)
            + float(answer_w) * float(answer_reward_gated)
            + tool_w * float(coac_tool_reward)
            - float(penalty)
        )
        reward = _clip(reward_before_clip, -1.0, _reward_max())
    else:
        reward_before_clip = -float(penalty)
        reward = _clip(reward_before_clip, -1.0, 0.0)

    metrics = {
        "reward": reward,
        "reward_before_no_answer_cap": reward_before_clip,
        "reward_before_clip": reward_before_clip,
        "reward/final_reward_before_clip": reward_before_clip,
        "reward/final_reward_after_clip": reward,
        "reward/agent_reward_enabled": int(agent_reward_enabled),
        "reward/agent_reward_progress": progress,
        "reward/progress_switch_step": _progress_switch_step(),
        "reward/progress_switch_enabled": int(_env_bool(AGENT_REWARD_PROGRESS_SWITCH_ENABLED_ENV, True)),
        "catastrophic_reward_value": catastrophic_reward_value,
        "reward/w_format": format_w,
        "reward/w_answer": answer_w,
        "reward/w_tool": tool_w,
        "reward/formula": "positive_format ? w_format*format + w_answer*answer + w_tool*tool_credit(coac_or_fixed_fallback) - penalty : -penalty",
        "agent_reward_enabled": int(agent_reward_enabled),
        "agent_reward_progress": progress,
        "progress_switch_enabled": int(_env_bool(AGENT_REWARD_PROGRESS_SWITCH_ENABLED_ENV, True)),
        "progress_switch_step": _progress_switch_step(),
        "first_tool_tag": first_turn_tool_tag,
        "reward_format_w": format_w,
        "reward_answer_w": answer_w,
        "w_format": format_w,
        "w_answer": answer_w,
        "w_tool": tool_w,
        "data_source": data_source,
        "dataset_name": dataset_name,
        "mix_source_dataset": extra_info.get("mix_source_dataset"),
        "mix_source_split": extra_info.get("mix_source_split"),
        "question_type": extra_info.get("question_type"),
        "scene_id": extra_info.get("scene_id"),
        "question_id": extra_info.get("question_id"),
        "scoring_text_source": scoring_text_source,
        "used_actor_response_text": scoring_text_source == "extra_info.response_text",
        "pred_answer_raw": parsed_answer,
        "normalized_gt_answers": gt_answers,
        "no_answer_cap_applied": False,
        "tool_call_token_count": tool_call_count,
        "first_visual_cue": first_visual_cue,
        **answer_metrics,
        **answer_gate_metrics,
        **format_metrics,
        **tool_execution_metrics,
        **gcs_metrics,
        **coac_metrics,
        **placeholder_format_metrics,
        **penalty_metrics,
        **final_length_metrics,
        **protocol_metrics,
        **catastrophic_metrics,
        **grounding_metrics,
    }
    metrics["parsed_answer"] = parsed_answer
    return float(reward), metrics


REWARD_MANAGER_EXTRA_KEYS = (
    "format_reward",
    "format_gate",
    "format_gate_tier",
    "format_autoclose_reward",
    "format_progress_enabled",
    "format_progress_reward",
    "format_progress_first_turn",
    "format_progress_reasoning",
    "format_progress_answer_body",
    "format_progress_answer_close",
    "agent_reward_enabled",
    "agent_reward_progress",
    "progress_switch_enabled",
    "progress_switch_step",
    "first_tool_tag",
    "dataset_name",
    "mix_source_dataset",
    "mix_source_split",
    "question_type",
    "scene_id",
    "question_id",
    "reward_format_w",
    "reward_answer_w",
    "w_format",
    "w_answer",
    "w_tool",
    "reward/final_reward_before_clip",
    "reward/final_reward_after_clip",
    "reward/agent_reward_enabled",
    "reward/agent_reward_progress",
    "reward/progress_switch_step",
    "reward/progress_switch_enabled",
    "reward/w_format",
    "reward/w_answer",
    "reward/w_tool",
    "reward/tool_weight",
    "answer_reward",
    "answer_reward_raw",
    "answer_rule_reward",
    "answer_rule_direct",
    "answer_rule_decision",
    "answer_judge_backend",
    "answer_judge_prompt_style",
    "answer_judge_called",
    "answer_judge_model",
    "answer_judge_status",
    "answer_judge_fallback",
    "answer_judge_error",
    "answer_judge_label",
    "answer_judge_reason",
    "answer_judge_raw_score",
    "answer_judge_score_1_to_5",
    "answer_judge_reward_mapping",
    "answer_judge_rubric_path",
    "scoring_text_source",
    "used_actor_response_text",
    "answer_reward_gated",
    "answer_gate_type",
    "answer_gate_cap",
    "answer_autoclose_scoring_enabled",
    "coac_enabled",
    "coac_delta",
    "coac_delta_group_norm",
    "coac_group_size",
    "coac_group_std",
    "coac_tau",
    "coac_r_tool",
    "coac_r_tool_override",
    "coac_fixed_r_tool",
    "coac_fallback_credit",
    "coac_tool_credit_source",
    "coac_fixed_fallback_applied",
    "coac/fallback_credit_rate",
    "coac_valid",
    "coac_copy_overlap",
    "coac_copy_thresh",
    "coac_copy_discount",
    "coac_copy_discount_enabled",
    "coac_copy_discount_applied",
    "coac_group_zscore_enabled",
    "answer_em",
    "answer_re_em",
    "answer_exact",
    "answer_token_f1",
    "answer_dataset_policy",
    "answer_short_gt",
    "answer_best_gt_index",
    "answer_best_gt",
    "answer_contains_bonus_applied",
    "pred_answer",
    "pred_answer_raw",
    "parsed_answer",
    "answer_parsed_from_answer_tag",
    "full_protocol_valid",
    "structural_protocol_valid",
    "first_turn_valid",
    "first_turn_invalid_reason",
    "final_turn_valid",
    "final_turn_invalid_reason",
    "catastrophic_invalid",
    "catastrophic_reward_value",
    "hard_invalid_reason",
    "catastrophic_invalid_reason",
    "has_think_grd",
    "has_tool_call",
    "has_reasoning",
    "has_answer",
    "rollout_has_tool_metrics",
    "rollout_tool_call_count",
    "rollout_tool_error_count",
    "rollout_cache_miss_count",
    "final_answer_autoclosed_count",
    "final_answer_autoclose_eligible_count",
    "format_minor_violation",
    "gcs_forced_mode",
    "gcs_forced_count",
    "gcs_coverage_fallback_count",
    "tool_call_token_count",
    "first_turn_tool_call_count",
    "first_turn_think_grd_count",
    "final_tool_call_count",
    "final_reasoning_count",
    "final_answer_count",
    "final_answer_open_count",
    "final_answer_close_count",
    "final_answer_unclosed_extractable",
    "answer_parsed_from_unclosed_answer_tag",
    "final_non_ascii_detected",
    "non_ascii_garbage",
    "non_ascii_ratio",
    "max_consecutive_non_ascii",
    "rare_unicode_garbage",
    "final_contamination_detected",
    "final_turn_contamination",
    "contamination_reason",
    "role_marker_contamination",
    "tool_response_contamination",
    "scenecot_instruction_contamination",
    "final_empty_answer_detected",
    "first_visual_cue",
    "has_gt_label",
    "grd_stats_valid",
    "grd_hit1",
    "grd_hit5",
    "grd_hit10",
    "grd_hit15",
    "grd_rank",
    "grd_prob",
    "grd_topk_ids",
    "grd_topk_probs",
    "format_placeholder_violation",
    "format_bad_tool_tag_placeholder",
    "format_bad_think_grd_placeholder",
    "penalty",
    "penalty_total",
    "penalty_repeated_dot",
    "penalty_repeated_dot_count",
    "penalty_repeated_punct_max_run",
    "penalty_dot_only",
    "penalty_repeated_sentence",
    "penalty_repeated_sentence_count",
    "penalty_repeated_sentence_max_count",
    "repeated_sentence_loop",
    "repeated_ngram_loop",
    "max_repeated_sentence_run",
    "max_same_sentence_count",
    "repeated_ngram_count",
    "repeated_ngram_ratio",
    "repetition_checked_full_response",
    "repetition_checked_final_turn",
    "repetition_checked_reasoning_span",
    "penalty_final_length",
    "penalty_final_reasoning_length",
    "penalty_final_answer_length",
    "final_reasoning_token_length",
    "final_answer_token_length",
    "final_reasoning_soft_token_limit",
    "final_reasoning_hard_token_limit",
    "final_answer_soft_token_limit",
    "final_answer_hard_token_limit",
    "final_length_tokenizer_status",
    "final_repeated_loop_detected",
    "no_answer_cap_applied",
)


def scenecot_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    reward, metrics = score_scenecot_solution(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    score = float(reward)
    reward_info = {
        "score": score,
        "acc": score,
    }
    for key in REWARD_MANAGER_EXTRA_KEYS:
        if key in metrics:
            reward_info[key] = metrics[key]
    return reward_info


compute_score = scenecot_reward
