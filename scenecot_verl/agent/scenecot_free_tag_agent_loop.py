from __future__ import annotations

import logging
import os
import json
import re
import hashlib
import random
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from scenecot_verl.protocol import (
    SCENECOT_OBSERVATION_INSTRUCTION,
    extract_final_assistant_turn,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,  # agent loop interface expected by verl, including run() input/output formats
    AgentLoopOutput,    # verl agent loop output format
    ToolListWrap,
    register,
)
from verl.tools.schemas import ToolResponse  # return type of the grounding tool
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from .scenecot_tag_parser import (
    THINK_GRD_END,
    VISUAL_CUE_TO_MODE,
    VISUAL_CUE_TOKENS,
    count_visual_cues,
    has_complete_answer,
    parse_action,
    strip_stop_after_first_cue,
)
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FIRST_TURN_MAX_TOKENS_ENV = "SCENECOT_FIRST_TURN_MAX_TOKENS"
FINAL_TURN_MAX_TOKENS_ENV = "SCENECOT_FINAL_TURN_MAX_TOKENS"
FIRST_TURN_TEMPERATURE_ENV = "SCENECOT_FIRST_TURN_TEMPERATURE"
ROLLOUT_DUMP_DIR_ENV = "SCENECOT_ROLLOUT_DUMP_DIR"
AUTOCLOSE_FINAL_ANSWER_ENV = "SCENECOT_AUTOCLOSE_FINAL_ANSWER"
FINAL_IGNORE_EOS_ENV = "SCENECOT_FINAL_IGNORE_EOS"
COAC_COUNTERFACTUAL_MODE_ENV = "SCENECOT_COAC_COUNTERFACTUAL_MODE"
COAC_RANDOM_SEED_ENV = "SCENECOT_COAC_RANDOM_SEED"
COAC_RANDOM_TOOL_POOL_ENV = "SCENECOT_COAC_RANDOM_TOOL_POOL"
COAC_VISUAL_ENABLE_ENV = "SCENECOT_COAC_VISUAL_ENABLE"
COAC_RANDOM_TOOL_MODES = ("prob", "loc_rect", "loc_polar", "highlight")
FINAL_ANSWER_STOP = "</answer>"
ANSWER_OPEN_TAG = "<answer>"
REASONING_RE = re.compile(r"<reasoning\s*>.*?</reasoning\s*>", flags=re.DOTALL | re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<answer\s*>", flags=re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer\s*>", flags=re.IGNORECASE)
TERMINAL_SPECIAL_TOKEN_RE = re.compile(
    r"(?:\s*(?:</s>|<\|endoftext\|>|<\|end_of_text\|>|<\|eot_id\|>|<\|im_end\|>))+\s*$",
    flags=re.IGNORECASE,
)
MODE_TO_VISUAL_CUE = {mode: cue for cue, mode in VISUAL_CUE_TO_MODE.items()}


def _merge_metrics(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0.0) + float(value)
        else:
            target[key] = value


def _truncate_after_answer(text: str) -> str:
    end_tag = FINAL_ANSWER_STOP
    end = text.find(end_tag)
    if end < 0:
        return text
    return text[: end + len(end_tag)]


def _autoclose_answer_insert_pos(text: str, *, finish_reason: str = "") -> int | None:
    """Return where a missing final ``</answer>`` may safely be inserted."""
    finish_reason_lower = str(finish_reason or "").strip().lower()
    if "length" in finish_reason_lower or "max_token" in finish_reason_lower:
        return None

    final_turn = extract_final_assistant_turn(str(text or "")).strip()
    reasoning_matches = list(REASONING_RE.finditer(final_turn))
    answer_open_matches = list(ANSWER_OPEN_RE.finditer(final_turn))
    answer_close_matches = list(ANSWER_CLOSE_RE.finditer(final_turn))
    if len(reasoning_matches) != 1 or len(answer_open_matches) != 1 or answer_close_matches:
        return None
    if reasoning_matches[0].end() > answer_open_matches[0].start():
        return None

    answer_body = final_turn[answer_open_matches[0].end() :]
    terminal_match = TERMINAL_SPECIAL_TOKEN_RE.search(answer_body)
    answer_content = answer_body[: terminal_match.start()] if terminal_match is not None else answer_body
    if not answer_content.strip() or answer_content.strip() in {"...", "…"}:
        return None

    full_terminal_match = TERMINAL_SPECIAL_TOKEN_RE.search(str(text or ""))
    return full_terminal_match.start() if full_terminal_match is not None else len(str(text or ""))


def _find_subsequence(values: list[int], pattern: list[int]) -> int:
    if not pattern:
        return -1
    last_start = len(values) - len(pattern)
    for start in range(max(0, last_start + 1)):
        if values[start : start + len(pattern)] == pattern:
            return start
    return -1


def _find_text_token_span(tokenizer: Any, token_ids: list[int], text: str) -> tuple[int, int] | None:
    target = str(text or "")
    if not token_ids or not target:
        return None
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    char_start = decoded.find(target)
    if char_start < 0:
        return None
    char_end = char_start + len(target)
    try:
        encoded = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None
    if list(encoded.get("input_ids", [])) != list(token_ids):
        return None
    token_indices = [
        index
        for index, (start, end) in enumerate(encoded.get("offset_mapping", []))
        if int(end) > char_start and int(start) < char_end
    ]
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1


def _append_unique_stop_strings(params: dict[str, Any], stop_strings: list[str]) -> None:
    raw_stop = params.get("stop")
    if raw_stop is None:
        stops: list[str] = []
    elif isinstance(raw_stop, str):
        stops = [raw_stop]
    else:
        stops = list(raw_stop)
    for stop in stop_strings:
        if stop not in stops:
            stops.append(stop)
    params["stop"] = stops


def _append_unique_bad_words(params: dict[str, Any], bad_words: tuple[str, ...] | list[str]) -> None:
    merged = list(params.get("bad_words") or [])
    for bad_word in bad_words:
        if bad_word not in merged:
            merged.append(bad_word)
    params["bad_words"] = merged


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_optional_float(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _is_object_token_cache_miss(exc: Exception) -> bool:
    return isinstance(exc, FileNotFoundError) and str(exc).startswith("Cached SceneCOT object tokens not found.")


def _native_finish_reason(output: Any) -> str:
    extra_fields = getattr(output, "extra_fields", None)
    if isinstance(extra_fields, dict):
        reason = extra_fields.get("finish_reason") or extra_fields.get("native_finish_reason")
        if reason is not None:
            return str(reason)
    return ""


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write_rollout_dump(payload: dict[str, Any]) -> None:
    dump_dir = os.getenv(ROLLOUT_DUMP_DIR_ENV, "").strip()
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, "rollouts.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    text = str(value).strip()
    return text or None


def _coac_counterfactual_mode() -> str:
    value = os.getenv(COAC_COUNTERFACTUAL_MODE_ENV, "").strip().lower()
    aliases = {"": "mask", "masked": "mask", "masked_token": "mask"}
    value = aliases.get(value, value)
    if value not in {"mask", "random_tokens", "random_tool"}:
        raise ValueError(
            f"{COAC_COUNTERFACTUAL_MODE_ENV} must be one of "
            "'mask', 'random_tokens', or 'random_tool'."
        )
    return value


def _stable_random(seed_parts: tuple[Any, ...]) -> random.Random:
    payload = "\x1f".join(str(part) for part in seed_parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))


def _random_tool_mode(actual_mode: str, *, seed_parts: tuple[Any, ...]) -> str:
    configured = os.getenv(COAC_RANDOM_TOOL_POOL_ENV, ",".join(COAC_RANDOM_TOOL_MODES))
    requested = [item.strip() for item in configured.split(",") if item.strip()]
    candidates = [
        mode
        for mode in requested
        if mode in COAC_RANDOM_TOOL_MODES
        and mode != actual_mode
        and (mode != "highlight" or _env_bool(COAC_VISUAL_ENABLE_ENV, False))
    ]
    if not candidates:
        candidates = [
            mode
            for mode in COAC_RANDOM_TOOL_MODES
            if mode != actual_mode
            and (mode != "highlight" or _env_bool(COAC_VISUAL_ENABLE_ENV, False))
        ]
    return _stable_random(seed_parts).choice(candidates)


SYSTEM_PROMPT_FILE_ENV = "SCENECOT_SYSTEM_PROMPT_FILE"
_SYSTEM_PROMPT_CACHE: dict[str, str] = {}


def _load_system_prompt_override() -> str | None:
    raw_path = os.getenv(SYSTEM_PROMPT_FILE_ENV, "").strip()
    if not raw_path:
        return None
    path = os.path.abspath(os.path.expanduser(raw_path))
    cached = _SYSTEM_PROMPT_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as handle:
        prompt = handle.read().strip()
    if not prompt:
        raise ValueError(f"{SYSTEM_PROMPT_FILE_ENV} points to an empty prompt file: {path}")
    _SYSTEM_PROMPT_CACHE[path] = prompt
    return prompt


def _apply_system_prompt_override(messages: list[Any]) -> list[Any]:
    override = _load_system_prompt_override()
    patched = list(messages)
    if override is None:
        return patched
    for index, message in enumerate(patched):
        if isinstance(message, dict) and message.get("role") == "system":
            updated = dict(message)
            updated["content"] = override
            patched[index] = updated
            return patched
    return [{"role": "system", "content": override}, *patched]



@register("scenecot_free_tag_agent")    # register this agent loop with verl so it can be selected by name
class SceneCOTFreeTagAgentLoop(AgentLoopBase):

    def __init__(
        self,
        *args,
        tools: Optional[ToolListWrap] = None,
        max_visual_cue_turns: int = 1,
        max_tokens_per_turn: int = 512,
        max_final_tokens: int = 1024,
        topk: int = 20,
        stop_on_answer: bool = True,
        insert_grounding_error: bool = True,
        block_visual_cues_after_max: bool = True,
        post_tool_answer_nudge: bool = True,
        post_tool_prefill: str = "",
        protocol_codec: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.max_visual_cue_turns = int(max_visual_cue_turns)
        self.max_tokens_per_turn = _env_int(FIRST_TURN_MAX_TOKENS_ENV, 96) # max tokens generated per turn before a tool call
        self.max_final_tokens = _env_int(FINAL_TURN_MAX_TOKENS_ENV, 512)
        self.topk = int(topk)
        self.stop_on_answer = bool(stop_on_answer)
        self.insert_grounding_error = bool(insert_grounding_error)
        self.block_visual_cues_after_max = bool(block_visual_cues_after_max)
        self.post_tool_answer_nudge = bool(post_tool_answer_nudge)
        self.post_tool_prefill = str(post_tool_prefill or "")
        self._configure_protocol(protocol_codec)
        self.prompt_length = self.rollout_config.prompt_length  # max input prompt length
        self.response_length = self.rollout_config.response_length  # max length of the whole response

    def _configure_protocol(self, protocol_codec: str | None) -> None:
        """Configure the shared literal free-tag protocol.

        Model-specific subclasses own any non-empty transport codec. Keeping
        that validation here prevents a literal-protocol loop from
        accidentally consuming another model's private token aliases.
        """
        normalized = str(protocol_codec or "").strip().lower()
        if normalized not in {"", "none"}:
            raise ValueError(
                "SceneCOTFreeTagAgentLoop only supports literal free tags; "
                "use SceneCOTQwen25AgentLoop for a Qwen2.5 protocol codec."
            )
        self.protocol_codec = normalized

    def _text_for_model(self, text: str) -> str:
        return text

    def _text_from_model(self, text: str) -> str:
        return text

    def _messages_for_model(self, messages: list[Any]) -> list[Any]:
        return messages

    def _tool_messages_for_model(self, messages: list[Any]) -> list[Any]:
        """Return the transport representation used for tool observations.

        Most models use the same representation for initial and tool messages.
        A model-specific subclass may override only this boundary when its SFT
        checkpoint learned observation tags as private tokenizer aliases while
        still requiring a literal instruction prompt.
        """
        return self._messages_for_model(messages)

    def _extra_final_stop_strings(self) -> list[str]:
        return []

    def _extra_visual_cue_stop_strings(self, visual_cue: str) -> list[str]:
        return []

    def _actor_forbidden_tokens(self) -> tuple[str, ...]:
        return ()

    def _post_tool_prefill_model_text(self, text: str) -> str:
        return self._text_for_model(text)

    def _observation_instruction(self) -> str:
        return SCENECOT_OBSERVATION_INSTRUCTION

    def _inline_tool_response_enabled(self, has_images: bool) -> bool:
        return False

    async def _encode_inline_tool_response(
        self,
        *,
        payload_text: str,
        new_images: list[Any],
        mm_processor_kwargs: dict[str, Any],
    ) -> list[int]:
        if new_images:
            raise ValueError(
                "The shared SceneCOT loop cannot inline image observations; "
                "a model-specific loop must define their multimodal boundary."
            )
        return await self._encode_text(f"\n{payload_text}")

    def _is_protocol_answer_stage(self, actor_text: str) -> bool:
        return False

    def _configure_protocol_answer_sampling(self, params: dict[str, Any]) -> int | None:
        return None

    def _sanitize_decoded_text(self, text: str) -> tuple[str, int]:
        return text, 0

    def _split_protocol_premature_answer_close(
        self,
        actor_text: str,
        chunk_text: str,
    ) -> tuple[str, bool]:
        return chunk_text, False

    def _protocol_fallback_answer(self, actor_text: str, chunk_text: str) -> str | None:
        return None

    def _public_first_turn_text(
        self,
        chunk_text: str,
        trigger: str | None,
        *,
        include_trigger: bool,
    ) -> str:
        return chunk_text

    def _protocol_transition_text(self) -> str | None:
        return None

    def _rollout_logprob_diagnostic_enabled(self, sampling_params: dict[str, Any]) -> bool:
        return False

    def _preserve_sampled_actor_token_ids(self) -> bool:
        return False

    def _resolve_sampled_prefix_ids(
        self,
        sampled_token_ids: list[int],
        *,
        decoded_text: str,
        retained_text: str,
    ) -> list[int]:
        if not retained_text:
            return []
        if decoded_text == retained_text:
            return sampled_token_ids
        if not decoded_text.startswith(retained_text):
            raise RuntimeError(
                "Autonomous protocol retained text is not a prefix of the "
                "decoded sampled output: "
                f"decoded_len={len(decoded_text)}, retained_len={len(retained_text)}, "
                f"decoded_tail={decoded_text[-200:]!r}, "
                f"retained_tail={retained_text[-200:]!r}."
            )
        for prefix_end in range(1, len(sampled_token_ids) + 1):
            prefix_text = self._decode_tokens(sampled_token_ids[:prefix_end])
            prefix_text, _ = self._sanitize_decoded_text(prefix_text)
            # Parser boundaries are character boundaries, not necessarily
            # tokenizer boundaries. A Qwen BPE token can contain the final
            # ``>`` of a stop tag plus following text. Keep the first original
            # sampled-token prefix that covers the retained character span.
            if prefix_text.startswith(retained_text):
                return sampled_token_ids[:prefix_end]
        # The full prefix is decoded and sanitized by the same methods used to
        # produce decoded_text, so the loop above must resolve whenever the
        # prefix invariant holds. Keep a defensive error for tokenizer bugs.
        raise RuntimeError(
            "Autonomous protocol could not map a valid retained character "
            "prefix to sampled tokens: "
            f"sampled_ids={sampled_token_ids[:32]}, "
            f"decoded_len={len(decoded_text)}, retained_len={len(retained_text)}, "
            f"decoded_tail={decoded_text[-200:]!r}, "
            f"retained_tail={retained_text[-200:]!r}."
        )

    def _rollout_preserved_prefix_metric_name(self) -> str:
        return "scenecot/rollout_preserved_prefix_tokens"

    def _augment_model_specific_metrics(
        self,
        *,
        metrics: dict[str, Any],
        scene_metrics: dict[str, Any],
        rollout_dump: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        """Allow model subclasses to expose namespaced compatibility metrics."""

    def _sampling_params_for_turn(
        self,
        sampling_params: dict[str, Any],
        *,
        max_tokens: int,
        remaining_response_tokens: int,
        block_visual_cues: bool = False,
        final_turn: bool = False,
        first_turn: bool = False,
    ) -> dict[str, Any]:
        params = dict(sampling_params)
        requested_max_tokens = int(max_tokens)
        remaining_tokens = max(0, int(remaining_response_tokens))
        if requested_max_tokens <= 0:
            params["max_tokens"] = remaining_tokens
        else:
            params["max_tokens"] = max(0, min(requested_max_tokens, remaining_tokens))
        _append_unique_stop_strings(
            params,
            [
                stop
                for token in VISUAL_CUE_TOKENS
                for stop in (
                    self._text_for_model(token),
                    *self._extra_visual_cue_stop_strings(token),
                )
            ],
        )
        if final_turn and self.stop_on_answer:
            _append_unique_stop_strings(params, [self._text_for_model(FINAL_ANSWER_STOP)])
            _append_unique_stop_strings(params, self._extra_final_stop_strings())
            # Preserve the sampled close tag in the PPO trajectory. During a
            # format-bootstrap run, ignore EOS so the policy must either sample
            # </answer> or exhaust the final-turn token budget.
            params["include_stop_str_in_output"] = True
            if _env_bool(FINAL_IGNORE_EOS_ENV, False):
                params["ignore_eos"] = True
        else:
            params.setdefault("include_stop_str_in_output", True)
        _append_unique_bad_words(params, self._actor_forbidden_tokens())
        if first_turn and not final_turn:
            first_turn_temperature = _env_optional_float(FIRST_TURN_TEMPERATURE_ENV)
            if first_turn_temperature is not None:
                params["temperature"] = first_turn_temperature
        if block_visual_cues:
            bad_words = list(params.get("bad_words") or [])
            for token in VISUAL_CUE_TOKENS:
                for cue in (
                    self._text_for_model(token),
                    *self._extra_visual_cue_stop_strings(token),
                ):
                    if cue not in bad_words:
                        bad_words.append(cue)
            params["bad_words"] = bad_words
        return params

    async def _encode_text(self, text: str) -> list[int]:
        if not text:
            return []
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(self._text_for_model(text), add_special_tokens=False),
        )

    async def _encode_post_tool_prefill(self, text: str) -> list[int]:
        if not text:
            return []
        model_text = self._post_tool_prefill_model_text(text)
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(model_text, add_special_tokens=False),
        )

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        return self._text_from_model(decoded)

    def _tool_response_text(self, text: str | None) -> str:
        text = text or ""
        if not self.post_tool_answer_nudge:
            return text
        nudge = self._observation_instruction()
        return f"{text}\n{nudge}" if text else nudge

    async def _append_text_tokens(self,*,text: str,mask_value: int,current_ids: list[int],response_ids: list[int],response_mask: list[int],) -> bool:
        token_ids = await self._encode_text(text)
        if not token_ids:
            return False
        remaining = self.response_length - len(response_mask)
        if remaining <= 0:
            return True
        truncated = False
        if len(token_ids) > remaining:
            token_ids = token_ids[:remaining]
            truncated = True
        current_ids.extend(token_ids)
        response_ids.extend(token_ids)
        response_mask.extend([mask_value] * len(token_ids))
        return truncated or len(response_mask) >= self.response_length

    async def _maybe_autoclose_final_answer(
        self,
        *,
        actor_text: str,
        finish_reason: str,
        current_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
    ) -> tuple[str, int, int]:
        """Append a parser-only answer close with mask 0 when safely eligible."""
        if not _env_bool(AUTOCLOSE_FINAL_ANSWER_ENV, False):
            return actor_text, 0, 0

        insert_pos = _autoclose_answer_insert_pos(actor_text, finish_reason=finish_reason)
        if insert_pos is None:
            return actor_text, 0, 0

        close_ids = await self._encode_text(FINAL_ANSWER_STOP)
        if not close_ids or len(response_mask) + len(close_ids) > self.response_length:
            return actor_text, 0, 1

        current_ids.extend(close_ids)
        response_ids.extend(close_ids)
        # The close tag is an environment-provided parsing aid, not a sampled
        # policy action, so it must not receive policy-gradient credit.
        response_mask.extend([0] * len(close_ids))
        repaired_text = actor_text[:insert_pos] + FINAL_ANSWER_STOP + actor_text[insert_pos:]
        return repaired_text, 1, 1

    async def _append_tool_response(
        self,
        *,
        tool_response: ToolResponse,    # result returned by the grounding tool
        current_ids: list[int],         # full model context: system+user at first, then assistant generations and tool responses are appended
        response_ids: list[int],        # response tokens for training, excluding the prompt
        response_mask: list[int],       
        image_data: list[Image.Image] | None,
        mm_processor_kwargs: dict[str, Any],
    ) -> tuple[bool, list[Image.Image] | None, tuple[int, int] | None]:
        new_images: list[Any] = []
        if tool_response.image:
            if not getattr(self.processor, "image_processor", None):
                raise ValueError(
                    "SceneCOT highlight tool returned an image, but the current processor has no image_processor."
                )
            new_images.extend([image for image in tool_response.image if image is not None])

        payload_text = tool_response.text or ""
        inline_tool_response = self._inline_tool_response_enabled(bool(new_images))
        if inline_tool_response:
            # Model-specific subclasses may preserve an SFT-compatible inline
            # observation. It remains environment-provided (mask 0).
            tool_ids = await self._encode_inline_tool_response(
                payload_text=payload_text,
                new_images=new_images,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        else:
            tool_text = self._tool_response_text(payload_text)
            if new_images:
                content: str | list[dict[str, Any]] = [{"type": "image"} for _ in new_images]
                if tool_text:
                    content.append({"type": "text", "text": tool_text})
                add_messages = [{"role": "tool", "content": content}]
            else:
                add_messages = [{"role": "tool", "content": tool_text}]

            tool_ids = await self.apply_chat_template(
                self._tool_messages_for_model(add_messages),
                images=new_images if new_images else None,
                videos=None,
                audios=None,
                mm_processor_kwargs=mm_processor_kwargs,
                remove_system_prompt=True,
            )
        payload_span: tuple[int, int] | None = None
        if payload_text:
            payload_ids = await self._encode_text(payload_text)
            payload_start = _find_subsequence(tool_ids, payload_ids)
            if payload_start >= 0:
                payload_span = (payload_start, payload_start + len(payload_ids))
            else:
                payload_span = _find_text_token_span(self.tokenizer, tool_ids, payload_text)
        remaining = self.response_length - len(response_mask)
        if remaining <= 0:
            return True, image_data, payload_span
        if new_images and len(tool_ids) > remaining:
            raise RuntimeError(
                "SceneCOT highlight tool response does not fit in the remaining response budget "
                f"({len(tool_ids)} image/tool tokens needed, {remaining} tokens remaining). "
                "Increase data.max_response_length / MAX_RESPONSE_LENGTH or reduce image processor max_pixels."
            )
        truncated = False
        if len(tool_ids) > remaining:
            tool_ids = tool_ids[:remaining]
            truncated = True
        if payload_span is not None and payload_span[1] > len(tool_ids):
            payload_span = None

        if new_images:
            if image_data is None:
                image_data = []
            elif not isinstance(image_data, list):
                image_data = [image_data]
            image_data.extend(new_images)

        current_ids.extend(tool_ids)
        response_ids.extend(tool_ids)
        response_mask.extend([0] * len(tool_ids))
        return truncated or len(response_mask) >= self.response_length, image_data, payload_span

    def _get_ground_tool(self):
        tool = self.tools.get("scenecot_ground")
        if tool is None:
            available = ", ".join(sorted(self.tools)) or "<none>"
            raise RuntimeError(f"SceneCOT grounding tool 'scenecot_ground' is not loaded. Available tools: {available}")
        return tool

    # Create a tool instance
    async def _create_ground_tool_instance(self, tool, tools_kwargs: dict[str, Any], kwargs: dict[str, Any]) -> str:
        create_bundle = tools_kwargs.get("scenecot_ground", {}) # fields: scene_id, question_id, gt_objects, dataset_name, grounding prompt, agent_pos_ori
        create_kwargs = create_bundle.get("create_kwargs", {}) if isinstance(create_bundle, dict) else {}
        reward_model = kwargs.get("reward_model", {}) or {} # reward model: style: rule, ground_truth: xxx
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        instance_id, _ = await tool.create(create_kwargs=create_kwargs, ground_truth=ground_truth)
        return instance_id

    async def _execute_ground_tool(
        self,
        *,
        tool,
        instance_id: str,
        query: str,
        mode: str,
        topk: int,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        parameters = {"query": query, "mode": mode, "topk": topk}
        return await tool.execute(instance_id, parameters)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        rollout_logprob_diagnostic = self._rollout_logprob_diagnostic_enabled(sampling_params)

        messages = _apply_system_prompt_override(list(kwargs["raw_prompt"]))
        multi_modal_data = await self.process_multi_modal_info(messages)
        image_data = multi_modal_data.get("images")
        mm_processor_kwargs = self._get_mm_processor_kwargs()

        prompt_ids = await self.apply_chat_template(
            self._messages_for_model(messages),
            images=image_data,
            videos=None,
            audios=None,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        current_ids = list(prompt_ids)  # full token list the LLM continues from; updated after each generation and tool call
        response_ids: list[int] = []    # excludes the initial prompt
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if rollout_logprob_diagnostic else None
        metrics: dict[str, Any] = {}
        extra_fields: dict[str, Any] = {"scenecot_actions": []}
        request_id = uuid4().hex
        actor_text = ""
        action_scan_start = 0   # offset in the actor text where action parsing starts; advanced after each turn
        visual_cue_count = 0    # number of visual cues triggered
        assistant_turns = 0     # number of LLM generation turns
        user_turns = 0      # number of tool-response turns
        tool_call_count = 0     # number of actions parsed and sent to the tool
        tool_error_count = 0    # number of failed tool calls
        cache_miss_count = 0
        obs_span: tuple[int, int] | None = None
        obs_context_end_pos: int | None = None
        obs_text_for_coac = ""
        mode_tag_end_pos: int | None = None
        coac_random_tool_mode: str | None = None
        coac_random_tool_prefix_ids: list[int] | None = None
        coac_random_tool_response_ids: list[int] | None = None
        coac_random_tool_payload_span: tuple[int, int] | None = None
        coac_random_tool_image_path: str | None = None
        coac_random_tool_error = ""
        answer_tag_pos: int | None = None
        first_generation_finish_reason: str | None = None
        final_generation_finish_reason: str | None = None
        first_generation_native_finish_reason: str | None = None
        final_generation_native_finish_reason: str | None = None
        final_answer_autoclosed_count = 0
        final_answer_autoclose_eligible_count = 0
        missing_think_grd_count = 0 # times the model triggered a visual cue without generating <think_grd>...</think_grd>
        has_think_grd_count = 0   # times the model triggered a visual cue and correctly generated <think_grd>...</think_grd>
        cue_counts = {"prob": 0, "loc_rect": 0, "loc_polar": 0, "highlight": 0}
        latest_tool_metrics: dict[str, Any] = {}
        post_tool_prefill_count = 0
        post_tool_prefill_token_count = 0
        protocol_transition_count = 0
        protocol_transition_token_count = 0
        protocol_fallback_answer_count = 0
        protocol_fallback_answer_token_count = 0
        actor_mm_placeholder_stripped_count = 0

        extra_info = kwargs.get("extra_info", {}) or {}
        forced_mode = _coerce_optional_str(kwargs.get("scenecot_forced_mode"))
        forced_trigger = MODE_TO_VISUAL_CUE.get(forced_mode or "")
        gcs_forced_injected_count = 0
        gcs_coverage_fallback_count = 0
        gcs_forced_masked_token_count = 0
        reward_model = kwargs.get("reward_model", {}) or {}
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        gt_answer_text = _first_ground_truth_text(ground_truth)
        tools_kwargs = extra_info.get("tools_kwargs", {})
        ground_tool = self._get_ground_tool()
        tool_instance_id: str | None = None

        try:
            tool_instance_id = await self._create_ground_tool_instance(ground_tool, tools_kwargs, kwargs)
            while len(response_mask) < self.response_length:

                remaining = self.response_length - len(response_mask)
                final_turn = visual_cue_count >= self.max_visual_cue_turns
                first_turn = assistant_turns == 0 and not final_turn
                max_tokens = self.max_final_tokens if final_turn else self.max_tokens_per_turn
                block_visual_cues = (
                    self.block_visual_cues_after_max and visual_cue_count >= self.max_visual_cue_turns
                )
                turn_sampling_params = self._sampling_params_for_turn(
                    sampling_params,
                    max_tokens=max_tokens,
                    remaining_response_tokens=remaining,
                    block_visual_cues=block_visual_cues,
                    final_turn=final_turn,
                    first_turn=first_turn,
                )
                protocol_answer_stage = final_turn and self._is_protocol_answer_stage(actor_text)
                if protocol_answer_stage:
                    answer_max_tokens = self._configure_protocol_answer_sampling(
                        turn_sampling_params
                    )
                    metrics["scenecot/protocol_answer_stage_count"] = (
                        metrics.get("scenecot/protocol_answer_stage_count", 0) + 1
                    )
                    if answer_max_tokens is not None:
                        metrics["scenecot/protocol_answer_max_tokens"] = answer_max_tokens
                if first_turn:
                    metrics["scenecot/first_turn_temperature"] = float(
                        turn_sampling_params.get("temperature", sampling_params.get("temperature", 0.0)) or 0.0
                    )
                if final_turn:
                    metrics["scenecot/final_turn_max_tokens"] = self.max_final_tokens
                    metrics["scenecot/final_generation_max_tokens"] = int(turn_sampling_params.get("max_tokens", 0) or 0)
                if block_visual_cues:
                    metrics["scenecot/block_visual_cues_after_max"] = 1
                if first_turn and forced_trigger is not None:
                    _append_unique_stop_strings(
                        turn_sampling_params,
                        [self._text_for_model(THINK_GRD_END)],
                    )

                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=current_ids,
                        sampling_params=turn_sampling_params,
                        image_data=image_data,
                        video_data=None,
                        audio_data=None,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
                else:
                    metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

                if not extra_fields.get("rollout_extra_fields"):
                    extra_fields["rollout_extra_fields"] = output.extra_fields

                stop_reason = str(output.stop_reason or "")
                native_finish_reason = _native_finish_reason(output)
                if first_turn and first_generation_finish_reason is None:
                    first_generation_finish_reason = stop_reason
                    first_generation_native_finish_reason = native_finish_reason
                if final_turn:
                    final_generation_finish_reason = stop_reason
                    final_generation_native_finish_reason = native_finish_reason

                decoded = self._decode_tokens(output.token_ids)
                decoded, stripped_mm_tokens = self._sanitize_decoded_text(decoded)
                if stripped_mm_tokens:
                    actor_mm_placeholder_stripped_count += stripped_mm_tokens
                    metrics["scenecot/actor_mm_placeholder_stripped_count"] = (
                        actor_mm_placeholder_stripped_count
                    )

                # if there is a visual cue, truncate at it; otherwise use the full decoded text
                chunk_text, trigger = strip_stop_after_first_cue(decoded)
                if self.stop_on_answer and not trigger:
                    chunk_text = _truncate_after_answer(chunk_text)
                # Preserve the pre-GCS-replacement view. Exact sampled-token
                # accounting must first resolve against what vLLM actually
                # sampled, not against the later text with its cue removed.
                sampled_chunk_text = chunk_text
                sampled_trigger = trigger

                if trigger is None and decoded != chunk_text:
                    _, trigger = strip_stop_after_first_cue(chunk_text)

                injected_forced_trigger: str | None = None
                forced_replaced = 0
                if first_turn and forced_trigger is not None:
                    if trigger is not None:
                        chunk_text = chunk_text[: -len(trigger)] if chunk_text.endswith(trigger) else chunk_text.replace(trigger, "", 1)
                        trigger = forced_trigger
                        injected_forced_trigger = forced_trigger
                        forced_replaced = 1
                    elif THINK_GRD_END in chunk_text:
                        trigger = forced_trigger
                        injected_forced_trigger = forced_trigger
                    else:
                        gcs_coverage_fallback_count += 1

                if trigger is None and output.stop_reason == "completed" and count_visual_cues(decoded) == 0:
                    metrics["scenecot/stop_without_visual_cue"] = metrics.get("scenecot/stop_without_visual_cue", 0) + 1

                if not chunk_text and not output.token_ids and injected_forced_trigger is None:
                    break

                protocol_transition = False
                if final_turn:
                    chunk_text, protocol_transition = (
                        self._split_protocol_premature_answer_close(actor_text, chunk_text)
                    )

                terminated = False
                protocol_fallback = None
                if final_turn and protocol_answer_stage:
                    protocol_fallback = self._protocol_fallback_answer(actor_text, chunk_text)
                if protocol_fallback:
                    fallback_token_count = len(await self._encode_text(protocol_fallback))
                    terminated = await self._append_text_tokens(
                        text=protocol_fallback,
                        mask_value=0,
                        current_ids=current_ids,
                        response_ids=response_ids,
                        response_mask=response_mask,
                    )
                    if fallback_token_count:
                        actor_text += protocol_fallback
                        protocol_fallback_answer_count += 1
                        protocol_fallback_answer_token_count += fallback_token_count

                if chunk_text:
                    public_chunk_text = (
                        self._public_first_turn_text(
                            chunk_text,
                            trigger,
                            include_trigger=injected_forced_trigger is None,
                        )
                        if first_turn
                        else chunk_text
                    )
                    if final_turn and answer_tag_pos is None:
                        answer_index = public_chunk_text.lower().find(ANSWER_OPEN_TAG)
                        if answer_index >= 0:
                            prefix_to_answer = public_chunk_text[: answer_index + len(ANSWER_OPEN_TAG)]
                            answer_tag_pos = len(response_ids) + len(await self._encode_text(prefix_to_answer))
                    if response_logprobs is not None:
                        # LLaMA's slow SentencePiece path can sample a dummy
                        # whitespace token before a user-defined XML token:
                        #   [29871, 31025] -> "<think_grd>"
                        # Re-encoding the decoded text with the fast tokenizer
                        # yields only [31025]. Keep the exact vLLM prefix here;
                        # otherwise actor log-probs would be recomputed for a
                        # different trajectory and the sync diagnostic would
                        # be invalid.
                        sampled_token_ids = list(output.token_ids)
                        if decoded == chunk_text:
                            sampled_prefix_end = len(sampled_token_ids)
                        else:
                            sampled_prefix_end = None
                            # Decode token prefixes instead of trying to
                            # re-encode chunk_text. SentencePiece
                            # decode->encode is not bijective around dummy
                            # whitespace and user-defined XML pieces.
                            for prefix_end in range(1, len(sampled_token_ids) + 1):
                                if self._decode_tokens(sampled_token_ids[:prefix_end]) == chunk_text:
                                    sampled_prefix_end = prefix_end
                            if sampled_prefix_end is None:
                                raise RuntimeError(
                                    "Rollout log-prob diagnostic cannot locate a vLLM token "
                                    "prefix that decodes to the retained generated chunk: "
                                    f"sampled_ids={sampled_token_ids[:32]}, "
                                    f"decoded={decoded[:200]!r}, chunk={chunk_text[:200]!r}."
                                )
                        if sampled_prefix_end <= 0:
                            raise RuntimeError(
                                "Rollout log-prob diagnostic resolved an empty sampled prefix "
                                "for a non-empty generated chunk."
                            )
                        sampled_prefix_ids = sampled_token_ids[:sampled_prefix_end]
                        if output.log_probs is None:
                            raise RuntimeError(
                                "Rollout log-prob diagnostic requested, but vLLM returned no log_probs."
                            )
                        if len(output.log_probs) < sampled_prefix_end:
                            raise RuntimeError(
                                "Rollout log-prob diagnostic received too few log_probs: "
                                f"{len(output.log_probs)} for {sampled_prefix_end} sampled-prefix tokens."
                            )
                        remaining_generated = self.response_length - len(response_mask)
                        recorded_prefix_ids = sampled_prefix_ids[:remaining_generated]
                        current_ids.extend(recorded_prefix_ids)
                        response_ids.extend(recorded_prefix_ids)
                        response_mask.extend([1] * len(recorded_prefix_ids))
                        response_logprobs.extend(
                            float(value) for value in output.log_probs[: len(recorded_prefix_ids)]
                        )
                        preserved_prefix_metric = self._rollout_preserved_prefix_metric_name()
                        metrics[preserved_prefix_metric] = (
                            metrics.get(preserved_prefix_metric, 0)
                            + max(0, len(sampled_prefix_ids) - len(await self._encode_text(chunk_text)))
                        )
                        terminated = (
                            len(recorded_prefix_ids) < len(sampled_prefix_ids)
                            or len(response_mask) >= self.response_length
                        )
                    elif self._preserve_sampled_actor_token_ids():
                        sampled_token_ids = list(output.token_ids)
                        sampled_retained_text = sampled_chunk_text
                        if (
                            injected_forced_trigger is not None
                            and forced_replaced
                            and sampled_trigger is not None
                        ):
                            if not sampled_retained_text.endswith(sampled_trigger):
                                raise RuntimeError(
                                    "GCS replacement expected the sampled visual cue "
                                    "at the end of the retained actor text: "
                                    f"cue={sampled_trigger!r}, "
                                    f"retained_tail={sampled_retained_text[-200:]!r}."
                                )
                            # Resolve directly to the token boundary before the
                            # sampled cue. Encoding the cue and removing that
                            # suffix is unsafe because Qwen may merge ``>`` and
                            # a trailing newline into a single token.
                            sampled_retained_text = sampled_retained_text[
                                : -len(sampled_trigger)
                            ]
                        sampled_prefix_ids = self._resolve_sampled_prefix_ids(
                            sampled_token_ids,
                            decoded_text=decoded,
                            retained_text=sampled_retained_text,
                        )
                        remaining_generated = self.response_length - len(response_mask)
                        recorded_prefix_ids = sampled_prefix_ids[:remaining_generated]
                        current_ids.extend(recorded_prefix_ids)
                        response_ids.extend(recorded_prefix_ids)
                        response_mask.extend([1] * len(recorded_prefix_ids))
                        metrics["scenecot/exact_sampled_actor_token_count"] = (
                            metrics.get("scenecot/exact_sampled_actor_token_count", 0)
                            + len(recorded_prefix_ids)
                        )
                        terminated = (
                            len(recorded_prefix_ids) < len(sampled_prefix_ids)
                            or len(response_mask) >= self.response_length
                        )
                    else:
                        terminated = await self._append_text_tokens(
                            text=chunk_text,
                            mask_value=1,
                            current_ids=current_ids,
                            response_ids=response_ids,
                            response_mask=response_mask,
                        )
                    if trigger is not None and injected_forced_trigger is None and mode_tag_end_pos is None:
                        mode_tag_end_pos = len(response_ids)
                    actor_text += public_chunk_text
                    assistant_turns += 1
                elif output.token_ids:
                    assistant_turns += 1

                if not terminated and protocol_transition:
                    transition_text = self._protocol_transition_text()
                    if transition_text is None:
                        raise RuntimeError(
                            "Protocol requested a masked transition without transition text."
                        )
                    transition_ids = await self._encode_post_tool_prefill(transition_text)
                    remaining_after_summary = self.response_length - len(response_mask)
                    if transition_ids and len(transition_ids) <= remaining_after_summary:
                        current_ids.extend(transition_ids)
                        response_ids.extend(transition_ids)
                        response_mask.extend([0] * len(transition_ids))
                        actor_text += transition_text
                        answer_tag_pos = len(response_ids)
                        protocol_transition_count += 1
                        protocol_transition_token_count += len(transition_ids)
                        # Continue the same assistant turn from the SFT answer
                        # anchor. The premature close is not in the trajectory.
                        continue
                    terminated = True

                if not terminated and injected_forced_trigger is not None:
                    injected_start = len(response_ids)
                    injected_token_count = len(await self._encode_text(injected_forced_trigger))
                    terminated = await self._append_text_tokens(
                        text=injected_forced_trigger,
                        mask_value=0,
                        current_ids=current_ids,
                        response_ids=response_ids,
                        response_mask=response_mask,
                    )
                    if response_logprobs is not None:
                        response_logprobs.extend([0.0] * (len(response_ids) - injected_start))
                    if injected_token_count:
                        if mode_tag_end_pos is None:
                            mode_tag_end_pos = len(response_ids)
                        actor_text += injected_forced_trigger
                        gcs_forced_injected_count += 1
                        gcs_forced_masked_token_count += injected_token_count
                        metrics["scenecot/gcs_forced_replaced_count"] = (
                            metrics.get("scenecot/gcs_forced_replaced_count", 0) + forced_replaced
                        )

                if terminated:
                    break

                if trigger is None:
                    if self.stop_on_answer and has_complete_answer(actor_text):
                        break
                    break

                visual_cue_count += 1
                cue_mode = VISUAL_CUE_TO_MODE.get(trigger)
                if cue_mode in cue_counts:
                    cue_counts[cue_mode] += 1
                metrics["scenecot/visual_cue_count"] = visual_cue_count
                if visual_cue_count > self.max_visual_cue_turns:
                    metrics["scenecot/max_visual_cue_turns_exceeded"] = 1
                    break

                action_text = actor_text[action_scan_start:]
                action = parse_action(action_text)
                action_scan_start = len(actor_text)
                if action is None:
                    missing_think_grd_count += 1
                    metrics["scenecot/missing_think_grd"] = metrics.get("scenecot/missing_think_grd", 0) + 1
                    break
                else:
                    has_think_grd_count += 1
                    tool_call_count += 1
                    with simple_timer("tool_calls", metrics):
                        try:
                            tool_response, _, tool_metrics = await self._execute_ground_tool(
                                tool=ground_tool,
                                instance_id=tool_instance_id,
                                query=action.query,
                                mode=action.mode,
                                topk=self.topk,
                            )
                        except Exception as exc:
                            if _is_object_token_cache_miss(exc):
                                cache_miss_count += 1
                                metrics["scenecot/cache_miss_count"] = cache_miss_count
                                logger.error("SceneCOT read-only object-token cache miss; aborting rollout: %s", exc)
                                raise
                            if not self.insert_grounding_error:
                                raise
                            tool_error = f"{type(exc).__name__}: {exc}"
                            if isinstance(exc, FileNotFoundError):
                                logger.warning("SceneCOT grounding tool execution failed: %s", tool_error)
                            else:
                                logger.exception("SceneCOT grounding tool execution failed")
                            object_token_cache_miss = _is_object_token_cache_miss(exc)
                            tool_response = ToolResponse(text=f"<grounding_error>{tool_error}</grounding_error>")
                            tool_metrics = {
                                "scenecot/tool_error": 1,
                                "scenecot/tool_call": 1,
                                "scenecot/cache_miss": 1 if object_token_cache_miss else 0,
                            }
                            if object_token_cache_miss:
                                cache_miss_count += 1
                            tool_error_count += 1
                        else:
                            if (
                                type(self) is SceneCOTFreeTagAgentLoop
                                and _coac_counterfactual_mode() == "random_tool"
                                and coac_random_tool_response_ids is None
                            ):
                                try:
                                    seed = _env_int(COAC_RANDOM_SEED_ENV, 42)
                                    coac_random_tool_mode = _random_tool_mode(
                                        action.mode,
                                        seed_parts=(
                                            seed,
                                            extra_info.get("scene_id"),
                                            extra_info.get("question_id"),
                                            kwargs.get("trajectory_info", {}).get("step")
                                            if isinstance(kwargs.get("trajectory_info"), dict)
                                            else None,
                                            kwargs.get("rollout_n"),
                                            kwargs.get("sample_index"),
                                            action.query,
                                        ),
                                    )
                                    counterfactual_tool_response = await ground_tool.render_last_result(
                                        tool_instance_id,
                                        query=action.query,
                                        mode=coac_random_tool_mode,
                                        topk=self.topk,
                                    )
                                    if counterfactual_tool_response.image:
                                        from scenecot_verl.agent.visual_cache import get_image_cache_path

                                        paths = [
                                            get_image_cache_path(image)
                                            for image in counterfactual_tool_response.image
                                        ]
                                        if len(paths) != 1 or paths[0] is None:
                                            raise RuntimeError(
                                                "Qwen3 COAC random highlight has no visual-cache image path."
                                            )
                                        coac_random_tool_image_path = paths[0]
                                    if not actor_text.endswith(action.trigger):
                                        raise RuntimeError(
                                            "Cannot locate the actual tool tag at the end of the Qwen3 COAC prefix."
                                        )
                                    counterfactual_prefix_text = (
                                        actor_text[: -len(action.trigger)]
                                        + MODE_TO_VISUAL_CUE[coac_random_tool_mode]
                                    )
                                    coac_random_tool_prefix_ids = await self._encode_text(
                                        counterfactual_prefix_text
                                    )
                                    temporary_current_ids: list[int] = []
                                    temporary_response_mask: list[int] = []
                                    coac_random_tool_response_ids = []
                                    truncated, _, coac_random_tool_payload_span = await self._append_tool_response(
                                        tool_response=counterfactual_tool_response,
                                        current_ids=temporary_current_ids,
                                        response_ids=coac_random_tool_response_ids,
                                        response_mask=temporary_response_mask,
                                        image_data=None,
                                        mm_processor_kwargs=mm_processor_kwargs,
                                    )
                                    if truncated:
                                        raise RuntimeError(
                                            "Qwen3 COAC random-tool response exceeds the response budget."
                                        )
                                except Exception as exc:
                                    coac_random_tool_prefix_ids = None
                                    coac_random_tool_response_ids = None
                                    coac_random_tool_payload_span = None
                                    coac_random_tool_image_path = None
                                    coac_random_tool_error = f"{type(exc).__name__}: {exc}"
                                    metrics["scenecot/coac_random_tool_build_error"] = 1
                                    logger.warning(
                                        "Qwen3 COAC random-tool counterfactual could not be built: %s",
                                        coac_random_tool_error,
                                    )
                    extra_fields["scenecot_actions"].append(
                        {
                            "trigger": action.trigger,
                            "mode": action.mode,
                            "query": action.query,
                        }
                    )

                _merge_metrics(metrics, tool_metrics)
                latest_tool_metrics.update(tool_metrics or {})
                tool_response_start = len(response_ids)
                terminated, image_data, payload_span = await self._append_tool_response(
                    tool_response=tool_response,
                    current_ids=current_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    image_data=image_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                tool_response_end = len(response_ids)
                if response_logprobs is not None:
                    response_logprobs.extend([0.0] * (tool_response_end - tool_response_start))
                if obs_span is None and tool_response_end > tool_response_start:
                    if payload_span is not None:
                        obs_span = (
                            tool_response_start + payload_span[0],
                            tool_response_start + payload_span[1],
                        )
                    else:
                        obs_span = None
                        metrics["scenecot/obs_payload_span_missing"] = (
                            metrics.get("scenecot/obs_payload_span_missing", 0) + 1
                        )
                    obs_context_end_pos = tool_response_end
                    obs_text_for_coac = tool_response.text or ""
                if not terminated and self.post_tool_prefill:
                    prefill_start = len(response_ids)
                    prefill_ids = await self._encode_post_tool_prefill(self.post_tool_prefill)
                    remaining_after_tool = self.response_length - len(response_mask)
                    if prefill_ids and len(prefill_ids) <= remaining_after_tool:
                        current_ids.extend(prefill_ids)
                        response_ids.extend(prefill_ids)
                        response_mask.extend([0] * len(prefill_ids))
                        actor_text += self.post_tool_prefill
                        post_tool_prefill_count += 1
                        post_tool_prefill_token_count += len(prefill_ids)
                    elif prefill_ids:
                        terminated = True
                    if response_logprobs is not None:
                        response_logprobs.extend([0.0] * (len(response_ids) - prefill_start))
                user_turns += 1
                if terminated:
                    break
        finally:
            if tool_instance_id is not None:
                await ground_tool.release(tool_instance_id)

        autoclose_finish_reason = " ".join(
            value
            for value in (
                final_generation_finish_reason,
                final_generation_native_finish_reason,
            )
            if value
        )
        autoclose_start = len(response_ids)
        (
            actor_text,
            final_answer_autoclosed_count,
            final_answer_autoclose_eligible_count,
        ) = await self._maybe_autoclose_final_answer(
            actor_text=actor_text,
            finish_reason=autoclose_finish_reason,
            current_ids=current_ids,
            response_ids=response_ids,
            response_mask=response_mask,
        )
        if response_logprobs is not None:
            response_logprobs.extend([0.0] * (len(response_ids) - autoclose_start))
            if len(response_logprobs) != len(response_ids):
                raise RuntimeError(
                    "Rollout log-prob diagnostic lost response alignment: "
                    f"log_probs={len(response_logprobs)}, response_ids={len(response_ids)}."
                )

        final_multi_modal_data = {}
        if image_data is not None:
            final_multi_modal_data["images"] = image_data

        metrics.setdefault("generate_sequences", 0.0)
        metrics.setdefault("tool_calls", 0.0)
        metrics.setdefault("compute_score", 0.0)
        metrics.setdefault("num_preempted", -1)
        metrics.setdefault("scenecot/exact_sampled_actor_token_count", 0)
        metrics.setdefault("scenecot/final_turn_max_tokens", self.max_final_tokens)
        metrics.setdefault("scenecot/final_generation_max_tokens", 0)
        metrics["scenecot/response_tokens"] = len(response_ids)
        metrics["scenecot/tool_response_tokens"] = sum(1 for value in response_mask if value == 0)
        metrics["scenecot/actor_tokens"] = sum(1 for value in response_mask if value == 1)
        metrics["scenecot/has_complete_answer"] = 1 if has_complete_answer(actor_text) else 0
        metrics["scenecot/tool_call_count"] = tool_call_count
        metrics["scenecot/tool_error_count"] = tool_error_count
        metrics["scenecot/cache_miss_count"] = cache_miss_count
        metrics["scenecot/missing_think_grd_count"] = missing_think_grd_count
        metrics["scenecot/visual_cue_count"] = visual_cue_count
        metrics["scenecot/has_answer_count"] = 1 if has_complete_answer(actor_text) else 0
        metrics["scenecot/has_think_grd_count"] = has_think_grd_count
        metrics["scenecot/cue_prob_count"] = cue_counts["prob"]
        metrics["scenecot/cue_loc_rect_count"] = cue_counts["loc_rect"]
        metrics["scenecot/cue_loc_polar_count"] = cue_counts["loc_polar"]
        metrics["scenecot/cue_highlight_count"] = cue_counts["highlight"]
        metrics["scenecot/gcs_forced_count"] = 1 if forced_trigger is not None else 0
        metrics["scenecot/gcs_forced_injected_count"] = gcs_forced_injected_count
        metrics["scenecot/gcs_forced_masked_token_count"] = gcs_forced_masked_token_count
        metrics["scenecot/gcs_coverage_fallback_count"] = gcs_coverage_fallback_count
        metrics["scenecot/post_tool_prefill_count"] = post_tool_prefill_count
        metrics["scenecot/post_tool_prefill_token_count"] = post_tool_prefill_token_count
        metrics["scenecot/protocol_transition_count"] = protocol_transition_count
        metrics["scenecot/protocol_transition_token_count"] = protocol_transition_token_count
        metrics["scenecot/protocol_fallback_answer_count"] = protocol_fallback_answer_count
        metrics["scenecot/protocol_fallback_answer_token_count"] = (
            protocol_fallback_answer_token_count
        )
        metrics["scenecot/actor_mm_placeholder_stripped_count"] = actor_mm_placeholder_stripped_count
        metrics["scenecot/final_answer_autoclosed_count"] = final_answer_autoclosed_count
        metrics["scenecot/final_answer_autoclose_eligible_count"] = final_answer_autoclose_eligible_count
        scene_metrics = {
            "tool_call_count": tool_call_count,
            "tool_error_count": tool_error_count,
            "cache_miss_count": cache_miss_count,
            "missing_think_grd_count": missing_think_grd_count,
            "visual_cue_count": visual_cue_count,
            "has_answer_count": 1 if has_complete_answer(actor_text) else 0,
            "has_think_grd_count": has_think_grd_count,
            "cue_prob_count": cue_counts["prob"],
            "cue_loc_rect_count": cue_counts["loc_rect"],
            "cue_loc_polar_count": cue_counts["loc_polar"],
            "cue_highlight_count": cue_counts["highlight"],
            "gcs_forced_mode": forced_mode,
            "gcs_forced_count": 1 if forced_trigger is not None else 0,
            "gcs_forced_injected_count": gcs_forced_injected_count,
            "gcs_forced_masked_token_count": gcs_forced_masked_token_count,
            "gcs_coverage_fallback_count": gcs_coverage_fallback_count,
            "post_tool_prefill_count": post_tool_prefill_count,
            "post_tool_prefill_token_count": post_tool_prefill_token_count,
            "protocol_transition_count": protocol_transition_count,
            "protocol_transition_token_count": protocol_transition_token_count,
            "protocol_fallback_answer_count": protocol_fallback_answer_count,
            "protocol_fallback_answer_token_count": protocol_fallback_answer_token_count,
            "actor_mm_placeholder_stripped_count": actor_mm_placeholder_stripped_count,
            "final_answer_autoclosed_count": final_answer_autoclosed_count,
            "final_answer_autoclose_eligible_count": final_answer_autoclose_eligible_count,
            "first_generation_finish_reason": first_generation_finish_reason,
            "final_generation_finish_reason": final_generation_finish_reason,
            "first_generation_native_finish_reason": first_generation_native_finish_reason,
            "final_generation_native_finish_reason": final_generation_native_finish_reason,
            **latest_tool_metrics,
        }
        extra_fields["scenecot_metrics"] = scene_metrics
        selected_action = extra_fields["scenecot_actions"][0] if extra_fields["scenecot_actions"] else {}
        extra_fields["scenecot_coac"] = {
            "obs_span": obs_span,
            "obs_context_end_pos": obs_context_end_pos,
            "mode_tag_end_pos": mode_tag_end_pos,
            "answer_tag_pos": answer_tag_pos,
            "gt_answer_text": gt_answer_text,
            "tool_called": tool_call_count > 0,
            "obs_text": obs_text_for_coac,
            "mode": selected_action.get("mode"),
            "tool_tag": selected_action.get("trigger"),
            # This marker intentionally excludes the Qwen2.5 and LLaVA
            # subclasses, even though they inherit this loop.
            "qwen3_counterfactual_supported": type(self) is SceneCOTFreeTagAgentLoop,
            "random_tool_mode": coac_random_tool_mode,
            "random_tool_prefix_ids": coac_random_tool_prefix_ids,
            "random_tool_response_ids": coac_random_tool_response_ids,
            "random_tool_payload_span": coac_random_tool_payload_span,
            "random_tool_image_path": coac_random_tool_image_path,
            "random_tool_error": coac_random_tool_error,
        }
        extra_fields["rollout_step"] = kwargs.get("trajectory_info", {}).get("step") if isinstance(kwargs.get("trajectory_info"), dict) else None
        extra_fields["rollout_n"] = kwargs.get("rollout_n")
        extra_fields["sample_index"] = kwargs.get("sample_index")
        extra_fields["response_text"] = actor_text

        rollout_dump = {
                "step": extra_fields.get("rollout_step"),
                "sample_index": kwargs.get("sample_index"),
                "rollout_n": kwargs.get("rollout_n"),
                "scene_id": extra_info.get("scene_id"),
                "question_id": extra_info.get("question_id"),
                "question_type": extra_info.get("question_type") or extra_info.get("qa_type"),
                "dataset_name": extra_info.get("dataset_name"),
                "mode": selected_action.get("mode"),
                "is_coverage": 1 if forced_mode else 0,
                "forced_mode": forced_mode,
                "coverage_fallback": 1 if gcs_coverage_fallback_count else 0,
                "tool_tag": selected_action.get("trigger"),
                "think_grd_query": selected_action.get("query"),
                "observation": obs_text_for_coac,
                "answer": actor_text,
                "gt_answer": gt_answer_text,
                "format_valid": has_complete_answer(actor_text),
                "tool_called": tool_call_count > 0,
                "obs_span": obs_span,
                "obs_context_end_pos": obs_context_end_pos,
                "mode_tag_end_pos": mode_tag_end_pos,
                "answer_tag_pos": answer_tag_pos,
                "gt_answer_text": gt_answer_text,
                "first_generation_finish_reason": first_generation_finish_reason,
                "final_generation_finish_reason": final_generation_finish_reason,
                "first_generation_native_finish_reason": first_generation_native_finish_reason,
                "final_generation_native_finish_reason": final_generation_native_finish_reason,
                "final_answer_autoclosed_count": final_answer_autoclosed_count,
                "final_answer_autoclose_eligible_count": final_answer_autoclose_eligible_count,
                "post_tool_prefill_count": post_tool_prefill_count,
                "post_tool_prefill_token_count": post_tool_prefill_token_count,
                "protocol_transition_count": protocol_transition_count,
                "protocol_transition_token_count": protocol_transition_token_count,
                "protocol_fallback_answer_count": protocol_fallback_answer_count,
                "protocol_fallback_answer_token_count": protocol_fallback_answer_token_count,
                "actor_mm_placeholder_stripped_count": actor_mm_placeholder_stripped_count,
                "tool_error_count": tool_error_count,
                "cache_miss_count": cache_miss_count,
            }
        self._augment_model_specific_metrics(
            metrics=metrics,
            scene_metrics=scene_metrics,
            rollout_dump=rollout_dump,
            state={
                "rollout_logprob_diagnostic": rollout_logprob_diagnostic,
                "protocol_answer_stage_count": int(
                    metrics.get("scenecot/protocol_answer_stage_count", 0)
                ),
                "protocol_answer_max_tokens": metrics.get(
                    "scenecot/protocol_answer_max_tokens"
                ),
                "protocol_transition_count": protocol_transition_count,
                "protocol_transition_token_count": protocol_transition_token_count,
                "protocol_fallback_answer_count": protocol_fallback_answer_count,
                "protocol_fallback_answer_token_count": protocol_fallback_answer_token_count,
                "actor_mm_placeholder_stripped_count": actor_mm_placeholder_stripped_count,
            },
        )
        _write_rollout_dump(rollout_dump)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                response_logprobs[: self.response_length] if response_logprobs is not None else None
            ),
            routed_experts=None,
            multi_modal_data=final_multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )
