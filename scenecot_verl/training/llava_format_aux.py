from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


LLAVA_FORMAT_AUX_ENABLE_ENV = "SCENECOT_LLAVA_FORMAT_AUX_ENABLE"
LLAVA_FORMAT_AUX_COEF_ENV = "SCENECOT_LLAVA_FORMAT_AUX_COEF"
LLAVA_FORMAT_AUX_REASONING_MAX_TOKENS_ENV = "SCENECOT_LLAVA_FORMAT_AUX_REASONING_MAX_TOKENS"
LLAVA_FORMAT_AUX_ANSWER_MAX_TOKENS_ENV = "SCENECOT_LLAVA_FORMAT_AUX_ANSWER_MAX_TOKENS"
LLAVA_FORMAT_TAGS = (
    "<reasoning>",
    "</reasoning>",
    "<answer>",
    "</answer>",
)


def is_llava_processor(processor: Any) -> bool:
    """Return whether the active multimodal processor is a Hugging Face LLaVA processor."""
    return processor is not None and "llava" in type(processor).__name__.lower()


def resolve_llava_format_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve and validate the four single-piece SceneCOT format tokens."""
    token_ids: dict[str, int] = {}
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for tag in LLAVA_FORMAT_TAGS:
        token_id = tokenizer.convert_tokens_to_ids(tag)
        if token_id is None or (unknown_id is not None and int(token_id) == int(unknown_id)):
            raise ValueError(f"LLaVA SceneCOT format tag {tag!r} is not in the tokenizer vocabulary.")
        encoded = list(tokenizer.encode(tag, add_special_tokens=False))
        if encoded.count(int(token_id)) != 1:
            raise ValueError(
                f"LLaVA SceneCOT format tag {tag!r} encoded as {encoded}; "
                f"expected exactly one occurrence of token ID {token_id}."
            )
        token_ids[tag] = int(token_id)
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError(f"LLaVA SceneCOT format tags do not map to unique token IDs: {token_ids}")
    return token_ids


def build_llava_canonical_format_target(
    *,
    response_valid: Sequence[int],
    actor_mask_valid: Sequence[int | bool],
    obs_context_end_pos: int | None,
    response_width: int,
    format_token_ids: Mapping[str, int],
    fallback_body_ids: Sequence[int],
    ignored_token_ids: set[int] | None = None,
    reasoning_max_tokens: int = 48,
    answer_max_tokens: int = 16,
) -> tuple[list[int], list[bool]] | None:
    """Repair one rollout into a teacher-forced canonical final turn.

    The prompt, first actor turn, and tool observation are retained verbatim.
    The sampled final turn supplies self-generated body tokens, while the four
    XML format tokens are rebuilt in canonical order. Only those four rebuilt
    tokens are marked as auxiliary targets; body tokens never become SFT
    labels. This creates a target even when the rollout omitted one or more
    tags and therefore avoids reinforcing only already-successful samples.
    """
    response_valid = [int(token_id) for token_id in response_valid]
    actor_mask_valid = [bool(value) for value in actor_mask_valid]
    if len(response_valid) != len(actor_mask_valid):
        raise ValueError(
            "response_valid and actor_mask_valid must have identical lengths; "
            f"got {len(response_valid)} and {len(actor_mask_valid)}."
        )
    if obs_context_end_pos is None:
        return None
    context_end = int(obs_context_end_pos)
    response_width = int(response_width)
    if context_end < 0 or context_end > len(response_valid) or response_width < context_end + len(LLAVA_FORMAT_TAGS):
        return None

    ordered_ids = [int(format_token_ids[tag]) for tag in LLAVA_FORMAT_TAGS]
    format_id_set = set(ordered_ids)
    ignored = set(ignored_token_ids or ())
    ignored.update(format_id_set)

    final_actor_ids = [
        response_valid[position]
        for position in range(context_end, len(response_valid))
        if actor_mask_valid[position]
    ]

    def _find(token_id: int, start: int = 0) -> int:
        try:
            return final_actor_ids.index(token_id, max(0, int(start)))
        except ValueError:
            return -1

    def _clean(values: Sequence[int]) -> list[int]:
        return [int(token_id) for token_id in values if int(token_id) not in ignored]

    reasoning_open_pos = _find(ordered_ids[0])
    reasoning_close_pos = _find(ordered_ids[1], reasoning_open_pos + 1 if reasoning_open_pos >= 0 else 0)
    answer_open_search_start = reasoning_close_pos + 1 if reasoning_close_pos >= 0 else 0
    answer_open_pos = _find(ordered_ids[2], answer_open_search_start)
    if answer_open_pos < 0:
        answer_open_pos = _find(ordered_ids[2])
    answer_close_pos = _find(ordered_ids[3], answer_open_pos + 1 if answer_open_pos >= 0 else 0)

    if reasoning_open_pos >= 0:
        reasoning_start = reasoning_open_pos + 1
        reasoning_end_candidates = [
            position
            for position in (reasoning_close_pos, answer_open_pos, answer_close_pos)
            if position >= reasoning_start
        ]
        reasoning_end = min(reasoning_end_candidates) if reasoning_end_candidates else len(final_actor_ids)
        reasoning_body = _clean(final_actor_ids[reasoning_start:reasoning_end])
    else:
        reasoning_end = answer_open_pos if answer_open_pos >= 0 else len(final_actor_ids)
        reasoning_body = _clean(final_actor_ids[:reasoning_end])

    if answer_open_pos >= 0:
        answer_start = answer_open_pos + 1
        answer_end = answer_close_pos if answer_close_pos >= answer_start else len(final_actor_ids)
        answer_body = _clean(final_actor_ids[answer_start:answer_end])
    else:
        answer_body = []

    all_body_tokens = _clean(final_actor_ids)
    fallback = _clean(fallback_body_ids)
    if not fallback:
        fallback = all_body_tokens[:1]
    if not fallback:
        return None

    reasoning_max_tokens = max(1, int(reasoning_max_tokens))
    answer_max_tokens = max(1, int(answer_max_tokens))
    if not reasoning_body:
        reasoning_body = all_body_tokens or fallback
    if not answer_body:
        # A short self-generated suffix provides a natural context for
        # ``</answer>`` without supervising answer content or using GT labels.
        answer_body = (all_body_tokens[-answer_max_tokens:] if all_body_tokens else fallback)
    reasoning_body = reasoning_body[:reasoning_max_tokens]
    answer_body = answer_body[:answer_max_tokens]

    available_body_tokens = response_width - context_end - len(ordered_ids)
    while len(reasoning_body) + len(answer_body) > available_body_tokens:
        if len(reasoning_body) > 1 and len(reasoning_body) >= len(answer_body):
            reasoning_body.pop()
        elif len(answer_body) > 1:
            answer_body.pop()
        elif reasoning_body:
            reasoning_body.pop()
        elif answer_body:
            answer_body.pop()
        else:
            break

    prefix = response_valid[:context_end]
    canonical = list(prefix)
    target_mask = [False] * len(prefix)
    for tag_index, body in (
        (0, reasoning_body),
        (1, ()),
        (2, answer_body),
        (3, ()),
    ):
        canonical.append(ordered_ids[tag_index])
        target_mask.append(True)
        canonical.extend(body)
        target_mask.extend([False] * len(body))

    if len(canonical) > response_width:
        raise AssertionError(f"canonical LLaVA format target exceeds response width: {len(canonical)} > {response_width}")
    if sum(target_mask) != len(LLAVA_FORMAT_TAGS):
        raise AssertionError(f"canonical LLaVA format target must contain four labels, got {sum(target_mask)}")
    return canonical, target_mask


def build_llava_format_aux_mask(
    *,
    responses: torch.Tensor,
    actor_response_mask: torch.Tensor,
    valid_response_mask: torch.Tensor,
    obs_context_end_positions: Sequence[int | None],
    autoclose_counts: Sequence[int],
    format_token_ids: Mapping[str, int],
) -> torch.Tensor:
    """Build targets for ordered final-turn format tokens already present in a rollout.

    Actor-sampled tags are eligible only after the tool observation and only
    while they form the unique prefix
    ``<reasoning></reasoning><answer></answer>``. A parser-inserted final
    ``</answer>`` is also eligible when autoclose metadata confirms its origin.
    Tool/instruction tokens remain excluded even though their response mask is
    zero and they may contain literal format examples.
    """
    if responses.ndim != 2:
        raise ValueError(f"responses must be rank 2, got shape {tuple(responses.shape)}")
    if actor_response_mask.shape != responses.shape or valid_response_mask.shape != responses.shape:
        raise ValueError(
            "responses, actor_response_mask, and valid_response_mask must have identical shapes; "
            f"got {tuple(responses.shape)}, {tuple(actor_response_mask.shape)}, "
            f"{tuple(valid_response_mask.shape)}"
        )
    batch_size = int(responses.shape[0])
    if len(obs_context_end_positions) != batch_size or len(autoclose_counts) != batch_size:
        raise ValueError(
            "SceneCOT format auxiliary metadata must have one value per response row; "
            f"got batch={batch_size}, obs={len(obs_context_end_positions)}, "
            f"autoclose={len(autoclose_counts)}"
        )

    ordered_ids = [int(format_token_ids[tag]) for tag in LLAVA_FORMAT_TAGS]
    result = torch.zeros_like(responses, dtype=torch.bool)
    actor_mask = actor_response_mask.to(torch.bool)
    valid_mask = valid_response_mask.to(torch.bool)

    for row in range(batch_size):
        context_end = obs_context_end_positions[row]
        if context_end is None:
            continue
        context_end = int(context_end)
        if context_end < 0 or context_end > int(responses.shape[1]):
            continue

        row_ids = responses[row]
        eligible_actor = actor_mask[row] & valid_mask[row]
        positions_by_id: dict[int, list[int]] = {}
        for token_id in ordered_ids:
            positions = torch.nonzero(
                eligible_actor & (row_ids == token_id),
                as_tuple=False,
            ).flatten()
            positions_by_id[token_id] = [int(pos) for pos in positions.tolist() if int(pos) >= context_end]

        if int(autoclose_counts[row] or 0) > 0:
            close_id = ordered_ids[-1]
            injected_close_positions = torch.nonzero(
                valid_mask[row] & ~actor_mask[row] & (row_ids == close_id),
                as_tuple=False,
            ).flatten()
            injected_after_context = [
                int(pos) for pos in injected_close_positions.tolist() if int(pos) >= context_end
            ]
            if injected_after_context:
                # Autoclose is appended after actor generation. Selecting the
                # last masked close excludes any literal tag in tool text.
                positions_by_id[close_id].append(injected_after_context[-1])

        previous_position = context_end - 1
        for token_id in ordered_ids:
            positions = sorted(set(positions_by_id[token_id]))
            # Do not reinforce duplicates or a tag that breaks the expected
            # ordered prefix. Later tags are unsafe once the prefix breaks.
            if len(positions) != 1 or positions[0] <= previous_position:
                break
            previous_position = positions[0]
            result[row, previous_position] = True

    return result
