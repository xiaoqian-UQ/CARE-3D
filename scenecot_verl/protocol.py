from __future__ import annotations

import re
from collections import Counter


SCENECOT_TOOL_TAGS = (
    "<list_obj_prob>",
    "<list_obj_loc_prob>",
    "<list_obj_loc_plr_prob>",
    "<highlight_obj>",
)
SCENECOT_OBSERVATION_INSTRUCTION = """<scenecot_instruction>
Use the observation to answer the original question.

Output exactly:
<reasoning>
brief evidence-based rationale
</reasoning>
<answer>
short final answer
</answer>

Use only the needed evidence. Do not list every candidate. Close </reasoning> before <answer>. Do not write any answer label. Do not use markdown. Do not request another tool.
</scenecot_instruction>"""

DOT_PUNCT_CHARS = frozenset(".。…·・")
_FINAL_ASSISTANT_MARKERS = (
    "\nassistant\n",
    "\nassistant:",
    "<|im_start|>assistant\n",
)
_OBSERVATION_MARKERS = (
    "<tool_response",
    "</tool_response>",
    "<|im_start|>tool",
    "\ntool\n",
    "\nuser\n",
    "\nuser:",
)
REASONING_OPEN = "<reasoning>"
ANSWER_OPEN_RE = re.compile(r"<answer\s*>", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def is_dot_only_text(text: str) -> bool:
    stripped = "".join(str(text or "").split())
    return bool(stripped) and all(char in DOT_PUNCT_CHARS for char in stripped)


def max_repeated_punct_run(text: str) -> int:
    max_run = 0
    current = 0
    for char in str(text or ""):
        if char in DOT_PUNCT_CHARS:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _first_tool_tag_end(text: str) -> int | None:
    best_pos: int | None = None
    best_end: int | None = None
    for tag in SCENECOT_TOOL_TAGS:
        pos = text.find(tag)
        if pos < 0:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_end = pos + len(tag)
    return best_end


def _last_marker_end(text: str, markers: tuple[str, ...]) -> int | None:
    best: tuple[int, int] | None = None
    for marker in markers:
        pos = text.rfind(marker)
        if pos < 0:
            continue
        candidate = (pos, pos + len(marker))
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best is not None else None


def _has_observation_or_final_signal(text: str) -> bool:
    if any(marker in text for marker in (*_OBSERVATION_MARKERS, *_FINAL_ASSISTANT_MARKERS)):
        return True
    first_tool_end = _first_tool_tag_end(text)
    if first_tool_end is None:
        return False
    finalish = text[first_tool_end:]
    return REASONING_OPEN in finalish or ANSWER_OPEN_RE.search(finalish) is not None


def extract_final_assistant_turn(response_text: str) -> str:
    text = str(response_text or "")
    marker_end = _last_marker_end(text, _FINAL_ASSISTANT_MARKERS)
    if marker_end is not None:
        return text[marker_end:]

    first_tool_end = _first_tool_tag_end(text)
    if first_tool_end is not None:
        finalish = text[first_tool_end:]
        reasoning_pos = finalish.find(REASONING_OPEN)
        answer_match = ANSWER_OPEN_RE.search(finalish)
        positions = [pos for pos in (reasoning_pos, answer_match.start() if answer_match else -1) if pos >= 0]
        if positions:
            return finalish[min(positions):]

    return text


def max_repeated_sentence_count(text: str) -> tuple[str, int]:
    normalized_text = re.sub(r"<[^>]+>", " ", str(text or ""))
    parts = re.split(r"(?<=[.!?。！？])\s*", normalized_text)
    sentences: list[str] = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip().lower()
        if len(sentence) < 20:
            continue
        sentences.append(sentence)
    if not sentences:
        return "", 0
    sentence, count = Counter(sentences).most_common(1)[0]
    return sentence, count


def max_repeated_ngram_count(
    text: str,
    *,
    min_ngram_size: int = 5,
    max_ngram_size: int = 12,
) -> tuple[str, int]:
    normalized_text = re.sub(r"<[^>]+>", " ", str(text or "")).lower()
    tokens = TOKEN_RE.findall(normalized_text)
    if len(tokens) < min_ngram_size * 2:
        return "", 0

    best_ngram = ""
    best_count = 0
    upper_size = min(max_ngram_size, len(tokens) // 2)
    for size in range(max(1, min_ngram_size), upper_size + 1):
        index = 0
        while index + size <= len(tokens):
            block = tokens[index : index + size]
            ngram_text = " ".join(block)
            if len(ngram_text) < 10:
                index += 1
                continue
            count = 1
            cursor = index + size
            while cursor + size <= len(tokens) and tokens[cursor : cursor + size] == block:
                count += 1
                cursor += size
            if count > best_count:
                best_ngram = ngram_text
                best_count = count
            index += size * count if count > 1 else 1
    return best_ngram, best_count


def repeated_punct_penalty(
    response_text: str,
    *,
    min_run: int = 4,
    max_penalty: float = 1.0,
) -> tuple[float, int, int]:
    text = str(response_text or "")
    dot_only = int(is_dot_only_text(text))
    max_run = max_repeated_punct_run(text)
    if dot_only:
        return min(1.0, float(max_penalty)), max_run, dot_only
    if max_run < max(1, int(min_run)):
        return 0.0, max_run, dot_only
    penalty = 0.2 + 0.05 * max_run
    return min(float(max_penalty), penalty), max_run, dot_only
