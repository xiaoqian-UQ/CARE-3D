"""Pure Python parser for SceneCOT free-tag visual cue rollouts."""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_GRD_START = "<think_grd>"
THINK_GRD_END = "</think_grd>"

VISUAL_CUE_TO_MODE = {
    "<list_obj_prob>": "prob",
    "<list_obj_loc_prob>": "loc_rect",
    "<list_obj_loc_plr_prob>": "loc_polar",
    "<highlight_obj>": "highlight",
}
VISUAL_CUE_TOKENS = list(VISUAL_CUE_TO_MODE.keys())

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL) # non-greedy match of any characters; DOTALL lets . match newlines

_THINK_GRD_RE = re.compile(r"<think_grd>(.*?)</think_grd>", flags=re.DOTALL)


@dataclass
class SceneCOTAction:
    trigger: str
    mode: str
    trigger_index: int
    query: str
    text_before_trigger: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def find_first_visual_cue(text: str) -> tuple[int, str, str] | None:
    earliest: tuple[int, str, str] | None = None
    for trigger, mode in VISUAL_CUE_TO_MODE.items():
        index = text.find(trigger)
        if index < 0:
            continue
        if earliest is None or index < earliest[0]:
            earliest = (index, trigger, mode)
    return earliest


def parse_latest_think_grd(text: str) -> str | None:
    matches = [match.group(1).strip() for match in _THINK_GRD_RE.finditer(text)]
    matches = [match for match in matches if match]
    return matches[-1] if matches else None


def parse_action(text: str) -> SceneCOTAction | None:
    cue = find_first_visual_cue(text)
    if cue is None:
        return None
    trigger_index, trigger, mode = cue
    text_before_trigger = text[:trigger_index]
    query = parse_latest_think_grd(text_before_trigger)
    if query is None:
        return None
    return SceneCOTAction(
        trigger=trigger,
        mode=mode,
        trigger_index=trigger_index,
        query=query,
        text_before_trigger=text_before_trigger,
    )


def has_complete_answer(text: str) -> bool:
    return ANSWER_RE.search(text) is not None


def extract_answer(text: str) -> str | None:
    matches = [match.group(1).strip() for match in ANSWER_RE.finditer(text)]
    matches = [match for match in matches if match]
    return matches[-1] if matches else None


def strip_stop_after_first_cue(text: str) -> tuple[str, str | None]:
    cue = find_first_visual_cue(text)
    if cue is None:
        return text, None
    index, trigger, _ = cue
    end = index + len(trigger)
    return text[:end], trigger


def count_visual_cues(text: str) -> int:
    return sum(text.count(trigger) for trigger in VISUAL_CUE_TOKENS)


def _self_test() -> None:
    text = f"{THINK_GRD_START}the narrow window on the right{THINK_GRD_END}\n<highlight_obj>"
    action = parse_action(text)
    assert action is not None
    assert action.mode == "highlight"
    assert action.query == "the narrow window on the right"
    assert not has_complete_answer(text)

    text = f"{THINK_GRD_START}chair{THINK_GRD_END}<list_obj_prob>"
    action = parse_action(text)
    assert action is not None and action.mode == "prob" and action.query == "chair"

    assert parse_action("<highlight_obj>") is None

    text = (
        f"{THINK_GRD_START}old query{THINK_GRD_END}\n"
        f"{THINK_GRD_START}new query{THINK_GRD_END}\n<list_obj_loc_prob>"
    )
    action = parse_action(text)
    assert action is not None and action.query == "new query" and action.mode == "loc_rect"

    stripped, trigger = strip_stop_after_first_cue(
        f"{THINK_GRD_START}window{THINK_GRD_END}<highlight_obj> extra text"
    )
    assert trigger == "<highlight_obj>"
    assert stripped.endswith("<highlight_obj>")
    assert "extra text" not in stripped

    assert extract_answer("<answer>No</answer>") == "No"
    assert count_visual_cues("<highlight_obj><list_obj_prob>") == 2


if __name__ == "__main__":
    _self_test()
    print("SceneCOT tag parser self-test passed.")
