from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from scenecot_verl.agent.visual_cache import SCENECOT_IMAGE_PATH_INFO_KEY

if TYPE_CHECKING:
    from .grounding_backend import GroundCandidate, GroundResult


def _display_name(candidate: GroundCandidate) -> str:
    return candidate.label or candidate.obj_id


def _fmt_float(value: float) -> str:
    return f"{float(value):.2f}"


def format_obj_prob(result: GroundResult, topk: int) -> str:
    items = result.candidates[:topk]
    body = " ".join(f"{_display_name(candidate)} {_fmt_float(candidate.prob)}" for candidate in items)
    return f"<obj_prob> {body} </obj_prob>"


def format_obj_loc_prob(result: GroundResult, topk: int) -> str:
    parts: list[str] = []
    for candidate in result.candidates[:topk]:
        loc = candidate.rect_loc
        loc_text = ",".join(_fmt_float(x) for x in loc) if loc is not None else "unknown"
        parts.append(f"{_display_name(candidate)}: {loc_text}; prob: {_fmt_float(candidate.prob)}")
    return f"<obj_loc_prob> {' '.join(parts)} </obj_loc_prob>"


def format_obj_loc_plr_prob(result: GroundResult, topk: int) -> str:
    parts: list[str] = []
    for candidate in result.candidates[:topk]:
        loc = candidate.polar_loc
        if loc is None or len(loc) < 2:
            loc_text = "unknown,unknown"
        else:
            loc_text = ",".join(_fmt_float(x) for x in loc[:2])
        parts.append(f"{_display_name(candidate)}: {loc_text}; prob: {_fmt_float(candidate.prob)}")
    return f"<obj_loc_plr_prob> {' '.join(parts)} </obj_loc_plr_prob>"


def get_highlight_image(result: GroundResult) -> Image.Image | str:
    if not result.candidates:
        raise ValueError("No candidates available for highlight mode.")
    top1 = result.candidates[0]
    if not top1.image_path:
        raise FileNotFoundError(
            f"No object image path for top1 object {top1.obj_id} in scene {result.scene_id}."
        )
    path = Path(top1.image_path)
    if not path.exists():
        raise FileNotFoundError(f"Top1 object image does not exist: {path}")
    image = Image.open(path).convert("RGB")
    image.info[SCENECOT_IMAGE_PATH_INFO_KEY] = str(path)
    return image
