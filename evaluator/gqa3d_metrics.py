from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _object_key(annotation: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    scene_id = annotation.get("scene_id")
    object_ids = _plain(annotation.get("obj_ids", annotation.get("object_ids")))
    if scene_id is None or object_ids is None:
        raise ValueError("GQA3D annotation must contain scene_id and obj_ids/object_ids")
    if not isinstance(object_ids, (list, tuple)):
        object_ids = [object_ids]
    return str(scene_id), tuple(_plain(object_id) for object_id in object_ids)


def summarize_gqa3d_case_obj(
    score_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    correct_threshold: float = 75.0,
) -> dict[str, Any]:
    """Compute Beacon3D-style per-case and strict per-object GQA3D scores."""
    if len(score_rows) != len(result_rows):
        raise ValueError(
            f"GQA3D score/result length mismatch: {len(score_rows)} != {len(result_rows)}"
        )
    if not score_rows:
        raise ValueError("cannot summarize an empty GQA3D result")

    object_scores: dict[tuple[str, tuple[Any, ...]], list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for position, (score_row, result_row) in enumerate(zip(score_rows, result_rows, strict=True)):
        if "gpt_score" not in score_row:
            raise ValueError(f"GQA3D score row {position} has no gpt_score")
        score = float(score_row["gpt_score"])
        annotation_index = int(_plain(result_row.get("index", position)))
        if annotation_index < 0 or annotation_index >= len(annotation_rows):
            raise ValueError(f"GQA3D annotation index out of range: {annotation_index}")
        annotation = annotation_rows[annotation_index]

        result_question = result_row.get("question")
        annotation_question = annotation.get("question")
        if result_question and annotation_question and str(result_question) != str(annotation_question):
            raise ValueError(
                f"GQA3D question mismatch at result row {position}, annotation {annotation_index}"
            )

        object_scores[_object_key(annotation)].append(score)
        all_scores.append(score)

    invalid_group_sizes = Counter(len(scores) for scores in object_scores.values() if len(scores) != 3)
    if invalid_group_sizes:
        raise ValueError(
            "each GQA3D object must have exactly three QA scores; "
            f"invalid group sizes: {dict(invalid_group_sizes)}"
        )

    correct_per_object = Counter(
        sum(score >= correct_threshold for score in scores) for scores in object_scores.values()
    )
    correct_objects = correct_per_object[3]
    object_count = len(object_scores)
    correct_questions = sum(score >= correct_threshold for score in all_scores)

    return {
        "Case": sum(all_scores) / len(all_scores),
        "Obj.": 100.0 * correct_objects / object_count,
        "qa_count": len(all_scores),
        "object_count": object_count,
        "correct_threshold": correct_threshold,
        "correct_question_count": correct_questions,
        "correct_question_rate": 100.0 * correct_questions / len(all_scores),
        "correct_object_count": correct_objects,
        "correct_questions_per_object": {
            str(count): correct_per_object[count] for count in range(4)
        },
    }
