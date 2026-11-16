#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from scenecot_verl.protocol import extract_final_assistant_turn

ANSWER_RE = re.compile(r"<answer\s*>(.*?)(?:</answer>|$)", flags=re.DOTALL | re.IGNORECASE)
ANSWER_PLACEHOLDERS = {"...", "…"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _is_placeholder_answer(value: Any) -> bool:
    return str(value or "").strip() in ANSWER_PLACEHOLDERS


def _clean_answer(value: str) -> str:
    answer = str(value or "").strip()
    for stop_token in ("<|im_end|>", "<|endoftext|>"):
        if stop_token in answer:
            answer = answer.split(stop_token, 1)[0].strip()
    return answer


def _extract_answer_from_output(value: Any) -> str | None:
    text = extract_final_assistant_turn(str(value or ""))
    matches = [_clean_answer(match.group(1)) for match in ANSWER_RE.finditer(text)]
    matches = [match for match in matches if match]
    for match in reversed(matches):
        if not _is_placeholder_answer(match):
            return match
    return matches[-1] if matches else None


def convert(
    verl_jsonl: Path,
    val_parquet: Path,
    output_root: Path,
    model_name: str,
    dataset_name: str,
) -> Path:
    generations = _load_jsonl(verl_jsonl)
    val_df = pd.read_parquet(val_parquet)

    if len(generations) > len(val_df):
        raise ValueError(
            f"{verl_jsonl} has {len(generations)} rows, but {val_parquet} has only {len(val_df)} rows"
        )

    results: list[dict[str, Any]] = []
    for position, gen in enumerate(generations):
        val_row = val_df.iloc[position]
        extra_info = val_row.get("extra_info") or {}
        reward_model = val_row.get("reward_model") or {}

        gt_answers = _to_list(extra_info.get("gt_answer"))
        if not gt_answers:
            gt_answers = _to_list(reward_model.get("ground_truth"))
        if not gt_answers:
            gt_answers = _to_list(gen.get("gts"))

        response_pred = _extract_answer_from_output(gen.get("output"))
        if response_pred is None or _is_placeholder_answer(response_pred):
            response_pred = gen.get("pred_answer")
        if response_pred is None or _is_placeholder_answer(response_pred):
            response_pred = gen.get("answer")
        if response_pred is None:
            response_pred = gen.get("output", "")

        results.append(
            {
                "index": int(extra_info.get("index", position)),
                "question": str(extra_info.get("question", "")),
                "question_type": str(extra_info.get("question_type", "unknown")),
                "scene_id": extra_info.get("scene_id"),
                "question_id": extra_info.get("question_id"),
                "response_pred": str(response_pred),
                "response_gt": [str(answer) for answer in gt_answers],
                "verl_score": gen.get("score"),
                "verl_reward": gen.get("reward"),
                "verl_answer_judge_score_1_to_5": gen.get("answer_judge_score_1_to_5"),
                "verl_answer_judge_label": gen.get("answer_judge_label"),
            }
        )

    output_dir = output_root / model_name / "eval_results" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VERL validation JSONL generations to the legacy MSQA evaluator results.json format."
    )
    parser.add_argument("--verl-jsonl", required=True, type=Path)
    parser.add_argument("--val-parquet", default=Path("data_assets/verl_parquet_msqa/val.parquet"), type=Path)
    parser.add_argument("--output-root", default=Path("outputs/eval_legacy"), type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", default="QACOTScanNetMSR3D")
    args = parser.parse_args()

    output_path = convert(
        verl_jsonl=args.verl_jsonl,
        val_parquet=args.val_parquet,
        output_root=args.output_root,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
