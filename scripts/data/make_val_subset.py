#!/usr/bin/env python
"""Select the fixed GRPO validation subset (20 MSQA val questions per type) by ID.

The subset is stored in assets/splits/val_msqa_20_per_type.json so that every run
monitors exactly the same 180 questions, in the same order, as the paper runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _key(extra) -> tuple[str, str, int]:
    return str(extra["scene_id"]), str(extra["question_id"]), int(extra["index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-parquet", type=Path, required=True, help="MSQA val.parquet from build_parquet.py")
    parser.add_argument("--ids", type=Path, default=Path("assets/splits/val_msqa_20_per_type.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    val = pd.read_parquet(args.val_parquet)
    position = {_key(extra): i for i, extra in enumerate(val["extra_info"])}
    if len(position) != len(val):
        raise ValueError(f"(scene_id, question_id, index) is not unique in {args.val_parquet}")

    wanted = json.loads(args.ids.read_text(encoding="utf-8"))["rows"]
    missing = [row for row in wanted if _key(row) not in position]
    if missing:
        raise ValueError(f"{len(missing)} subset rows are missing from {args.val_parquet}, e.g. {missing[0]}")

    subset = val.iloc[[position[_key(row)] for row in wanted]].reset_index(drop=True)

    def _annotate(extra):
        # Same bookkeeping fields as the original (mixed-parquet) subset; logged only.
        extra = dict(extra)
        extra["mix_source_dataset"] = str(extra.get("dataset_name") or "msqa").lower()
        extra["mix_source_split"] = "val"
        extra["mix_original_index"] = extra.get("index")
        return extra

    subset["extra_info"] = subset["extra_info"].map(_annotate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(args.output, index=False)
    print(f"Wrote {len(subset)} rows to {args.output}")


if __name__ == "__main__":
    main()
