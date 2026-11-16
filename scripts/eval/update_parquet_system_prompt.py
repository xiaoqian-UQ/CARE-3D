from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenecot_verl.data.build_parquet import (
    DEFAULT_SYSTEM_PROMPT,
    MODEL_PROMPT_SOURCES,
    load_system_prompt,
)


def _as_prompt_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"prompt must be a list-like chat, got {type(value).__name__}")
    return value


def _as_message_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value)
    raise ValueError(f"prompt message must be dict-like, got {type(value).__name__}")


USER_PROMPT_SOURCES = ("keep", *MODEL_PROMPT_SOURCES)


def _replace_prompt(
    prompt_value: Any,
    extra_value: Any,
    *,
    system_prompt: str,
    user_prompt_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = [_as_message_dict(message) for message in _as_prompt_list(prompt_value)]
    if not prompt:
        raise ValueError("prompt is empty")
    if prompt[0].get("role") != "system":
        raise ValueError("prompt[0] must be the system message")
    prompt[0]["content"] = system_prompt

    extra = extra_value if isinstance(extra_value, dict) else dict(extra_value)
    extra = dict(extra)
    if user_prompt_source != "keep":
        user_prompt = extra.get(user_prompt_source)
        if user_prompt is None or not str(user_prompt).strip():
            raise ValueError(f"extra_info.{user_prompt_source} is missing or empty")
        if len(prompt) < 2 or prompt[1].get("role") != "user":
            raise ValueError("prompt[1] must be the user message")
        prompt[1]["content"] = str(user_prompt).strip()
        extra["model_prompt_source"] = user_prompt_source

    extra.pop("expected_first_visual_cue", None)
    extra.pop("cue_weight", None)

    return prompt, extra


def _update_one(
    path: Path,
    output_path: Path,
    *,
    system_prompt: str,
    user_prompt_source: str,
) -> None:
    dataframe = pd.read_parquet(path)
    if "prompt" not in dataframe.columns:
        raise ValueError(f"{path} has no prompt column")
    if "extra_info" not in dataframe.columns:
        raise ValueError(f"{path} has no extra_info column")

    updated = dataframe.apply(
        lambda row: _replace_prompt(
            row["prompt"],
            row["extra_info"],
            system_prompt=system_prompt,
            user_prompt_source=user_prompt_source,
        ),
        axis=1,
    )
    dataframe["prompt"] = updated.map(lambda item: item[0])
    dataframe["extra_info"] = updated.map(lambda item: item[1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(f"Wrote {len(dataframe)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Update SceneCOT parquet chat prompts.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional prompt file. Defaults to build_parquet.DEFAULT_SYSTEM_PROMPT.",
    )
    parser.add_argument(
        "--user-prompt-source",
        choices=USER_PROMPT_SOURCES,
        default="keep",
        help=(
            "Set prompt[1].content from this extra_info field. "
            "Use question for question-only policy prompts."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Only valid with one input unless --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input parquet after writing a temporary sibling file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None and (args.in_place or len(args.inputs) != 1):
        raise ValueError("--output can only be used with exactly one input and without --in-place")
    if args.output is None and not args.in_place:
        raise ValueError("Use --in-place or provide --output for a single input")

    system_prompt = (
        load_system_prompt(args.system_prompt_file)
        if args.system_prompt_file is not None
        else DEFAULT_SYSTEM_PROMPT
    )

    for input_path in args.inputs:
        path = input_path.expanduser()
        output_path = path if args.in_place else args.output.expanduser()
        _update_one(
            path,
            output_path,
            system_prompt=system_prompt,
            user_prompt_source=args.user_prompt_source,
        )


if __name__ == "__main__":
    main()
