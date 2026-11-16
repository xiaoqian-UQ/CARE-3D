"""
python scenecot_verl/data/build_parquet.py \
--train-input data_assets/GQA3D \
--test-input data_assets/GQA3D \
--val-input data_assets/GQA3D \
--output-dir data_assets/verl_parquet_gqa3d \
--dataset-name gqa3d
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


MODEL_PROMPT_SOURCES = ("grounding_prompt", "question")


DEFAULT_SYSTEM_PROMPT = """You are a precise 3D scene assistant.

You must obtain one observation before answering.

First assistant turn must contain exactly:
<think_grd>
short evidence query
</think_grd>
one tool from Tool list

The query line must be a short noun phrase from the current question, usually 2-8 words.
Use a concrete object, relation, viewpoint cue, or evidence phrase.
Do not answer, reason, or explain tool choice in the first turn.
The fourth line must be exactly one tool tag from Tool list, with no label, colon, or extra text.

Tool list:
<list_obj_prob>
Returns ranked object/category evidence with confidence scores.

<list_obj_loc_prob>
Returns ranked object evidence with scene-coordinate 3D centers, sizes, and confidence scores.

<list_obj_loc_plr_prob>
Returns ranked object evidence with viewer-relative angle_deg, distance, and confidence scores.

<highlight_obj>
Returns visual/crop evidence for one queried object.

Numeric interpretation:
- prob/confidence: higher means more likely to match the query.
- <list_obj_loc_prob>: [center_x, center_y, center_z, size_x, size_y, size_z]. Centers are scene-coordinate positions. Sizes are extents along each axis.
- <list_obj_loc_plr_prob>: object_name: angle_deg,distance; prob: confidence. Example: crate: 49.54,1.50; prob: 0.75. The first number is angle_deg, the second number is distance. angle_deg is relative to viewer forward: 0 front, positive left, negative right, +/-180 behind. clock hour = round(((-angle_deg) mod 360) / 30), with 0 mapped to 12.

Choose one tool from Tool list according to what returned evidence would make the answer best supported.

first-turn example:
Question: How many chairs are at my 5 o'clock?
<think_grd>
chairs at my 5 o'clock
</think_grd>
<list_obj_loc_plr_prob>

After receiving the observation, answer exactly:
<reasoning>
brief evidence-based rationale
</reasoning>
<answer>
short final answer
</answer>

Final-answer rules:
- Use the observation and the original question.
- In <reasoning>, write a brief evidence-based rationale, usually one to three short sentences.
- Mention only evidence needed for the answer; do not list every candidate.
- Close </reasoning> before <answer>.
- In <answer>, write only the short final answer.
- Do not request another tool.
- Do not use markdown, bullets, headings, bold text, or code.
- Do not repeat any sentence."""


def _candidate_filename(split: str, dataset_name: str) -> str:
    dataset = dataset_name.lower()
    if dataset == "msqa":
        return f"situated_qa_{split}_pure_txt.json"
    return f"gqa3d_{split}.json"


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_python_scalar(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python_scalar(v) for v in value]
    if is_missing(value):
        return None
    return value


def build_grounding_prompt(situation: Optional[str], question: str) -> str:
    question_text = "" if is_missing(question) else str(question).strip()
    if is_missing(situation):
        prompt = question_text
    else:
        situation_text = str(situation).strip()
        prompt = f"{situation_text} {question_text}" if situation_text else question_text
    return re.sub(r"\s+", " ", prompt).strip()


def _stringify_scalar(value: Any) -> str:   # convert a scalar value to a string
    value = to_python_scalar(value)
    if is_missing(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int):
        return str(value)
    raise ValueError(f"Expected a scalar value, got {type(value).__name__}: {value!r}")


def normalize_gt_answer(value: Any) -> list[str]:
    if is_missing(value):
        raise ValueError("gt_answer is required for every split and cannot be empty.")
    if not isinstance(value, list):
        raise ValueError(f"gt_answer must be a list, got {type(value).__name__}.")

    answers = [_stringify_scalar(item) for item in value]
    answers = [item for item in answers if item]
    if not answers:
        raise ValueError("gt_answer list is empty after cleaning.")
    return answers


def normalize_gt_objects(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if not isinstance(value, list):
        raise ValueError(f"gt_objects must be a list, got {type(value).__name__}.")

    objects = [_stringify_scalar(item) for item in value]
    objects = [item for item in objects if item]
    return objects


def build_agent_pos_ori(example: dict[str, Any]) -> list[list[float]] | None:
    position = example.get("position")
    orientation_angle = example.get("orientation_angle")
    if is_missing(position) or is_missing(orientation_angle):
        return None
    if not isinstance(position, list):
        raise ValueError(f"position must be a list, got {type(position).__name__}.")
    pos = [float(value) for value in position]
    angle = float(orientation_angle)
    return [pos, [math.cos(angle), math.sin(angle), 0.0]]


def load_system_prompt(path: Optional[Path]) -> str:
    if path is None:
        return DEFAULT_SYSTEM_PROMPT
    prompt_path = path.expanduser()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt file does not exist: {prompt_path}")
    content = prompt_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"System prompt file is empty: {prompt_path}")
    return content


def build_prompt(question: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def model_prompt_text(question: str, grounding_prompt: str, source: str) -> str:
    if source == "question":
        return question
    if source == "grounding_prompt":
        return grounding_prompt
    raise ValueError(f"Unsupported model prompt source: {source}")


def normalize_optional_text(value: Any) -> Optional[str]:
    value = to_python_scalar(value)
    if is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_required_text(value: Any, field_name: str, split: str, idx: int) -> str:
    value = to_python_scalar(value)
    if is_missing(value):
        raise ValueError(f"{split}[{idx}] is missing required field: {field_name}")
    text = _stringify_scalar(value)
    if not text:
        raise ValueError(f"{split}[{idx}] has empty required field: {field_name}")
    return text


def build_row(
    example: dict[str, Any],
    idx: int,
    split: str,
    args: argparse.Namespace,
    system_prompt: str,
) -> dict[str, Any]:
    scene_id = _normalize_required_text(example.get("scene_id"), "scene_id", split, idx)
    question = _normalize_required_text(example.get("question"), "question", split, idx)
    situation = normalize_optional_text(example.get("situation"))
    grounding_prompt = build_grounding_prompt(situation, question)
    if not grounding_prompt:
        raise ValueError(f"{split}[{idx}] grounding_prompt is empty after cleaning.")

    question_id_raw = example.get("index")
    question_id = f"{split}_{idx}" if is_missing(question_id_raw) else _stringify_scalar(question_id_raw)
    if not question_id:
        question_id = f"{split}_{idx}"

    try:
        gt_answer = normalize_gt_answer(example.get("answers"))
    except ValueError as exc:
        raise ValueError(f"{split}[{idx}] invalid gt_answer: {exc}") from exc

    try:
        gt_objects = normalize_gt_objects(example.get("obj_ids"))
    except ValueError as exc:
        raise ValueError(f"{split}[{idx}] invalid gt_objects: {exc}") from exc

    question_type_raw = example.get("type")
    question_type = None if is_missing(question_type_raw) else _stringify_scalar(question_type_raw)
    agent_pos_ori = build_agent_pos_ori(example)
    requires_grounding = len(gt_objects) > 0
    gt_answer_with_cot_raw = example.get("answer_with_cot")
    gt_answer_with_cot = None if is_missing(gt_answer_with_cot_raw) else str(to_python_scalar(gt_answer_with_cot_raw))

    prompt_source = str(args.model_prompt_source)
    model_prompt = model_prompt_text(question, grounding_prompt, prompt_source)
    prompt = build_prompt(question=model_prompt, system_prompt=system_prompt)

    extra_info: dict[str, Any] = {
        "split": split,
        "index": idx,
        "dataset_name": args.dataset_name,
        "scene_id": scene_id,
        "question_id": question_id,
        "question": question,
        "situation": situation,
        "grounding_prompt": grounding_prompt,
        "model_prompt_source": prompt_source,
        "agent_pos_ori": agent_pos_ori,
        "gt_answer": gt_answer,
        "gt_objects": gt_objects,
        "question_type": question_type,
        "requires_grounding": requires_grounding,
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "scenecot_ground": {
                "create_kwargs": {
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "gt_objects": gt_objects,
                    "dataset_name": args.dataset_name,
                    "grounding_prompt": grounding_prompt,
                    "agent_pos_ori": agent_pos_ori,
                }
            }
        },
    }

    if args.include_gt_cot and not args.exclude_gt_cot:
        extra_info["gt_answer_with_cot"] = gt_answer_with_cot

    return {
        "data_source": args.data_source,
        "agent_name": args.agent_name,
        "prompt": prompt,
        "ability": args.ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": gt_answer,
        },
        "extra_info": extra_info,
    }


def _ensure_dict_records(records: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of dict records.")

    cleaned: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{path} record {idx} must be a dict, got {type(record).__name__}.")
        cleaned.append(dict(record))
    return cleaned


def _read_json(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file {path}: {exc}") from exc

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    elif isinstance(payload, dict):
        raise ValueError(
            f"Unsupported JSON structure in {path}. Expected list[dict] or "
            '{"data": list[dict]}.'
        )
    else:
        raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")

    return _ensure_dict_records(records, path)


def read_any_format(path: Path, split: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"Unsupported input file type for {split}: {path}")


def build_split(
    input_path: Path,
    split: str,
    args: argparse.Namespace,
    system_prompt: str,
) -> list[dict[str, Any]]:
    examples = read_any_format(input_path, split)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    rows = [build_row(example, idx, split, args, system_prompt) for idx, example in enumerate(examples)]
    validate_rows(rows, split=split)
    print_stats(rows, split=split)
    return rows


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import pandas as pd  # type: ignore

        dataframe = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise RuntimeError("Please install pyarrow or fastparquet to write parquet files.") from exc
    except ValueError as exc:
        if "Unable to find a usable engine" in str(exc) or "pyarrow" in str(exc) or "fastparquet" in str(exc):
            raise RuntimeError("Please install pyarrow or fastparquet to write parquet files.") from exc
        raise


def _gt_answer_is_missing(value: Any) -> bool:
    if isinstance(value, list):
        return not [item for item in value if not is_missing(item)]
    return is_missing(value)


def print_stats(rows: list[dict[str, Any]], split: str) -> None:
    missing_scene_id = 0
    missing_question = 0
    missing_situation = 0
    with_situation = 0
    missing_gt_answer = 0
    missing_gt_objects = 0
    requires_grounding = 0
    question_types: Counter[str] = Counter()

    for row in rows:
        extra = row.get("extra_info") or {}
        if is_missing(extra.get("scene_id")):
            missing_scene_id += 1
        if is_missing(extra.get("question")):
            missing_question += 1
        if is_missing(extra.get("situation")):
            missing_situation += 1
        else:
            with_situation += 1
        if is_missing(extra.get("grounding_prompt")):
            raise ValueError(f"{split} row has empty grounding_prompt.")
        if _gt_answer_is_missing(extra.get("gt_answer")):
            missing_gt_answer += 1
        if not extra.get("gt_objects"):
            missing_gt_objects += 1
        if bool(extra.get("requires_grounding")):
            requires_grounding += 1
        question_type = extra.get("question_type")
        question_types["<None>" if is_missing(question_type) else str(question_type)] += 1

    print(f"\n=== {split} ===")
    print(f"num rows: {len(rows)}")
    print(f"missing scene_id count: {missing_scene_id}")
    print(f"missing question count: {missing_question}")
    print(f"missing situation count: {missing_situation}")
    print(f"with situation count: {with_situation}")
    print(f"missing gt_answer count: {missing_gt_answer}")
    print(f"missing gt_objects count: {missing_gt_objects}")
    print(f"requires_grounding count: {requires_grounding}")
    first_grounding_prompt = (rows[0].get("extra_info") or {}).get("grounding_prompt") if rows else None
    print(f"first grounding_prompt: {first_grounding_prompt}")
    print("question_type distribution:")
    for question_type, count in question_types.most_common():
        print(f"  {question_type}: {count}")
    print("first row preview as pretty JSON:")
    preview = rows[0] if rows else None
    print(json.dumps(preview, ensure_ascii=False, indent=2, default=str))


def _validate_prompt_shape(row: dict[str, Any], split: str, idx: int) -> None:
    prompt = row.get("prompt")
    extra = row.get("extra_info") or {}
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError(f"{split}[{idx}] prompt must be a two-message raw chat list.")
    if prompt[0].get("role") != "system" or prompt[1].get("role") != "user":
        raise ValueError(f"{split}[{idx}] prompt roles must be system then user.")

    source = str(extra.get("model_prompt_source") or "grounding_prompt")
    if source not in MODEL_PROMPT_SOURCES:
        raise ValueError(f"{split}[{idx}] unsupported model_prompt_source: {source}")
    expected_user = str(extra.get(source))
    if prompt[1].get("content") != expected_user:
        raise ValueError(
            f"{split}[{idx}] user prompt must contain only {source}. "
            "Do not add gt_answer, gt_objects, gt_answer_with_cot, images, or tool outputs."
        )


def _validate_no_exact_cot_leak(row: dict[str, Any], split: str, idx: int) -> None:
    extra = row.get("extra_info") or {}
    gt_answer_with_cot = extra.get("gt_answer_with_cot")
    if is_missing(gt_answer_with_cot):
        return
    prompt_text = "\n".join(str(message.get("content", "")) for message in row.get("prompt", []))
    gt_cot_text = str(gt_answer_with_cot).strip()
    if len(gt_cot_text) >= 16 and gt_cot_text in prompt_text:
        raise ValueError(f"{split}[{idx}] prompt contains gt_answer_with_cot, which is forbidden.")


def _validate_no_object_id_leak(row: dict[str, Any], split: str, idx: int) -> None:
    extra = row.get("extra_info") or {}
    user_prompt = row["prompt"][1]["content"]
    source = str(extra.get("model_prompt_source") or "grounding_prompt")
    if source not in MODEL_PROMPT_SOURCES:
        raise ValueError(f"{split}[{idx}] unsupported model_prompt_source: {source}")
    expected_prompt = str(extra.get(source, ""))
    if user_prompt != expected_prompt:
        raise ValueError(f"{split}[{idx}] user prompt was not built from the allowed template.")


def validate_rows(rows: list[dict[str, Any]], split: str) -> None:
    if not rows:
        raise ValueError(f"{split} split produced 0 rows.")

    missing_scene_id = 0
    missing_question = 0
    missing_gt_answer = 0
    missing_gt_objects = 0

    for idx, row in enumerate(rows):
        for key in ["data_source", "agent_name", "prompt", "ability", "reward_model", "extra_info"]:
            if key not in row:
                raise ValueError(f"{split}[{idx}] missing top-level key: {key}")

        reward_model = row.get("reward_model")
        if not isinstance(reward_model, Mapping):
            raise ValueError(f"{split}[{idx}] reward_model must be a dict.")
        if reward_model.get("style") != "rule":
            raise ValueError(f"{split}[{idx}] reward_model.style must be 'rule'.")
        if _gt_answer_is_missing(reward_model.get("ground_truth")):
            missing_gt_answer += 1

        extra = row.get("extra_info") or {}
        if is_missing(extra.get("scene_id")):
            missing_scene_id += 1
        if is_missing(extra.get("question")):
            missing_question += 1
        grounding_prompt = extra.get("grounding_prompt")
        if is_missing(grounding_prompt):
            raise ValueError(f"{split}[{idx}] extra_info.grounding_prompt must be present and non-empty.")
        tools_kwargs = extra.get("tools_kwargs") or {}
        ground_tool = tools_kwargs.get("scenecot_ground") or {}
        create_kwargs = ground_tool.get("create_kwargs") or {}
        tool_grounding_prompt = create_kwargs.get("grounding_prompt")
        if is_missing(tool_grounding_prompt):
            raise ValueError(
                f"{split}[{idx}] tools_kwargs.scenecot_ground.create_kwargs.grounding_prompt "
                "must be present and non-empty."
            )
        if tool_grounding_prompt != grounding_prompt:
            raise ValueError(
                f"{split}[{idx}] tool grounding_prompt does not match extra_info.grounding_prompt."
            )
        if create_kwargs.get("agent_pos_ori") != extra.get("agent_pos_ori"):
            raise ValueError(f"{split}[{idx}] tool agent_pos_ori does not match extra_info.agent_pos_ori.")
        user_content = row["prompt"][1]["content"]
        if str(grounding_prompt) not in user_content:
            raise ValueError(f"{split}[{idx}] user prompt must contain grounding_prompt.")
        if _gt_answer_is_missing(extra.get("gt_answer")):
            missing_gt_answer += 1
        if not extra.get("gt_objects"):
            missing_gt_objects += 1
        if extra.get("need_tools_kwargs") is not True:
            raise ValueError(f"{split}[{idx}] extra_info.need_tools_kwargs must be True.")
        if not ground_tool:
            raise ValueError(f"{split}[{idx}] missing scenecot_ground tools_kwargs.")

        _validate_prompt_shape(row, split, idx)
        _validate_no_exact_cot_leak(row, split, idx)
        _validate_no_object_id_leak(row, split, idx)

    if missing_scene_id:
        raise ValueError(f"{split} has {missing_scene_id} rows missing scene_id.")
    if missing_question:
        raise ValueError(f"{split} has {missing_question} rows missing question.")
    if missing_gt_answer:
        raise ValueError(f"{split} has {missing_gt_answer} rows missing gt_answer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build verl AgentLoop/GRPO parquet files from SceneCOT annotations."
    )
    parser.add_argument("--train-input", type=Path, required=True, help="Train annotation file or directory.")
    parser.add_argument("--test-input", type=Path, required=True, help="Test annotation file or directory.")
    parser.add_argument("--val-input", type=Path, default=None, help="Optional val annotation file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output parquet directory.")
    parser.add_argument("--data-source", default="scenecot")
    parser.add_argument("--agent-name", default="scenecot_free_tag_agent")
    parser.add_argument("--ability", default="3d_scene_reasoning")
    parser.add_argument("--dataset-name", default="scenecot")
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument(
        "--model-prompt-source",
        choices=MODEL_PROMPT_SOURCES,
        default="grounding_prompt",
        help=(
            "Which extra_info field is used as the user message shown to the policy. "
            "Tools metadata still keeps grounding_prompt."
        ),
    )
    parser.add_argument(
        "--include-gt-cot",
        action="store_true",
        default=True,
        help="Store gt_answer_with_cot in extra_info for debug metadata.",
    )
    parser.add_argument(
        "--exclude-gt-cot",
        action="store_true",
        help="Do not store gt_answer_with_cot in extra_info.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be a positive integer.")

    random.seed(args.seed)
    system_prompt = load_system_prompt(args.system_prompt_file)

    split_inputs: list[tuple[str, Path]] = [
        ("train", args.train_input),
        ("test", args.test_input),
    ]
    if args.val_input is not None:
        split_inputs.append(("val", args.val_input))

    built: dict[str, list[dict[str, Any]]] = {}
    for split, input_path in split_inputs:
        resolved = input_path.expanduser()
        if not resolved.suffix:
            resolved = resolved / _candidate_filename(split, args.dataset_name)
        print(f"Reading {split} from: {resolved}")
        built[split] = build_split(resolved, split, args, system_prompt)

    if args.dry_run:
        print("\nDry run enabled: parquet files were not written.")
        return

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in built.items():
        output_path = output_dir / f"{split}.parquet"
        write_parquet(rows, output_path)
        print(f"Wrote {split}: {output_path}")


if __name__ == "__main__":
    main()
