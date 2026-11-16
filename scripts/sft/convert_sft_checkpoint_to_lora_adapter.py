#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file


LORA_PREFIX = "llm_model."
LORA_DEFAULT_SUFFIXES = (
    (".lora_A.default.weight", ".lora_A.weight"),
    (".lora_B.default.weight", ".lora_B.weight"),
)


def load_state_dict(path: Path) -> Mapping[str, Any]:
    if path.suffix == ".safetensors":
        return load_safetensors(str(path), device="cpu")

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, Mapping):
        nested = state.get("state_dict") or state.get("model")
        if isinstance(nested, Mapping):
            return nested
        return state
    raise TypeError(f"Unsupported checkpoint payload type: {type(state).__name__}")


def convert_lora_key(key: str, architecture: str = "native") -> str | None:
    if not key.startswith(LORA_PREFIX) or ".lora_" not in key:
        return None

    converted = key.removeprefix(LORA_PREFIX)
    if architecture == "llava-hf":
        legacy_prefix = "base_model.model.model."
        modern_prefix = "base_model.model.model.language_model."
        if not converted.startswith(legacy_prefix):
            raise ValueError(
                "Unexpected legacy LLaVA LoRA key; expected it to start with "
                f"{legacy_prefix!r}, got {converted!r}."
            )
        converted = modern_prefix + converted.removeprefix(legacy_prefix)
    for old, new in LORA_DEFAULT_SUFFIXES:
        if converted.endswith(old):
            return converted[: -len(old)] + new
    return None


def build_adapter_config(
    base_model_name_or_path: str,
    rank: int,
    alpha: int,
    architecture: str = "native",
    language_model_only: bool = False,
) -> dict[str, Any]:
    if architecture == "llava-hf":
        # A suffix-only module list would also match CLIP q/k/v projections in
        # modern HF LLaVA.  Restrict LoRA to the language model, matching the
        # original SceneCOT SFT model where the vision tower was separate.
        target_modules: str | list[str] = (
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        )
    elif language_model_only:
        # Qwen2.5-VL reuses names such as down_proj inside visual blocks.  A
        # suffix-only target list would recreate visual LoRA modules even when
        # their tensors were intentionally left out of the adapter file.  vLLM
        # multimodal rollout only supports language-model LoRA, so constrain
        # PEFT module creation as well as checkpoint tensor selection.
        target_modules = (
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        )
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return {
        "task_type": None,
        "peft_type": "LORA",
        "auto_mapping": None,
        "peft_version": "0.19.1",
        "base_model_name_or_path": base_model_name_or_path,
        "revision": None,
        "inference_mode": False,
        "r": rank,
        "target_modules": target_modules,
        "exclude_modules": None,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "use_rslora": False,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "trainable_token_indices": None,
        "loftq_config": {},
        "eva_config": None,
        "corda_config": None,
        "lora_ga_config": None,
        "use_dora": False,
        "alora_invocation_tokens": None,
        "use_qalora": False,
        "qalora_group_size": 16,
        "layer_replication": None,
        "lora_bias": False,
        "target_parameters": None,
        "use_bdlora": None,
        "arrow_config": None,
        "ensure_weight_tying": False,
    }


def infer_rank(adapter_state: Mapping[str, torch.Tensor]) -> int:
    for key, value in adapter_state.items():
        if key.endswith(".lora_A.weight"):
            return int(value.shape[0])
    raise ValueError("Could not infer LoRA rank because no lora_A tensors were found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a SceneCOT SFT checkpoint into a PEFT LoRA adapter directory."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model-name-or-path", default="model_assets/qwen3-vl-4b-instruct")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--architecture",
        choices=("native", "llava-hf"),
        default="native",
        help="Use llava-hf when converting a legacy SceneCOT LLaVA adapter for modern HF LLaVA.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help=(
            "Keep only LoRA tensors under language_model and write a PEFT regex "
            "that cannot recreate visual LoRA modules. Required for Qwen-VL "
            "adapters used by vLLM multimodal rollout."
        ),
    )
    args = parser.parse_args()

    state = load_state_dict(args.checkpoint.expanduser())
    adapter_state: dict[str, torch.Tensor] = {}
    skipped_non_language_lora = 0
    for key, value in state.items():
        converted_key = convert_lora_key(str(key), architecture=args.architecture)
        if converted_key is None:
            continue
        if args.language_model_only and ".language_model." not in converted_key:
            skipped_non_language_lora += 1
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"LoRA value is not a tensor: {key}")
        adapter_state[converted_key] = value.detach().cpu()

    if not adapter_state:
        raise ValueError(f"No SceneCOT LoRA tensors found in {args.checkpoint}")

    rank = args.lora_rank if args.lora_rank is not None else infer_rank(adapter_state)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_file(adapter_state, str(output_dir / "adapter_model.safetensors"))
    adapter_config = build_adapter_config(
        base_model_name_or_path=args.base_model_name_or_path,
        rank=rank,
        alpha=args.lora_alpha,
        architecture=args.architecture,
        language_model_only=args.language_model_only,
    )
    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=4)
        f.write("\n")

    print(f"Wrote {len(adapter_state)} LoRA tensors to {output_dir}")
    if args.language_model_only:
        print(f"Skipped {skipped_non_language_lora} non-language LoRA tensors")
    print(f"rank={rank}, lora_alpha={args.lora_alpha}")


if __name__ == "__main__":
    main()
