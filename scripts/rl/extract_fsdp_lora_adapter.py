#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.distributed._tensor import Placement

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor  # type: ignore


def load_world_size(actor_dir: Path) -> int:
    config_path = actor_dir / "fsdp_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing FSDP config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    world_size = config.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Invalid world_size in {config_path}: {world_size!r}")
    return world_size


def extract_shard_info(state: dict[str, Any], world_size: int) -> tuple[int, tuple[str, ...]]:
    for value in state.values():
        if isinstance(value, DTensor):
            mesh_dim_names = tuple(value.device_mesh.mesh_dim_names)
            mesh_shape = tuple(value.device_mesh.mesh.shape)
            if mesh_dim_names not in (("fsdp",), ("dp", "fsdp"), ("ddp", "fsdp")):
                raise NotImplementedError(f"Unsupported DTensor mesh_dim_names: {mesh_dim_names}")
            return int(mesh_shape[-1]), mesh_dim_names
    return world_size, ("fsdp",)


def normalize_placements(
    placements: tuple[Placement, ...] | None,
    mesh_dim_names: tuple[str, ...],
) -> tuple[Placement, ...] | None:
    if placements is None:
        return None
    if mesh_dim_names and mesh_dim_names[0] in ("dp", "ddp"):
        return placements[1:]
    return placements


def local_tensor_and_placement(
    tensor: Any,
    mesh_dim_names: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[Placement, ...] | None]:
    if isinstance(tensor, DTensor):
        placements = normalize_placements(tuple(tensor.placements), mesh_dim_names)
        return tensor._local_tensor.bfloat16(), placements
    if isinstance(tensor, torch.Tensor):
        return tensor.bfloat16(), None
    raise TypeError(f"Unsupported LoRA tensor type: {type(tensor).__name__}")


def merge_tensors(tensors: list[torch.Tensor], placements: tuple[Placement, ...] | None) -> torch.Tensor:
    if placements:
        shard_placements = [placement for placement in placements if placement.is_shard()]
        unsupported = [
            placement
            for placement in placements
            if not placement.is_replicate() and not placement.is_shard()
        ]
        if unsupported:
            raise NotImplementedError(f"Unsupported DTensor placement(s): {unsupported}")
        if not shard_placements:
            return tensors[0].contiguous()
        if len(shard_placements) != 1:
            raise NotImplementedError(f"Only one sharded placement is supported, got placements={placements}")
        return torch.cat(tensors, dim=shard_placements[0].dim).contiguous()
    return torch.cat(tensors, dim=0).contiguous()


def peft_key(name: str) -> str:
    return name.replace(".default.weight", ".weight")


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def collect_lora_tensors(
    state: dict[str, Any],
    *,
    lora_names: list[str],
    rank_id: int,
    mesh_dim_names: tuple[str, ...],
    lora_shards: dict[str, list[torch.Tensor]],
    placements_by_name: dict[str, tuple[Placement, ...] | None],
) -> None:
    for name in lora_names:
        if name not in state:
            raise KeyError(f"{name} missing from rank {rank_id}")
        tensor, placements = local_tensor_and_placement(state[name], mesh_dim_names)
        lora_shards[name].append(tensor)
        if name not in placements_by_name:
            placements_by_name[name] = placements
        elif placements_by_name[name] != placements:
            raise ValueError(f"Placement mismatch for {name}: {placements_by_name[name]} vs {placements}")


def extract_lora_adapter(actor_dir: Path, output_dir: Path, rank: int, alpha: int) -> Path:
    world_size = load_world_size(actor_dir)
    print(f"world_size: {world_size}", flush=True)

    rank0_path = actor_dir / f"model_world_size_{world_size}_rank_0.pt"
    print(f"loading rank 0 to discover LoRA keys: {rank0_path}", flush=True)
    rank0_state = load_checkpoint(rank0_path)
    lora_names = sorted(name for name in rank0_state if "lora_" in name)
    if not lora_names:
        raise ValueError(f"No LoRA parameters found in {rank0_path}")
    print(f"found {len(lora_names)} LoRA tensors", flush=True)
    total_shards, mesh_dim_names = extract_shard_info(rank0_state, world_size)
    print(f"mesh_dim_names: {mesh_dim_names}", flush=True)
    print(f"total FSDP shards to load: {total_shards}", flush=True)

    lora_shards: dict[str, list[torch.Tensor]] = {name: [] for name in lora_names}
    placements_by_name: dict[str, tuple[Placement, ...] | None] = {}

    collect_lora_tensors(
        rank0_state,
        lora_names=lora_names,
        rank_id=0,
        mesh_dim_names=mesh_dim_names,
        lora_shards=lora_shards,
        placements_by_name=placements_by_name,
    )
    del rank0_state
    gc.collect()

    for rank_id in range(1, total_shards):
        shard_path = actor_dir / f"model_world_size_{world_size}_rank_{rank_id}.pt"
        print(f"loading shard {rank_id}/{total_shards - 1}: {shard_path}", flush=True)
        state = load_checkpoint(shard_path)
        collect_lora_tensors(
            state,
            lora_names=lora_names,
            rank_id=rank_id,
            mesh_dim_names=mesh_dim_names,
            lora_shards=lora_shards,
            placements_by_name=placements_by_name,
        )
        del state
        gc.collect()

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_params: OrderedDict[str, torch.Tensor] = OrderedDict()
    target_modules: set[str] = set()
    inferred_ranks: set[int] = set()

    for name in lora_names:
        merged = merge_tensors(lora_shards[name], placements_by_name[name])
        key = peft_key(name)
        lora_params[key] = merged
        parts = key.split(".")
        if len(parts) >= 3:
            target_modules.add(parts[-3])
        if ".lora_A." in key:
            inferred_ranks.add(int(merged.shape[0]))
        elif ".lora_B." in key:
            inferred_ranks.add(int(merged.shape[1]))

    if inferred_ranks and inferred_ranks != {rank}:
        raise ValueError(
            f"Requested rank={rank}, but merged LoRA tensor ranks={sorted(inferred_ranks)}; "
            "check checkpoint sharding and rank before exporting the adapter"
        )

    try:
        import peft
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Please install peft and safetensors in the active environment.") from exc

    peft_config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=sorted(target_modules),
    ).to_dict()
    peft_config["task_type"] = (
        peft_config["task_type"].value
        if hasattr(peft_config["task_type"], "value")
        else (peft_config["task_type"] or None)
    )
    peft_config["peft_type"] = (
        peft_config["peft_type"].value
        if hasattr(peft_config["peft_type"], "value")
        else (peft_config["peft_type"] or None)
    )
    peft_config["target_modules"] = list(peft_config["target_modules"])

    config_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"
    config_path.write_text(json.dumps(peft_config, indent=4) + "\n", encoding="utf-8")
    save_file(lora_params, str(model_path))

    print(f"wrote {config_path}", flush=True)
    print(f"wrote {model_path}", flush=True)
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract only the LoRA adapter from a VERL FSDP actor checkpoint.")
    parser.add_argument("--actor-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--alpha", default=32, type=int)
    args = parser.parse_args()

    extract_lora_adapter(
        actor_dir=args.actor_dir,
        output_dir=args.target_dir,
        rank=args.rank,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
