#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from verl.utils.transformers_compat import unpack_visual_output
except ImportError:

    def unpack_visual_output(visual_output: Any) -> tuple[torch.Tensor, Any]:
        if hasattr(visual_output, "pooler_output"):
            return visual_output.pooler_output, getattr(visual_output, "deepstack_features", None)
        if isinstance(visual_output, tuple):
            return visual_output
        return visual_output, None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PLACEHOLDER_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Qwen3-VL visual embeddings for SceneCOT highlight images."
    )
    parser.add_argument("--image-root", type=Path, default=Path("data_assets/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("model_assets/qwen3-vl-4b-instruct"))
    parser.add_argument("--pattern", default="**/*")
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--cache-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-rank", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def discover_images(root: Path, pattern: str, limit: int | None) -> list[Path]:
    paths = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return paths[:limit] if limit is not None else paths


def partition_paths(paths: list[Path], num_partitions: int, partition_rank: int) -> list[Path]:
    return paths[partition_rank::num_partitions]


def partition_tag(num_partitions: int, partition_rank: int) -> str:
    return f"part_{partition_rank:03d}_of_{num_partitions:03d}"


def shard_stem(shard_id: int, num_partitions: int, partition_rank: int) -> str:
    if num_partitions == 1:
        return f"shard_{shard_id:06d}"
    return f"part_{partition_rank:03d}_shard_{shard_id:06d}"


def atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temp_name = handle.name
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def atomic_text_dump(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, path)


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:
        for path in paths:
            images.append(Image.open(path).convert("RGB"))
        return images
    except Exception:
        for image in images:
            image.close()
        raise


def close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


def visual_seqlens(image_grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    return (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // (merge_size * merge_size)


def build_meta(
    *,
    shard_id: int,
    shard_name: str,
    paths: list[Path],
    image_root: Path,
    image_grid_thw: torch.Tensor,
    seqlens: torch.Tensor,
) -> dict[str, Any]:
    offsets = [0]
    for length in seqlens.tolist():
        offsets.append(offsets[-1] + int(length))

    entries: list[dict[str, Any]] = []
    for row, path in enumerate(paths):
        try:
            rel_path = str(path.relative_to(image_root))
        except ValueError:
            rel_path = str(path)
        entries.append(
            {
                "image_path": rel_path,
                "shard": shard_name,
                "row": row,
                "offset": offsets[row],
                "length": offsets[row + 1] - offsets[row],
                "image_grid_thw": [int(x) for x in image_grid_thw[row].tolist()],
            }
        )

    return {
        "shard_id": shard_id,
        "shard": shard_name,
        "num_images": len(paths),
        "total_visual_tokens": offsets[-1],
        "entries": entries,
    }


def compute_batch(
    *,
    processor: Any,
    visual: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    min_pixels: int,
    max_pixels: int,
    merge_size: int,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    images = load_images(paths)
    try:
        processor_inputs = processor(
            text=[PLACEHOLDER_TEXT] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
            size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
        )
    finally:
        close_images(images)

    pixel_values = processor_inputs["pixel_values"].to(device)
    image_grid_thw = processor_inputs["image_grid_thw"].to(device)
    with torch.inference_mode():
        image_embeds, deepstack_embeds = unpack_visual_output(
            visual(pixel_values.to(dtype=visual.dtype), grid_thw=image_grid_thw)
        )

    seqlens = visual_seqlens(image_grid_thw, merge_size)
    expected = int(seqlens.sum().item())
    if int(image_embeds.shape[0]) != expected:
        raise ValueError(f"image_embeds has {image_embeds.shape[0]} rows, expected {expected}")

    deepstack_list = list(deepstack_embeds or [])
    for idx, tensor in enumerate(deepstack_list):
        if int(tensor.shape[0]) != expected:
            raise ValueError(f"deepstack[{idx}] has {tensor.shape[0]} rows, expected {expected}")

    return (
        image_embeds.detach().to("cpu", dtype=cache_dtype).contiguous(),
        [tensor.detach().to("cpu", dtype=cache_dtype).contiguous() for tensor in deepstack_list],
        image_grid_thw.detach().to("cpu", dtype=torch.int32).contiguous(),
        seqlens.detach().to("cpu", dtype=torch.int32).contiguous(),
    )


def compute_shard(
    *,
    processor: Any,
    visual: torch.nn.Module,
    paths: list[Path],
    image_root: Path,
    shard_id: int,
    shard_name: str,
    batch_size: int,
    device: torch.device,
    min_pixels: int,
    max_pixels: int,
    merge_size: int,
    cache_dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_parts: list[torch.Tensor] = []
    grid_parts: list[torch.Tensor] = []
    seqlen_parts: list[torch.Tensor] = []
    deepstack_parts: list[list[torch.Tensor]] | None = None

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        image_embeds, deepstack, image_grid_thw, seqlens = compute_batch(
            processor=processor,
            visual=visual,
            paths=batch_paths,
            device=device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            merge_size=merge_size,
            cache_dtype=cache_dtype,
        )
        image_parts.append(image_embeds)
        grid_parts.append(image_grid_thw)
        seqlen_parts.append(seqlens)
        if deepstack_parts is None:
            deepstack_parts = [[] for _ in deepstack]
        if len(deepstack_parts) != len(deepstack):
            raise ValueError(f"Deepstack layer count changed from {len(deepstack_parts)} to {len(deepstack)}")
        for idx, tensor in enumerate(deepstack):
            deepstack_parts[idx].append(tensor)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    image_embeds = torch.cat(image_parts, dim=0)
    image_grid_thw = torch.cat(grid_parts, dim=0)
    seqlens = torch.cat(seqlen_parts, dim=0)
    offsets = torch.zeros(len(seqlens) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(seqlens.to(torch.int64), dim=0)
    deepstack = [torch.cat(parts, dim=0) for parts in (deepstack_parts or [])]

    payload = {
        "format_version": 1,
        "image_embeds": image_embeds,
        "deepstack_image_embeds": deepstack,
        "image_grid_thw": image_grid_thw,
        "seqlens": seqlens,
        "offsets": offsets,
    }
    meta = build_meta(
        shard_id=shard_id,
        shard_name=shard_name,
        paths=paths,
        image_root=image_root,
        image_grid_thw=image_grid_thw,
        seqlens=seqlens,
    )
    return payload, meta


def merge_partition_outputs(output_dir: Path, num_partitions: int) -> None:
    manifests: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for rank in range(num_partitions):
        tag = partition_tag(num_partitions, rank)
        manifest_path = output_dir / f"manifest.{tag}.json"
        index_path = output_dir / f"index.{tag}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing partition manifest: {manifest_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing partition index: {index_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        with index_path.open(encoding="utf-8") as handle:
            entries.extend(json.loads(line) for line in handle if line.strip())

    base = dict(manifests[0])
    base.pop("partition_rank", None)
    base.pop("partition_tag", None)
    base["num_partitions"] = num_partitions
    base["num_images"] = sum(int(manifest["num_images"]) for manifest in manifests)
    base["total_images_written"] = sum(int(manifest["total_images_written"]) for manifest in manifests)
    base["total_visual_tokens"] = sum(int(manifest["total_visual_tokens"]) for manifest in manifests)
    base["mean_visual_tokens_per_image"] = (
        base["total_visual_tokens"] / base["total_images_written"] if base["total_images_written"] else 0.0
    )
    base["shards"] = [shard for manifest in manifests for shard in manifest["shards"]]

    layer_counts = {manifest.get("deepstack_layers") for manifest in manifests if "deepstack_layers" in manifest}
    if len(layer_counts) > 1:
        raise ValueError(f"Partition deepstack layer counts differ: {sorted(layer_counts)}")
    if layer_counts:
        base["deepstack_layers"] = layer_counts.pop()

    entries.sort(key=lambda item: item["image_path"])
    atomic_text_dump(
        output_dir / "index.jsonl",
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
    )
    atomic_json_dump(output_dir / "manifest.json", base)
    print(f"merged partitions: {num_partitions}")
    print(f"done: {base['total_images_written']} images")
    print(f"total visual tokens: {base['total_visual_tokens']}")
    print(f"mean visual tokens/image: {base['mean_visual_tokens_per_image']:.2f}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("--batch-size and --shard-size must be positive.")
    if args.min_pixels <= 0 or args.max_pixels <= 0 or args.min_pixels > args.max_pixels:
        raise ValueError("--min-pixels and --max-pixels must satisfy 0 < min <= max.")
    if args.num_partitions <= 0:
        raise ValueError("--num-partitions must be positive.")
    if args.partition_rank < 0 or args.partition_rank >= args.num_partitions:
        raise ValueError("--partition-rank must satisfy 0 <= rank < num_partitions.")

    image_root = args.image_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if args.merge_only:
        merge_partition_outputs(output_dir, args.num_partitions)
        return

    all_paths = discover_images(image_root, args.pattern, args.limit)
    paths = partition_paths(all_paths, args.num_partitions, args.partition_rank)
    num_shards = (len(paths) + args.shard_size - 1) // args.shard_size
    tag = partition_tag(args.num_partitions, args.partition_rank)
    print(f"images: {len(paths)} / {len(all_paths)}")
    print(f"partition: {args.partition_rank}/{args.num_partitions} ({tag})")
    print(f"image_root: {image_root}")
    print(f"output_dir: {output_dir}")
    print(f"model_path: {model_path}")
    print(f"processor size: shortest_edge={args.min_pixels}, longest_edge={args.max_pixels}")
    print(f"batch_size: {args.batch_size}; shard_size: {args.shard_size}; shards: {num_shards}")
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache_dtype = dtype_from_name(args.cache_dtype)
    torch_dtype = dtype_from_name(args.torch_dtype)

    print("loading processor")
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    print(f"merge_size: {merge_size}")

    print("loading Qwen3-VL model")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    visual = model.visual.eval().to(device)
    for param in visual.parameters():
        param.requires_grad_(False)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "image_root": str(image_root),
        "model_path": str(model_path),
        "num_images": len(paths),
        "total_num_images": len(all_paths),
        "pattern": args.pattern,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "processor_kwargs": {"size": {"shortest_edge": args.min_pixels, "longest_edge": args.max_pixels}},
        "merge_size": merge_size,
        "torch_dtype": args.torch_dtype,
        "cache_dtype": args.cache_dtype,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "num_partitions": args.num_partitions,
        "partition_rank": args.partition_rank,
        "partition_tag": tag,
        "shards": [],
    }

    if args.num_partitions == 1:
        index_tmp = output_dir / "index.jsonl.tmp"
        index_path = output_dir / "index.jsonl"
        manifest_path = output_dir / "manifest.json"
    else:
        index_tmp = output_dir / f"index.{tag}.jsonl.tmp"
        index_path = output_dir / f"index.{tag}.jsonl"
        manifest_path = output_dir / f"manifest.{tag}.json"
    index_tmp.unlink(missing_ok=True)

    total_images = 0
    total_tokens = 0
    for shard_id, start in enumerate(range(0, len(paths), args.shard_size)):
        shard_paths = paths[start : start + args.shard_size]
        shard_name = f"{shard_stem(shard_id, args.num_partitions, args.partition_rank)}.pt"
        shard_path = output_dir / shard_name
        meta_path = output_dir / f"{shard_stem(shard_id, args.num_partitions, args.partition_rank)}.meta.json"
        if args.resume and shard_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"[{shard_id + 1}/{num_shards}] skip {shard_name}")
        else:
            print(f"[{shard_id + 1}/{num_shards}] compute {shard_name} ({len(shard_paths)} images)")
            payload, meta = compute_shard(
                processor=processor,
                visual=visual,
                paths=shard_paths,
                image_root=image_root,
                shard_id=shard_id,
                shard_name=shard_name,
                batch_size=args.batch_size,
                device=device,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                merge_size=merge_size,
                cache_dtype=cache_dtype,
            )
            manifest["deepstack_layers"] = len(payload["deepstack_image_embeds"])
            atomic_torch_save(shard_path, payload)
            atomic_json_dump(meta_path, meta)
            del payload
            gc.collect()

        total_images += int(meta["num_images"])
        total_tokens += int(meta["total_visual_tokens"])
        manifest["shards"].append(
            {
                "shard": meta["shard"],
                "meta": meta_path.name,
                "num_images": meta["num_images"],
                "total_visual_tokens": meta["total_visual_tokens"],
            }
        )
        with index_tmp.open("a", encoding="utf-8") as handle:
            for entry in meta["entries"]:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if index_tmp.exists():
        os.replace(index_tmp, index_path)
    manifest["total_images_written"] = total_images
    manifest["total_visual_tokens"] = total_tokens
    manifest["mean_visual_tokens_per_image"] = total_tokens / total_images if total_images else 0.0
    atomic_json_dump(manifest_path, manifest)
    print(f"done: {total_images} images")
    print(f"total visual tokens: {total_tokens}")
    print(f"mean visual tokens/image: {manifest['mean_visual_tokens_per_image']:.2f}")
    if args.num_partitions > 1:
        print(f"partition manifest: {manifest_path}")
        print("run all partitions, then rerun with --merge-only to create manifest.json and index.jsonl")


if __name__ == "__main__":
    main()
