from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from PIL import Image


SCENECOT_IMAGE_PATH_INFO_KEY = "scenecot_image_path"


def get_image_cache_path(image: Any) -> str | None:
    if isinstance(image, Image.Image):
        value = image.info.get(SCENECOT_IMAGE_PATH_INFO_KEY)
        return str(value) if value else None
    return None


class Qwen3VLVisualCache:
    def __init__(self, root: str | os.PathLike[str], *, max_shards: int = 2) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        index_path = self.root / "index.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing visual cache manifest: {manifest_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing visual cache index: {index_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.image_root = Path(self.manifest.get("image_root", "")).expanduser()
        self.max_shards = max(1, int(max_shards))
        self._index: dict[str, dict[str, Any]] = {}
        self._shards: OrderedDict[str, dict[str, Any]] = OrderedDict()

        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                self._add_index_entry(entry)

    @property
    def processor_kwargs(self) -> dict[str, Any]:
        value = self.manifest.get("processor_kwargs", {})
        return value if isinstance(value, dict) else {}

    def _add_index_entry(self, entry: dict[str, Any]) -> None:
        rel_path = str(entry["image_path"])
        keys = [rel_path, Path(rel_path).name]
        if self.image_root:
            full_path = self.image_root / rel_path
            keys.extend([str(full_path), str(full_path.resolve())])
        for key in keys:
            # Prefer exact relative/absolute paths. Basename fallback is best-effort.
            self._index.setdefault(key, entry)

    def _lookup_entry(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        raw = str(path)
        candidates = [raw]
        expanded = Path(raw).expanduser()
        candidates.append(str(expanded))
        try:
            candidates.append(str(expanded.resolve()))
        except OSError:
            pass
        if self.image_root:
            try:
                candidates.append(str(expanded.resolve().relative_to(self.image_root.resolve())))
            except (OSError, ValueError):
                pass
        candidates.append(expanded.name)

        for candidate in candidates:
            entry = self._index.get(candidate)
            if entry is not None:
                return entry
        raise KeyError(f"Image is not in visual cache: {raw}")

    def _load_shard(self, shard_name: str) -> dict[str, Any]:
        cached = self._shards.get(shard_name)
        if cached is not None:
            self._shards.move_to_end(shard_name)
            return cached

        shard_path = self.root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing visual cache shard: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._shards[shard_name] = payload
        while len(self._shards) > self.max_shards:
            self._shards.popitem(last=False)
        return payload

    def _load_one(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        entry = self._lookup_entry(path)
        payload = self._load_shard(str(entry["shard"]))
        offset = int(entry["offset"])
        length = int(entry["length"])
        end = offset + length

        image_embeds = payload["image_embeds"][offset:end].clone().contiguous()
        deepstack = [
            tensor[offset:end].clone().contiguous() for tensor in payload.get("deepstack_image_embeds", [])
        ]
        image_grid_thw = torch.tensor([entry["image_grid_thw"]], dtype=torch.long)
        return {
            "image_embeds": image_embeds,
            "deepstack_image_embeds": deepstack,
            "image_grid_thw": image_grid_thw,
        }

    def load_images(self, paths: list[str | os.PathLike[str]]) -> dict[str, Any]:
        loaded = [self._load_one(path) for path in paths]
        image_embeds = torch.cat([item["image_embeds"] for item in loaded], dim=0)
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in loaded], dim=0)
        layer_count = len(loaded[0]["deepstack_image_embeds"]) if loaded else 0
        deepstack = []
        for layer_index in range(layer_count):
            deepstack.append(torch.cat([item["deepstack_image_embeds"][layer_index] for item in loaded], dim=0))

        images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
        return {
            "image_embeds": image_embeds,
            "deepstack_image_embeds": deepstack,
            "image_grid_thw": image_grid_thw,
            "images_seqlens": images_seqlens,
        }

