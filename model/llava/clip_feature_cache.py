from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class LlavaClipFeatureCache:
    """Read sharded CLIP features produced by the SceneCOT precompute script."""

    def __init__(self, root: str | Path, max_shards: int = 2):
        self.root = Path(root).expanduser().resolve()
        self.max_shards = max(1, int(max_shards))
        manifest_path = self.root / "manifest.json"
        index_path = self.root / "index.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing LLaVA CLIP cache manifest: {manifest_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing LLaVA CLIP cache index: {index_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("cache_boundary") != "clip_vision_tower_before_mm_projector":
            raise ValueError(
                "Unsupported LLaVA cache boundary: "
                f"{self.manifest.get('cache_boundary')!r}"
            )
        self.feature_shape = tuple(int(x) for x in self.manifest["feature_shape"])
        self.index: dict[str, dict[str, Any]] = {}
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                key = self.normalize_key(entry["image_path"])
                if key in self.index:
                    raise ValueError(f"Duplicate LLaVA CLIP cache key: {key}")
                self.index[key] = entry
        expected = int(self.manifest.get("num_images", len(self.index)))
        if len(self.index) != expected:
            raise ValueError(
                f"LLaVA CLIP cache index has {len(self.index)} entries, manifest says {expected}"
            )
        self._shards: OrderedDict[str, torch.Tensor] = OrderedDict()

    @staticmethod
    def normalize_key(image_path: str | Path) -> str:
        return str(image_path).replace("\\", "/").lstrip("./")

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_name: str) -> torch.Tensor:
        cached = self._shards.pop(shard_name, None)
        if cached is not None:
            self._shards[shard_name] = cached
            return cached

        shard_path = self.root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing LLaVA CLIP cache shard: {shard_path}")
        try:
            # mmap avoids reading an entire ~144 MiB shard when a random
            # training sample needs only one ~1.1 MiB feature row.
            payload = torch.load(
                shard_path, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            try:
                payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(shard_path, map_location="cpu")
        features = payload.get("clip_features")
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError(f"Invalid clip_features payload in {shard_path}")
        if tuple(features.shape[1:]) != self.feature_shape:
            raise ValueError(
                f"Feature shape mismatch in {shard_path}: {tuple(features.shape[1:])} "
                f"!= {self.feature_shape}"
            )
        self._shards[shard_name] = features
        while len(self._shards) > self.max_shards:
            self._shards.popitem(last=False)
        return features

    def load(self, image_path: str | Path) -> torch.Tensor:
        key = self.normalize_key(image_path)
        entry = self.index.get(key)
        if entry is None:
            raise KeyError(f"Image is not in LLaVA CLIP cache: {key}")
        features = self._load_shard(str(entry["shard"]))
        row = int(entry["row"])
        if not 0 <= row < features.shape[0]:
            raise IndexError(f"Invalid cache row {row} for {entry['shard']}")
        return features[row]
