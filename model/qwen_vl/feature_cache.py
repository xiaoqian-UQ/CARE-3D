from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class Qwen25VLFeatureCache:
    """Read sharded, frozen Qwen2.5-VL visual features.

    The cache boundary is after the Qwen visual patch merger.  Each cached
    row can therefore be inserted directly after SceneCOT's ``<highlight_obj>``
    action without running the image processor or visual tower online.
    """

    CACHE_BOUNDARY = "qwen2_5_vl_visual_after_patch_merger"

    def __init__(self, root: str | Path, *, max_shards: int = 2) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_shards = max(1, int(max_shards))
        manifest_path = self.root / "manifest.json"
        index_path = self.root / "index.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing Qwen2.5-VL cache manifest: {manifest_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing Qwen2.5-VL cache index: {index_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("cache_boundary") != self.CACHE_BOUNDARY:
            raise ValueError(
                "Unsupported Qwen2.5-VL cache boundary: "
                f"{self.manifest.get('cache_boundary')!r}"
            )
        if self.manifest.get("model_type") != "qwen2_5_vl":
            raise ValueError(
                f"Expected a qwen2_5_vl cache, got {self.manifest.get('model_type')!r}"
            )

        self.feature_shape = tuple(int(value) for value in self.manifest["feature_shape"])
        if len(self.feature_shape) != 2:
            raise ValueError(f"Invalid Qwen2.5-VL feature shape: {self.feature_shape}")

        self.image_root = Path(self.manifest.get("image_root", "")).expanduser()
        self.index: dict[str, dict[str, Any]] = {}
        self._ambiguous_keys: set[str] = set()
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                self._add_index_entry(entry)

        expected = int(self.manifest.get("num_images", len(self.index)))
        unique_entries = {id(entry) for entry in self.index.values()}
        if len(unique_entries) != expected:
            raise ValueError(
                f"Qwen2.5-VL cache index has {len(unique_entries)} entries, manifest says {expected}"
            )
        self._shards: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._dummy_feature: torch.Tensor | None = None

    @staticmethod
    def normalize_key(image_path: str | Path) -> str:
        return str(image_path).replace("\\", "/").lstrip("./")

    def _add_index_entry(self, entry: dict[str, Any]) -> None:
        rel_path = self.normalize_key(entry["image_path"])
        keys = [rel_path, Path(rel_path).name]
        if self.image_root:
            full_path = self.image_root / rel_path
            keys.extend(
                [
                    self.normalize_key(full_path),
                    self.normalize_key(full_path.resolve()),
                ]
            )
        for key in keys:
            if key in self._ambiguous_keys:
                continue
            existing = self.index.get(key)
            if existing is not None and existing is not entry:
                # Basenames should be unique for the current ScanNet object
                # image layout.  Refuse ambiguous fallback keys rather than
                # silently loading the wrong object image.
                if key == Path(key).name:
                    self.index.pop(key, None)
                    self._ambiguous_keys.add(key)
                    continue
                raise ValueError(f"Duplicate Qwen2.5-VL cache key: {key}")
            self.index[key] = entry

    def __len__(self) -> int:
        return int(self.manifest.get("num_images", 0))

    def _lookup_entry(self, image_path: str | Path) -> dict[str, Any]:
        raw = Path(image_path).expanduser()
        candidates = [self.normalize_key(raw)]
        try:
            resolved = raw.resolve()
            candidates.append(self.normalize_key(resolved))
            if self.image_root:
                candidates.append(self.normalize_key(resolved.relative_to(self.image_root.resolve())))
        except (OSError, ValueError):
            pass
        candidates.append(raw.name)
        for candidate in candidates:
            entry = self.index.get(candidate)
            if entry is not None:
                return entry
        raise KeyError(f"Image is not in Qwen2.5-VL cache: {image_path}")

    def _load_shard(self, shard_name: str) -> torch.Tensor:
        cached = self._shards.pop(shard_name, None)
        if cached is not None:
            self._shards[shard_name] = cached
            return cached

        shard_path = self.root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing Qwen2.5-VL cache shard: {shard_path}")
        try:
            payload = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            try:
                payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(shard_path, map_location="cpu")
        features = payload.get("image_features")
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError(f"Invalid image_features payload in {shard_path}")
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
        entry = self._lookup_entry(image_path)
        features = self._load_shard(str(entry["shard"]))
        row = int(entry["row"])
        if not 0 <= row < features.shape[0]:
            raise IndexError(f"Invalid cache row {row} for {entry['shard']}")
        return features[row]

    def load_dummy(self) -> torch.Tensor:
        if self._dummy_feature is not None:
            return self._dummy_feature
        dummy_name = self.manifest.get("dummy_feature", "dummy_feature.pt")
        dummy_path = self.root / str(dummy_name)
        if not dummy_path.is_file():
            raise FileNotFoundError(f"Missing Qwen2.5-VL dummy feature: {dummy_path}")
        try:
            payload = torch.load(dummy_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(dummy_path, map_location="cpu")
        feature = payload.get("image_feature")
        if not isinstance(feature, torch.Tensor) or tuple(feature.shape) != self.feature_shape:
            raise ValueError(
                f"Invalid dummy Qwen2.5-VL feature shape in {dummy_path}: "
                f"{getattr(feature, 'shape', None)}"
            )
        self._dummy_feature = feature
        return feature
