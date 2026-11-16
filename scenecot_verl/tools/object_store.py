from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from data.data_utils import (
    LabelConverter,
    construct_bbox_corners,
    convert_pc_to_box,
    eval_ref_one_sample,
    point_cloud_iou,
)


@dataclass
class ObjectInputPack:  # object-level input pack: per-object point clouds, image features, locations, etc.
    dataset_name: str
    scene_id: str
    question_id: str
    grounding_prompt: str

    obj_fts_img: torch.Tensor
    obj_fts_vox: torch.Tensor
    obj_fts: torch.Tensor
    obj_locs: torch.Tensor
    obj_masks: torch.Tensor | None

    obj_ids: list[str]
    obj_labels: list[str] | None
    obj_locs_rect: list[list[float]]
    obj_locs_polar: list[list[float]] | None
    agent_pos_ori: list[list[float]] | None
    object_image_paths: list[str] | None
    all_obj_imgs: list[Any] | None

    raw: dict[str, Any]


def _default_data_root() -> Path:
    return Path(os.environ.get("SCENECOT_DATA_ROOT", "./data_assets")).expanduser()


def _as_path(value: str | Path | None, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.expanduser()
    return Path(value).expanduser()


def _find_scannet_label_path(scan_family_root: Path) -> Path | None:
    candidates = [
        scan_family_root / "annotations" / "meta_data" / "scannetv2-labels.combined.tsv",
        scan_family_root / "scannet" / "annotations" / "meta_data" / "scannetv2-labels.combined.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(scan_family_root.glob("**/scannetv2-labels.combined.tsv"))
    return matches[0] if matches else None


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor_feature(value: Any, dim: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(dim, dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    else:
        tensor = torch.as_tensor(value).float()
    if tensor.numel() != dim:
        raise ValueError(f"Feature dimension mismatch: expected {dim}, got {tensor.numel()} for value {value}.")
    #     tensor = tensor.reshape(-1)
    #     if tensor.numel() < dim:
    #         tensor = torch.cat([tensor, torch.zeros(dim - tensor.numel())], dim=0)
    #     else:
    #         tensor = tensor[:dim]
    return tensor.reshape(dim)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _compute_loc(obj_pcd: np.ndarray) -> np.ndarray:
    xyz = obj_pcd[:, :3]
    center = xyz.mean(0)
    size = xyz.max(0) - xyz.min(0)
    return np.concatenate([center, size], 0)


def _compute_polar(loc: list[float], agent_pos_ori: list[list[float]] | None) -> list[float]:
    if agent_pos_ori is None:
        x, y = float(loc[0]), float(loc[1])
        angle = math.degrees(math.atan2(y, x))
        distance = math.sqrt(x * x + y * y)
        return [angle, distance]

    agent_pos = np.asarray(agent_pos_ori[0], dtype=np.float32)
    agent_ori = np.asarray(agent_pos_ori[1], dtype=np.float32)
    orientation_vec = agent_ori[:2]
    norm = np.linalg.norm(orientation_vec)
    if norm < 1e-6:
        orientation_vec = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        orientation_vec = orientation_vec / norm

    theta = math.atan2(float(orientation_vec[1]), float(orientation_vec[0]))
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    rotation_matrix = np.asarray([[cos_theta, sin_theta], [-sin_theta, cos_theta]], dtype=np.float32)
    relative_xy = np.asarray(loc[:2], dtype=np.float32) - agent_pos[:2]
    relative_xy = rotation_matrix @ relative_xy
    angle = math.degrees(math.atan2(float(relative_xy[1]), float(relative_xy[0])))
    distance = math.sqrt(float(relative_xy[0]) ** 2 + float(relative_xy[1]) ** 2)
    return [angle, distance]


def _normalize_agent_pos_ori(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if len(value) != 2:
        raise ValueError(f"agent_pos_ori must be [position, orientation], got {value!r}.")
    position, orientation = value
    if position is None or orientation is None:
        return None
    pos = [float(item) for item in position]
    ori = [float(item) for item in orientation]
    return [pos, ori]


def _preprocess_obj_pcds(
    obj_pcds: list[np.ndarray],
    obj_ids: list[str],
    scene_id: str,
    *,
    obj_num_points: int,
) -> torch.Tensor:
    processed: list[np.ndarray] = []
    # iterate over each object's point cloud
    for obj_pcd, obj_id in zip(obj_pcds, obj_ids):
        obj_pcd = np.array(obj_pcd, copy=True)
        if len(obj_pcd) == 0:
            obj_pcd = np.zeros((1, 6), dtype=np.float32)
        # center of the object's point cloud
        center = obj_pcd[:, :3].mean(0)
        # per-object sampling seed so the same object is always sampled identically
        rng = np.random.default_rng(_stable_seed(scene_id, obj_id))
        indices = rng.choice(len(obj_pcd), size=obj_num_points, replace=len(obj_pcd) < obj_num_points)
        sampled = obj_pcd[indices].astype(np.float32, copy=True)
        # center the point cloud on its centroid
        sampled[:, :3] = sampled[:, :3] - center
        # max distance from any point to the origin
        max_dist = np.sqrt((sampled[:, :3] ** 2).sum(1)).max()

        if max_dist < 1e-6:
            max_dist = 1.0
        # normalize object scale to a reasonable range so large values do not destabilize training
        sampled[:, :3] = sampled[:, :3] / max_dist
        processed.append(sampled)
    return torch.from_numpy(np.stack(processed, 0)).float()


class SceneObjectStore:
    """Load SceneCOT object-level inputs needed by online PQ3D."""

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        msqa_root: str | Path | None = None,
        gqa3d_root: str | Path | None = None,
        object_feature_root: str | Path | None = None,
        object_image_root: str | Path | None = None,
        scene_feature_root: str | Path | None = None,
        scan_family_root: str | Path | None = None,
        split: str | None = None,
        device: str | None = None,
        pc_type: str = "auto",
        obj_num_points: int = 1024,
        obj_img_feat_dim: int = 768,
        obj_vox_feat_dim: int = 768,
        nms_iou_threshold: float = 0.25,
        iou_threshold: float = 0.25,
    ) -> None:
        data_root_path = _as_path(data_root, _default_data_root())
        self.data_root = data_root_path
        self.msqa_root = _as_path(msqa_root, data_root_path / "MSQA")
        self.gqa3d_root = _as_path(gqa3d_root, data_root_path / "GQA3D")
        self.sceneverse_root = _as_path(scene_feature_root, data_root_path / "SceneVerse")
        self.object_feature_root = _as_path(object_feature_root, data_root_path / "scene-verse-pred-all" / "ScanNet")
        self.object_image_root = _as_path(object_image_root, data_root_path / "scenecot_imgs" / "imgs" / "scannet")
        self.scan_family_root = _as_path(scan_family_root, data_root_path / "scan_family")
        self.split = split
        self.device = device
        self.pc_type = pc_type
        self.obj_num_points = int(obj_num_points)
        self.obj_img_feat_dim = int(obj_img_feat_dim)
        self.obj_vox_feat_dim = int(obj_vox_feat_dim)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.iou_threshold = float(iou_threshold)
        self._scene_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._pack_cache: dict[tuple[str, str, str, str, str], ObjectInputPack] = {}

        label_path = _find_scannet_label_path(self.scan_family_root)
        self.label_converter = LabelConverter(str(label_path)) if label_path is not None else None

    def _dataset_pc_type(self, dataset_name: str) -> str:
        if self.pc_type != "auto":
            return self.pc_type
        dataset = dataset_name.lower()
        if dataset == "gqa3d":
            return "gt"
        if dataset in {"msqa", "msr3d"}:
            return "pred"
        return "gt"

    def _image_path(self, scene_id: str, inst_id: str | int) -> str | None:
        path = self.object_image_root / f"{scene_id}_inst{inst_id}_0.jpg"
        if path.exists():
            return str(path)
        matches = sorted(self.object_image_root.glob(f"{scene_id}_inst{inst_id}_*_0.jpg"))
        if not matches:
            matches = sorted(self.object_image_root.glob(f"{scene_id}_inst{inst_id}_*.jpg"))
        return str(matches[0]) if matches else None


    # Load the object-level data PQ3D needs: point clouds, images, voxel features
    def _load_scene(self, domain: str, scene_id: str, *, need_pred: bool) -> dict[str, Any]:
        key = (domain, scene_id)
        cached = self._scene_cache.get(key)
        if cached is not None and (not need_pred or "obj_pcds_pred" in cached):
            return cached

        scan_data_dirs = [
            self.sceneverse_root / domain / "scan_data",
            self.sceneverse_root / "scan_data",
        ]
        pcd_candidates = [
            scan_data_dir / "pcd_with_global_alignment" / f"{scene_id}.pth"
            for scan_data_dir in scan_data_dirs
        ]
        pcd_path = next((path for path in pcd_candidates if path.exists()), None)
        if pcd_path is None:
            tried = "\n  ".join(str(path) for path in pcd_candidates)
            raise FileNotFoundError(f"Scene point cloud not found. Tried:\n  {tried}")
        pcd_data = torch.load(pcd_path, weights_only=False, map_location="cpu")
        points, colors, points_inst_ids = _to_numpy(pcd_data[0]), _to_numpy(pcd_data[1]), _to_numpy(pcd_data[-1])
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], axis=1)

        label_candidates = [
            scan_data_dir / "instance_id_to_label" / f"{scene_id}.pth"
            for scan_data_dir in scan_data_dirs
        ]
        label_path = next((path for path in label_candidates if path.exists()), None)
        if label_path is None:
            tried = "\n  ".join(str(path) for path in label_candidates)
            raise FileNotFoundError(f"Instance label file not found. Tried:\n  {tried}")
        inst_id2label = torch.load(label_path, weights_only=False, map_location="cpu")

        obj_pcds_gt: dict[int, np.ndarray] = {}
        for inst_id, _label in inst_id2label.items():
            mask = points_inst_ids == inst_id
            if mask.any():
                obj_pcds_gt[int(inst_id)] = pcds[mask]

        scene: dict[str, Any] = {
            "obj_pcds_gt": obj_pcds_gt,
            "obj_labels": {int(k): str(v) for k, v in inst_id2label.items()},
        }

        scene.update(self._load_offline_features(scene_id, obj_pcds_gt, suffix="gt"))

        if need_pred:
            scene.update(self._load_pred_objects(scene_id, pcds, obj_pcds_gt))
            scene.update(self._load_offline_features(scene_id, scene["obj_pcds_pred"], suffix="pred"))

        self._scene_cache[key] = scene
        return scene

    def _load_offline_features(
        self, scene_id: str, obj_pcds: dict[int, np.ndarray], *, suffix: str
    ) -> dict[str, dict[int, torch.Tensor]]:
        img_features = {int(i): torch.zeros(self.obj_img_feat_dim) for i in obj_pcds.keys()}
        vox_features = {int(i): torch.zeros(self.obj_vox_feat_dim) for i in obj_pcds.keys()}
        image_path = self.object_feature_root / f"image_obj_feat_{suffix}" / f"{scene_id}.pth"
        voxel_path = self.object_feature_root / f"voxel_obj_feat_{suffix}" / f"{scene_id}.pth"
        if image_path.exists():
            feat_dict = torch.load(image_path, weights_only=False, map_location="cpu")
            for obj_id in img_features:
                if obj_id in feat_dict:
                    img_features[obj_id] = _tensor_feature(feat_dict[obj_id], self.obj_img_feat_dim)
        if voxel_path.exists():
            feat_dict = torch.load(voxel_path, weights_only=False, map_location="cpu")
            for obj_id in vox_features:
                if obj_id in feat_dict:
                    vox_features[obj_id] = _tensor_feature(feat_dict[obj_id], self.obj_vox_feat_dim)
        return {f"obj_fts_img_{suffix}": img_features, f"obj_fts_vox_{suffix}": vox_features}

    def _load_pred_objects(
        self, scene_id: str, pcds: np.ndarray, obj_pcds_gt: dict[int, np.ndarray]
    ) -> dict[str, Any]:
        from scipy import sparse

        mask_dir = self.object_feature_root / "mask"
        mask_path = mask_dir / f"{scene_id}.mask.npz"
        score_path = mask_dir / f"{scene_id}.score.npy"
        label_path = mask_dir / f"{scene_id}.label.npy"
        for path in [mask_path, score_path, label_path]:
            if not path.exists():
                raise FileNotFoundError(f"Predicted object metadata not found: {path}")

        obj_masks = np.array(sparse.load_npz(mask_path).todense())[:100, :]
        obj_masks_len = len(obj_masks)
        obj_mask_scores = np.array(np.load(score_path))[:obj_masks_len]
        label_ids = np.load(label_path)

        pred_raw: dict[int, np.ndarray] = {}
        for i in range(obj_masks.shape[0]):
            mask = obj_masks[i] == 1
            if mask.any():
                pred_raw[i] = pcds[mask]

        order = sorted(range(len(obj_mask_scores)), key=lambda k: -obj_mask_scores[k])
        nms_masks: list[np.ndarray] = []
        pred_labels: dict[int, str] = {}
        nms_id = 0
        while order:
            i = order.pop(0)
            if i not in pred_raw:
                continue
            nms_masks.append(obj_masks[i])
            label_id = int(label_ids[i])
            if self.label_converter and label_id in self.label_converter.scannet_raw_id_to_raw_name:
                pred_labels[nms_id] = self.label_converter.scannet_raw_id_to_raw_name[label_id]
            else:
                pred_labels[nms_id] = f"label_{label_id}"
            nms_id += 1
            for j in order.copy():
                if j not in pred_raw:
                    order.remove(j)
                    continue
                iou = point_cloud_iou(obj_masks[i], obj_masks[j])
                if iou > self.nms_iou_threshold and label_ids[i] == label_ids[j]:
                    order.remove(j)

        obj_pcds_pred: dict[int, np.ndarray] = {}
        for i, mask_values in enumerate(nms_masks):
            mask = mask_values == 1
            if mask.any():
                obj_pcds_pred[i] = pcds[mask]

        gt_to_pred_id: dict[int, tuple[int, int]] = {}
        for gt_id, gt_pcd in obj_pcds_gt.items():
            iou_max = 0.0
            best_pred = -1
            gt_center, gt_size = convert_pc_to_box(gt_pcd)
            for pred_id, pred_pcd in obj_pcds_pred.items():
                pred_center, pred_size = convert_pc_to_box(pred_pcd)
                current_iou = eval_ref_one_sample(
                    construct_bbox_corners(pred_center, pred_size),
                    construct_bbox_corners(gt_center, gt_size),
                )
                if current_iou > iou_max:
                    iou_max = current_iou
                    best_pred = pred_id
            if best_pred != -1 and iou_max > 0:
                gt_to_pred_id[int(gt_id)] = (int(best_pred), 1 if iou_max > self.iou_threshold else 0)

        pred_to_gt_id = {pred_id: gt_id for gt_id, (pred_id, _flag) in gt_to_pred_id.items()}
        return {
            "obj_pcds_pred": obj_pcds_pred,
            "obj_label_names_pred": pred_labels,
            "gt_to_pred_id": gt_to_pred_id,
            "pred_to_gt_id": pred_to_gt_id,
        }

    def load(
        self,
        *,
        dataset_name: str,
        scene_id: str,
        question_id: str,
        grounding_prompt: str,
        agent_pos_ori: Any = None,
    ) -> ObjectInputPack:
        if not grounding_prompt or not str(grounding_prompt).strip():
            raise ValueError("grounding_prompt must be non-empty.")
        agent_pos_ori = _normalize_agent_pos_ori(agent_pos_ori)
        key = (dataset_name, scene_id, question_id, grounding_prompt, repr(agent_pos_ori))
        if key in self._pack_cache:
            return self._pack_cache[key]

        pc_type = self._dataset_pc_type(dataset_name)
        scene = self._load_scene("ScanNet", scene_id, need_pred=pc_type == "pred")
        if pc_type == "pred":
            obj_pcd_dict = scene["obj_pcds_pred"]
            img_feat_dict = scene["obj_fts_img_pred"]
            vox_feat_dict = scene["obj_fts_vox_pred"]
            labels_dict = scene["obj_label_names_pred"]
            image_id_map = scene.get("pred_to_gt_id", {})
        else:
            obj_pcd_dict = scene["obj_pcds_gt"]
            img_feat_dict = scene["obj_fts_img_gt"]
            vox_feat_dict = scene["obj_fts_vox_gt"]
            labels_dict = scene["obj_labels"]
            image_id_map = {}

        obj_keys = sorted(int(k) for k in obj_pcd_dict.keys())
        if not obj_keys:
            raise ValueError(f"No objects found for scene_id={scene_id}, pc_type={pc_type}.")

        obj_ids = [str(i) for i in obj_keys]
        obj_labels = [str(labels_dict.get(i, "")) if labels_dict is not None else None for i in obj_keys]
        obj_pcds = [obj_pcd_dict[i] for i in obj_keys]
        obj_locs_rect = [_compute_loc(obj_pcd_dict[i]).astype(float).tolist() for i in obj_keys]
        obj_locs_polar = [_compute_polar(loc, agent_pos_ori) for loc in obj_locs_rect]
        obj_fts = _preprocess_obj_pcds(obj_pcds, obj_ids, scene_id, obj_num_points=self.obj_num_points)
        obj_fts_img = torch.stack([_tensor_feature(img_feat_dict.get(i), self.obj_img_feat_dim) for i in obj_keys])
        obj_fts_vox = torch.stack([_tensor_feature(vox_feat_dict.get(i), self.obj_vox_feat_dim) for i in obj_keys])
        obj_locs = torch.tensor(obj_locs_rect, dtype=torch.float32)
        obj_masks = torch.ones(len(obj_ids), dtype=torch.bool)

        object_image_paths: list[str | None] = []
        for obj_id in obj_keys:
            image_obj_id = image_id_map.get(obj_id, obj_id)
            object_image_paths.append(self._image_path(scene_id, image_obj_id))
        paths_or_none = [p for p in object_image_paths]

        pack = ObjectInputPack(
            dataset_name=dataset_name,
            scene_id=scene_id,
            question_id=question_id,
            grounding_prompt=grounding_prompt,
            obj_fts_img=obj_fts_img,
            obj_fts_vox=obj_fts_vox,
            obj_fts=obj_fts,
            obj_locs=obj_locs,
            obj_masks=obj_masks,
            obj_ids=obj_ids,
            obj_labels=obj_labels,
            obj_locs_rect=obj_locs_rect,
            obj_locs_polar=obj_locs_polar,
            agent_pos_ori=agent_pos_ori,
            object_image_paths=paths_or_none,
            all_obj_imgs=None,
            raw={
                "pc_type": pc_type,
                "gt_to_pred_id": scene.get("gt_to_pred_id", {}),
                "pred_to_gt_id": scene.get("pred_to_gt_id", {}),
            },
        )

        n = len(obj_ids)
        if pack.obj_fts_img.shape[0] != n or pack.obj_fts_vox.shape[0] != n or pack.obj_fts.shape[0] != n:
            raise ValueError(f"Object feature count mismatch for scene_id={scene_id}.")
        if pack.obj_locs.shape[0] != n or len(pack.obj_locs_rect) != n:
            raise ValueError(f"Object location count mismatch for scene_id={scene_id}.")
        if pack.object_image_paths is not None and len(pack.object_image_paths) != n:
            raise ValueError(f"Object image path count mismatch for scene_id={scene_id}.")

        self._pack_cache[key] = pack
        return pack


ObjectStore = SceneObjectStore
