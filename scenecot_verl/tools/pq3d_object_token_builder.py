'''
Encode the object-level data prepared by object_store into object tokens with the PQ3D pcd_encoder.
'''


from __future__ import annotations
import hashlib
import json
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import model  # noqa: F401 - registers model modules.
from data.data_utils import PromptType
from model.build import build_module

from .object_store import ObjectInputPack
from .query_encoder import torch_dtype_from_string


@dataclass
class ObjectTokenPack:
    object_tokens: torch.Tensor
    obj_ids: list[str]
    obj_labels: list[str] | None
    obj_locs_rect: list[list[float]]
    obj_locs_polar: list[list[float]] | None
    object_image_paths: list[str] | None
    all_obj_imgs: list[Any] | None
    object_token_offset: int
    raw_scene_tokens_shape: tuple[int, ...]


def _cache_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return _cache_jsonable(value.detach().cpu().tolist())
    if isinstance(value, tuple):
        return [_cache_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_cache_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _cache_jsonable(item) for key, item in sorted(value.items())}
    if hasattr(value, "tolist"):
        return _cache_jsonable(value.tolist())
    return repr(value)


def _safe_cache_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return name[:80] or "unknown"


def object_token_cache_key(
    *,
    dataset_name: str,
    scene_id: str,
    question_id: str,
    grounding_prompt: str,
    agent_pos_ori: Any = None,
    pc_type: str | None = None,
) -> str:
    payload = {
        "schema": "scenecot_object_tokens_v1",
        "dataset_name": str(dataset_name),
        "scene_id": str(scene_id),
        "question_id": str(question_id),
        "grounding_prompt": str(grounding_prompt),
        "agent_pos_ori": _cache_jsonable(agent_pos_ori),
    }
    if pc_type is not None:
        payload["pc_type"] = str(pc_type)
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def object_token_cache_path(
    cache_root: str | Path,
    *,
    dataset_name: str,
    scene_id: str,
    question_id: str,
    grounding_prompt: str,
    agent_pos_ori: Any = None,
    pc_type: str | None = None,
) -> Path:
    digest = object_token_cache_key(
        dataset_name=dataset_name,
        scene_id=scene_id,
        question_id=question_id,
        grounding_prompt=grounding_prompt,
        agent_pos_ori=agent_pos_ori,
        pc_type=pc_type,
    )
    filename = f"{_safe_cache_name(scene_id)}__{_safe_cache_name(question_id)}__{digest[:16]}.pt"
    return Path(cache_root).expanduser() / _safe_cache_name(dataset_name.lower()) / digest[:2] / filename


def serialize_object_token_pack(pack: ObjectTokenPack, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": _cache_jsonable(metadata or {}),
        "object_tokens": pack.object_tokens.detach().cpu().float(),
        "obj_ids": [str(item) for item in pack.obj_ids],
        "obj_labels": None if pack.obj_labels is None else [None if item is None else str(item) for item in pack.obj_labels],
        "obj_locs_rect": [[float(value) for value in loc] for loc in pack.obj_locs_rect],
        "obj_locs_polar": None
        if pack.obj_locs_polar is None
        else [[float(value) for value in loc] for loc in pack.obj_locs_polar],
        "object_image_paths": pack.object_image_paths,
        "all_obj_imgs": None,
        "object_token_offset": int(pack.object_token_offset),
        "raw_scene_tokens_shape": tuple(int(value) for value in pack.raw_scene_tokens_shape),
    }


def deserialize_object_token_pack(payload: dict[str, Any]) -> ObjectTokenPack:
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"Unsupported object token cache schema_version={schema_version}.")
    object_tokens = payload["object_tokens"]
    if not isinstance(object_tokens, torch.Tensor):
        object_tokens = torch.as_tensor(object_tokens)
    return ObjectTokenPack(
        object_tokens=object_tokens.detach().cpu().float(),
        obj_ids=[str(item) for item in payload["obj_ids"]],
        obj_labels=payload.get("obj_labels"),
        obj_locs_rect=[[float(value) for value in loc] for loc in payload["obj_locs_rect"]],
        obj_locs_polar=None
        if payload.get("obj_locs_polar") is None
        else [[float(value) for value in loc] for loc in payload["obj_locs_polar"]],
        object_image_paths=payload.get("object_image_paths"),
        all_obj_imgs=payload.get("all_obj_imgs"),
        object_token_offset=int(payload["object_token_offset"]),
        raw_scene_tokens_shape=tuple(int(value) for value in payload["raw_scene_tokens_shape"]),
    )


def save_object_token_pack(path: str | Path, pack: ObjectTokenPack, metadata: dict[str, Any] | None = None) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    torch.save(serialize_object_token_pack(pack, metadata), tmp_path)
    tmp_path.replace(output_path)
    return output_path


def load_object_token_pack(path: str | Path) -> ObjectTokenPack:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object token cache dict, got {type(payload).__name__}.")
    return deserialize_object_token_pack(payload)


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    ckpt_path = Path(path).expanduser()
    if ckpt_path.is_dir():
        for filename in ("pytorch_model.bin", "model.safetensors"):
            candidate = ckpt_path / filename
            if candidate.is_file():
                ckpt_path = candidate
                break
        else:
            for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
                index_path = ckpt_path / index_name
                if not index_path.is_file():
                    continue
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                shard_names = sorted(set(payload.get("weight_map", {}).values()))
                merged: dict[str, torch.Tensor] = {}
                for shard_name in shard_names:
                    merged.update(load_checkpoint_state_dict(ckpt_path / shard_name))
                return merged
            raise FileNotFoundError(
                f"No pytorch_model.bin, model.safetensors, or sharded weight index found under {ckpt_path}"
            )
    if not ckpt_path.exists() or not ckpt_path.is_file():
        raise FileNotFoundError(f"SFT checkpoint file not found: {ckpt_path}")

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Mapping[str, Any] = load_file(str(ckpt_path), device="cpu")
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError(f"Expected checkpoint state dict, got {type(state).__name__}.")
    nested = state.get("state_dict")
    if not isinstance(nested, Mapping):
        nested = state.get("model")
    if isinstance(nested, Mapping):
        state = nested
    return {str(k): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def find_training_config_path(path: str | Path, explicit_path: str | Path | None = None) -> Path | None:
    if explicit_path:
        cfg_path = Path(explicit_path).expanduser()
        return cfg_path if cfg_path.exists() else None

    current = Path(path).expanduser()
    if current.is_file():
        current = current.parent
    for candidate_dir in [current, *current.parents]:
        cfg_path = candidate_dir / "config.yaml"
        if cfg_path.exists():
            return cfg_path
    return None


def load_training_config(path: str | Path, explicit_path: str | Path | None = None) -> Any | None:
    cfg_path = find_training_config_path(path, explicit_path)
    if cfg_path is None:
        return None
    return OmegaConf.load(cfg_path)


def component_state_dict(       # select one component's params from a checkpoint state dict and strip the prefix
    state_dict: dict[str, torch.Tensor],
    component_names: list[str],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    suffixes = [name + "." for name in component_names]
    for raw_key, value in state_dict.items():
        for suffix in suffixes:
            if raw_key.startswith(suffix):
                result[raw_key[len(suffix) :]] = value
                break
    return result


def load_component(     # load pcd_encoder, pcd_proj and grounding head weights
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    component_names: list[str],
) -> None:
    filtered = component_state_dict(state_dict, component_names)
    if not filtered:
        warnings.warn(f"No checkpoint keys found for component aliases: {component_names}")
        return
    incompatible = module.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys:
        warnings.warn(f"Missing keys while loading {component_names}: {incompatible.missing_keys[:20]}")
    if incompatible.unexpected_keys:
        warnings.warn(f"Unexpected keys while loading {component_names}: {incompatible.unexpected_keys[:20]}")


def _ensure_accelerate_state() -> None:
    # Original SceneCOT modules use ``accelerate.logging.get_logger``. When the
    # tool is run outside the trainer, that logger needs a minimal PartialState.
    try:
        from accelerate.state import PartialState

        PartialState()
    except Exception:
        pass


def build_pq3d_cfg( # load the default PQ3D config, apply overrides, and return the final config
    *,
    hidden_dim: int,
    pq3d_tokenizer_path: str | None,
    pointnet_tokenizer_path: str | None,
    query3d_pretrain_path: str | None,
    use_embodied_token: bool | None = None,
) -> Any:
    cfg_path = Path("configs/vision3d/pq3d.yaml")

    pq3d_cfg = OmegaConf.load(cfg_path)
    root = OmegaConf.create(
        {
            "vision3d": pq3d_cfg,
            "data": {
                "pq3d_tokenizer_path": pq3d_tokenizer_path
                or "${oc.env:SCENECOT_PQ3D_TOKENIZER_PATH,${oc.env:SCENECOT_MODEL_ROOT,./model_assets}/clip-vit-large-patch14}",
            },
            "llm": {"hidden_dim": hidden_dim},
        }
    )
    if pq3d_tokenizer_path:
        root.vision3d.query3d_cfg.model.txt_encoder.args.weights = pq3d_tokenizer_path
    if pointnet_tokenizer_path:
        root.vision3d.query3d_cfg.model.pc_encoder.args.pretrained = pointnet_tokenizer_path
    if query3d_pretrain_path:
        root.vision3d.query3d_pretrain_path = query3d_pretrain_path
    if use_embodied_token is not None:
        root.vision3d.use_embodied_token = bool(use_embodied_token)
    OmegaConf.resolve(root)
    return root.vision3d


class PQ3DObjectTokenBuilder:   # encodes object-level data from object_store into object tokens with the PQ3D pcd_encoder
    def __init__(
        self,
        *,
        sft_checkpoint_path: str,
        hidden_dim: int = 2560,
        device: str = "cuda",
        dtype: str | torch.dtype = "bf16",
        pq3d_tokenizer_path: str | None = None,
        pointnet_tokenizer_path: str | None = None,
        query3d_pretrain_path: str | None = None,
        use_embodied_token: bool | None = None,
        training_config_path: str | None = None,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.device = torch.device(device)
        self.dtype = torch_dtype_from_string(dtype)
        training_cfg = load_training_config(sft_checkpoint_path, training_config_path)
        if use_embodied_token is None and training_cfg is not None:
            use_embodied_token = OmegaConf.select(training_cfg, "vision3d.use_embodied_token")
            if use_embodied_token is not None:
                warnings.warn(f"Using vision3d.use_embodied_token={use_embodied_token} from training config.")

        # 
        self.cfg = build_pq3d_cfg(
            hidden_dim=self.hidden_dim,
            pq3d_tokenizer_path=pq3d_tokenizer_path,    # CLIP tokenizer path for the text encoder
            pointnet_tokenizer_path=pointnet_tokenizer_path,    # tokenizer path for the PointNet encoder
            query3d_pretrain_path=query3d_pretrain_path,    # initialize from this path; text/unified/mv/voxel/coord/box weights are then overwritten from best.pth
            use_embodied_token=use_embodied_token,
        )
        _ensure_accelerate_state()
        self.pcd_encoder = build_module(self.cfg).to(self.device)
        self.pcd_proj = torch.nn.Linear(int(self.cfg.hidden_dim), self.hidden_dim).to(self.device)

        # load checkpoint weights
        state = load_checkpoint_state_dict(sft_checkpoint_path)
        load_component(self.pcd_encoder, state, ["pcd_encoder"])
        load_component(self.pcd_proj, state, ["pcd_proj"])


        self.pcd_encoder.eval().requires_grad_(False)
        self.pcd_proj.eval().requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.query3d_cfg.model.txt_encoder.args.weights,
            trust_remote_code=True,
        )

    def _build_data_dict(self, pack: ObjectInputPack) -> dict[str, torch.Tensor]:
        prompt_ids = self.tokenizer(
            [pack.grounding_prompt],
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
        ).input_ids
        n = len(pack.obj_ids)
        data_dict = {
            "obj_fts_img": pack.obj_fts_img.unsqueeze(0).to(self.device),
            "obj_fts_vox": pack.obj_fts_vox.unsqueeze(0).to(self.device),
            "obj_fts": pack.obj_fts.unsqueeze(0).to(self.device),
            "obj_locs": pack.obj_locs.unsqueeze(0).to(self.device),
            "obj_masks": torch.ones((1, n), dtype=torch.bool, device=self.device)
            if pack.obj_masks is None
            else pack.obj_masks.unsqueeze(0).bool().to(self.device),
            "prompt": prompt_ids.to(self.device),
            "prompt_pad_masks": torch.ones_like(prompt_ids, dtype=torch.bool, device=self.device),
            "prompt_type": torch.tensor([PromptType.TXT], dtype=torch.long, device=self.device),
            # PQ3D embodied-token path expects a 4D orientation Fourier input.
            "anchor_orientation": torch.zeros((1, 4), dtype=torch.float32, device=self.device),
        }
        return data_dict

    @torch.no_grad()
    def build(self, pack: ObjectInputPack) -> ObjectTokenPack:
        data_dict = self._build_data_dict(pack)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            # encode scene_tokens with the PQ3D pcd_encoder, then project to hidden_dim with pcd_proj to get the final object tokens
            data_dict = self.pcd_encoder(data_dict)
            scene_tokens = self.pcd_proj(data_dict["scene_tokens"].to(self.device))

        scene_tokens = scene_tokens.squeeze(0).float()
        raw_shape = tuple(data_dict["scene_tokens"].shape)
        num_objects = len(pack.obj_ids)

        if getattr(self.pcd_encoder, "use_embodied_token", False):
            object_token_offset = 1
        elif scene_tokens.shape[0] == num_objects:
            object_token_offset = 0
        elif scene_tokens.shape[0] == num_objects + 1:
            object_token_offset = 1
            warnings.warn(
                f"Detected one leading non-object token for scene_id={pack.scene_id}; "
                "treating it as anchor/embodied token."
            )
        else:
            raise ValueError(
                "Cannot determine PQ3D object token offset: "
                f"scene_id={pack.scene_id}, question_id={pack.question_id}, "
                f"scene_tokens shape={tuple(scene_tokens.shape)}, len(obj_ids)={num_objects}, "
                f"len(all_obj_imgs)={len(pack.all_obj_imgs) if pack.all_obj_imgs is not None else None}, "
                f"len(obj_locs)={len(pack.obj_locs_rect)}"
            )

        object_tokens = scene_tokens[object_token_offset:]
        if object_tokens.shape[0] != num_objects:
            raise ValueError(
                "PQ3D object token count mismatch: "
                f"scene_id={pack.scene_id}, question_id={pack.question_id}, "
                f"scene_tokens shape={tuple(scene_tokens.shape)}, object_token_offset={object_token_offset}, "
                f"len(obj_ids)={num_objects}, len(all_obj_imgs)="
                f"{len(pack.all_obj_imgs) if pack.all_obj_imgs is not None else None}, "
                f"len(obj_locs)={len(pack.obj_locs_rect)}"
            )
        if object_tokens.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Object token dim must be {self.hidden_dim}, got {object_tokens.shape[1]} "
                f"for scene_id={pack.scene_id}."
            )
        if object_tokens.shape[0] != len(pack.obj_locs_rect):
            raise ValueError(f"object_tokens and obj_locs count mismatch for scene_id={pack.scene_id}.")
        if pack.object_image_paths is not None and len(pack.object_image_paths) != object_tokens.shape[0]:
            raise ValueError(f"object image count mismatch for scene_id={pack.scene_id}.")

        return ObjectTokenPack(
            object_tokens=object_tokens,    # [num_objects, hidden_dim]
            obj_ids=pack.obj_ids,           # [num__objects]
            obj_labels=pack.obj_labels,     # [num_objects]
            obj_locs_rect=pack.obj_locs_rect,   # per-object bbox location [x, y, z, w, h, d]
            obj_locs_polar=pack.obj_locs_polar, # per-object angle/distance relative to the agent
            object_image_paths=pack.object_image_paths,
            all_obj_imgs=pack.all_obj_imgs,
            object_token_offset=object_token_offset,
            raw_scene_tokens_shape=raw_shape,
        )
