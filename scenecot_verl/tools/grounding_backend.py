from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from model.grounding_module import OBJMaskDecoder

from .object_store import SceneObjectStore
from .pq3d_object_token_builder import (
    ObjectTokenPack,
    PQ3DObjectTokenBuilder,
    load_training_config,
    load_checkpoint_state_dict,
    load_component,
    load_object_token_pack,
    object_token_cache_key,
    object_token_cache_path,
    save_object_token_pack,
)
from .query_encoder import AverageEmbeddingQueryEncoder, torch_dtype_from_string

logger = logging.getLogger(__name__)


class ObjectTokenCacheMissError(FileNotFoundError):
    """Raised when read-only object-token cache lookup misses."""


@dataclass
class GroundingBackendConfig:
    # model path
    sft_checkpoint_path: str
    model_path: str | None = None
    # Backward-compatible alias used by the existing Qwen tool configs.
    qwen_model_path: str | None = None

    # data path
    data_root: str | None = None
    msqa_root: str | None = None
    gqa3d_root: str | None = None
    object_feature_root: str | None = None
    object_image_root: str | None = None
    scene_feature_root: str | None = None
    scan_family_root: str | None = None

    # pq3d config
    pq3d_tokenizer_path: str | None = None
    pointnet_tokenizer_path: str | None = None
    query3d_pretrain_path: str | None = None
    training_config_path: str | None = None


    device: str = "cuda"
    dtype: str = "bf16"
    hidden_dim: int = 2560
    topk: int = 15
    cache_object_tokens: bool = True    # cache object tokens in memory to avoid recomputation
    object_token_cache_root: str | None = None
    object_token_cache_mode: str = "auto"


    pc_type: str = "auto"
    use_embodied_token: bool | None = None
    grounding_loss_type: str = "auto"
    grounding_module_type: str = "auto"
    infer_prob_type: str = "sigmoid"
    infer_ce_softmax_temperature: float = 0.5
    infer_mismatch_policy: str = "zero"
    infer_aggregation: str = "max"


@dataclass
class GroundCandidate:
    obj_id: str
    label: str | None
    prob: float
    rect_loc: list[float] | None
    polar_loc: list[float] | None
    image_path: str | None


@dataclass
class GroundResult:
    dataset_name: str
    scene_id: str
    question_id: str
    grounding_prompt: str
    query_text: str

    pred_obj_prob: torch.Tensor
    logits: torch.Tensor
    ranked_obj_ids: list[str]
    top1_obj_id: str
    top1_prob: float
    candidates: list[GroundCandidate]
    object_token_shape: tuple[int, ...]


def _grounding_head_cfg(config: GroundingBackendConfig) -> dict:
    return {
        "hidden_dim": config.hidden_dim,
        "num_heads": 8,
        "obj_mask_token_idx": 999,
        "loss_type": config.grounding_loss_type,
        "grounding_module_type": config.grounding_module_type,
        "grd_text_hidden_states": "average_embedding",
        "infer_prob_type": config.infer_prob_type,
        "infer_ce_softmax_temperature": config.infer_ce_softmax_temperature,
        "infer_mismatch_policy": config.infer_mismatch_policy,
        "infer_aggregation": config.infer_aggregation,
    }


class SceneCOTGroundingBackend:
    def __init__(self, config: GroundingBackendConfig | dict) -> None:
        explicit_keys = set(config.keys()) if isinstance(config, dict) else set()
        if isinstance(config, dict):
            config = GroundingBackendConfig(**config)
        self._apply_training_config(config, explicit_keys)
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch_dtype_from_string(config.dtype)
        self.object_token_cache_root = self._normalize_cache_root(config.object_token_cache_root)
        self.object_token_cache_mode = self._normalize_cache_mode(config.object_token_cache_mode)
        if self.object_token_cache_mode == "read_only" and self.object_token_cache_root is None:
            raise ValueError("object_token_cache_root is required when object_token_cache_mode=read_only.")
        query_model_path = config.model_path or config.qwen_model_path
        if not query_model_path:
            raise ValueError("GroundingBackendConfig requires model_path (or legacy qwen_model_path).")
        self.query_encoder = AverageEmbeddingQueryEncoder(
            query_model_path,
            hidden_dim=config.hidden_dim,
            device=config.device,
            dtype=config.dtype,
        )
        self.object_store = SceneObjectStore(
            data_root=config.data_root,
            msqa_root=config.msqa_root,
            gqa3d_root=config.gqa3d_root,
            object_feature_root=config.object_feature_root,
            object_image_root=config.object_image_root,
            scene_feature_root=config.scene_feature_root,
            scan_family_root=config.scan_family_root,
            device=config.device,
            pc_type=config.pc_type,
        )
        self.object_token_builder: PQ3DObjectTokenBuilder | None = None
        self.grounding_head = OBJMaskDecoder(_grounding_head_cfg(config)).to(self.device)

        state = load_checkpoint_state_dict(config.sft_checkpoint_path)
        load_component(self.grounding_head, state, ["grounding_head"])
        self.grounding_head.eval().requires_grad_(False)
        self._object_token_cache: dict[str, ObjectTokenPack] = {}

    @staticmethod
    def _normalize_cache_root(value: str | None) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        return Path(value).expanduser()

    @staticmethod
    def _normalize_cache_mode(value: str | None) -> str:
        mode = str(value or "auto").strip().lower()
        if mode in {"none", "false", "disabled"}:
            mode = "off"
        if mode not in {"auto", "read_only", "write", "off"}:
            raise ValueError(
                "object_token_cache_mode must be one of auto, read_only, write, off; "
                f"got {value!r}."
            )
        return mode

    def _get_object_token_builder(self) -> PQ3DObjectTokenBuilder:
        if self.object_token_builder is None:
            config = self.config
            self.object_token_builder = PQ3DObjectTokenBuilder(
                sft_checkpoint_path=config.sft_checkpoint_path,
                hidden_dim=config.hidden_dim,
                device=config.device,
                dtype=config.dtype,
                pq3d_tokenizer_path=config.pq3d_tokenizer_path,
                pointnet_tokenizer_path=config.pointnet_tokenizer_path,
                query3d_pretrain_path=config.query3d_pretrain_path,
                use_embodied_token=config.use_embodied_token,
                training_config_path=config.training_config_path,
            )
        return self.object_token_builder

    def _cache_path(
        self,
        *,
        dataset_name: str,
        scene_id: str,
        question_id: str,
        grounding_prompt: str,
        agent_pos_ori: object = None,
        pc_type: str | None = None,
    ) -> Path | None:
        if self.object_token_cache_root is None or self.object_token_cache_mode == "off":
            return None
        return object_token_cache_path(
            self.object_token_cache_root,
            dataset_name=dataset_name,
            scene_id=scene_id,
            question_id=question_id,
            grounding_prompt=grounding_prompt,
            agent_pos_ori=agent_pos_ori,
            pc_type=pc_type,
        )

    @staticmethod
    def _apply_training_config(config: GroundingBackendConfig, explicit_keys: set[str]) -> None:
        training_cfg = load_training_config(config.sft_checkpoint_path, config.training_config_path)
        if training_cfg is None:
            if config.grounding_loss_type == "auto":
                config.grounding_loss_type = "ce"
            if config.grounding_module_type == "auto":
                config.grounding_module_type = "concat"
            return

        if "hidden_dim" not in explicit_keys:
            hidden_dim = OmegaConf.select(training_cfg, "grounding.hidden_dim")
            if hidden_dim is not None:
                config.hidden_dim = int(hidden_dim)
        if config.use_embodied_token is None and "use_embodied_token" not in explicit_keys:
            use_embodied_token = OmegaConf.select(training_cfg, "vision3d.use_embodied_token")
            if use_embodied_token is not None:
                config.use_embodied_token = bool(use_embodied_token)
        if config.grounding_loss_type == "auto":
            loss_type = OmegaConf.select(training_cfg, "grounding.loss_type")
            config.grounding_loss_type = str(loss_type or "ce")
        if config.grounding_module_type == "auto":
            module_type = OmegaConf.select(training_cfg, "grounding.grounding_module_type")
            config.grounding_module_type = str(module_type or "concat")
        if "infer_prob_type" not in explicit_keys:
            infer_prob_type = OmegaConf.select(training_cfg, "grounding.infer_prob_type")
            if infer_prob_type is not None:
                config.infer_prob_type = str(infer_prob_type)
        if "infer_ce_softmax_temperature" not in explicit_keys:
            temperature = OmegaConf.select(training_cfg, "grounding.infer_ce_softmax_temperature")
            if temperature is not None:
                config.infer_ce_softmax_temperature = float(temperature)
        if "infer_mismatch_policy" not in explicit_keys:
            policy = OmegaConf.select(training_cfg, "grounding.infer_mismatch_policy")
            if policy is not None:
                config.infer_mismatch_policy = str(policy)
        if "infer_aggregation" not in explicit_keys:
            aggregation = OmegaConf.select(training_cfg, "grounding.infer_aggregation")
            if aggregation is not None:
                config.infer_aggregation = str(aggregation)

    def _get_object_tokens(
        self,
        *,
        dataset_name: str,
        scene_id: str,
        question_id: str,
        grounding_prompt: str,
        agent_pos_ori: object = None,
    ) -> ObjectTokenPack:
        pc_type = self.object_store._dataset_pc_type(dataset_name)
        key = object_token_cache_key(
            dataset_name=dataset_name,
            scene_id=scene_id,
            question_id=question_id,
            grounding_prompt=grounding_prompt,
            agent_pos_ori=agent_pos_ori,
            pc_type=pc_type,
        )
        if self.config.cache_object_tokens and key in self._object_token_cache:
            return self._object_token_cache[key]

        cache_path = self._cache_path(
            dataset_name=dataset_name,
            scene_id=scene_id,
            question_id=question_id,
            grounding_prompt=grounding_prompt,
            agent_pos_ori=agent_pos_ori,
            pc_type=pc_type,
        )
        if cache_path is not None and cache_path.exists():
            object_pack = load_object_token_pack(cache_path)
            if self.config.cache_object_tokens:
                self._object_token_cache[key] = object_pack
            return object_pack

        if self.object_token_cache_mode == "read_only":
            message = (
                "Cached SceneCOT object tokens not found. "
                f"Expected: {cache_path}. The object-token cache is missing; set "
                "SCENECOT_OBJECT_TOKEN_CACHE_MODE=off (online encoding, the default) "
                "or point SCENECOT_OBJECT_TOKEN_CACHE_ROOT to a valid cache."
            )
            logger.error(message)
            raise ObjectTokenCacheMissError(message)

        object_input = self.object_store.load(
            dataset_name=dataset_name,
            scene_id=scene_id,
            question_id=question_id,
            grounding_prompt=grounding_prompt,
            agent_pos_ori=agent_pos_ori,
        )

        object_pack = self._get_object_token_builder().build(object_input)
        if cache_path is not None and self.object_token_cache_mode in {"auto", "write"}:
            save_object_token_pack(
                cache_path,
                object_pack,
                metadata={
                    "dataset_name": dataset_name,
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "grounding_prompt": grounding_prompt,
                    "agent_pos_ori": agent_pos_ori,
                    "hidden_dim": self.config.hidden_dim,
                    "pc_type": pc_type,
                },
            )
        if self.config.cache_object_tokens:
            self._object_token_cache[key] = object_pack
        return object_pack

    @torch.no_grad()
    def score(
        self,
        dataset_name: str,
        scene_id: str,
        question_id: str,
        grounding_prompt: str,
        query_text: str,
        agent_pos_ori: object = None,
        topk: Optional[int] = None,
    ) -> GroundResult:

        # grounding query embedding
        query_vec = self.query_encoder.encode(query_text).to(self.device)
        if query_vec.shape != (self.config.hidden_dim,):
            raise ValueError(f"query vector dim must be {self.config.hidden_dim}, got {tuple(query_vec.shape)}.")

        # object token building
        object_pack = self._get_object_tokens(
            dataset_name=dataset_name,
            scene_id=scene_id,
            question_id=question_id,
            grounding_prompt=grounding_prompt,
            agent_pos_ori=agent_pos_ori,
        )
        object_tokens = object_pack.object_tokens.to(self.device)
        if object_tokens.ndim != 2 or object_tokens.shape[1] != self.config.hidden_dim:
            raise ValueError(
                f"object token dim must be {self.config.hidden_dim}, got {tuple(object_tokens.shape)}."
            )

        
        num_objects = object_tokens.shape[0]
        if num_objects != len(object_pack.obj_ids):
            raise ValueError("object_tokens.shape[0] must equal len(obj_ids).")
        if num_objects != len(object_pack.obj_locs_rect):
            raise ValueError("object_tokens.shape[0] must equal len(obj_locs).")
        if object_pack.object_image_paths is not None and len(object_pack.object_image_paths) != num_objects:
            raise ValueError("object image count must equal num_objects.")


        obj_masks = torch.ones((1, num_objects), dtype=torch.bool, device=self.device)


        logits = self.grounding_head.pred_object_logits(
            object_tokens.unsqueeze(0),
            query_vec.unsqueeze(0),
            obj_masks,
            self.device,
        )


        logits = logits.reshape(-1)
        probs = self.grounding_head._inference_logits_to_prob(logits).reshape(-1)
        if logits.numel() == num_objects + 1:
            logits = logits[:num_objects]
            probs = probs[:num_objects]
        if logits.numel() != num_objects:
            raise ValueError(f"logits length must equal num_objects={num_objects}, got {logits.numel()}.")
        if probs.numel() != num_objects:
            raise ValueError(f"pred_obj_prob length must equal num_objects={num_objects}, got {probs.numel()}.")

        # move results to CPU as float for downstream processing
        probs_cpu = probs.detach().float().cpu()
        logits_cpu = logits.detach().float().cpu()

        k = min(int(topk or self.config.topk), num_objects)
        ranked_indices = torch.argsort(probs_cpu, descending=True).tolist()
        top_indices = ranked_indices[:k]
        candidates: list[GroundCandidate] = []

        # build the candidate list: object ID, label, probability, location and image path
        for idx in top_indices:
            label = object_pack.obj_labels[idx] if object_pack.obj_labels else None
            image_path = object_pack.object_image_paths[idx] if object_pack.object_image_paths else None
            polar = object_pack.obj_locs_polar[idx] if object_pack.obj_locs_polar else None
            candidates.append(
                GroundCandidate(
                    obj_id=object_pack.obj_ids[idx],
                    label=label if label else None,
                    prob=float(probs_cpu[idx].item()),
                    rect_loc=object_pack.obj_locs_rect[idx],
                    polar_loc=polar,
                    image_path=image_path,
                )
            )
        if not candidates:
            raise ValueError(f"No grounding candidates produced for scene_id={scene_id}.")
            
        return GroundResult(
            dataset_name=dataset_name,
            scene_id=scene_id,
            question_id=question_id,
            grounding_prompt=grounding_prompt,
            query_text=query_text,
            pred_obj_prob=probs_cpu,
            logits=logits_cpu,
            ranked_obj_ids=[object_pack.obj_ids[i] for i in ranked_indices],
            top1_obj_id=candidates[0].obj_id,
            top1_prob=candidates[0].prob,
            candidates=candidates,
            object_token_shape=tuple(object_pack.object_tokens.shape),
        )
