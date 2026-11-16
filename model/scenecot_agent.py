import copy
import math
import os
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
                         T5ForConditionalGeneration, T5Tokenizer

from model.build import build_module
from model.llava.llava_llama import LlavaLlamaForCausalLM
from model.transformers import AttentionPooling
from model.utils import disabled_train, maybe_autocast, predict_with_beam_search
from model.grounding_module import OBJMaskDecoder
from data.cot_utils import *
from model.llava.llava_llama import load_pretrained_llava
from model.llava.clip_feature_cache import LlavaClipFeatureCache
from model.gemma3.feature_cache import Gemma3FeatureCache
from model.qwen_vl.feature_cache import Qwen25VLFeatureCache
from model.qwen_vl.image_processing import DEFAULT_QWEN_IMAGE_EDGE, DEFAULT_QWEN_IMAGE_PIXELS

logger = get_logger(__name__)

torch_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def validate_qwen_cot_action_tokens(tokenizer):
    """Fail early if the configured SceneCOT action IDs do not match the tokenizer."""
    if ACTION_TOKENS != QWEN_VL_ACTION_TOKENS:
        raise ValueError("Qwen-VL training requires SCENECOT_COT_TOKEN_SET=qwen")

    mismatches = []
    for token, expected_id in QWEN_VL_ACTION_TOKENS.items():
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids != [expected_id]:
            mismatches.append((token, expected_id, token_ids))

    if mismatches:
        details = "; ".join(
            f"{token!r}: expected {expected_id}, got {token_ids}"
            for token, expected_id, token_ids in mismatches[:5]
        )
        raise ValueError(
            "The Qwen tokenizer is incompatible with SceneCOT's fixed CoT action token IDs. "
            f"First mismatches: {details}"
        )


def validate_gemma3_cot_action_tokens(tokenizer):
    """Fail early when Gemma's dynamically selected unused tokens drift."""
    if os.environ.get("SCENECOT_COT_TOKEN_SET", "").lower() != "gemma3":
        raise ValueError("Gemma 3 training requires SCENECOT_COT_TOKEN_SET=gemma3")

    mismatches = []
    for token, expected_id in ACTION_TOKENS.items():
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids != [expected_id]:
            mismatches.append((token, expected_id, token_ids))
    if mismatches:
        details = "; ".join(
            f"{token!r}: expected {expected_id}, got {token_ids}"
            for token, expected_id, token_ids in mismatches[:5]
        )
        raise ValueError(
            "The Gemma 3 tokenizer is incompatible with the selected SceneCOT "
            f"action tokens. First mismatches: {details}"
        )


class QwenVLVisionTower(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    @property
    def dtype(self):
        return next(self.visual.parameters()).dtype

    @property
    def device(self):
        return next(self.visual.parameters()).device

    def _infer_grid_thw(self, pixel_values):
        seq_len = pixel_values.shape[0]
        side = math.isqrt(seq_len)
        if side * side != seq_len:
            raise ValueError(
                "Cannot reconstruct Qwen-VL image_grid_thw from a non-square patch tensor: "
                f"received {seq_len} patches"
            )
        merge_size = getattr(self.visual, "spatial_merge_size", 1)
        if side % merge_size != 0:
            raise ValueError(f"Qwen-VL patch-grid side {side} is not divisible by merge size {merge_size}")
        return torch.tensor([[1, side, side]], device=pixel_values.device, dtype=torch.long)

    @torch.no_grad()
    def forward(self, images):
        if images.dim() == 2:
            images = images.unsqueeze(0)
        if images.dim() != 3:
            raise ValueError(f"Qwen-VL image tensors should have shape [B, N, C], got {tuple(images.shape)}")

        features = []
        for image in images:
            image = image.to(device=self.device, dtype=self.dtype)
            grid_thw = self._infer_grid_thw(image)
            image_features = self.visual(image, grid_thw=grid_thw)
            if isinstance(image_features, tuple):
                image_features = image_features[0]
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(0)
            features.append(image_features.float())

        if all(x.shape == features[0].shape for x in features):
            return torch.cat(features, dim=0)

        max_len = max(x.shape[1] for x in features)
        hidden_size = features[0].shape[-1]
        padded = torch.zeros(len(features), max_len, hidden_size, device=features[0].device, dtype=features[0].dtype)
        for idx, feature in enumerate(features):
            padded[idx, :feature.shape[1]] = feature
        return padded


class Gemma3VisionTower(nn.Module):
    """Expose frozen Gemma 3 image features without duplicating model weights."""

    def __init__(self, multimodal_model):
        super().__init__()
        object.__setattr__(self, "_model_ref", weakref.ref(multimodal_model))

    def _model(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Gemma 3 multimodal model is no longer available")
        return model

    @property
    def dtype(self):
        return next(self._model().model.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self._model().model.vision_tower.parameters()).device

    @torch.no_grad()
    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(
                "Gemma 3 image tensors should have shape [B, C, H, W], "
                f"got {tuple(images.shape)}"
            )
        model = self._model()
        return model.get_image_features(
            images.to(device=self.device, dtype=self.dtype)
        ).float()


class SceneCOTAgent(nn.Module):
    @staticmethod
    def _mask_predicted_ids_for_grounding(predicted_ids, targets):
        """Keep predictions only at supervised output-token positions.

        During teacher-forced SFT, the target mask already identifies the
        generated answer span after prompt, scene, image, and padding tokens
        have been inserted.  Using that mask is tokenizer-independent and does
        not require an untrained model to predict EOS correctly.
        """
        if predicted_ids.shape != targets.shape:
            raise ValueError(
                "Grounding prediction/target shape mismatch: "
                f"{tuple(predicted_ids.shape)} != {tuple(targets.shape)}"
            )
        return predicted_ids.masked_fill(targets < 0, -100)

    @property
    def llm_hidden_size(self):
        if hasattr(self.llm_model.config, 'hidden_size'):
            return self.llm_model.config.hidden_size
        return self.llm_model.config.text_config.hidden_size

    @property
    def llm_dtype(self):
        return next(self.llm_model.parameters()).dtype

    def _get_decoder_layers(self):
        llm_model = self._get_base_llm_model()
        candidates = [
            ('model', 'model', 'layers'),
            ('model', 'language_model', 'layers'),
            ('language_model', 'model', 'layers'),
        ]
        for path in candidates:
            module = llm_model
            for attr in path:
                if not hasattr(module, attr):
                    break
                module = getattr(module, attr)
            else:
                return module
        raise AttributeError(f"Cannot find decoder layers for {self.llm_model.__class__.__name__}")

    def _get_base_llm_model(self):
        if hasattr(self.llm_model, 'get_base_model'):
            return self.llm_model.get_base_model()
        return self.llm_model

    def _forward_decoder_only_llm(self, **kwargs):
        output_hidden_states = kwargs.get('output_hidden_states', False)
        llm_model = self._get_base_llm_model()
        model_type = getattr(llm_model.config, 'model_type', '')
        if model_type == 'qwen3_vl' and output_hidden_states:
            model_kwargs = dict(kwargs)
            model_kwargs.pop('return_dict', None)
            model_outputs = llm_model.model(**model_kwargs)
            hidden_states = model_outputs[0]    # the first element of model_outputs is the last-layer hidden states
            logits = llm_model.lm_head(hidden_states)
            hidden_state_stack = getattr(model_outputs, 'hidden_states', None)
            if hidden_state_stack is None:
                hidden_state_stack = (hidden_states,)
            return SimpleNamespace(logits=logits, hidden_states=hidden_state_stack)
        return self.llm_model(**kwargs)

    def __init__(self, cfg):
        super().__init__()

        # LLM
        self.llm_is_decoder_only = True
        if 'vicuna' in cfg.llm.name.lower():
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = LlamaForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
            self.llm_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        elif 'opt' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        elif 'llama3' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.bfloat16)
            self.llm_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        elif 'gemma3' in cfg.llm.name.lower():
            model_kwargs = {'dtype': torch_dtype_map[cfg.llm.torch_dtype]}
            if getattr(cfg.llm, 'attn_implementation', None):
                model_kwargs['attn_implementation'] = cfg.llm.attn_implementation
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            validate_gemma3_cot_action_tokens(self.llm_tokenizer)
            self.llm_model = AutoModelForImageTextToText.from_pretrained(
                cfg.llm.cfg_path, **model_kwargs
            )
        elif 'gemma' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.bfloat16)
        elif 't5' in cfg.llm.name.lower():
            self.llm_is_decoder_only = False
            self.llm_tokenizer = T5Tokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = T5ForConditionalGeneration.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        elif 'llava' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, use_fast=False, truncation_side=cfg.llm.truncation_side)
            self.llm_model = LlavaLlamaForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        elif 'qwen' in cfg.llm.name.lower():
            model_kwargs = {'torch_dtype': torch_dtype_map[cfg.llm.torch_dtype]}
            if getattr(cfg.llm, 'attn_implementation', None):
                model_kwargs['attn_implementation'] = cfg.llm.attn_implementation
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            validate_qwen_cot_action_tokens(self.llm_tokenizer)
            self.llm_model = AutoModelForImageTextToText.from_pretrained(cfg.llm.cfg_path, **model_kwargs)
            self.use_chat_template = True
        else:
            raise NotImplementedError

        logger.info(f"Build {cfg.llm.name} from {cfg.llm.cfg_path}")
        self.is_gemma3 = 'gemma3' in cfg.llm.name.lower()
        if self.is_gemma3:
            self.img_start_token_id = int(self.llm_model.config.boi_token_index)
            self.img_end_token_id = int(self.llm_model.config.eoi_token_index)
        else:
            self.img_start_token_id = IMG_START_TOKEN_ID
            self.img_end_token_id = IMG_END_TOKEN_ID

        for param in self.llm_model.parameters():
            param.requires_grad = False
        # self.llm_model.eval()
        # self.llm_model.train = disabled_train
        self.blind = cfg.blind  # blind mode: use only input text tokens, no scene tokens
        logger.info(f"Freeze LLM, blind={self.blind}")

        # 2D vision
        # self.img_encoder = build_module(cfg.vision2d)
        # self.img_proj = nn.Linear(
        #     self.img_encoder.out_channels, self.llm_model.config.hidden_size
        # )

        self.qwen_vl_cache_mode = os.getenv(
            "SCENECOT_QWEN_VL_CACHE_MODE", "off"
        ).strip().lower()
        if self.qwen_vl_cache_mode not in {"off", "read_only", "fallback"}:
            raise ValueError(
                "SCENECOT_QWEN_VL_CACHE_MODE must be one of: off, read_only, fallback"
            )
        self.gemma3_cache_mode = os.getenv(
            "SCENECOT_GEMMA3_CACHE_MODE", "off"
        ).strip().lower()
        if self.gemma3_cache_mode not in {"off", "read_only", "fallback"}:
            raise ValueError(
                "SCENECOT_GEMMA3_CACHE_MODE must be one of: off, read_only, fallback"
            )

        # 2D image encoder
        if 'llava' in cfg.llm.name.lower():
            model_path = cfg.llm.cfg_path
            vision_model_path = cfg.llm.vision_model_path
            tokenizer, mm_model, self.img_transform = load_pretrained_llava(model_path=model_path, vision_tower_path=vision_model_path)
            self.vision_tower = copy.deepcopy(mm_model.get_model().get_vision_tower())
            self.mm_projector = copy.deepcopy(mm_model.get_model().mm_projector)
            self.mm_projector = self.mm_projector.float()
            self.vision_tower.eval()
            self.vision_tower.train = disabled_train
            print(f"vision_tower built from llava")
            print(f"mm_projector built from llava")


            # project image feature generated by clip to the same dimension as LLM.
            if not cfg.cot.freeze_mm_projector:
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
                print("enable mm_projector training")
            else:
                for p in self.mm_projector.parameters():
                    p.requires_grad = False
                print("freeze mm_projector")
        elif 'gemma3' in cfg.llm.name.lower():
            self.vision_tower = (
                None
                if self.gemma3_cache_mode == "read_only"
                else Gemma3VisionTower(self.llm_model)
            )
            self.mm_projector = nn.Identity()
            if self.vision_tower is None:
                print("Gemma 3 vision_tower replaced by read-only feature cache")
            else:
                print("vision_tower built from Gemma 3 SigLIP")
            print("mm_projector uses Gemma 3's frozen multimodal projector")
        elif 'qwen' in cfg.llm.name.lower():
            if self.qwen_vl_cache_mode == "read_only":
                # The base Qwen model still owns its frozen visual module, but
                # strict cache mode does not need SceneCOT's second deep copy.
                self.vision_tower = None
                print("Qwen-VL vision_tower replaced by read-only feature cache")
            else:
                self.vision_tower = QwenVLVisionTower(copy.deepcopy(self.llm_model.model.visual))
            self.mm_projector = nn.Identity()   # no mm_projector: the Qwen-VL vision tower already outputs suitable features
            print(f"vision_tower built from qwen-vl")
            print(f"mm_projector built from qwen-vl")
            if not cfg.cot.freeze_mm_projector:
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
                print("enable mm_projector training")
            else:
                for p in self.mm_projector.parameters():
                    p.requires_grad = False
                print("freeze mm_projector")

        self.llava_clip_cache_mode = os.getenv(
            "SCENECOT_LLAVA_CLIP_CACHE_MODE", "off"
        ).strip().lower()
        if self.llava_clip_cache_mode not in {"off", "read_only", "fallback"}:
            raise ValueError(
                "SCENECOT_LLAVA_CLIP_CACHE_MODE must be one of: off, read_only, fallback"
            )
        self.llava_clip_feature_cache = None
        self._llava_dummy_clip_feature = None
        self._llava_dummy_cache_warning_emitted = False
        if 'llava' in cfg.llm.name.lower() and self.llava_clip_cache_mode != "off":
            cache_root = os.getenv("SCENECOT_LLAVA_CLIP_CACHE_DIR", "").strip()
            if not cache_root:
                raise ValueError(
                    "SCENECOT_LLAVA_CLIP_CACHE_DIR is required when "
                    "SCENECOT_LLAVA_CLIP_CACHE_MODE is enabled"
                )
            max_shards = int(os.getenv("SCENECOT_LLAVA_CLIP_CACHE_MAX_SHARDS", "2"))
            self.llava_clip_feature_cache = LlavaClipFeatureCache(
                cache_root, max_shards=max_shards
            )
            expected_shape = (
                int(self.vision_tower.num_patches),
                int(self.vision_tower.hidden_size),
            )
            if self.llava_clip_feature_cache.feature_shape != expected_shape:
                raise ValueError(
                    "LLaVA CLIP cache/model feature mismatch: "
                    f"cache={self.llava_clip_feature_cache.feature_shape}, "
                    f"model={expected_shape}"
                )
            manifest = self.llava_clip_feature_cache.manifest
            if int(manifest.get("select_layer", self.vision_tower.select_layer)) != int(
                self.vision_tower.select_layer
            ):
                raise ValueError("LLaVA CLIP cache uses a different vision select layer")
            if manifest.get("select_feature", self.vision_tower.select_feature) != self.vision_tower.select_feature:
                raise ValueError("LLaVA CLIP cache uses a different vision select feature")
            logger.info(
                "Loaded LLaVA CLIP feature cache from %s (%d images, mode=%s)",
                cache_root,
                len(self.llava_clip_feature_cache),
                self.llava_clip_cache_mode,
            )

        self.qwen_vl_feature_cache = None
        self._qwen_dummy_cache_warning_emitted = False
        if 'qwen' in cfg.llm.name.lower() and self.qwen_vl_cache_mode != "off":
            cache_root = os.getenv("SCENECOT_QWEN_VL_CACHE_DIR", "").strip()
            if not cache_root:
                raise ValueError(
                    "SCENECOT_QWEN_VL_CACHE_DIR is required when "
                    "SCENECOT_QWEN_VL_CACHE_MODE is enabled"
                )
            max_shards = int(os.getenv("SCENECOT_QWEN_VL_CACHE_MAX_SHARDS", "2"))
            self.qwen_vl_feature_cache = Qwen25VLFeatureCache(
                cache_root, max_shards=max_shards
            )
            if self.qwen_vl_feature_cache.feature_shape[-1] != self.llm_hidden_size:
                raise ValueError(
                    "Qwen2.5-VL cache/model hidden-size mismatch: "
                    f"cache={self.qwen_vl_feature_cache.feature_shape[-1]}, "
                    f"model={self.llm_hidden_size}"
                )
            manifest = self.qwen_vl_feature_cache.manifest
            expected_edge = int(os.getenv("SCENECOT_QWEN_IMAGE_EDGE", DEFAULT_QWEN_IMAGE_EDGE))
            expected_budget = int(
                os.getenv("SCENECOT_QWEN_IMAGE_PIXELS", DEFAULT_QWEN_IMAGE_PIXELS)
            )
            if int(manifest.get("image_edge", -1)) != expected_edge:
                raise ValueError(
                    f"Qwen2.5-VL cache image_edge={manifest.get('image_edge')} "
                    f"does not match training image_edge={expected_edge}"
                )
            if int(manifest.get("pixel_budget", -1)) != expected_budget:
                raise ValueError(
                    f"Qwen2.5-VL cache pixel_budget={manifest.get('pixel_budget')} "
                    f"does not match training pixel_budget={expected_budget}"
                )
            logger.info(
                "Loaded Qwen2.5-VL feature cache from %s (%d images, shape=%s, mode=%s)",
                cache_root,
                len(self.qwen_vl_feature_cache),
                self.qwen_vl_feature_cache.feature_shape,
                self.qwen_vl_cache_mode,
            )

        self.gemma3_feature_cache = None
        self._gemma3_dummy_cache_warning_emitted = False
        if self.is_gemma3 and self.gemma3_cache_mode != "off":
            cache_root = os.getenv("SCENECOT_GEMMA3_CACHE_DIR", "").strip()
            if not cache_root:
                raise ValueError(
                    "SCENECOT_GEMMA3_CACHE_DIR is required when "
                    "SCENECOT_GEMMA3_CACHE_MODE is enabled"
                )
            max_shards = int(os.getenv("SCENECOT_GEMMA3_CACHE_MAX_SHARDS", "2"))
            self.gemma3_feature_cache = Gemma3FeatureCache(
                cache_root, max_shards=max_shards
            )
            expected_shape = (
                int(self.llm_model.config.mm_tokens_per_image),
                self.llm_hidden_size,
            )
            if self.gemma3_feature_cache.feature_shape != expected_shape:
                raise ValueError(
                    "Gemma 3 cache/model feature mismatch: "
                    f"cache={self.gemma3_feature_cache.feature_shape}, "
                    f"model={expected_shape}"
                )
            logger.info(
                "Loaded Gemma 3 feature cache from %s (%d images, shape=%s, mode=%s)",
                cache_root,
                len(self.gemma3_feature_cache),
                self.gemma3_feature_cache.feature_shape,
                self.gemma3_cache_mode,
            )

        # 3D vision: project 3D point-cloud encoder features to the LLM hidden size.
        self.pcd_encoder = build_module(cfg.vision3d)
        if 'llava' in cfg.llm.name.lower() and not hasattr(self, 'vision_tower'):
            self.pcd_proj = copy.deepcopy(self.llm_model.get_model().mm_projector)
            self.pcd_proj.float()
            for p in self.pcd_proj.parameters():
                p.requires_grad = True
        
        else:
            self.pcd_proj = nn.Linear(
                cfg.vision3d.hidden_dim, self.llm_hidden_size
            )
        self.vision3d_context = nullcontext
        if hasattr(cfg.vision3d, 'freeze') and cfg.vision3d.freeze:
            for p in self.pcd_encoder.parameters():
                p.requires_grad = False
            self.pcd_encoder.eval()
            self.pcd_encoder.train = disabled_train
            for p in self.pcd_proj.parameters():
                p.requires_grad = False
            self.pcd_proj.eval()
            self.pcd_proj.train = disabled_train
            self.vision3d_context = torch.no_grad
            logger.info(f"Freeze 3D encoder")

        # 3D vision object-centric encoder (unused)
        self.obj_encoder = build_module(cfg.vision3d_obj)



        # LoRA
        if cfg.llm.lora.flag:
            logger.info(f"Apply LoRA with configs: {cfg.llm.lora}")
            lora_config = LoraConfig(
                r=cfg.llm.lora.rank,
                lora_alpha=cfg.llm.lora.alpha,
                target_modules=cfg.llm.lora.target_modules,
                lora_dropout=cfg.llm.lora.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config=lora_config)
        else:
            self.llm_model.eval()
            self.llm_model.train = disabled_train

        self.max_context_len = cfg.llm.max_context_len
        self.max_out_len = cfg.llm.max_out_len

        # unused
        self.clip_txt_guidance = cfg.clip_txt_guidance.flag
        if self.clip_txt_guidance:
            # OpenAI CLIP is an optional dependency and newer minimal
            # environments may not provide its legacy pkg_resources import.
            # Import it only when this disabled-by-default feature is used.
            import clip

            logger.info("Add CLIP semantics guidance")
            self.clip_model = clip.load('RN50')[0]
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
            self.clip_model.train = disabled_train
            self.clip_proj = nn.Linear(cfg.clip_txt_guidance.clip_out_dim, self.llm_hidden_size)

        # grounding module
        if hasattr(cfg, 'grounding') and cfg.grounding.enable:
            self.grounding_head = OBJMaskDecoder(cfg.grounding)
            logger.info(f"Build grounding module: {cfg.grounding.loss_type}")
            self.grd_token_id =self.llm_tokenizer(GRD_TOKEN_TXT, add_special_tokens=False).input_ids[-1]
            print('found grd_token_id:', self.grd_token_id)

            # Unfreeze the last N layers of the LLM
            if cfg.grounding.unfreeze_llm_layer.flag:
                num_unfreeze = cfg.grounding.unfreeze_llm_layer.unfreeze_layers_num  # Specify the number of layers to unfreeze

                for layer in self._get_decoder_layers()[-num_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                print(f"Unfreezing the last {num_unfreeze} layers of the LLM")

            if cfg.vision3d.name.lower()== "pq3d":
                self.grd_use_scene_tokens = True
            else:
                self.grd_use_scene_tokens = False

            # Optional (not used in the paper): keep LoRA / PQ3D / projector fixed and train only the grounding head.
            if cfg.grounding.get('train_only_head', False):
                for name, param in self.named_parameters():
                    param.requires_grad = name.startswith('grounding_head.')
                logger.info("Train grounding head only; all other parameters are frozen")

        if hasattr(cfg, 'cot'):
            self.cot_mask_obj_prob_loc_token = cfg.cot.mask_obj_prob_loc_token  # True or False
            self.cot_mask_observation = bool(cfg.cot.get('mask_observation', False))
            if hasattr(cfg.cot, 'cot_answer_training'):
                self.cot_answer_training = cfg.cot.cot_answer_training
            self.max_cot_obj_num = cfg.cot.max_cot_obj_num
            self.use_oracle_obj_content = cfg.cot.use_oracle_obj_content    # at inference, use oracle object content or let the model generate it (currently False: model-generated)

            print(f"cot_mask_obj_prob_loc_token: {self.cot_mask_obj_prob_loc_token}")
            print(f"cot_mask_observation: {self.cot_mask_observation}")

            if hasattr(cfg.cot, 'cot_no_scene_tokens'):
                self.cot_no_scene_tokens = cfg.cot.cot_no_scene_tokens

            print(f"cot_no_scene_tokens: {self.cot_no_scene_tokens}")

        self.cfg = cfg

        # NOTE: do not use
        # load the predicted object mask by the best counting/existence expert
        print(f"cfg.grounding:")
        print(cfg.grounding)
        if hasattr(cfg.grounding, "moe_flag") and cfg.grounding.moe_flag:
            self.moe_flag = cfg.grounding.moe_flag
            self.moe_type_list = cfg.grounding.moe_type_list
            self.obj_prob_dict = torch.load(cfg.grounding.obj_prob_dict_path)

    # NOTE: do not use
    def encode_images(self, images):
        if hasattr(self, 'vision_tower') and self.cot_no_scene_tokens:
            image_features = self.vision_tower(images)
            image_features = self.mm_projector(image_features)
            return image_features
        else:
            ValueError("No vision tower found!")

    def _project_llava_clip_feature(self, clip_feature):
        projector_parameter = next(self.mm_projector.parameters(), None)
        projector_dtype = (
            projector_parameter.dtype if projector_parameter is not None else clip_feature.dtype
        )
        clip_feature = clip_feature.to(
            device=self.device, dtype=projector_dtype, non_blocking=True
        )
        if clip_feature.ndim == 2:
            clip_feature = clip_feature.unsqueeze(0)
        return self.mm_projector(clip_feature)

    def _get_llava_dummy_clip_feature(self):
        if self._llava_dummy_clip_feature is None:
            dummy_pixels = torch.zeros(1, 3, 336, 336, device=self.device)
            with torch.inference_mode():
                dummy_feature = self.vision_tower(dummy_pixels).squeeze(0)
            self._llava_dummy_clip_feature = dummy_feature.detach().cpu()
        if not self._llava_dummy_cache_warning_emitted:
            logger.warning(
                "At least one object image is missing; using one shared dummy CLIP feature."
            )
            self._llava_dummy_cache_warning_emitted = True
        return self._llava_dummy_clip_feature

    def _prepare_qwen_cached_feature(self, feature):
        feature = feature.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        return feature

    def _get_qwen_dummy_cached_feature(self):
        if not self._qwen_dummy_cache_warning_emitted:
            logger.warning(
                "At least one object image is missing; using the cached blank Qwen2.5-VL feature."
            )
            self._qwen_dummy_cache_warning_emitted = True
        return self._prepare_qwen_cached_feature(
            self.qwen_vl_feature_cache.load_dummy()
        )

    def _prepare_gemma3_cached_feature(self, feature):
        feature = feature.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        return feature

    def _get_gemma3_dummy_cached_feature(self):
        if not self._gemma3_dummy_cache_warning_emitted:
            logger.warning(
                "At least one object image is missing; using the cached blank Gemma 3 feature."
            )
            self._gemma3_dummy_cache_warning_emitted = True
        return self._prepare_gemma3_cached_feature(
            self.gemma3_feature_cache.load_dummy()
        )

    def _encode_visual_cue_image(
        self,
        data_dict,
        batch_idx,
        *,
        image_field,
        path_field,
        object_idx=None,
    ):
        """Load frozen visual features when enabled, otherwise encode pixels online."""
        if self.llava_clip_feature_cache is not None:
            paths = data_dict.get(path_field)
            cache_key = None
            if paths is not None:
                cache_key = paths[batch_idx]
                if object_idx is not None:
                    cache_key = cache_key[object_idx]
            if cache_key:
                try:
                    clip_feature = self.llava_clip_feature_cache.load(cache_key)
                    return self._project_llava_clip_feature(clip_feature)
                except (KeyError, FileNotFoundError, IndexError, ValueError):
                    if self.llava_clip_cache_mode == "read_only":
                        raise
                    logger.warning(
                        "LLaVA CLIP cache miss for %s; falling back to online encoding.",
                        cache_key,
                    )
            elif self.llava_clip_cache_mode == "read_only":
                return self._project_llava_clip_feature(self._get_llava_dummy_clip_feature())

        if self.qwen_vl_feature_cache is not None:
            paths = data_dict.get(path_field)
            cache_key = None
            if paths is not None:
                cache_key = paths[batch_idx]
                if object_idx is not None:
                    cache_key = cache_key[object_idx]
            if cache_key:
                try:
                    return self._prepare_qwen_cached_feature(
                        self.qwen_vl_feature_cache.load(cache_key)
                    )
                except (KeyError, FileNotFoundError, IndexError, ValueError):
                    if self.qwen_vl_cache_mode == "read_only":
                        raise
                    logger.warning(
                        "Qwen2.5-VL cache miss for %s; falling back to online encoding.",
                        cache_key,
                    )
            elif self.qwen_vl_cache_mode == "read_only":
                return self._get_qwen_dummy_cached_feature()

        if self.gemma3_feature_cache is not None:
            paths = data_dict.get(path_field)
            cache_key = None
            if paths is not None:
                cache_key = paths[batch_idx]
                if object_idx is not None:
                    cache_key = cache_key[object_idx]
            if cache_key:
                try:
                    return self._prepare_gemma3_cached_feature(
                        self.gemma3_feature_cache.load(cache_key)
                    )
                except (KeyError, FileNotFoundError, IndexError, ValueError):
                    if self.gemma3_cache_mode == "read_only":
                        raise
                    logger.warning(
                        "Gemma 3 cache miss for %s; falling back to online encoding.",
                        cache_key,
                    )
            elif self.gemma3_cache_mode == "read_only":
                return self._get_gemma3_dummy_cached_feature()

        images = data_dict.get(image_field)
        if images is None:
            raise KeyError(
                f"{image_field} is unavailable and no usable cached visual feature was found"
            )
        image = images[batch_idx]
        if object_idx is not None:
            image = image[object_idx]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if self.vision_tower is None:
            raise RuntimeError("Online vision tower is unavailable in read-only cache mode")
        return self.mm_projector(self.vision_tower(image.to(self.device)))

    @property
    def device(self):
        return list(self.parameters())[0].device

    def count_params(self, parameters):
        tot = sum([math.prod(p.shape) for p in parameters])
        return tot

    def show_params_size(self, tot):
        if tot >= 1e9:
            return '{:.1f}B'.format(tot / 1e9)
        elif tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}k'.format(tot / 1e3)

    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        learnable_params_size = self.count_params(learnable_named_params.values())
        frozen_params_size = self.count_params(frozen_named_params.values())
        logger.info(
            f"Build SceneCOT with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
        return learnable_named_params



    # Convert the batch's input-text token ids and attention mask into Qwen's input format.
    def build_input_sequence(self, data_dict):

        device = self.device
        bs = len(data_dict['input_txt'])

        input_txt_tokens = data_dict['input_txt_ids']
        input_txt_masks = data_dict['input_txt_mask']

        pad_length = torch.logical_not(input_txt_masks).sum(-1)
        input_txt_tokens_right_justified = input_txt_tokens.clone()
        input_txt_masks_right_justified = input_txt_masks.clone()
        for i in range(bs):
            this_pad_len = pad_length[i]
            if this_pad_len > 0:
                input_txt_tokens_right_justified[i, :this_pad_len] = input_txt_tokens[i, -this_pad_len:]
                input_txt_masks_right_justified[i, :this_pad_len] = 0
                input_txt_tokens_right_justified[i, this_pad_len:] = input_txt_tokens[i, :-this_pad_len]
                input_txt_masks_right_justified[i, this_pad_len:] = 1

        input_txt_embeds = self.llm_model.get_input_embeddings()(input_txt_tokens_right_justified)

        input_embeds = torch.cat([input_txt_embeds], dim=1)
        attn_masks = torch.cat([input_txt_masks_right_justified], dim=1)
        return input_embeds, attn_masks
       


    def build_new_inputs_embeds_targets(self, inputs_embeds, attention_mask, text_output_embeds, text_output_mask, text_output_ids, data_dict):
        
        # concatenate prompt text and output text into new inputs_embeds and attention_mask
        inputs_embeds = torch.cat([inputs_embeds, text_output_embeds], dim=1)   
        attention_mask = torch.cat([attention_mask, text_output_mask], dim=1)  

        # targets shaped like attention_mask, initialized to -100 (ignored positions)
        targets = torch.zeros_like(attention_mask).long().fill_(-100)   
        targets_idx = text_output_mask.bool()  # non-pad token positions in the output text
        targets[:, -targets_idx.shape[1]:][targets_idx] = text_output_ids[targets_idx]  # fill the output-text token ids into targets
    
        new_targets_list = []   # per-sample training loss targets
        new_inputs_embeds_list = [] # per-sample LLM input embeddings: prompt embeddings + GT COT embeddings
        new_attention_mask_list = []
        new_token_type_ids_list = []


        # if there is a highlight_obj, insert the image embedding after that token so Qwen generates the rest of the COT conditioned on the image
        for i in range(inputs_embeds.shape[0]):
            img_indicator_idx = torch.where(targets[i] == HIGHLIGHT_OBJ_TOKEN_ID)[0]
            if len(img_indicator_idx) == 0:
                new_targets_list.append(targets[i])
                new_inputs_embeds_list.append(inputs_embeds[i])
                new_attention_mask_list.append(attention_mask[i])
                new_token_type_ids_list.append(torch.zeros_like(attention_mask[i]))
                continue

            img_embeds = self._encode_visual_cue_image(
                data_dict,
                i,
                image_field='gt_img',
                path_field='gt_img_path',
            )

            # wrap as <img_start>, img_embeds, <img_end> so Qwen generates the rest of the COT conditioned on the image
            img_start_embeds = self.llm_model.get_input_embeddings()(torch.tensor(self.img_start_token_id).to(self.device)).unsqueeze(0).unsqueeze(0)
            img_end_embeds = self.llm_model.get_input_embeddings()(torch.tensor(self.img_end_token_id).to(self.device)).unsqueeze(0).unsqueeze(0)
            img_embeds_with_pre_suffix = torch.cat([img_start_embeds, img_embeds, img_end_embeds], dim=1)   # (1, 578, D)

            # embeddings from the start up to highlight_obj, followed by the image
            img_masks = torch.ones(img_embeds_with_pre_suffix.shape[:2], dtype=torch.int64, device=self.device) # every token in img_embeds_with_pre_suffix is valid
            inputs_embeds_with_img_embeds_prefix = torch.cat([inputs_embeds[i:i+1, :img_indicator_idx+1, :], img_embeds_with_pre_suffix], dim=1)  # embeddings of the sample from the start up to highlight_obj
            attention_mask_with_img_embeds_prefix = torch.cat([attention_mask[i:i+1, :img_indicator_idx+1], img_masks], dim=1) # mark the image embedding positions as valid too

            # append the original text after highlight_obj: prompt + COT up to highlight_obj + image embedding + remaining text
            targets_suffix = targets[i:i+1, img_indicator_idx+1:]  # after <highlight_obj>, do not include <highlight_obj>
            inputs_embeds_suffix = inputs_embeds[i:i+1, img_indicator_idx+1:, :]
            attention_mask_suffix = attention_mask[i:i+1, img_indicator_idx+1:]
            inputs_embeds_with_img = torch.cat([inputs_embeds_with_img_embeds_prefix, inputs_embeds_suffix], dim=1)   
            attention_mask_with_img = torch.cat([attention_mask_with_img_embeds_prefix, attention_mask_suffix], dim=1)   
            targets_img_mask = torch.zeros_like(img_masks).long().fill_(-100)   
            new_targets = torch.cat([targets[i:i+1, :img_indicator_idx+1], targets_img_mask, targets_suffix], dim=1)      
            new_token_type_ids = torch.zeros_like(attention_mask_with_img)
            if self.is_gemma3:
                image_block_start = img_indicator_idx + 1
                new_token_type_ids[
                    :, image_block_start + 1:image_block_start + img_embeds_with_pre_suffix.shape[1] - 1
                ] = 1



            assert new_targets.shape[1] == inputs_embeds_with_img.shape[1]
            new_targets_list.append(new_targets.squeeze(0))   
            new_inputs_embeds_list.append(inputs_embeds_with_img.squeeze(0))
            new_attention_mask_list.append(attention_mask_with_img.squeeze(0))
            new_token_type_ids_list.append(new_token_type_ids.squeeze(0))
        
        max_tensor_len = max([t.shape[0] for t in new_targets_list])
        new_targets = torch.zeros(inputs_embeds.shape[0], max_tensor_len).long().fill_(-100).to(self.device)
        new_inputs_embeds = torch.zeros(inputs_embeds.shape[0], max_tensor_len, inputs_embeds.shape[-1]).to(self.device)
        new_attention_mask = torch.zeros(inputs_embeds.shape[0], max_tensor_len).long().fill_(0).to(self.device)
        new_token_type_ids = torch.zeros(
            inputs_embeds.shape[0], max_tensor_len, dtype=torch.long, device=self.device
        )
        for i in range(len(new_targets_list)):
            new_targets[i, :new_targets_list[i].shape[0]] = new_targets_list[i]
            new_inputs_embeds[i, :new_inputs_embeds_list[i].shape[0], :] = new_inputs_embeds_list[i]
            new_attention_mask[i, :new_attention_mask_list[i].shape[0]] = new_attention_mask_list[i]
            new_token_type_ids[i, :new_token_type_ids_list[i].shape[0]] = new_token_type_ids_list[i]

        assert new_targets.shape[1] == new_inputs_embeds.shape[1] == new_attention_mask.shape[1]
        
        return new_inputs_embeds, new_attention_mask, new_targets, new_token_type_ids
        # attention_mask tells the LLM which positions are real input and which are padding
        # targets tells the loss which positions to compute and which to ignore


    # unused
    def compute_grounding_logits(self, obj_embeds, txt_embeds, obj_masks):
        """
        obj_embeds: [B, N, D]
        txt_embeds: [B, T, D]
        obj_masks: [B, N], 1 for valid and 0 for pad
        """
        obj_masks = obj_masks.bool()
        txt_feat = self.txt_pool(txt_embeds)   # [B, D]
        txt_feat = F.normalize(txt_feat, dim=-1)   # [B, D]
        obj_feat = F.normalize(obj_embeds, dim=-1)   # [B, N, D]
        sim = torch.einsum('bnd,bd->bn', obj_feat, txt_feat)   # [B, N]
        return sim.masked_fill_(~obj_masks, -torch.inf)


# data_dict
# → tokenize input_txt
# → encode 3D objects into scene_tokens/object features
# → tokenize output_gt
# → build the teacher-forcing input: prompt + GT COT
# → LLM forward to compute the generation loss
# → grounding head takes the query from <think_grd> in the GT COT and computes the grounding loss
# → return loss_gen / loss_grd / pred_obj_prob


    def forward(self, data_dict):


        device = self.device
        bs = len(data_dict['input_txt'])

        # get embedding of input_txt
        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'left'

        input_txt_tokenized = self.llm_tokenizer(
            data_dict['input_txt'],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        ).to(device)   # [BOS, tokens, PAD]

        data_dict['input_txt_ids'] = input_txt_tokenized.input_ids
        data_dict['input_txt_embed'] = self.llm_model.get_input_embeddings()(input_txt_tokenized.input_ids)
        data_dict['input_txt_mask'] = input_txt_tokenized.attention_mask

        with self.vision3d_context():
            if self.blind:
                data_dict['scene_tokens'] = torch.zeros(bs, 1, data_dict['input_txt_embed'].shape[-1]).to(device)
                data_dict['scene_masks'] = torch.zeros(bs, 1, dtype=torch.bool).to(device)
            else:
                try:
                    data_dict = self.pcd_encoder(data_dict)
                except Exception:
                    print('in scenecot_agent.py: error in pcd_encoder')
                    if 'obj_fts' in data_dict:
                        print(f'shape of obj_fts: {data_dict["obj_fts"].shape}, dtype: {data_dict["obj_fts"].dtype}')
                    if 'obj_masks' in data_dict:
                        print(f'shape of obj_masks: {data_dict["obj_masks"].shape}, dtype: {data_dict["obj_masks"].dtype}')
                    if 'obj_locs' in data_dict:
                        print(f'shape of obj_locs: {data_dict["obj_locs"].shape}, dtype: {data_dict["obj_locs"].dtype}')
                    if 'anchor_loc' in data_dict:
                        print(f'shape of anchor_loc: {data_dict["anchor_loc"].shape}')
                    raise
                data_dict['scene_tokens'] = self.pcd_proj(data_dict['scene_tokens'].to(device)) # fuses point-cloud, 2D, voxel and location information; 768 -> 2560

        inputs_embeds, attention_mask = self.build_input_sequence(data_dict=data_dict)
        # (B, T1+O+T2, D), (B, T1+O+T2)

        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in data_dict['output_gt']], # output_gt is the full answer with COT
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_out_len,
            add_special_tokens=False,
        ).to(device)


        text_output_ids = text_output_tokens.input_ids
        text_output_mask = text_output_tokens.attention_mask
        text_output_embeds = self.llm_model.get_input_embeddings()(text_output_ids)   # (B, T3, D)
        

        if not self.llm_is_decoder_only:
            labels = text_output_ids.clone()
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
            with maybe_autocast(self):
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            logits = outputs.logits.float()
            # empirical: encoder-decoder sequence does not need to shift
            num_tokens_for_loss = (labels >= 0).int().sum(1)   # (B,)
            logits = rearrange(logits, 'b t v -> (b t) v')
            labels = rearrange(labels, 'b t -> (b t)')
            loss = F.cross_entropy(logits, labels, reduction='none')
            loss = rearrange(loss, '(b t) -> b t', b=bs)
            loss = loss.sum(1) / num_tokens_for_loss   # (B,)
        else:
            
            inputs_embeds, attention_mask, targets, token_type_ids = self.build_new_inputs_embeds_targets(inputs_embeds, attention_mask, text_output_embeds, text_output_mask, text_output_ids, data_dict)
            # build new inputs_embeds and targets; 
            # insert image tokens; 
            # # mask the tokens before <highlight_obj> and after </highlight_obj>
            
            # mask token loss between <obj_prob>,</obj_prob; <obj_loc>,</obj_loc>; <obj_loc_plr>,</obj_loc_plr>   
            if hasattr(self, 'cot_mask_obj_prob_loc_token') and self.cot_mask_obj_prob_loc_token:
                assert self.max_out_len > 500    # shorter than 500 will miss the tokens after </obj_prob>


                # mask out the tokens between obj_prob/obj_loc/obj_loc_plr tags, if present
                for i in range(bs):
                    if len(torch.where(targets[i] == OBJ_PROB_START_TOKEN_ID)[0]) > 0 and len(torch.where(targets[i] == OBJ_PROB_END_TOKEN_ID)[0]) > 0:
                        start_idx = torch.where(targets[i] == OBJ_PROB_START_TOKEN_ID)[0]
                        end_idx = torch.where(targets[i] == OBJ_PROB_END_TOKEN_ID)[0]
                        targets[i, start_idx:end_idx+1] = -100    # including the start and end token
                    if len(torch.where(targets[i] == OBJ_LOC_START_TOKEN_ID)[0]) > 0 and len(torch.where(targets[i] == OBJ_LOC_END_TOKEN_ID)[0]) > 0:
                        start_idx = torch.where(targets[i] == OBJ_LOC_START_TOKEN_ID)[0]
                        end_idx = torch.where(targets[i] == OBJ_LOC_END_TOKEN_ID)[0]
                        targets[i, start_idx:end_idx+1] = -100
                    if len(torch.where(targets[i] == OBJ_LOC_PLR_START_TOKEN_ID)[0]) > 0 and len(torch.where(targets[i] == OBJ_LOC_PLR_END_TOKEN_ID)[0]) > 0:
                        start_idx = torch.where(targets[i] == OBJ_LOC_PLR_START_TOKEN_ID)[0]
                        end_idx = torch.where(targets[i] == OBJ_LOC_PLR_END_TOKEN_ID)[0]
                        targets[i, start_idx:end_idx+1] = -100
                    
            with maybe_autocast(self):  # mixed precision (e.g. fp16) where appropriate, to save memory and speed things up
                llm_forward_kwargs = {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': attention_mask,
                    'return_dict': True,
                    'output_hidden_states': True,
                }
                if self.is_gemma3:
                    llm_forward_kwargs['token_type_ids'] = token_type_ids
                outputs = self._forward_decoder_only_llm(
                    **llm_forward_kwargs
                )

            logits = outputs.logits.float()

            # shift so the logits at step t align with the label of token t+1
            shift_logits = logits[..., :-1, :].contiguous() # logits over the whole vocabulary for the next token at each step
            shift_labels = targets[..., 1:].contiguous()    # correct next-token id at each step
            num_tokens_for_loss = (shift_labels >= 0).int().sum(1)   # (B,)

            shift_logits = rearrange(shift_logits, 'b t v -> (b t) v')
            shift_labels = rearrange(shift_labels, 'b t -> (b t)')

            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = rearrange(loss, '(b t) -> b t', b=bs)
            loss = loss.sum(1) / num_tokens_for_loss   # (B,)

        if hasattr(self, 'grounding_head'):
            modified_pred_input_ids = self._mask_predicted_ids_for_grounding(
                logits.contiguous().argmax(-1),
                targets,
            )
                
            last_hidden_states = outputs.hidden_states[-1]
            last_hidden_states = last_hidden_states.float()
            
            if self.grd_use_scene_tokens:
                # pq3d setting
                obj_tokens = data_dict['scene_tokens']
            else:
                data_dict = self.obj_encoder(data_dict)
                obj_tokens = self.pcd_proj(data_dict['obj_tokens'])

            obj_masks = data_dict['obj_masks']
            self.grounding_head = self.grounding_head.to(self.device)
            
            # The grounding head uses the text between <think_grd> and </think_grd> in the GT as the query and computes its logits against all object tokens.
            obj_mask_loss, all_pred_obj_mask = self.grounding_head(input_ids=modified_pred_input_ids, gt_input_ids=targets, hidden_states=last_hidden_states, obj_tokens=obj_tokens, obj_mask_token_idx=self.grd_token_id, obj_masks=obj_masks, gt_grounding_obj_mask=data_dict['grounding_obj_mask_gt'], llm_model=self.llm_model, device=self.device)
            data_dict['loss_grd'] = obj_mask_loss    # (B,)
            data_dict['pred_obj_prob'] = all_pred_obj_mask
            loss = loss + obj_mask_loss*self.cfg.grounding.grd_loss_weight

        data_dict['loss_gen'] = loss
        return data_dict

    def build_obj_prob_loc_txt(self, pred_obj_prob, obj_labels, obj_masks, obj_locs, build_choice='prob', obj_max_num=20, agent_pos_ori=[[0,0,0], [0,0,0]], device='cuda',coord_type='rectangle'):   # prob or loc
        '''
            pred_obj_prob: (N,)
        '''
        pred_obj_prob = pred_obj_prob[obj_masks]   # keep only real objects' probabilities, dropping padding
        sorted_indices = torch.argsort(pred_obj_prob, descending=True)  # indices sorted by probability, descending
        if hasattr(self, 'max_cot_obj_num'):
            obj_max_num = self.max_cot_obj_num                                                
        top_k_indices = sorted_indices[:obj_max_num]
        top_k_labels = [obj_labels[i] for i in top_k_indices]
        top_k_probs = [pred_obj_prob[i].item() for i in top_k_indices]
        
        if build_choice == 'prob':
            output_string = " ".join(f"{label} {prob:.2f}" for label, prob in zip(top_k_labels, top_k_probs))
        elif build_choice == 'loc':
            agent_pos = agent_pos_ori[0]
            agent_ori = agent_pos_ori[1]
            orientation_vec = agent_ori[:2]
            norm = torch.linalg.norm(orientation_vec)
            orientation_vec = orientation_vec / norm
            theta = math.atan2(orientation_vec[1], orientation_vec[0])  # Convert to angle
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            rotation_matrix = torch.tensor([[cos_theta, sin_theta], [-sin_theta, cos_theta]]).to(device)
            top_k_relative_loc_str = []
            for i in top_k_indices:
                obj_loc = obj_locs[i]
                relative_loc = obj_loc[:2] - agent_pos[:2]
                relative_loc = torch.matmul(rotation_matrix, relative_loc[:2])
                relative_loc = relative_loc.tolist()
                relative_loc.append(obj_loc[2] - agent_pos[2])
                # relative_loc_str = ','.join([f"{coord:.1f}" for coord in relative_loc])
                if coord_type == 'rectangle':
                    obj_sizes = obj_loc[3:6]
                    relative_loc_str = ",".join([f"{coord:.1f}" for coord in relative_loc])
                    relative_loc_str += ","+",".join([f"{size:.1f}" for size in obj_sizes])
                elif coord_type == 'polar':
                    theta_obj = math.atan2(relative_loc[1], relative_loc[0])*180/math.pi
                    distance_obj = math.sqrt(relative_loc[0] ** 2 + relative_loc[1] ** 2)
                    relative_loc_str = f"{theta_obj:.1f}, {distance_obj:.1f}"
                top_k_relative_loc_str.append(relative_loc_str)
            output_string = " ".join(f"{label}: {loc_str}; prob: {prob:.2f}" for label, prob, loc_str in zip(top_k_labels, top_k_probs, top_k_relative_loc_str))
        else:
            ValueError("build_choice should be either 'prob' or 'loc'")
        
        return output_string

    @torch.no_grad()
    def generate_based_on_visual_cues(self, inputs_embeds, final_sequences, 
                                      attention_mask, pred_obj_prob, obj_masks,
                                      obj_labels=None, obj_locs=None,
                                      oracle_obj_prob=None, oracle_obj_loc=None,
                                      agent_pos_ori=[[0,0,0], [0,0,0]],
                                      num_beams=5, use_nucleus_sampling=False, top_p=0.9, 
                                      repetition_penalty=3.0, length_penalty=1, num_captions=1, temperature=1,device='cuda',data_dict=None):
        bs = inputs_embeds.shape[0]
        if pred_obj_prob.shape[1] == 1:
            pred_obj_prob = pred_obj_prob.squeeze(1)
        device = self.device
        model = self.llm_model
        tokenizer = self.llm_tokenizer
        output_after_visual_cues = []
        output_with_visual_cues = []
        for i in range(bs):
            new_token_type_ids = None
            sequence = final_sequences[i]
            list_obj_prob_idx = (sequence == LIST_OBJ_PROB_TOKEN_ID).nonzero(as_tuple=True)[0]
            list_obj_loc_idx = (sequence == LIST_OBJ_LOC_TOKEN_ID).nonzero(as_tuple=True)[0]
            list_obj_loc_plr_idx = (sequence == LIST_OBJ_LOC_PLR_TOKEN_ID).nonzero(as_tuple=True)[0]
            list_highlight_obj_idx = (sequence == HIGHLIGHT_OBJ_TOKEN_ID).nonzero(as_tuple=True)[0]
            print(f"LIST_OBJ_PROB_TOKEN_ID: {LIST_OBJ_PROB_TOKEN_ID}")
            print(f"sequence: {sequence}")
            if list_obj_prob_idx.numel() > 0 and list_obj_loc_idx.numel() == 0 and list_obj_loc_plr_idx.numel() == 0 and list_highlight_obj_idx.numel() == 0:
                list_obj_prob_idx = list_obj_prob_idx[0]
                if self.cot_mask_observation:
                    obj_prob_content = ""
                elif self.use_oracle_obj_content:
                    obj_prob_content = oracle_obj_prob[i]
                    print(f'obj_prob_oracle_content: {obj_prob_content}')
                else:
                    obj_prob_content = self.build_obj_prob_loc_txt(pred_obj_prob[i], obj_labels=obj_labels[i], obj_masks=obj_masks[i], obj_locs=obj_locs[i], build_choice='prob', device=device)
                obj_prob = f"\n{COT_INDICATORS_TOKENIZE['<obj_prob>']}{obj_prob_content}{COT_INDICATORS_TOKENIZE['</obj_prob>']}"
                obj_prob_tokenized = tokenizer(obj_prob, return_tensors='pt', add_special_tokens=False)
                obj_prob_tokens = obj_prob_tokenized.input_ids[0].to(device)
                obj_prob_inputs_embeds = model.get_input_embeddings()(obj_prob_tokens)
                obj_prob_attention_mask = obj_prob_tokenized.attention_mask[0].to(device)
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)   
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_obj_prob_idx+1], obj_prob_inputs_embeds] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_obj_prob_idx+1], obj_prob_attention_mask], dim=0)
                new_seqeunce = torch.cat([sequence[:list_obj_prob_idx+1], obj_prob_tokens], dim=0)    # do not contain the inputs embeds
                # print(f"new_seqeunce: {new_seqeunce}")
                visual_cues_content = obj_prob
            elif list_obj_loc_idx.numel() > 0 and list_obj_prob_idx.numel() == 0 and list_obj_loc_plr_idx.numel() == 0 and list_highlight_obj_idx.numel() == 0:
                list_obj_loc_idx = list_obj_loc_idx[0]
                if self.cot_mask_observation:
                    obj_loc_content = ""
                elif self.use_oracle_obj_content:
                    obj_loc_content = oracle_obj_loc[i]
                else:
                    obj_loc_content = self.build_obj_prob_loc_txt(pred_obj_prob[i], obj_labels=obj_labels[i], obj_masks=obj_masks[i], obj_locs=obj_locs[i], build_choice='loc', agent_pos_ori=agent_pos_ori[i], device=device, coord_type='rectangle')
                obj_loc = f"\n{COT_INDICATORS_TOKENIZE['<obj_loc_prob>']}{obj_loc_content}{COT_INDICATORS_TOKENIZE['</obj_loc_prob>']}" 
                obj_loc_tokenized = tokenizer(obj_loc, return_tensors='pt', add_special_tokens=False)
                obj_loc_tokens = obj_loc_tokenized.input_ids[0].to(device)
                obj_loc_inputs_embeds = model.get_input_embeddings()(obj_loc_tokens)
                obj_loc_attention_mask = obj_loc_tokenized.attention_mask[0].to(device)
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)   
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_obj_loc_idx+1], obj_loc_inputs_embeds] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_obj_loc_idx+1], obj_loc_attention_mask], dim=0)
                new_seqeunce = torch.cat([sequence[:list_obj_loc_idx+1], obj_loc_tokens], dim=0)  # do not contain the inputs embeds
                visual_cues_content = obj_loc
            elif list_obj_loc_plr_idx.numel() > 0 and list_obj_prob_idx.numel() == 0 and list_obj_loc_idx.numel() == 0 and list_highlight_obj_idx.numel() == 0:
                list_obj_loc_plr_idx = list_obj_loc_plr_idx[0]
                if self.cot_mask_observation:
                    obj_loc_content = ""
                elif self.use_oracle_obj_content:
                    obj_loc_content = oracle_obj_loc[i]
                else:
                    obj_loc_content = self.build_obj_prob_loc_txt(pred_obj_prob[i], obj_labels=obj_labels[i], obj_masks=obj_masks[i], obj_locs=obj_locs[i], build_choice='loc', agent_pos_ori=agent_pos_ori[i], device=device, coord_type='polar')
                obj_loc = f"\n{COT_INDICATORS_TOKENIZE['<obj_loc_plr_prob>']}{obj_loc_content}{COT_INDICATORS_TOKENIZE['</obj_loc_plr_prob>']}"
                obj_loc_tokenized = tokenizer(obj_loc, return_tensors='pt', add_special_tokens=False)
                obj_loc_tokens = obj_loc_tokenized.input_ids[0].to(device)
                obj_loc_inputs_embeds = model.get_input_embeddings()(obj_loc_tokens)
                obj_loc_attention_mask = obj_loc_tokenized.attention_mask[0].to(device)
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_obj_loc_plr_idx+1], obj_loc_inputs_embeds] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_obj_loc_plr_idx+1], obj_loc_attention_mask], dim=0)
                new_seqeunce = torch.cat([sequence[:list_obj_loc_plr_idx+1], obj_loc_tokens], dim=0)  # do not contain the inputs embeds
                visual_cues_content = obj_loc
            elif list_highlight_obj_idx.numel() > 0 and list_obj_prob_idx.numel() == 0 and list_obj_loc_idx.numel() == 0 and list_obj_loc_plr_idx.numel() == 0:
                list_highlight_obj_idx = list_highlight_obj_idx[0]
                if self.cot_mask_observation:
                    # Keep the learned image boundary, but remove every image
                    # patch embedding. This preserves the second generation
                    # stage without exposing any highlighted-object evidence.
                    img_embeds = None
                else:
                    # obj_id = data_dict['obj_ids'][0]  # attribute/description only has one object
                    # img = data_dict['obj_imgs'][obj_id]
                    sorted_indices = torch.argsort(pred_obj_prob[i][obj_masks[i]], descending=True)
                    if hasattr(self, 'max_cot_obj_num'):
                        obj_max_num = self.max_cot_obj_num
                    top_k_indices = sorted_indices[:obj_max_num]
                    selected_object_idx = int(top_k_indices[0].item())
                    img_embeds = self._encode_visual_cue_image(
                        data_dict,
                        i,
                        image_field='all_obj_imgs',
                        path_field='all_obj_img_paths',
                        object_idx=selected_object_idx,
                    )
                img_start_embeds = model.get_input_embeddings()(torch.tensor(self.img_start_token_id).to(device)).unsqueeze(0).unsqueeze(0)   # (1, 1, D)
                img_end_embeds = model.get_input_embeddings()(torch.tensor(self.img_end_token_id).to(device)).unsqueeze(0).unsqueeze(0)   # (1, 1, D)
                image_parts = [img_start_embeds, img_end_embeds]
                if img_embeds is not None:
                    image_parts.insert(1, img_embeds)
                img_embeds_with_pre_suffix = torch.cat(image_parts, dim=1)
                img_masks = torch.ones(img_embeds_with_pre_suffix.shape[:2], dtype=torch.int64, device=device) # (1, 576)
                img_embeds_with_pre_suffix = img_embeds_with_pre_suffix.squeeze(0)
                img_masks = img_masks.squeeze(0)
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_highlight_obj_idx+1], img_embeds_with_pre_suffix] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_highlight_obj_idx+1], img_masks], dim=0)
                if self.is_gemma3:
                    new_token_type_ids = torch.zeros_like(new_attention_mask, dtype=torch.long)
                    image_block_start = new_attention_mask.shape[0] - img_embeds_with_pre_suffix.shape[0]
                    new_token_type_ids[image_block_start + 1:-1] = 1
                new_seqeunce = sequence[:list_highlight_obj_idx+1]   # remove the content after highlight_obj
            else:
                new_inputs_embeds = inputs_embeds[i]
                new_attention_mask = attention_mask[i]
                visual_cues_content = ''
                new_seqeunce = sequence
            generation_kwargs = {}
            if new_token_type_ids is not None:
                generation_kwargs['token_type_ids'] = new_token_type_ids.unsqueeze(0)
            new_outputs = model.generate(
                    inputs_embeds=new_inputs_embeds.unsqueeze(0).to(self.llm_dtype),
                    attention_mask=new_attention_mask.unsqueeze(0),
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=128,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    **generation_kwargs,
                    )   
            new_txt_output = tokenizer.decode(new_outputs.sequences[0], skip_special_tokens=True)   # extactly the same output with batch decode
            new_output_with_visual_cues = tokenizer.decode(new_seqeunce, skip_special_tokens=True)
            print(f'output with visual cues: {new_output_with_visual_cues}')
            print(f'output after visual cues: {new_txt_output}')
            output_after_visual_cues.append(new_txt_output)
            output_with_visual_cues.append(new_output_with_visual_cues)

        return output_with_visual_cues, output_after_visual_cues




    # Inference pipeline
    # 1. tokenize input question
    # 2. encode 3d scene tokens
    # 3: build LLM input embeddings (prompt + scene tokens)
    # 4: LLM generate first response
    # 5: grounding head computes the grounding loss and pred_obj_prob from the query in the generated <think_grd> and the object tokens
    # 6: build visual cues from pred_obj_prob and predefined templates, then regenerate the answer with them
    # 7: return the final generated answer
    
    @torch.no_grad()
    def generate(
        self,
        data_dict,
        use_nucleus_sampling=False,
        num_beams=5,
        top_p=0.9,
        repetition_penalty=3.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        """
        data_dict requires the same keys as forward() except output_gt
        """
        device = self.device
        bs = len(data_dict['input_txt'])

        # get embedding of input_txt
        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'left'
        input_txt_tokenized = self.llm_tokenizer(
            data_dict['input_txt'],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        ).to(device)   # [BOS, tokens, PAD]
        data_dict['input_txt_ids'] = input_txt_tokenized.input_ids
        data_dict['input_txt_embed'] = self.llm_model.get_input_embeddings()(input_txt_tokenized.input_ids)
        data_dict['input_txt_mask'] = input_txt_tokenized.attention_mask

        if 'scene_tokens' not in data_dict:
            if self.blind:
                data_dict['scene_tokens'] = torch.zeros(bs, 1, data_dict['input_txt_embed'].shape[-1]).to(device)
                data_dict['scene_masks'] = torch.zeros(bs, 1, dtype=torch.bool).to(device)
            else:
                data_dict = self.pcd_encoder(data_dict)
                data_dict['scene_tokens'] = self.pcd_proj(data_dict['scene_tokens'].to(device))

        inputs_embeds, attention_mask = self.build_input_sequence(data_dict=data_dict)


        # NOTE: stage 1: LLM generates the first response
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds.to(self.llm_dtype),
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=self.max_out_len,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

        final_sequences = outputs.sequences
        # grounding
        if hasattr(self, 'grounding_head'):
            hidden_states = getattr(outputs, 'hidden_states', None)  # step x layer x (bs x beam) x token_num x hidden_dim
            if hidden_states is None:
                predicted_obj_prob = torch.zeros(bs, data_dict['obj_masks'].shape[1], device=device)
                data_dict['pred_obj_prob'] = predicted_obj_prob
            else:
                decoded_sequences = []
                for seq in final_sequences:
                    decoded_sequences.append(self.llm_tokenizer.decode(seq, skip_special_tokens=True))

                batch_size = final_sequences.size(0)  
                hidden_size = hidden_states[0][-1].size(-1)
                predicted_tokens, predicted_sequences = predict_with_beam_search(self.llm_model, hidden_states, self.llm_tokenizer, num_beams=num_beams, batch_size=batch_size, device=device, repetition_penalty=repetition_penalty)
                for i in range(batch_size):
                    eos_token_id = self.llm_tokenizer.eos_token_id
                    if len(torch.nonzero(predicted_tokens[i] == eos_token_id)) == 0:
                        end_idx = predicted_tokens[i].shape[0]
                    else:
                        end_idx = torch.nonzero(predicted_tokens[i] == eos_token_id)[0][0]
                    predicted_tokens[i, end_idx+1:] = -100
                beam_indices = outputs.beam_indices if hasattr(outputs, 'beam_indices') else None
                hidden_states = outputs.hidden_states
                last_hidden_states = []
                for i in range(min(len(hidden_states), beam_indices.shape[1])):
                    last_hidden_states.append(hidden_states[i][-1][beam_indices[:, i], -1, :])
                last_hidden_states = torch.stack(last_hidden_states, dim=1)
                predicted_tokens = predicted_tokens.to(device)
                predicted_tokens = predicted_tokens[:, :beam_indices.shape[1]] 
                
                if self.grd_use_scene_tokens:
                    # pq3d
                    obj_tokens = data_dict['scene_tokens']
                else:
                    data_dict = self.obj_encoder(data_dict)
                    obj_tokens = self.pcd_proj(data_dict['obj_tokens'])
                obj_masks = data_dict['obj_masks']
                self.grounding_head = self.grounding_head.to(self.device)
                last_hidden_states = last_hidden_states.float() 
                predicted_obj_prob = self.grounding_head(input_ids=predicted_tokens, hidden_states=last_hidden_states, obj_tokens=obj_tokens, obj_mask_token_idx=self.grd_token_id, obj_masks=obj_masks, gt_grounding_obj_mask=data_dict['grounding_obj_mask_gt'], llm_model=self.llm_model,device=self.device)
                data_dict['pred_obj_prob'] = predicted_obj_prob
                # print(f'predicted_obj_prob: {predicted_obj_prob}')
                if hasattr(self, 'moe_flag') and self.moe_flag:
                    question_type = data_dict['type']
                    if question_type in self.moe_type_list:
                        question_index = str(data_dict['index'])
                        predicted_obj_prob = self.obj_prob_dict[question_index]
                        print(f"$$$$$$$$$$$$$$$$$$$$$$$$")
                        print("Successfully load the predicted_obj_prob from the dictionary")

        if self.llm_tokenizer.unk_token_id is not None:
            final_sequences[final_sequences == self.llm_tokenizer.unk_token_id] = self.llm_tokenizer.eos_token_id
        # data_dict['output_tokens'] = outputs   # unable to gather variable-length tensors
        if hasattr(self, 'grounding_head') and hasattr(self, 'cot_mask_obj_prob_loc_token') and self.cot_mask_obj_prob_loc_token:

            # NOTE: stage 2: build visual cues from pred_obj_prob and predefined templates, then regenerate the answer
            output_with_visual_cues, output_after_visual_cues = self.generate_based_on_visual_cues(inputs_embeds, 
            final_sequences, attention_mask, predicted_obj_prob, data_dict['obj_masks'],
            obj_labels=data_dict['gt_labels'], obj_locs=data_dict['obj_locs'], 
            oracle_obj_prob=data_dict['oracle_obj_prob'], oracle_obj_loc=data_dict['oracle_obj_loc'],
            agent_pos_ori=data_dict['agent_pos_ori'],
            num_beams=num_beams, use_nucleus_sampling=use_nucleus_sampling, 
            top_p=top_p, repetition_penalty=repetition_penalty, 
            length_penalty=length_penalty, num_captions=num_captions, 
            temperature=temperature,data_dict=data_dict)
            output_txt_list = []
            for i in range(bs):
                output_txt = replace_cot_tokens_with_indicators(output_with_visual_cues[i]+output_after_visual_cues[i])   # replace the cot tokens with indicators
                output_txt_list.append(output_txt.strip())
            data_dict['output_txt'] = output_txt_list
        elif hasattr(self, 'grounding_head') and hasattr(self, 'cot_mask_obj_prob_loc_token') and not self.cot_mask_obj_prob_loc_token:
            output_txt_raw = self.llm_tokenizer.batch_decode(final_sequences, skip_special_tokens=True)
            output_txt = []
            for i in range(bs):
                output_txt.append(replace_cot_tokens_with_indicators(output_txt_raw[i]))
            data_dict['output_txt'] = output_txt
        else:
            output_txt = self.llm_tokenizer.batch_decode(final_sequences, skip_special_tokens=True)
            output_txt = [txt.strip() for txt in output_txt]
            data_dict['output_txt'] = output_txt
            # if hasattr(self, 'cot_answer_training') and self.cot_answer_training:
            #     for i in range(bs):
            #         output_txt = replace_cot_tokens_with_indicators(output_txt[i], COT_INDICATORS_TOKENIZE)

        return data_dict
