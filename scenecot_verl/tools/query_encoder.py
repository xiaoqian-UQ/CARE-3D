from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


# Qwen-VL and modern Hugging Face LLaVA use the first key.  The remaining
# candidates keep the loader useful for older LLaVA/Llama checkpoints.
EMBEDDING_KEY_CANDIDATES = (
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def torch_dtype_from_string(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        if dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError(f"Unsupported SceneCOT tool dtype: {dtype}.")
        return dtype
    normalized = str(dtype).lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "SceneCOT tool dtype must be one of bf16/bfloat16, fp16/float16, "
            f"or fp32/float32; got {dtype!r}."
        ) from exc


def _tensor_from_mapping(state: Mapping[str, object], source: Path) -> torch.Tensor:
    nested = state.get("state_dict")
    if not isinstance(nested, Mapping):
        nested = state.get("model")
    if isinstance(nested, Mapping):
        state = nested
    for key in EMBEDDING_KEY_CANDIDATES:
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float()
    raise KeyError(
        f"No input embedding tensor found in {source}. Tried: "
        + ", ".join(EMBEDDING_KEY_CANDIDATES)
    )


def _load_safetensors_file(path: Path) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in EMBEDDING_KEY_CANDIDATES:
            if key in keys:
                return handle.get_tensor(key).float()
    raise KeyError(
        f"No input embedding tensor found in {path}. Tried: "
        + ", ".join(EMBEDDING_KEY_CANDIDATES)
    )


def _load_indexed_weight(model_dir: Path, index_path: Path) -> torch.Tensor:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})
    for key in EMBEDDING_KEY_CANDIDATES:
        shard_name = weight_map.get(key)
        if not shard_name:
            continue
        shard_path = model_dir / shard_name
        if shard_path.suffix == ".safetensors":
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                return handle.get_tensor(key).float()
        state = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError(f"Unsupported checkpoint payload type in {shard_path}: {type(state).__name__}")
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float()
        return _tensor_from_mapping(state, shard_path)
    raise KeyError(
        f"No input embedding entry found in {index_path}. Tried: "
        + ", ".join(EMBEDDING_KEY_CANDIDATES)
    )


def load_embedding_weight(model_path_or_ckpt: str | Path) -> torch.Tensor:
    """Load input embeddings from a local Qwen-VL, HF LLaVA, or Llama checkpoint."""
    source = Path(model_path_or_ckpt).expanduser()
    if source.is_file():
        if source.suffix == ".safetensors":
            return _load_safetensors_file(source)
        state = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError(f"Unsupported checkpoint payload type in {source}: {type(state).__name__}")
        return _tensor_from_mapping(state, source)

    if not source.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {source}")

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = source / index_name
        if index_path.is_file():
            return _load_indexed_weight(source, index_path)

    for filename in ("model.safetensors", "pytorch_model.bin"):
        weight_path = source / filename
        if not weight_path.is_file():
            continue
        if weight_path.suffix == ".safetensors":
            return _load_safetensors_file(weight_path)
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError(f"Unsupported checkpoint payload type in {weight_path}: {type(state).__name__}")
        return _tensor_from_mapping(state, weight_path)

    raise FileNotFoundError(
        f"No supported Hugging Face weight file found under {source}; expected a "
        "safetensors or pytorch_model.bin file/index."
    )


class AverageEmbeddingQueryEncoder:
    """Encode grounding text by averaging the actor model's frozen token embeddings."""

    def __init__(
        self,
        model_path_or_ckpt: str,
        *,
        hidden_dim: int = 2560,
        device: str = "cuda",
        dtype: str | torch.dtype = "bf16",
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.device = torch.device(device)
        self.dtype = torch_dtype_from_string(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_ckpt, trust_remote_code=True)
        weight = load_embedding_weight(model_path_or_ckpt)
        if weight.ndim != 2 or int(weight.shape[1]) != self.hidden_dim:
            raise ValueError(
                f"Input embedding shape must be (vocab, {self.hidden_dim}), got {tuple(weight.shape)} "
                f"from {model_path_or_ckpt}."
            )
        self.embedding = torch.nn.Embedding.from_pretrained(weight.to(self.dtype), freeze=True).to(self.device)
        self.embedding.eval()

    @torch.no_grad()
    def encode(self, query_text: str) -> torch.Tensor:
        if query_text is None or not str(query_text).strip():
            raise ValueError("Grounding query text is empty.")
        encoded = self.tokenizer(str(query_text).strip(), add_special_tokens=False, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        if input_ids.numel() == 0:
            raise ValueError("Grounding query tokenized to zero tokens.")
        embeds = self.embedding(input_ids).float()
        query = embeds.mean(dim=1).squeeze(0)
        if query.shape != (self.hidden_dim,):
            raise ValueError(f"Query vector shape must be ({self.hidden_dim},), got {tuple(query.shape)}.")
        return query
