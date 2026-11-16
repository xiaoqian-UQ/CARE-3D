# Shared environment for every CARE-3D script. Source it from the repository root:
#   source scripts/env.sh
# All paths can be overridden by exporting the variable before sourcing.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

# Data / model / output roots (relative to the repository root unless absolute).
export SCENECOT_DATA_ROOT="${SCENECOT_DATA_ROOT:-data_assets}"
export SCENECOT_MODEL_ROOT="${SCENECOT_MODEL_ROOT:-model_assets}"
export SCENECOT_EXP_ROOT="${SCENECOT_EXP_ROOT:-outputs}"

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${SCENECOT_MODEL_ROOT}/qwen3-vl-4b-instruct}"
export SCENECOT_PQ3D_TOKENIZER_PATH="${SCENECOT_PQ3D_TOKENIZER_PATH:-${SCENECOT_MODEL_ROOT}/clip-vit-large-patch14}"
export SCENECOT_POINTNET_TOKENIZER_PATH="${SCENECOT_POINTNET_TOKENIZER_PATH:-${SCENECOT_MODEL_ROOT}/pointnet_tokenizer.pth}"
export SCENECOT_QUERY3D_PRETRAIN_PATH="${SCENECOT_QUERY3D_PRETRAIN_PATH:-${SCENECOT_MODEL_ROOT}/query3d_pretrain/pytorch_model.bin}"

# Default experiment locations produced by the pipeline.
export SFT_DIR="${SFT_DIR:-${SCENECOT_EXP_ROOT}/CARE3D_sft_qwen3vl4b}"
export SFT_EXPORT_DIR="${SFT_EXPORT_DIR:-${SCENECOT_EXP_ROOT}/sft_export}"
export RL_PARQUET_DIR="${RL_PARQUET_DIR:-${SCENECOT_DATA_ROOT}/verl_parquet}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_file() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required input: ${path}" >&2
      return 1
    fi
  done
}
