#!/usr/bin/env bash
# Package the SFT result for GRPO:
#   ${SFT_EXPORT_DIR}/lora_adapter/   PEFT adapter that initialises the actor (vLLM + FSDP)
#   ${SFT_EXPORT_DIR}/grounder/       full SFT checkpoint + config.yaml, loaded by the frozen
#                                     grounding tool (PQ3D encoder, projector, grounding head)
#
#   bash scripts/sft/export_for_rl.sh [path/to/best.pth]   (default: ${SFT_DIR}/best.pth)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

SFT_CKPT_DIR="${1:-${SFT_DIR}/best.pth}"
SFT_CONFIG="$(dirname "${SFT_CKPT_DIR}")/config.yaml"
require_file "${SFT_CKPT_DIR}/pytorch_model.bin" "${SFT_CONFIG}"

"${PYTHON_BIN}" scripts/sft/convert_sft_checkpoint_to_lora_adapter.py \
  --checkpoint "${SFT_CKPT_DIR}/pytorch_model.bin" \
  --output-dir "${SFT_EXPORT_DIR}/lora_adapter" \
  --base-model-name-or-path "${QWEN_MODEL_PATH}" \
  --lora-alpha 32

# The grounding tool locates config.yaml in a parent directory of the checkpoint file
# (it needs e.g. vision3d.use_embodied_token=False), so keep the two together.
mkdir -p "${SFT_EXPORT_DIR}/grounder/best.pth"
ln -sfn "$(cd "${SFT_CKPT_DIR}" && pwd)/pytorch_model.bin" "${SFT_EXPORT_DIR}/grounder/best.pth/pytorch_model.bin"
cp "${SFT_CONFIG}" "${SFT_EXPORT_DIR}/grounder/config.yaml"
echo "SFT export ready: ${SFT_EXPORT_DIR}"
