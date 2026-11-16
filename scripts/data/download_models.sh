#!/usr/bin/env bash
# Download the base VLM and the pretrained 3D-encoder weights into ${SCENECOT_MODEL_ROOT}.
#
#   bash scripts/data/download_models.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

MODEL_ROOT="${SCENECOT_MODEL_ROOT}"
mkdir -p "${MODEL_ROOT}"

# Base agent: Qwen3-VL-4B-Instruct (revision used in the paper).
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --revision "${QWEN_REVISION:-ebb281ec70b05090aa6165b016eac8ec08e71b17}" \
  --local-dir "${MODEL_ROOT}/qwen3-vl-4b-instruct"

# PQ3D: PointNet++ tokenizer, Query3D (SceneVerse) pre-training and the CLIP text encoder.
huggingface-cli download EricLHK/SceneCOT \
  --revision "${SCENECOT_MODEL_REVISION:-be06ebc5e1faf3c0a88010ada348ad72146e5481}" \
  --include "pointnet_tokenizer.pth" "query3d_pretrain/*" "clip-vit-large-patch14/*" \
  --exclude "clip-vit-large-patch14/pytorch_model.bin" \
  --local-dir "${MODEL_ROOT}"

require_file \
  "${QWEN_MODEL_PATH}/config.json" \
  "${SCENECOT_POINTNET_TOKENIZER_PATH}" \
  "${SCENECOT_QUERY3D_PRETRAIN_PATH}" \
  "${SCENECOT_PQ3D_TOKENIZER_PATH}/config.json"
echo "Model assets OK: ${MODEL_ROOT}"
