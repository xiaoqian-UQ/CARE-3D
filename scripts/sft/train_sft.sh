#!/usr/bin/env bash
# SFT (paper Table 7): jointly train the agent LoRA, the PQ3D 3D encoder / projector
# and the grounding head on MSQA + GQA3D CoT data.
#   AdamW (0.9, 0.999), lr 2e-5 (one shared group), wd 0.05, 400-step linear warm-up + cosine
#   decay, 5 epochs, bf16, 2x H100 80GB, per-GPU batch 2 x grad-accum 5 (effective batch 20).
#
#   bash scripts/sft/train_sft.sh [extra hydra overrides]
# Output: ${SFT_DIR}/best.pth/pytorch_model.bin (+ config.yaml). Re-running resumes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export SCENECOT_LLM_PATH="${QWEN_MODEL_PATH}"
export SCENECOT_COT_TOKEN_SET=qwen   # Qwen3-VL ids for the SceneCOT action tokens
NUM_GPUS="${NUM_GPUS:-2}"

require_file "${QWEN_MODEL_PATH}/config.json" "${SCENECOT_POINTNET_TOKENIZER_PATH}" \
  "${SCENECOT_QUERY3D_PRETRAIN_PATH}" "${SCENECOT_PQ3D_TOKENIZER_PATH}/config.json"

"${PYTHON_BIN}" launch.py --mode accelerate --gpu_per_node "${NUM_GPUS}" --num_nodes 1 \
  --mixed_precision bf16 --cpu_per_task 10 --port "${MASTER_PORT:-2021}" \
  --config configs/default_cot.yaml \
  model=SceneCOTAgent \
  task=scenecot_scanent_msqa_gqa3d \
  note=sft_qwen3vl4b \
  llm=qwen3-vl-4b-instruct \
  llm.max_out_len=768 \
  llm.max_context_len=1024 \
  vision3d.name=PQ3D \
  vision3d.use_embodied_token=False \
  grounding.enable=True \
  grounding.hidden_dim=2560 \
  grounding.loss_type=ce \
  grounding.grd_loss_weight=0.1 \
  grounding.grd_text_hidden_states=average_embedding \
  grounding.use_region_mask=True \
  cot.mask_obj_prob_loc_token=True \
  cot.use_oracle_obj_content=False \
  cot.cot_no_scene_tokens=True \
  cot.gqa3d_val_set_size=500 \
  cot.msr3d_val_set_size=1000 \
  task.training.lr=2e-5 \
  task.training.epochs=5 \
  task.leomix.LeoMix.debug_size=40000000 \
  training.gradient_accumulation_steps=5 \
  data.cotqa.msr3d.pc_type=pred \
  data.cotqa.gqa3d.pc_type=gt \
  data.cotqa.use_pred_for_train=False \
  data.nms_iou_threshold=1 \
  dataloader.train.batchsize=2 \
  dataloader.train.num_workers=10 \
  dataloader.eval.batchsize=2 \
  dataloader.eval.num_workers=4 \
  eval.val_interval=1 \
  "$@"
