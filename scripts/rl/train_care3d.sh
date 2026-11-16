#!/usr/bin/env bash
# CARE-3D GRPO post-training of Qwen3-VL-4B on 2x H100 80GB (paper Table 8).
#
#   export GEMINI_API_KEY=...          # answer judge used for the reward (see README)
#   bash scripts/rl/train_care3d.sh [extra hydra overrides]
#
# Init: SFT LoRA adapter (actor) + frozen SFT grounder (tool). Trainable: actor LoRA only.
# AdamW (0.9, 0.999), lr 1e-6 (constant), wd 0.01, KL 0.02 (low-var-kl), bf16, 600 GRPO steps,
# 24 questions x 8 rollouts per step (6 free + 2 GCS coverage rollouts, 1 coverage after step 200),
# COAC tool credit with group z-score (text + visual observations).
# Checkpoints: checkpoints/${EXPERIMENT_NAME}/global_step_*/actor
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-care3d_qwen3vl4b}"
export SCENECOT_SFT_ADAPTER="${SCENECOT_SFT_ADAPTER:-${SFT_EXPORT_DIR}/lora_adapter}"
export SCENECOT_SFT_CKPT="${SCENECOT_SFT_CKPT:-${SFT_EXPORT_DIR}/grounder/best.pth/pytorch_model.bin}"
export TRAIN_PARQUET="${TRAIN_PARQUET:-${RL_PARQUET_DIR}/msqa/train.parquet}"
export VAL_PARQUET="${VAL_PARQUET:-${RL_PARQUET_DIR}/msqa/val_msqa_20_per_type.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCENECOT_EXP_ROOT}/rl/${EXPERIMENT_NAME}}"

# ---- optimisation / batch (Table 8) ----
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-600}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export LORA_RANK=16 LORA_ALPHA=32
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
export AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"   # each worker hosts one grounding tool
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.30}"
export ROLLOUT_CKPT_BUCKET_MB="${ROLLOUT_CKPT_BUCKET_MB:-2048}"
# Safe with weight updates: verl resets the vLLM prefix cache after every weight sync.
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"

# ---- GCS: mode-coverage sampling inside the rollout group ----
export SCENECOT_GCS_COVERAGE_NUM="${SCENECOT_GCS_COVERAGE_NUM:-2}"
export SCENECOT_GCS_COVERAGE_POLICY="${SCENECOT_GCS_COVERAGE_POLICY:-least_used}"
export SCENECOT_GCS_COVERAGE_SEED="${SCENECOT_GCS_COVERAGE_SEED:-42}"
export SCENECOT_GCS_COVERAGE_WARMUP_STEP="${SCENECOT_GCS_COVERAGE_WARMUP_STEP:-0}"
export SCENECOT_GCS_COVERAGE_WARMUP_NUM="${SCENECOT_GCS_COVERAGE_WARMUP_NUM:-0}"
export SCENECOT_GCS_COVERAGE_ANNEAL_STEP="${SCENECOT_GCS_COVERAGE_ANNEAL_STEP:-200}"
export SCENECOT_GCS_COVERAGE_ANNEAL_NUM="${SCENECOT_GCS_COVERAGE_ANNEAL_NUM:-1}"
export SCENECOT_GCS_COVERAGE_MODE_POOL="${SCENECOT_GCS_COVERAGE_MODE_POOL:-prob,loc_rect,loc_polar,highlight}"
export SCENECOT_MODE_STRATIFIED_ROLLOUT=0
export SCENECOT_DIAGNOSTIC_ROLLOUT_ONLY=0

# ---- COAC: counterfactual observation credit for the tool call ----
export SCENECOT_COAC_ENABLE="${SCENECOT_COAC_ENABLE:-1}"
export SCENECOT_COAC_GROUP_ZSCORE="${SCENECOT_COAC_GROUP_ZSCORE:-1}"
export SCENECOT_COAC_VISUAL_ENABLE="${SCENECOT_COAC_VISUAL_ENABLE:-1}"
export SCENECOT_COAC_COUNTERFACTUAL_MODE="${SCENECOT_COAC_COUNTERFACTUAL_MODE:-mask}"
export SCENECOT_COAC_COPY_DISCOUNT_ENABLE="${SCENECOT_COAC_COPY_DISCOUNT_ENABLE:-0}"
export SCENECOT_COAC_FALLBACK_CREDIT="${SCENECOT_COAC_FALLBACK_CREDIT:-0.5}"

# ---- reward ----
export SCENECOT_REWARD_MAX="${SCENECOT_REWARD_MAX:-2.2}"
export SCENECOT_ENABLE_AGENT_REWARD="${SCENECOT_ENABLE_AGENT_REWARD:-1}"
export SCENECOT_ENABLE_PROGRESS_SWITCH="${SCENECOT_ENABLE_PROGRESS_SWITCH:-1}"
export SCENECOT_PROGRESS_FORMAT_REWARD_ENABLE="${SCENECOT_PROGRESS_FORMAT_REWARD_ENABLE:-0}"
export SCENECOT_AUTOCLOSE_FINAL_ANSWER="${SCENECOT_AUTOCLOSE_FINAL_ANSWER:-1}"
export SCENECOT_ANSWER_JUDGE_BACKEND="${SCENECOT_ANSWER_JUDGE_BACKEND:-gemini}"
export SCENECOT_ANSWER_JUDGE_MODEL="${SCENECOT_ANSWER_JUDGE_MODEL:-gemini-2.5-flash-lite}"
export SCENECOT_PC_TYPE=gt

export TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"

require_file "${SCENECOT_SFT_ADAPTER}/adapter_config.json" "${SCENECOT_SFT_ADAPTER}/adapter_model.safetensors"
if [[ "${PRINT_HYDRA_CMD_ONLY:-0}" != 1 ]]; then
  if [[ "${SCENECOT_ANSWER_JUDGE_BACKEND}" == gemini && -z "${GEMINI_API_KEY:-}" ]]; then
    echo "Export GEMINI_API_KEY (or set SCENECOT_ANSWER_JUDGE_BACKEND=rule / use run_with_local_qwen_judge.py)." >&2
    exit 1
  fi
  if [[ "${SCENECOT_ANSWER_JUDGE_BACKEND}" != rule ]]; then
    "${PYTHON_BIN}" scripts/rl/check_answer_judge.py   # one real judge request before allocating GPUs
  fi
fi
mkdir -p "${OUTPUT_DIR}"

bash scripts/rl/launch_verl.sh \
  actor_rollout_ref.model.lora_adapter_path="${SCENECOT_SFT_ADAPTER}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.resume_mode="${RESUME_MODE:-auto}" \
  trainer.val_before_train=False \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  trainer.test_freq="${TEST_FREQ:-100}" \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-2}" \
  trainer.log_val_generations=5 \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  "$@" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
