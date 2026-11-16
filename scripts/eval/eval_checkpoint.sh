#!/usr/bin/env bash
# Greedy evaluation of a LoRA policy on the MSQA and GQA3D (Beacon3D) test sets with the full
# agent protocol (grounding tool in the loop). Works for the SFT adapter and for GRPO checkpoints.
#
#   # GRPO checkpoint (the LoRA adapter is extracted from the FSDP actor checkpoint first)
#   bash scripts/eval/eval_checkpoint.sh --actor-dir checkpoints/care3d_qwen3vl4b/global_step_600/actor --name care3d_step600
#   # any PEFT adapter, e.g. the SFT model
#   bash scripts/eval/eval_checkpoint.sh --adapter outputs/sft_export/lora_adapter --name sft
#
# Outputs: outputs/eval/<name>/<dataset>/0.jsonl (raw rollouts) and
#          outputs/eval_legacy/<name>_<dataset>_test/eval_results/<DATASET>/results.json
# Score them with scripts/eval/score_gpt.sh <name>.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

ADAPTER="" ACTOR_DIR="" NAME="" DATASETS="${DATASETS:-msqa gqa3d}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter) ADAPTER="$2"; shift 2 ;;
    --actor-dir) ACTOR_DIR="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${NAME}" ]] || { echo "--name is required" >&2; exit 2; }
if [[ -n "${ACTOR_DIR}" ]]; then
  ADAPTER_ROOT="${SCENECOT_EXP_ROOT}/merged/${NAME}"
  ADAPTER="${ADAPTER_ROOT}/lora_adapter"
  if [[ ! -f "${ADAPTER}/adapter_model.safetensors" ]]; then
    "${PYTHON_BIN}" scripts/rl/extract_fsdp_lora_adapter.py --actor-dir "${ACTOR_DIR}" \
      --target-dir "${ADAPTER_ROOT}" --rank 16 --alpha 32
  fi
fi
require_file "${ADAPTER}/adapter_config.json" "${ADAPTER}/adapter_model.safetensors"

# Evaluation protocol: greedy decoding (verl val_kwargs), one rollout, no COAC / GCS.
export SCENECOT_COAC_ENABLE=0 SCENECOT_COAC_GROUP_ZSCORE=0 SCENECOT_COAC_VISUAL_ENABLE=0
export SCENECOT_MODE_STRATIFIED_ROLLOUT=0
export SCENECOT_GCS_COVERAGE_NUM=0 SCENECOT_GCS_COVERAGE_WARMUP_NUM=0 SCENECOT_GCS_COVERAGE_ANNEAL_NUM=0
export SCENECOT_ENABLE_PROGRESS_SWITCH=0 SCENECOT_PROGRESS_FORMAT_REWARD_ENABLE=0
export SCENECOT_AUTOCLOSE_FINAL_ANSWER=1
export SCENECOT_ANSWER_JUDGE_BACKEND=rule   # rewards are only logged; scoring uses GPT-4o offline
export SCENECOT_VISUAL_CACHE_DIR="" SCENECOT_OBJECT_TOKEN_CACHE_ROOT="" SCENECOT_OBJECT_TOKEN_CACHE_MODE=off
export ROLLOUT_N=1 TOTAL_TRAINING_STEPS=1
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
export TRAIN_BATCH_SIZE="${N_GPUS_PER_NODE}" PPO_MINI_BATCH_SIZE="${N_GPUS_PER_NODE}" TRAIN_MAX_SAMPLES=2
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}" VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
export AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-8}"
export MM_PROCESSOR_MAX_PIXELS="${MM_PROCESSOR_MAX_PIXELS:-393216}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"

for DATASET in ${DATASETS}; do
  case "${DATASET}" in
    msqa) DATASET_NAME=QACOTScanNetMSR3D ;;
    gqa3d) DATASET_NAME=GQA3D ;;
    *) echo "Unknown dataset: ${DATASET}" >&2; exit 2 ;;
  esac
  OUT_DIR="${SCENECOT_EXP_ROOT}/eval/${NAME}/${DATASET}"
  TEST_PARQUET="${OUT_DIR}/test_default_prompt.parquet"
  mkdir -p "${OUT_DIR}"
  require_file "${RL_PARQUET_DIR}/${DATASET}/test.parquet"
  # Pin the evaluation prompt to build_parquet.DEFAULT_SYSTEM_PROMPT.
  "${PYTHON_BIN}" scripts/eval/update_parquet_system_prompt.py "${RL_PARQUET_DIR}/${DATASET}/test.parquet" \
    --output "${TEST_PARQUET}" --user-prompt-source keep
  NUM_ROWS="$("${PYTHON_BIN}" -c "import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))" "${TEST_PARQUET}")"

  TRAIN_PARQUET="${TEST_PARQUET}" VAL_PARQUET="${TEST_PARQUET}" EXPERIMENT_NAME="eval_${NAME}_${DATASET}" \
  RAY_TEMP_DIR="/tmp/ray_${USER:-user}_eval_$$_${DATASET}" \
  bash scripts/rl/launch_verl.sh \
    actor_rollout_ref.model.lora_adapter_path="${ADAPTER}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.device=cuda \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.validation_data_dir="${OUT_DIR}" \
    trainer.log_val_generations="${NUM_ROWS}" \
    2>&1 | tee "${OUT_DIR}/generate.log"
  [[ "${PRINT_HYDRA_CMD_ONLY:-0}" == 1 ]] && continue

  "${PYTHON_BIN}" scripts/eval/convert_verl_jsonl_to_msqa_results.py \
    --verl-jsonl "${OUT_DIR}/0.jsonl" \
    --val-parquet "${TEST_PARQUET}" \
    --model-name "${NAME}_${DATASET}_test" \
    --dataset-name "${DATASET_NAME}" \
    --output-root "${SCENECOT_EXP_ROOT}/eval_legacy"
done
echo "Done. Score with: bash scripts/eval/score_gpt.sh ${NAME}"
