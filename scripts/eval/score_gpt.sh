#!/usr/bin/env bash
# GPT-Score (gpt-4o-2024-11-20, 1-5 rubric) for predictions produced by eval_checkpoint.sh.
#   MSQA:     per-type and overall GPT-Score
#   Beacon3D: Case (mean score) and Obj. (all three questions of an object scored >= 75)
#
#   export OPENAI_API_KEY=...
#   bash scripts/eval/score_gpt.sh <name> [msqa gqa3d]
# Scoring resumes from eval_results_with_gpt_score.json if interrupted.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

NAME="${1:?usage: score_gpt.sh <name> [datasets...]}"; shift
DATASETS="${*:-msqa gqa3d}"
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "Export OPENAI_API_KEY first." >&2; exit 1; }

RESULT_ROOT="$(cd "${SCENECOT_EXP_ROOT}" && pwd)/eval_legacy"
export SCENECOT_EVAL_ASSETS="$(cd "${SCENECOT_DATA_ROOT}" && pwd)/evaluation_assets"
export SCENECOT_GPT_SCORE_PROMPT="${SCENECOT_GPT_SCORE_PROMPT:-${REPO_ROOT}/assets/eval/MSQA_eval_prompt.json}"
export GQA3D_TEST_ANNOTATION="$(cd "${SCENECOT_DATA_ROOT}" && pwd)/GQA3D/gqa3d_test.json"

for DATASET in ${DATASETS}; do
  case "${DATASET}" in
    msqa) DATASET_NAME=QACOTScanNetMSR3D ;;
    gqa3d) DATASET_NAME=GQA3D ;;
    *) echo "Unknown dataset: ${DATASET}" >&2; exit 2 ;;
  esac
  MODEL_NAME="${NAME}_${DATASET}_test"
  require_file "${RESULT_ROOT}/${MODEL_NAME}/eval_results/${DATASET_NAME}/results.json"
  "${PYTHON_BIN}" evaluator/msqa_evaluator_offline.py \
    gpt_model=gpt-4o-2024-11-20 gpt_score_flag=True \
    "result_dir=${RESULT_ROOT}" \
    "evaluated_models=[${MODEL_NAME}]" "evaluate_dataset=[${DATASET_NAME}]" \
    "evaluation_protocol=${DATASET}" \
    hydra.output_subdir=null "hydra.run.dir=${RESULT_ROOT}/${MODEL_NAME}/hydra"
done
echo "Summaries: ${RESULT_ROOT}/${NAME}_msqa_test/eval_results/MSQA_eval_results_with_gpt_score.json"
echo "           ${RESULT_ROOT}/${NAME}_gqa3d_test/eval_results/GQA3D_eval_results_with_gpt_score.json"
