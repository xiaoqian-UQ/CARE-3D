#!/usr/bin/env bash
# Build the verl parquet files for GRPO training / evaluation.
#
#   ${RL_PARQUET_DIR}/msqa/{train,val,test}.parquet     MSQA (GRPO is trained on MSQA train)
#   ${RL_PARQUET_DIR}/gqa3d/{train,val,test}.parquet    GQA3D / Beacon3D (test only is used)
#   ${RL_PARQUET_DIR}/msqa/val_msqa_20_per_type.parquet fixed 180-question validation subset
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

MSQA_DIR="${SCENECOT_DATA_ROOT}/MSQA"
GQA3D_DIR="${SCENECOT_DATA_ROOT}/GQA3D"

"${PYTHON_BIN}" scenecot_verl/data/build_parquet.py \
  --train-input "${MSQA_DIR}/situated_qa_train_pure_txt.json" \
  --val-input "${MSQA_DIR}/situated_qa_val_pure_txt.json" \
  --test-input "${MSQA_DIR}/situated_qa_test_pure_txt.json" \
  --output-dir "${RL_PARQUET_DIR}/msqa" \
  --dataset-name msqa

"${PYTHON_BIN}" scenecot_verl/data/build_parquet.py \
  --train-input "${GQA3D_DIR}/gqa3d_train.json" \
  --val-input "${GQA3D_DIR}/gqa3d_val.json" \
  --test-input "${GQA3D_DIR}/gqa3d_test.json" \
  --output-dir "${RL_PARQUET_DIR}/gqa3d" \
  --dataset-name gqa3d

"${PYTHON_BIN}" scripts/data/make_val_subset.py \
  --val-parquet "${RL_PARQUET_DIR}/msqa/val.parquet" \
  --output "${RL_PARQUET_DIR}/msqa/val_msqa_20_per_type.parquet"

echo "Expected rows: MSQA train/val/test = 145260/889/826, GQA3D test = 1191, val subset = 180."
