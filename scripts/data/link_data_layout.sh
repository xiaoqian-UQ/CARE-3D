#!/usr/bin/env bash
# Create the symlinks expected by the SFT data loader and the RL grounding tool, then check
# that every required input exists. Safe to re-run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

DATA_ROOT="$(cd "${SCENECOT_DATA_ROOT}" && pwd)"

link() {  # link <target> <link_name>
  local target="$1" name="$2"
  if [[ -L "${name}" && ! -e "${name}" ]]; then rm "${name}"; fi   # dangling link
  if [[ ! -e "${name}" ]]; then
    mkdir -p "$(dirname "${name}")"
    ln -s "${target}" "${name}"
  fi
}

# ScanNetBase expects annotations/, mask/ and scan_data/ directly under scan_family/.
link "${DATA_ROOT}/scan_family/scannet/annotations" "${DATA_ROOT}/scan_family/annotations"
link "${DATA_ROOT}/scan_family/scannet/mask"        "${DATA_ROOT}/scan_family/mask"
link "${DATA_ROOT}/sceneverse/scan_data"            "${DATA_ROOT}/scan_family/scan_data"
# The SFT loader reads point clouds from SceneVerse/ScanNet/scan_data.
link "${DATA_ROOT}/sceneverse/scan_data"            "${DATA_ROOT}/SceneVerse/ScanNet/scan_data"
# The RL tool reads the annotations from MSQA/ and GQA3D/.
link "${DATA_ROOT}/scenecot_cot_data/MSQA"          "${DATA_ROOT}/MSQA"
link "${DATA_ROOT}/scenecot_cot_data/GQA3D"         "${DATA_ROOT}/GQA3D"

require_file \
  "${DATA_ROOT}/MSQA/situated_qa_train_pure_txt.json" \
  "${DATA_ROOT}/MSQA/situated_qa_val_pure_txt.json" \
  "${DATA_ROOT}/MSQA/situated_qa_test_pure_txt.json" \
  "${DATA_ROOT}/GQA3D/gqa3d_train.json" \
  "${DATA_ROOT}/GQA3D/gqa3d_val.json" \
  "${DATA_ROOT}/GQA3D/gqa3d_test.json" \
  "${DATA_ROOT}/scan_family/annotations" \
  "${DATA_ROOT}/scan_family/mask" \
  "${DATA_ROOT}/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment" \
  "${DATA_ROOT}/scene-verse-pred-all/ScanNet/image_obj_feat_gt" \
  "${DATA_ROOT}/ScanNet/scene0000_00.pt" \
  "${DATA_ROOT}/scannet" \
  "${DATA_ROOT}/evaluation_assets/testset_mapping.json"
echo "Data layout OK: ${DATA_ROOT}"
