#!/usr/bin/env bash
# Download and lay out the SceneCOT-185K data (MSQA + GQA3D/Beacon3D annotations, ScanNet
# object features, object images and point clouds) under ${SCENECOT_DATA_ROOT}.
#
#   bash scripts/data/download_data.sh
#
# ~16 GB of zip files are downloaded (~30 GB after extraction). Set KEEP_ZIPS=1 to keep them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

HF_DATA_REPO="${HF_DATA_REPO:-EricLHK/SceneCOT}"
HF_DATA_REVISION="${HF_DATA_REVISION:-d2742f1834f9c2c699d41b61466889dea560fe6d}"
DATA_ROOT="${SCENECOT_DATA_ROOT}"
DOWNLOAD_DIR="${DATA_ROOT}/_hf_download"
mkdir -p "${DATA_ROOT}" "${DOWNLOAD_DIR}"

huggingface-cli download "${HF_DATA_REPO}" --repo-type dataset --revision "${HF_DATA_REVISION}" \
  --local-dir "${DOWNLOAD_DIR}"

# zip archive -> directory it creates inside ${DATA_ROOT}
declare -A ZIPS=(
  [scan_family.zip]=scan_family/scannet               # ScanNet annotations / instance masks
  [sceneverse.zip]=sceneverse/scan_data               # point clouds + instance labels
  [scene-verse-pred-all.zip]=scene-verse-pred-all     # offline object features (gt / predicted masks)
  [LEO-2_feature_ScanNet.zip]=ScanNet                 # 2D object features
  [scenecot_imgs.zip]=scannet                         # object crops used by <highlight_obj>
)
for zip in "${!ZIPS[@]}"; do
  if [[ -e "${DATA_ROOT}/${ZIPS[$zip]}" ]]; then
    echo "[skip] ${ZIPS[$zip]} exists"
  else
    echo "[unzip] ${zip}"
    unzip -q "${DOWNLOAD_DIR}/${zip}" -d "${DATA_ROOT}"
  fi
done

for dir in scenecot_cot_data evaluation_assets; do
  [[ -e "${DATA_ROOT}/${dir}" ]] || cp -r "${DOWNLOAD_DIR}/${dir}" "${DATA_ROOT}/${dir}"
done
[[ "${KEEP_ZIPS:-0}" == 1 ]] || rm -rf "${DOWNLOAD_DIR}"

bash scripts/data/link_data_layout.sh
