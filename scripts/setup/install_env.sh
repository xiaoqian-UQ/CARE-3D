#!/usr/bin/env bash
# Create the single conda environment used for SFT, GRPO and evaluation.
# Tested on Linux, CUDA 12.8 driver/toolkit, 2x NVIDIA H100 80GB.
#
#   bash scripts/setup/install_env.sh            # creates env "care3d"
#   ENV_NAME=myenv bash scripts/setup/install_env.sh
#
# Requirements: conda and a CUDA 12.x toolkit (12.8 recommended) to compile flash-attn and the
# PointNet++ extension; its major version must match PyTorch's cu128 build. The toolkit is taken
# from CUDA_HOME, else the nvcc on PATH, else /usr/local/cuda-12*. Java (METEOR metric) is
# installed from conda-forge if missing. The script can be re-run to resume after a failure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ENV_NAME="${ENV_NAME:-care3d}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-8}"

cuda_major() { "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p'; }
candidates=()
if [[ -n "${CUDA_HOME:-}" ]]; then candidates+=("${CUDA_HOME}"); fi
if command -v nvcc >/dev/null 2>&1; then candidates+=("$(dirname "$(dirname "$(command -v nvcc)")")"); fi
candidates+=(/usr/local/cuda $(ls -d /usr/local/cuda-12* 2>/dev/null | sort -V -r))
REQUESTED_CUDA_HOME="${CUDA_HOME:-}"
CUDA_HOME=""
for candidate in "${candidates[@]}"; do
  if [[ "$(cuda_major "${candidate}")" == 12 ]]; then CUDA_HOME="${candidate}"; break; fi
done
if [[ -n "${REQUESTED_CUDA_HOME}" && "${CUDA_HOME}" != "${REQUESTED_CUDA_HOME}" ]]; then
  echo "CUDA_HOME=${REQUESTED_CUDA_HOME} is not a CUDA 12.x toolkit; looking for another one." >&2
fi
if [[ -z "${CUDA_HOME}" ]]; then
  echo "No CUDA 12.x toolkit found. Set CUDA_HOME to one (e.g. /usr/local/cuda-12.8) or load it" \
       "(e.g. 'module load cuda/12.8'); PyTorch is built with CUDA 12.8." >&2
  exit 1
fi
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}"
echo "Using CUDA toolkit ${CUDA_HOME}: $(nvcc --version | tail -1)"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

# 1) PyTorch + vLLM (vllm==0.12.0 requires torch==2.9.0)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.12.0

# 2) Python dependencies (versions of the environment used for the paper)
pip install -r requirements.txt

# 3) Compiled extras. flash-attn 2.8.3 has no pre-built wheel for torch 2.9, so pip compiles it
#    from source, which takes a long time (lower MAX_JOBS if the build runs out of memory).
#    SKIP_FLASH_ATTN=1 skips it; verl then computes log-probs with a plain PyTorch kernel
#    instead of flash-attn's (same maths, slightly different numerics from the paper runs).
if [[ "${SKIP_FLASH_ATTN:-0}" != 1 ]] && ! python -c "import flash_attn" 2>/dev/null; then
  pip install flash-attn==2.8.3 --no-build-isolation
fi
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

# 4) Vendored (patched) verl. Do NOT install verl from PyPI.
pip install --no-deps -e verl

# 5) PointNet++ CUDA extension used by the PQ3D 3D encoder
(cd model/pointnetpp && rm -rf build ./*.egg-info && pip install --no-build-isolation -e .)

# 6) Sanity checks
python -c "from model.pointnetpp.pointnetpp import PointNetPP; print('PointNet++ OK')"
PYTHONPATH="$PWD:$PWD/verl" python -c "
import torch, transformers, vllm, peft, verl
from scenecot_verl.agent.scenecot_free_tag_agent_loop import SceneCOTFreeTagAgentLoop
from scenecot_verl.tools.scenecot_ground_tool import SceneCOTGroundTool
from scenecot_verl.rewards.scenecot_reward import scenecot_reward
print('torch', torch.__version__, '| transformers', transformers.__version__, '| vllm', vllm.__version__, '| peft', peft.__version__)
print('CARE-3D imports OK')
"
# Java is needed by the METEOR scorer that the SFT trainer builds at start-up.
if ! command -v java >/dev/null 2>&1; then
  conda install -y -c conda-forge openjdk=11
fi
echo "Environment '${ENV_NAME}' is ready."
