<!-- TODO: paper title, venue, authors -->
<h2 align="center">
  <b>CARE-3D: &lt;Paper Title&gt;</b>
</h2>

<h3 align="center">
&lt;Venue Year&gt;
</h3>

<div align="center" margin-bottom="6em">
<a target="_blank" rel="external nofollow noopener" href="TODO">Author 1</a>,
<a target="_blank" rel="external nofollow noopener" href="TODO">Author 2</a>,
<a target="_blank" rel="external nofollow noopener" href="TODO">Author 3</a>
</div>
&nbsp;

<!-- TODO: teaser figure
<div align="middle">
<img src="static/images/teaser.jpg" width="85%" alt="CARE-3D Teaser">
</div>
-->

<b>CARE-3D</b> trains a Qwen3-VL-4B agent to answer situated questions in 3D scenes in two turns:
it writes a grounding query and picks one evidence tool (object list, 3D boxes, observer-centric
polar locations or an object image), then reasons over the returned evidence and answers. After
SFT, the agent is post-trained with GRPO using **COAC** (counterfactual observation credit for the
tool call) and **GCS** (tool-coverage sampling within each rollout group).

## 🔥 News
- \[2026-11\] Training and evaluation code released

## 🚀 Get Started

Tested on Linux, CUDA 12.8, 2x NVIDIA H100 80GB. Run all scripts from the repository root.

1. Clone the repository and create the environment.
```shell
git clone https://github.com/xiaoqian-UQ/CARE-3D
cd CARE-3D
bash scripts/setup/install_env.sh    # creates the conda env "care3d"
conda activate care3d
```

2. Download pretrained weights and data.
```shell
bash scripts/data/download_models.sh    # Qwen3-VL-4B-Instruct, PQ3D, CLIP (~12 GB)
bash scripts/data/download_data.sh      # SceneCOT-185K (~16 GB)
bash scripts/data/build_rl_parquet.sh   # verl parquet files for GRPO / evaluation
```

Data, weights and outputs default to `./data_assets`, `./model_assets` and `./outputs`; override
with `SCENECOT_DATA_ROOT`, `SCENECOT_MODEL_ROOT` and `SCENECOT_EXP_ROOT`.

## 🕹 Training

```shell
# SFT
bash scripts/sft/train_sft.sh
bash scripts/sft/export_for_rl.sh

# GRPO with COAC + GCS (600 steps)
export GEMINI_API_KEY=...    # answer judge: gemini-2.5-flash-lite
bash scripts/rl/train_care3d.sh
```

## 📊 Evaluation

MSQA and Beacon3D (GQA3D) test sets, scored with GPT-Score (`gpt-4o-2024-11-20`):
```shell
bash scripts/eval/eval_checkpoint.sh --actor-dir checkpoints/care3d_qwen3vl4b/global_step_600/actor --name care3d
export OPENAI_API_KEY=...
bash scripts/eval/score_gpt.sh care3d
```

Every hyper-parameter is an environment variable documented in the scripts (e.g.
[scripts/rl/train_care3d.sh](scripts/rl/train_care3d.sh)). Changes to the vendored verl are listed
in [verl/CARE3D_CHANGES.md](verl/CARE3D_CHANGES.md).

## 📝 TODO List

- [x] Training code
- [x] Evaluation code
- [ ] arXiv paper
- [ ] Model checkpoints

<!-- TODO: choose a license for the CARE-3D code. -->
