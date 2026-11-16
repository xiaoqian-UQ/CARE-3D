#!/usr/bin/env bash
# Generic launcher for the SceneCOT agent loop on verl (GRPO). Every knob is an environment
# variable; extra Hydra overrides are appended from "$@". Called by scripts/rl/train_care3d.sh
# and scripts/eval/eval_checkpoint.sh -- use those entry points rather than this file.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

# ---- grounding tool (configs/rl/scenecot_ground_tool.yaml) ----
export SCENECOT_ACTOR_MODEL_PATH="${SCENECOT_ACTOR_MODEL_PATH:-${QWEN_MODEL_PATH}}"
export SCENECOT_SFT_CKPT="${SCENECOT_SFT_CKPT:-${SFT_EXPORT_DIR}/grounder/best.pth/pytorch_model.bin}"
export MSQA_ROOT="${MSQA_ROOT:-${SCENECOT_DATA_ROOT}/MSQA}"
export GQA3D_ROOT="${GQA3D_ROOT:-${SCENECOT_DATA_ROOT}/GQA3D}"
export SCENECOT_OBJECT_FEATURE_ROOT="${SCENECOT_OBJECT_FEATURE_ROOT:-${SCENECOT_DATA_ROOT}/scene-verse-pred-all/ScanNet}"
export SCENECOT_SCENE_FEATURE_ROOT="${SCENECOT_SCENE_FEATURE_ROOT:-${SCENECOT_DATA_ROOT}/sceneverse}"
export SCENECOT_OBJECT_IMAGE_ROOT="${SCENECOT_OBJECT_IMAGE_ROOT:-${SCENECOT_DATA_ROOT}/scannet}"
export SCENECOT_POINTNET_TOKENIZER="${SCENECOT_POINTNET_TOKENIZER:-${SCENECOT_POINTNET_TOKENIZER_PATH}}"
export SCENECOT_TOOL_DEVICE="${SCENECOT_TOOL_DEVICE:-cuda}"
export SCENECOT_PC_TYPE="${SCENECOT_PC_TYPE:-gt}"
export SCENECOT_OBJECT_TOKEN_CACHE_ROOT="${SCENECOT_OBJECT_TOKEN_CACHE_ROOT:-}"
export SCENECOT_OBJECT_TOKEN_CACHE_MODE="${SCENECOT_OBJECT_TOKEN_CACHE_MODE:-off}"
# Optional pre-computed Qwen3-VL features of the highlight images (see README); off = online.
export SCENECOT_VISUAL_CACHE_DIR="${SCENECOT_VISUAL_CACHE_DIR:-}"
export SCENECOT_VISUAL_CACHE_MODE="${SCENECOT_VISUAL_CACHE_MODE:-read_only}"
export SCENECOT_VISUAL_CACHE_MAX_SHARDS="${SCENECOT_VISUAL_CACHE_MAX_SHARDS:-4}"
export SCENECOT_AGENT_LOOP_NUM_GPUS="${SCENECOT_AGENT_LOOP_NUM_GPUS:-0}"
export SCENECOT_AGENT_LOOP_NUM_CPUS="${SCENECOT_AGENT_LOOP_NUM_CPUS:-1}"
export SCENECOT_AGENT_LOOP_CUDA_VISIBLE_DEVICES="${SCENECOT_AGENT_LOOP_CUDA_VISIBLE_DEVICES-${CUDA_VISIBLE_DEVICES:-0,1}}"

# ---- reward: answer judge ----
export SCENECOT_ANSWER_JUDGE_BACKEND="${SCENECOT_ANSWER_JUDGE_BACKEND:-rule}"
export SCENECOT_ANSWER_JUDGE_MODEL="${SCENECOT_ANSWER_JUDGE_MODEL:-gemini-2.5-flash-lite}"
export SCENECOT_ANSWER_JUDGE_PROMPT_STYLE="${SCENECOT_ANSWER_JUDGE_PROMPT_STYLE:-msqa_1to5}"
export SCENECOT_ANSWER_JUDGE_RUBRIC_PATH="${SCENECOT_ANSWER_JUDGE_RUBRIC_PATH:-assets/eval/MSQA_eval_prompt.json}"
export SCENECOT_ANSWER_JUDGE_TIMEOUT="${SCENECOT_ANSWER_JUDGE_TIMEOUT:-8}"
export SCENECOT_ANSWER_JUDGE_MIN_F1="${SCENECOT_ANSWER_JUDGE_MIN_F1:-0}"
export SCENECOT_ANSWER_JUDGE_MAX_F1="${SCENECOT_ANSWER_JUDGE_MAX_F1:-1}"

# ---- reward: format / length / repetition shaping ----
export SCENECOT_ENABLE_AGENT_REWARD="${SCENECOT_ENABLE_AGENT_REWARD:-1}"
export SCENECOT_ENABLE_PROGRESS_SWITCH="${SCENECOT_ENABLE_PROGRESS_SWITCH:-1}"
export SCENECOT_PROGRESS_SWITCH_STEP="${SCENECOT_PROGRESS_SWITCH_STEP:-200}"
export SCENECOT_FORMAT_W_START="${SCENECOT_FORMAT_W_START:-1.20}"
export SCENECOT_FORMAT_W_END="${SCENECOT_FORMAT_W_END:-0.80}"
export SCENECOT_ANSWER_W_START="${SCENECOT_ANSWER_W_START:-0.80}"
export SCENECOT_ANSWER_W_END="${SCENECOT_ANSWER_W_END:-1.20}"
export SCENECOT_FIRST_TURN_MAX_TOKENS="${SCENECOT_FIRST_TURN_MAX_TOKENS:-96}"
export SCENECOT_FINAL_TURN_MAX_TOKENS="${SCENECOT_FINAL_TURN_MAX_TOKENS:-512}"
export SCENECOT_FINAL_LENGTH_PENALTY="${SCENECOT_FINAL_LENGTH_PENALTY:-1}"
export SCENECOT_FINAL_REASONING_SOFT_TOKEN_LIMIT="${SCENECOT_FINAL_REASONING_SOFT_TOKEN_LIMIT:-128}"
export SCENECOT_FINAL_REASONING_HARD_TOKEN_LIMIT="${SCENECOT_FINAL_REASONING_HARD_TOKEN_LIMIT:-256}"
export SCENECOT_FINAL_REASONING_MAX_PENALTY="${SCENECOT_FINAL_REASONING_MAX_PENALTY:-0.10}"
export SCENECOT_FINAL_ANSWER_SOFT_TOKEN_LIMIT="${SCENECOT_FINAL_ANSWER_SOFT_TOKEN_LIMIT:-40}"
export SCENECOT_FINAL_ANSWER_HARD_TOKEN_LIMIT="${SCENECOT_FINAL_ANSWER_HARD_TOKEN_LIMIT:-80}"
export SCENECOT_FINAL_ANSWER_MAX_PENALTY="${SCENECOT_FINAL_ANSWER_MAX_PENALTY:-0.05}"
export SCENECOT_MAX_REPEAT_PUNCT_RUN="${SCENECOT_MAX_REPEAT_PUNCT_RUN:-4}"
export SCENECOT_SEVERE_REPEATED_PUNCT_RUN="${SCENECOT_SEVERE_REPEATED_PUNCT_RUN:-12}"
export SCENECOT_REPEATED_PUNCT_MAX_PENALTY="${SCENECOT_REPEATED_PUNCT_MAX_PENALTY:-1.0}"
export SCENECOT_REPEATED_SENTENCE_THRESHOLD="${SCENECOT_REPEATED_SENTENCE_THRESHOLD:-3}"
export SCENECOT_TOO_LONG_TOKENIZER_PATH="${SCENECOT_TOO_LONG_TOKENIZER_PATH:-${SCENECOT_ACTOR_MODEL_PATH}}"
export SCENECOT_TOO_LONG_TOKENIZER_LOCAL_FILES_ONLY="${SCENECOT_TOO_LONG_TOKENIZER_LOCAL_FILES_ONLY:-1}"

# ---- COAC (counterfactual tool credit) and GCS (coverage sampling); off unless enabled ----
export SCENECOT_COAC_ENABLE="${SCENECOT_COAC_ENABLE:-0}"
export SCENECOT_COAC_TAU="${SCENECOT_COAC_TAU:-0.5}"
export SCENECOT_COAC_COPY_THRESH="${SCENECOT_COAC_COPY_THRESH:-0.6}"
export SCENECOT_COAC_COPY_DISCOUNT="${SCENECOT_COAC_COPY_DISCOUNT:-0.5}"
export SCENECOT_COAC_GROUP_ZSCORE="${SCENECOT_COAC_GROUP_ZSCORE:-0}"
export SCENECOT_GCS_COVERAGE_NUM="${SCENECOT_GCS_COVERAGE_NUM:-0}"
export SCENECOT_GCS_COVERAGE_POLICY="${SCENECOT_GCS_COVERAGE_POLICY:-least_used}"
export SCENECOT_GCS_COVERAGE_SEED="${SCENECOT_GCS_COVERAGE_SEED:-42}"
export SCENECOT_GCS_COVERAGE_ANNEAL_STEP="${SCENECOT_GCS_COVERAGE_ANNEAL_STEP:-200}"
export SCENECOT_GCS_COVERAGE_ANNEAL_NUM="${SCENECOT_GCS_COVERAGE_ANNEAL_NUM:-1}"

# ---- data ----
TRAIN_PARQUET="${TRAIN_PARQUET:-${RL_PARQUET_DIR}/msqa/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${RL_PARQUET_DIR}/msqa/val_msqa_20_per_type.parquet}"
require_file "${TRAIN_PARQUET}" "${VAL_PARQUET}" "${SCENECOT_ACTOR_MODEL_PATH}/config.json" "${SCENECOT_SFT_CKPT}"

# ---- resources / batch ----
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
NNODES="${NNODES:-1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-${N_GPUS_PER_NODE}}"
# Ray needs a short temp path (unix socket length limit).
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_${USER:-user}_care3d_$$}"
ROLLOUT_STANDALONE="${ROLLOUT_STANDALONE:-False}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-0}"
ROLLOUT_N_GPUS_PER_NODE="${ROLLOUT_N_GPUS_PER_NODE:-${N_GPUS_PER_NODE}}"
ROLLOUT_CKPT_BACKEND="${ROLLOUT_CKPT_BACKEND:-naive}"
ROLLOUT_CKPT_BUCKET_MB="${ROLLOUT_CKPT_BUCKET_MB:-1024}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1700}"
MM_PROCESSOR_MAX_PIXELS="${MM_PROCESSOR_MAX_PIXELS:-524288}"
MM_PROCESSOR_SHORTEST_EDGE="${MM_PROCESSOR_SHORTEST_EDGE:-3136}"
MM_PROCESSOR_LONGEST_EDGE="${MM_PROCESSOR_LONGEST_EDGE:-131072}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-600}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.30}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-4096}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1024}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
# Qwen3-VL mRoPE: keep remove-padding off; with it on, trajectories without image tokens get
# position_ids of the wrong length and the forward pass fails.
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"
# fsdp_size=1: one full actor replica per GPU; DDP syncs the LoRA gradients.
ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:-1}"
REF_FSDP_SIZE="${REF_FSDP_SIZE:-1}"
# The reference policy is the base model with the adapter disabled.
SHARE_REF_WITH_ACTOR="${SHARE_REF_WITH_ACTOR:-True}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# vLLM supports LoRA on the language model only; never target visual.* modules.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
SEED="${SEED:-42}"

AGENT_CONFIG="${AGENT_CONFIG:-configs/rl/scenecot_free_tag_agent.yaml}"
AGENT_LOOP_NAME="${AGENT_LOOP_NAME:-scenecot_free_tag_agent}"
export SCENECOT_FORCE_AGENT_LOOP_NAME="${SCENECOT_FORCE_AGENT_LOOP_NAME:-${AGENT_LOOP_NAME}}"
TOOL_CONFIG="${TOOL_CONFIG:-configs/rl/scenecot_ground_tool.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-care3d_qwen3vl4b}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

mkdir -p "${RAY_TEMP_DIR}"

# The agent-loop manager lives in Ray's TaskRunner; propagate the GCS / judge settings
# explicitly (also when attaching to an already running Ray instance).
GCS_RUNTIME_ARGS=()
while IFS= read -r name; do
  GCS_RUNTIME_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${name}='${!name}'")
done < <(compgen -e SCENECOT_GCS_; compgen -e SCENECOT_ANSWER_JUDGE_)

HYDRA_CMD=(
  "${PYTHON_BIN}" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="${TRAIN_PARQUET}"
  data.val_files="${VAL_PARQUET}"
  data.train_max_samples="${TRAIN_MAX_SAMPLES}"
  data.val_max_samples="${VAL_MAX_SAMPLES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.return_raw_chat=True
  data.return_multi_modal_inputs=False
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  +data.mm_processor_kwargs.max_pixels="${MM_PROCESSOR_MAX_PIXELS}"
  +data.mm_processor_kwargs.size.shortest_edge="${MM_PROCESSOR_SHORTEST_EDGE}"
  +data.mm_processor_kwargs.size.longest_edge="${MM_PROCESSOR_LONGEST_EDGE}"
  data.filter_overlong_prompts=True
  data.truncation=error
  data.trust_remote_code=True
  data.dataloader_num_workers=8
  data.seed="${SEED}"
  actor_rollout_ref.model.path="${SCENECOT_ACTOR_MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
  actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}"
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}"
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"
  actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}"
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${ACTOR_FSDP_SIZE}"
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.data_loader_seed="${SEED}"
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  ++actor_rollout_ref.rollout.standalone="${ROLLOUT_STANDALONE}"
  actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}"
  actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_N_GPUS_PER_NODE}"
  actor_rollout_ref.rollout.checkpoint_engine.backend="${ROLLOUT_CKPT_BACKEND}"
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${ROLLOUT_CKPT_BUCKET_MB}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}"
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable
  actor_rollout_ref.rollout.agent.default_agent_loop="${AGENT_LOOP_NAME}"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}"
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=scenecot_verl.agent.scenecot_agent_loop_manager.SceneCOTAgentLoopManager
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}"
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.ref.share_with_actor="${SHARE_REF_WITH_ACTOR}"
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.fsdp_config.fsdp_size="${REF_FSDP_SIZE}"
  actor_rollout_ref.ref.fsdp_config.param_offload=False
  actor_rollout_ref.ref.use_torch_compile=False
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
  custom_reward_function.path=scenecot_verl/rewards/scenecot_reward.py
  custom_reward_function.name=scenecot_reward
  reward.custom_reward_function.path=scenecot_verl/rewards/scenecot_reward.py
  reward.custom_reward_function.name=scenecot_reward
  reward.reward_manager.source=importlib
  reward.reward_manager.name=SceneCOTNaiveRewardManager
  reward.reward_manager.module.path=scenecot_verl/rewards/reward_manager.py
  trainer.project_name=care3d
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.logger="${TRAINER_LOGGER}"
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.total_epochs="${TOTAL_TRAINING_STEPS}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  +ray_kwargs.ray_init.num_gpus="${RAY_NUM_GPUS}"
  +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}"
  ++ray_kwargs.ray_init.include_dashboard=False
  "${GCS_RUNTIME_ARGS[@]}"
  "$@"
)

printf 'Final Hydra command:\n'
printf ' %q' "${HYDRA_CMD[@]}"
printf '\n'
if [[ "${PRINT_HYDRA_CMD_ONLY:-0}" == "1" ]]; then
  exit 0
fi
"${HYDRA_CMD[@]}"
