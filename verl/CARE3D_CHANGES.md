# Changes to the vendored verl

`verl/` is a vendored copy of [verl](https://github.com/verl-project/verl) (`0.8.0.dev`, Apache-2.0)
with the project-specific changes listed below. Install it from the repository root
(`pip install --no-deps -e verl`); the PyPI release does not contain these changes.

| File | Change |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` | COAC: builds the real / counterfactual teacher-forced batches (text and image observations), scores the reference answer with the current actor, computes per-group z-score credit (`_compute_scenecot_group_zscore_credit`) and passes it to the reward. Optional Qwen3-VL visual-feature cache and a (disabled) format-auxiliary loss hook. |
| `verl/experimental/agent_loop/agent_loop.py` | Passes `rollout_n`, `sample_index`, `trajectory_info` and `validate` to the agent loop (needed by GCS); computes the global rollout index before the batch is split across agent-loop workers; builds object-dtype `reward_extra_info` columns so list-valued grounding statistics do not break `DataProto.concat`; optional visual-feature cache. |
| `verl/trainer/ppo/metric_utils.py` | Validation metric aggregation ignores `None` / non-finite values (e.g. grounding rank on questions without ground-truth objects). |
| `verl/utils/weight_sync_ipc.py`, `verl/workers/rollout/vllm_rollout/{vllm_rollout,vllm_async_server,utils}.py` | ZMQ IPC socket names for FSDP -> vLLM weight sync are namespaced by `sha256(ray_gcs_address, ray_job_id)` so that concurrent Ray clusters on one node do not collide. |
| `verl/workers/utils/losses.py` | Optional format-auxiliary loss term (coefficient 0, i.e. disabled, for CARE-3D). |
| `verl/trainer/main_ppo_sync.py` | Logs the SceneCOT reward metrics (format tiers, judge / COAC / GCS statistics). |

When upgrading verl, re-apply these changes and run `pytest tests/` from the repository root.
