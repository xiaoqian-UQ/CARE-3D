from __future__ import annotations

import asyncio
import importlib
import os
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from scenecot_verl.agent.gcs_coverage import GCS_POLICIES, coverage_rng, sample_coverage_modes

# Import the registered loops before AgentLoopWorker loads the YAML registry.
# Otherwise Hydra's first lazy import runs @register after the YAML was loaded,
# replacing the rich model-specific config with only ``{"_target_": ...}``.
_scenecot_free_tag_agent_loop = importlib.import_module(
    "scenecot_verl.agent.scenecot_free_tag_agent_loop"
)


GCS_MODES = ("prob", "loc_rect", "loc_polar", "highlight")
GCS_COAC_TEXT_MODES = ("prob", "loc_rect", "loc_polar")
GCS_MODE_ALIASES = {
    "prob": "prob",
    "list_obj_prob": "prob",
    "<list_obj_prob>": "prob",
    "loc_rect": "loc_rect",
    "list_obj_loc_prob": "loc_rect",
    "<list_obj_loc_prob>": "loc_rect",
    "loc_polar": "loc_polar",
    "list_obj_loc_plr_prob": "loc_polar",
    "<list_obj_loc_plr_prob>": "loc_polar",
    "highlight": "highlight",
    "highlight_obj": "highlight",
    "<highlight_obj>": "highlight",
}
GCS_COVERAGE_MODE_POOL_ENV = "SCENECOT_GCS_COVERAGE_MODE_POOL"
COAC_ENABLE_ENV = "SCENECOT_COAC_ENABLE"
COAC_GROUP_ZSCORE_ENV = "SCENECOT_COAC_GROUP_ZSCORE"
COAC_VISUAL_ENABLE_ENV = "SCENECOT_COAC_VISUAL_ENABLE"
FORCE_AGENT_LOOP_NAME_ENV = "SCENECOT_FORCE_AGENT_LOOP_NAME"


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _cuda_visible_devices() -> list[str]:
    value = os.environ.get("SCENECOT_AGENT_LOOP_CUDA_VISIBLE_DEVICES")
    if value is None:
        return ["0"]
    if value.strip() == "":
        return [""]
    return [device.strip() for device in value.split(",") if device.strip()]


def _agent_loop_config_value(config, key: str, default):
    agent_config = config.actor_rollout_ref.rollout.agent
    path = agent_config.get("agent_loop_config_path")
    if not path:
        return default
    selected_agent_name = agent_config.get("default_agent_loop", "scenecot_free_tag_agent")
    try:
        configs = OmegaConf.load(resolve_config_path(path))
        for item in configs:
            if item.get("name") == selected_agent_name:
                return item.get(key, default)
    except Exception:
        return default
    return default


def _normalize_gcs_mode(value) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return GCS_MODE_ALIASES.get(text)


def _parse_coverage_mode_pool(raw_value) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    else:
        items = list(raw_value)
    modes: list[str] = []
    for item in items:
        mode = _normalize_gcs_mode(item)
        if mode is not None and mode not in modes:
            modes.append(mode)
    return tuple(modes)


def _default_coverage_mode_pool() -> tuple[str, ...]:
    if _env_bool(COAC_ENABLE_ENV, False) and _env_bool(COAC_GROUP_ZSCORE_ENV, False):
        if _env_bool(COAC_VISUAL_ENABLE_ENV, False):
            return GCS_MODES
        return GCS_COAC_TEXT_MODES
    return GCS_MODES


# Create several agent-loop workers, split each batch across them, and collect the rollouts they return
class SceneCOTAgentLoopManager(AgentLoopManager):

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(SceneCOTAgentLoopWorker)
        super().__init__(*args, **kwargs)
        self.coverage_num = int(os.getenv("SCENECOT_GCS_COVERAGE_NUM", _agent_loop_config_value(self.config, "coverage_num", 0)))
        self.coverage_warmup_step = int(
            os.getenv(
                "SCENECOT_GCS_COVERAGE_WARMUP_STEP",
                _agent_loop_config_value(self.config, "coverage_warmup_step", 0),
            )
        )
        self.coverage_warmup_num = int(
            os.getenv(
                "SCENECOT_GCS_COVERAGE_WARMUP_NUM",
                _agent_loop_config_value(self.config, "coverage_warmup_num", 0),
            )
        )
        self.coverage_anneal_step = int(
            os.getenv("SCENECOT_GCS_COVERAGE_ANNEAL_STEP", _agent_loop_config_value(self.config, "coverage_anneal_step", 200))
        )
        self.coverage_anneal_num = int(
            os.getenv("SCENECOT_GCS_COVERAGE_ANNEAL_NUM", _agent_loop_config_value(self.config, "coverage_anneal_num", 1))
        )
        configured_pool = os.getenv(
            GCS_COVERAGE_MODE_POOL_ENV,
            _agent_loop_config_value(self.config, "coverage_mode_pool", None),
        )
        self.coverage_mode_pool = _parse_coverage_mode_pool(configured_pool) or _default_coverage_mode_pool()
        self.coverage_policy = str(os.getenv(
            "SCENECOT_GCS_COVERAGE_POLICY",
            _agent_loop_config_value(self.config, "coverage_policy", "least_used"),
        )).strip().lower()
        if self.coverage_policy not in GCS_POLICIES:
            raise ValueError(f"Unknown GCS coverage policy: {self.coverage_policy!r}")
        self.coverage_seed = int(os.getenv(
            "SCENECOT_GCS_COVERAGE_SEED",
            _agent_loop_config_value(self.config, "coverage_seed", 42),
        ))
        print(f"GCS policy={self.coverage_policy} seed={self.coverage_seed} "
              f"pool={self.coverage_mode_pool} coverage_num={self.coverage_num}", flush=True)

    @staticmethod
    def _apply_forced_agent_loop_name(
        prompts: DataProto,
        default_agent_loop: str | None = None,
    ) -> None:
        """Override stale parquet routing for a model-specific training wrapper.

        Ray does not reliably propagate arbitrary driver environment variables
        to the TaskRunner that owns this manager.  The resolved Hydra default
        is already model-specific, so use it when the optional environment
        override is absent instead of trusting an ``agent_name`` persisted in
        a shared Qwen/LLaVA parquet.
        """
        agent_name = os.getenv(FORCE_AGENT_LOOP_NAME_ENV, "").strip()
        if not agent_name:
            agent_name = str(default_agent_loop or "").strip()
        if not agent_name:
            return
        prompts.non_tensor_batch["agent_name"] = np.full(
            len(prompts),
            agent_name,
            dtype=object,
        )

    def _effective_coverage_num(self, prompts: DataProto) -> int:
        if prompts.meta_info.get("validate", False):
            return 0
        coverage_num = max(0, self.coverage_num)
        step_value = prompts.meta_info.get("global_steps", -1)
        step = int(step_value) if step_value is not None else -1
        if 0 <= step < self.coverage_warmup_step:
            coverage_num = max(0, self.coverage_warmup_num)
        elif coverage_num > 0 and step >= self.coverage_anneal_step:
            coverage_num = min(coverage_num, max(0, self.coverage_anneal_num))
        rollout_n = int(self.rollout_config.n)
        return min(coverage_num, max(0, rollout_n - 1))

    @staticmethod
    def _ensure_rollout_numbers(prompts: DataProto) -> np.ndarray:
        if "__rollout_n_global__" in prompts.non_tensor_batch:
            return prompts.non_tensor_batch["__rollout_n_global__"]
        rollout_keys = prompts.non_tensor_batch.get("uid")
        if rollout_keys is None:
            rollout_keys = prompts.non_tensor_batch.get("index")
        if rollout_keys is None:
            rollout_keys = np.arange(len(prompts), dtype=np.int64)
        seen_rollouts = {}
        rollout_n = np.empty(len(prompts), dtype=np.int64)
        for i, key in enumerate(rollout_keys):
            key = key.item() if hasattr(key, "item") else key
            rollout_n[i] = seen_rollouts.get(key, 0)
            seen_rollouts[key] = rollout_n[i] + 1
        prompts.non_tensor_batch["__rollout_n_global__"] = rollout_n
        return rollout_n

    @staticmethod
    def _group_keys(prompts: DataProto) -> np.ndarray:
        keys = prompts.non_tensor_batch.get("uid")
        if keys is None:
            keys = prompts.non_tensor_batch.get("index")
        if keys is None:
            keys = np.arange(len(prompts), dtype=object)
        return keys

    @staticmethod
    def _mode_from_actions(actions) -> str | None:
        if isinstance(actions, list) and actions:
            action = actions[0]
            if isinstance(action, dict):
                mode = action.get("mode")
                return str(mode) if mode else None
        return None

    @staticmethod
    def _sample_coverage_modes(counts: dict[str, int], n: int, mode_pool: tuple[str, ...] | None = None) -> list[str]:
        return sample_coverage_modes(counts, n, mode_pool or GCS_MODES)

    async def _dispatch(self, prompts: DataProto) -> DataProto:
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        metrics = [item.meta_info.pop("metrics") for item in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def _generate_sequences_async(self, prompts: DataProto) -> DataProto:
        self._apply_forced_agent_loop_name(
            prompts,
            default_agent_loop=self.rollout_config.agent.default_agent_loop,
        )
        coverage_num = self._effective_coverage_num(prompts)
        if coverage_num <= 0:
            return await super().generate_sequences(prompts)

        rollout_n = self._ensure_rollout_numbers(prompts)
        group_keys = self._group_keys(prompts)
        total_rollout_n = int(self.rollout_config.n)
        n_free = max(1, total_rollout_n - coverage_num)
        free_indices = np.nonzero(rollout_n < n_free)[0]
        coverage_indices = np.nonzero((rollout_n >= n_free) & (rollout_n < total_rollout_n))[0]
        if len(coverage_indices) == 0 or len(free_indices) == 0:
            return await super().generate_sequences(prompts)

        free_output = await self._dispatch(prompts[free_indices])
        free_timing = free_output.meta_info.pop("timing", {})

        counts_by_group: dict[Any, dict[str, int]] = {}
        actions_arr = free_output.non_tensor_batch.get("scenecot_actions", np.array([], dtype=object))
        for original_idx, actions in zip(free_indices, actions_arr, strict=False):
            key = group_keys[original_idx].item() if hasattr(group_keys[original_idx], "item") else group_keys[original_idx]
            mode = self._mode_from_actions(actions)
            counts = counts_by_group.setdefault(key, {mode_name: 0 for mode_name in self.coverage_mode_pool})
            if mode in counts:
                counts[mode] += 1

        plans: dict[Any, list[str]] = {}
        step = int(prompts.meta_info.get("global_steps", 0) or 0)
        group_count = len(set(group_keys[coverage_indices]))
        for original_idx in coverage_indices:
            key = group_keys[original_idx].item() if hasattr(group_keys[original_idx], "item") else group_keys[original_idx]
            if key not in plans:
                group_index = len(plans)
                # Stateless schedule: reproducible on resume, independent of
                # uuid4 group IDs, worker scheduling and observed tool usage.
                uniform_offset = (
                    self.coverage_seed + max(0, step - 1) * group_count * coverage_num
                    + group_index * coverage_num
                )
                plans[key] = sample_coverage_modes(
                    counts_by_group.get(key, {}),
                    coverage_num,
                    mode_pool=self.coverage_mode_pool,
                    policy=self.coverage_policy,
                    rng=coverage_rng(self.coverage_seed, step, group_index),
                    uniform_offset=uniform_offset,
                )

        coverage_batch = prompts[coverage_indices]
        forced_modes = np.empty(len(coverage_indices), dtype=object)
        used_by_group: dict[Any, int] = {}
        for local_idx, original_idx in enumerate(coverage_indices):
            key = group_keys[original_idx].item() if hasattr(group_keys[original_idx], "item") else group_keys[original_idx]
            offset = used_by_group.get(key, 0)
            plan = plans.get(key) or []
            forced_modes[local_idx] = plan[offset % len(plan)] if plan else None
            used_by_group[key] = offset + 1
        coverage_batch.non_tensor_batch["scenecot_forced_mode"] = forced_modes
        coverage_output = await self._dispatch(coverage_batch)
        coverage_timing = coverage_output.meta_info.pop("timing", {})

        combined = DataProto.concat([free_output, coverage_output])
        restore_order = np.argsort(np.concatenate([free_indices, coverage_indices]))
        combined.reorder(torch.as_tensor(restore_order, dtype=torch.long))
        combined.meta_info["timing"] = {
            **{f"gcs/free/{key}": value for key, value in free_timing.items()},
            **{f"gcs/coverage/{key}": value for key, value in coverage_timing.items()},
            "gcs/coverage_num": coverage_num,
            "gcs/free_count": int(len(free_indices)),
            "gcs/coverage_count": int(len(coverage_indices)),
            **{f"gcs/policy/{policy}": int(self.coverage_policy == policy) for policy in GCS_POLICIES},
            **{f"gcs/forced_mode/{mode}/count": int(np.count_nonzero(forced_modes == mode))
               for mode in self.coverage_mode_pool},
        }
        return combined

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        return await self._generate_sequences_async(prompts)

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers
        num_gpus = _env_float("SCENECOT_AGENT_LOOP_NUM_GPUS", 0.0)
        num_cpus = _env_float("SCENECOT_AGENT_LOOP_NUM_CPUS", 1.0)
        cuda_devices = _cuda_visible_devices()

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            cuda_visible_devices = cuda_devices[i % len(cuda_devices)]
            options = {
                "name": f"scenecot_agent_loop_worker_{i}_{uuid4().hex[:8]}",
                "num_cpus": num_cpus,
                "runtime_env": {
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "SCENECOT_AGENT_LOOP_ASSIGNED_CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                    },
                },
                "scheduling_strategy": ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=True
                ),
            }
            if num_gpus > 0:
                options["num_gpus"] = num_gpus

            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(**options).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )


class SceneCOTAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker that can run CUDA-only SceneCOT grounding kernels."""

    def __init__(self, *args, **kwargs):
        cuda_visible_devices = os.environ.get(
            "SCENECOT_AGENT_LOOP_ASSIGNED_CUDA_VISIBLE_DEVICES",
            os.environ.get("SCENECOT_AGENT_LOOP_CUDA_VISIBLE_DEVICES", "0"),
        )
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(
            "SceneCOTAgentLoopWorker CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            flush=True,
        )
        super().__init__(*args, **kwargs)
