from __future__ import annotations

import sys
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

for module_name in list(sys.modules):
    if module_name == "verl" or module_name.startswith("verl."):
        sys.modules.pop(module_name, None)

from scenecot_verl.agent.scenecot_agent_loop_manager import SceneCOTAgentLoopManager
from verl.protocol import DataProto


def test_model_wrapper_overrides_stale_parquet_agent_name(monkeypatch):
    class Prompts:
        def __init__(self):
            self.non_tensor_batch = {
                "agent_name": np.array(["scenecot_free_tag_agent"] * 3, dtype=object)
            }

        def __len__(self):
            return 3

    prompts = Prompts()
    monkeypatch.setenv("SCENECOT_FORCE_AGENT_LOOP_NAME", "scenecot_qwen25_agent")

    SceneCOTAgentLoopManager._apply_forced_agent_loop_name(prompts)

    assert prompts.non_tensor_batch["agent_name"].tolist() == [
        "scenecot_qwen25_agent",
        "scenecot_qwen25_agent",
        "scenecot_qwen25_agent",
    ]


def test_hydra_default_overrides_stale_agent_when_ray_drops_env(monkeypatch):
    class Prompts:
        def __init__(self):
            self.non_tensor_batch = {
                "agent_name": np.array(
                    ["scenecot_free_tag_agent"] * 2,
                    dtype=object,
                )
            }

        def __len__(self):
            return 2

    prompts = Prompts()
    monkeypatch.delenv("SCENECOT_FORCE_AGENT_LOOP_NAME", raising=False)

    SceneCOTAgentLoopManager._apply_forced_agent_loop_name(
        prompts,
        default_agent_loop="scenecot_llava15_agent",
    )

    assert prompts.non_tensor_batch["agent_name"].tolist() == [
        "scenecot_llava15_agent",
        "scenecot_llava15_agent",
    ]


def test_gcs_weighted_sampling_covers_zero_count_modes():
    selected = SceneCOTAgentLoopManager._sample_coverage_modes(
        {"prob": 0, "loc_rect": 0, "loc_polar": 3, "highlight": 3},
        2,
    )

    assert set(selected) == {"prob", "loc_rect"}


def test_gcs_weighted_sampling_is_without_replacement():
    selected = SceneCOTAgentLoopManager._sample_coverage_modes(
        {"prob": 1, "loc_rect": 1, "loc_polar": 1, "highlight": 1},
        3,
    )

    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_gcs_coverage_pool_can_exclude_highlight_when_highlight_dominates():
    selected = SceneCOTAgentLoopManager._sample_coverage_modes(
        {"prob": 1, "loc_rect": 1, "loc_polar": 1, "highlight": 99},
        2,
        mode_pool=("prob", "loc_rect", "loc_polar"),
    )

    assert selected
    assert "highlight" not in selected


def test_gcs_coverage_can_be_disabled_during_format_warmup():
    manager = object.__new__(SceneCOTAgentLoopManager)
    manager.coverage_num = 1
    manager.coverage_warmup_step = 200
    manager.coverage_warmup_num = 0
    manager.coverage_anneal_step = 200
    manager.coverage_anneal_num = 1
    manager.rollout_config = SimpleNamespace(n=8)

    assert manager._effective_coverage_num(SimpleNamespace(meta_info={"global_steps": 199})) == 0
    assert manager._effective_coverage_num(SimpleNamespace(meta_info={"global_steps": 0})) == 0
    assert manager._effective_coverage_num(SimpleNamespace(meta_info={"global_steps": 200})) == 1
    assert manager._effective_coverage_num(
        SimpleNamespace(meta_info={"global_steps": 200, "validate": True})
    ) == 0


@pytest.mark.parametrize("policy", ["least_used", "random", "uniform"])
def test_policies_share_dispatch_budget_and_restore_order(policy):
    manager = object.__new__(SceneCOTAgentLoopManager)
    manager.coverage_num = 2
    manager.coverage_warmup_step = 0
    manager.coverage_warmup_num = 0
    manager.coverage_anneal_step = 200
    manager.coverage_anneal_num = 2
    manager.coverage_policy = policy
    manager.coverage_seed = 42
    manager.coverage_mode_pool = ("prob", "loc_rect", "loc_polar", "highlight")
    manager.rollout_config = SimpleNamespace(n=8, agent=SimpleNamespace(default_agent_loop="scenecot_free_tag_agent"))
    dispatches = []

    async def dispatch(batch):
        forced = batch.non_tensor_batch.get("scenecot_forced_mode", np.full(len(batch), "highlight", dtype=object))
        dispatches.append((len(batch), forced.tolist()))
        actions = np.empty(len(batch), dtype=object)
        actions[:] = [[{"mode": mode}] for mode in forced]
        return DataProto.from_dict(
            tensors={"row": batch.batch["row"].clone()},
            non_tensors={"scenecot_actions": actions},
            meta_info={"timing": {}},
        )

    manager._dispatch = dispatch
    prompts = DataProto.from_dict(
        tensors={"row": torch.arange(16).unsqueeze(1)},
        non_tensors={"uid": np.array(["new-uuid-a"] * 8 + ["new-uuid-b"] * 8, dtype=object)},
        meta_info={"global_steps": 10},
    )
    output = asyncio.run(manager._generate_sequences_async(prompts))
    assert [n for n, _ in dispatches] == [12, 4]
    assert output.batch["row"].flatten().tolist() == list(range(16))
    metrics = output.meta_info["timing"]
    assert metrics["gcs/coverage_count"] == 4
    assert metrics["gcs/free_count"] == 12
    assert metrics[f"gcs/policy/{policy}"] == 1
    if policy == "uniform":
        assert all(metrics[f"gcs/forced_mode/{m}/count"] == 1 for m in manager.coverage_mode_pool)
    # The trainer regenerates UUIDs on resume; allocation must not depend on them.
    prompts.non_tensor_batch["uid"] = np.array(["different-a"] * 8 + ["different-b"] * 8, dtype=object)
    asyncio.run(manager._generate_sequences_async(prompts))
    assert dispatches[1] == dispatches[3]
