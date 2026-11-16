from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_verl_tool_stubs() -> None:
    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        return module

    class BaseTool:
        def __init__(self, config, tool_schema):
            self.config = config
            self.tool_schema = tool_schema
            self.name = getattr(getattr(tool_schema, "function", None), "name", "scenecot_ground")

    class OpenAIFunctionToolSchema:
        def __init__(self, payload):
            self.payload = payload
            self.function = SimpleNamespace(name=payload["function"]["name"])

        @classmethod
        def model_validate(cls, payload):
            return cls(payload)

    class ToolResponse:
        def __init__(self, text="", image=None):
            self.text = text
            self.image = image

    base_tool_module = types.ModuleType("verl.tools.base_tool")
    base_tool_module.BaseTool = BaseTool
    schemas_module = types.ModuleType("verl.tools.schemas")
    schemas_module.OpenAIFunctionToolSchema = OpenAIFunctionToolSchema
    schemas_module.ToolResponse = ToolResponse

    modules = {
        "verl": package("verl"),
        "verl.tools": package("verl.tools"),
        "verl.tools.base_tool": base_tool_module,
        "verl.tools.schemas": schemas_module,
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


_install_verl_tool_stubs()

from scenecot_verl.tools.scenecot_ground_tool import SceneCOTGroundTool


class _FakeBackend:
    def __init__(self, result):
        self.result = result
        self.config = SimpleNamespace(topk=15)
        self.score_calls = 0

    def score(self, **_kwargs):
        self.score_calls += 1
        return self.result


def _fake_result(ranked_ids: list[str], sorted_probs: list[float]):
    candidates = [
        SimpleNamespace(
            obj_id=obj_id,
            label=f"obj_{obj_id}",
            prob=prob,
            rect_loc=None,
            polar_loc=None,
            image_path=None,
        )
        for obj_id, prob in zip(ranked_ids[:15], sorted_probs[:15])
    ]
    return SimpleNamespace(
        ranked_obj_ids=ranked_ids,
        pred_obj_prob=torch.tensor(sorted_probs, dtype=torch.float32),
        top1_prob=sorted_probs[0],
        candidates=candidates,
    )


async def _execute_tool(*, gt_objects: list[str], pc_type: str, result):
    tool = SceneCOTGroundTool(
        {
            "qwen_model_path": "unused",
            "sft_checkpoint_path": "unused",
            "pc_type": pc_type,
            "topk": 15,
        }
    )
    tool.backend = _FakeBackend(result)
    instance_id, _ = await tool.create(
        create_kwargs={
            "scene_id": "scene0000_00",
            "question_id": "q0",
            "dataset_name": "msqa",
            "grounding_prompt": "find the target",
            "gt_objects": gt_objects,
        }
    )
    _, _, metrics = await tool.execute(instance_id, {"query": "target", "mode": "prob", "topk": 15})
    return metrics


def test_grounding_stats_multi_target_uses_best_rank_and_prob():
    result = _fake_result(["3", "2", "4", "1"], [0.9, 0.8, 0.7, 0.1])

    metrics = asyncio.run(_execute_tool(gt_objects=["4", "2"], pc_type="gt", result=result))

    assert metrics["scenecot/has_gt_label"] is True
    assert metrics["scenecot/grd_stats_valid"] is True
    assert metrics["scenecot/grd_hit15"] == 1.0
    assert metrics["scenecot/grd_rank"] == 2
    assert metrics["scenecot/grd_prob"] == pytest.approx(0.8)
    assert metrics["scenecot/grd_topk_ids"] == ["3", "2", "4", "1"]
    assert metrics["scenecot/ground_hit1"] == 0.0
    assert metrics["scenecot/ground_hit5"] == 1.0


def test_grounding_stats_no_gt_outputs_none_grd_metrics():
    result = _fake_result(["3", "2", "4", "1"], [0.9, 0.8, 0.7, 0.1])

    metrics = asyncio.run(_execute_tool(gt_objects=[], pc_type="gt", result=result))

    assert metrics["scenecot/has_gt_label"] is False
    assert metrics["scenecot/grd_stats_valid"] is True
    assert metrics["scenecot/grd_hit15"] is None
    assert metrics["scenecot/grd_rank"] is None
    assert metrics["scenecot/grd_prob"] is None
    assert metrics["scenecot/grd_topk_ids"] is None
    assert metrics["scenecot/grd_topk_probs"] is None


def test_grounding_stats_missing_gt_in_ranked_list_has_no_rank_or_prob():
    result = _fake_result(["3", "2", "4", "1"], [0.9, 0.8, 0.7, 0.1])

    metrics = asyncio.run(_execute_tool(gt_objects=["99"], pc_type="gt", result=result))

    assert metrics["scenecot/has_gt_label"] is True
    assert metrics["scenecot/grd_stats_valid"] is True
    assert metrics["scenecot/grd_hit15"] == 0.0
    assert metrics["scenecot/grd_rank"] is None
    assert metrics["scenecot/grd_prob"] is None
    assert metrics["scenecot/grd_topk_ids"] == ["3", "2", "4", "1"]


def test_grounding_stats_invalid_when_pc_type_is_not_gt():
    result = _fake_result(["3", "2", "4", "1"], [0.9, 0.8, 0.7, 0.1])

    metrics = asyncio.run(_execute_tool(gt_objects=["2"], pc_type="pred", result=result))

    assert metrics["scenecot/has_gt_label"] is True
    assert metrics["scenecot/grd_stats_valid"] is False
    assert metrics["scenecot/grd_hit15"] is None
    assert metrics["scenecot/grd_rank"] is None
    assert metrics["scenecot/grd_prob"] is None
    assert metrics["scenecot/grd_topk_ids"] is None
    assert metrics["scenecot/ground_hit5"] == 0.0


def test_random_tool_render_reuses_last_grounding_result():
    result = _fake_result(["3", "2"], [0.9, 0.8])
    tool = SceneCOTGroundTool(
        {
            "qwen_model_path": "unused",
            "sft_checkpoint_path": "unused",
            "pc_type": "gt",
            "topk": 15,
        }
    )
    backend = _FakeBackend(result)
    tool.backend = backend

    async def run():
        instance_id, _ = await tool.create(
            create_kwargs={
                "scene_id": "scene0000_00",
                "question_id": "q0",
                "dataset_name": "msqa",
                "grounding_prompt": "find the target",
                "gt_objects": [],
            }
        )
        await tool.execute(instance_id, {"query": "target", "mode": "prob", "topk": 15})
        return await tool.render_last_result(
            instance_id,
            query="target",
            mode="loc_rect",
            topk=15,
        )

    response = asyncio.run(run())

    assert backend.score_calls == 1
    assert response.text.startswith("<obj_loc_prob>")
