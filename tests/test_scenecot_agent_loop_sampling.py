from __future__ import annotations

import asyncio
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest


def _install_verl_stubs() -> None:
    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        return module

    class AgentLoopBase:
        pass

    class AgentLoopOutput:
        pass

    class ToolListWrap:
        tools = []

    class ToolResponse:
        def __init__(self, text="", image=None):
            self.text = text
            self.image = image

    class OpenAIFunctionToolSchema:
        def __init__(self, payload=None):
            self.payload = payload or {}
            function = self.payload.get("function", {}) if isinstance(self.payload, dict) else {}
            self.function = types.SimpleNamespace(name=function.get("name", "scenecot_ground"))

        @classmethod
        def model_validate(cls, payload):
            return cls(payload)

    class TokenOutput:
        pass

    def register(_name: str):
        return lambda cls: cls

    agent_loop_module = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop_module.AgentLoopBase = AgentLoopBase
    agent_loop_module.AgentLoopOutput = AgentLoopOutput
    agent_loop_module.ToolListWrap = ToolListWrap
    agent_loop_module.register = register

    schemas_module = types.ModuleType("verl.tools.schemas")
    schemas_module.OpenAIFunctionToolSchema = OpenAIFunctionToolSchema
    schemas_module.ToolResponse = ToolResponse

    profiler_module = types.ModuleType("verl.utils.profiler")
    profiler_module.simple_timer = lambda *_args, **_kwargs: nullcontext()

    replica_module = types.ModuleType("verl.workers.rollout.replica")
    replica_module.TokenOutput = TokenOutput

    modules = {
        "verl": package("verl"),
        "verl.experimental": package("verl.experimental"),
        "verl.experimental.agent_loop": package("verl.experimental.agent_loop"),
        "verl.experimental.agent_loop.agent_loop": agent_loop_module,
        "verl.tools": package("verl.tools"),
        "verl.tools.schemas": schemas_module,
        "verl.utils": package("verl.utils"),
        "verl.utils.profiler": profiler_module,
        "verl.workers": package("verl.workers"),
        "verl.workers.rollout": package("verl.workers.rollout"),
        "verl.workers.rollout.replica": replica_module,
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


_install_verl_stubs()

from scenecot_verl.agent.scenecot_free_tag_agent_loop import (
    FINAL_ANSWER_STOP,
    SceneCOTFreeTagAgentLoop,
    _autoclose_answer_insert_pos,
    _coac_counterfactual_mode,
    _is_object_token_cache_miss,
    _random_tool_mode,
)


def test_qwen3_coac_counterfactual_defaults_to_mask(monkeypatch):
    monkeypatch.delenv("SCENECOT_COAC_COUNTERFACTUAL_MODE", raising=False)
    assert _coac_counterfactual_mode() == "mask"


def test_qwen3_random_tool_is_deterministic_and_excludes_actual_mode(monkeypatch):
    monkeypatch.delenv("SCENECOT_COAC_VISUAL_ENABLE", raising=False)
    monkeypatch.setenv(
        "SCENECOT_COAC_RANDOM_TOOL_POOL",
        "prob,loc_rect,loc_polar,highlight",
    )
    first = _random_tool_mode("prob", seed_parts=(42, "scene", "question", 3))
    second = _random_tool_mode("prob", seed_parts=(42, "scene", "question", 3))

    assert first == second
    assert first in {"loc_rect", "loc_polar"}


def test_qwen3_random_tool_can_select_highlight_when_visual_coac_is_enabled(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_VISUAL_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_RANDOM_TOOL_POOL", "highlight")

    assert _random_tool_mode("prob", seed_parts=(42, "sample")) == "highlight"


def test_object_token_cache_miss_detection_ignores_other_file_not_found_errors():
    assert _is_object_token_cache_miss(
        FileNotFoundError("Cached SceneCOT object tokens not found. path=/tmp/missing.pt")
    )
    assert not _is_object_token_cache_miss(
        FileNotFoundError("No object image path for top1 object 13 in scene scene0551_00.")
    )


def test_autoclose_eligibility_requires_complete_reasoning_and_nonempty_answer():
    valid_unclosed = (
        "<think_grd>chair</think_grd><list_obj_prob>"
        "<reasoning>Visible evidence.</reasoning><answer>chair"
    )

    assert _autoclose_answer_insert_pos(valid_unclosed, finish_reason="completed") == len(valid_unclosed)
    assert _autoclose_answer_insert_pos(valid_unclosed, finish_reason="length") is None
    assert _autoclose_answer_insert_pos(valid_unclosed + "</answer>", finish_reason="completed") is None
    assert _autoclose_answer_insert_pos(
        valid_unclosed.replace("<answer>chair", "<answer>   "),
        finish_reason="completed",
    ) is None


def _make_loop() -> SceneCOTFreeTagAgentLoop:
    return object.__new__(SceneCOTFreeTagAgentLoop)


def test_final_sampling_params_use_remaining_budget_without_final_overrides():
    loop = _make_loop()
    loop.stop_on_answer = True

    params = loop._sampling_params_for_turn(
        {"temperature": 0.9, "top_p": 0.95},
        max_tokens=0,
        remaining_response_tokens=384,
        final_turn=True,
    )

    assert params["max_tokens"] == 384
    assert params["stop"]
    assert FINAL_ANSWER_STOP in params["stop"]
    assert params["include_stop_str_in_output"] is True
    assert params["temperature"] == 0.9
    assert params["top_p"] == 0.95
    assert "extra_body" not in params


def test_final_sampling_params_can_require_sampled_answer_close(monkeypatch):
    monkeypatch.setenv("SCENECOT_FINAL_IGNORE_EOS", "1")
    loop = _make_loop()
    loop.stop_on_answer = True

    params = loop._sampling_params_for_turn(
        {"include_stop_str_in_output": False},
        max_tokens=64,
        remaining_response_tokens=64,
        final_turn=True,
    )

    assert params["include_stop_str_in_output"] is True
    assert params["ignore_eos"] is True
    assert FINAL_ANSWER_STOP in params["stop"]


def test_first_turn_sampling_params_do_not_use_final_answer_stop():
    loop = _make_loop()
    loop.stop_on_answer = True

    params = loop._sampling_params_for_turn(
        {},
        max_tokens=96,
        remaining_response_tokens=128,
        final_turn=False,
    )

    assert params["max_tokens"] == 96
    assert params["stop"]
    assert FINAL_ANSWER_STOP not in params["stop"]
    assert "ignore_eos" not in params
    assert "extra_body" not in params


def test_first_turn_temperature_env_only_overrides_first_turn(monkeypatch):
    monkeypatch.setenv("SCENECOT_FIRST_TURN_TEMPERATURE", "1.0")
    loop = _make_loop()
    loop.stop_on_answer = True

    first_params = loop._sampling_params_for_turn(
        {"temperature": 0.7, "top_p": 0.9},
        max_tokens=96,
        remaining_response_tokens=128,
        final_turn=False,
        first_turn=True,
    )
    later_params = loop._sampling_params_for_turn(
        {"temperature": 0.7, "top_p": 0.9},
        max_tokens=96,
        remaining_response_tokens=128,
        final_turn=False,
        first_turn=False,
    )
    final_params = loop._sampling_params_for_turn(
        {"temperature": 0.7, "top_p": 0.9},
        max_tokens=96,
        remaining_response_tokens=128,
        final_turn=True,
        first_turn=True,
    )

    assert first_params["temperature"] == 1.0
    assert later_params["temperature"] == 0.7
    assert final_params["temperature"] == 0.7


def test_autoclose_appends_parser_token_with_zero_response_mask(monkeypatch):
    monkeypatch.setenv("SCENECOT_AUTOCLOSE_FINAL_ANSWER", "1")
    loop = _make_loop()
    loop.response_length = 32

    async def encode_text(text: str) -> list[int]:
        assert text == FINAL_ANSWER_STOP
        return [91, 92]

    loop._encode_text = encode_text
    actor_text = (
        "<think_grd>chair</think_grd><list_obj_prob>"
        "<reasoning>Visible evidence.</reasoning><answer>chair"
    )
    current_ids = [1]
    response_ids = [2]
    response_mask = [1]

    repaired, applied, eligible = asyncio.run(
        loop._maybe_autoclose_final_answer(
            actor_text=actor_text,
            finish_reason="completed",
            current_ids=current_ids,
            response_ids=response_ids,
            response_mask=response_mask,
        )
    )

    assert repaired.endswith("</answer>")
    assert (applied, eligible) == (1, 1)
    assert current_ids[-2:] == [91, 92]
    assert response_ids[-2:] == [91, 92]
    assert response_mask[-2:] == [0, 0]
