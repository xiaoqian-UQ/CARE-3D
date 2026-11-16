from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_converter_module():
    path = Path(__file__).resolve().parents[1] / "scripts/eval/convert_verl_jsonl_to_msqa_results.py"
    spec = importlib.util.spec_from_file_location("convert_verl_jsonl_to_msqa_results", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_answer_ignores_tool_instruction_answer_tag():
    module = _load_converter_module()
    output = """<think_grd>
chair
</think_grd>
<highlight_obj>user
<tool_response>
<visual_observation/>
<scenecot_instruction>Close </reasoning> before <answer>. Do not write Answer:</scenecot_instruction>
</tool_response>
assistant
<reasoning>The object is visible.</reasoning>
<answer>chair</answer>"""

    assert module._extract_answer_from_output(output) == "chair"
