from __future__ import annotations

from pathlib import Path

import pytest

from scenecot_verl.rewards.scenecot_reward import (
    _get_agent_reward_progress,
    _merge_scenecot_reward_extra_info,
    score_scenecot_solution,
)


DEFAULT_EXTRA_INFO = {
    "dataset_name": "msqa",
    "question": "Which chair is near the table?",
    "qa_type": "spatial",
}


def _score(
    solution: str,
    extra_info: dict | None = None,
    ground_truth: list[str] | None = None,
    **kwargs,
):
    return score_scenecot_solution(
        data_source="scenecot",
        solution_str=solution,
        ground_truth=ground_truth or ["chair"],
        extra_info=extra_info or dict(DEFAULT_EXTRA_INFO),
        **kwargs,
    )


def _valid_solution(
    answer: str = "chair",
    reasoning: str = "The object is visible.",
    query: str = "chair near table",
    tool_tag: str = "list_obj_loc_prob",
) -> str:
    return (
        f"<think_grd>\n{query}\n</think_grd>\n"
        f"<{tool_tag}>"
        f"<reasoning>{reasoning}</reasoning>"
        f"<answer>{answer}</answer>"
    )


def _observed_solution(final_text: str, query: str = "chair near table", tool_tag: str = "list_obj_loc_prob") -> str:
    return (
        f"<think_grd>\n{query}\n</think_grd>\n<{tool_tag}>user\n"
        "<tool_response>\n<visual_observation/>\n</tool_response>\n"
        f"assistant\n{final_text}"
    )


@pytest.mark.parametrize("step", [0, 199])
def test_agent_reward_progress_switch_early_stage_by_default(step):
    reward, metrics = _score(_valid_solution("table"), ground_truth=["chair"], global_step=step)

    assert metrics["reward/agent_reward_progress"] == 0.0
    assert metrics["reward/progress_switch_enabled"] == 1
    assert metrics["reward/w_format"] == pytest.approx(1.2)
    assert metrics["reward/w_answer"] == pytest.approx(0.8)
    assert "reward/w_query" not in metrics
    assert reward == pytest.approx(1.2)


@pytest.mark.parametrize("step", [200, 800])
def test_agent_reward_progress_late_stage_no_longer_changes_score(step):
    reward, metrics = _score(_valid_solution("table"), ground_truth=["chair"], global_step=step)

    assert metrics["reward/agent_reward_progress"] == 1.0
    assert metrics["reward/w_format"] == pytest.approx(0.8)
    assert metrics["reward/w_answer"] == pytest.approx(1.2)
    assert "reward/w_query" not in metrics
    assert reward == pytest.approx(0.8)


def test_agent_reward_progress_switch_step_env(monkeypatch):
    monkeypatch.setenv("SCENECOT_ENABLE_PROGRESS_SWITCH", "1")
    monkeypatch.setenv("SCENECOT_PROGRESS_SWITCH_STEP", "300")

    assert _get_agent_reward_progress(global_step=299) == 0.0
    assert _get_agent_reward_progress(global_step=300) == 1.0


def test_agent_reward_progress_switch_reads_rollout_step(monkeypatch):
    monkeypatch.setenv("SCENECOT_ENABLE_PROGRESS_SWITCH", "1")
    monkeypatch.setenv("SCENECOT_PROGRESS_SWITCH_STEP", "67")
    monkeypatch.setenv("SCENECOT_FORMAT_W_START", "1.5")
    monkeypatch.setenv("SCENECOT_FORMAT_W_END", "0.8")
    monkeypatch.setenv("SCENECOT_ANSWER_W_START", "0")
    monkeypatch.setenv("SCENECOT_ANSWER_W_END", "1.2")

    early_extra = {**DEFAULT_EXTRA_INFO, "rollout_step": 1}
    early_reward, early_metrics = _score(
        _valid_solution("table"),
        extra_info=early_extra,
        ground_truth=["chair"],
    )
    assert early_metrics["reward/agent_reward_progress"] == 0.0
    assert early_metrics["reward/w_format"] == pytest.approx(1.5)
    assert early_metrics["reward/w_answer"] == pytest.approx(0.0)
    assert early_reward == pytest.approx(1.5)

    late_extra = {**DEFAULT_EXTRA_INFO, "rollout_step": 67}
    late_reward, late_metrics = _score(
        _valid_solution("table"),
        extra_info=late_extra,
        ground_truth=["chair"],
    )
    assert late_metrics["reward/agent_reward_progress"] == 1.0
    assert late_metrics["reward/w_format"] == pytest.approx(0.8)
    assert late_metrics["reward/w_answer"] == pytest.approx(1.2)
    assert late_reward == pytest.approx(0.8)


@pytest.mark.parametrize(
    "batch_fields",
    [
        {"rollout_step": 12},
        {"tool_extra_fields": {"rollout_step": 12}},
    ],
)
def test_reward_recompute_normalizes_rollout_step(batch_fields):
    merged = _merge_scenecot_reward_extra_info(dict(DEFAULT_EXTRA_INFO), batch_fields)

    assert merged["rollout_step"] == 12
    assert merged["global_step"] == 12


def test_agent_reward_force_progress_env(monkeypatch):
    monkeypatch.setenv("SCENECOT_FORCE_AGENT_REWARD_PROGRESS", "0.5")
    reward, metrics = _score(_valid_solution("table"), ground_truth=["chair"], global_step=0)

    assert metrics["reward/agent_reward_progress"] == pytest.approx(0.5)
    assert metrics["reward/w_format"] == pytest.approx(1.0)
    assert metrics["reward/w_answer"] == pytest.approx(1.0)
    assert "reward/w_query" not in metrics
    assert reward == pytest.approx(1.0)


def test_agent_reward_progress_defaults_to_late_stage_when_step_missing():
    reward, metrics = _score(_valid_solution("table"), ground_truth=["chair"])

    assert metrics["reward/agent_reward_progress"] == 1.0
    assert reward == pytest.approx(0.8)


def test_case_a_valid_protocol_correct_answer_gets_binary_format_and_answer_reward():
    reward, metrics = _score(_valid_solution("chair"), ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["format_reward"] == 1.0
    assert metrics["format_gate_tier"] == "valid"
    assert metrics["answer_reward"] == 1.0
    assert "tool_reward" not in metrics
    assert "query_reward" not in metrics
    assert metrics["agent_reward_enabled"] == 1
    assert metrics["full_protocol_valid"] == 1
    assert metrics["first_turn_valid"] == 1
    assert metrics["final_turn_valid"] == 1
    assert metrics["catastrophic_invalid"] == 0
    assert metrics["answer_parsed_from_answer_tag"] == 1
    assert metrics["pred_answer_raw"] == "chair"
    assert metrics["penalty_total"] == 0.0


def test_case_b_valid_protocol_wrong_answer_keeps_format_and_agent_reward():
    reward, metrics = _score(_valid_solution("table"), ground_truth=["chair"])

    assert reward == pytest.approx(0.8)
    assert metrics["format_reward"] == 1.0
    assert metrics["answer_reward"] == 0.0
    assert "tool_reward" not in metrics
    assert "query_reward" not in metrics
    assert metrics["full_protocol_valid"] == 1
    assert metrics["answer_parsed_from_answer_tag"] == 1


def test_coac_tool_credit_is_default_off_and_optional(monkeypatch):
    extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_delta": 0.25,
        "coac_valid": True,
        "coac_obs_text": "The observation discusses a sofa.",
    }
    reward_off, metrics_off = _score(_valid_solution("table"), extra_info=extra, ground_truth=["chair"])

    assert reward_off == pytest.approx(0.8)
    assert metrics_off["coac_enabled"] == 0
    assert metrics_off["coac_r_tool"] == 0.0

    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    reward_on, metrics_on = _score(_valid_solution("table"), extra_info=extra, ground_truth=["chair"])

    assert metrics_on["coac_enabled"] == 1
    assert metrics_on["coac_r_tool"] == pytest.approx(0.5)
    assert metrics_on["coac_tool_credit_source"] == "coac_delta"
    assert reward_on == pytest.approx(0.9)


def test_coac_skipped_highlight_falls_back_to_fixed_tool_credit(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_delta": 0.0,
        "coac_valid": False,
        "coac_obs_text": "<visual_observation/>",
    }

    reward, metrics = _score(
        _valid_solution("table", tool_tag="highlight_obj"),
        extra_info=extra,
        ground_truth=["chair"],
    )

    assert metrics["coac_enabled"] == 1
    assert metrics["coac_valid"] == 0
    assert metrics["coac_r_tool"] == pytest.approx(0.5)
    assert metrics["coac_fixed_r_tool"] == pytest.approx(0.5)
    assert metrics["coac_fallback_credit"] == pytest.approx(0.5)
    assert metrics["coac_tool_credit_source"] == "fixed_fallback"
    assert metrics["coac_fixed_fallback_applied"] == 1
    assert metrics["coac/fallback_credit_rate"] == pytest.approx(1.0)
    assert reward == pytest.approx(0.9)


def test_coac_fallback_credit_env_override(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_FALLBACK_CREDIT", "0.25")
    extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_delta": 0.0,
        "coac_valid": False,
        "coac_obs_text": "<visual_observation/>",
    }

    reward, metrics = _score(
        _valid_solution("table", tool_tag="highlight_obj"),
        extra_info=extra,
        ground_truth=["chair"],
    )

    assert metrics["coac_r_tool"] == pytest.approx(0.25)
    assert metrics["coac_fallback_credit"] == pytest.approx(0.25)
    assert metrics["coac/fallback_credit_rate"] == pytest.approx(1.0)
    assert reward == pytest.approx(0.85)


def test_coac_copy_discount_is_monitor_only_by_default(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_delta": 0.5,
        "coac_valid": True,
        "coac_obs_text": "chair chair chair",
    }

    reward, metrics = _score(_valid_solution("table"), extra_info=extra, ground_truth=["chair"])

    assert metrics["coac_copy_overlap"] == pytest.approx(1.0)
    assert metrics["coac_copy_discount_enabled"] == 0
    assert metrics["coac_copy_discount_applied"] == 0
    assert metrics["coac_r_tool"] == pytest.approx(1.0)
    assert reward == pytest.approx(1.0)


def test_coac_copy_discount_reduces_surface_copy_credit_when_enabled(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_COPY_DISCOUNT_ENABLE", "1")
    extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_delta": 0.5,
        "coac_valid": True,
        "coac_obs_text": "chair chair chair",
    }

    reward, metrics = _score(_valid_solution("table"), extra_info=extra, ground_truth=["chair"])

    assert metrics["coac_copy_discount_enabled"] == 1
    assert metrics["coac_copy_discount_applied"] == 1
    assert metrics["coac_r_tool"] == pytest.approx(0.5)
    assert reward == pytest.approx(0.9)


def test_reward_max_2_2_preserves_tool_credit_for_correct_answer(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_GROUP_ZSCORE", "1")

    high_tool_extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_valid": True,
        "coac_delta": 0.0,
        "coac_r_tool_override": 0.93,
        "coac_obs_text": "chair",
    }
    low_tool_extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_valid": True,
        "coac_delta": 0.0,
        "coac_r_tool_override": 0.10,
        "coac_obs_text": "chair",
    }

    high_reward, high_metrics = _score(_valid_solution("chair"), extra_info=high_tool_extra, ground_truth=["chair"])
    low_reward, low_metrics = _score(_valid_solution("chair"), extra_info=low_tool_extra, ground_truth=["chair"])

    assert high_metrics["reward/w_format"] == pytest.approx(0.8)
    assert high_metrics["reward/w_answer"] == pytest.approx(1.2)
    assert high_metrics["coac_r_tool"] == pytest.approx(0.93)
    assert low_metrics["coac_r_tool"] == pytest.approx(0.10)
    assert high_reward == pytest.approx(2.186)
    assert low_reward == pytest.approx(2.020)
    assert high_reward > low_reward


def test_reward_max_env_2_0_reproduces_old_tool_credit_clip(monkeypatch):
    monkeypatch.setenv("SCENECOT_COAC_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_COAC_GROUP_ZSCORE", "1")
    monkeypatch.setenv("SCENECOT_REWARD_MAX", "2.0")
    high_tool_extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_valid": True,
        "coac_delta": 0.0,
        "coac_r_tool_override": 0.93,
        "coac_obs_text": "chair",
    }
    low_tool_extra = {
        **DEFAULT_EXTRA_INFO,
        "coac_valid": True,
        "coac_delta": 0.0,
        "coac_r_tool_override": 0.10,
        "coac_obs_text": "chair",
    }

    high_reward, _ = _score(_valid_solution("chair"), extra_info=high_tool_extra, ground_truth=["chair"])
    low_reward, _ = _score(_valid_solution("chair"), extra_info=low_tool_extra, ground_truth=["chair"])

    assert high_reward == pytest.approx(2.0)
    assert low_reward == pytest.approx(2.0)


def test_case_c_answer_only_does_not_get_answer_reward_without_protocol():
    reward, metrics = _score("<answer>chair</answer>", ground_truth=["chair"])

    assert reward == pytest.approx(0.0)
    assert metrics["full_protocol_valid"] == 0
    assert metrics["first_turn_valid"] == 0
    assert metrics["final_turn_valid"] == 0
    assert metrics["format_reward"] == 0.0
    assert metrics["answer_reward"] == 0.0
    assert metrics["answer_parsed_from_answer_tag"] == 1
    assert metrics["answer_judge_called"] == 0


def test_case_d_reasoning_correct_but_answer_wrong_scores_only_answer_tag():
    solution = _valid_solution("table", reasoning="The correct answer is chair.")
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(0.8)
    assert metrics["full_protocol_valid"] == 1
    assert metrics["format_reward"] == 1.0
    assert metrics["answer_reward"] == 0.0
    assert metrics["pred_answer_raw"] == "table"


def test_protocol_violation_gets_no_format_or_answer_reward():
    reward, metrics = _score("prefix text\n" + _valid_solution("chair"), ground_truth=["chair"])

    assert reward == pytest.approx(0.0)
    assert metrics["format_reward"] == 0.0
    assert metrics["format_gate_tier"] == "invalid"
    assert metrics["answer_reward"] == 0.0
    assert "tool_reward" not in metrics


def test_tool_execution_metrics_do_not_change_reward_score():
    tool_error_extra = {
        **DEFAULT_EXTRA_INFO,
        "scenecot_tool_metrics": {"tool_call_count": 1, "tool_error_count": 1},
    }
    reward_error, metrics_error = _score(
        _valid_solution("chair"),
        extra_info=tool_error_extra,
        ground_truth=["chair"],
    )
    reward_wrong, metrics_wrong = _score(_valid_solution("table"), ground_truth=["chair"])

    assert reward_error == pytest.approx(2.0)
    assert metrics_error["rollout_tool_call_count"] == 1
    assert metrics_error["rollout_tool_error_count"] == 1
    assert "tool_reward" not in metrics_error
    assert reward_wrong == pytest.approx(0.8)
    assert "tool_reward" not in metrics_wrong


def test_tool_metrics_passthrough_includes_cache_miss_only():
    extra = {
        **DEFAULT_EXTRA_INFO,
        "scenecot_tool_metrics": {
            "tool_call_count": 1,
            "tool_error_count": 0,
            "cache_miss_count": 0,
        },
    }

    reward, metrics = _score(_valid_solution("chair", tool_tag="highlight_obj"), extra_info=extra, ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["rollout_has_tool_metrics"] == 1
    assert metrics["rollout_tool_call_count"] == 1
    assert metrics["rollout_tool_error_count"] == 0
    assert metrics["rollout_cache_miss_count"] == 0


def test_self_closed_answer_remains_valid_without_autoclose_flag():
    response_text = _valid_solution("chair")
    extra = {
        **DEFAULT_EXTRA_INFO,
        "response_text": response_text,
        "scenecot_tool_metrics": {"tool_call_count": 1, "tool_error_count": 0},
    }

    reward, metrics = _score("unused raw solution", extra_info=extra, ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["format_gate_tier"] == "valid"
    assert metrics["final_turn_valid"] == 1


def test_empty_answer_stays_catastrophic_without_autoclose_override():
    response_text = _valid_solution("   ")
    extra = {
        **DEFAULT_EXTRA_INFO,
        "response_text": response_text,
        "scenecot_tool_metrics": {"tool_call_count": 1, "tool_error_count": 0},
    }

    reward, metrics = _score("unused raw solution", extra_info=extra, ground_truth=["chair"])

    assert reward == pytest.approx(-1.0)
    assert metrics["format_gate_tier"] == "catastrophic"
    assert metrics["format_reward"] == 0.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["answer_reward"] == 0.0


def test_catastrophic_reward_can_be_neutralized_for_format_bootstrap(monkeypatch):
    monkeypatch.setenv("SCENECOT_CATASTROPHIC_REWARD", "0")

    reward, metrics = _score("." * 64, ground_truth=["chair"])

    assert metrics["catastrophic_invalid"] == 1
    assert metrics["catastrophic_reward_value"] == 0.0
    assert reward == 0.0


def test_final_reward_formula_snapshot_fixed_cases():
    cases = [
        ("valid_correct", _valid_solution("chair"), ["chair"], dict(DEFAULT_EXTRA_INFO), 2.0),
        ("valid_wrong", _valid_solution("table"), ["chair"], dict(DEFAULT_EXTRA_INFO), 0.8),
        ("invalid_prefix", "prefix text\n" + _valid_solution("chair"), ["chair"], dict(DEFAULT_EXTRA_INFO), 0.0),
        (
            "unclosed_answer_invalid",
            "<think_grd>\nchair near table\n</think_grd>\n"
            "<list_obj_loc_prob>"
            "<reasoning>The object is visible.</reasoning>"
            "<answer>chair",
            ["chair"],
            dict(DEFAULT_EXTRA_INFO),
            0.0,
        ),
        (
            "tool_error",
            _valid_solution("chair"),
            ["chair"],
            {**DEFAULT_EXTRA_INFO, "scenecot_tool_metrics": {"tool_call_count": 1, "tool_error_count": 1}},
            2.0,
        ),
        ("catastrophic", "." * 64, ["chair"], dict(DEFAULT_EXTRA_INFO), -1.0),
    ]

    snapshot = {}
    for name, solution, ground_truth, extra_info, expected in cases:
        reward, metrics = _score(solution, extra_info=extra_info, ground_truth=ground_truth)
        snapshot[name] = {
            "score": reward,
            "format_gate_tier": metrics["format_gate_tier"],
            "answer_reward": metrics["answer_reward"],
        }
        assert reward == pytest.approx(expected)

    assert snapshot == {
        "valid_correct": {"score": pytest.approx(2.0), "format_gate_tier": "valid", "answer_reward": 1.0},
        "valid_wrong": {"score": pytest.approx(0.8), "format_gate_tier": "valid", "answer_reward": 0.0},
        "invalid_prefix": {"score": pytest.approx(0.0), "format_gate_tier": "invalid", "answer_reward": 0.0},
        "unclosed_answer_invalid": {"score": pytest.approx(0.0), "format_gate_tier": "invalid", "answer_reward": 0.0},
        "tool_error": {"score": pytest.approx(2.0), "format_gate_tier": "valid", "answer_reward": 1.0},
        "catastrophic": {
            "score": pytest.approx(-1.0),
            "format_gate_tier": "catastrophic",
            "answer_reward": 0.0,
        },
    }


def test_agent_reward_disable_switch_uses_unit_weights(monkeypatch):
    monkeypatch.setenv("SCENECOT_ENABLE_AGENT_REWARD", "0")

    reward, metrics = _score(_valid_solution("chair"), ground_truth=["chair"])
    assert reward == pytest.approx(2.0)
    assert metrics["agent_reward_enabled"] == 0
    assert metrics["w_format"] == pytest.approx(1.0)
    assert metrics["w_answer"] == pytest.approx(1.0)
    assert metrics["full_protocol_valid"] == 1

    invalid_reward, invalid_metrics = _score(
        "<think_grd>\nchair near table\n</think_grd>\n<list_obj_loc_prob><answer>chair</answer>",
        ground_truth=["chair"],
    )
    assert invalid_reward == pytest.approx(0.0)
    assert invalid_metrics["catastrophic_invalid"] == 0
    assert invalid_metrics["full_protocol_valid"] == 0

    catastrophic_reward, catastrophic_metrics = _score("." * 64, ground_truth=["chair"])
    assert catastrophic_reward == -1.0
    assert catastrophic_metrics["catastrophic_invalid"] == 1


def test_case_e_repeated_sentence_loop_is_catastrophic_invalid():
    repeated = (
        "The object is clearly visible in the observation. "
        "The object is clearly visible in the observation. "
        "The object is clearly visible in the observation."
    )
    reward, metrics = _score(_valid_solution("chair", reasoning=repeated), ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["hard_invalid_reason"] == "repeated_sentence_loop"
    assert metrics["full_protocol_valid"] == 0
    assert metrics["format_reward"] == 0.0
    assert metrics["answer_reward"] == 0.0
    assert metrics["penalty_repeated_sentence"] == 1.0


def test_case_f_duplicated_answer_is_protocol_invalid_without_answer_reward():
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>chair</answer><answer>chair</answer>"
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(0.0)
    assert metrics["full_protocol_valid"] == 0
    assert metrics["final_turn_valid"] == 0
    assert metrics["format_reward"] == 0.0
    assert metrics["format_gate_tier"] == "invalid"
    assert metrics["answer_reward"] == 0.0
    assert metrics["answer_parsed_from_answer_tag"] == 0
    assert metrics["final_turn_invalid_reason"] == "final_turn_answer_count"


def test_unclosed_answer_with_extractable_content_is_invalid_and_does_not_score_answer():
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>chair"
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(0.0)
    assert metrics["full_protocol_valid"] == 0
    assert metrics["format_reward"] == 0.0
    assert metrics["format_gate_tier"] == "invalid"
    assert metrics["answer_reward"] == 0.0
    assert "tool_reward" not in metrics
    assert metrics["answer_parsed_from_answer_tag"] == 1
    assert metrics["answer_parsed_from_unclosed_answer_tag"] == 1
    assert metrics["final_answer_unclosed_extractable"] == 1
    assert metrics["final_turn_invalid_reason"] == "final_turn_answer_count"


def test_agent_autoclosed_answer_gets_minor_format_credit_and_answer_reward():
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>chair</answer>"
    )
    extra = {
        **DEFAULT_EXTRA_INFO,
        "scenecot_tool_metrics": {
            "tool_call_count": 1,
            "tool_error_count": 0,
            "final_answer_autoclosed_count": 1,
            "final_answer_autoclose_eligible_count": 1,
        },
    }

    reward, metrics = _score(solution, extra_info=extra, ground_truth=["chair"])

    assert reward == pytest.approx(1.44)
    assert metrics["format_reward"] == pytest.approx(0.3)
    assert metrics["format_gate_tier"] == "minor_violation"
    assert metrics["format_minor_violation"] == 1
    assert metrics["structural_protocol_valid"] == 1
    assert metrics["full_protocol_valid"] == 0
    assert metrics["final_turn_valid"] == 1
    assert metrics["answer_reward"] == 1.0
    assert metrics["final_answer_autoclosed_count"] == 1


def test_agent_autoclosed_answer_can_be_parser_only_without_training_credit(monkeypatch):
    monkeypatch.setenv("SCENECOT_AUTOCLOSE_FORMAT_REWARD", "0")
    monkeypatch.setenv("SCENECOT_SCORE_AUTOCLOSED_ANSWER", "0")
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>chair</answer>"
    )
    extra = {
        **DEFAULT_EXTRA_INFO,
        "scenecot_tool_metrics": {
            "tool_call_count": 1,
            "tool_error_count": 0,
            "final_answer_autoclosed_count": 1,
            "final_answer_autoclose_eligible_count": 1,
        },
    }

    reward, metrics = _score(solution, extra_info=extra, ground_truth=["chair"])

    assert reward == pytest.approx(0.0)
    assert metrics["format_reward"] == 0.0
    assert metrics["format_gate_tier"] == "minor_violation"
    assert metrics["answer_reward"] == 0.0
    assert metrics["answer_autoclose_scoring_enabled"] == 0
    assert metrics["full_protocol_valid"] == 0


def test_agent_autoclose_keeps_progress_prefix_credit_without_unlocking_answer(monkeypatch):
    monkeypatch.setenv("SCENECOT_PROGRESS_FORMAT_REWARD_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_AUTOCLOSE_FORMAT_REWARD", "0")
    # Leave SCORE_AUTOCLOSED_ANSWER at its backward-compatible default. A
    # progress-only format score must still not unlock answer correctness.
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>chair</answer>"
    )
    extra = {
        **DEFAULT_EXTRA_INFO,
        "scenecot_tool_metrics": {
            "tool_call_count": 1,
            "tool_error_count": 0,
            "final_answer_autoclosed_count": 1,
            "final_answer_autoclose_eligible_count": 1,
        },
    }

    reward, metrics = _score(solution, extra_info=extra, ground_truth=["chair"])

    assert metrics["format_gate_tier"] == "minor_violation"
    assert metrics["format_progress_first_turn"] == pytest.approx(0.25)
    assert metrics["format_progress_reasoning"] == pytest.approx(0.25)
    assert metrics["format_progress_answer_body"] == pytest.approx(0.20)
    assert metrics["format_progress_answer_close"] == 0.0
    assert metrics["format_reward"] == pytest.approx(0.70)
    assert metrics["answer_reward"] == 0.0
    assert reward == pytest.approx(0.8 * 0.70)


def test_coac_recompute_uses_flattened_repaired_response_text():
    repaired_response = _valid_solution("chair")
    contaminated_decoded_transcript = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>Assistant: <answer>wrong</answer></reasoning>"
        "<answer>wrong</answer>"
    )
    recompute_extra_info = _merge_scenecot_reward_extra_info(
        dict(DEFAULT_EXTRA_INFO),
        {
            "extra_info": dict(DEFAULT_EXTRA_INFO),
            # AgentLoop._postprocess has flattened these rollout extra_fields.
            "response_text": repaired_response,
            "scenecot_metrics": {
                "tool_call_count": 1,
                "tool_error_count": 0,
                "final_answer_autoclosed_count": 1,
                "final_answer_autoclose_eligible_count": 1,
            },
        },
    )

    reward, metrics = _score(
        contaminated_decoded_transcript,
        extra_info=recompute_extra_info,
        ground_truth=["chair"],
    )

    assert recompute_extra_info["response_text"] == repaired_response
    assert metrics["used_actor_response_text"] == 1
    assert metrics["format_reward"] == pytest.approx(0.3)
    assert metrics["format_minor_violation"] == 1
    assert metrics["answer_reward"] == 1.0
    assert reward == pytest.approx(1.44)


def test_unclosed_answer_without_content_stays_zero_format():
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>   "
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(0.0)
    assert metrics["format_reward"] == 0.0
    assert metrics["format_gate_tier"] == "invalid"
    assert metrics["answer_reward"] == 0.0
    assert metrics["answer_parsed_from_unclosed_answer_tag"] == 0
    assert metrics["final_answer_unclosed_extractable"] == 0


def test_empty_final_answer_is_catastrophic_when_final_format_is_attempted():
    reward, metrics = _score(_valid_solution(""), ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["hard_invalid_reason"] == "empty_final_answer"
    assert metrics["answer_reward"] == 0.0


def test_final_turn_tool_request_is_catastrophic():
    solution = _valid_solution("chair") + "<highlight_obj>"
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["hard_invalid_reason"] == "final_turn_tool_tag"
    assert metrics["answer_reward"] == 0.0


def test_dot_only_output_is_catastrophic():
    reward, metrics = _score("." * 64, ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["hard_invalid_reason"] == "dot_only_output"
    assert metrics["penalty_repeated_dot"] == 1.0
    assert metrics["penalty_dot_only"] == 1


def test_invalid_protocol_missing_reasoning_is_zero_minus_penalty_not_catastrophic():
    solution = "<think_grd>\nchair near table\n</think_grd>\n<list_obj_loc_prob><answer>chair</answer>"
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(0.0)
    assert metrics["catastrophic_invalid"] == 0
    assert metrics["full_protocol_valid"] == 0
    assert metrics["final_turn_valid"] == 0
    assert metrics["format_reward"] == 0.0
    assert metrics["answer_reward"] == 0.0


def test_progressive_format_reward_is_hierarchical_and_keeps_answer_gated(monkeypatch):
    monkeypatch.setenv("SCENECOT_PROGRESS_FORMAT_REWARD_ENABLE", "1")
    monkeypatch.setenv("SCENECOT_SCORE_AUTOCLOSED_ANSWER", "0")
    unclosed = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>chair"
    )

    reward, metrics = _score(unclosed, ground_truth=["chair"])

    assert metrics["full_protocol_valid"] == 0
    assert metrics["format_gate_tier"] == "progress"
    assert metrics["format_progress_first_turn"] == pytest.approx(0.25)
    assert metrics["format_progress_reasoning"] == pytest.approx(0.25)
    assert metrics["format_progress_answer_body"] == pytest.approx(0.20)
    assert metrics["format_progress_answer_close"] == 0.0
    assert metrics["format_reward"] == pytest.approx(0.70)
    assert metrics["answer_reward"] == 0.0
    assert reward == pytest.approx(0.8 * 0.70)


def test_progressive_format_reward_does_not_skip_missing_reasoning(monkeypatch):
    monkeypatch.setenv("SCENECOT_PROGRESS_FORMAT_REWARD_ENABLE", "1")
    missing_reasoning = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob><answer>chair</answer>"
    )

    reward, metrics = _score(missing_reasoning, ground_truth=["chair"])

    assert metrics["format_gate_tier"] == "progress"
    assert metrics["format_reward"] == pytest.approx(0.25)
    assert metrics["format_progress_first_turn"] == pytest.approx(0.25)
    assert metrics["format_progress_reasoning"] == 0.0
    assert metrics["format_progress_answer_body"] == 0.0
    assert metrics["format_progress_answer_close"] == 0.0
    assert metrics["answer_reward"] == 0.0
    assert reward == pytest.approx(0.8 * 0.25)


def test_observed_response_with_tool_response_and_assistant_marker_is_valid():
    reward, metrics = _score(
        _observed_solution("<reasoning>The object is visible.</reasoning>\n<answer>chair</answer>"),
        ground_truth=["chair"],
    )

    assert reward == pytest.approx(2.0)
    assert metrics["full_protocol_valid"] == 1
    assert metrics["first_turn_valid"] == 1
    assert metrics["final_turn_valid"] == 1
    assert metrics["answer_reward"] == 1.0


def test_tool_instruction_answer_tag_does_not_pollute_parsed_answer():
    solution = """<think_grd>
chair near table
</think_grd>
<list_obj_loc_prob>user
<tool_response>
<visual_observation/>
<scenecot_instruction>Close </reasoning> before <answer>wrong</answer>.</scenecot_instruction>
</tool_response>
assistant
<reasoning>The object is visible.</reasoning>
<answer>chair</answer>"""

    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["pred_answer_raw"] == "chair"
    assert metrics["answer_reward"] == 1.0
    assert metrics["full_protocol_valid"] == 1


def test_final_length_penalty_subtracts_from_valid_protocol(monkeypatch):
    monkeypatch.setenv("SCENECOT_TOO_LONG_TOKENIZER_PATH", "/definitely/missing/tokenizer")
    monkeypatch.setenv("SCENECOT_FINAL_REASONING_SOFT_TOKEN_LIMIT", "2")
    monkeypatch.setenv("SCENECOT_FINAL_REASONING_HARD_TOKEN_LIMIT", "10")
    monkeypatch.setenv("SCENECOT_FINAL_REASONING_MAX_PENALTY", "0.10")

    reward, metrics = _score(_valid_solution("chair", reasoning="one two three four"), ground_truth=["chair"])

    assert metrics["full_protocol_valid"] == 1
    assert metrics["penalty_final_reasoning_length"] > 0.0
    assert metrics["penalty_final_length"] > 0.0
    assert reward < 2.0


def test_paired_tool_xml_is_accepted_for_first_turn_protocol():
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>chair</list_obj_loc_prob>"
        "<reasoning>The object is visible.</reasoning>"
        "<answer>chair</answer>"
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["first_turn_valid"] == 1
    assert metrics["full_protocol_valid"] == 1


def test_case_g_repeated_sentence_inside_truncated_reasoning_is_catastrophic():
    solution = (
        "<think_grd>\nfind the object for seating\n</think_grd>\n"
        "<highlight_obj>\n"
        "<reasoning>\n"
        "The object used for seating is a black chair.\n"
        "It is at your 12 o'clock.\n"
        "It is at your 12 o'clock.\n"
        "It is at your 12 o'clock.\n"
        "It is at your 12 o'clock."
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["hard_invalid_reason"] == "repeated_sentence_loop"
    assert metrics["format_reward"] == 0.0
    assert metrics["answer_reward"] == 0.0
    assert metrics["repeated_sentence_loop"] == 1
    assert metrics["max_repeated_sentence_run"] >= 4
    assert metrics["repetition_checked_full_response"] == 1
    assert metrics["repetition_checked_final_turn"] == 1
    assert metrics["repetition_checked_reasoning_span"] == 1


def test_case_h_repeated_phrase_without_punctuation_is_ngram_catastrophic():
    repeated = " ".join(["at your 12 o'clock"] * 5)
    solution = (
        "<think_grd>\nchair location\n</think_grd>\n"
        "<highlight_obj>"
        f"<reasoning>{repeated}"
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["repeated_ngram_loop"] == 1
    assert metrics["repeated_ngram_ratio"] > 0.5
    assert metrics["hard_invalid_reason"] == "repeated_sentence_loop"


def test_case_i_concise_reasoning_with_repeated_object_is_not_loop():
    solution = (
        "<think_grd>\nchair near table\n</think_grd>\n"
        "<list_obj_loc_prob>"
        "<reasoning>The tool observation highlights a chair. "
        "The chair is the likely object used for sitting.</reasoning>"
        "<answer>chair</answer>"
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["catastrophic_invalid"] == 0
    assert metrics["repeated_sentence_loop"] == 0
    assert metrics["repeated_ngram_loop"] == 0
    assert metrics["full_protocol_valid"] == 1


def test_repetition_detection_ignores_repeated_tool_observation_when_actor_text_is_clean():
    repeated_observation = " ".join(["chair"] * 20)
    solution = (
        "<think_grd>\nchairs near me\n</think_grd>\n"
        "<list_obj_prob>user\n"
        f"<tool_response>\n<obj_prob> {repeated_observation} </obj_prob>\n</tool_response>\n"
        "assistant\n"
        "<reasoning>The observation identifies a chair.</reasoning>"
        "<answer>chair</answer>"
    )
    actor_text = (
        "<think_grd>\nchairs near me\n</think_grd>\n"
        "<list_obj_prob>"
        "<reasoning>The observation identifies a chair.</reasoning>"
        "<answer>chair</answer>"
    )

    reward, metrics = _score(
        solution,
        extra_info={**DEFAULT_EXTRA_INFO, "scenecot_actor_generated_text": actor_text},
        ground_truth=["chair"],
    )

    assert reward == pytest.approx(2.0)
    assert metrics["penalty_scope"] == "response_mask_actor_tokens"
    assert metrics["repetition_scope"] == "response_mask_actor_tokens"
    assert metrics["catastrophic_invalid"] == 0
    assert metrics["repeated_sentence_loop"] == 0
    assert metrics["repeated_ngram_loop"] == 0
    assert metrics["full_protocol_valid"] == 1


def test_repetition_detection_still_flags_actor_generated_loop():
    repeated_reasoning = " ".join(["at your 12 o'clock"] * 5)
    solution = (
        "<think_grd>\nchair location\n</think_grd>\n"
        "<list_obj_prob>user\n"
        "<tool_response>\n<obj_prob> chair 0.90 chair 0.80 </obj_prob>\n</tool_response>\n"
        "assistant\n"
        f"<reasoning>{repeated_reasoning}</reasoning>"
        "<answer>chair</answer>"
    )
    actor_text = (
        "<think_grd>\nchair location\n</think_grd>\n"
        "<list_obj_prob>"
        f"<reasoning>{repeated_reasoning}</reasoning>"
        "<answer>chair</answer>"
    )

    reward, metrics = _score(
        solution,
        extra_info={**DEFAULT_EXTRA_INFO, "scenecot_actor_generated_text": actor_text},
        ground_truth=["chair"],
    )

    assert reward == -1.0
    assert metrics["penalty_scope"] == "response_mask_actor_tokens"
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["repeated_ngram_loop"] == 1
    assert metrics["hard_invalid_reason"] == "repeated_sentence_loop"


def test_harmless_non_ascii_punctuation_is_not_catastrophic():
    solution = _valid_solution("chair", reasoning="The chair is at the user’s 12 o’clock.")
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["non_ascii_ratio"] > 0.0
    assert metrics["non_ascii_garbage"] == 0
    assert metrics["catastrophic_invalid"] == 0


def test_repeated_non_latin_garbage_is_catastrophic():
    solution = _valid_solution("chair", reasoning="ᠮᠮᠮᠮᠮᠮᠮᠮᠮᠮ")
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["hard_invalid_reason"] == "non_ascii_garbage"
    assert metrics["non_ascii_garbage"] == 1
    assert metrics["max_consecutive_non_ascii"] >= 8
    assert metrics["rare_unicode_garbage"] == 1


def test_natural_user_and_tool_words_are_not_contamination():
    solution = _valid_solution(
        "chair",
        reasoning="The user is looking for a chair. The tool observation indicates a chair.",
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == pytest.approx(2.0)
    assert metrics["final_turn_contamination"] == 0
    assert metrics["role_marker_contamination"] == 0
    assert metrics["tool_response_contamination"] == 0


def test_role_marker_contamination_is_catastrophic():
    solution = (
        "<think_grd>\nchair\n</think_grd>\n"
        "<highlight_obj>"
        "<reasoning>Assistant: <answer>chair</answer></reasoning>"
        "<answer>chair</answer>"
    )
    reward, metrics = _score(solution, ground_truth=["chair"])

    assert reward == -1.0
    assert metrics["catastrophic_invalid"] == 1
    assert metrics["hard_invalid_reason"] == "final_turn_contamination"
    assert metrics["final_turn_contamination"] == 1
    assert metrics["contamination_reason"] == "role_marker"
    assert metrics["role_marker_contamination"] == 1


def test_grounding_passthrough_does_not_change_reward_score():
    samples = [
        (_valid_solution("chair"), ["chair"]),
        (_valid_solution("table"), ["chair"]),
        ("<answer>chair</answer>", ["chair"]),
        (_observed_solution("<reasoning>The object is visible.</reasoning><answer>chair</answer>"), ["chair"]),
        (_valid_solution("chair", reasoning="The object is visible. " * 2), ["chair"]),
        (_valid_solution("chair", query="chair by the table", tool_tag="highlight_obj"), ["chair"]),
    ]
    tool_metrics = {
        "scenecot/has_gt_label": True,
        "scenecot/grd_stats_valid": True,
        "scenecot/ground_hit1": 0.0,
        "scenecot/ground_hit5": 1.0,
        "scenecot/ground_hit10": 1.0,
        "scenecot/grd_hit15": 1.0,
        "scenecot/grd_rank": 2,
        "scenecot/grd_prob": 0.7,
        "scenecot/grd_topk_ids": ["3", "2"],
        "scenecot/grd_topk_probs": [0.9, 0.7],
    }

    for solution, ground_truth in samples:
        base_extra = {**DEFAULT_EXTRA_INFO, "gt_objects": ["2"]}
        with_metrics = {**base_extra, "scenecot_tool_metrics": tool_metrics}

        reward_without, _ = _score(solution, extra_info=base_extra, ground_truth=ground_truth)
        reward_with, metrics_with = _score(solution, extra_info=with_metrics, ground_truth=ground_truth)

        assert reward_with == reward_without
        assert metrics_with["grd_rank"] == 2
        assert metrics_with["grd_prob"] == 0.7


def test_no_forbidden_agent_reward_surfaces_in_reward_file():
    reward_source = (
        Path(__file__).resolve().parents[1] / "scenecot_verl" / "rewards" / "scenecot_reward.py"
    ).read_text(encoding="utf-8")
    forbidden = [
        "tool" + "_choice_reward",
        "grounding" + "_reward",
        "evidence" + "_use_reward",
        "evidence" + "_support_reward",
        "answer" + "_reward_eff",
        "SCENECOT" + "_GROUNDING_W",
        "SCENECOT" + "_EVIDENCE_USE_W",
        "SCENECOT" + "_EVIDENCE_GATE_ANSWER",
        "SCENECOT" + "_ENABLE_GROUP_RELATIVE_QUERY_REWARD",
        "SCENECOT" + "_GROUP_RELATIVE_QUERY_ALPHA",
        "ground" + "_hit1",
        "ground" + "_hit5",
        "ground" + "_hit10",
        "top1" + "_prob",
        "query" + "_slot_coverage",
        "query" + "_reward",
        "group" + "_relative_query_reward",
        "w" + "_query",
        "SCENECOT" + "_QUERY_W",
        "tool" + "_efficiency",
        "w" + "_eff",
        "SCENECOT" + "_TOOL_EFFICIENCY",
    ]

    for term in forbidden:
        assert term not in reward_source
