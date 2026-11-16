import io
import importlib
import json
from urllib import error

import pytest

from scenecot_verl.rewards import answer_judge_client as client
reward = importlib.import_module("scenecot_verl.rewards.scenecot_reward")


@pytest.fixture(autouse=True)
def clean_judge_env(monkeypatch):
    import os
    for name in os.environ:
        if name.startswith("SCENECOT_ANSWER_JUDGE_") or name in {"OPENAI_API_KEY", "GEMINI_API_KEY"}:
            monkeypatch.delenv(name, raising=False)
    reward._http_answer_judge_cached.cache_clear()
    reward._gemini_answer_judge_cached.cache_clear()


def apply_judge(metrics=None):
    return reward.maybe_apply_answer_judge(
        0.2, metrics if metrics is not None else {"answer_rule_direct": 0, "answer_token_f1": 0.5},
        question="What color is the chair?", pred_answer="blue", gt_answers=["blue"],
    )


@pytest.mark.parametrize("backend", ["gemini", "openai", "local_qwen"])
@pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
def test_every_backend_uses_identical_prompt_and_score_mapping(monkeypatch, backend, score):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", backend)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    prompts = []
    payload = {"score_1_to_5": score, "reason": "test"}

    def fake_http(*args, **kwargs):
        prompts.append(args[2])
        return payload

    def fake_gemini(req, **kwargs):
        prompts.append(json.loads(req.data)["contents"][0]["parts"][0]["text"])
        return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [
            {"text": json.dumps(payload)}]}}]}).encode())

    monkeypatch.setattr(reward, "request_judge", fake_http)
    monkeypatch.setattr(reward.request, "urlopen", fake_gemini)
    value, metrics = apply_judge()
    assert value == (score - 1) / 4
    assert metrics["answer_judge_status"] == backend
    assert metrics["answer_judge_score_1_to_5"] == score
    assert metrics["answer_judge_called"] == 1
    assert prompts == [reward._build_gemini_answer_judge_prompt(
        "msqa_1to5", "What color is the chair?", "blue", ["blue"], reward.DEFAULT_ANSWER_JUDGE_RUBRIC_PATH,
    )]


def test_openai_request_uses_luna_responses_schema_and_ignores_reasoning_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    def post(url, payload, headers, timeout, retries):
        assert url == "https://api.openai.com/v1/responses"
        assert payload["model"] == "gpt-5.6-luna"
        assert payload["reasoning"] == {"effort": "none"}
        assert payload["store"] is False
        assert payload["text"]["format"]["schema"]["additionalProperties"] is False
        assert headers == {"Authorization": "Bearer dummy"}
        return {"status": "completed", "output": [
            {"type": "reasoning", "content": [{"type": "output_text", "text": "incorrect JSON"}]},
            {"type": "message", "content": [{"type": "output_text", "text": '{"score_1_to_5":5,"reason":"ok"}'}]},
        ]}
    monkeypatch.setattr(client, "post_json", post)
    assert client.request_judge("openai", "gpt-5.6-luna", "prompt", "msqa_1to5",
                                base_url="https://api.openai.com/v1")["score_1_to_5"] == 5


@pytest.mark.parametrize("model", ["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3.5-2B"])
def test_local_qwen_uses_loopback_chat_no_cloud_key_and_no_thinking(monkeypatch, model):
    def post(url, payload, headers, timeout, retries):
        assert url == "http://127.0.0.1:1234/v1/chat/completions"
        assert not headers
        assert payload["model"] == model
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["temperature"] == 0
        assert payload["response_format"]["type"] == "json_schema"
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": '{"score_1_to_5":4,"reason":"ok"}'}}]}
    monkeypatch.setattr(client, "post_json", post)
    assert client.request_judge("local_qwen", model, "prompt", "msqa_1to5",
                                base_url="http://127.0.0.1:1234/v1")["score_1_to_5"] == 4
    with pytest.raises(ValueError, match="loopback"):
        client.request_judge("local_qwen", model, "prompt", "msqa_1to5",
                             base_url="https://example.com/v1")


@pytest.mark.parametrize("bad", [0, 6, 2.5, "5", True, float("nan")])
def test_invalid_grades_are_rejected_not_clipped(bad):
    with pytest.raises(ValueError):
        client.validate_judge_payload({"score_1_to_5": bad, "reason": "test"}, "msqa_1to5")


@pytest.mark.parametrize("body", [
    {"status": "incomplete", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
])
def test_incomplete_or_refused_openai_response_is_rejected(monkeypatch, body):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(client, "post_json", lambda *a, **kw: body)
    with pytest.raises(ValueError):
        client.request_judge("openai", "gpt-5.6-luna", "prompt", "msqa_1to5", base_url="https://api.openai.com/v1")


def test_openai_length_retry_preserves_prompt_and_rejects_partial_grade(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    calls = []
    def post(url, payload, *args):
        calls.append(json.loads(json.dumps(payload)))
        if len(calls) == 1:
            return {"id": "resp_test", "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 1100, "output_tokens": 512},
                    "output": [{"type": "message", "content": [{"type": "output_text",
                                "text": '{"score_1_to_5":5,"reason":"private"}'}]}]}
        return {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": '{"score_1_to_5":3,"reason":"ok"}'}]}]}
    monkeypatch.setattr(client, "post_json", post)
    result = client.request_judge("openai", "gpt-5.6-luna", "private prompt", "msqa_1to5",
                                  base_url="https://api.openai.com/v1", max_tokens=512)
    assert result["score_1_to_5"] == 3
    assert [p["max_output_tokens"] for p in calls] == [512, 1024]
    assert calls[0]["input"] == calls[1]["input"]
    assert calls[0]["text"] == calls[1]["text"]
    assert '"incomplete_reason": "max_output_tokens"' in caplog.text
    assert '"output_tokens": 512' in caplog.text
    assert "private" not in caplog.text


@pytest.mark.parametrize("retries,start,expected", [
    (0, 512, [512]), (2, 512, [512, 1024, 2048]), (5, 512, [512, 1024, 2048]),
    (2, 4096, [4096]),
])
def test_openai_truncation_retries_have_attempt_and_token_limits(monkeypatch, retries, start, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    limits = []
    def post(url, payload, *args):
        limits.append(payload["max_output_tokens"])
        return {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    monkeypatch.setattr(client, "post_json", post)
    with pytest.raises(ValueError, match="max_output_tokens"):
        client.request_judge("openai", "gpt-5.6-luna", "prompt", "msqa_1to5",
                             base_url="https://api.openai.com/v1", max_tokens=start, retries=retries)
    assert limits == expected


@pytest.mark.parametrize("body,expected", [
    ({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}, "content_filter"),
    ({"status": "failed", "error": {"code": "server_error", "message": "private"}}, "server_error"),
    ({"status": "incomplete"}, '"incomplete_reason": null'),
    ({"status": "completed", "output": []}, "empty content"),
    ({"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "private invalid JSON"}]}]}, "invalid JSON"),
])
def test_openai_other_failures_report_metadata_without_retry_or_output(monkeypatch, body, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    calls = []
    def post(*args):
        calls.append(1)
        return body
    monkeypatch.setattr(client, "post_json", post)
    with pytest.raises(ValueError, match=expected) as exc:
        client.request_judge("openai", "gpt-5.6-luna", "private prompt", "msqa_1to5",
                             base_url="https://api.openai.com/v1")
    assert len(calls) == 1
    assert "private" not in str(exc.value)


def test_truncated_local_response_is_rejected_even_if_json_looks_valid(monkeypatch):
    monkeypatch.setattr(client, "post_json", lambda *a, **kw: {"choices": [{
        "finish_reason": "length", "message": {"content": '{"score_1_to_5":5,"reason":"ok"}'}}]})
    with pytest.raises(ValueError, match="truncated"):
        client.request_judge("local_qwen", "Qwen/Qwen3.5-2B", "prompt", "msqa_1to5", base_url="http://localhost/v1")


def test_local_length_retry_only_accepts_completed_result(monkeypatch, caplog):
    calls = []
    def post(url, payload, *args):
        calls.append(json.loads(json.dumps(payload)))
        if len(calls) == 1:
            return {"usage": {"prompt_tokens": 1100, "completion_tokens": 512}, "choices": [{
                "finish_reason": "length", "message": {"content": '{"score_1_to_5":5,"reason":"private text"}'}}]}
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": '{"score_1_to_5":3,"reason":"complete"}'}}]}
    monkeypatch.setattr(client, "post_json", post)
    result = client.request_judge("local_qwen", "Qwen/Qwen3-VL-4B-Instruct", "private prompt",
                                  "msqa_1to5", base_url="http://localhost/v1", max_tokens=512)
    assert result["score_1_to_5"] == 3
    assert [p["max_tokens"] for p in calls] == [512, 1024]
    assert calls[0]["messages"] == calls[1]["messages"]
    assert calls[0]["response_format"] == calls[1]["response_format"]
    assert "finish_reason=length" in caplog.text
    assert "private" not in caplog.text


@pytest.mark.parametrize("retries,expected", [(0, [512]), (2, [512, 1024])])
def test_local_truncation_retries_are_bounded(monkeypatch, retries, expected):
    limits = []
    def post(url, payload, *args):
        limits.append(payload["max_tokens"])
        return {"choices": [{"finish_reason": "length", "message": {"content": "private"}}]}
    monkeypatch.setattr(client, "post_json", post)
    with pytest.raises(ValueError, match="finish_reason=length") as exc:
        client.request_judge("local_qwen", "Qwen/Qwen3-VL-4B-Instruct", "prompt", "msqa_1to5",
                             base_url="http://localhost/v1", max_tokens=512, retries=retries)
    assert limits == expected
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("choices,diagnostic", [
    ([], "no_choices"),
    ([{"finish_reason": "content_filter", "message": {"content": "private"}}], "content_filter"),
    ([{"finish_reason": "stop", "message": {"content": ""}}], "empty content"),
    ([{"finish_reason": "stop", "message": {"content": "private invalid JSON"}}], "invalid JSON"),
])
def test_local_other_failures_report_metadata_without_retry_or_text(monkeypatch, choices, diagnostic):
    calls = []
    def post(*args):
        calls.append(1)
        return {"choices": choices}
    monkeypatch.setattr(client, "post_json", post)
    with pytest.raises(ValueError, match=diagnostic) as exc:
        client.request_judge("local_qwen", "Qwen/Qwen3-VL-4B-Instruct", "prompt", "msqa_1to5",
                             base_url="http://localhost/v1")
    assert len(calls) == 1
    assert "private" not in str(exc.value)


def test_direct_rules_skip_judge_and_failures_are_explicit(monkeypatch):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", "local_qwen")
    def fail(*args, **kwargs):
        raise TimeoutError("offline")
    monkeypatch.setattr(reward, "request_judge", fail)
    _, metrics = apply_judge({"answer_rule_direct": 1, "answer_rule_decision": "em"})
    assert metrics["answer_judge_called"] == 0
    assert metrics["answer_judge_status"] == "em"
    value, metrics = apply_judge()
    assert value == 0.2
    assert metrics["answer_judge_fallback"] == 1
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "raise")
    with pytest.raises(RuntimeError, match="local_qwen answer judge failed"):
        apply_judge()


def test_local_truncation_falls_back_at_1024_and_next_sample_can_recover(monkeypatch):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", "local_qwen")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_MAX_TOKENS", "512")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "raise")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_TRUNCATION", "fallback")
    limits = []

    def post(url, payload, *args):
        limits.append(payload["max_tokens"])
        # A parseable partial score must not be used, or cached as a judgement.
        finish = "length" if len(limits) <= 2 else "stop"
        return {"choices": [{"finish_reason": finish, "message": {
            "content": '{"score_1_to_5":5,"reason":"test"}'}}]}

    monkeypatch.setattr(client, "post_json", post)
    value, metrics = apply_judge()
    assert limits == [512, 1024]
    assert value == 0.2
    assert metrics["answer_judge_fallback"] == 1
    assert metrics["answer_judge_status"] == "fallback_rule_on_truncation"
    assert metrics["answer_judge_score_1_to_5"] is None
    assert "finish_reason=length" in metrics["answer_judge_error"]
    value, metrics = apply_judge()
    assert limits == [512, 1024, 512]
    assert value == 1.0
    assert metrics["answer_judge_fallback"] == 0
    assert metrics["answer_judge_status"] == "local_qwen"


def test_openai_truncation_falls_back_and_next_request_recovers(monkeypatch):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_MAX_TOKENS", "512")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "raise")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_TRUNCATION", "fallback")
    limits = []
    def post(url, payload, *args):
        limits.append(payload["max_output_tokens"])
        result = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": '{"score_1_to_5":5,"reason":"test"}'}]}]}
        if len(limits) <= 3:
            result.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return result
    monkeypatch.setattr(client, "post_json", post)
    value, metrics = apply_judge()
    assert limits == [512, 1024, 2048]
    assert value == 0.2
    assert metrics["answer_judge_fallback"] == 1
    assert metrics["answer_judge_status"] == "fallback_rule_on_truncation"
    assert metrics["answer_judge_score_1_to_5"] is None
    assert "max_output_tokens" in metrics["answer_judge_error"]
    value, metrics = apply_judge()
    assert limits == [512, 1024, 2048, 512]
    assert value == 1.0
    assert metrics["answer_judge_fallback"] == 0
    assert metrics["answer_judge_status"] == "openai"


@pytest.mark.parametrize("body", [
    {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
    {"status": "incomplete"},
    {"status": "failed", "error": {"code": "server_error"}},
])
def test_openai_non_length_incomplete_does_not_trigger_truncation_fallback(monkeypatch, body):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "raise")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_TRUNCATION", "fallback")
    monkeypatch.setattr(client, "post_json", lambda *args: body)
    with pytest.raises(RuntimeError, match="not completed"):
        apply_judge()


@pytest.mark.parametrize("backend", ["local_qwen", "openai"])
@pytest.mark.parametrize("failure", [TimeoutError("offline"), ValueError("invalid JSON")])
def test_truncation_fallback_does_not_swallow_other_failures(monkeypatch, backend, failure):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", backend)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "raise")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_TRUNCATION", "fallback")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(reward, "request_judge", fail)
    with pytest.raises(RuntimeError, match=f"{backend} answer judge failed"):
        apply_judge()


@pytest.mark.parametrize("backend", ["local_qwen", "openai"])
def test_truncation_can_still_be_configured_to_raise(monkeypatch, backend):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", backend)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "raise")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_TRUNCATION", "raise")
    def fail(*args, **kwargs):
        raise client.JudgeOutputTruncatedError("finish_reason=length")
    monkeypatch.setattr(reward, "request_judge", fail)
    with pytest.raises(RuntimeError, match="finish_reason=length"):
        apply_judge()


def test_missing_openai_key_cannot_silently_disable_strict_judge(monkeypatch):
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", "openai")
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_ON_ERROR", "raise")
    with pytest.raises(RuntimeError, match="missing_api_key"):
        apply_judge()


def test_cache_is_separate_for_backend_model_endpoint_and_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    calls = []
    def judge(*args, **kwargs):
        calls.append((args, kwargs))
        return {"score_1_to_5": 4, "reason": "test"}
    monkeypatch.setattr(reward, "request_judge", judge)
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", "openai")
    apply_judge(); apply_judge()
    assert len(calls) == 1
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_MODEL", "different-model")
    apply_judge()
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_REASONING_EFFORT", "low")
    apply_judge()
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BACKEND", "local_qwen")
    apply_judge()
    monkeypatch.setenv("SCENECOT_ANSWER_JUDGE_BASE_URL", "http://localhost:9999/v1")
    apply_judge()
    assert len(calls) == 5


def test_transient_http_failure_is_retried_without_leaking_response(monkeypatch):
    calls = []
    def opener(req, **kwargs):
        calls.append(req)
        if len(calls) == 1:
            raise error.HTTPError(req.full_url, 429, "rate limit", {}, io.BytesIO(b"private response"))
        return io.BytesIO(b'{"status":"completed"}')
    monkeypatch.setattr(client.request, "urlopen", opener)
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    assert client.post_json("https://api.openai.com/v1/responses", {}, {}, 1, 2) == {"status": "completed"}
    assert len(calls) == 2
