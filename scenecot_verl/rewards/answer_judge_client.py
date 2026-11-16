"""HTTP transports for training answer judges; no model/SDK imports in Ray workers."""
from __future__ import annotations

import json
import logging
import os
import time
from urllib import error, parse, request

logger = logging.getLogger(__name__)


class JudgeOutputTruncatedError(ValueError):
    """The judge exhausted its output budget without a completed response."""


def openai_response_diagnostic(response: dict, max_tokens: int) -> str:
    """Log completion metadata only, never prompts, output text or error messages."""
    details = response.get("incomplete_details") or {}
    api_error = response.get("error") or {}
    usage = response.get("usage") or {}
    diagnostic = {
        "response_id": response.get("id"),
        "status": response.get("status"),
        "incomplete_reason": details.get("reason"),
        "error_code": api_error.get("code"),
        "max_output_tokens": max_tokens,
        "usage": {k: v for k in ("input_tokens", "output_tokens", "total_tokens")
                  if type(v := usage.get(k)) is int},
    }
    return json.dumps(diagnostic, ensure_ascii=False)


def local_response_diagnostic(response: dict, max_tokens: int) -> str:
    """Safe completion metadata, without prompts, generated text or credentials."""
    choices = response.get("choices") or []
    finish = choices[0].get("finish_reason") if choices else "no_choices"
    allowed = {"stop", "length", "content_filter", "tool_calls", "function_call", "no_choices", None}
    finish = finish if isinstance(finish, (str, type(None))) and finish in allowed else "unknown"
    usage = response.get("usage") or {}
    counts = {k: v for k in ("prompt_tokens", "completion_tokens")
              if type(v := usage.get(k)) is int}
    return f"finish_reason={finish}, max_tokens={max_tokens}, usage={counts}"


def judge_schema(style: str) -> dict:
    if style == "msqa_1to5":
        properties = {
            "score_1_to_5": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "reason": {"type": "string"},
        }
    elif style == "equivalence":
        properties = {
            "score": {"type": "number", "enum": [0.0, 0.5, 1.0]},
            "label": {"type": "string", "enum": ["incorrect", "partially_correct", "equivalent"]},
            "reason": {"type": "string"},
        }
    else:
        raise ValueError(f"Unsupported judge prompt style: {style}")
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def validate_judge_payload(payload: dict, style: str) -> dict:
    schema = judge_schema(style)
    if not isinstance(payload, dict) or set(payload) != set(schema["required"]):
        raise ValueError("Judge response does not match the requested JSON fields")
    for name, spec in schema["properties"].items():
        value = payload[name]
        if spec["type"] == "string" and not isinstance(value, str):
            raise ValueError(f"Judge {name} must be a string")
        if spec["type"] in {"integer", "number"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Judge {name} must be numeric")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"Judge {name} is outside the allowed values")
    return payload


def post_json(url: str, payload: dict, headers: dict, timeout: float, retries: int) -> dict:
    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Content-Type": "application/json", **headers})
    local = parse.urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}
    # Keep local questions on this node even if the cluster exports HTTP_PROXY.
    opener = request.build_opener(request.ProxyHandler({})).open if local else request.urlopen
    for attempt in range(retries + 1):
        try:
            with opener(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("Answer judge HTTP response must be a JSON object")
                return result
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            if status not in {408, 429, 500, 502, 503, 504} or attempt == retries:
                # Do not include headers, URLs containing credentials, or response bodies.
                raise RuntimeError(f"Answer judge HTTP {status}") from None
        except (TimeoutError, error.URLError, OSError):
            if attempt == retries:
                raise RuntimeError("Answer judge connection failed or timed out") from None
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Answer judge retry loop exhausted")


def request_judge(
    backend: str, model: str, prompt: str, style: str, *,
    base_url: str, max_tokens: int = 128, reasoning_effort: str = "none",
    timeout: float = 30.0, retries: int = 2,
) -> dict:
    schema = judge_schema(style)
    if backend == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload = {
            "model": model, "input": prompt, "store": False,
            "max_output_tokens": max_tokens, "reasoning": {"effort": reasoning_effort},
            "text": {"format": {"type": "json_schema", "name": "answer_judgement",
                                "strict": True, "schema": schema}},
        }
        for attempt in range(max(0, retries) + 1):
            response = post_json(base_url.rstrip("/") + "/responses", payload,
                                 {"Authorization": f"Bearer {key}"}, timeout, retries)
            diagnostic = openai_response_diagnostic(response, payload["max_output_tokens"])
            if response.get("status") == "completed":
                break
            reason = (response.get("incomplete_details") or {}).get("reason")
            if response.get("status") == "incomplete" and reason == "max_output_tokens":
                next_limit = min(payload["max_output_tokens"] * 2, 2048)
                if attempt < retries and next_limit > payload["max_output_tokens"]:
                    logger.warning("OpenAI judge output truncated (%s); retrying with max_output_tokens=%d",
                                   diagnostic, next_limit)
                    payload["max_output_tokens"] = next_limit
                    continue
                raise JudgeOutputTruncatedError(
                    f"OpenAI judge response was truncated ({diagnostic})"
                )
            raise ValueError(f"OpenAI judge response was not completed ({diagnostic})")
        parts = [part for item in response.get("output", []) if item.get("type") == "message"
                 for part in item.get("content", [])]
        if any(part.get("type") == "refusal" for part in parts):
            raise ValueError(f"OpenAI judge refused the request ({diagnostic})")
        text = "".join(part.get("text", "") for part in parts if part.get("type") == "output_text")
        if not text.strip():
            raise ValueError(f"OpenAI judge returned empty content ({diagnostic})")
        try:
            return validate_judge_payload(json.loads(text), style)
        except (ValueError, TypeError):
            raise ValueError(f"OpenAI judge returned invalid JSON or score fields ({diagnostic})") from None
    elif backend == "local_qwen":
        if parse.urlparse(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("local_qwen requires a loopback URL on the training node")
        payload = {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "answer_judgement", "strict": True, "schema": schema}},
        }
        for attempt in range(max(0, retries) + 1):
            response = post_json(base_url.rstrip("/") + "/chat/completions", payload, {}, timeout, retries)
            choices = response.get("choices") or []
            diagnostic = local_response_diagnostic(response, payload["max_tokens"])
            finish = choices[0].get("finish_reason") if choices else None
            if finish == "length":
                # Reissue the unchanged rubric/question with more output space.
                # Never accept a partial grade, even if its JSON happens to parse.
                next_limit = min(payload["max_tokens"] * 2, 1024)
                if attempt < retries and next_limit > payload["max_tokens"]:
                    logger.warning("Local judge output truncated (%s); retrying with max_tokens=%d",
                                   diagnostic, next_limit)
                    payload["max_tokens"] = next_limit
                    continue
                raise JudgeOutputTruncatedError(
                    f"Local judge response was truncated ({diagnostic})"
                )
            if finish != "stop":
                raise ValueError(f"Local judge response was empty, truncated, or unfinished ({diagnostic})")
            text = choices[0].get("message", {}).get("content") or ""
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Local judge returned empty content ({diagnostic})")
            try:
                return validate_judge_payload(json.loads(text), style)
            except (ValueError, TypeError):
                raise ValueError(f"Local judge returned invalid JSON or score fields ({diagnostic})") from None
    else:
        raise ValueError(f"Unsupported judge transport: {backend}")
    return validate_judge_payload(json.loads(text), style)
