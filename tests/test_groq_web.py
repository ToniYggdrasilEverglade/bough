"""Groq tool-use handling: gpt-oss calling an undeclared tool, and explicit web requests using browser_search."""
import io
import json
import os
import tempfile
import urllib.error

from fastapi.testclient import TestClient

import backend.models.http_model as hm
from backend.app import create_app
from backend.config import Settings
from backend.models.http_model import wants_web

GROQ = "https://api.groq.com/openai/v1/chat/completions"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok(content, executed=None):
    msg = {"role": "assistant", "content": content}
    if executed is not None:
        msg["executed_tools"] = executed
    return FakeResponse(json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]}).encode())


def tool_use_failed(name="browser.search"):
    body = {"error": {"message": "Tool choice is none, but model called a tool", "type": "invalid_request_error",
                      "code": "tool_use_failed", "failed_generation": json.dumps({"name": name, "arguments": {}})}}
    return urllib.error.HTTPError(GROQ, 400, "Bad Request", {}, io.BytesIO(json.dumps(body).encode()))


def client(model="openai/gpt-oss-120b", **kw):
    s = Settings(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"), model_backend="http", model_url=GROQ,
                 model_name=model, model_api_key="gsk_test", **kw)
    return TestClient(create_app(s))


def chat(c, text):
    return c.post("/api/sessions/s1/chat", json={"text": text}).json()


def test_wants_web_only_on_explicit_requests():
    for t in ("use the web to find out who invented the lightbulb first", "search the web for groq pricing",
              "look it up online", "google it", "can you check online for the latest release",
              "do a web search on rust 2026"):
        assert wants_web({"request": t}), t
    for t in ("who invented the lightbulb", "build a web app in fastapi", "what is a web socket",
              "explain the internet protocol suite"):
        assert not wants_web({"request": t}), t


def test_web_request_uses_groq_browser_search(monkeypatch):
    bodies = []

    def fake(req, timeout):
        body = json.loads(req.data)
        bodies.append(body)
        assert timeout >= 60
        executed = [{"index": 0, "type": "browser_search",
                     "arguments": json.dumps({"query": "who invented the lightbulb first"}), "output": "..."}]
        return ok("I searched the web: Humphry Davy demonstrated an arc lamp around 1806; "
                  "Joseph Swan and Thomas Edison made practical incandescent bulbs around 1879.", executed)

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    r = chat(client(), "use the web to find out who invented the lightbulb first")
    assert bodies[0]["tools"] == [{"type": "browser_search"}] and bodies[0]["tool_choice"] == "required"
    assert "response_format" not in bodies[0]
    assert "Edison" in r["final"]["text"] and r["status"] == "completed"
    assert len(bodies) == 1, "grounding must accept the claim: the search really ran, no repair call"


def test_plain_request_has_no_tools(monkeypatch):
    bodies = []

    def fake(req, timeout):
        bodies.append(json.loads(req.data))
        return ok("Thomas Edison is usually credited with the first practical incandescent bulb in 1879.")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    chat(client(), "who invented the lightbulb")
    assert "tools" not in bodies[0]


def test_undeclared_tool_call_is_retried_without_tools(monkeypatch):
    """Non-search model (or web search off): gpt-oss calls a tool anyway -> one retry with a no-tools note."""
    bodies = []

    def fake(req, timeout):
        body = json.loads(req.data)
        bodies.append(body)
        if len(bodies) == 1:
            raise tool_use_failed()
        return ok("Thomas Edison and Joseph Swan made the first practical incandescent bulbs around 1879.")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    r = chat(client(model_web_search="off"), "use the web to find out who invented the lightbulb first")
    assert "Edison" in r["final"]["text"]
    assert len(bodies) == 2 and "tools" not in bodies[1]
    assert any("No tools are available" in m["content"] for m in bodies[1]["messages"] if m["role"] == "system")


def test_tool_call_twice_reports_real_cause(monkeypatch):
    def fake(req, timeout):
        raise tool_use_failed("browser.open")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    r = chat(client(model="qwen/qwen3-32b"), "use the web to find out who invented the lightbulb first")
    assert r["final"]["action_kind"] == "notice"
    assert "could not be reached" not in r["final"]["text"]   # it was a tool-call problem, not an outage
    assert "kept calling a tool Bough did not offer (browser.open)" in r["final"]["text"]


def _err429(msg, headers=None):
    body = {"error": {"message": msg, "type": "tokens", "code": "rate_limit_exceeded"}}
    return urllib.error.HTTPError(GROQ, 429, "Too Many Requests", headers or {}, io.BytesIO(json.dumps(body).encode()))


TOO_LARGE = ("Request too large for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on "
             "output tokens per minute (OTPM): Limit 1000, Requested 1033. The request's expected output tokens "
             "exceed the enforced limit; reduce max_tokens (or the request's expected output) and try again.")


def test_otpm_too_large_shrinks_cap_and_remembers(monkeypatch):
    caps = []

    def fake(req, timeout):
        cap = json.loads(req.data)["max_completion_tokens"]
        caps.append(cap)
        if cap > 1000:
            raise _err429(TOO_LARGE)
        return ok("Hello!")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    c = client(model="qwen/qwen3.8-27b")
    assert chat(c, "hello")["final"]["text"] == "Hello!"
    assert caps == [6000, 750]
    chat(c, "hello again")
    assert caps[2] == 750, "the learned cap is reused, no wasted request"


def test_short_rate_limit_waits_then_retries(monkeypatch):
    calls = []

    def fake(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise _err429("Rate limit reached for model `qwen/qwen3.8-27b` ... (OTPM): Limit 1000, Used 900, "
                          "Requested 300. Please try again in 0.2s.")
        return ok("Hello!")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    assert chat(client(model="qwen/qwen3.8-27b"), "hello")["final"]["text"] == "Hello!"
    assert len(calls) == 2


def test_long_rate_limit_says_when(monkeypatch):
    def fake(req, timeout):
        raise _err429("Rate limit reached ... Please try again in 42.5s.")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    text = chat(client(model="qwen/qwen3.8-27b"), "hello")["final"]["text"]
    assert "per-minute limit" in text and "43s" in text
