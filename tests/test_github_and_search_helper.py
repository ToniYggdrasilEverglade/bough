"""GitHub lookups for repo questions, and web search for Groq models without browser_search."""
import io
import json
import os
import tempfile
import urllib.error

from fastapi.testclient import TestClient

import backend.models.http_model as hm
from backend.app import create_app
from backend.config import Settings
from backend.github_tool import GitHubQuery, GitHubTool, detect

GROQ = "https://api.groq.com/openai/v1/chat/completions"
Q = ("whats the most recent issue oppened in the public github repo torvalds/linux? include its tittle, "
     "auther,exact creation time, and link")


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def gh_opener(routes, seen=None):
    def op(req, timeout):
        path = req.full_url.replace("https://api.github.com", "").split("?")[0]
        if seen is not None:
            seen.append((req.full_url, req.get_method(), req.headers.get("Authorization")))
        if path not in routes:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b"{}"))
        r = routes[path]
        if isinstance(r, Exception):
            raise r
        return Resp(json.dumps(r).encode())
    return op


LINUX = {"full_name": "torvalds/linux", "html_url": "https://github.com/torvalds/linux", "has_issues": False,
         "open_issues_count": 3}
REPO = {"full_name": "a/b", "html_url": "https://github.com/a/b", "has_issues": True, "open_issues_count": 2}
ISSUES = [
    {"number": 9, "title": "PR, not an issue", "user": {"login": "x"}, "created_at": "2026-11-06T10:00:00Z",
     "state": "open", "html_url": "https://github.com/a/b/pull/9", "pull_request": {}},
    {"number": 8, "title": "Crash on start", "user": {"login": "alice"}, "created_at": "2026-11-05T09:08:07Z",
     "state": "open", "html_url": "https://github.com/a/b/issues/8"},
]


def test_detect():
    assert detect(Q) == GitHubQuery("torvalds", "linux", "issues")
    assert detect("latest release of https://github.com/psf/requests") == GitHubQuery("psf", "requests", "releases")
    assert detect("newest PR in github repo a/b.git") == GitHubQuery("a", "b", "pulls")
    assert detect("what is 3/4 of an issue") is None            # no github/repo word, no URL
    assert detect("tell me about github") is None
    assert detect("torvalds/linux on github") is None            # repo but nothing to look up


def test_issues_disabled_is_reported_not_guessed():
    out = GitHubTool(opener=gh_opener({"/repos/torvalds/linux": LINUX})).lookup(detect(Q))
    assert out["ok"]
    body = json.loads(out["data"]["text"])
    assert "Issues are disabled" in body["note"] and body["newest_first"] == []
    assert out["data"]["final_url"] == "https://github.com/torvalds/linux"


def test_newest_issue_skips_pull_requests_and_uses_token():
    seen = []
    tool = GitHubTool(token="ghp_test", opener=gh_opener({"/repos/a/b": REPO, "/repos/a/b/issues": ISSUES}, seen))
    out = tool.lookup(GitHubQuery("a", "b", "issues"))
    first = json.loads(out["data"]["text"])["newest_first"][0]
    assert first == {"number": 8, "title": "Crash on start", "author": "alice", "created_at": "2026-11-05T09:08:07Z",
                     "state": "open", "url": "https://github.com/a/b/issues/8"}
    assert all(m == "GET" and a == "Bearer ghp_test" for _, m, a in seen)
    assert "sort=created" in seen[1][0] and "direction=desc" in seen[1][0]


def test_errors_are_truthful():
    assert GitHubTool(opener=gh_opener({})).lookup(GitHubQuery("no", "pe", "issues"))["error"]["code"] == "NOT_FOUND"
    limited = urllib.error.HTTPError("u", 403, "Forbidden", {}, io.BytesIO(b"{}"))
    out = GitHubTool(opener=gh_opener({"/repos/a/b": limited})).lookup(GitHubQuery("a", "b", "issues"))
    assert out["error"]["code"] == "RATE_LIMITED" and "BOUGH_GITHUB_TOKEN" in out["error"]["message"]


def settings(model="qwen/qwen3.8-27b", **kw):
    return Settings(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"), model_backend="http", model_url=GROQ,
                    model_name=model, model_api_key="gsk_test", **kw)


def chat(c, text):
    return c.post("/api/sessions/s1/chat", json={"text": text}).json()


def ok(content, executed=None):
    msg = {"role": "assistant", "content": content}
    if executed is not None:
        msg["executed_tools"] = executed
    return Resp(json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]}).encode())


def test_repo_question_reaches_model_with_github_data(monkeypatch):
    bodies = []

    gh = gh_opener({"/repos/torvalds/linux": LINUX})

    def fake(req, timeout):
        if req.full_url.startswith("https://api.github.com"):
            return gh(req, timeout)
        bodies.append(json.loads(req.data))
        return ok("torvalds/linux has GitHub issues disabled, so there is no most recent issue. "
                  "See https://github.com/torvalds/linux")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    final = chat(TestClient(create_app(settings())), Q)
    sent = json.dumps(bodies[-1])
    assert "Issues are disabled" in sent and "tools" not in bodies[-1]
    assert "issues disabled" in json.dumps(final)


def test_github_lookup_off(monkeypatch):
    called = []
    def fake(req, timeout):
        if "api.github.com" in req.full_url:
            called.append(1)
        return ok("answer")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    chat(TestClient(create_app(settings(github_lookup="off"))), Q)
    assert called == []


def test_qwen_web_request_uses_gpt_oss_helper(monkeypatch):
    bodies = []
    executed = [{"type": "browser.search", "arguments": json.dumps({"query": "who invented the lightbulb"})}]

    def fake(req, timeout):
        b = json.loads(req.data)
        bodies.append(b)
        if b["model"] == "openai/gpt-oss-20b":
            return ok("- Joseph Swan demonstrated a carbon lamp in 1878 (https://example.org/swan)", executed)
        return ok("Joseph Swan (1878), per https://example.org/swan; Edison's design followed in 1879.")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    final = chat(TestClient(create_app(settings())), "use the web to find out who invented the lightbulb first")
    helper, main = bodies[0], bodies[-1]
    assert helper["model"] == "openai/gpt-oss-20b" and helper["tools"] == [{"type": "browser_search"}]
    assert main["model"] == "qwen/qwen3.8-27b" and "tools" not in main
    assert "Joseph Swan demonstrated" in json.dumps(main["messages"])
    assert "Joseph Swan" in json.dumps(final)


def test_no_helper_without_web_request(monkeypatch):
    bodies = []
    monkeypatch.setattr(hm.urllib.request, "urlopen", lambda req, timeout: (bodies.append(json.loads(req.data)), ok("hi"))[1])
    chat(TestClient(create_app(settings())), "who invented the lightbulb first")
    assert {b["model"] for b in bodies} == {"qwen/qwen3.8-27b"}


def test_helper_disabled(monkeypatch):
    bodies = []
    monkeypatch.setattr(hm.urllib.request, "urlopen", lambda req, timeout: (bodies.append(json.loads(req.data)), ok("hi"))[1])
    chat(TestClient(create_app(settings(model_web_search_helper=""))), "use the web to find the weather")
    assert {b["model"] for b in bodies} == {"qwen/qwen3.8-27b"}
