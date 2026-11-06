"""GitHub lookups for repo questions, and web search for Groq models without browser_search."""
import asyncio
import io
import json
import os
import tempfile

from fastapi.testclient import TestClient

import backend.models.http_model as hm
from backend.app import create_app
from backend.config import Settings
from backend.github_tool import GitHubQuery, GitHubTool, detect
from backend.web_tool import find_allowlisted_url

GROQ = "https://api.groq.com/openai/v1/chat/completions"
Q = ("whats the most recent issue oppened in the public github repo torvalds/linux? include its tittle, "
     "auther,exact creation time, and link")


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_fetch(routes, seen=None):
    """Stands in for WebTool.fetch: returns the Rust tool's envelope."""
    async def fetch(url):
        if seen is not None:
            seen.append(url)
        path = url.replace("https://api.github.com", "").split("?")[0]
        r = routes.get(path)
        if r is None:
            return {"version": 1, "ok": False, "data": None, "error": {"code": "HTTP_ERROR", "message": "HTTP status 404"}}
        if isinstance(r, dict) and r.get("_error"):
            return {"version": 1, "ok": False, "data": None, "error": r["_error"]}
        text = json.dumps(r)
        return {"version": 1, "ok": True, "error": None, "data": {
            "requested_url": url, "final_url": url, "status": 200, "content_type": "application/json",
            "title": None, "text": text, "truncated": False}}
    return fetch


def run(coro):
    return asyncio.run(coro)


LINUX = {"full_name": "torvalds/linux", "html_url": "https://github.com/torvalds/linux", "has_issues": False,
         "open_issues_count": 3}
REPO = {"full_name": "a/b", "html_url": "https://github.com/a/b", "has_issues": True, "open_issues_count": 2}
SEARCH = {"total_count": 1, "items": [
    {"number": 8, "title": "Crash on start", "user": {"login": "alice"}, "created_at": "2026-11-05T09:08:07Z",
     "state": "open", "html_url": "https://github.com/a/b/issues/8"}]}


def test_detect():
    assert detect(Q) == GitHubQuery("torvalds", "linux", "issues")
    assert detect("latest release of https://github.com/psf/requests") == GitHubQuery("psf", "requests", "releases")
    assert detect("newest PR in github repo a/b.git") == GitHubQuery("a", "b", "pulls")
    assert detect("what is 3/4 of an issue") is None            # no github/repo word, no URL
    assert detect("tell me about github") is None
    assert detect("torvalds/linux on github") is None            # repo but nothing to look up


def test_issues_disabled_is_reported_not_guessed():
    out = run(GitHubTool(fake_fetch({"/repos/torvalds/linux": LINUX})).lookup(detect(Q)))
    assert out["ok"]
    body = json.loads(out["data"]["text"])
    assert "Issues are disabled" in body["note"] and body["newest_first"] == []
    assert out["data"]["final_url"] == "https://github.com/torvalds/linux"


def test_newest_issue_uses_issue_search_through_web_fetch():
    seen = []
    out = run(GitHubTool(fake_fetch({"/repos/a/b": REPO, "/search/issues": SEARCH}, seen)).lookup(
        GitHubQuery("a", "b", "issues")))
    first = json.loads(out["data"]["text"])["newest_first"][0]
    assert first == {"number": 8, "title": "Crash on start", "author": "alice", "created_at": "2026-11-05T09:08:07Z",
                     "state": "open", "url": "https://github.com/a/b/issues/8"}
    assert "q=repo:a%2Fb+is:issue" in seen[1] and "sort=created" in seen[1] and "order=desc" in seen[1]


def test_truncated_list_is_retried_with_one_item():
    seen = []
    calls = {"n": 0}

    async def fetch(url):
        seen.append(url)
        if "/search/issues" in url:
            calls["n"] += 1
            if calls["n"] == 1:
                return {"version": 1, "ok": True, "error": None, "data": {"content_type": "application/json",
                        "text": '{"items": [', "truncated": True, "final_url": url}}
        return await fake_fetch({"/repos/a/b": REPO, "/search/issues": SEARCH})(url)

    out = run(GitHubTool(fetch).lookup(GitHubQuery("a", "b", "issues")))
    assert out["ok"] and "per_page=1" in seen[-1]


def test_errors_are_truthful():
    assert run(GitHubTool(fake_fetch({})).lookup(GitHubQuery("no", "pe", "issues")))["error"]["code"] == "NOT_FOUND"
    off = {"/repos/a/b": {"_error": {"code": "PERMISSION_DENIED", "message": "disabled"}}}
    out = run(GitHubTool(fake_fetch(off)).lookup(GitHubQuery("a", "b", "issues")))
    assert "BOUGH_WEB_FETCH_ENABLED" in out["error"]["message"]
    blocked = {"/repos/a/b": {"_error": {"code": "URL_BLOCKED", "message": "Hostname not allowlisted"}}}
    out = run(GitHubTool(fake_fetch(blocked)).lookup(GitHubQuery("a", "b", "issues")))
    assert "api.github.com" in out["error"]["message"]
    limited = {"/repos/a/b": {"_error": {"code": "HTTP_ERROR", "message": "HTTP status 403"}}}
    assert run(GitHubTool(fake_fetch(limited)).lookup(GitHubQuery("a", "b", "issues")))["error"]["code"] == "RATE_LIMITED"


def test_find_allowlisted_url():
    hosts = ("example.com", "api.github.com")
    assert find_allowlisted_url("read https://example.com/a?b=1, please", hosts) == "https://example.com/a?b=1"
    assert find_allowlisted_url("see https://evil.test/x and https://example.com/", hosts) == "https://example.com/"
    assert find_allowlisted_url("see https://evil.test/x", hosts) is None
    assert find_allowlisted_url("no url here", hosts) is None


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
    fetched = []

    def fake(req, timeout):
        bodies.append(json.loads(req.data))
        return ok("torvalds/linux has GitHub issues disabled, so there is no most recent issue. "
                  "See https://github.com/torvalds/linux")

    monkeypatch.setattr(hm.urllib.request, "urlopen", fake)
    monkeypatch.setattr("backend.web_tool.WebTool.fetch",
                        lambda self, url: fake_fetch({"/repos/torvalds/linux": LINUX}, fetched)(url))
    final = chat(TestClient(create_app(settings())), Q)
    assert fetched == ["https://api.github.com/repos/torvalds/linux"]
    assert "Issues are disabled" in json.dumps(bodies[-1]) and "tools" not in bodies[-1]
    assert "issues disabled" in json.dumps(final)


def test_pasted_url_is_fetched_only_when_allowlisted(monkeypatch):
    fetched = []

    async def fetch(self, url):
        fetched.append(url)
        return {"version": 1, "ok": True, "error": None, "data": {"requested_url": url, "final_url": url,
                "status": 200, "content_type": "text/html", "title": "Example", "text": "Example Domain body",
                "truncated": False}}

    bodies = []
    monkeypatch.setattr("backend.web_tool.WebTool.fetch", fetch)
    monkeypatch.setattr(hm.urllib.request, "urlopen",
                        lambda req, timeout: (bodies.append(json.loads(req.data)), ok("It is the Example Domain page."))[1])
    on = dict(web_fetch_enabled=True, web_fetch_allowed_hosts=("example.com",))
    chat(TestClient(create_app(settings(**on))), "what is on https://example.com/ ?")
    assert fetched == ["https://example.com/"] and "Example Domain body" in json.dumps(bodies[-1])
    chat(TestClient(create_app(settings(**on))), "what is on https://other.test/ ?")
    chat(TestClient(create_app(settings(web_fetch_enabled=True, web_fetch_allowed_hosts=("example.com",),
                                        web_fetch_auto=False))), "what is on https://example.com/ ?")
    assert fetched == ["https://example.com/"]


def test_github_lookup_off(monkeypatch):
    called = []
    monkeypatch.setattr(hm.urllib.request, "urlopen", lambda req, timeout: ok("answer"))
    monkeypatch.setattr("backend.web_tool.WebTool.fetch", lambda self, url: called.append(url))
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
