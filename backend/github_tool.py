"""Read-only GitHub lookups for questions about a public repository (issues, pull requests, releases, commits).

The root model cannot browse, so "what's the newest issue in torvalds/linux?" used to get a "go check it
yourself" reply. When a message names a repository (github.com/owner/repo, or owner/repo next to the word
GitHub/repo) and asks about one of those things, Bough asks the GitHub REST API directly and hands the answer
to the model as data.tool_result - the same slot a /fetch result uses, so the grounding branch treats it as a
real lookup. The requests go through the trusted web_fetch tool (GET only, allowlist enforced by the Rust CLI),
so api.github.com must be allowlisted. BOUGH_GITHUB_LOOKUP=off disables it.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

API = "https://api.github.com"
_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_REPO = r"[A-Za-z0-9._-]{1,100}"
_URL = re.compile(rf"github\.com/({_NAME})/({_REPO})", re.I)
_SLUG = re.compile(rf"(?<![\w./-])({_NAME})/({_REPO})(?![\w/-])")
_GITHUB_WORD = re.compile(r"\b(?:github|repo|repository)\b", re.I)
KINDS = (
    ("pulls", re.compile(r"\b(?:pull\s*requests?|prs?|pulls?|merge\s+requests?)\b", re.I)),
    ("issues", re.compile(r"\b(?:issues?|bugs?\s+reports?|tickets?)\b", re.I)),
    ("releases", re.compile(r"\b(?:releases?|versions?|tags?)\b", re.I)),
    ("commits", re.compile(r"\b(?:commits?|pushed|changes?\s+(?:made|pushed)|last\s+change)\b", re.I)),
    ("repo", re.compile(r"\b(?:stars?|forks?|description|license|default\s+branch|language|watchers?)\b", re.I)),
)
LIST_LIMIT = 5


@dataclass(frozen=True)
class GitHubQuery:
    owner: str
    repo: str
    kind: str          # issues | pulls | releases | commits | repo

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def detect(text: str) -> Optional[GitHubQuery]:
    """A repo plus something to look up, or None. Never guesses a repo the user did not name."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = _URL.search(text)
    if m is None and _GITHUB_WORD.search(text):
        m = _SLUG.search(text)
    if m is None:
        return None
    owner, repo = m.group(1), m.group(2).rstrip(".")
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not repo or repo in (".", ".."):
        return None
    for kind, rx in KINDS:
        if rx.search(text):
            return GitHubQuery(owner, repo, kind)
    return None


def failed(code: str, message: str) -> Dict[str, Any]:
    return {"version": 1, "tool": "github", "ok": False, "data": None, "error": {"code": code, "message": message}}


class LookupError_(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


FetchFn = Callable[[str], Awaitable[Dict[str, Any]]]


class GitHubTool:
    """Every request goes through the trusted web_fetch tool (Rust CLI + wrapper), so its allowlist,
    public-IP check, size and time limits apply. api.github.com must be in BOUGH_WEB_FETCH_ALLOWED_HOSTS."""

    def __init__(self, fetch: FetchFn):
        self._fetch = fetch

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = API + path + ("?" + urllib.parse.urlencode(params, safe=":+") if params else "")
        env = await self._fetch(url)
        if not isinstance(env, dict) or not env.get("ok"):
            err = (env or {}).get("error") or {}
            code, msg = str(err.get("code") or "UNKNOWN"), str(err.get("message") or "")
            if code == "PERMISSION_DENIED":
                raise LookupError_(code, "Web fetch is off. Set BOUGH_WEB_FETCH_ENABLED=true and add api.github.com "
                                         "to BOUGH_WEB_FETCH_ALLOWED_HOSTS.")
            if code == "URL_BLOCKED":
                raise LookupError_(code, "api.github.com is not in BOUGH_WEB_FETCH_ALLOWED_HOSTS.")
            if code == "HTTP_ERROR" and "404" in msg:
                raise LookupError_("NOT_FOUND", "not found")
            if code == "HTTP_ERROR" and ("403" in msg or "429" in msg):
                raise LookupError_("RATE_LIMITED", "GitHub API rate limit reached (60 requests/hour, search 10/minute "
                                                   "without a token). Try again later.")
            raise LookupError_(code, msg or "web fetch failed")
        data = env.get("data") or {}
        if data.get("truncated"):
            raise LookupError_("TRUNCATED", "response larger than the web_fetch text limit")
        try:
            return json.loads(data.get("text") or "")
        except ValueError:
            raise LookupError_("BAD_RESPONSE", f"api.github.com did not return JSON ({data.get('content_type')})")

    async def _list(self, path: str, params: Dict[str, Any]) -> Any:
        """Lists carry full bodies (issue text, release notes); retry with one item if the page is too big."""
        try:
            return await self._get(path, params)
        except LookupError_ as e:
            if e.code != "TRUNCATED":
                raise
            return await self._get(path, dict(params, per_page=1))

    async def lookup(self, q: GitHubQuery) -> Dict[str, Any]:
        """Always returns the fetch envelope, never raises."""
        base = f"/repos/{urllib.parse.quote(q.owner)}/{urllib.parse.quote(q.repo)}"
        try:
            info = await self._get(base)
            items: List[Dict[str, Any]] = []
            note = ""
            if q.kind in ("issues", "pulls"):
                if q.kind == "issues" and not info.get("has_issues", True):
                    note = (f"Issues are disabled on {info.get('full_name', q.full_name)}: the repository has no "
                            "GitHub issues at all. Its open_issues_count counts open pull requests only.")
                else:
                    kind = "issue" if q.kind == "issues" else "pr"
                    raw = await self._list("/search/issues", {"q": f"repo:{q.full_name} is:{kind}", "sort": "created",
                                                              "order": "desc", "per_page": 3})
                    items = [_issue(i) for i in (raw.get("items") or [])]
                    if not items:
                        note = "No issues found." if kind == "issue" else "No pull requests found."
            elif q.kind == "releases":
                raw = await self._list(base + "/releases", {"per_page": 3})
                items = [{"name": r.get("name") or r.get("tag_name"), "tag": r.get("tag_name"),
                          "author": (r.get("author") or {}).get("login"), "published_at": r.get("published_at"),
                          "prerelease": r.get("prerelease"), "url": r.get("html_url")} for r in raw]
                if not items:
                    raw = await self._get(base + "/tags", {"per_page": LIST_LIMIT})
                    items = [{"tag": t.get("name"), "commit": (t.get("commit") or {}).get("sha", "")[:12]} for t in raw]
                    note = "No GitHub releases; these are the newest tags." if items else "No releases or tags."
            elif q.kind == "commits":
                raw = await self._list(base + "/commits", {"per_page": LIST_LIMIT})
                items = [{"sha": c.get("sha", "")[:12],
                          "message": ((c.get("commit") or {}).get("message") or "").split("\n", 1)[0][:200],
                          "author": (c.get("author") or {}).get("login")
                          or ((c.get("commit") or {}).get("author") or {}).get("name"),
                          "date": ((c.get("commit") or {}).get("committer") or {}).get("date"),
                          "url": c.get("html_url")} for c in raw]
        except LookupError_ as e:
            if e.code == "NOT_FOUND":
                return failed("NOT_FOUND", f"GitHub has no public repository {q.full_name} (HTTP 404).")
            return failed(e.code, str(e))
        except (ValueError, TypeError, AttributeError) as e:
            return failed("BAD_RESPONSE", f"Unexpected GitHub API response ({type(e).__name__}).")

        repo_info = {k: info.get(k) for k in ("full_name", "html_url", "description", "has_issues",
                                              "open_issues_count", "stargazers_count", "forks_count",
                                              "default_branch", "pushed_at", "archived")}
        payload = {"source": "GitHub REST API via web_fetch",
                   "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "repository": repo_info, "lookup": q.kind, "note": note or None,
                   "newest_first" if q.kind != "repo" else "items": items}
        text = json.dumps(payload, ensure_ascii=False, indent=1)
        final_url = repo_info.get("html_url") or f"https://github.com/{q.full_name}"
        if q.kind != "repo" and not (q.kind == "issues" and note.startswith("Issues are disabled")):
            final_url = f"{final_url}/{q.kind}"
        return {"version": 1, "tool": "github", "ok": True, "error": None,
                "data": {"final_url": final_url, "status": 200, "content_type": "application/json",
                         "title": f"GitHub {q.kind} for {repo_info.get('full_name') or q.full_name}",
                         "text": text, "chars": len(text), "truncated": False}}


def _issue(i: Dict[str, Any]) -> Dict[str, Any]:
    return {"number": i.get("number"), "title": i.get("title"), "author": (i.get("user") or {}).get("login"),
            "created_at": i.get("created_at"), "state": i.get("state"), "url": i.get("html_url")}
