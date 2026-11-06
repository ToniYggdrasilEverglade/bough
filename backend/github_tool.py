"""Read-only GitHub lookups for questions about a public repository (issues, pull requests, releases, commits).

The root model cannot browse, so "what's the newest issue in torvalds/linux?" used to get a "go check it
yourself" reply. When a message names a repository (github.com/owner/repo, or owner/repo next to the word
GitHub/repo) and asks about one of those things, Bough asks the GitHub REST API directly and hands the answer
to the model as data.tool_result - the same slot a /fetch result uses, so the grounding branch treats it as a
real lookup. Only GET requests to api.github.com, no writes. BOUGH_GITHUB_LOOKUP=off disables it.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


class GitHubTool:
    def __init__(self, token: str = "", timeout_s: float = 8.0, opener=None):
        self._token = token
        self.timeout_s = timeout_s
        self._open = opener                 # None = urllib.request.urlopen (looked up per call)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "Bough/1.0"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        with (self._open or urllib.request.urlopen)(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read(2_000_000).decode("utf-8"))

    def lookup(self, q: GitHubQuery) -> Dict[str, Any]:
        """Blocking; call through asyncio.to_thread. Always returns the fetch envelope, never raises."""
        base = f"/repos/{urllib.parse.quote(q.owner)}/{urllib.parse.quote(q.repo)}"
        try:
            info = self._get(base)
            items: List[Dict[str, Any]] = []
            note = ""
            if q.kind == "issues":
                if not info.get("has_issues", True):
                    note = (f"Issues are disabled on {info.get('full_name', q.full_name)}: the repository has no "
                            "GitHub issues at all. Its open_issues_count counts open pull requests only.")
                else:
                    raw = self._get(base + "/issues", {"state": "all", "sort": "created", "direction": "desc",
                                                       "per_page": 30})
                    items = [_issue(i) for i in raw if "pull_request" not in i][:LIST_LIMIT]
                    if not items:
                        note = "No issues found (pull requests are excluded)."
            elif q.kind == "pulls":
                raw = self._get(base + "/pulls", {"state": "all", "sort": "created", "direction": "desc",
                                                  "per_page": LIST_LIMIT})
                items = [_issue(i) for i in raw]
                if not items:
                    note = "No pull requests found."
            elif q.kind == "releases":
                raw = self._get(base + "/releases", {"per_page": LIST_LIMIT})
                items = [{"name": r.get("name") or r.get("tag_name"), "tag": r.get("tag_name"),
                          "author": (r.get("author") or {}).get("login"), "published_at": r.get("published_at"),
                          "prerelease": r.get("prerelease"), "url": r.get("html_url")} for r in raw]
                if not items:
                    raw = self._get(base + "/tags", {"per_page": LIST_LIMIT})
                    items = [{"tag": t.get("name"), "commit": (t.get("commit") or {}).get("sha", "")[:12]} for t in raw]
                    note = "No GitHub releases; these are the newest tags." if items else "No releases or tags."
            elif q.kind == "commits":
                raw = self._get(base + "/commits", {"per_page": LIST_LIMIT})
                items = [{"sha": c.get("sha", "")[:12],
                          "message": ((c.get("commit") or {}).get("message") or "").split("\n", 1)[0][:200],
                          "author": (c.get("author") or {}).get("login")
                          or ((c.get("commit") or {}).get("author") or {}).get("name"),
                          "date": ((c.get("commit") or {}).get("committer") or {}).get("date"),
                          "url": c.get("html_url")} for c in raw]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return failed("NOT_FOUND", f"GitHub has no public repository {q.full_name} (HTTP 404).")
            if e.code in (403, 429):
                return failed("RATE_LIMITED", "GitHub API rate limit reached (60 requests/hour without a token). "
                              "Set BOUGH_GITHUB_TOKEN to raise it, or try again later.")
            return failed("HTTP_ERROR", f"GitHub API returned HTTP {e.code}.")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return failed("NETWORK", f"Could not reach api.github.com ({type(e).__name__}).")
        except (ValueError, TypeError, AttributeError) as e:
            return failed("BAD_RESPONSE", f"Unexpected GitHub API response ({type(e).__name__}).")

        repo_info = {k: info.get(k) for k in ("full_name", "html_url", "description", "has_issues",
                                              "open_issues_count", "stargazers_count", "forks_count",
                                              "default_branch", "pushed_at", "archived")}
        payload = {"source": "GitHub REST API", "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
