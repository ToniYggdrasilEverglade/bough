"""Supabase (Postgres via PostgREST) memory backend.

The server holds no database credentials. Every request carries the signed-in user's own access
token plus the public publishable key, so Postgres row-level security restricts each call to that
user's rows. If a token is missing or expired PostgREST answers 401 and the call fails closed.

Schema and policies live in supabase/migrations/. Ranking is identical to the SQLite backend: rows
are fetched (bounded by max_records) and scored in Python, so no search logic lives in the database.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..errors import MemoryRejected, MemoryUnavailable
from ..schemas import MemoryRecord, new_id
from .base import Owner
from .store import check_kind_origin, clean_text, rank_rows
from .text import token_set

log = logging.getLogger("bough.memory.supabase")
_FIELDS = "id,kind,text,origin,created_at,source_session,source_cycle"


class SupabaseStore:
    name = "supabase"

    def __init__(self, url: str, publishable_key: str, max_records: int = 1000, timeout_s: float = 10.0):
        self.base = url.rstrip("/") + "/rest/v1/"
        self.key = publishable_key
        self.max_records = max_records
        self.timeout_s = timeout_s

    def close(self) -> None:
        return None

    # ------------------------------------------------------------ transport
    def _request(self, owner: Owner, method: str, path: str, *, params: Optional[Dict[str, str]] = None,
                 body: Any = None, prefer: Optional[str] = None) -> Tuple[Any, Dict[str, str]]:
        if not owner.token:
            raise MemoryUnavailable("persistent memory requires a signed-in user")
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        headers = {"apikey": self.key, "Authorization": f"Bearer {owner.token}", "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                return (json.loads(raw) if raw else None), dict(resp.headers)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read()).get("message", "")
            except Exception:  # noqa: BLE001
                pass
            if e.code in (401, 403):
                raise MemoryUnavailable("memory access denied (sign in again)") from None
            log.warning("supabase %s %s -> %s %s", method, path, e.code, detail[:200])
            raise MemoryUnavailable(f"memory service error ({e.code})") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            log.warning("supabase %s %s failed: %s", method, path, type(e).__name__)
            raise MemoryUnavailable("memory service unreachable") from None

    @staticmethod
    def _rec(r: Dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(r["id"], r["kind"], r["text"], r["origin"], r["created_at"],
                            r.get("source_session"), r.get("source_cycle"))

    # ------------------------------------------------------------ memories
    def add(self, owner: Owner, text: str, kind: str = "fact", origin: str = "user_stated",
            session_id: Optional[str] = None, cycle_id: Optional[str] = None) -> Tuple[MemoryRecord, bool]:
        check_kind_origin(kind, origin)
        text = clean_text(text)
        existing, _ = self._request(owner, "GET", "memories",
                                    params={"select": _FIELDS, "text_lower": f"eq.{text.lower()}", "limit": "1"})
        if existing:
            return self._rec(existing[0]), False
        _, hdrs = self._request(owner, "HEAD", "memories", params={"select": "id"}, prefer="count=exact")
        n = _count_from_range(hdrs)
        if n is not None and n >= self.max_records:
            raise MemoryRejected(f"memory limit reached ({self.max_records}); delete something first")
        row = {"id": new_id("mem_"), "kind": kind, "text": text, "origin": origin,
               "source_session": session_id, "source_cycle": cycle_id, "tokens": " ".join(sorted(token_set(text)))}
        created, _ = self._request(owner, "POST", "memories", params={"select": _FIELDS}, body=row,
                                   prefer="return=representation")
        return self._rec(created[0]), True

    def list(self, owner: Owner, limit: int = 500) -> List[MemoryRecord]:
        rows, _ = self._request(owner, "GET", "memories",
                                params={"select": _FIELDS, "order": "created_at.desc", "limit": str(limit)})
        return [self._rec(r) for r in rows or []]

    def get(self, owner: Owner, memory_id: str) -> Optional[MemoryRecord]:
        rows, _ = self._request(owner, "GET", "memories", params={"select": _FIELDS, "id": f"eq.{memory_id}"})
        return self._rec(rows[0]) if rows else None

    def delete(self, owner: Owner, memory_id: str) -> bool:
        rows, _ = self._request(owner, "DELETE", "memories", params={"id": f"eq.{memory_id}", "select": "id"},
                                prefer="return=representation")
        return bool(rows)

    def search(self, owner: Owner, query: str, limit: int = 5, min_score: float = 0.15,
               allowed_origins: Iterable[str] = ("user_stated", "system_summary")) -> List[dict]:
        q = token_set(query)
        origins = list(allowed_origins)
        if not q or not origins:
            return []
        rows, _ = self._request(owner, "GET", "memories", params={
            "select": _FIELDS + ",tokens", "origin": "in.(" + ",".join(origins) + ")",
            "order": "created_at.desc", "limit": str(self.max_records)})
        return rank_rows(q, [(self._rec(r), set((r.get("tokens") or "").split())) for r in rows or []], min_score, limit)

    # ------------------------------------------------------------ adaptation + feedback
    def get_all_bias(self, owner: Owner) -> Dict[str, float]:
        rows, _ = self._request(owner, "GET", "adaptation", params={"select": "kind,bias"})
        return {r["kind"]: float(r["bias"]) for r in rows or []}

    def apply_bias_delta(self, owner: Owner, kind: str, delta: float, limit: float) -> Tuple[float, float]:
        res, _ = self._request(owner, "POST", "rpc/apply_bias_delta",
                               body={"p_kind": kind, "p_delta": delta, "p_limit": limit})
        row = res[0] if isinstance(res, list) else res
        return float(row["old_bias"]), float(row["new_bias"])

    def log_feedback(self, owner: Owner, session_id: str, cycle_id: str, proposal_id: Optional[str],
                     proposal_kind: str, rating: str) -> str:
        fid = new_id("fb_")
        self._request(owner, "POST", "feedback", body={
            "id": fid, "session_id": session_id, "cycle_id": cycle_id, "proposal_id": proposal_id,
            "proposal_kind": proposal_kind, "rating": rating}, prefer="return=minimal")
        return fid

    def list_feedback(self, owner: Owner) -> List[dict]:
        rows, _ = self._request(owner, "GET", "feedback", params={"select": "*", "order": "created_at.asc"})
        return list(rows or [])

    def counts(self, owner: Owner) -> Dict[str, int]:
        out = {}
        for t in ("memories", "feedback", "adaptation"):
            _, hdrs = self._request(owner, "HEAD", t, params={"select": "*"}, prefer="count=exact")
            out[t] = _count_from_range(hdrs) or 0
        return out


def _count_from_range(headers: Dict[str, str]) -> Optional[int]:
    rng = headers.get("Content-Range") or headers.get("content-range") or ""
    total = rng.rsplit("/", 1)[-1]
    return int(total) if total.isdigit() else None
