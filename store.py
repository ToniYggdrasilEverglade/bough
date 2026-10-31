"""SQLite-backed long-term memory, feedback log and bounded adaptation weights.

Only explicit, approved records are written here. Records carry an `origin` so user-provided
statements are never confused with system summaries or model interpretations.
"""
from __future__ import annotations

import math
import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..errors import MemoryRejected
from ..schemas import MemoryRecord, new_id, utcnow_iso
from .text import token_set

KINDS = ("fact", "preference", "summary")
ORIGINS = ("user_stated", "system_summary", "model_interpretation")
MAX_TEXT = 500
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL, origin TEXT NOT NULL,
  created_at TEXT NOT NULL, source_session TEXT, source_cycle TEXT, tokens TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback(
  id TEXT PRIMARY KEY, session_id TEXT, cycle_id TEXT, proposal_id TEXT,
  proposal_kind TEXT NOT NULL, rating TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS adaptation(
  kind TEXT PRIMARY KEY, bias REAL NOT NULL, updated_at TEXT NOT NULL);
"""


class MemoryStore:
    def __init__(self, path: str = ":memory:", max_records: int = 1000):
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.max_records = max_records
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ memories
    @staticmethod
    def _row(r: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(r["id"], r["kind"], r["text"], r["origin"], r["created_at"],
                            r["source_session"], r["source_cycle"])

    def add(self, text: str, kind: str = "fact", origin: str = "user_stated",
            session_id: Optional[str] = None, cycle_id: Optional[str] = None) -> Tuple[MemoryRecord, bool]:
        """Store a record. Returns (record, created). Identical text returns the existing record."""
        if kind not in KINDS:
            raise MemoryRejected(f"kind must be one of {KINDS}")
        if origin not in ORIGINS:
            raise MemoryRejected(f"origin must be one of {ORIGINS}")
        text = " ".join(_CTRL.sub(" ", text or "").split())
        if not text:
            raise MemoryRejected("memory text is empty")
        if len(text) > MAX_TEXT:
            raise MemoryRejected(f"memory text is longer than {MAX_TEXT} characters")
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE lower(text)=lower(?)", (text,)).fetchone()
            if row:
                return self._row(row), False
            n = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if n >= self.max_records:
                raise MemoryRejected(f"memory limit reached ({self.max_records}); delete something first")
            rec = MemoryRecord(new_id("mem_"), kind, text, origin, utcnow_iso(), session_id, cycle_id)
            self._conn.execute(
                "INSERT INTO memories(id,kind,text,origin,created_at,source_session,source_cycle,tokens) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (rec.id, kind, text, origin, rec.created_at, session_id, cycle_id,
                 " ".join(sorted(token_set(text)))))
            self._conn.commit()
            return rec, True

    def list(self, limit: int = 500) -> List[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row(r) if r else None

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def search(self, query: str, limit: int = 5, min_score: float = 0.15,
               allowed_origins: Iterable[str] = ("user_stated", "system_summary")) -> List[dict]:
        """Keyword-overlap search: score = |overlap| / sqrt(|query| * |memory|). No embeddings."""
        q = token_set(query)
        if not q:
            return []
        origins = list(allowed_origins)
        if not origins:
            return []
        marks = ",".join("?" for _ in origins)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE origin IN ({marks}) ORDER BY created_at DESC", origins).fetchall()
        scored = []
        for r in rows:
            m = set(r["tokens"].split())
            if not m:
                continue
            ov = len(q & m)
            if ov == 0:
                continue
            score = ov / math.sqrt(len(q) * len(m))
            if score >= min_score:
                d = self._row(r).to_dict()
                d["score"] = round(score, 3)
                scored.append(d)
        scored.sort(key=lambda d: d["created_at"], reverse=True)   # newest first...
        scored.sort(key=lambda d: -d["score"])                     # ...then stable sort by score
        return scored[:limit]

    # ------------------------------------------------------------ adaptation + feedback
    def get_all_bias(self) -> Dict[str, float]:
        with self._lock:
            return {r["kind"]: r["bias"] for r in self._conn.execute("SELECT kind,bias FROM adaptation")}

    def apply_bias_delta(self, kind: str, delta: float, limit: float) -> Tuple[float, float]:
        with self._lock:
            row = self._conn.execute("SELECT bias FROM adaptation WHERE kind=?", (kind,)).fetchone()
            old = row["bias"] if row else 0.0
            new = max(-limit, min(limit, old + delta))
            self._conn.execute(
                "INSERT INTO adaptation(kind,bias,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(kind) DO UPDATE SET bias=excluded.bias, updated_at=excluded.updated_at",
                (kind, new, utcnow_iso()))
            self._conn.commit()
            return old, new

    def log_feedback(self, session_id: str, cycle_id: str, proposal_id: Optional[str],
                     proposal_kind: str, rating: str) -> str:
        fid = new_id("fb_")
        with self._lock:
            self._conn.execute("INSERT INTO feedback VALUES(?,?,?,?,?,?,?)",
                               (fid, session_id, cycle_id, proposal_id, proposal_kind, rating, utcnow_iso()))
            self._conn.commit()
        return fid

    def list_feedback(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM feedback ORDER BY created_at")]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {t: self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("memories", "feedback", "adaptation")}
