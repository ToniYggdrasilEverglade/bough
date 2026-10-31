"""SQLite-backed long-term memory, feedback log and bounded adaptation weights.

Only explicit, approved records are written here. Records carry an `origin` so user-provided
statements are never confused with system summaries or model interpretations.

Rows are scoped by `owner` (see memory/base.py). Databases created before owners existed are
migrated in place: existing rows are assigned to the LOCAL owner.
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
from .base import KINDS, MAX_TEXT, ORIGINS, Owner
from .text import token_set

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id TEXT PRIMARY KEY, owner TEXT NOT NULL DEFAULT 'local', kind TEXT NOT NULL, text TEXT NOT NULL,
  origin TEXT NOT NULL, created_at TEXT NOT NULL, source_session TEXT, source_cycle TEXT, tokens TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS memories_owner ON memories(owner, created_at DESC);
CREATE TABLE IF NOT EXISTS feedback(
  id TEXT PRIMARY KEY, owner TEXT NOT NULL DEFAULT 'local', session_id TEXT, cycle_id TEXT, proposal_id TEXT,
  proposal_kind TEXT NOT NULL, rating TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS adaptation(
  owner TEXT NOT NULL DEFAULT 'local', kind TEXT NOT NULL, bias REAL NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(owner, kind));
"""


def clean_text(text: str) -> str:
    text = " ".join(_CTRL.sub(" ", text or "").split())
    if not text:
        raise MemoryRejected("memory text is empty")
    if len(text) > MAX_TEXT:
        raise MemoryRejected(f"memory text is longer than {MAX_TEXT} characters")
    return text


def check_kind_origin(kind: str, origin: str) -> None:
    if kind not in KINDS:
        raise MemoryRejected(f"kind must be one of {KINDS}")
    if origin not in ORIGINS:
        raise MemoryRejected(f"origin must be one of {ORIGINS}")


class MemoryStore:
    name = "sqlite"

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
            self._migrate_pre_owner()
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _columns(self, table: str) -> List[str]:
        return [r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")]

    def _migrate_pre_owner(self) -> None:
        """Add the owner column to databases created before multi-user support."""
        for table in ("memories", "feedback"):
            cols = self._columns(table)
            if cols and "owner" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN owner TEXT NOT NULL DEFAULT 'local'")
        cols = self._columns("adaptation")
        if cols and "owner" not in cols:          # primary key changes need a rebuild
            self._conn.executescript("""
                ALTER TABLE adaptation RENAME TO adaptation_old;
                CREATE TABLE adaptation(owner TEXT NOT NULL DEFAULT 'local', kind TEXT NOT NULL, bias REAL NOT NULL,
                                        updated_at TEXT NOT NULL, PRIMARY KEY(owner, kind));
                INSERT INTO adaptation(owner, kind, bias, updated_at)
                    SELECT 'local', kind, bias, updated_at FROM adaptation_old;
                DROP TABLE adaptation_old;
            """)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ memories
    @staticmethod
    def _row(r: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(r["id"], r["kind"], r["text"], r["origin"], r["created_at"],
                            r["source_session"], r["source_cycle"])

    def add(self, owner: Owner, text: str, kind: str = "fact", origin: str = "user_stated",
            session_id: Optional[str] = None, cycle_id: Optional[str] = None) -> Tuple[MemoryRecord, bool]:
        """Store a record. Returns (record, created). Identical text returns the existing record."""
        check_kind_origin(kind, origin)
        text = clean_text(text)
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE owner=? AND lower(text)=lower(?)",
                                     (owner.user_id, text)).fetchone()
            if row:
                return self._row(row), False
            n = self._conn.execute("SELECT COUNT(*) FROM memories WHERE owner=?", (owner.user_id,)).fetchone()[0]
            if n >= self.max_records:
                raise MemoryRejected(f"memory limit reached ({self.max_records}); delete something first")
            rec = MemoryRecord(new_id("mem_"), kind, text, origin, utcnow_iso(), session_id, cycle_id)
            self._conn.execute(
                "INSERT INTO memories(id,owner,kind,text,origin,created_at,source_session,source_cycle,tokens) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (rec.id, owner.user_id, kind, text, origin, rec.created_at, session_id, cycle_id,
                 " ".join(sorted(token_set(text)))))
            self._conn.commit()
            return rec, True

    def list(self, owner: Owner, limit: int = 500) -> List[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories WHERE owner=? ORDER BY created_at DESC LIMIT ?",
                                      (owner.user_id, limit)).fetchall()
        return [self._row(r) for r in rows]

    def get(self, owner: Owner, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM memories WHERE owner=? AND id=?", (owner.user_id, memory_id)).fetchone()
        return self._row(r) if r else None

    def delete(self, owner: Owner, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE owner=? AND id=?", (owner.user_id, memory_id))
            self._conn.commit()
            return cur.rowcount > 0

    def search(self, owner: Owner, query: str, limit: int = 5, min_score: float = 0.15,
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
                f"SELECT * FROM memories WHERE owner=? AND origin IN ({marks}) ORDER BY created_at DESC",
                [owner.user_id, *origins]).fetchall()
        return rank_rows(q, [(self._row(r), set(r["tokens"].split())) for r in rows], min_score, limit)

    # ------------------------------------------------------------ adaptation + feedback
    def get_all_bias(self, owner: Owner) -> Dict[str, float]:
        with self._lock:
            return {r["kind"]: r["bias"] for r in
                    self._conn.execute("SELECT kind,bias FROM adaptation WHERE owner=?", (owner.user_id,))}

    def apply_bias_delta(self, owner: Owner, kind: str, delta: float, limit: float) -> Tuple[float, float]:
        with self._lock:
            row = self._conn.execute("SELECT bias FROM adaptation WHERE owner=? AND kind=?",
                                     (owner.user_id, kind)).fetchone()
            old = row["bias"] if row else 0.0
            new = max(-limit, min(limit, old + delta))
            self._conn.execute(
                "INSERT INTO adaptation(owner,kind,bias,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(owner,kind) DO UPDATE SET bias=excluded.bias, updated_at=excluded.updated_at",
                (owner.user_id, kind, new, utcnow_iso()))
            self._conn.commit()
            return old, new

    def log_feedback(self, owner: Owner, session_id: str, cycle_id: str, proposal_id: Optional[str],
                     proposal_kind: str, rating: str) -> str:
        fid = new_id("fb_")
        with self._lock:
            self._conn.execute("INSERT INTO feedback(id,owner,session_id,cycle_id,proposal_id,proposal_kind,rating,created_at) "
                               "VALUES(?,?,?,?,?,?,?,?)",
                               (fid, owner.user_id, session_id, cycle_id, proposal_id, proposal_kind, rating, utcnow_iso()))
            self._conn.commit()
        return fid

    def list_feedback(self, owner: Owner) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM feedback WHERE owner=? ORDER BY created_at", (owner.user_id,))]

    def counts(self, owner: Owner) -> Dict[str, int]:
        with self._lock:
            return {t: self._conn.execute(f"SELECT COUNT(*) FROM {t} WHERE owner=?", (owner.user_id,)).fetchone()[0]
                    for t in ("memories", "feedback", "adaptation")}


def rank_rows(q: set, rows: List[Tuple[MemoryRecord, set]], min_score: float, limit: int) -> List[dict]:
    """Shared ranking for all backends: newest first, then stable sort by score."""
    scored = []
    for rec, m in rows:
        if not m:
            continue
        ov = len(q & m)
        if ov == 0:
            continue
        score = ov / math.sqrt(len(q) * len(m))
        if score >= min_score:
            d = rec.to_dict()
            d["score"] = round(score, 3)
            scored.append(d)
    scored.sort(key=lambda d: d["created_at"], reverse=True)
    scored.sort(key=lambda d: -d["score"])
    return scored[:limit]
