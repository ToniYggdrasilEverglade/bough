"""Built-in knowledge seeds: read-only guidance shipped with the app.

Seeds are NOT personal memory. They live in `backend/memory/seeds/*.json`, are versioned with the code,
belong to no user, and can never be written to or deleted through the API. At retrieval time they are
ranked with the same keyword-overlap scoring as real memories (`rank_rows`) so they reach the planner and
the language model as memories with `origin == "seed"`, and evaluation accepts their ids in `supported_by`.

Why a file and not a table: the same seeds must be visible to every user on both the SQLite and the
Supabase backend, they must be read-only, and they change only when the code changes. A JSON directory
gives all of that with zero migration surface. Set BRAIN_SEED_ENABLED=false to turn them off.

File format:
    {"domain": "python", "description": "...",
     "entries": [{"id": "py-001", "text": "...", "tags": ["python", "venv"], "kind": "guideline"}]}
`tags` are only used for retrieval (they are tokenised together with the text); ids must be unique
across all files and are exposed as "seed:<id>". An optional `"pin": "always" | "code" | "writing"`
includes the entry whenever the request's detected domain matches (or always), regardless of keyword
overlap - that is how the handful of non-negotiable rules (answer first, complete code) always reach
the model while the rest of the seeds compete on relevance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..schemas import MemoryRecord
from .store import rank_rows
from .text import token_set

log = logging.getLogger("brain.memory.seed")

SEED_DIR = Path(__file__).with_name("seeds")
SEED_ORIGIN = "seed"
SEED_CREATED_AT = "2026-11-01T00:00:00.000+00:00"   # constant: seeds have no meaningful age
MAX_TEXT = 500
PIN_SCORE = 0.5                     # pinned entries outrank keyword hits but stay below exact user memories
PIN_VALUES = ("always", "code", "writing", "general")


class SeedStore:
    """In-process, immutable index over the seed JSON files."""

    def __init__(self, directory: Optional[Path] = None, enabled: bool = True):
        self.directory = Path(directory) if directory else SEED_DIR
        self.enabled = enabled
        self._rows: List[Tuple[MemoryRecord, Set[str]]] = []
        self._by_id: Dict[str, MemoryRecord] = {}
        self._pins: Dict[str, List[MemoryRecord]] = {}      # pin value -> records
        self.domains: Dict[str, int] = {}
        self.problems: List[str] = []
        if enabled:
            self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if not self.directory.is_dir():
            self.problems.append(f"seed directory missing: {self.directory}")
            return
        for path in sorted(self.directory.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                domain = str(doc.get("domain") or path.stem)
                entries = doc.get("entries")
                if not isinstance(entries, list):
                    raise ValueError("'entries' must be a list")
            except (OSError, ValueError) as e:
                self.problems.append(f"{path.name}: {e}")
                continue
            count = 0
            for i, e in enumerate(entries):
                err = self._add(domain, e)
                if err:
                    self.problems.append(f"{path.name}[{i}]: {err}")
                else:
                    count += 1
            self.domains[domain] = count
        for p in self.problems:
            log.warning("seed: %s", p)
        log.info("seed: loaded %d entries from %d file(s)", len(self._rows), len(self.domains))

    def _add(self, domain: str, e: object) -> Optional[str]:
        if not isinstance(e, dict):
            return "entry must be an object"
        raw_id, text = e.get("id"), e.get("text")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return "missing id"
        if not isinstance(text, str) or not text.strip():
            return f"{raw_id}: missing text"
        text = " ".join(text.split())
        if len(text) > MAX_TEXT:
            return f"{raw_id}: text longer than {MAX_TEXT} chars"
        tags = e.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return f"{raw_id}: tags must be a list of strings"
        mid = f"seed:{raw_id.strip()}"
        if mid in self._by_id:
            return f"{raw_id}: duplicate id"
        kind = e.get("kind", "guideline")
        if not isinstance(kind, str) or not kind:
            kind = "guideline"
        pin = e.get("pin")
        if pin is not None and pin not in PIN_VALUES:
            return f"{raw_id}: pin must be one of {PIN_VALUES}"
        rec = MemoryRecord(id=mid, kind=kind, text=text, origin=SEED_ORIGIN, created_at=SEED_CREATED_AT,
                           source_session=None, source_cycle=domain)
        tokens = token_set(text) | token_set(" ".join(tags))
        self._rows.append((rec, tokens))
        self._by_id[mid] = rec
        if pin:
            self._pins.setdefault(pin, []).append(rec)
        return None

    # ------------------------------------------------------------------ read API
    def __len__(self) -> int:
        return len(self._rows)

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self._by_id.get(memory_id)

    def search(self, query: str, limit: int = 3, min_score: float = 0.15, domain: str = "general") -> List[dict]:
        """Pinned entries for `domain` first, then keyword hits scored exactly like the memory backends.
        Returns dicts shaped like `MemoryRecord.to_dict()` + score."""
        if not self.enabled or not self._rows or limit <= 0:
            return []
        out: List[dict] = []
        seen: Set[str] = set()
        for rec in self._pins.get("always", []) + self._pins.get(domain, []):
            if rec.id not in seen:
                d = rec.to_dict()
                d["score"] = PIN_SCORE
                d["pinned"] = True
                out.append(d)
                seen.add(rec.id)
        q = token_set(query)
        if q:
            for d in rank_rows(q, self._rows, min_score, limit + len(seen)):
                if d["id"] not in seen:
                    out.append(d)
                    seen.add(d["id"])
        return out[: max(limit, len(self._pins.get("always", [])) + len(self._pins.get(domain, [])))]

    def describe(self) -> Dict[str, object]:
        return {"enabled": self.enabled, "count": len(self._rows), "domains": dict(self.domains),
                "pinned": {k: [r.id for r in v] for k, v in self._pins.items()}, "problems": list(self.problems)}

    def ids(self) -> Iterable[str]:
        return self._by_id.keys()
