"""Shared types for memory backends.

Every store method takes an `Owner` first. With auth off there is exactly one owner (`LOCAL`);
with Supabase auth each signed-in user is an owner and the Supabase backend forwards the user's own
JWT so Postgres row-level security - not application code - decides which rows are visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

from ..schemas import MemoryRecord

KINDS = ("fact", "preference", "summary")
ORIGINS = ("user_stated", "system_summary", "model_interpretation")
MAX_TEXT = 500


@dataclass(frozen=True)
class Owner:
    user_id: str                      # "local" when auth is off; Supabase auth.users.id otherwise
    token: Optional[str] = None       # the user's access token, forwarded only by the Supabase backend

    def __repr__(self) -> str:        # never print tokens
        return f"Owner({self.user_id!r})"


LOCAL = Owner("local")


class MemoryBackend(Protocol):
    name: str
    max_records: int

    def close(self) -> None: ...
    def add(self, owner: Owner, text: str, kind: str = "fact", origin: str = "user_stated",
            session_id: Optional[str] = None, cycle_id: Optional[str] = None) -> Tuple[MemoryRecord, bool]: ...
    def list(self, owner: Owner, limit: int = 500) -> List[MemoryRecord]: ...
    def get(self, owner: Owner, memory_id: str) -> Optional[MemoryRecord]: ...
    def delete(self, owner: Owner, memory_id: str) -> bool: ...
    def search(self, owner: Owner, query: str, limit: int = 5, min_score: float = 0.15,
               allowed_origins: Iterable[str] = ("user_stated", "system_summary")) -> List[dict]: ...
    def get_all_bias(self, owner: Owner) -> Dict[str, float]: ...
    def apply_bias_delta(self, owner: Owner, kind: str, delta: float, limit: float) -> Tuple[float, float]: ...
    def log_feedback(self, owner: Owner, session_id: str, cycle_id: str, proposal_id: Optional[str],
                     proposal_kind: str, rating: str) -> str: ...
    def list_feedback(self, owner: Owner) -> List[dict]: ...
    def counts(self, owner: Owner) -> Dict[str, int]: ...
