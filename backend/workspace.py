"""Working memory / shared workspace. One instance per session; nothing is shared between sessions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .errors import StaleCycleError
from .schemas import clip, jsonable, new_id, utcnow_iso

KINDS = ("goal", "observation", "memory", "proposal", "evaluation", "selected_action", "tool_result")


@dataclass
class WorkspaceItem:
    item_id: str
    kind: str
    source: str
    timestamp: str
    cycle_id: str
    content: Dict[str, Any]
    priority: float
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)

    def preview(self, n: int = 140) -> Dict[str, Any]:
        return {"item_id": self.item_id, "kind": self.kind, "source": self.source,
                "timestamp": self.timestamp, "cycle_id": self.cycle_id, "priority": self.priority,
                "preview": clip(json.dumps(self.content, ensure_ascii=False, sort_keys=True), n)}


class Workspace:
    def __init__(self, session_id: str, max_items: int, max_item_chars: int = 1500):
        self.session_id = session_id
        self.max_items = max_items
        self.max_item_chars = max_item_chars
        self.current_cycle_id: Optional[str] = None
        self._items: List[WorkspaceItem] = []
        self.last_evicted: List[str] = []

    def begin_cycle(self, cycle_id: str) -> None:
        self.current_cycle_id = cycle_id

    def add(self, kind: str, source: str, cycle_id: str, content: Dict[str, Any],
            priority: float = 0.5) -> Optional[WorkspaceItem]:
        """Add an item. Returns None for an exact duplicate inside the same cycle.
        Raises StaleCycleError if `cycle_id` is not the session's current cycle."""
        if kind not in KINDS:
            raise ValueError(f"unknown workspace item kind: {kind}")
        if cycle_id != self.current_cycle_id:
            raise StaleCycleError(f"cycle {cycle_id} is not current for session {self.session_id}")
        content = jsonable(content)
        blob = json.dumps(content, ensure_ascii=False, sort_keys=True)
        if len(blob) > self.max_item_chars:
            content = {"truncated": True, "preview": blob[: self.max_item_chars]}
            blob = json.dumps(content, ensure_ascii=False)
        fp = hashlib.sha1(f"{kind}|{blob}".encode()).hexdigest()
        if any(i.cycle_id == cycle_id and i.fingerprint == fp for i in self._items):
            return None
        item = WorkspaceItem(new_id("ws_"), kind, source, utcnow_iso(), cycle_id, content, priority, fp)
        self._items.append(item)
        self.last_evicted = []
        while len(self._items) > self.max_items:
            victims = [i for i in self._items if not (i.kind == "goal" and i.cycle_id == self.current_cycle_id)]
            if not victims:
                break
            v = min(victims, key=lambda i: (i.cycle_id == self.current_cycle_id, i.priority, i.timestamp))
            self._items.remove(v)
            self.last_evicted.append(v.item_id)
        return item

    def items(self, kind: Optional[str] = None, cycle_id: Optional[str] = None,
              exclude_cycle: Optional[str] = None) -> List[WorkspaceItem]:
        return [i for i in self._items
                if (kind is None or i.kind == kind)
                and (cycle_id is None or i.cycle_id == cycle_id)
                and (exclude_cycle is None or i.cycle_id != exclude_cycle)]

    def __len__(self) -> int:
        return len(self._items)

    def snapshot(self, preview_chars: int = 140) -> List[Dict[str, Any]]:
        return [i.preview(preview_chars) for i in self._items]
