"""Per-cycle execution context: budget counters, collected events, bookkeeping."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class ModelBudget:
    limit: int
    used: int = 0

    def try_consume(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


@dataclass
class CycleContext:
    session_id: str
    cycle_id: str
    text: str
    budget: ModelBudget
    events: List[dict] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)       # stage outputs for the result record
    components: List[str] = field(default_factory=list)
    limits_hit: List[dict] = field(default_factory=list)
    parent_id: Optional[str] = None
    terminal: bool = False
    cancel_reason: Optional[str] = None
    started: float = field(default_factory=time.monotonic)
    emit: Optional[Callable[..., Awaitable[Any]]] = None     # set by the orchestrator
