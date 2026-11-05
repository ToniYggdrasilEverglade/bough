"""Typed in-process event bus with duplicate suppression and runaway-loop protection.

The orchestrator is the only component that routes work; the bus exists for observability and for
subscribers (WebSocket broadcaster, tests). Guards:
  * duplicate event_ids are ignored (no duplicate processing),
  * per-cycle total and per-(source,type) caps raise LoopGuardError (no self-triggering loops),
  * subscriber exceptions never propagate back into the cycle and never publish new events.
"""
from __future__ import annotations

import inspect
import logging
from collections import Counter, OrderedDict, defaultdict, deque
from typing import Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Union

from .errors import LoopGuardError
from .schemas import Event, EventType

log = logging.getLogger("bough.bus")

# Terminal / bookkeeping events are always allowed so a tripped guard can still report itself.
_EXEMPT = {EventType.CYCLE_ERROR, EventType.CYCLE_CANCELLED, EventType.CYCLE_TIMEOUT,
           EventType.RESPONSE_FINAL, EventType.LIMIT_REACHED}

Callback = Callable[[Event], Union[None, Awaitable[None]]]


class _Sub:
    def __init__(self, cb: Callback, session_id: Optional[str], types: Optional[set]):
        self.cb, self.session_id, self.types = cb, session_id, types


class EventBus:
    def __init__(self, max_events_per_cycle: int = 200, max_per_source_type: int = 10,
                 history_per_session: int = 1000, dedupe_window: int = 20000):
        self.max_events_per_cycle = max_events_per_cycle
        self.max_per_source_type = max_per_source_type
        self._subs: List[_Sub] = []
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self._dedupe_window = dedupe_window
        self._counts: Dict[str, Counter] = defaultdict(Counter)
        self._totals: Dict[str, int] = defaultdict(int)
        self._history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=history_per_session))
        self.duplicates_dropped = 0
        self.subscriber_errors = 0

    def subscribe(self, callback: Callback, *, session_id: Optional[str] = None,
                  event_types: Optional[Iterable[EventType]] = None) -> Callable[[], None]:
        sub = _Sub(callback, session_id, set(event_types) if event_types else None)
        self._subs.append(sub)

        def unsubscribe() -> None:
            if sub in self._subs:
                self._subs.remove(sub)
        return unsubscribe

    async def publish(self, event: Event) -> bool:
        if event.event_id in self._seen:
            self.duplicates_dropped += 1
            return False
        if event.event_type not in _EXEMPT:
            key = f"{event.source}|{event.event_type.value}"
            self._totals[event.cycle_id] += 1
            self._counts[event.cycle_id][key] += 1
            if self._totals[event.cycle_id] > self.max_events_per_cycle:
                raise LoopGuardError(f"cycle exceeded {self.max_events_per_cycle} events")
            if self._counts[event.cycle_id][key] > self.max_per_source_type:
                raise LoopGuardError(f"'{key}' fired more than {self.max_per_source_type} times in one cycle")
        self._seen[event.event_id] = None
        while len(self._seen) > self._dedupe_window:
            self._seen.popitem(last=False)
        self._history[event.session_id].append(event.to_dict())
        for sub in list(self._subs):
            if sub.session_id is not None and sub.session_id != event.session_id:
                continue
            if sub.types is not None and event.event_type not in sub.types:
                continue
            try:
                res = sub.cb(event)
                if inspect.isawaitable(res):
                    await res
            except Exception:  # noqa: BLE001 - a broken subscriber must not break the cycle
                self.subscriber_errors += 1
                log.warning("event subscriber failed", exc_info=False)
        return True

    def history(self, session_id: str, cycle_id: Optional[str] = None) -> List[dict]:
        items = list(self._history.get(session_id, ()))
        return [e for e in items if cycle_id is None or e["cycle_id"] == cycle_id]

    def end_cycle(self, cycle_id: str) -> None:
        self._counts.pop(cycle_id, None)
        self._totals.pop(cycle_id, None)
