"""Typed schemas for events and component outputs (stdlib dataclasses, validated at boundaries)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / enums into plain JSON-compatible structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj


def clip(text: Any, n: int) -> str:
    s = str(text)
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


# ---------------------------------------------------------------- events
class EventType(str, Enum):
    INPUT_RECEIVED = "input.received"
    CYCLE_STARTED = "cycle.started"
    COMPONENT_ACTIVATED = "component.activated"
    PERCEPTION_DONE = "perception.done"
    MEMORY_RETRIEVED = "memory.retrieved"
    ATTENTION_DONE = "attention.done"
    WORKSPACE_UPDATED = "workspace.updated"
    MODEL_CALL = "model.call"
    PLANNING_DONE = "planning.done"
    ROOT_DONE = "root.done"                 # root pipeline: free-form model draft (or repair) finished
    BRANCHES_DONE = "branches.done"         # root pipeline: rule-based branch checks on a draft
    EVALUATION_DONE = "evaluation.done"
    ACTION_SELECTED = "action.selected"
    TOOL_STARTED = "tool.started"
    TOOL_DONE = "tool.done"
    MEMORY_WRITE = "memory.write"
    LANGUAGE_DONE = "language.done"
    RESPONSE_FINAL = "response.final"
    FEEDBACK_RECORDED = "feedback.recorded"
    CYCLE_CANCELLED = "cycle.cancelled"
    CYCLE_TIMEOUT = "cycle.timeout"
    CYCLE_ERROR = "cycle.error"
    LIMIT_REACHED = "limit.reached"


@dataclass(frozen=True)
class Event:
    event_id: str
    session_id: str
    cycle_id: str
    source: str
    event_type: EventType
    timestamp: str
    payload: Dict[str, Any]
    parent_event_id: Optional[str] = None

    @staticmethod
    def create(session_id: str, cycle_id: str, source: str, event_type: EventType,
               payload: Optional[Dict[str, Any]] = None, parent_event_id: Optional[str] = None) -> "Event":
        return Event(new_id("ev_"), session_id, cycle_id, source, event_type, utcnow_iso(),
                     jsonable(payload or {}), parent_event_id)

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


# ---------------------------------------------------------------- perception
@dataclass
class RawInput:
    modality: str            # "text" now; "image" | "audio" | "sensor" later
    data: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    text: str
    type: str                # quoted | number | proper_noun


@dataclass
class Constraint:
    id: str
    type: str                # max_words | max_sentences | style | format | forbid_term
    value: Any
    source_text: str


@dataclass
class Observation:
    raw_text: str                       # original input, untouched
    request: str                        # what the system should act on
    intent: str                         # question | request | statement | save_memory
    entities: List[Entity] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    underspecified: bool = False
    missing: List[str] = field(default_factory=list)
    save_text: Optional[str] = None     # only for explicit "remember ..." requests
    save_kind: Optional[str] = None     # fact | preference
    modality: str = "text"
    domain: str = "general"             # code | writing | general (rule-based guess, drives scoring only)

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


# ---------------------------------------------------------------- planning / evaluation / selection
@dataclass
class Proposal:
    proposal_id: str
    kind: str                           # answer | answer_from_memory | clarify | acknowledge_save | notice
    purpose: str
    required_inputs: List[str]
    expected_result: str
    content_spec: Dict[str, Any]
    supported_by: List[str] = field(default_factory=list)       # memory ids
    honors_constraints: List[str] = field(default_factory=list)  # constraint ids (a claim, checked later)
    model_confidence: float = 0.0       # UNCALIBRATED self-report; only used as a last tie-breaker
    origin: str = "rules"               # rules | model

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass
class Evaluation:
    proposal_id: str
    kind: str
    scores: Dict[str, float]            # relevance, constraint_consistency, support, safety (+ risk)
    total: float
    flags: List[str]
    blocked: bool
    summary: str
    model_review: Optional[Dict[str, Any]] = None
    independent_verification: bool = False   # always False: the same model is never an independent verifier

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass
class SelectedAction:
    proposal_id: Optional[str]
    kind: str
    content_spec: Dict[str, Any]
    rule: str
    reason: str
    ranking: List[Dict[str, Any]]
    model_confidence: float = 0.0
    supported_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass
class MemoryRecord:
    id: str
    kind: str          # fact | preference | summary
    text: str
    origin: str        # user_stated | system_summary | model_interpretation | seed
    created_at: str
    source_session: Optional[str] = None
    source_cycle: Optional[str] = None

    TRUST = {
        "user_stated": "user-provided",
        "system_summary": "system-generated summary (derived from the conversation)",
        "model_interpretation": "model-generated interpretation (unverified, not a fact)",
        "seed": "built-in guidance shipped with the app (read-only, not about you)",
    }

    def to_dict(self) -> Dict[str, Any]:
        d = jsonable(self)
        d["trust"] = self.TRUST.get(self.origin, "unknown")
        return d
