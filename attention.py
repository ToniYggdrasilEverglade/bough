"""Attention: decides what reaches each component. Pure Python, no model call.

It filters retrieved memories (relevance threshold, duplicates, budget), keeps at most a couple of
relevant items from earlier cycles, and builds the trimmed per-component views. Components never see
the raw workspace."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import Settings
from ..memory.text import token_set
from ..schemas import Observation, clip, jsonable
from ..workspace import WorkspaceItem

MEM_CHARS = 300
CONTEXT_BUDGET = 2
LANGUAGE_MEMORY_BUDGET = 3


@dataclass
class AttentionResult:
    memories: List[Dict[str, Any]] = field(default_factory=list)
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    context: List[Dict[str, Any]] = field(default_factory=list)
    budgets: Dict[str, int] = field(default_factory=dict)


class Attention:
    def __init__(self, settings: Settings):
        self.settings = settings

    def attend(self, obs: Observation, retrieved: List[Dict[str, Any]],
               prior_items: List[WorkspaceItem]) -> AttentionResult:
        s = self.settings
        res = AttentionResult(budgets={"memories": s.max_retrieved_memories, "context": CONTEXT_BUDGET,
                                       "proposals": s.max_proposals})
        seen_text = set()
        for m in sorted(retrieved, key=lambda m: -m.get("score", 0)):
            norm = " ".join(m["text"].lower().split())
            if obs.intent == "save_memory":
                res.dropped.append({"id": m["id"], "reason": "not_needed_for_save_request"})
            elif m.get("score", 0) < s.memory_min_score:
                res.dropped.append({"id": m["id"], "reason": "below_relevance_threshold"})
            elif norm in seen_text:
                res.dropped.append({"id": m["id"], "reason": "duplicate"})
            elif len(res.memories) >= s.max_retrieved_memories:
                res.dropped.append({"id": m["id"], "reason": "over_memory_budget"})
            else:
                seen_text.add(norm)
                res.memories.append(m)
        # Context from earlier cycles: previous goals / selected actions that overlap the new request.
        req = token_set(obs.request)
        cands = []
        for it in prior_items:
            if it.kind not in ("goal", "selected_action"):
                continue
            text = it.content.get("goal") or it.content.get("kind") or ""
            cands.append((len(req & token_set(str(text))), it.timestamp, it, str(text)))
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        for overlap, _, it, text in cands[:CONTEXT_BUDGET]:
            if overlap > 0:
                res.context.append({"item_id": it.item_id, "kind": it.kind, "text": clip(text, MEM_CHARS)})
        return res

    def view(self, component: str, obs: Observation, att: AttentionResult, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "request": obs.request, "intent": obs.intent,
            "constraints": jsonable(obs.constraints)}
        mems = [{"id": m["id"], "text": clip(m["text"], MEM_CHARS), "kind": m["kind"], "origin": m["origin"],
                 "trust": m.get("trust"), "score": m.get("score")} for m in att.memories]
        if component == "planning":
            base.update({"entities": jsonable(obs.entities), "underspecified": obs.underspecified,
                         "missing": obs.missing, "memories": mems, "context": att.context,
                         "limits": {"max_proposals": self.settings.max_proposals}})
        elif component == "evaluation":
            base.update({"memories": mems[:self.settings.max_retrieved_memories]})
        elif component == "language":
            base.update({"memories": mems[:LANGUAGE_MEMORY_BUDGET]})
        if extra:
            base.update(extra)
        return base