"""Planning: proposes a small set of candidate actions.

Deterministic cases (explicit save requests, missing information) are handled by rules and cost no
model call. Open-ended requests use ONE model call, validated against a strict schema. If the model
fails, the planner falls back to rule-based proposals and says so."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import Settings
from ..context import CycleContext
from ..errors import ModelFailure
from ..models.gateway import ModelGateway
from ..models.validation import validate_plan
from ..schemas import Observation, Proposal


@dataclass
class PlanOutcome:
    proposals: List[Proposal]
    source: str                                  # rules | model | fallback
    notes: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    model_error: str | None = None


def _rule(kind: str, purpose: str, inputs: List[str], expected: str, spec: Dict[str, Any],
          obs: Observation, supported_by: List[str] | None = None) -> Proposal:
    return Proposal("", kind, purpose, inputs, expected, spec, supported_by or [],
                    [c.id for c in obs.constraints], 1.0, "rules")


class Planner:
    def __init__(self, settings: Settings, gateway: ModelGateway):
        self.settings, self.gateway = settings, gateway

    async def plan(self, ctx: CycleContext, obs: Observation, view: Dict[str, Any],
                   memory_enabled: bool) -> PlanOutcome:
        if obs.request == "/fetch" or obs.request.startswith("/fetch "):
            parts = obs.request.split(maxsplit=2)
            if len(parts) < 2:
                props = [_rule("clarify", "Ask for the URL", [], "User supplies URL",
                               {"question":"Use /fetch https://example.com/ followed by an optional question."}, obs)]
            else:
                question = parts[2] if len(parts) > 2 else "Summarize the fetched page concisely."
                props = [_rule("web_fetch", "Fetch the explicitly requested public page", ["url"],
                               "Return a source-grounded reply or a truthful fetch error",
                               {"url":parts[1], "points":[question]}, obs)]
            return self._finish(props, "rules", ["explicit /fetch command; permission checked at execution"], [])
        if obs.intent == "save_memory":
            if obs.underspecified:
                props = [_rule("clarify", "Ask what should be remembered", ["request"], "User states what to save",
                               {"question": "What would you like me to remember?"}, obs)]
            elif memory_enabled:
                props = [_rule("acknowledge_save", "Save the user's explicit statement to long-term memory",
                               ["save_text", "memory_enabled"], "A user_stated memory record is stored",
                               {"template": "acknowledge_save"}, obs)]
            else:
                props = [_rule("notice", "Explain that nothing was saved because memory is off",
                               ["memory_enabled"], "User knows how to enable memory",
                               {"code": "memory_disabled"}, obs)]
            return self._finish(props, "rules", ["no model call needed: explicit save request"], [])

        try:
            plan = await self.gateway.call(ctx, "plan", view, lambda raw: validate_plan(raw, self.settings.max_proposals))
            return self._finish(plan.proposals, "model", [], plan.rejected)
        except ModelFailure as f:
            props: List[Proposal] = []
            if obs.underspecified:
                props.append(_rule("clarify", "Ask for the missing information", ["request"],
                                   "User supplies the missing detail",
                                   {"question": "Could you tell me " + ", ".join(obs.missing) + "?"}, obs))
            else:
                props.append(_rule("notice", "Tell the user the language model could not be used",
                                   [], "User understands the limitation",
                                   {"code": "model_unavailable", "reason": f.code}, obs))
            mems = view.get("memories", [])
            if mems and not obs.underspecified:
                top = mems[:2]
                props.append(_rule("answer_from_memory", "Offer relevant saved memories without model help",
                                   ["request", "memory"], "Saved memories shown verbatim",
                                   {"points": [m["text"] for m in top], "render": "template"}, obs,
                                   [m["id"] for m in top]))
            return self._finish(props, "fallback", [f"model planning failed ({f.code}); using rule-based fallback"],
                                [], model_error=f"{f.code}: {f.detail}" if f.detail else f.code)

    def _finish(self, props: List[Proposal], source: str, notes: List[str], rejected: List[str],
                model_error: str | None = None) -> PlanOutcome:
        props = props[: self.settings.max_proposals]
        for i, p in enumerate(props, 1):
            p.proposal_id = f"p{i}"
        return PlanOutcome(props, source, notes, rejected, model_error)
