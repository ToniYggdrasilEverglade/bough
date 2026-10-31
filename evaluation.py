"""Evaluation: scores every proposal against explicit, documented criteria.

Scores come from ordinary Python checks. An OPTIONAL same-model review (disabled by default) can add
notes and may lower the support score by a bounded amount. That review is a self-check by the same
model, so results are always marked independent_verification=False. Model-reported confidence is shown
but is not part of the score."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Set

from ..config import Settings
from ..context import CycleContext
from ..errors import ModelFailure
from ..memory.text import token_set
from ..models.gateway import ModelGateway
from ..models.validation import validate_review
from ..schemas import Evaluation, Observation, Proposal

CRITERIA = {
    "relevance": "How well the proposal fits the request (token overlap + whether the kind suits the situation).",
    "constraint_consistency": "Share of the user's explicit constraints the proposal declares it honours "
                              "(a claim; the language step re-checks word/sentence/term limits).",
    "support": "Is it grounded? Memory answers: fraction of cited memory ids that were really retrieved "
               "(unknown ids = 0 and blocked). Plain answers rely on unverified model knowledge, so score low.",
    "safety": "1 - execution risk. Risk is rule-based by action kind. Free-text required_inputs "
              "describe information needs, not execution permissions; unsupported action kinds are blocked.",
}
BASE_RISK = {"web_fetch": 0.3, "answer": 0.2, "answer_from_memory": 0.15, "clarify": 0.05, "acknowledge_save": 0.15, "notice": 0.0}
REVIEW_SUPPORT_PENALTY = 0.2


def _r(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 3)


class Evaluator:
    def __init__(self, settings: Settings, gateway: ModelGateway):
        self.settings, self.gateway = settings, gateway

    async def evaluate(self, ctx: CycleContext, obs: Observation, proposals: List[Proposal],
                       retrieved_ids: Set[str], view: Dict[str, Any]) -> List[Evaluation]:
        review_task = None
        model_props = [p for p in proposals if p.origin == "model"]
        if self.settings.enable_model_review and model_props:
            data = dict(view, proposals=[{"proposal_id": p.proposal_id, "kind": p.kind, "purpose": p.purpose,
                                          "content_spec": p.content_spec, "supported_by": p.supported_by}
                                         for p in model_props])
            review_task = asyncio.create_task(self._review(ctx, data))
        # Independent of the review call, so both run concurrently.
        evals = [self._score(p, obs, retrieved_ids) for p in proposals]
        reviews = await review_task if review_task else {}
        for ev in evals:
            rv = reviews.get(ev.proposal_id)
            if rv is None:
                continue
            ev.model_review = {**rv, "same_model_review": True, "independent_verification": False}
            if "unsupported_claim" in rv["flags"]:
                ev.scores["support"] = _r(ev.scores["support"] - REVIEW_SUPPORT_PENALTY)
                ev.flags.append("reviewer_flagged_unsupported_claim")
                self._total(ev)
        return evals

    async def _review(self, ctx: CycleContext, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await self.gateway.call(ctx, "review", data, validate_review)
        except ModelFailure:
            return {}

    # ------------------------------------------------------------------
    def _score(self, p: Proposal, obs: Observation, retrieved_ids: Set[str]) -> Evaluation:
        flags: List[str] = []
        spec = p.content_spec
        text = " ".join([p.purpose, *spec.get("points", []), spec.get("question", "")])
        req = token_set(obs.request)
        overlap = min(1.0, 2 * len(req & token_set(text)) / max(1, len(req)))
        mock = self.gateway.adapter.is_mock

        # relevance
        if p.kind == "web_fetch":
            rel = 1.0
        elif p.kind == "acknowledge_save":
            rel = 1.0 if obs.intent == "save_memory" else 0.0
        elif p.kind == "notice":
            rel = 1.0 if obs.intent == "save_memory" else 0.5
        elif p.kind == "clarify":
            rel = 1.0 if obs.underspecified else 0.45
        elif obs.underspecified:
            rel = 0.3
        else:
            rel = 0.6 + 0.4 * overlap

        # support
        if p.kind == "web_fetch":
            sup = 0.5  # proposal only: no page has been fetched yet
        elif p.kind in ("acknowledge_save", "notice"):
            sup = 1.0
        elif p.kind == "clarify":
            sup = 0.8 if obs.underspecified else 0.5
        elif p.kind == "answer_from_memory":
            if not p.supported_by:
                sup = 0.0
                flags.append("memory_answer_without_citations")
            else:
                valid = [m for m in p.supported_by if m in retrieved_ids]
                sup = len(valid) / len(p.supported_by)
                if len(valid) < len(p.supported_by):
                    flags.append("unsupported_memory_citation")
        else:  # plain answer
            sup = 0.25 if mock else (0.5 if retrieved_ids else 0.35)
            flags.append("mock_model_no_real_reasoning" if mock else "relies_on_unverified_model_knowledge")

        # constraints
        con = 1.0
        if obs.constraints:
            if p.origin == "rules":
                con = 1.0
            else:
                ids = {c.id for c in obs.constraints}
                con = len(ids & set(p.honors_constraints)) / len(ids)
                if con < 1.0:
                    flags.append("constraint_not_declared_honored")
            for c in obs.constraints:
                if c.type == "forbid_term" and re.search(rf"\b{re.escape(str(c.value))}\b", text, re.I):
                    con = 0.0
                    flags.append("plan_contains_forbidden_term")

        # risk / permissions
        risk = BASE_RISK.get(p.kind, 1.0)
        blocked = False
        if p.kind == "web_fetch" and p.origin != "rules":
            risk, blocked = 1.0, True
            flags.append("web_fetch_requires_explicit_user_command")
        # required_inputs is descriptive text, not an execution-permission field.
        # Mentioning HTTP/files/etc. does not execute those operations.
        # Executable action kinds remain unsupported (the unknown_kind check below).
        if p.kind == "answer_from_memory" and (not p.supported_by or sup == 0):
            blocked = True
        if p.kind not in BASE_RISK:
            risk, blocked = 1.0, True
            flags.append("unknown_kind")

        ev = Evaluation(p.proposal_id, p.kind,
                        {"relevance": _r(rel), "constraint_consistency": _r(con), "support": _r(sup),
                         "safety": _r(1 - risk), "risk": _r(risk)},
                        0.0, flags, blocked, "")
        self._total(ev)
        return ev

    def _total(self, ev: Evaluation) -> None:
        s, sc = self.settings, ev.scores
        ev.total = round(s.w_relevance * sc["relevance"] + s.w_constraints * sc["constraint_consistency"]
                         + s.w_support * sc["support"] + s.w_safety * sc["safety"], 3)
        ev.summary = (f"{ev.kind}: relevance {sc['relevance']:.2f}, constraints {sc['constraint_consistency']:.2f}, "
                      f"support {sc['support']:.2f}, risk {sc['risk']:.2f} → total {ev.total:.2f}"
                      + (f"; BLOCKED" if ev.blocked else "")
                      + (f"; flags: {', '.join(ev.flags)}" if ev.flags else ""))