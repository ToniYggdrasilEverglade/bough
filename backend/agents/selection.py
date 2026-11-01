"""Action selection: explicit, inspectable rules in plain Python. No model involved.

Rules, applied in order:
  R1 exclude_blocked           proposals flagged as blocked by evaluation are never eligible.
  R2 clarify_when_essential_missing   if perception found essential information missing and an eligible
                               clarify proposal exists, choose it (never guess).
  R3 best_adjusted_score       otherwise pick the highest (evaluation total + bounded adaptation bias) among
                               the eligible NON-clarify proposals (when nothing is missing, a clarifying question
                               is never allowed to out-score an acceptable answer).
                               Ties: higher support, lower risk, higher model confidence (uncalibrated), earlier order.
  R3b clarify_fallback         if no answer reached min_select_score but the model offered an eligible clarify
                               proposal that does, use that specific question.
  R4 no_acceptable_proposal    if nothing is eligible or nothing reaches min_select_score,
                               ask a generic clarifying question instead of answering.
Exactly one action is returned per cycle.
"""
from __future__ import annotations

from typing import Dict, List

from ..config import Settings
from ..schemas import Evaluation, Observation, Proposal, SelectedAction

RULES_TEXT = [
    "R1 exclude_blocked: proposals blocked by evaluation are not eligible.",
    "R2 clarify_when_essential_missing: if essential info is missing and a clarify proposal is eligible, choose it.",
    "R3 best_adjusted_score: highest evaluation total + bounded bias among non-clarify proposals; "
    "ties → support, safety, model confidence, order. Clarify cannot beat an acceptable answer.",
    "R3b clarify_fallback: no answer reached the threshold but the model's clarify proposal did → ask that question.",
    "R4 no_acceptable_proposal: if nothing is eligible or nothing reaches the threshold, ask a generic clarifying question.",
]


class Selector:
    def __init__(self, settings: Settings):
        self.settings = settings

    def select(self, obs: Observation, proposals: List[Proposal], evaluations: List[Evaluation],
               bias: Dict[str, float]) -> SelectedAction:
        by_id = {p.proposal_id: p for p in proposals}
        ranking = []
        for order, ev in enumerate(evaluations):
            b = bias.get(ev.kind, 0.0)
            ranking.append({"proposal_id": ev.proposal_id, "kind": ev.kind, "eval_total": ev.total,
                            "bias": round(b, 3), "adjusted": round(ev.total + b, 3), "blocked": ev.blocked,
                            "_order": order})
        eligible = [r for r in ranking if not r["blocked"]]

        def done(r, rule, reason):
            p = by_id[r["proposal_id"]]
            return SelectedAction(p.proposal_id, p.kind, p.content_spec, rule, reason, _clean(ranking),
                                  p.model_confidence, p.supported_by)

        if obs.underspecified:
            clar = [r for r in eligible if r["kind"] == "clarify"]
            if clar:
                return done(max(clar, key=lambda r: r["adjusted"]), "R2 clarify_when_essential_missing",
                            "Essential information is missing: " + "; ".join(obs.missing))
        def key(r):
            ev = next(e for e in evaluations if e.proposal_id == r["proposal_id"])
            p = by_id[r["proposal_id"]]
            return (r["adjusted"], ev.scores["support"], ev.scores["safety"], p.model_confidence, -r["_order"])

        answers = [r for r in eligible if r["kind"] != "clarify"]
        if answers:
            best = max(answers, key=key)
            if best["adjusted"] >= self.settings.min_select_score:
                return done(best, "R3 best_adjusted_score",
                            f"Highest adjusted score {best['adjusted']:.3f} (eval {best['eval_total']:.3f} "
                            f"+ bias {best['bias']:+.3f}) among {len(answers)} eligible non-clarify proposal(s)")
        clar = [r for r in eligible if r["kind"] == "clarify"]
        if clar:
            best_c = max(clar, key=key)
            if best_c["adjusted"] >= self.settings.min_select_score:
                return done(best_c, "R3b clarify_fallback",
                            "No answer proposal reached the minimum score; using the model's clarifying question "
                            f"(adjusted {best_c['adjusted']:.3f})")
        return SelectedAction(
            None, "clarify",
            {"question": "I couldn't produce a sufficiently supported answer. Could you rephrase or add detail?"},
            "R4 no_acceptable_proposal",
            "No eligible proposal reached the minimum score" if eligible else "Every proposal was blocked",
            _clean(ranking))


def _clean(ranking):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in ranking]
