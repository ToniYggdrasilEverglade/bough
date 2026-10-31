"""Feedback / adaptation.

What this is: a bounded, documented RULE adjustment. A "useful" rating adds +bias_step to the selection
bias of the kind of action that was chosen; "not useful" subtracts it; the bias is clamped to
[-bias_limit, +bias_limit] and added to evaluation totals during selection.

What this is NOT: no neural network is trained, no weights of any model change, and no memories are
edited. The only things written are a feedback log row and one bias number per action kind.
"""
from __future__ import annotations

from ..config import Settings

ADAPTATION_DOC = (
    "Rating a reply adjusts one number: the selection bias for the kind of action that produced it "
    "(+/- step, clamped to +/- limit). This is a hand-written rule, not model training. "
    "Memories and model weights are never changed by feedback."
)


class Adaptation:
    def __init__(self, settings: Settings):
        self.settings = settings

    def delta(self, rating: str) -> float:
        return self.settings.bias_step if rating == "useful" else -self.settings.bias_step

    def clamp(self, v: float) -> float:
        lim = self.settings.bias_limit
        return max(-lim, min(lim, v))
