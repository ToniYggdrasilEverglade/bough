"""Strict validation of model output. Anything that does not match raises ModelOutputError.

Proposal kinds a model may emit are an allowlist; anything else (e.g. "run_shell") is dropped and
reported, so model output can never grant itself new capabilities.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from ..errors import ModelOutputError
from ..schemas import Proposal

MODEL_PROPOSAL_KINDS = ("answer", "answer_from_memory", "clarify")
REVIEW_FLAGS = ("unsupported_claim", "constraint_risk", "off_topic", "unverified_knowledge")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _text(v: Any, field: str, max_len: int, allow_empty: bool = False) -> str:
    if not isinstance(v, str):
        raise ModelOutputError(f"{field} must be a string")
    v = _CTRL.sub(" ", v).strip()
    if not v and not allow_empty:
        raise ModelOutputError(f"{field} must not be empty")
    return v[:max_len]


def _str_list(v: Any, field: str, max_items: int, max_len: int) -> List[str]:
    if not isinstance(v, list):
        raise ModelOutputError(f"{field} must be a list")
    return [_text(x, field, max_len) for x in v[:max_items]]


@dataclass
class PlanResult:
    proposals: List[Proposal]
    rejected: List[str]


def validate_plan(raw: Any, max_proposals: int) -> PlanResult:
    if not isinstance(raw, dict) or not isinstance(raw.get("proposals"), list):
        raise ModelOutputError("plan must be an object with a 'proposals' list")
    good: List[Proposal] = []
    rejected: List[str] = []
    for i, item in enumerate(raw["proposals"]):
        try:
            if not isinstance(item, dict):
                raise ModelOutputError("proposal must be an object")
            kind = item.get("kind")
            if kind not in MODEL_PROPOSAL_KINDS:
                raise ModelOutputError(f"kind {str(kind)[:30]!r} is not an allowed proposal kind")
            conf = item.get("model_confidence")
            if isinstance(conf, bool) or not isinstance(conf, (int, float)):
                raise ModelOutputError("model_confidence must be a number")
            spec_in = item.get("content_spec")
            if not isinstance(spec_in, dict):
                raise ModelOutputError("content_spec must be an object")
            spec: Dict[str, Any] = {}
            if "points" in spec_in:
                spec["points"] = _str_list(spec_in["points"], "content_spec.points", 6, 300)
            if "question" in spec_in:
                spec["question"] = _text(spec_in["question"], "content_spec.question", 300)
            if kind == "clarify" and "question" not in spec:
                raise ModelOutputError("clarify proposals need content_spec.question")
            if kind != "clarify" and not spec.get("points"):
                raise ModelOutputError("content_spec.points required")
            good.append(Proposal(
                proposal_id="", kind=kind,
                purpose=_text(item.get("purpose"), "purpose", 300),
                required_inputs=_str_list(item.get("required_inputs", []), "required_inputs", 6, 60),
                expected_result=_text(item.get("expected_result"), "expected_result", 300),
                content_spec=spec,
                supported_by=_str_list(item.get("supported_by", []), "supported_by", 6, 64),
                honors_constraints=_str_list(item.get("honors_constraints", []), "honors_constraints", 10, 16),
                model_confidence=max(0.0, min(1.0, float(conf))), origin="model"))
        except ModelOutputError as e:
            rejected.append(f"proposal[{i}]: {e}")
    if not good:
        raise ModelOutputError("no valid proposals in model output: " + "; ".join(rejected[:3]))
    return PlanResult(good[:max_proposals], rejected + ([f"{len(good) - max_proposals} valid proposal(s) dropped: over limit"]
                                                         if len(good) > max_proposals else []))


def validate_review(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("reviews"), list):
        raise ModelOutputError("review must be an object with a 'reviews' list")
    out: Dict[str, Dict[str, Any]] = {}
    for r in raw["reviews"]:
        if not isinstance(r, dict) or not isinstance(r.get("proposal_id"), str):
            continue
        flags = [f for f in r.get("flags", []) if isinstance(f, str) and f in REVIEW_FLAGS] \
            if isinstance(r.get("flags", []), list) else []
        note = r.get("note", "")
        out[r["proposal_id"]] = {"note": _text(note, "note", 300, allow_empty=True) if isinstance(note, str) else "",
                                 "flags": flags}
    return out


def validate_verbalize(raw: Any, max_text: int = 16000) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise ModelOutputError("verbalize output must be an object")
    return {"proposal_id": _text(raw.get("proposal_id"), "proposal_id", 64),
            "action_kind": _text(raw.get("action_kind"), "action_kind", 40),
            "text": _text(raw.get("text"), "text", max_text)}


_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)


def validate_text(raw: Any, max_text: int = 16000) -> str:
    """Free-form reply from the root model. Only control characters are scrubbed; newlines/tabs survive."""
    if not isinstance(raw, str):
        raise ModelOutputError("reply must be text")
    tools = getattr(raw, "executed_tools", None)   # read before any str operation drops the attribute
    # Reasoning models (Qwen3 etc.) may prepend a <think>...</think> block; the user never needs it.
    raw = _THINK.sub("", raw)
    text = _CTRL.sub(" ", raw).strip()
    if not text:
        raise ModelOutputError("reply was empty")
    if tools:   # keep the record of provider-side tools (Groq browser_search) for the grounding branch
        from .http_model import ModelText
        return ModelText(text[:max_text], tools)
    return text[:max_text]
