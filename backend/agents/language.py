"""Language: turns the SELECTED action into text. It cannot change the action.

* clarify / acknowledge_save / notice are rendered from templates (no model call).
* answer / answer_from_memory use the model once; the output must echo the selected proposal_id and
  action_kind or it is rejected and the template is used instead.
* Word / sentence / forbidden-term limits are re-checked in Python after generation (answer/clarify only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..context import CycleContext
from ..errors import ModelFailure
from ..models.gateway import ModelGateway
from ..models.validation import validate_verbalize
from ..schemas import Constraint, Observation, SelectedAction, clip

MODEL_RENDERED = ("answer", "answer_from_memory", "web_fetch")


@dataclass
class LanguageResult:
    text: str
    method: str                       # template | model | template_fallback
    post_checks: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def apply_constraints(text: str, constraints: List[Constraint]) -> tuple[str, List[str]]:
    checks: List[str] = []
    for c in constraints:
        if c.type == "forbid_term":
            pat = re.compile(rf"\b{re.escape(str(c.value))}\b", re.I)
            sents = re.split(r"(?<=[.!?])\s+", text)
            kept = [s for s in sents if not pat.search(s)]
            if len(kept) != len(sents):
                checks.append(f"removed_sentences_containing_forbidden_term:{c.value}")
                text = " ".join(kept) or "I can't respond to that within the constraints you stated."
    for c in constraints:
        if c.type == "max_sentences" and int(c.value) > 0:
            sents = re.split(r"(?<=[.!?])\s+", text)
            if len(sents) > int(c.value):
                text = " ".join(sents[: int(c.value)])
                checks.append(f"truncated_to_max_sentences:{c.value}")
    for c in constraints:
        if c.type == "max_words" and int(c.value) > 0:
            words = text.split()
            if len(words) > int(c.value):
                text = " ".join(words[: int(c.value)])
                checks.append(f"truncated_to_max_words:{c.value}")
    return text, checks


class LanguageAgent:
    def __init__(self, settings: Settings, gateway: ModelGateway):
        self.settings, self.gateway = settings, gateway

    async def render(self, ctx: CycleContext, obs: Observation, sel: SelectedAction, view: Dict[str, Any],
                     memory_write: Dict[str, Any]) -> LanguageResult:
        spec = sel.content_spec
        tool_result = view.get("tool_result") or {}
        use_model = sel.kind in MODEL_RENDERED and spec.get("render") != "template"
        if sel.kind == "web_fetch" and not tool_result.get("ok"):
            use_model = False
        res: Optional[LanguageResult] = None
        notes: List[str] = []
        if use_model:
            data = dict(view, selected={"proposal_id": sel.proposal_id, "kind": sel.kind, "content_spec": spec})
            try:
                out = await self.gateway.call(ctx, "verbalize", data,
                                              lambda raw: validate_verbalize(raw, self.settings.max_reply_chars))
                if out["action_kind"] != sel.kind or out["proposal_id"] != sel.proposal_id:
                    notes.append("model_changed_action_rejected")
                else:
                    res = LanguageResult(out["text"], "model")
            except ModelFailure as f:
                notes.append(f"model_unavailable_for_language:{f.code}")
        if res is None:
            res = LanguageResult(self._web_template(tool_result) if sel.kind == "web_fetch" else self._template(sel, obs, memory_write),
                                 "template" if not use_model else "template_fallback", notes=notes)
        else:
            res.notes = notes
        if sel.kind in ("answer", "answer_from_memory", "clarify", "web_fetch"):
            res.text, res.post_checks = apply_constraints(res.text, obs.constraints)
        res.text = clip(res.text, self.settings.max_reply_chars)
        return res

    @staticmethod
    def _web_template(result):
        if not result.get("ok"):
            error = result.get("error") or {}
            return "Web fetch failed (" + str(error.get("code", "UNKNOWN")) + "): " + str(error.get("message", "No page was retrieved."))
        page = result["data"]
        note = "\n[Source text was shortened.]" if page.get("truncated") or page.get("context_truncated") else ""
        return ("Fetched: " + page["final_url"] + "\nTitle: " + (page.get("title") or "Untitled") +
                "\n\nPage excerpt (not a model summary):\n" + page.get("text", "")[:2400] + note)

    def _template(self, sel: SelectedAction, obs: Observation, mw: Dict[str, Any]) -> str:
        spec, kind = sel.content_spec, sel.kind
        if kind == "clarify":
            return spec.get("question", "Could you tell me more?")
        if kind == "acknowledge_save":
            st = mw.get("status")
            rec = mw.get("record") or {}
            if st == "saved":
                return f"Saved to long-term memory as a {rec.get('kind')}: “{rec.get('text')}”. You can delete it from the memory panel."
            if st == "duplicate":
                return f"That is already in long-term memory: “{rec.get('text')}”. Nothing new was saved."
            if st == "failed":
                return f"I couldn't save that: {mw.get('reason', 'unknown error')}."
            return f"Nothing was saved ({mw.get('reason', 'memory is off')})."
        if kind == "notice":
            if spec.get("code") == "memory_disabled":
                return ("Memory is turned off for this session, so nothing was saved. "
                        "Turn on “Persistent memory” in the header if you want me to keep this.")
            if spec.get("code") == "model_unavailable":
                why = {"unavailable": "the model endpoint returned an error or could not be reached",
                       "timeout": "the model took too long to answer",
                       "malformed_output": "the model's reply could not be used",
                       "budget_exceeded": "the model-call budget for this message ran out"}.get(
                           spec.get("reason"), spec.get("reason") or "error")
                detail = f" Details: {spec['detail']}." if spec.get("detail") else ""
                return (f"The language model could not be used for this request ({why}), so I can only give a "
                        f"limited, rule-based reply and I can't answer your question yet.{detail}")
            return spec.get("text", "Notice.")
        if kind == "answer_from_memory":
            pts = "; ".join(f"“{p}”" for p in spec.get("points", []))
            return f"From your saved memories: {pts}."
        return (f"I couldn't produce a model-written answer to: “{clip(obs.request, 200)}”. "
                "Please try again, or check the model settings.")
