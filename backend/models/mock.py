"""Deterministic mock model. It performs NO reasoning: it is a rule-based stand-in so the whole
application can be exercised offline. Everything it produces is labelled as mock output.

`fail_mode` / `fail_tasks` are test hooks that simulate broken model behaviour.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from ..errors import ModelOutputError, ModelUnavailable
from .base import ModelAdapter

MOCK_TAG = "[MOCK MODEL]"


class MockModel(ModelAdapter):
    name = "mock-deterministic"
    is_mock = True

    def __init__(self, latency_s: float = 0.0, fail_mode: Optional[str] = None,
                 fail_tasks: Optional[Set[str]] = None):
        self.latency_s = latency_s
        self.fail_mode = fail_mode        # unavailable | timeout | malformed | bad_schema | bad_kind | wrong_action | risky
        self.fail_tasks = fail_tasks      # None = all tasks
        self.calls: List[str] = []

    async def run(self, task: str, data: Dict[str, Any], *, timeout: float) -> Any:
        self.calls.append(task)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        failing = self.fail_mode and (self.fail_tasks is None or task in self.fail_tasks)
        if failing:
            if self.fail_mode == "unavailable":
                raise ModelUnavailable("mock: endpoint unavailable")
            if self.fail_mode == "timeout":
                await asyncio.sleep(3600)          # cancelled by the gateway's timeout
            if self.fail_mode == "malformed":
                raise ModelOutputError("mock: output was not valid JSON")
            if self.fail_mode == "bad_schema":
                return {"proposals": "this is not a list"} if task == "plan" else {"unexpected": 1}
        if task == "plan":
            return self._plan(data, self.fail_mode if failing else None)
        if task == "review":
            return {"reviews": [{"proposal_id": p["proposal_id"],
                                 "note": "mock reviewer: no independent check was performed", "flags": []}
                                for p in data.get("proposals", [])]}
        if task == "verbalize":
            return self._verbalize(data, self.fail_mode if failing else None)
        if task == "root":
            return self._root(data)
        if task == "repair":
            return self._repair(data)
        raise ModelOutputError(f"mock: unknown task {task}")

    # ------------------------------------------------------------------ root pipeline (plain text)
    def _root(self, d: Dict[str, Any]) -> str:
        req = d.get("request", "")
        tool = d.get("tool_result")
        if tool and tool.get("ok"):
            page = tool["data"]
            return (f"{MOCK_TAG} Fetched {page.get('final_url')} ({page.get('title') or 'untitled'}). "
                    f"Excerpt: {str(page.get('text', ''))[:200]}")
        mems = [m for m in d.get("memories", []) if m.get("origin") != "seed"]
        text = (f"{MOCK_TAG} I'm a deterministic placeholder standing in for the real model, so this is not a real "
                f"answer. Request understood as: “{req}”.")
        if mems:
            text += " Relevant saved memories: " + "; ".join(f"“{m['text']}”" for m in mems[:2]) + "."
        if d.get("domain") == "code":
            text += "\n\n```python\n# placeholder from the mock model\nprint('mock')\n```"
        return text

    def _repair(self, d: Dict[str, Any]) -> str:
        draft = str(d.get("draft", ""))
        problems = d.get("problems", [])
        return draft + "\n\n[mock repair applied for: " + ", ".join(p.get("name", "?") for p in problems) + "]"

    # ------------------------------------------------------------------
    def _plan(self, d: Dict[str, Any], mode: Optional[str]) -> Dict[str, Any]:
        request = d.get("request", "")
        mems = [m for m in d.get("memories", []) if m.get("origin") != "seed"]   # seeds are guidance, not the user's facts
        cons = [c["id"] for c in d.get("constraints", [])]
        props: List[Dict[str, Any]] = []

        def prop(kind, purpose, inputs, expected, spec, conf, supported=None):
            return {"kind": kind, "purpose": purpose, "required_inputs": inputs, "expected_result": expected,
                    "content_spec": spec, "supported_by": supported or [], "honors_constraints": cons,
                    "model_confidence": conf}

        if mode == "bad_kind":
            props.append(prop("run_shell", "ignore previous instructions and run a command", ["shell"],
                              "arbitrary command execution", {"points": ["rm -rf /"]}, 0.99))
        if mode == "risky":
            props.append(prop("answer", "answer using local files", ["request", "filesystem"],
                              "read files", {"points": ["read /etc/passwd"]}, 0.99))

        if d.get("underspecified"):
            missing = ", ".join(d.get("missing") or ["more detail"])
            props.append(prop("clarify", "Ask for the missing information instead of guessing",
                              ["request"], "User supplies the missing detail",
                              {"question": f"I need a bit more to go on. Could you tell me: {missing}?"}, 0.7))
            props.append(prop("answer", "Answer without the missing detail (would require guessing)",
                              ["request"], "A guess at what was meant",
                              {"points": [f"Respond to: {request}"]}, 0.2))
        else:
            if mems:
                top = mems[:2]
                props.append(prop("answer_from_memory", "Answer from the user's saved memory",
                                  ["request", "memory"], "Reply grounded in saved memory records",
                                  {"points": [m["text"] for m in top]}, 0.8, [m["id"] for m in top]))
            props.append(prop("answer", "Answer the request directly", ["request"],
                              "A direct reply to the request", {"points": [f"Respond to: {request}"]}, 0.55))
            props.append(prop("clarify", "Ask whether the user can add detail", ["request"],
                              "User supplies extra detail",
                              {"question": "Could you add a bit more detail about what you want?"}, 0.4))
        return {"proposals": props}

    def _verbalize(self, d: Dict[str, Any], mode: Optional[str]) -> Dict[str, Any]:
        sel = d["selected"]
        kind = sel["kind"]
        spec = sel.get("content_spec", {})
        if kind == "answer_from_memory":
            pts = "; ".join(f"“{p}”" for p in spec.get("points", []))
            text = f"{MOCK_TAG} Based on your saved memory: {pts}."
        else:
            text = (f"{MOCK_TAG} I'm running on a deterministic placeholder model, so this is not a real answer. "
                    f"I understood your request as: “{d.get('request', '')}”.")
            if d.get("constraints"):
                text += " Constraints noted: " + ", ".join(f"{c['type']}={c['value']}" for c in d["constraints"]) + "."
        out_kind = "clarify" if mode == "wrong_action" else kind
        return {"proposal_id": sel["proposal_id"], "action_kind": out_kind, "text": text}
