"""Regression tests for the clarify fix and built-in seeds. Run: pytest -q (from the repo root).

A fake "chatty" planner always proposes a clarify next to a real answer. Before the fix, the clarify
out-scored the answer on ordinary coding requests (0.67 vs 0.66 with review off); the pipeline must now
answer, keep clarifying when something is genuinely missing, and surface seeds as read-only guidance."""
from __future__ import annotations

import asyncio
import os
import tempfile

from backend.agents.perception import detect_domain
from backend.config import Settings
from backend.memory.base import LOCAL
from backend.memory.seed import SeedStore
from backend.memory.store import MemoryStore
from backend.models.base import ModelAdapter
from backend.orchestrator import Orchestrator

CODE = "def reverse(items):\n    return items[::-1]\n"


class ChattyModel(ModelAdapter):
    is_mock = False
    name = "fake:chatty"

    def __init__(self):
        self.calls = []

    async def run(self, task, data, *, timeout):
        self.calls.append((task, data))
        if task == "plan":
            seeds = [m["id"] for m in data.get("memories", []) if m["origin"] == "seed"]
            return {"proposals": [
                {"kind": "clarify", "purpose": "Ask which Python version", "required_inputs": ["request"],
                 "expected_result": "User says version", "content_spec": {"question": "Which Python version?"},
                 "supported_by": [], "honors_constraints": [], "model_confidence": 0.9},
                {"kind": "answer", "purpose": "Give slicing implementation", "required_inputs": ["request"],
                 "expected_result": "Working function", "content_spec": {"points": ["Use slicing items[::-1]"]},
                 "supported_by": seeds[:2], "honors_constraints": [], "model_confidence": 0.6},
            ]}
        if task == "review":
            return {"reviews": [{"proposal_id": p["proposal_id"], "note": "", "flags": ["unsupported_claim"]}
                                for p in data["proposals"]]}
        sel = data["selected"]
        text = ("```python\n" + CODE + "```") if sel["kind"] == "answer" else sel["content_spec"]["question"]
        return {"proposal_id": sel["proposal_id"], "action_kind": sel["kind"], "text": text}


def run(text, memory_enabled=True, **settings_kw):
    d = tempfile.mkdtemp()
    settings_kw.setdefault("pipeline", "proposals")   # these tests cover the council flow explicitly
    s = Settings(db_path=os.path.join(d, "t.sqlite3"), model_backend="http", model_url="http://x/", **settings_kw)
    model = ChattyModel()
    orch = Orchestrator(s, MemoryStore(s.db_path, 100), model)
    sid = orch.bind(LOCAL, "s1")
    orch.sessions.get(sid).memory_enabled = memory_enabled
    return asyncio.run(orch.process(sid, text)), model, orch


def test_coding_request_is_answered_not_clarified_even_with_review():
    r, model, _ = run("write a python function that reverses a list", enable_model_review=True)
    assert r["final"]["action_kind"] == "answer"
    assert r["selected"]["rule"].startswith("R3 ")
    assert "items[::-1]" in r["final"]["text"]
    ev = {e["kind"]: e for e in r["evaluations"]}
    assert ev["clarify"]["scores"]["relevance"] == 0.2
    assert "clarify_without_missing_information" in ev["clarify"]["flags"]
    assert "reviewer_flagged_unsupported_claim_ignored_for_actionable_request" in ev["answer"]["flags"]
    assert ev["answer"]["total"] > ev["clarify"]["total"]


def test_underspecified_request_still_clarifies():
    r, _, _ = run("fix it")
    assert r["final"]["action_kind"] == "clarify"
    assert r["selected"]["rule"].startswith("R2 ")


def test_seeds_reach_planner_and_language_with_domain():
    r, model, _ = run("write a python function that reverses a list")
    plan = next(d for t, d in model.calls if t == "plan")
    assert plan["domain"] == "code"
    ids = [m["id"] for m in plan["memories"] if m["origin"] == "seed"]
    assert {"seed:style-001", "seed:style-002", "seed:code-001"} <= set(ids)
    lang = next(d for t, d in model.calls if t == "verbalize")
    assert any(m["id"] == "seed:code-001" for m in lang["memories"])


def test_seeds_consulted_when_personal_memory_is_off():
    r, model, _ = run("write a rust cli that fetches a url", memory_enabled=False)
    plan = next(d for t, d in model.calls if t == "plan")
    assert any(m["id"] == "seed:code-001" for m in plan["memories"])
    assert r["final"]["action_kind"] == "answer"


def test_seeds_can_be_disabled():
    r, model, orch = run("write a python function that reverses a list", seed_enabled=False)
    plan = next(d for t, d in model.calls if t == "plan")
    assert plan["memories"] == []
    assert orch.describe()["seeds"] == {"enabled": False, "count": 0, "domains": {}, "pinned": {}, "problems": []}


def test_seed_files_are_valid_and_unique():
    st = SeedStore()
    assert st.problems == []
    assert len(st) >= 100
    assert set(st.describe()["pinned"]) == {"always", "code", "writing"}


def test_seed_search_pins_and_keywords():
    st = SeedStore()
    hits = st.search("how do I set up a systemd service", 4, 0.15, detect_domain("how do I set up a systemd service"))
    ids = [h["id"] for h in hits]
    assert ids[:2] == ["seed:style-001", "seed:style-002"]
    assert "seed:sh-005" in ids
    assert all(h.get("pinned") for h in st.search("", 4))          # no keywords: only the always-on pins


def test_domain_detection():
    assert detect_domain("my code throws TypeError") == "code"
    assert detect_domain("rewrite this email to sound formal") == "writing"
    assert detect_domain("what's the capital of france") == "general"


def test_seed_ids_are_not_user_memories():
    _, _, orch = run("hello")
    assert orch.store.delete(LOCAL, "seed:code-001") is False
    assert orch.store.get(LOCAL, "seed:code-001") is None
