"""Root pipeline (Yggdrasil): the model answers freely once, rule-based branches filter the draft,
the decision step publishes / fixes / repairs once / falls back. A scripted fake model lets each
branch be exercised deterministically."""
import asyncio, os, tempfile
import pytest
from backend.config import Settings
from backend.models.base import ModelAdapter
from backend.errors import ModelUnavailable
from backend.memory.store import MemoryStore
from backend.memory.base import LOCAL
from backend.orchestrator import Orchestrator
from backend.agents.branches import Branches
from backend.schemas import Observation, Constraint

GOOD_CODE = "Use slicing:\n\n```python\ndef reverse(items):\n    return items[::-1]\n```\n"


class ScriptedModel(ModelAdapter):
    is_mock = False
    name = "fake:scripted"

    def __init__(self, root, repair=None, fail_root=False):
        self.root, self.repair, self.fail_root, self.calls = root, repair, fail_root, []

    async def run(self, task, data, *, timeout):
        self.calls.append((task, data))
        if task == "root":
            if self.fail_root:
                raise ModelUnavailable("down")
            return self.root(data) if callable(self.root) else self.root
        if task == "repair":
            return self.repair(data) if callable(self.repair) else (self.repair or data["draft"])
        raise AssertionError(f"root pipeline must not call task {task!r}")


def run(text, model, memory_enabled=True, prior=None, **kw):
    d = tempfile.mkdtemp()
    s = Settings(db_path=os.path.join(d, "t.sqlite3"), model_backend="http", model_url="http://x/", **kw)
    orch = Orchestrator(s, MemoryStore(s.db_path, 100), model)
    sid = orch.bind(LOCAL, "s1")
    orch.sessions.get(sid).memory_enabled = memory_enabled

    async def go():
        out = []
        for t in (prior or []):
            out.append(await orch.process(sid, t))
        out.append(await orch.process(sid, text))
        return out[-1]
    return asyncio.run(go()), orch


def test_default_pipeline_is_root():
    assert Settings().pipeline == "root"


def test_clean_draft_published_untouched():
    m = ScriptedModel(GOOD_CODE)
    r, _ = run("write a python function that reverses a list", m)
    f = r["final"]
    assert f["text"] == GOOD_CODE.strip()
    assert f["action_kind"] == "answer" and f["method"] == "root_model"
    assert r["selected"]["rule"].startswith("D4")
    assert [t for t, _ in m.calls] == ["root"]
    root_data = m.calls[0][1]
    assert root_data["request"] == "write a python function that reverses a list"
    assert "conversation_history" in root_data
    assert any(mm["origin"] == "seed" for mm in root_data["memories"])
    assert "root" in r["activated_components"] and "branches" in r["activated_components"]
    assert "planning" not in r["activated_components"]


def test_filler_and_question_trigger_one_repair():
    m = ScriptedModel("Great question! Which Python version are you on?", repair=GOOD_CODE)
    r, _ = run("write a python function that reverses a list", m)
    assert [t for t, _ in m.calls] == ["root", "repair"]
    rep = m.calls[1][1]
    names = {p["name"] for p in rep["problems"]}
    assert "no_filler_opener" in names and "answers_instead_of_asking" in names and "code_block_present" in names
    assert rep["draft"].startswith("Great question")
    assert r["final"]["text"] == GOOD_CODE.strip()
    assert r["selected"]["rule"] == "D2 repaired"
    rounds = r["branches"]["rounds"]
    assert len(rounds) == 2 and rounds[1]["report"]["failed"] == []


def test_placeholder_code_flagged_and_bad_repair_rejected():
    draft = "```python\ndef reverse(items):\n    # TODO: implement\n    ...\n```"
    m = ScriptedModel(draft, repair="Great question! " + draft)      # repair makes it worse
    r, _ = run("write a python function that reverses a list", m)
    assert r["selected"]["rule"].startswith("D3")
    assert r["final"]["text"] == draft
    assert "no_placeholders" in r["final"]["notes"]           # remaining failure is visible, not hidden


def test_deterministic_fixes_without_model_repair():
    m = ScriptedModel("The capital of France is Paris. My key is gsk_abcdefghijklmnopqrstuvwxyz0123456789 ok.")
    r, _ = run("what is the capital of France in at most 12 words", m)
    assert [t for t, _ in m.calls] == ["root"]                  # fix-severity failures never cost a model call
    assert "gsk_" not in r["final"]["text"]
    assert any(x.startswith("redacted_secret") for x in r["final"]["post_checks"])
    assert r["selected"]["rule"] == "D1 published_with_fixes"


def test_question_allowed_when_underspecified():
    m = ScriptedModel("Which file do you mean, and in which language?")
    r, _ = run("fix it", m)
    assert [t for t, _ in m.calls] == ["root"]
    assert r["final"]["action_kind"] == "clarify"
    assert r["selected"]["rule"].startswith("D4")


def test_fake_tool_claim_repaired():
    m = ScriptedModel("I searched the web and the answer is 42.", repair="The answer is 42.")
    r, _ = run("what is six times seven", m)
    assert [t for t, _ in m.calls] == ["root", "repair"]
    assert m.calls[1][1]["problems"][0]["name"] == "no_fake_tool_use"
    assert r["final"]["text"] == "The answer is 42."


def test_model_down_gives_truthful_notice():
    m = ScriptedModel(None, fail_root=True)
    r, _ = run("what is six times seven", m)
    assert r["final"]["action_kind"] == "notice"
    assert "could not be used" in r["final"]["text"]
    assert r["selected"]["rule"] == "D0 model_unavailable"


def test_save_memory_is_rule_only():
    m = ScriptedModel("should never be used")
    r, orch = run("remember that my laptop runs Kali", m)
    assert m.calls == []
    assert r["final"]["action_kind"] == "acknowledge_save"
    assert r["memory_write"]["status"] == "saved"
    r2, _ = run("remember that my laptop runs Kali", ScriptedModel("x"), memory_enabled=False)
    assert r2["final"]["action_kind"] == "notice" and "Memory is turned off" in r2["final"]["text"]


def test_fetch_without_url_is_rule_only():
    m = ScriptedModel("never")
    r, _ = run("/fetch", m)
    assert m.calls == [] and r["final"]["action_kind"] == "clarify"


def test_repair_disabled_keeps_original_with_flags():
    m = ScriptedModel("Sure! Here you go: 42.")
    r, _ = run("what is six times seven", m, root_repair_enabled=False)
    assert [t for t, _ in m.calls] == ["root"]
    assert r["selected"]["rule"] == "D3 repair_skipped"
    assert "no_filler_opener" in r["final"]["notes"]


def test_branches_unit_checks():
    b = Branches(16000)
    obs = Observation(raw_text="x", request="write a function", intent="request", domain="code",
                      constraints=[Constraint("c1", "forbid_term", "foo", "no foo")])
    rep = b.inspect(obs, "Here: foo\n```py\nprint(1)\n", user_memories=0, tool_result=None)
    failed = {c.name for c in rep.failed}
    assert "forbid_term:foo" in failed and "fences_closed" in failed
    assert rep.needs_fix and rep.needs_repair and not rep.blocked
    assert b.inspect(obs, "   ", user_memories=0, tool_result=None).blocked
