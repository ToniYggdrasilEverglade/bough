"""Central orchestrator: runs one bounded processing cycle per request.

    input -> perception || memory retrieval -> attention -> workspace -> planning -> evaluation
          -> action selection -> (tool: web_fetch) -> (memory write) -> language -> final response -> [feedback]

Tool note: the only executable action is `web_fetch`, and it runs only when selection picked a
rule-originated proposal created by the explicit `/fetch <url> [question]` command (model-proposed
web_fetch is blocked by evaluation). The URL goes to WebTool, which defers to the trusted wrapper +
Rust CLI for allowlisting, DNS/IP policy and size limits; the model never chooses URLs. The result
is posted to the workspace and handed to the language agent as untrusted source text.

Memory write note: the prompt lists storage after language output. Here the (rule-based) write is
committed right after selection and BEFORE verbalisation so the reply can report what really
happened instead of promising it. It is skipped unless the selected action is `acknowledge_save`,
the session has memory enabled, and the cycle is still current.

Guarantees: one final response per cycle; a newer request cancels the older cycle; cancelled or stale
cycles cannot write memory, touch the workspace, or publish a final response; model calls, time,
proposals, workspace size and retries are bounded (see Settings).
"""
from __future__ import annotations

import asyncio
import json
import functools
import logging
from typing import Any, Dict, List, Optional

from .agents.adaptation import ADAPTATION_DOC, Adaptation
from .agents.attention import Attention
from .agents.branches import Branches, apply_fixes, classify
from .agents.evaluation import CRITERIA, Evaluator
from .agents.language import LanguageAgent
from .agents.perception import Perception, detect_domain
from .agents.planning import Planner
from .agents.selection import RULES_TEXT, Selector
from .config import Settings
from .context import CycleContext, ModelBudget
from .errors import (BoughError, CycleSuperseded, InputRejected, LoopGuardError, MemoryRejected, MemoryUnavailable, ModelFailure, NotFound,
                     StaleCycleError)
from .event_bus import EventBus
from .memory.base import MemoryBackend
from .models.base import ModelAdapter
from .models.gateway import ModelGateway
from .models.validation import validate_text
from .schemas import Event, EventType, RawInput, SelectedAction, clip, jsonable, new_id, utcnow_iso
from .memory.base import Owner
from .memory.seed import SeedStore
from .sessions import Session, SessionManager
from . import github_tool
from .web_tool import WebTool

DECISION_RULES = [
    "R0 rule-only paths: explicit 'remember ...' saves and /fetch run without asking the model to decide.",
    "D0 blocked / model_unavailable: no usable draft -> truthful notice, nothing invented.",
    "D4 published_clean: every branch check passed -> the model's text is published untouched.",
    "D1 published_with_fixes: only deterministic fixes were needed (forbidden terms, length limits, secret redaction).",
    "D2 repaired: at least one check needed a rewrite -> ONE bounded repair call; the rewrite is re-inspected and used only if it fails fewer checks.",
    "D3 repair_rejected / repair_unavailable / repair_skipped / published_with_flags: original kept, remaining failures shown as flags.",
]

log = logging.getLogger("bough.orchestrator")
ET = EventType


class Orchestrator:
    def __init__(self, settings: Settings, store: MemoryBackend, adapter: ModelAdapter,
                 bus: Optional[EventBus] = None):
        self.settings, self.store, self.adapter = settings, store, adapter
        self.bus = bus or EventBus(settings.max_events_per_cycle, settings.max_events_per_source_type)
        self.sessions = SessionManager(settings)
        self.gateway = ModelGateway(adapter, settings)
        self.perception = Perception(settings)
        self.attention = Attention(settings)
        self.planner = Planner(settings, self.gateway)
        self.evaluator = Evaluator(settings, self.gateway)
        self.selector = Selector(settings)
        self.language = LanguageAgent(settings, self.gateway)
        self.adaptation = Adaptation(settings)
        self.web_tool = WebTool(settings)
        self.github = github_tool.GitHubTool(settings.github_token)
        self.seeds = SeedStore(enabled=settings.seed_enabled)
        self.branches = Branches(settings.max_reply_chars)

    # ================================================================ public API
    def describe(self) -> Dict[str, Any]:
        return {"settings": self.settings.public_dict(),
                "model": {"name": self.adapter.name, "is_mock": self.adapter.is_mock,
                          "backend": self.settings.model_backend},
                "auth": {"mode": self.settings.auth_mode, "supabase_url": self.settings.supabase_url or None,
                         "publishable_key": self.settings.supabase_publishable_key or None},
                "memory": {"backend": self.store.name},
                "seeds": self.seeds.describe(),
                "criteria": CRITERIA, "selection_rules": RULES_TEXT, "adaptation": ADAPTATION_DOC,
                "pipeline": self.settings.pipeline, "decision_rules": DECISION_RULES,
                "components": (["perception", "attention", "memory", "workspace", "tool", "root", "branches",
                                "decision", "language", "feedback"] if self.settings.pipeline == "root" else
                               ["perception", "attention", "memory", "workspace", "planning", "evaluation",
                                "selection", "tool", "language", "feedback"])}

    def bind(self, owner: Owner, session_id: str, create: bool = True) -> str:
        """Resolve a client session id for `owner`, (re)attach the owner's current credentials to the
        session, and return the internal id used by every other method."""
        sid = self.sessions.scoped_id(owner, session_id)
        sess = self.sessions.get(sid, create=create)
        sess.owner = owner
        return sid

    def session_state(self, session_id: str) -> Dict[str, Any]:
        s = self.sessions.get(session_id)
        return {"session_id": s.session_id, "memory_enabled": s.memory_enabled,
                "workspace": s.workspace.snapshot(), "events": self.bus.history(session_id)[-300:],
                "running": bool(s.task and not s.task.done()),
                "conversation_history": {"exchanges": len(s.conversation),
                                         "dropped_exchanges": s.history_dropped}}

    def set_memory_enabled(self, session_id: str, enabled: bool) -> Dict[str, Any]:
        s = self.sessions.get(session_id)
        s.memory_enabled = bool(enabled)
        return {"session_id": session_id, "memory_enabled": s.memory_enabled}

    async def process(self, session_id: str, text: str, modality: str = "text") -> Dict[str, Any]:
        task = await self.submit(session_id, text, modality)
        return await task

    async def submit(self, session_id: str, text: str, modality: str = "text") -> "asyncio.Task":
        if not isinstance(text, str) or not text.strip():
            raise InputRejected("message is empty")
        if len(text) > self.settings.max_input_chars:
            raise InputRejected(f"message is longer than {self.settings.max_input_chars} characters")
        sess = self.sessions.get(session_id)
        old_task, old_ctx = sess.task, sess.ctx
        cycle_id = new_id("cy_")
        # Mark the new cycle current BEFORE touching the old one: any late write from the old cycle is stale.
        sess.current_cycle_id = cycle_id
        sess.workspace.begin_cycle(cycle_id)
        if old_task is not None and not old_task.done():
            old_ctx.cancel_reason = "superseded_by_newer_request"
            old_task.cancel()
            await asyncio.gather(old_task, return_exceptions=True)
        if old_ctx is not None and not old_ctx.terminal:
            await self._finish_cancelled(old_ctx, old_ctx.cancel_reason or "superseded_by_newer_request")
        ctx = CycleContext(session_id, cycle_id, text, ModelBudget(self.settings.max_model_calls))
        ctx.emit = functools.partial(self._emit, ctx)
        ctx.data["clear_conversation"] = text.strip() == "/clear"
        if ctx.data["clear_conversation"]:
            sess.clear_conversation()
        ctx.data["conversation_history"] = sess.conversation_snapshot()
        raw = RawInput(modality, text)
        sess.ctx = ctx
        sess.task = asyncio.create_task(self._run(ctx, sess, raw))
        return sess.task

    async def cancel(self, session_id: str, reason: str = "cancelled_by_user") -> bool:
        sess = self.sessions.get(session_id, create=False)
        if sess.task is None or sess.task.done():
            return False
        sess.ctx.cancel_reason = reason
        sess.task.cancel()
        await asyncio.gather(sess.task, return_exceptions=True)
        return True

    # ================================================================ cycle runner
    async def _run(self, ctx: CycleContext, sess: Session, raw: RawInput) -> Dict[str, Any]:
        status, final, reason = "completed", None, None
        try:
            await self._emit(ctx, "user", ET.INPUT_RECEIVED, {
                "chars": len(raw.data), "modality": raw.modality, "text": clip(raw.data, 300),
                "memory_enabled": sess.memory_enabled})
            limit = self.settings.cycle_timeout_s
            probe = {"request": raw.data if isinstance(raw.data, str) else ""}
            if any(callable(getattr(self.adapter, n, None)) and getattr(self.adapter, n)("root", probe)
                   for n in ("uses_web_search", "uses_search_helper")):
                # Groq browser search: leave room for the search plus a possible repair call.
                limit = max(limit, self.settings.model_web_search_timeout_s + 2 * self.settings.model_timeout_s + 10)
            if self.settings.github_lookup == "auto" and github_tool.detect(probe["request"]) is not None:
                limit += 2 * self.github.timeout_s        # repo info + one list request
            final = await asyncio.wait_for(self._pipeline(ctx, sess, raw), timeout=limit)
        except asyncio.TimeoutError:
            status, reason = "timeout", f"cycle exceeded {limit}s"
            ctx.limits_hit.append({"limit": "cycle_timeout_s", "value": limit})
            await self._emit(ctx, "orchestrator", ET.CYCLE_TIMEOUT, {"limit_s": limit})
            final = await self._notice_final(ctx, sess, "The processing time limit was reached before a response "
                                                         "was ready. Nothing was saved from this request.")
        except asyncio.CancelledError:
            status, reason = "cancelled", ctx.cancel_reason or "cancelled"
            await self._finish_cancelled(ctx, reason)
        except CycleSuperseded as e:
            status, reason = "cancelled", str(e) or "superseded"
            await self._finish_cancelled(ctx, reason)
        except LoopGuardError as e:
            status, reason = "error", f"loop guard: {e}"
            await self._error(ctx, sess, reason)
        except Exception as e:  # noqa: BLE001 - never crash the server because of one cycle
            log.exception("cycle failed")
            status, reason = "error", f"internal error ({type(e).__name__})"
            await self._error(ctx, sess, reason)
            final = ctx.data.get("final")
        finally:
            self.bus.end_cycle(ctx.cycle_id)
        return self._result(ctx, status, final, reason)

    async def _pipeline(self, ctx: CycleContext, sess: Session, raw: RawInput) -> Dict[str, Any]:
        if ctx.data.get("clear_conversation"):
            return await self._notice_final(
                ctx, sess, "Conversation context cleared. Persistent memories were not deleted.")

        def model_view(component, **more):
            # Earlier workspace goals are not a substitute for conversational turns.
            # Avoid reintroducing old goals after /clear.
            return self.attention.view(component, obs, att, extra={
                "request": obs.request,
                "context": [],
                "conversation_history": ctx.data["conversation_history"],
                **more,
            })

        s = self.settings
        cid = ctx.cycle_id
        started = await self._emit(ctx, "orchestrator", ET.CYCLE_STARTED, {
            "model": {"name": self.adapter.name, "is_mock": self.adapter.is_mock},
            "limits": {k: getattr(s, k) for k in ("max_model_calls", "cycle_timeout_s", "max_proposals",
                                                  "workspace_max_items", "max_retrieved_memories", "max_retries")}})
        ctx.parent_id = started.event_id

        # ---- 1. perception and memory retrieval are independent (both need only the raw text) -> concurrent
        await self._activate(ctx, "perception", "new input")
        await self._activate(ctx, "memory", "retrieve candidates" if sess.memory_enabled else "memory disabled")

        async def retrieve():
            # Built-in seeds are not personal memory: they are consulted regardless of the memory toggle
            # (attention gives them their own budget so they never displace the user's own records).
            seed_rows = self.seeds.search(raw.data, s.max_seed_memories * 2, s.memory_min_score, detect_domain(raw.data))
            if not sess.memory_enabled:
                return seed_rows, "memory_disabled"
            origins = ["user_stated", "system_summary"] + (["model_interpretation"] if s.include_model_interpretations else [])
            try:
                rows = await asyncio.to_thread(self.store.search, sess.owner, raw.data, s.max_retrieved_memories * 3,
                                               s.memory_min_score, origins)
            except MemoryUnavailable as e:        # remote backend down: degrade to "no memories", say so
                log.warning("memory retrieval unavailable: %s", e)
                return seed_rows, f"memory_unavailable: {e}"
            return rows + seed_rows, None

        async def perceive():
            return self.perception.perceive(raw)

        obs, (rows, skip) = await asyncio.gather(perceive(), retrieve())
        self._current(ctx, sess)
        ctx.data["observation"] = obs.to_dict()
        ev_p = await self._emit(ctx, "perception", ET.PERCEPTION_DONE, obs.to_dict())
        ev_m = await self._emit(ctx, "memory", ET.MEMORY_RETRIEVED, {
            "candidates": len(rows), "skipped_reason": skip,
            "records": [self._mem_brief(m) for m in rows]}, parent=ev_p.event_id)
        ctx.parent_id = ev_m.event_id

        # ---- 2. attention + workspace
        await self._activate(ctx, "attention", "filter memories and context under budget")
        prior = sess.workspace.items(exclude_cycle=cid)
        att = self.attention.attend(obs, rows, prior)
        ctx.data["memories"] = [self._mem_brief(m) for m in att.memories]
        ctx.data["memories_dropped"] = att.dropped
        ev_a = await self._emit(ctx, "attention", ET.ATTENTION_DONE, {
            "kept": [m["id"] for m in att.memories], "dropped": att.dropped, "context": att.context,
            "budgets": att.budgets})
        ctx.parent_id = ev_a.event_id
        await self._activate(ctx, "workspace", "post goal, observation and memories")
        self._ws(ctx, sess, "goal", "orchestrator", {"goal": obs.request, "intent": obs.intent}, 1.0)
        self._ws(ctx, sess, "observation", "perception", obs.to_dict(), 0.8)
        for m in att.memories:
            self._ws(ctx, sess, "memory", "memory", self._mem_brief(m), 0.6)
        await self._workspace_event(ctx, sess)

        if s.pipeline == "root":
            return await self._root_pipeline(ctx, sess, obs, att, model_view)

        # ---- 3. planning
        await self._activate(ctx, "planning", "propose candidate actions")
        plan_view = model_view("planning")
        plan = await self.planner.plan(ctx, obs, plan_view, sess.memory_enabled)
        self._current(ctx, sess)
        ctx.data["proposals"] = [p.to_dict() for p in plan.proposals]
        ev_pl = await self._emit(ctx, "planning", ET.PLANNING_DONE, {
            "source": plan.source, "proposals": ctx.data["proposals"], "notes": plan.notes,
            "rejected": plan.rejected, "model_error": plan.model_error, "model_calls_used": ctx.budget.used})
        ctx.parent_id = ev_pl.event_id
        for p in plan.proposals:
            self._ws(ctx, sess, "proposal", "planning", p.to_dict(), 0.5)

        # ---- 4. evaluation
        await self._activate(ctx, "evaluation", "score proposals")
        evals = await self.evaluator.evaluate(ctx, obs, plan.proposals, {m["id"] for m in att.memories},
                                              model_view("evaluation"))
        self._current(ctx, sess)
        ctx.data["evaluations"] = [e.to_dict() for e in evals]
        ev_e = await self._emit(ctx, "evaluation", ET.EVALUATION_DONE, {
            "evaluations": ctx.data["evaluations"],
            "note": "Scores are rule-based. Any model review is a same-model self-check, not independent verification."})
        ctx.parent_id = ev_e.event_id
        for e in evals:
            self._ws(ctx, sess, "evaluation", "evaluation", e.to_dict(), 0.5)

        # ---- 5. selection (exactly one action)
        await self._activate(ctx, "selection", "apply selection rules")
        bias = await self._effective_bias(sess)
        sel = self.selector.select(obs, plan.proposals, evals, bias)
        ctx.data["selected"] = sel.to_dict()
        ev_s = await self._emit(ctx, "selection", ET.ACTION_SELECTED, sel.to_dict())
        ctx.parent_id = ev_s.event_id
        self._ws(ctx, sess, "selected_action", "selection", {"kind": sel.kind, "proposal_id": sel.proposal_id,
                                                              "rule": sel.rule}, 0.9)
        await self._workspace_event(ctx, sess)

        # ---- 5b. tool execution (only for the explicitly selected web_fetch action)
        tool_result: Optional[Dict[str, Any]] = None
        if sel.kind == "web_fetch":
            tool_result = await self._run_web_fetch(ctx, sess, sel)
            ctx.data["tool_result"] = self._tool_brief(tool_result)

        # ---- 6. memory write (rule-based, opt-in, explicit only)
        mw = await self._maybe_store(ctx, sess, obs, sel)
        ctx.data["memory_write"] = mw
        ev_w = await self._emit(ctx, "memory", ET.MEMORY_WRITE, mw)
        ctx.parent_id = ev_w.event_id

        # ---- 7. language
        await self._activate(ctx, "language", "render selected action")
        self._current(ctx, sess)
        lang_view = model_view("language", tool_result=tool_result) if tool_result is not None else model_view("language")
        lang = await self.language.render(ctx, obs, sel, lang_view, mw)
        self._current(ctx, sess)
        ev_l = await self._emit(ctx, "language", ET.LANGUAGE_DONE, {
            "method": lang.method, "post_checks": lang.post_checks, "notes": lang.notes,
            "action_kind": sel.kind, "model_calls_used": ctx.budget.used})
        ctx.parent_id = ev_l.event_id

        final = {"text": lang.text, "action_kind": sel.kind, "proposal_id": sel.proposal_id,
                 "method": lang.method, "post_checks": lang.post_checks, "notes": lang.notes,
                 "mock": self.adapter.is_mock, "tool_result": ctx.data.get("tool_result")}
        return await self._publish_final(ctx, sess, final, obs.request)

    # ================================================================ root pipeline (Bough)
    async def _root_pipeline(self, ctx: CycleContext, sess: Session, obs, att, model_view) -> Dict[str, Any]:
        """Trunk: the model answers the request freely, once, in plain text. Branches: rule-based agents
        inspect that draft (constraints, completeness, directness, grounding, safety, size). Decision:
        publish / fix / one bounded repair call / notice. The model never sees a schema and never decides
        whether its own output is acceptable."""
        s = self.settings
        tool_result: Optional[Dict[str, Any]] = None

        # ---- 3. rule-only paths: explicit memory saves and the /fetch command never need the model to decide
        if obs.intent == "save_memory":
            if obs.underspecified:
                sel = self._rule_action("clarify", {"question": "What would you like me to remember?"},
                                        "R0 save_memory_underspecified", "explicit remember-request without content")
            elif sess.memory_enabled:
                sel = self._rule_action("acknowledge_save", {"template": "acknowledge_save"}, "R0 save_memory",
                                        "explicit remember-request; stored verbatim, no model involved")
            else:
                sel = self._rule_action("notice", {"code": "memory_disabled"}, "R0 save_memory_memory_off",
                                        "explicit remember-request while memory is off")
            return await self._root_finish_template(ctx, sess, obs, sel, model_view)

        fetch_question: Optional[str] = None
        if obs.request == "/fetch" or obs.request.startswith("/fetch "):
            parts = obs.request.split(maxsplit=2)
            if len(parts) < 2:
                sel = self._rule_action("clarify", {"question": "Use /fetch https://example.com/ followed by an optional question."},
                                        "R0 fetch_without_url", "/fetch needs a URL")
                return await self._root_finish_template(ctx, sess, obs, sel, model_view)
            fetch_question = parts[2] if len(parts) > 2 else "Summarize the fetched page concisely."
            sel = self._rule_action("web_fetch", {"url": parts[1], "points": [fetch_question]}, "R0 explicit_fetch",
                                    "URL came from the user's /fetch command")
            tool_result = await self._run_web_fetch(ctx, sess, sel)
            ctx.data["tool_result"] = self._tool_brief(tool_result)
            if not tool_result.get("ok"):
                return await self._root_finish_template(ctx, sess, obs, sel, model_view, tool_result=tool_result)

        # ---- 3b. GitHub lookup: the user named a repo and asked for its issues / PRs / releases / commits
        if tool_result is None and s.github_lookup == "auto":
            gh_query = github_tool.detect(obs.request)
            if gh_query is not None:
                tool_result = await self._run_github(ctx, sess, gh_query)
                ctx.data["tool_result"] = self._tool_brief(tool_result)

        # ---- 4. ROOT: one free-form model call
        await self._activate(ctx, "root", "model answers the request freely (plain text, no schema)")
        extra = {"tool_result": tool_result} if tool_result is not None else {}
        if fetch_question:
            extra["request"] = fetch_question
        view = model_view("language", **extra)
        draft, model_error = await self._root_call(ctx, "root", view)
        self._current(ctx, sess)
        ctx.data["root"] = {"method": "model" if draft is not None else "unavailable", "model_error": model_error,
                            "chars": len(draft or ""), "rounds": 1}
        ev_r = await self._emit(ctx, "root", ET.ROOT_DONE, {
            "round": 1, "task": "root", "ok": draft is not None, "model_error": model_error,
            "chars": len(draft or ""), "preview": clip(draft or "", 400), "model_calls_used": ctx.budget.used})
        ctx.parent_id = ev_r.event_id
        if draft is None:
            sel = self._rule_action("notice", {"code": "model_unavailable", "reason": model_error,
                                               "detail": ctx.data.get("root_model_detail", "")},
                                    "D0 model_unavailable", "root model produced no usable draft")
            return await self._root_finish_template(ctx, sess, obs, sel, model_view, tool_result=tool_result)
        self._ws(ctx, sess, "draft", "root", {"preview": clip(draft, 300), "chars": len(draft)}, 0.7)
        if tool_result is None and getattr(draft, "executed_tools", None):
            # The provider really searched (Groq browser_search). Record it so the grounding branch accepts an
            # honest "I searched the web" and the trace shows what was looked up.
            tool_result, brief = self._provider_search_result(draft.executed_tools)
            ctx.data["tool_result"] = brief
            ev_t = await self._emit(ctx, "tool", ET.TOOL_DONE, brief)
            ctx.parent_id = ev_t.event_id
            self._ws(ctx, sess, "tool_result", "tool", brief, 0.85)

        # ---- 5. BRANCHES: rule-based inspection (no model)
        await self._activate(ctx, "branches", "inspect the draft: constraints, completeness, directness, grounding, safety")
        user_mem = sum(1 for m in att.memories if m.get("origin") != "seed")
        report = self.branches.inspect(obs, draft, user_memories=user_mem, tool_result=tool_result, round_no=1)
        ev_b = await self._emit(ctx, "branches", ET.BRANCHES_DONE, dict(report.to_dict(), round=1))
        ctx.parent_id = ev_b.event_id
        self._ws(ctx, sess, "branch_report", "branches", {"round": 1, "failed": [c.name for c in report.failed],
                                                            "score": report.score}, 0.6)
        rounds = [{"round": 1, "report": report.to_dict()}]

        # ---- 6. DECISION (rules)
        await self._activate(ctx, "decision", "publish, fix, repair once, or fall back")
        final_text, final_report, rule, reason = draft, report, "D4 published_clean", "every branch check passed"
        repair_info: Dict[str, Any] = {"attempted": False}
        if report.blocked:
            sel = self._rule_action("notice", {"code": "model_unavailable", "reason": "empty draft"},
                                    "D0 blocked", "; ".join(c.detail for c in report.failed if c.severity == "block"))
            return await self._root_finish_template(ctx, sess, obs, sel, model_view, tool_result=tool_result)
        if report.needs_repair and s.root_repair_enabled and ctx.budget.remaining > 0:
            problems = [{"name": c.name, "branch": c.branch, "detail": c.detail}
                        for c in report.failed if c.severity in ("repair", "fix")]
            repaired, rep_err = await self._root_call(ctx, "repair", dict(view, draft=draft, problems=problems))
            self._current(ctx, sess)
            ctx.data["root"]["rounds"] = 2
            ev_r2 = await self._emit(ctx, "root", ET.ROOT_DONE, {
                "round": 2, "task": "repair", "ok": repaired is not None, "model_error": rep_err,
                "chars": len(repaired or ""), "preview": clip(repaired or "", 400), "model_calls_used": ctx.budget.used,
                "problems": problems})
            ctx.parent_id = ev_r2.event_id
            repair_info = {"attempted": True, "ok": repaired is not None, "model_error": rep_err, "problems": problems}
            if repaired is not None:
                report2 = self.branches.inspect(obs, repaired, user_memories=user_mem, tool_result=tool_result, round_no=2)
                ev_b2 = await self._emit(ctx, "branches", ET.BRANCHES_DONE, dict(report2.to_dict(), round=2))
                ctx.parent_id = ev_b2.event_id
                rounds.append({"round": 2, "report": report2.to_dict()})
                if not report2.blocked and len(report2.failed) < len(report.failed):
                    final_text, final_report = repaired, report2
                    rule, reason = "D2 repaired", f"repair fixed {len(report.failed) - len(report2.failed)} of {len(report.failed)} failed checks"
                    repair_info["used"] = True
                else:
                    rule, reason = "D3 repair_rejected", "repaired draft was not better; original kept"
                    repair_info["used"] = False
            else:
                rule, reason = "D3 repair_unavailable", f"repair call failed ({rep_err}); original kept"
        elif report.needs_repair:
            rule, reason = "D3 repair_skipped", ("repair disabled" if not s.root_repair_enabled else "model call budget exhausted")

        final_text, fixes = apply_fixes(final_text, obs, s.max_reply_chars)
        if fixes and rule == "D4 published_clean":
            rule, reason = "D1 published_with_fixes", "deterministic fixes applied"
        remaining = [c.name for c in final_report.failed if c.severity == "repair"]
        if remaining and not rule.startswith("D2"):
            rule = rule if rule.startswith("D3") else "D3 published_with_flags"
            reason = reason + f"; still failing: {', '.join(remaining)}"
        kind = classify(final_text, obs, tool_result)
        sel = SelectedAction(proposal_id=None, kind=kind, content_spec={"source": "root"}, rule=rule, reason=reason,
                             ranking=[], supported_by=[m["id"] for m in att.memories if m.get("origin") != "seed"])
        ctx.data["selected"] = sel.to_dict()
        ctx.data["branches"] = {"rounds": rounds, "final_failed": [c.name for c in final_report.failed],
                                "repair": repair_info, "fixes": fixes}
        ev_s = await self._emit(ctx, "decision", ET.ACTION_SELECTED, dict(sel.to_dict(), repair=repair_info, fixes=fixes,
                                                                          still_failing=remaining))
        ctx.parent_id = ev_s.event_id
        self._ws(ctx, sess, "selected_action", "decision", {"kind": kind, "rule": rule}, 0.9)
        await self._workspace_event(ctx, sess)

        mw = await self._maybe_store(ctx, sess, obs, sel)
        ctx.data["memory_write"] = mw
        ev_w = await self._emit(ctx, "memory", ET.MEMORY_WRITE, mw)
        ctx.parent_id = ev_w.event_id

        await self._activate(ctx, "language", "publish the filtered reply")
        self._current(ctx, sess)
        ev_l = await self._emit(ctx, "language", ET.LANGUAGE_DONE, {
            "method": "root_model", "post_checks": fixes, "notes": remaining, "action_kind": kind,
            "model_calls_used": ctx.budget.used})
        ctx.parent_id = ev_l.event_id
        final = {"text": final_text, "action_kind": kind, "proposal_id": None, "method": "root_model",
                 "post_checks": fixes, "notes": remaining, "rule": rule, "mock": self.adapter.is_mock,
                 "tool_result": ctx.data.get("tool_result")}
        return await self._publish_final(ctx, sess, final, obs.request)

    async def _root_call(self, ctx: CycleContext, task: str, data: Dict[str, Any]):
        """One gated plain-text model call. Returns (text | None, error_code | None)."""
        try:
            text = await self.gateway.call(ctx, task, data, lambda raw: validate_text(raw, self.settings.max_reply_chars))
            return text, None
        except ModelFailure as f:
            ctx.data["root_model_detail"] = clip(f.detail or "", 300)
            return None, f.code

    @staticmethod
    def _rule_action(kind: str, spec: Dict[str, Any], rule: str, reason: str) -> SelectedAction:
        return SelectedAction(proposal_id=None, kind=kind, content_spec=spec, rule=rule, reason=reason, ranking=[])

    async def _root_finish_template(self, ctx, sess, obs, sel: SelectedAction, model_view, tool_result=None):
        """Rule-decided outcomes (saves, fetch errors, model failures): template text, no model call."""
        await self._activate(ctx, "decision", "rule-decided action, no model involved")
        ctx.data["selected"] = sel.to_dict()
        ev_s = await self._emit(ctx, "decision", ET.ACTION_SELECTED, sel.to_dict())
        ctx.parent_id = ev_s.event_id
        self._ws(ctx, sess, "selected_action", "decision", {"kind": sel.kind, "rule": sel.rule}, 0.9)
        mw = await self._maybe_store(ctx, sess, obs, sel)
        ctx.data["memory_write"] = mw
        ev_w = await self._emit(ctx, "memory", ET.MEMORY_WRITE, mw)
        ctx.parent_id = ev_w.event_id
        await self._activate(ctx, "language", "render rule-decided action from template")
        spec = dict(sel.content_spec, render="template")
        tsel = SelectedAction(sel.proposal_id, sel.kind, spec, sel.rule, sel.reason, sel.ranking)
        view = model_view("language", tool_result=tool_result) if tool_result is not None else model_view("language")
        lang = await self.language.render(ctx, obs, tsel, view, mw)
        ev_l = await self._emit(ctx, "language", ET.LANGUAGE_DONE, {
            "method": lang.method, "post_checks": lang.post_checks, "notes": lang.notes,
            "action_kind": sel.kind, "model_calls_used": ctx.budget.used})
        ctx.parent_id = ev_l.event_id
        final = {"text": lang.text, "action_kind": sel.kind, "proposal_id": None, "method": lang.method,
                 "post_checks": lang.post_checks, "notes": lang.notes, "rule": sel.rule,
                 "mock": self.adapter.is_mock, "tool_result": ctx.data.get("tool_result")}
        return await self._publish_final(ctx, sess, final, obs.request)

    # ================================================================ helpers
    async def _emit(self, ctx: CycleContext, source: str, etype: EventType, payload: Optional[dict] = None,
                    parent: Optional[str] = None) -> Event:
        ev = Event.create(ctx.session_id, ctx.cycle_id, source, etype, payload, parent or ctx.parent_id)
        if await self.bus.publish(ev):
            ctx.events.append(ev.to_dict())
        return ev

    async def _activate(self, ctx: CycleContext, component: str, reason: str) -> Event:
        ev = await self._emit(ctx, "orchestrator", ET.COMPONENT_ACTIVATED, {"component": component, "reason": reason})
        if component not in ctx.components:
            ctx.components.append(component)
        return ev

    def _current(self, ctx: CycleContext, sess: Session) -> None:
        if sess.current_cycle_id != ctx.cycle_id or ctx.cancel_reason is not None:
            raise CycleSuperseded(ctx.cancel_reason or "superseded_by_newer_request")

    def _ws(self, ctx, sess, kind, source, content, priority):
        try:
            return sess.workspace.add(kind, source, ctx.cycle_id, content, priority)
        except StaleCycleError as e:
            raise CycleSuperseded(str(e)) from None

    async def _workspace_event(self, ctx, sess):
        await self._emit(ctx, "workspace", ET.WORKSPACE_UPDATED, {
            "size": len(sess.workspace), "max_items": sess.workspace.max_items,
            "evicted": sess.workspace.last_evicted,
            "items": [i for i in sess.workspace.snapshot(120)]})

    @staticmethod
    def _mem_brief(m: Dict[str, Any]) -> Dict[str, Any]:
        return {k: m.get(k) for k in ("id", "kind", "text", "origin", "trust", "score", "created_at")}

    async def _effective_bias(self, sess: Session) -> Dict[str, float]:
        persisted = {}
        if sess.memory_enabled:
            try:
                persisted = await asyncio.to_thread(self.store.get_all_bias, sess.owner)
            except MemoryUnavailable as e:
                log.warning("bias lookup unavailable: %s", e)
        kinds = set(persisted) | set(sess.bias)
        return {k: self.adaptation.clamp(persisted.get(k, 0.0) + sess.bias.get(k, 0.0)) for k in kinds}

    # ---------------------------------------------------------------- tool: web_fetch
    async def _run_web_fetch(self, ctx: CycleContext, sess: Session, sel) -> Dict[str, Any]:
        """Execute the selected web_fetch. Permission, allowlist and network policy live in the trusted
        wrapper/CLI; this method only sequences, bounds and records. A stale or cancelled cycle never
        launches the tool, and a tool crash degrades to a truthful error result, never an exception."""
        url = str(sel.content_spec.get("url", ""))
        self._current(ctx, sess)                      # a stale cycle must not launch tools
        await self._activate(ctx, "tool", "execute selected web_fetch")
        ev_t = await self._emit(ctx, "tool", ET.TOOL_STARTED, {
            "tool": "web_fetch", "url": clip(url, 2048), "proposal_id": sel.proposal_id,
            "note": "URL came from the user's /fetch command; allowlist and network policy are enforced by the CLI."})
        ctx.parent_id = ev_t.event_id
        try:
            result = await self.web_tool.fetch(url)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - a tool bug must not crash the cycle
            log.exception("web_fetch tool failed")
            result = {"version": 1, "ok": False, "data": None,
                      "error": {"code": "TOOL_ERROR", "message": f"web fetch tool failed ({type(e).__name__})"}}
        self._current(ctx, sess)
        result = self._bound_tool_result(result)
        brief = self._tool_brief(result)
        ev_d = await self._emit(ctx, "tool", ET.TOOL_DONE, brief)
        ctx.parent_id = ev_d.event_id
        self._ws(ctx, sess, "tool_result", "tool", brief, 0.85)
        await self._workspace_event(ctx, sess)
        return result

    async def _run_github(self, ctx: CycleContext, sess: Session, q) -> Dict[str, Any]:
        """Read-only GitHub REST lookup; failures come back as a truthful error result, never an exception."""
        self._current(ctx, sess)
        await self._activate(ctx, "tool", f"GitHub API: {q.kind} of {q.full_name}")
        ev_t = await self._emit(ctx, "tool", ET.TOOL_STARTED, {
            "tool": "github", "url": f"https://api.github.com/repos/{q.full_name}", "lookup": q.kind,
            "note": "The user named this repository; read-only GET requests to api.github.com."})
        ctx.parent_id = ev_t.event_id
        try:
            result = await asyncio.to_thread(self.github.lookup, q)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("github lookup failed")
            result = github_tool.failed("TOOL_ERROR", f"GitHub lookup failed ({type(e).__name__})")
        self._current(ctx, sess)
        tool = result.get("tool", "github")
        result = dict(self._bound_tool_result(result), tool=tool)
        brief = self._tool_brief(result)
        ev_d = await self._emit(ctx, "tool", ET.TOOL_DONE, brief)
        ctx.parent_id = ev_d.event_id
        self._ws(ctx, sess, "tool_result", "tool", brief, 0.85)
        await self._workspace_event(ctx, sess)
        return result

    @staticmethod
    def _provider_search_result(executed: List[Dict[str, Any]]):
        queries = []
        for t in executed[:5]:
            args = t.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                args = {}
            q = args.get("query") if isinstance(args, dict) else None
            if isinstance(q, str) and q.strip():
                queries.append(clip(q.strip(), 200))
        result = {"version": 1, "ok": True, "error": None, "provider_search": True,
                  "data": {"final_url": None, "status": None, "content_type": None, "title": "Groq browser search",
                           "text": "", "chars": 0, "truncated": False, "context_truncated": False}}
        brief = {"tool": "web_search", "ok": True, "provider": "groq browser_search",
                 "searches": len(executed), "queries": queries}
        return result, brief

    def _bound_tool_result(self, result: Any) -> Dict[str, Any]:
        """Normalise to the wrapper's envelope and cap the page text that reaches the model."""
        if not isinstance(result, dict) or "ok" not in result:
            return {"version": 1, "ok": False, "data": None,
                    "error": {"code": "TOOL_ERROR", "message": "web fetch tool returned an unexpected result"}}
        if not result.get("ok") or not isinstance(result.get("data"), dict):
            return {"version": result.get("version", 1), "ok": False, "data": None,
                    "error": result.get("error") or {"code": "UNKNOWN", "message": "No page was retrieved."}}
        page = dict(result["data"])
        text = page.get("text") or ""
        limit = self.settings.web_fetch_context_chars
        page["context_truncated"] = len(text) > limit
        page["text"] = text[:limit]
        page["chars"] = len(text)
        return {"version": result.get("version", 1), "ok": True, "data": page, "error": None}

    @staticmethod
    def _tool_brief(result: Dict[str, Any]) -> Dict[str, Any]:
        """Event/workspace-sized summary: never the full page text."""
        if not result.get("ok"):
            err = result.get("error") or {}
            return {"tool": result.get("tool", "web_fetch"), "ok": False,
                    "error": {"code": err.get("code"), "message": clip(err.get("message", ""), 300)}}
        d = result["data"]
        return {"tool": result.get("tool", "web_fetch"), "ok": True, "final_url": d.get("final_url"), "status": d.get("status"),
                "content_type": d.get("content_type"), "title": clip(d.get("title") or "", 200) or None,
                "chars": d.get("chars"), "truncated": bool(d.get("truncated")),
                "context_truncated": bool(d.get("context_truncated")), "excerpt": clip(d.get("text", ""), 300)}

    async def _maybe_store(self, ctx: CycleContext, sess: Session, obs, sel) -> Dict[str, Any]:
        if sel.kind != "acknowledge_save":
            reason = ("memory_disabled" if obs.intent == "save_memory" and not sess.memory_enabled
                      else "no_explicit_save_request")
            return {"status": "skipped", "reason": reason}
        if not sess.memory_enabled:
            return {"status": "skipped", "reason": "memory_disabled"}
        self._current(ctx, sess)                      # a stale cycle must not write
        try:
            rec, created = await asyncio.to_thread(
                self.store.add, sess.owner, obs.save_text or "", obs.save_kind or "fact", "user_stated",
                ctx.session_id, ctx.cycle_id)
            return {"status": "saved" if created else "duplicate", "record": rec.to_dict()}
        except (MemoryRejected, MemoryUnavailable) as e:
            return {"status": "failed", "reason": str(e)}
        except Exception as e:  # noqa: BLE001
            log.exception("memory write failed")
            return {"status": "failed", "reason": f"storage error ({type(e).__name__})"}

    async def _publish_final(self, ctx: CycleContext, sess: Session, final: Dict[str, Any],
                             user_text: str) -> Dict[str, Any]:
        self._current(ctx, sess)                      # outdated cycles never publish
        ctx.terminal = True
        ctx.data["final"] = final
        await self._emit(ctx, "orchestrator", ET.RESPONSE_FINAL, final)
        self._current(ctx, sess)
        if not ctx.data.get("clear_conversation") and final["action_kind"] != "notice":
            sess.add_exchange(user_text, final["text"])
        sess.add_record(ctx.cycle_id, {"cycle_id": ctx.cycle_id, "user_text": user_text,
                                       "final_text": final["text"], "action_kind": final["action_kind"],
                                       "proposal_id": final.get("proposal_id"), "rated": False,
                                       "summary_saved": False})
        return final

    async def _notice_final(self, ctx, sess, text):
        try:
            return await self._publish_final(ctx, sess, {"text": text, "action_kind": "notice", "proposal_id": None,
                                                         "method": "template", "post_checks": [], "notes": [],
                                                         "mock": self.adapter.is_mock}, ctx.text)
        except CycleSuperseded:
            await self._finish_cancelled(ctx, "superseded_by_newer_request")
            return None

    async def _finish_cancelled(self, ctx: CycleContext, reason: str) -> None:
        if ctx.terminal:
            return
        ctx.terminal = True
        await self._emit(ctx, "orchestrator", ET.CYCLE_CANCELLED, {"reason": reason})

    async def _error(self, ctx: CycleContext, sess: Session, message: str) -> None:
        await self._emit(ctx, "orchestrator", ET.CYCLE_ERROR, {"message": message})
        if not ctx.terminal:
            await self._notice_final(ctx, sess, "Something went wrong while processing this request, so I stopped "
                                                "instead of guessing. Nothing was saved.")

    def _result(self, ctx: CycleContext, status: str, final: Optional[dict], reason: Optional[str]) -> Dict[str, Any]:
        d = ctx.data
        return {"cycle_id": ctx.cycle_id, "session_id": ctx.session_id, "status": status, "reason": reason,
                "final": final, "activated_components": ctx.components, "observation": d.get("observation"),
                "memories_retrieved": d.get("memories", []), "memories_dropped": d.get("memories_dropped", []),
                "proposals": d.get("proposals", []), "evaluations": d.get("evaluations", []),
                "selected": d.get("selected"), "tool_result": d.get("tool_result"), "memory_write": d.get("memory_write"),
                "root": d.get("root"), "branches": d.get("branches"), "pipeline": self.settings.pipeline,
                "model": {"name": self.adapter.name, "is_mock": self.adapter.is_mock},
                "model_calls_used": ctx.budget.used, "model_calls_limit": ctx.budget.limit,
                "limits_hit": ctx.limits_hit, "events": ctx.events}

    # ================================================================ feedback + memory actions
    async def record_feedback(self, session_id: str, cycle_id: str, rating: str) -> Dict[str, Any]:
        if rating not in ("useful", "not_useful"):
            raise InputRejected("rating must be 'useful' or 'not_useful'")
        sess = self.sessions.get(session_id, create=False)
        rec = sess.records.get(cycle_id)
        if rec is None:
            raise NotFound("no completed reply with that cycle id in this session")
        if rec["rated"]:
            raise InputRejected("this reply was already rated")
        kind, delta = rec["action_kind"], self.adaptation.delta(rating)
        if sess.memory_enabled:
            old, new = await asyncio.to_thread(self.store.apply_bias_delta, sess.owner, kind, delta, self.settings.bias_limit)
            await asyncio.to_thread(self.store.log_feedback, sess.owner, session_id, cycle_id, rec["proposal_id"], kind, rating)
            persisted = True
        else:
            old = sess.bias.get(kind, 0.0)
            new = self.adaptation.clamp(old + delta)
            sess.bias[kind] = new
            persisted = False
        rec["rated"] = True
        ctx = CycleContext(session_id, cycle_id, "", ModelBudget(0))
        ev = await self._emit(ctx, "feedback", ET.FEEDBACK_RECORDED, {
            "rating": rating, "proposal_id": rec["proposal_id"], "action_kind": kind,
            "mechanism": "selection_bias_rule", "bias_before": round(old, 3), "bias_after": round(new, 3),
            "persisted": persisted,
            "note": "Rule adjustment only. No model was trained and no memory was changed."})
        self.bus.end_cycle(cycle_id)
        return ev.to_dict()["payload"] | {"cycle_id": cycle_id}

    async def save_summary(self, session_id: str, cycle_id: str) -> Dict[str, Any]:
        sess = self.sessions.get(session_id, create=False)
        if not sess.memory_enabled:
            raise MemoryRejected("persistent memory is off for this session")
        rec = sess.records.get(cycle_id)
        if rec is None:
            raise NotFound("no completed reply with that cycle id in this session")
        text = (f"On {utcnow_iso()[:10]} the user asked: “{clip(rec['user_text'], 160)}”; "
                f"the system gave a {rec['action_kind']} reply.")
        record, created = await asyncio.to_thread(self.store.add, sess.owner, text, "summary", "system_summary",
                                                  session_id, cycle_id)
        rec["summary_saved"] = True
        return {"created": created, "record": record.to_dict()}
