"""Central orchestrator: runs one bounded processing cycle per request.

    input -> perception || memory retrieval -> attention -> workspace -> planning -> evaluation
          -> action selection -> (memory write) -> language -> final response -> [feedback]

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
import functools
import logging
from typing import Any, Dict, List, Optional

from .agents.adaptation import ADAPTATION_DOC, Adaptation
from .agents.attention import Attention
from .agents.evaluation import CRITERIA, Evaluator
from .agents.language import LanguageAgent
from .agents.perception import Perception
from .agents.planning import Planner
from .agents.selection import RULES_TEXT, Selector
from .config import Settings
from .context import CycleContext, ModelBudget
from .errors import (BrainError, CycleSuperseded, InputRejected, LoopGuardError, MemoryRejected, NotFound,
                     StaleCycleError)
from .event_bus import EventBus
from .memory.store import MemoryStore
from .models.base import ModelAdapter
from .models.gateway import ModelGateway
from .schemas import Event, EventType, RawInput, clip, jsonable, new_id, utcnow_iso
from .sessions import Session, SessionManager

log = logging.getLogger("brain.orchestrator")
ET = EventType


class Orchestrator:
    def __init__(self, settings: Settings, store: MemoryStore, adapter: ModelAdapter,
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

    # ================================================================ public API
    def describe(self) -> Dict[str, Any]:
        return {"settings": self.settings.public_dict(),
                "model": {"name": self.adapter.name, "is_mock": self.adapter.is_mock,
                          "backend": self.settings.model_backend},
                "criteria": CRITERIA, "selection_rules": RULES_TEXT, "adaptation": ADAPTATION_DOC,
                "components": ["perception", "attention", "memory", "workspace", "planning", "evaluation",
                               "selection", "language", "feedback"]}

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
            final = await asyncio.wait_for(self._pipeline(ctx, sess, raw), timeout=self.settings.cycle_timeout_s)
        except asyncio.TimeoutError:
            status, reason = "timeout", f"cycle exceeded {self.settings.cycle_timeout_s}s"
            ctx.limits_hit.append({"limit": "cycle_timeout_s", "value": self.settings.cycle_timeout_s})
            await self._emit(ctx, "orchestrator", ET.CYCLE_TIMEOUT, {"limit_s": self.settings.cycle_timeout_s})
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

        def model_view(component):
            # Earlier workspace goals are not a substitute for conversational turns.
            # Avoid reintroducing old goals after /clear.
            return self.attention.view(component, obs, att, extra={
                "request": obs.request,
                "context": [],
                "conversation_history": ctx.data["conversation_history"],
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
            if not sess.memory_enabled:
                return [], "memory_disabled"
            origins = ["user_stated", "system_summary"] + (["model_interpretation"] if s.include_model_interpretations else [])
            rows = await asyncio.to_thread(self.store.search, raw.data, s.max_retrieved_memories * 3,
                                           s.memory_min_score, origins)
            return rows, None

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

        # ---- 6. memory write (rule-based, opt-in, explicit only)
        mw = await self._maybe_store(ctx, sess, obs, sel)
        ctx.data["memory_write"] = mw
        ev_w = await self._emit(ctx, "memory", ET.MEMORY_WRITE, mw)
        ctx.parent_id = ev_w.event_id

        # ---- 7. language
        await self._activate(ctx, "language", "render selected action")
        self._current(ctx, sess)
        lang = await self.language.render(ctx, obs, sel, model_view("language"), mw)
        self._current(ctx, sess)
        ev_l = await self._emit(ctx, "language", ET.LANGUAGE_DONE, {
            "method": lang.method, "post_checks": lang.post_checks, "notes": lang.notes,
            "action_kind": sel.kind, "model_calls_used": ctx.budget.used})
        ctx.parent_id = ev_l.event_id

        final = {"text": lang.text, "action_kind": sel.kind, "proposal_id": sel.proposal_id,
                 "method": lang.method, "post_checks": lang.post_checks, "notes": lang.notes,
                 "mock": self.adapter.is_mock}
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
        persisted = await asyncio.to_thread(self.store.get_all_bias) if sess.memory_enabled else {}
        kinds = set(persisted) | set(sess.bias)
        return {k: self.adaptation.clamp(persisted.get(k, 0.0) + sess.bias.get(k, 0.0)) for k in kinds}

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
                self.store.add, obs.save_text or "", obs.save_kind or "fact", "user_stated",
                ctx.session_id, ctx.cycle_id)
            return {"status": "saved" if created else "duplicate", "record": rec.to_dict()}
        except MemoryRejected as e:
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
                "selected": d.get("selected"), "memory_write": d.get("memory_write"),
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
            old, new = await asyncio.to_thread(self.store.apply_bias_delta, kind, delta, self.settings.bias_limit)
            await asyncio.to_thread(self.store.log_feedback, session_id, cycle_id, rec["proposal_id"], kind, rating)
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
        record, created = await asyncio.to_thread(self.store.add, text, "summary", "system_summary",
                                                  session_id, cycle_id)
        rec["summary_saved"] = True
        return {"created": created, "record": record.to_dict()}
