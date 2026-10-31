"""FastAPI transport layer: REST + WebSocket. All behaviour lives in the Orchestrator; this file only
validates requests, enforces the localhost/Origin policy and forwards real backend events."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import LOOPBACK_HOSTS, Settings
from .errors import BrainError, InputRejected, MemoryRejected, NotFound
from .memory.store import MemoryStore
from .models.base import ModelAdapter
from .models.factory import build_adapter
from .orchestrator import Orchestrator

log = logging.getLogger("brain.app")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ChatBody(BaseModel):
    text: str = Field(..., max_length=20000)


class SettingsBody(BaseModel):
    memory_enabled: bool


class FeedbackBody(BaseModel):
    cycle_id: str = Field(..., max_length=64)
    rating: str = Field(..., max_length=16)


class MemoryBody(BaseModel):
    text: str = Field(..., max_length=2000)
    kind: str = Field("fact", max_length=16)


def _origin_allowed(origin: Optional[str]) -> bool:
    if origin is None:
        return True

    try:
        parsed = urlsplit(origin)

        if parsed.scheme not in ("http", "https"):
            return False

        if parsed.hostname in LOOPBACK_HOSTS:
            return True

        allowed = {
            value.strip().rstrip("/")
            for value in os.getenv(
                "BRAIN_ALLOWED_ORIGINS", ""
            ).split(",")
            if value.strip()
        }

        print(f"Allowed origins: {allowed}")
        print(f"Request origin: {origin}")
        return origin in allowed

    except ValueError:
        return False


def create_app(settings: Optional[Settings] = None, adapter: Optional[ModelAdapter] = None,
               store: Optional[MemoryStore] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or MemoryStore(settings.db_path, settings.max_memory_records)
    adapter = adapter or build_adapter(settings)
    orch = Orchestrator(settings, store, adapter)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        store.close()

    app = FastAPI(title="Brain AI prototype", lifespan=lifespan)
    app.state.orch = orch

    @app.exception_handler(BrainError)
    async def brain_error(_, exc: BrainError):
        code = 404 if isinstance(exc, NotFound) else 409 if isinstance(exc, MemoryRejected) else 400
        return JSONResponse({"error": str(exc)}, status_code=code)

    # ------------------------------------------------------------------ REST
    @app.get("/api/config")
    async def get_config():
        return orch.describe()

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str):
        return orch.session_state(sid)

    @app.put("/api/sessions/{sid}/settings")
    async def put_settings(sid: str, body: SettingsBody):
        return orch.set_memory_enabled(sid, body.memory_enabled)

    @app.post("/api/sessions/{sid}/chat")
    async def chat(sid: str, body: ChatBody):
        return await orch.process(sid, body.text)

    @app.post("/api/sessions/{sid}/cancel")
    async def cancel(sid: str):
        return {"cancelled": await orch.cancel(sid)}

    @app.post("/api/sessions/{sid}/feedback")
    async def feedback(sid: str, body: FeedbackBody):
        return await orch.record_feedback(sid, body.cycle_id, body.rating)

    @app.post("/api/sessions/{sid}/cycles/{cid}/save-summary")
    async def save_summary(sid: str, cid: str):
        return await orch.save_summary(sid, cid)

    @app.get("/api/memories")
    async def list_memories():
        recs = await asyncio.to_thread(store.list)
        return {"memories": [r.to_dict() for r in recs]}

    @app.post("/api/memories")
    async def add_memory(body: MemoryBody):
        rec, created = await asyncio.to_thread(store.add, body.text, body.kind, "user_stated")
        return {"created": created, "record": rec.to_dict()}

    @app.delete("/api/memories/{mid}")
    async def delete_memory(mid: str):
        if not await asyncio.to_thread(store.delete, mid):
            raise HTTPException(404, "memory not found")
        return {"deleted": mid}

    @app.get("/api/adaptation")
    async def adaptation():
        return {"bias": await asyncio.to_thread(store.get_all_bias),
                "limit": settings.bias_limit, "step": settings.bias_step,
                "explanation": orch.describe()["adaptation"]}

    # ------------------------------------------------------------------ WebSocket
    @app.websocket("/ws/{sid}")
    async def ws(websocket: WebSocket, sid: str):
        if not _origin_allowed(websocket.headers.get("origin")):
            await websocket.close(code=1008)
            return
        try:
            orch.sessions.get(sid)
        except InputRejected:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        lock = asyncio.Lock()

        async def send(obj):
            async with lock:
                await websocket.send_json(obj)

        async def on_event(ev):
            await send({"type": "event", "event": ev.to_dict()})

        unsubscribe = orch.bus.subscribe(on_event, session_id=sid)
        pending: set = set()

        async def deliver(task):
            try:
                await send({"type": "result", "result": await task})
            except Exception:  # noqa: BLE001 - socket closed
                pass

        try:
            await send({"type": "hello", "state": orch.session_state(sid), "config": orch.describe()})
            while True:
                msg = await websocket.receive_json()
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "message":
                    try:
                        task = await orch.submit(sid, msg.get("text", ""))
                    except BrainError as e:
                        await send({"type": "error", "message": str(e)})
                        continue
                    t = asyncio.create_task(deliver(task))
                    pending.add(t)
                    t.add_done_callback(pending.discard)
                elif kind == "cancel":
                    await orch.cancel(sid)
                else:
                    await send({"type": "error", "message": "unknown message type"})
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()
            try:
                await orch.cancel(sid, "client_disconnected")
            except BrainError:
                pass

    # static dashboard last, so /api and /ws take precedence
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app
