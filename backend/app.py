"""FastAPI transport layer: REST + WebSocket. All behaviour lives in the Orchestrator; this file only
validates requests, enforces the Origin policy and authentication, and forwards real backend events.

Auth: with BOUGH_AUTH_MODE=supabase every /api route except GET /api/config needs
`Authorization: Bearer <supabase access token>`; the WebSocket must send
{"type": "auth", "token": "..."} as its first frame. Session ids are namespaced per user."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import Authenticator, Principal
from .config import LOOPBACK_HOSTS, Settings
from .errors import (AuthRequired, BoughError, InputRejected, MemoryRejected, MemoryUnavailable, NotFound,
                     STTUnavailable, TTSUnavailable)
from .memory import MemoryBackend, build_store
from .models.base import ModelAdapter
from .models.factory import build_adapter
from .orchestrator import Orchestrator
from .stt import GroqSTT, build_stt
from .tts import TTS, build_tts

log = logging.getLogger("bough.app")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ChatBody(BaseModel):
    text: str = Field(..., max_length=20000)


class SettingsBody(BaseModel):
    memory_enabled: bool


class FeedbackBody(BaseModel):
    cycle_id: str = Field(..., max_length=64)
    rating: str = Field(..., max_length=16)


class TTSBody(BaseModel):
    text: str = Field(..., max_length=20000)
    voice_id: Optional[str] = Field(None, max_length=64)


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
                "BOUGH_ALLOWED_ORIGINS", os.getenv("BRAIN_ALLOWED_ORIGINS", "")   # BRAIN_*: pre-rename name
            ).split(",")
            if value.strip()
        }

        return origin.rstrip("/") in allowed

    except ValueError:
        return False


def create_app(settings: Optional[Settings] = None, adapter: Optional[ModelAdapter] = None,
               store: Optional[MemoryBackend] = None, authenticator: Optional[Authenticator] = None,
               tts: Optional[TTS] = None, stt: Optional[GroqSTT] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or build_store(settings)
    adapter = adapter or build_adapter(settings)
    auth = authenticator or Authenticator(settings)
    orch = Orchestrator(settings, store, adapter)
    tts = tts or build_tts(settings)
    stt = stt or build_stt(settings)
    if auth.required:
        log.info("auth: supabase (%s); memory backend: %s", settings.supabase_url, store.name)
    else:
        log.info("auth: off (single local user); memory backend: %s", store.name)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        store.close()

    app = FastAPI(title="Bough prototype", lifespan=lifespan)
    app.state.orch = orch
    app.state.auth = auth
    app.state.tts = tts
    app.state.stt = stt

    @app.exception_handler(BoughError)
    async def bough_error(_, exc: BoughError):
        code = (401 if isinstance(exc, AuthRequired) else 404 if isinstance(exc, NotFound)
                else 409 if isinstance(exc, MemoryRejected) else 503 if isinstance(exc, (MemoryUnavailable, TTSUnavailable, STTUnavailable)) else 400)
        headers = {"WWW-Authenticate": "Bearer"} if code == 401 else None
        return JSONResponse({"error": str(exc)}, status_code=code, headers=headers)

    async def current_user(authorization: Optional[str] = Header(default=None)) -> Optional[Principal]:
        return await auth.authenticate(auth.bearer(authorization))

    def bind(principal: Optional[Principal], sid: str, create: bool = True) -> str:
        return orch.bind(auth.owner_of(principal), sid, create=create)

    # ------------------------------------------------------------------ REST
    def describe() -> dict:
        return {**orch.describe(), "tts": tts.describe() if tts else {"enabled": False},
                "stt": stt.describe() if stt else {"enabled": False}}

    @app.get("/api/config")
    async def get_config():
        return describe()

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str, user: Optional[Principal] = Depends(current_user)):
        return orch.session_state(bind(user, sid))

    @app.put("/api/sessions/{sid}/settings")
    async def put_settings(sid: str, body: SettingsBody, user: Optional[Principal] = Depends(current_user)):
        return orch.set_memory_enabled(bind(user, sid), body.memory_enabled)

    @app.post("/api/sessions/{sid}/chat")
    async def chat(sid: str, body: ChatBody, user: Optional[Principal] = Depends(current_user)):
        return await orch.process(bind(user, sid), body.text)

    @app.post("/api/sessions/{sid}/cancel")
    async def cancel(sid: str, user: Optional[Principal] = Depends(current_user)):
        return {"cancelled": await orch.cancel(bind(user, sid, create=False))}

    @app.post("/api/sessions/{sid}/feedback")
    async def feedback(sid: str, body: FeedbackBody, user: Optional[Principal] = Depends(current_user)):
        return await orch.record_feedback(bind(user, sid, create=False), body.cycle_id, body.rating)

    @app.post("/api/sessions/{sid}/cycles/{cid}/save-summary")
    async def save_summary(sid: str, cid: str, user: Optional[Principal] = Depends(current_user)):
        return await orch.save_summary(bind(user, sid, create=False), cid)

    @app.get("/api/memories")
    async def list_memories(user: Optional[Principal] = Depends(current_user)):
        recs = await asyncio.to_thread(store.list, auth.owner_of(user))
        return {"memories": [r.to_dict() for r in recs]}

    @app.post("/api/memories")
    async def add_memory(body: MemoryBody, user: Optional[Principal] = Depends(current_user)):
        rec, created = await asyncio.to_thread(store.add, auth.owner_of(user), body.text, body.kind, "user_stated")
        return {"created": created, "record": rec.to_dict()}

    @app.delete("/api/memories/{mid}")
    async def delete_memory(mid: str, user: Optional[Principal] = Depends(current_user)):
        if not await asyncio.to_thread(store.delete, auth.owner_of(user), mid):
            raise HTTPException(404, "memory not found")
        return {"deleted": mid}

    @app.get("/api/adaptation")
    async def adaptation(user: Optional[Principal] = Depends(current_user)):
        return {"bias": await asyncio.to_thread(store.get_all_bias, auth.owner_of(user)),
                "limit": settings.bias_limit, "step": settings.bias_step,
                "explanation": orch.describe()["adaptation"]}

    @app.post("/api/tts")
    async def text_to_speech(body: TTSBody, user: Optional[Principal] = Depends(current_user)):
        """Read a reply aloud (ElevenLabs or Groq). Same auth as every other /api route; the key stays here."""
        if tts is None:
            raise TTSUnavailable("text-to-speech is off (set BOUGH_TTS_BACKEND=elevenlabs or groq)")
        audio = await tts.synthesize(body.text, body.voice_id)
        return Response(audio, media_type=tts.media_type, headers={"Cache-Control": "no-store"})

    @app.post("/api/transcribe")
    async def transcribe(request: Request, user: Optional[Principal] = Depends(current_user)):
        """Mic button: raw audio body (Content-Type audio/webm, audio/ogg, ...) -> Groq Whisper -> {"text"}."""
        if stt is None:
            raise STTUnavailable("speech input is off (set BOUGH_STT_BACKEND=groq)")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > stt.max_bytes:
            raise InputRejected(f"recording is larger than {stt.max_bytes // 1_000_000} MB")
        audio = bytearray()
        async for chunk in request.stream():
            audio += chunk
            if len(audio) > stt.max_bytes:
                raise InputRejected(f"recording is larger than {stt.max_bytes // 1_000_000} MB")
        return {"text": await stt.transcribe(bytes(audio), request.headers.get("content-type"))}

    @app.get("/api/me")
    async def me(user: Optional[Principal] = Depends(current_user)):
        if user is None:
            return {"auth": "off", "user_id": "local", "email": None}
        return {"auth": "supabase", "user_id": user.user_id, "email": user.email, "expires_at": user.expires_at}

    # ------------------------------------------------------------------ WebSocket
    @app.websocket("/ws/{client_sid}")
    async def ws(websocket: WebSocket, client_sid: str):
        if not _origin_allowed(websocket.headers.get("origin")):
            await websocket.close(code=1008)
            return
        try:
            orch.sessions.validate_id(client_sid)
        except InputRejected:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        lock = asyncio.Lock()

        # Browsers cannot set headers on WebSocket upgrades, so the token travels in the first frame
        # (never in the URL, which ends up in proxy and server logs). Unauthenticated sockets are closed
        # before any session is created or any event is streamed.
        principal: Optional[Principal] = None
        if auth.required:
            try:
                first = await asyncio.wait_for(websocket.receive_json(), timeout=10)
                token = first.get("token") if isinstance(first, dict) and first.get("type") == "auth" else None
                principal = await auth.authenticate(token if isinstance(token, str) else None)
            except (AuthRequired, asyncio.TimeoutError, ValueError, WebSocketDisconnect) as e:
                if not isinstance(e, WebSocketDisconnect):
                    try:
                        await websocket.send_json({"type": "error", "message": str(e) or "authentication failed"})
                        await websocket.close(code=4401)
                    except Exception:  # noqa: BLE001
                        pass
                return
        try:
            sid = bind(principal, client_sid)
        except InputRejected:
            await websocket.close(code=1008)
            return

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
            await send({"type": "hello", "state": orch.session_state(sid), "config": describe(),
                        "user": {"user_id": principal.user_id, "email": principal.email} if principal else None})
            while True:
                msg = await websocket.receive_json()
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "message":
                    try:
                        task = await orch.submit(sid, msg.get("text", ""))
                    except BoughError as e:
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
            except BoughError:
                pass

    # static dashboard last, so /api and /ws take precedence
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app
