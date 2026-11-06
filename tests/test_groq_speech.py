"""Groq speech: Orpheus TTS (200-char pieces joined into one WAV) and Whisper transcription for the mic.
The network is never touched: urllib.request.urlopen is replaced with fakes."""
import asyncio
import io
import json
import os
import struct
import tempfile
import urllib.error

import pytest
from fastapi.testclient import TestClient

from backend import stt as stt_mod
from backend import tts as tts_mod
from backend.app import create_app
from backend.config import Settings
from backend.errors import InputRejected, STTUnavailable, TTSUnavailable
from backend.stt import GroqSTT
from backend.tts import GroqTTS, join_wavs, split_for_groq

GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"


def settings(**kw):
    base = dict(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"), groq_api_key="gsk_test")
    base.update(kw)
    return Settings(**base)


def wav(pcm: bytes, streamed=False) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
    size = 0xFFFFFFFF if streamed else len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"LIST" + struct.pack("<I", 4) + b"INFO" + b"data" + struct.pack("<I", size) + pcm)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body):
    return urllib.error.HTTPError("https://api.groq.com", code, "err", {}, io.BytesIO(json.dumps(body).encode()))


# ------------------------------------------------------------------ config
def test_groq_key_falls_back_to_model_key_on_groq():
    s = Settings(db_path="x", model_backend="http", model_url=GROQ_CHAT, model_api_key="gsk_model", tts_backend="groq")
    assert s.groq_key() == "gsk_model"
    assert Settings(db_path="x", model_url="http://127.0.0.1:11434/v1/chat/completions",
                    model_api_key="other").groq_key() == ""
    with pytest.raises(ValueError):
        Settings(db_path="x", tts_backend="groq")
    with pytest.raises(ValueError):
        Settings(db_path="x", stt_backend="groq")


def test_groq_key_never_public():
    s = settings(tts_backend="groq", stt_backend="groq")
    pub = json.dumps(s.public_dict())
    assert "gsk_test" not in pub and "gsk_test" not in repr(s)
    assert s.public_dict()["groq_api_key_configured"] is True


# ------------------------------------------------------------------ chunking + WAV
def test_split_respects_200_chars():
    text = ("This is a sentence that keeps going. " * 20) + ("x" * 450)
    pieces = split_for_groq(text)
    assert all(0 < len(p) <= 200 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_join_wavs_concatenates_pcm():
    joined = join_wavs([wav(b"\x01\x00" * 3), wav(b"\x02\x00" * 2, streamed=True)])
    assert joined[:4] == b"RIFF" and joined[8:12] == b"WAVE"
    data_at = joined.index(b"data")
    size = struct.unpack("<I", joined[data_at + 4:data_at + 8])[0]
    assert size == 10 and joined[data_at + 8:] == b"\x01\x00" * 3 + b"\x02\x00" * 2
    assert struct.unpack("<I", joined[4:8])[0] == len(joined) - 8


def test_join_rejects_non_wav():
    with pytest.raises(TTSUnavailable):
        join_wavs([b"ID3mp3", b"ID3mp3"])


# ------------------------------------------------------------------ Groq TTS
def test_groq_tts_requests(monkeypatch):
    calls = []

    def fake(req, timeout):
        calls.append(req)
        return FakeResponse(wav(b"\x05\x00"))

    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", fake)
    t = GroqTTS(settings(tts_backend="groq", tts_max_chars=1000))
    audio = asyncio.run(t.synthesize("Hello there. " * 40))
    assert audio[:4] == b"RIFF" and len(calls) >= 3
    body = json.loads(calls[0].data)
    assert calls[0].full_url == "https://api.groq.com/openai/v1/audio/speech"
    assert calls[0].get_header("Authorization") == "Bearer gsk_test"
    assert body["model"] == "canopylabs/orpheus-v1-english" and body["voice"] == "autumn"
    assert body["response_format"] == "wav" and all(len(json.loads(c.data)["input"]) <= 200 for c in calls)


def test_groq_tts_rejects_bad_voice(monkeypatch):
    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(InputRejected):
        asyncio.run(GroqTTS(settings(tts_backend="groq")).synthesize("hi", voice_id="../x"))


@pytest.mark.parametrize("code,body,needle", [
    (401, {"error": {"message": "Invalid API Key"}}, "key is wrong"),
    (400, {"error": {"message": "The model requires terms acceptance", "code": "model_terms_required"}}, "terms"),
    (429, {"error": {"message": "Rate limit reached"}}, "rate limit"),
])
def test_groq_tts_errors_explained(monkeypatch, code, body, needle):
    def boom(req, timeout):
        raise http_error(code, body)
    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", boom)
    with pytest.raises(TTSUnavailable) as e:
        asyncio.run(GroqTTS(settings(tts_backend="groq")).synthesize("hi"))
    assert needle in str(e.value) and "gsk_test" not in str(e.value)


# ------------------------------------------------------------------ Groq Whisper
def test_groq_stt_multipart(monkeypatch):
    calls = []

    def fake(req, timeout):
        calls.append(req)
        return FakeResponse(json.dumps({"text": "  what time is it  "}).encode())

    monkeypatch.setattr(stt_mod.urllib.request, "urlopen", fake)
    text = asyncio.run(GroqSTT(settings(stt_backend="groq")).transcribe(b"OggS-audio", "audio/webm;codecs=opus"))
    assert text == "what time is it"
    req = calls[0]
    assert req.full_url == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert req.get_header("Content-type").startswith("multipart/form-data; boundary=")
    body = req.data
    assert b'name="model"\r\n\r\nwhisper-large-v3-turbo' in body and b'name="language"\r\n\r\nen' in body
    assert b'filename="speech.webm"' in body and b"OggS-audio" in body


def test_groq_stt_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(stt_mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    s = GroqSTT(settings(stt_backend="groq", stt_max_bytes=10))
    with pytest.raises(InputRejected):
        asyncio.run(s.transcribe(b"x", "text/plain"))
    with pytest.raises(InputRejected):
        asyncio.run(s.transcribe(b"", "audio/webm"))
    with pytest.raises(InputRejected):
        asyncio.run(s.transcribe(b"x" * 11, "audio/webm"))


def test_groq_stt_error(monkeypatch):
    def boom(req, timeout):
        raise http_error(413, {"error": {"message": "too big"}})
    monkeypatch.setattr(stt_mod.urllib.request, "urlopen", boom)
    with pytest.raises(STTUnavailable) as e:
        asyncio.run(GroqSTT(settings(stt_backend="groq")).transcribe(b"x", "audio/ogg"))
    assert "25 MB" in str(e.value)


# ------------------------------------------------------------------ routes, full voice loop on one Groq key
def test_voice_loop_on_one_groq_key(monkeypatch):
    """Mic audio -> Whisper -> Groq chat model -> reply -> Orpheus voice, all with BOUGH_MODEL_API_KEY."""
    seen = []

    def fake(req, timeout):
        seen.append(req.full_url)
        assert req.get_header("Authorization") == "Bearer gsk_model"
        if req.full_url.endswith("/audio/transcriptions"):
            return FakeResponse(json.dumps({"text": "Do Python lists keep order?"}).encode())
        if req.full_url.endswith("/chat/completions"):
            msg = {"content": "<think>easy</think>Yes. Python lists keep insertion order.", "role": "assistant"}
            return FakeResponse(json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]}).encode())
        assert req.full_url.endswith("/audio/speech")
        assert "think" not in json.loads(req.data)["input"]
        return FakeResponse(wav(b"\x07\x00"))

    import backend.models.http_model as hm
    for mod in (hm, tts_mod, stt_mod):
        monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
    s = Settings(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"), model_backend="http", model_url=GROQ_CHAT,
                 model_name="qwen/qwen3-32b", model_api_key="gsk_model", tts_backend="groq", stt_backend="groq")
    c = TestClient(create_app(s))
    cfg = c.get("/api/config").json()
    assert cfg["tts"]["provider"] == "groq" and cfg["stt"]["enabled"] and cfg["tts"]["attribution"] is None

    heard = c.post("/api/transcribe", content=b"webm-bytes", headers={"Content-Type": "audio/webm"}).json()
    assert heard == {"text": "Do Python lists keep order?"}
    reply = c.post("/api/sessions/s1/chat", json={"text": heard["text"]}).json()["final"]["text"]
    assert "insertion order" in reply
    r = c.post("/api/tts", json={"text": reply})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    assert [u.rsplit("/", 1)[1] for u in seen] == ["transcriptions", "completions", "speech"]


def test_transcribe_off_by_default():
    c = TestClient(create_app(Settings(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"))))
    assert c.get("/api/config").json()["stt"] == {"enabled": False}
    r = c.post("/api/transcribe", content=b"x", headers={"Content-Type": "audio/webm"})
    assert r.status_code == 503 and "BOUGH_STT_BACKEND" in r.json()["error"]
