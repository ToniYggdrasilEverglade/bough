"""ElevenLabs text-to-speech: free-plan defaults, request shape, error hints, and the /api/tts route.
The network is never touched: urllib.request.urlopen is replaced with a fake."""
import asyncio
import io
import json
import os
import tempfile
import urllib.error

import pytest
from fastapi.testclient import TestClient

from backend import tts as tts_mod
from backend.app import create_app
from backend.config import Settings
from backend.errors import InputRejected, TTSUnavailable
from backend.tts import ElevenLabsTTS, speakable


def settings(**kw):
    d = tempfile.mkdtemp()
    base = dict(db_path=os.path.join(d, "t.sqlite3"), tts_backend="elevenlabs", elevenlabs_api_key="sk_test")
    base.update(kw)
    return Settings(**base)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        return FakeResponse(b"ID3fake-mp3")

    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", fake_urlopen)
    return calls


def http_error(code, status):
    body = json.dumps({"detail": {"status": status, "message": "raw"}}).encode()
    return urllib.error.HTTPError("https://api.elevenlabs.io", code, "err", {}, io.BytesIO(body))


def test_defaults_fit_free_plan():
    s = Settings()
    assert s.tts_backend == "off"
    assert s.elevenlabs_model_id == "eleven_flash_v2_5"
    assert s.elevenlabs_output_format == "mp3_44100_128"
    assert s.tts_max_chars <= 1000


def test_paid_formats_refused_unless_allowed():
    with pytest.raises(ValueError):
        settings(elevenlabs_output_format="mp3_44100_192")
    settings(elevenlabs_output_format="mp3_44100_192", elevenlabs_allow_paid_formats=True)


def test_key_required_and_never_public():
    with pytest.raises(ValueError):
        Settings(tts_backend="elevenlabs")
    pub = settings().public_dict()
    assert "elevenlabs_api_key" not in pub and "sk_test" not in json.dumps(pub)
    assert pub["elevenlabs_api_key_configured"] is True
    assert "sk_test" not in repr(settings())


def test_speakable_skips_code():
    out = speakable("## Fix\nUse `items[::-1]`:\n\n```python\nprint('x')\n```\nDone **now**.")
    assert "print" not in out and "code block omitted" in out
    assert "items[::-1]" in out and "**" not in out and "##" not in out


def test_request_shape(captured):
    audio = asyncio.run(ElevenLabsTTS(settings()).synthesize("Hello there"))
    assert audio == b"ID3fake-mp3"
    req = captured[0]
    assert req.full_url == ("https://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb"
                            "?output_format=mp3_44100_128")
    assert req.get_header("Xi-api-key") == "sk_test"
    assert json.loads(req.data) == {"text": "Hello there", "model_id": "eleven_flash_v2_5"}


def test_long_text_clipped(captured):
    asyncio.run(ElevenLabsTTS(settings(tts_max_chars=20)).synthesize("word " * 50))
    assert len(json.loads(captured[0].data)["text"]) <= 20


def test_bad_voice_and_empty_text(captured):
    t = ElevenLabsTTS(settings())
    with pytest.raises(InputRejected):
        asyncio.run(t.synthesize("hi", voice_id="../../v1/user"))
    with pytest.raises(InputRejected):
        asyncio.run(t.synthesize("   \n  "))
    assert not captured


@pytest.mark.parametrize("code,status,needle", [
    (402, "paid_plan_required", "Voice Library"),
    (401, "quota_exceeded", "credits"),
    (401, "", "API key is wrong"),
])
def test_free_plan_errors_explained(monkeypatch, code, status, needle):
    def boom(req, timeout):
        raise http_error(code, status)
    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", boom)
    with pytest.raises(TTSUnavailable) as e:
        asyncio.run(ElevenLabsTTS(settings()).synthesize("hi"))
    assert needle in str(e.value) and "sk_test" not in str(e.value)


def test_route_off_by_default():
    c = TestClient(create_app(Settings(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"))))
    assert c.get("/api/config").json()["tts"] == {"enabled": False}
    r = c.post("/api/tts", json={"text": "hi"})
    assert r.status_code == 503 and "off" in r.json()["error"]


def test_route_returns_audio(captured):
    c = TestClient(create_app(settings()))
    cfg = c.get("/api/config").json()["tts"]
    assert cfg["enabled"] and cfg["attribution"] == "Voice by ElevenLabs"
    r = c.post("/api/tts", json={"text": "Hello"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and r.content == b"ID3fake-mp3"


def test_speakable_groq_reply_shapes():
    reply = ("<think>The user wants a list.</think>\n"
             "See [the docs](https://docs.python.org/3/) or https://example.com/x?y=1.\n\n"
             "| Model | Speed |\n|---|---|\n| flash | fast |\n\n"
             "[reply cut off at the completion-token limit]")
    out = speakable(reply)
    assert "think" not in out.lower() and "user wants" not in out
    assert "the docs" in out and "docs.python.org" not in out and "example.com" not in out
    assert "---" not in out and "|" not in out and "Model, Speed" in out
    assert "cut off" not in out


def test_groq_reply_reaches_elevenlabs(monkeypatch):
    """Full path: Groq chat-completions (faked) -> root pipeline -> reply -> /api/tts -> ElevenLabs (faked)."""
    seen = {}
    groq_reply = "<think>short answer</think>Python lists keep insertion order. See [docs](https://docs.python.org/3/)."

    def fake_urlopen(req, timeout):
        host = req.full_url.split("/")[2]
        seen.setdefault(host, []).append(req)
        if host == "api.groq.com":
            assert req.get_header("Authorization") == "Bearer gsk_test"
            body = {"choices": [{"message": {"content": groq_reply}, "finish_reason": "stop"}]}
            return FakeResponse(json.dumps(body).encode())
        assert host == "api.elevenlabs.io"
        return FakeResponse(b"ID3groq-audio")

    import backend.models.http_model as hm
    monkeypatch.setattr(hm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", fake_urlopen)
    c = TestClient(create_app(settings(model_backend="http", model_name="qwen/qwen3-32b", model_api_key="gsk_test",
                                       model_url="https://api.groq.com/openai/v1/chat/completions")))
    result = c.post("/api/sessions/s1/chat", json={"text": "Do Python lists keep order?"}).json()
    reply = result["final"]["text"]
    assert "insertion order" in reply and seen["api.groq.com"]

    r = c.post("/api/tts", json={"text": reply})
    assert r.status_code == 200 and r.content == b"ID3groq-audio"
    spoken = json.loads(seen["api.elevenlabs.io"][0].data)["text"]
    assert "insertion order" in spoken and "think" not in spoken and "docs.python.org" not in spoken
