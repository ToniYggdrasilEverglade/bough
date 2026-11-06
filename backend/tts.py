"""Optional text-to-speech for replies through the ElevenLabs API.

POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=...
headers: xi-api-key; body: {"text", "model_id"}; response: audio bytes.

The API key lives only in Settings and here. The browser never sees it: it asks the backend for audio
(POST /api/tts) and the backend calls ElevenLabs on its behalf.
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from .config import Settings
from .errors import InputRejected, TTSUnavailable

API_BASE = "https://api.elevenlabs.io/v1/text-to-speech/"
MAX_AUDIO_BYTES = 20_000_000
_FENCE = re.compile(r"```[\w+#-]*[ \t]*\n[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MD_NOISE = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|>\s?)", re.M)
_ID = re.compile(r"^[A-Za-z0-9]{1,64}$")

MEDIA_TYPES = {"mp3": "audio/mpeg", "opus": "audio/ogg", "pcm": "audio/L16", "ulaw": "audio/basic",
               "alaw": "audio/basic", "wav": "audio/wav"}


def speakable(text: str) -> str:
    """Turn a chat reply into something worth reading aloud: code blocks are announced, not read."""
    t = _FENCE.sub(" (code block omitted) ", text or "")
    t = _INLINE_CODE.sub(r"\1", t)
    t = _MD_NOISE.sub("", t)
    t = t.replace("**", "").replace("__", "")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip()


def media_type(output_format: str) -> str:
    return MEDIA_TYPES.get(output_format.split("_", 1)[0], "application/octet-stream")


class ElevenLabsTTS:
    def __init__(self, settings: Settings):
        self._key = settings.elevenlabs_api_key
        self.voice_id = settings.elevenlabs_voice_id
        self.model_id = settings.elevenlabs_model_id
        self.output_format = settings.elevenlabs_output_format
        self.max_chars = settings.tts_max_chars
        self.timeout = settings.tts_timeout_s
        self.media_type = media_type(self.output_format)

    def describe(self) -> dict:
        return {"enabled": True, "provider": "elevenlabs", "attribution": "Voice by ElevenLabs",
                "attribution_url": "https://elevenlabs.io/text-to-speech-api", "voice_id": self.voice_id,
                "model_id": self.model_id, "output_format": self.output_format, "max_chars": self.max_chars}

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        spoken = speakable(text)
        if not spoken:
            raise InputRejected("nothing to read aloud")
        if len(spoken) > self.max_chars:  # clip on a word boundary instead of refusing
            cut = spoken[: self.max_chars]
            spoken = cut[: cut.rfind(" ")] if " " in cut else cut
        voice = voice_id or self.voice_id
        if not _ID.match(voice):
            raise InputRejected("invalid voice id")
        return await asyncio.to_thread(self._post, spoken, voice)

    def _post(self, text: str, voice: str) -> bytes:
        url = API_BASE + voice + "?" + urllib.parse.urlencode({"output_format": self.output_format})
        body = json.dumps({"text": text, "model_id": self.model_id}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "xi-api-key": self._key, "Content-Type": "application/json", "Accept": self.media_type})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                audio = res.read(MAX_AUDIO_BYTES + 1)
        except urllib.error.HTTPError as e:
            status, msg = _detail(e)
            raise TTSUnavailable(_explain(e.code, status, msg)) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise TTSUnavailable(f"ElevenLabs unreachable: {getattr(e, 'reason', e)}") from None
        if len(audio) > MAX_AUDIO_BYTES:
            raise TTSUnavailable("ElevenLabs audio exceeded the size limit")
        if not audio:
            raise TTSUnavailable("ElevenLabs returned no audio")
        return audio


def _detail(e: urllib.error.HTTPError) -> tuple[str, str]:
    """ElevenLabs errors look like {"detail": {"status": "...", "message": "..."}}. Never echo headers."""
    try:
        d = json.loads(e.read(4096) or b"{}").get("detail")
        if isinstance(d, dict):
            return str(d.get("status") or ""), str(d.get("message") or "")[:200]
        return "", str(d or "")[:200]
    except (ValueError, AttributeError, OSError):
        return "", ""


# Free-plan problems get a plain-English fix instead of a bare status code.
_HINTS = {
    "paid_plan_required": "this voice needs a paid ElevenLabs plan. Voice Library voices are blocked for free API "
                          "users; set BOUGH_ELEVENLABS_VOICE_ID to a default voice or one you designed/cloned",
    "quota_exceeded": "your ElevenLabs credits for this month are used up (free plan: 10,000 credits, about "
                      "20,000 characters with eleven_flash_v2_5). They reset at the start of your billing cycle",
    "voice_not_found": "voice not found. Default voices are retired on 2026-12-31; pick a new voice in "
                       "My Voices and set BOUGH_ELEVENLABS_VOICE_ID",
    "missing_permissions": "the API key lacks the Text to Speech permission",
    "invalid_api_key": "the ElevenLabs API key is wrong",
    "too_many_concurrent_requests": "too many requests at once for your plan; try again in a moment",
}


def _explain(code: int, status: str, msg: str) -> str:
    hint = _HINTS.get(status) or {401: _HINTS["invalid_api_key"], 402: _HINTS["paid_plan_required"],
                                  429: _HINTS["too_many_concurrent_requests"]}.get(code)
    return f"ElevenLabs HTTP {code}: " + (hint or msg or "request failed")


def build_tts(settings: Settings) -> Optional[ElevenLabsTTS]:
    return ElevenLabsTTS(settings) if settings.tts_backend == "elevenlabs" else None
