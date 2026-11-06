"""Optional text-to-speech for replies: ElevenLabs or Groq (Orpheus).

ElevenLabs: POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=...
            headers: xi-api-key; body: {"text", "model_id"}; response: audio bytes.
Groq:       POST https://api.groq.com/openai/v1/audio/speech
            headers: Authorization: Bearer; body: {"model", "input" (<= 200 chars), "voice", "response_format": "wav"}.
            Longer replies are split into <= 200-char pieces and the WAVs are joined into one file.

API keys live only in Settings and here. The browser never sees them: it asks the backend for audio
(POST /api/tts) and the backend calls the provider on its behalf.
"""
from __future__ import annotations

import asyncio
import json
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Union

from .config import Settings
from .errors import InputRejected, TTSUnavailable

API_BASE = "https://api.elevenlabs.io/v1/text-to-speech/"
MAX_AUDIO_BYTES = 20_000_000
_FENCE = re.compile(r"```[\w+#-]*[ \t]*\n[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MD_NOISE = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|>\s?)", re.M)
_THINK = re.compile(r"<think>[\s\S]*?(</think>|$)\s*", re.I)   # Qwen3/DeepSeek reasoning on Groq
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")       # [text](url) -> text
_URL = re.compile(r"https?://\S+")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", re.M)
_CUT_NOTE = "[reply cut off at the completion-token limit]"
_ID = re.compile(r"^[A-Za-z0-9]{1,64}$")

MEDIA_TYPES = {"mp3": "audio/mpeg", "opus": "audio/ogg", "pcm": "audio/L16", "ulaw": "audio/basic",
               "alaw": "audio/basic", "wav": "audio/wav"}


def speakable(text: str) -> str:
    """Turn a chat reply into something worth reading aloud: code blocks are announced, not read."""
    t = _THINK.sub("", text or "").replace(_CUT_NOTE, "")
    t = _FENCE.sub(" (code block omitted) ", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _URL.sub("(link)", t)
    t = _TABLE_RULE.sub("", t)
    t = re.sub(r"\s*\|\s*", ", ", t)                                       # table cells read as a list
    t = _INLINE_CODE.sub(r"\1", t)
    t = _MD_NOISE.sub("", t)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"^[, ]+|[, ]+$", "", t, flags=re.M)
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


# ---------------------------------------------------------------------------------------------- Groq
GROQ_SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"
GROQ_CHUNK_CHARS = 200          # Orpheus hard limit per request
GROQ_PARALLEL = 2               # concurrent piece requests; keeps free-tier rate limits happy
_GROQ_VOICE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_for_groq(text: str, limit: int = GROQ_CHUNK_CHARS) -> List[str]:
    """Pack sentences into pieces of <= limit chars; overlong sentences break at commas, then spaces."""
    pieces: List[str] = []
    for sentence in (x.strip() for x in _SENTENCE.split(text)):
        while len(sentence) > limit:
            cut = max(sentence.rfind(", ", 0, limit), sentence.rfind(" ", 0, limit))
            cut = cut + 1 if cut > 0 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    packed: List[str] = []
    for p in pieces:
        if packed and len(packed[-1]) + 1 + len(p) <= limit:
            packed[-1] += " " + p
        else:
            packed.append(p)
    return packed


def _wav_parts(wav: bytes):
    """Return (fmt chunk body, PCM data) from a RIFF/WAVE file. Tolerates streamed size fields (0xFFFFFFFF)."""
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise TTSUnavailable("Groq returned audio that is not WAV")
    pos, fmt = 12, None
    while pos + 8 <= len(wav):
        cid, size = wav[pos:pos + 4], struct.unpack("<I", wav[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = wav[body:body + size]
        elif cid == b"data":
            end = len(wav) if size in (0, 0xFFFFFFFF) or body + size > len(wav) else body + size
            if fmt is None:
                break
            return fmt, wav[body:end]
        pos = body + size + (size & 1)
    raise TTSUnavailable("Groq WAV audio had no fmt/data chunk")


def join_wavs(wavs: List[bytes]) -> bytes:
    if len(wavs) == 1:
        return wavs[0]
    fmt, chunks = None, []
    for w in wavs:
        f, data = _wav_parts(w)
        if fmt is None:
            fmt = f
        elif f[:16] != fmt[:16]:
            raise TTSUnavailable("Groq audio pieces had different formats")
        chunks.append(data)
    data = b"".join(chunks)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data)


def groq_error(e: urllib.error.HTTPError, what: str) -> str:
    """Groq errors look like {"error": {"message", "type", "code"}}. Turn the common ones into fixes."""
    code, msg = "", ""
    try:
        err = json.loads(e.read(4096) or b"{}").get("error") or {}
        code, msg = str(err.get("code") or ""), str(err.get("message") or "")[:300]
    except (ValueError, AttributeError, OSError):
        pass
    if e.code == 401:
        hint = "the Groq API key is wrong (GROQ_API_KEY or BOUGH_MODEL_API_KEY)"
    elif code == "model_terms_required" or "terms" in msg.lower():
        hint = ("accept this model's terms once in the Groq console (open the model in the Playground at "
                "https://console.groq.com/playground and accept), then try again")
    elif e.code == 429:
        hint = "Groq rate limit reached for your plan; wait a moment and try again" + (f" ({msg})" if msg else "")
    elif e.code == 413:
        hint = "the recording is too large for Groq (free tier: 25 MB)"
    else:
        hint = msg or "request failed"
    return f"Groq {what} HTTP {e.code}: {hint}"


class GroqTTS:
    media_type = "audio/wav"

    def __init__(self, settings: Settings):
        self._key = settings.groq_key()
        self.model_id = settings.groq_tts_model
        self.voice_id = settings.groq_tts_voice
        self.max_chars = settings.tts_max_chars
        self.timeout = settings.tts_timeout_s

    def describe(self) -> dict:
        return {"enabled": True, "provider": "groq", "attribution": None, "attribution_url": None,
                "voice_id": self.voice_id, "model_id": self.model_id, "output_format": "wav",
                "max_chars": self.max_chars}

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        spoken = speakable(text)
        if not spoken:
            raise InputRejected("nothing to read aloud")
        if len(spoken) > self.max_chars:
            cut = spoken[: self.max_chars]
            spoken = cut[: cut.rfind(" ")] if " " in cut else cut
        voice = (voice_id or self.voice_id).lower()
        if not _GROQ_VOICE.match(voice):
            raise InputRejected("invalid voice id")
        pieces = split_for_groq(spoken)
        gate = asyncio.Semaphore(GROQ_PARALLEL)

        async def one(piece: str) -> bytes:
            async with gate:
                return await asyncio.to_thread(self._post, piece, voice)

        return join_wavs(list(await asyncio.gather(*(one(p) for p in pieces))))

    def _post(self, text: str, voice: str) -> bytes:
        body = json.dumps({"model": self.model_id, "input": text, "voice": voice, "response_format": "wav"}).encode()
        req = urllib.request.Request(GROQ_SPEECH_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "Accept": "audio/wav",
            "User-Agent": "Bough/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                audio = res.read(MAX_AUDIO_BYTES + 1)
        except urllib.error.HTTPError as e:
            raise TTSUnavailable(groq_error(e, "speech")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise TTSUnavailable(f"Groq unreachable: {getattr(e, 'reason', e)}") from None
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            raise TTSUnavailable("Groq returned no audio or too much audio")
        return audio


TTS = Union[ElevenLabsTTS, GroqTTS]


def build_tts(settings: Settings) -> Optional[TTS]:
    if settings.tts_backend == "elevenlabs":
        return ElevenLabsTTS(settings)
    if settings.tts_backend == "groq":
        return GroqTTS(settings)
    return None
