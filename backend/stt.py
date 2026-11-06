"""Optional speech-to-text for the mic button through Groq Whisper.

POST https://api.groq.com/openai/v1/audio/transcriptions (multipart: file, model, language, response_format=json)
The browser uploads its recording to POST /api/transcribe; the backend forwards it with the Groq key, which
never reaches the browser. Groq bills a minimum of 10 seconds per request.
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
import uuid
from typing import Optional

from .config import Settings
from .errors import InputRejected, STTUnavailable
from .tts import groq_error

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# Browser MediaRecorder types -> file extension Groq accepts (flac mp3 mp4 mpeg mpga m4a ogg wav webm)
EXTENSIONS = {"audio/webm": "webm", "video/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a",
              "audio/x-m4a": "m4a", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
              "audio/wave": "wav", "audio/flac": "flac"}


def _base_type(content_type: Optional[str]) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


class GroqSTT:
    def __init__(self, settings: Settings):
        self._key = settings.groq_key()
        self.model_id = settings.groq_stt_model
        self.language = settings.stt_language
        self.max_bytes = settings.stt_max_bytes
        self.timeout = max(settings.tts_timeout_s, 30.0)

    def describe(self) -> dict:
        return {"enabled": True, "provider": "groq", "model_id": self.model_id, "language": self.language or None,
                "max_bytes": self.max_bytes}

    async def transcribe(self, audio: bytes, content_type: Optional[str]) -> str:
        ext = EXTENSIONS.get(_base_type(content_type))
        if ext is None:
            raise InputRejected(f"unsupported audio type {_base_type(content_type) or '(none)'}")
        if not audio:
            raise InputRejected("empty recording")
        if len(audio) > self.max_bytes:
            raise InputRejected(f"recording is larger than {self.max_bytes // 1_000_000} MB")
        text = await asyncio.to_thread(self._post, audio, ext, _base_type(content_type))
        return re.sub(r"\s+", " ", text).strip()

    def _post(self, audio: bytes, ext: str, mime: str) -> str:
        boundary = "bough-" + uuid.uuid4().hex
        fields = {"model": self.model_id, "response_format": "json", "temperature": "0"}
        if self.language:
            fields["language"] = self.language
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in fields.items()]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.{ext}"\r\n'
                     f"Content-Type: {mime}\r\n\r\n".encode() + audio + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(GROQ_TRANSCRIBE_URL, data=b"".join(parts), method="POST", headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json", "User-Agent": "Bough/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                raw = res.read(1_000_000)
        except urllib.error.HTTPError as e:
            raise STTUnavailable(groq_error(e, "transcription")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise STTUnavailable(f"Groq unreachable: {getattr(e, 'reason', e)}") from None
        try:
            text = json.loads(raw)["text"]
        except (ValueError, KeyError, TypeError):
            raise STTUnavailable("unexpected response from Groq transcription") from None
        if not isinstance(text, str):
            raise STTUnavailable("unexpected response from Groq transcription")
        return text


def build_stt(settings: Settings) -> Optional[GroqSTT]:
    return GroqSTT(settings) if settings.stt_backend == "groq" else None
