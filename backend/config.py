"""Configuration. Everything comes from environment variables (optionally a .env file).

Secrets (the model and ElevenLabs API keys) live only here and in the HTTP adapter. `public_dict()` is the only
representation that is ever sent to the browser, and it never contains the key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from urllib.parse import urlsplit, urlunsplit

try:  # optional convenience
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# ElevenLabs output formats the free plan cannot use (MP3 192 kbps: Creator+; 44.1 kHz PCM/WAV: Pro+).
# Set BOUGH_ELEVENLABS_ALLOW_PAID_FORMATS=true if you upgrade.
PAID_ONLY_FORMATS = {"mp3_44100_192", "pcm_44100", "wav_44100", "pcm_48000", "wav_48000"}


def redact_url(url: str) -> str:
    """Strip credentials and query strings from a URL so it is safe to display."""
    try:
        p = urlsplit(url)
        netloc = p.netloc
        if p.username or p.password:
            host = p.hostname or ""
            if p.port:
                host += f":{p.port}"
            netloc = f"***@{host}"
        return urlunsplit((p.scheme, netloc, p.path, "", ""))
    except ValueError:
        return "<invalid-url>"


@dataclass(frozen=True)
class Settings:
    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000
    allow_non_local_bind: bool = False
    # --- storage ---
    memory_backend: str = "sqlite"         # "sqlite" | "supabase" (supabase requires auth_mode=supabase)
    db_path: str = "data/bough.sqlite3"
    # --- auth (Supabase email+password; JWTs verified against the project's JWKS) ---
    auth_mode: str = "off"                 # "off" (single local user, loopback only) | "supabase"
    supabase_url: str = ""                 # https://<ref>.supabase.co
    supabase_publishable_key: str = ""     # public by design (ships to the browser); RLS guards the data
    jwks_cache_s: float = 3600.0
    # --- pipeline ---
    pipeline: str = "root"                 # "root": model answers freely, agents filter (Bough) | "proposals": council flow
    root_repair_enabled: bool = True       # allow ONE bounded model rewrite when branch checks fail
    # --- model adapter ---
    model_backend: str = "mock"            # "mock" | "http"
    model_url: str = field(default="http://127.0.0.1:11434/v1/chat/completions", repr=False)  # may embed credentials
    model_name: str = "llama3.2"
    model_api_key: str = field(default="", repr=False)   # SECRET: never exposed, not even in repr()
    model_json_mode: bool = True
    model_plan_tokens: int = 1500          # completion-token caps per task (256 truncated every real answer)
    model_review_tokens: int = 600
    model_reply_tokens: int = 6000
    max_reply_chars: int = 16000           # final reply clip; code answers need far more than the old 4000
    model_timeout_s: float = 20.0
    # "auto": when the user explicitly asks for the web ("use the web", "search online", ...) and the model is a
    # Groq gpt-oss model, enable Groq's server-side browser_search for that reply. "off": never search.
    model_web_search: str = "auto"
    model_web_search_timeout_s: float = 60.0
    mock_latency_s: float = 0.0
    # --- web fetching (trusted server configuration only) ---
    web_fetch_enabled: bool = False
    web_fetch_executable: str = field(default="", repr=False)
    web_fetch_wrapper: str = field(default="", repr=False)
    web_fetch_allowed_hosts: tuple[str, ...] = ()
    web_fetch_context_chars: int = 6000    # page text handed to the language model (the CLI returns up to 64 KiB)
    # --- text-to-speech (optional; ElevenLabs) ---
    tts_backend: str = "off"               # "off" | "elevenlabs" | "groq"
    elevenlabs_api_key: str = field(default="", repr=False)  # SECRET: never exposed, not even in repr()
    # Defaults are chosen for the ElevenLabs FREE plan (10,000 credits/month, non-commercial, attribution):
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"       # "George", a default voice (free tier OK; retired 2026-12-31)
    elevenlabs_model_id: str = "eleven_flash_v2_5"          # 0.5 credit/char -> ~20,000 chars/month on free
    elevenlabs_output_format: str = "mp3_44100_128"         # highest format on free (192 kbps / 44.1 kHz PCM are paid)
    elevenlabs_allow_paid_formats: bool = False  # set true after upgrading to use 192 kbps MP3 / 44.1 kHz PCM
    # Groq speech (Orpheus TTS + Whisper STT). Key: GROQ_API_KEY, else BOUGH_MODEL_API_KEY when the model is on Groq.
    groq_api_key: str = field(default="", repr=False)        # SECRET
    groq_tts_model: str = "canopylabs/orpheus-v1-english"     # English only; 200 chars per request, WAV only
    groq_tts_voice: str = "autumn"                           # autumn, diana, hannah, austin, daniel, troy
    stt_backend: str = "off"               # "off" | "groq": mic button -> Groq Whisper -> text
    groq_stt_model: str = "whisper-large-v3-turbo"
    stt_language: str = "en"               # ISO-639-1 hint for Whisper; empty = auto-detect
    stt_max_bytes: int = 25_000_000        # Groq free tier upload limit
    tts_max_chars: int = 1000              # per reply; longer replies are clipped (ElevenLabs bills per character)
    tts_timeout_s: float = 30.0
    # --- limits ---
    max_model_calls: int = 3               # per cycle, retries included
    cycle_timeout_s: float = 30.0
    max_proposals: int = 4
    workspace_max_items: int = 40
    max_retrieved_memories: int = 5
    max_retries: int = 1                   # extra attempts after a malformed/timeout model result
    max_input_chars: int = 2000
    max_sessions: int = 50
    max_events_per_cycle: int = 200
    max_events_per_source_type: int = 10
    max_memory_records: int = 1000
    # --- memory / attention ---
    memory_min_score: float = 0.15
    include_model_interpretations: bool = False
    seed_enabled: bool = True              # built-in, read-only knowledge seeds (backend/memory/seeds/*.json)
    max_seed_memories: int = 4             # separate budget: seeds never crowd out the user's own memories
    # --- evaluation / selection ---
    enable_model_review: bool = False
    w_relevance: float = 0.35
    w_constraints: float = 0.20
    w_support: float = 0.25
    w_safety: float = 0.20
    min_select_score: float = 0.35
    # --- adaptation ---
    bias_step: float = 0.03
    bias_limit: float = 0.15

    def __post_init__(self):
        if self.model_backend not in ("mock", "http"):
            raise ValueError("BOUGH_MODEL_BACKEND must be 'mock' or 'http'")
        if self.pipeline not in ("root", "proposals"):
            raise ValueError("BOUGH_PIPELINE must be 'root' or 'proposals'")
        if urlsplit(self.model_url).scheme not in ("http", "https"):
            raise ValueError("BOUGH_MODEL_URL must start with http:// or https://")
        for name in ("max_model_calls", "max_proposals", "workspace_max_items",
                     "max_retrieved_memories", "max_input_chars", "max_sessions",
                     "max_events_per_cycle", "max_events_per_source_type", "max_memory_records",
                     "web_fetch_context_chars", "max_seed_memories", "model_plan_tokens",
                     "model_review_tokens", "model_reply_tokens", "max_reply_chars"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.cycle_timeout_s <= 0 or self.model_timeout_s <= 0:
            raise ValueError("timeouts must be > 0")
        if not (0 < self.bias_limit <= 0.5) or not (0 < self.bias_step <= self.bias_limit):
            raise ValueError("bias_step must be in (0, bias_limit] and bias_limit in (0, 0.5]")
        if self.auth_mode not in ("off", "supabase"):
            raise ValueError("BOUGH_AUTH_MODE must be 'off' or 'supabase'")
        if self.memory_backend not in ("sqlite", "supabase"):
            raise ValueError("BOUGH_MEMORY_BACKEND must be 'sqlite' or 'supabase'")
        if self.auth_mode == "supabase" or self.memory_backend == "supabase":
            u = urlsplit(self.supabase_url)
            if u.scheme != "https" or not u.netloc or u.path not in ("", "/"):
                raise ValueError("BOUGH_SUPABASE_URL must look like https://<ref>.supabase.co")
            if not self.supabase_publishable_key:
                raise ValueError("BOUGH_SUPABASE_PUBLISHABLE_KEY is required")
        if self.memory_backend == "supabase" and self.auth_mode != "supabase":
            raise ValueError("BOUGH_MEMORY_BACKEND=supabase needs BOUGH_AUTH_MODE=supabase (rows are per signed-in user)")
        if self.model_web_search not in ("auto", "off"):
            raise ValueError("BOUGH_MODEL_WEB_SEARCH must be 'auto' or 'off'")
        if self.model_web_search_timeout_s <= 0:
            raise ValueError("model_web_search_timeout_s must be > 0")
        if self.tts_backend not in ("off", "elevenlabs", "groq"):
            raise ValueError("BOUGH_TTS_BACKEND must be 'off', 'elevenlabs' or 'groq'")
        if self.stt_backend not in ("off", "groq"):
            raise ValueError("BOUGH_STT_BACKEND must be 'off' or 'groq'")
        if (self.tts_backend == "groq" or self.stt_backend == "groq") and not self.groq_key():
            raise ValueError("Groq speech needs GROQ_API_KEY (or BOUGH_MODEL_API_KEY with a Groq BOUGH_MODEL_URL)")
        if not (1 <= self.stt_max_bytes <= 100_000_000):
            raise ValueError("stt_max_bytes must be between 1 and 100000000")
        if self.tts_backend == "elevenlabs" and not self.elevenlabs_api_key:
            raise ValueError("BOUGH_TTS_BACKEND=elevenlabs needs ELEVENLABS_API_KEY")
        if not (1 <= self.tts_max_chars <= 10000):
            raise ValueError("tts_max_chars must be between 1 and 10000")
        if self.elevenlabs_output_format in PAID_ONLY_FORMATS and not self.elevenlabs_allow_paid_formats:
            raise ValueError(f"{self.elevenlabs_output_format} needs a paid ElevenLabs plan; use mp3_44100_128")
        if self.tts_timeout_s <= 0:
            raise ValueError("tts_timeout_s must be > 0")
        if self.jwks_cache_s <= 0:
            raise ValueError("jwks_cache_s must be > 0")

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env=None, load_env_file: bool = True) -> "Settings":
        if load_env_file and load_dotenv is not None and env is None:
            load_dotenv()
        e = os.environ if env is None else env

        def s(name, default):
            return e.get(name, default)

        def i(name, default):
            return int(e.get(name, default))

        def f(name, default):
            return float(e.get(name, default))

        def b(name, default):
            return str(e.get(name, default)).strip().lower() in ("1", "true", "yes", "on")

        d = cls()
        return cls(
            host=s("BOUGH_HOST", d.host), port=i("BOUGH_PORT", d.port),
            allow_non_local_bind=b("BOUGH_ALLOW_NON_LOCAL_BIND", d.allow_non_local_bind),
            db_path=s("BOUGH_DB_PATH", d.db_path),
            memory_backend=s("BOUGH_MEMORY_BACKEND", d.memory_backend).lower(),
            auth_mode=s("BOUGH_AUTH_MODE", d.auth_mode).lower(),
            supabase_url=s("BOUGH_SUPABASE_URL", d.supabase_url).rstrip("/"),
            supabase_publishable_key=s("BOUGH_SUPABASE_PUBLISHABLE_KEY", d.supabase_publishable_key),
            jwks_cache_s=f("BOUGH_JWKS_CACHE_S", d.jwks_cache_s),
            pipeline=s("BOUGH_PIPELINE", d.pipeline).lower(),
            root_repair_enabled=b("BOUGH_ROOT_REPAIR", d.root_repair_enabled),
            model_backend=s("BOUGH_MODEL_BACKEND", d.model_backend).lower(),
            model_url=s("BOUGH_MODEL_URL", d.model_url), model_name=s("BOUGH_MODEL_NAME", d.model_name),
            model_api_key=s("BOUGH_MODEL_API_KEY", d.model_api_key),
            model_json_mode=b("BOUGH_MODEL_JSON_MODE", d.model_json_mode),
            model_timeout_s=f("BOUGH_MODEL_TIMEOUT_S", d.model_timeout_s),
            model_web_search=s("BOUGH_MODEL_WEB_SEARCH", d.model_web_search).lower(),
            model_web_search_timeout_s=f("BOUGH_MODEL_WEB_SEARCH_TIMEOUT_S", d.model_web_search_timeout_s),
            web_fetch_enabled=b("BOUGH_WEB_FETCH_ENABLED", d.web_fetch_enabled),
            web_fetch_executable=s("BOUGH_WEB_FETCH_EXECUTABLE", d.web_fetch_executable),
            web_fetch_wrapper=s("BOUGH_WEB_FETCH_WRAPPER", d.web_fetch_wrapper),
            web_fetch_allowed_hosts=tuple(x.strip() for x in s("BOUGH_WEB_FETCH_ALLOWED_HOSTS", "").split(",") if x.strip()),
            web_fetch_context_chars=i("BOUGH_WEB_FETCH_CONTEXT_CHARS", d.web_fetch_context_chars),
            tts_backend=s("BOUGH_TTS_BACKEND", d.tts_backend).lower(),
            elevenlabs_api_key=s("ELEVENLABS_API_KEY", d.elevenlabs_api_key),
            elevenlabs_voice_id=s("BOUGH_ELEVENLABS_VOICE_ID", d.elevenlabs_voice_id),
            elevenlabs_model_id=s("BOUGH_ELEVENLABS_MODEL_ID", d.elevenlabs_model_id),
            elevenlabs_output_format=s("BOUGH_ELEVENLABS_OUTPUT_FORMAT", d.elevenlabs_output_format),
            elevenlabs_allow_paid_formats=b("BOUGH_ELEVENLABS_ALLOW_PAID_FORMATS", d.elevenlabs_allow_paid_formats),
            groq_api_key=s("GROQ_API_KEY", d.groq_api_key),
            groq_tts_model=s("BOUGH_GROQ_TTS_MODEL", d.groq_tts_model),
            groq_tts_voice=s("BOUGH_GROQ_TTS_VOICE", d.groq_tts_voice).lower(),
            stt_backend=s("BOUGH_STT_BACKEND", d.stt_backend).lower(),
            groq_stt_model=s("BOUGH_GROQ_STT_MODEL", d.groq_stt_model),
            stt_language=s("BOUGH_STT_LANGUAGE", d.stt_language).strip().lower(),
            stt_max_bytes=i("BOUGH_STT_MAX_BYTES", d.stt_max_bytes),
            tts_max_chars=i("BOUGH_TTS_MAX_CHARS", d.tts_max_chars),
            tts_timeout_s=f("BOUGH_TTS_TIMEOUT_S", d.tts_timeout_s),
            mock_latency_s=f("BOUGH_MOCK_LATENCY_S", d.mock_latency_s),
            max_model_calls=i("BOUGH_MAX_MODEL_CALLS", d.max_model_calls),
            cycle_timeout_s=f("BOUGH_CYCLE_TIMEOUT_S", d.cycle_timeout_s),
            max_proposals=i("BOUGH_MAX_PROPOSALS", d.max_proposals),
            workspace_max_items=i("BOUGH_WORKSPACE_MAX_ITEMS", d.workspace_max_items),
            max_retrieved_memories=i("BOUGH_MAX_RETRIEVED_MEMORIES", d.max_retrieved_memories),
            max_retries=i("BOUGH_MAX_RETRIES", d.max_retries),
            max_input_chars=i("BOUGH_MAX_INPUT_CHARS", d.max_input_chars),
            max_sessions=i("BOUGH_MAX_SESSIONS", d.max_sessions),
            max_memory_records=i("BOUGH_MAX_MEMORY_RECORDS", d.max_memory_records),
            memory_min_score=f("BOUGH_MEMORY_MIN_SCORE", d.memory_min_score),
            seed_enabled=b("BOUGH_SEED_ENABLED", d.seed_enabled),
            max_seed_memories=i("BOUGH_MAX_SEED_MEMORIES", d.max_seed_memories),
            model_plan_tokens=i("BOUGH_MODEL_PLAN_TOKENS", d.model_plan_tokens),
            model_review_tokens=i("BOUGH_MODEL_REVIEW_TOKENS", d.model_review_tokens),
            model_reply_tokens=i("BOUGH_MODEL_REPLY_TOKENS", d.model_reply_tokens),
            max_reply_chars=i("BOUGH_MAX_REPLY_CHARS", d.max_reply_chars),
            include_model_interpretations=b("BOUGH_INCLUDE_MODEL_INTERPRETATIONS", d.include_model_interpretations),
            enable_model_review=b("BOUGH_ENABLE_MODEL_REVIEW", d.enable_model_review),
            min_select_score=f("BOUGH_MIN_SELECT_SCORE", d.min_select_score),
            bias_step=f("BOUGH_BIAS_STEP", d.bias_step), bias_limit=f("BOUGH_BIAS_LIMIT", d.bias_limit),
        )

    def groq_key(self) -> str:
        """GROQ_API_KEY, or the model key when the chat model itself runs on Groq (one key for everything)."""
        if self.groq_api_key:
            return self.groq_api_key
        return self.model_api_key if urlsplit(self.model_url).hostname == "api.groq.com" else ""

    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS

    def public_dict(self) -> dict:
        """The ONLY settings view that may reach the browser. No secrets, no credentials in URLs."""
        out = {}
        for fld in fields(self):
            if fld.name in ("model_api_key", "model_url", "db_path", "web_fetch_executable", "web_fetch_wrapper",
                            "elevenlabs_api_key", "groq_api_key"):
                continue
            out[fld.name] = getattr(self, fld.name)
        out["model_url"] = redact_url(self.model_url) if self.model_backend == "http" else None
        out["model_api_key_configured"] = bool(self.model_api_key)
        out["elevenlabs_api_key_configured"] = bool(self.elevenlabs_api_key)
        out["groq_api_key_configured"] = bool(self.groq_key())
        return out
