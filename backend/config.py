"""Configuration. Everything comes from environment variables (optionally a .env file).

Secrets (the model API key) live only here and in the HTTP adapter. `public_dict()` is the only
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
    db_path: str = "data/brain.sqlite3"
    # --- auth (Supabase email+password; JWTs verified against the project's JWKS) ---
    auth_mode: str = "off"                 # "off" (single local user, loopback only) | "supabase"
    supabase_url: str = ""                 # https://<ref>.supabase.co
    supabase_publishable_key: str = ""     # public by design (ships to the browser); RLS guards the data
    jwks_cache_s: float = 3600.0
    # --- pipeline ---
    pipeline: str = "root"                 # "root": model answers freely, agents filter (Yggdrasil) | "proposals": council flow
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
    mock_latency_s: float = 0.0
    # --- web fetching (trusted server configuration only) ---
    web_fetch_enabled: bool = False
    web_fetch_executable: str = field(default="", repr=False)
    web_fetch_wrapper: str = field(default="", repr=False)
    web_fetch_allowed_hosts: tuple[str, ...] = ()
    web_fetch_context_chars: int = 6000    # page text handed to the language model (the CLI returns up to 64 KiB)
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
            raise ValueError("BRAIN_MODEL_BACKEND must be 'mock' or 'http'")
        if self.pipeline not in ("root", "proposals"):
            raise ValueError("BRAIN_PIPELINE must be 'root' or 'proposals'")
        if urlsplit(self.model_url).scheme not in ("http", "https"):
            raise ValueError("BRAIN_MODEL_URL must start with http:// or https://")
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
            raise ValueError("BRAIN_AUTH_MODE must be 'off' or 'supabase'")
        if self.memory_backend not in ("sqlite", "supabase"):
            raise ValueError("BRAIN_MEMORY_BACKEND must be 'sqlite' or 'supabase'")
        if self.auth_mode == "supabase" or self.memory_backend == "supabase":
            u = urlsplit(self.supabase_url)
            if u.scheme != "https" or not u.netloc or u.path not in ("", "/"):
                raise ValueError("BRAIN_SUPABASE_URL must look like https://<ref>.supabase.co")
            if not self.supabase_publishable_key:
                raise ValueError("BRAIN_SUPABASE_PUBLISHABLE_KEY is required")
        if self.memory_backend == "supabase" and self.auth_mode != "supabase":
            raise ValueError("BRAIN_MEMORY_BACKEND=supabase needs BRAIN_AUTH_MODE=supabase (rows are per signed-in user)")
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
            host=s("BRAIN_HOST", d.host), port=i("BRAIN_PORT", d.port),
            allow_non_local_bind=b("BRAIN_ALLOW_NON_LOCAL_BIND", d.allow_non_local_bind),
            db_path=s("BRAIN_DB_PATH", d.db_path),
            memory_backend=s("BRAIN_MEMORY_BACKEND", d.memory_backend).lower(),
            auth_mode=s("BRAIN_AUTH_MODE", d.auth_mode).lower(),
            supabase_url=s("BRAIN_SUPABASE_URL", d.supabase_url).rstrip("/"),
            supabase_publishable_key=s("BRAIN_SUPABASE_PUBLISHABLE_KEY", d.supabase_publishable_key),
            jwks_cache_s=f("BRAIN_JWKS_CACHE_S", d.jwks_cache_s),
            pipeline=s("BRAIN_PIPELINE", d.pipeline).lower(),
            root_repair_enabled=b("BRAIN_ROOT_REPAIR", d.root_repair_enabled),
            model_backend=s("BRAIN_MODEL_BACKEND", d.model_backend).lower(),
            model_url=s("BRAIN_MODEL_URL", d.model_url), model_name=s("BRAIN_MODEL_NAME", d.model_name),
            model_api_key=s("BRAIN_MODEL_API_KEY", d.model_api_key),
            model_json_mode=b("BRAIN_MODEL_JSON_MODE", d.model_json_mode),
            model_timeout_s=f("BRAIN_MODEL_TIMEOUT_S", d.model_timeout_s),
            web_fetch_enabled=b("BRAIN_WEB_FETCH_ENABLED", d.web_fetch_enabled),
            web_fetch_executable=s("BRAIN_WEB_FETCH_EXECUTABLE", d.web_fetch_executable),
            web_fetch_wrapper=s("BRAIN_WEB_FETCH_WRAPPER", d.web_fetch_wrapper),
            web_fetch_allowed_hosts=tuple(x.strip() for x in s("BRAIN_WEB_FETCH_ALLOWED_HOSTS", "").split(",") if x.strip()),
            web_fetch_context_chars=i("BRAIN_WEB_FETCH_CONTEXT_CHARS", d.web_fetch_context_chars),
            mock_latency_s=f("BRAIN_MOCK_LATENCY_S", d.mock_latency_s),
            max_model_calls=i("BRAIN_MAX_MODEL_CALLS", d.max_model_calls),
            cycle_timeout_s=f("BRAIN_CYCLE_TIMEOUT_S", d.cycle_timeout_s),
            max_proposals=i("BRAIN_MAX_PROPOSALS", d.max_proposals),
            workspace_max_items=i("BRAIN_WORKSPACE_MAX_ITEMS", d.workspace_max_items),
            max_retrieved_memories=i("BRAIN_MAX_RETRIEVED_MEMORIES", d.max_retrieved_memories),
            max_retries=i("BRAIN_MAX_RETRIES", d.max_retries),
            max_input_chars=i("BRAIN_MAX_INPUT_CHARS", d.max_input_chars),
            max_sessions=i("BRAIN_MAX_SESSIONS", d.max_sessions),
            max_memory_records=i("BRAIN_MAX_MEMORY_RECORDS", d.max_memory_records),
            memory_min_score=f("BRAIN_MEMORY_MIN_SCORE", d.memory_min_score),
            seed_enabled=b("BRAIN_SEED_ENABLED", d.seed_enabled),
            max_seed_memories=i("BRAIN_MAX_SEED_MEMORIES", d.max_seed_memories),
            model_plan_tokens=i("BRAIN_MODEL_PLAN_TOKENS", d.model_plan_tokens),
            model_review_tokens=i("BRAIN_MODEL_REVIEW_TOKENS", d.model_review_tokens),
            model_reply_tokens=i("BRAIN_MODEL_REPLY_TOKENS", d.model_reply_tokens),
            max_reply_chars=i("BRAIN_MAX_REPLY_CHARS", d.max_reply_chars),
            include_model_interpretations=b("BRAIN_INCLUDE_MODEL_INTERPRETATIONS", d.include_model_interpretations),
            enable_model_review=b("BRAIN_ENABLE_MODEL_REVIEW", d.enable_model_review),
            min_select_score=f("BRAIN_MIN_SELECT_SCORE", d.min_select_score),
            bias_step=f("BRAIN_BIAS_STEP", d.bias_step), bias_limit=f("BRAIN_BIAS_LIMIT", d.bias_limit),
        )

    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS

    def public_dict(self) -> dict:
        """The ONLY settings view that may reach the browser. No secrets, no credentials in URLs."""
        out = {}
        for fld in fields(self):
            if fld.name in ("model_api_key", "model_url", "db_path", "web_fetch_executable", "web_fetch_wrapper"):
                continue
            out[fld.name] = getattr(self, fld.name)
        out["model_url"] = redact_url(self.model_url) if self.model_backend == "http" else None
        out["model_api_key_configured"] = bool(self.model_api_key)
        return out
