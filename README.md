# Bough

An experimental AI application with a Python/FastAPI backend, a browser dashboard, configurable model pipelines, and persistent memory. It runs locally with a deterministic mock model by default; an OpenAI-compatible HTTP endpoint can provide real model responses.

## Features

- Browser dashboard served by the backend.
- REST API and WebSocket events for chat sessions.
- Request cancellation, feedback, and saved cycle summaries.
- Persistent memory using local SQLite or Supabase.
- Optional Supabase authentication with user-scoped sessions and memory.
- Configurable root and proposal pipelines, model-call budgets, timeouts, and response limits.
- Optional read-only knowledge seeds and an allowlisted web-fetch integration.
- Optional text-to-speech for replies through ElevenLabs (free-plan defaults) or Groq Orpheus.
- Optional voice input: a Speak button transcribes your voice with Groq Whisper and reads the reply aloud.

## Quick start

Requires Python 3.10 or newer. Run commands from the repository root.

```sh
git clone https://github.com/ToniYggdrasilEverglade/bough.git
cd bough
python -m venv .venv
```

Activate the virtual environment:

```sh
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and start the application:

```sh
python -m pip install -r requirements.txt
python -m backend
```

Open http://127.0.0.1:8000 in your browser. FastAPI's interactive API documentation is available at http://127.0.0.1:8000/docs.

The default configuration uses a mock model, local SQLite storage, and no login. No model API key is required for this mode. The mock model is for exercising the application, not real AI reasoning.

## Configuration

For custom settings, copy `.env.example` to `.env` in the repository root and edit the values. Do not commit `.env` or API keys.

| Variable | Purpose | Default |
| --- | --- | --- |
| `BOUGH_HOST` | Server bind address | `127.0.0.1` |
| `BOUGH_PORT` | Server port | `8000` |
| `BOUGH_MODEL_BACKEND` | Model adapter: `mock` or `http` | `mock` |
| `BOUGH_AUTH_MODE` | Authentication: `off` or `supabase` | `off` |
| `BOUGH_MEMORY_BACKEND` | Memory storage: `sqlite` or `supabase` | `sqlite` |
| `BOUGH_DB_PATH` | SQLite database path | `data/bough.sqlite3` |
| `BOUGH_PIPELINE` | Pipeline: `root` or `proposals` | `root` |

See `.env.example` for model token limits, request budgets, knowledge seeds, allowed origins, and other options.

### Connect a model

Set the HTTP adapter and your provider's OpenAI-compatible chat-completions endpoint in `.env`:

```dotenv
BOUGH_MODEL_BACKEND=http
BOUGH_MODEL_URL=https://your-provider.example/v1/chat/completions
BOUGH_MODEL_NAME=your-model-name
BOUGH_MODEL_API_KEY=your-api-key
```

For a local Ollama server, the example configuration uses:

```dotenv
BOUGH_MODEL_BACKEND=http
BOUGH_MODEL_URL=http://127.0.0.1:11434/v1/chat/completions
BOUGH_MODEL_NAME=llama3.2
BOUGH_MODEL_JSON_MODE=true
```

The local model server must be running and the selected model must be available. Restart Bough after changing configuration.

### Supabase authentication and storage

To enable Supabase sign-in, configure:

```dotenv
BOUGH_AUTH_MODE=supabase
BOUGH_SUPABASE_URL=https://<project-ref>.supabase.co
BOUGH_SUPABASE_PUBLISHABLE_KEY=your-publishable-key
```

SQLite can still be used with authentication enabled. To store memory in Supabase instead, apply the SQL migrations in `supabase/migrations/` to your project and set `BOUGH_MEMORY_BACKEND=supabase`. This backend requires Supabase authentication and uses each user's JWT with row-level security.

With authentication enabled, API requests require a bearer token except for `GET /api/config`. WebSocket clients must send an authentication message as their first frame:

```json
{"type": "auth", "token": "<supabase-access-token>"}
```

### Optional text-to-speech (ElevenLabs free plan)

Replies get a **Listen** button that reads them aloud with [ElevenLabs](https://elevenlabs.io/text-to-speech-api). The backend calls ElevenLabs, so the API key never reaches the browser. Code blocks are skipped rather than read aloud.

1. Create a free account and an API key at https://elevenlabs.io/app/settings/api-keys (the Text to Speech permission is enough).
2. Add to `.env` and restart Bough:

```dotenv
BOUGH_TTS_BACKEND=elevenlabs
ELEVENLABS_API_KEY=your-elevenlabs-key
```

The defaults are chosen for the free plan:

| Setting | Default | Why |
| --- | --- | --- |
| `BOUGH_ELEVENLABS_MODEL_ID` | `eleven_flash_v2_5` | Half the credits per character, so the 10,000 monthly free credits cover about 20,000 characters |
| `BOUGH_ELEVENLABS_OUTPUT_FORMAT` | `mp3_44100_128` | Best format on free; 192 kbps MP3 and 44.1 kHz PCM/WAV need a paid plan and are refused unless `BOUGH_ELEVENLABS_ALLOW_PAID_FORMATS=true` |
| `BOUGH_ELEVENLABS_VOICE_ID` | `JBFqnCBsd6RMkjVDRZzb` | A default voice; free API users cannot use Voice Library voices (HTTP 402 `paid_plan_required`) |
| `BOUGH_TTS_MAX_CHARS` | `1000` | Caps credits spent per reply; longer replies are clipped |

Free-plan terms: personal, non-commercial use only, with attribution. The dashboard shows a "Voice by ElevenLabs" link next to each Listen button while text-to-speech is on. ElevenLabs is retiring its default voices on December 31, 2026; after that, set `BOUGH_ELEVENLABS_VOICE_ID` to a voice from My Voices (for example one made with Voice Design).

### Web search on Groq

Ask explicitly ("use the web to ...", "search online for ...", "look it up online") and, when `BOUGH_MODEL_NAME` is `openai/gpt-oss-20b` or `openai/gpt-oss-120b` on Groq, Bough turns on Groq's built-in [browser search](https://console.groq.com/docs/browser-search) for that one reply. Groq runs the search on its side; Bough records the queries in the trace, and the grounding branch accepts "I searched the web" only when a search really ran. Searched replies get `BOUGH_MODEL_WEB_SEARCH_TIMEOUT_S` (default 60 s). Set `BOUGH_MODEL_WEB_SEARCH=off` to never search.

Other Groq models (for example `qwen/qwen3.8-27b` or the Llama models) have no browser search. For them, an explicit web request is first searched by a helper model, `BOUGH_MODEL_WEB_SEARCH_HELPER` (default `openai/gpt-oss-20b`, same Groq key, at most `BOUGH_MODEL_WEB_SEARCH_HELPER_TOKENS`=700 output tokens), and your main model writes the answer from its findings. Set `BOUGH_MODEL_WEB_SEARCH_HELPER=` (empty) to turn the helper off.

### GitHub lookups

Questions that name a repository (`github.com/owner/repo`, or `owner/repo` together with the word GitHub/repo) and ask about its issues, pull requests, releases, commits or stats are answered from the GitHub REST API, with any model. The requests go through the Rust `web_fetch` tool (GET only; its allowlist, public-IP check and size/time limits apply), so web fetch must be enabled and `api.github.com` allowlisted. Issues come from the issue search (`is:issue`, newest first), so pull requests are not mixed in; repos with issues turned off (such as `torvalds/linux`) are reported as such. GitHub allows 60 anonymous requests/hour per IP (search: 10/minute). `BOUGH_GITHUB_LOOKUP=off` disables it.

### Groq free-tier output limits

Groq's free tier caps output tokens per minute per model (for example 1,000 for `qwen/qwen3.8-27b`). If a request is rejected as "Request too large ... (OTPM): Limit N", Bough lowers its output cap to 75% of N, resends at once, and keeps that cap for later requests (the terminal prints `[model limit] output cap lowered to ...`). If the per-minute window is merely used up, Bough waits up to 8 seconds and retries once; longer waits are reported in the reply ("Try again in about 43s"). To avoid the first rejected request entirely, set `BOUGH_MODEL_PLAN_TOKENS`, `BOUGH_MODEL_REVIEW_TOKENS` and `BOUGH_MODEL_REPLY_TOKENS` below your limit.

### Groq voice and mic input

Groq can handle the whole voice loop with the same key you use for the model: Whisper turns your speech into text, the Groq model answers, and Orpheus reads the reply aloud. Speech uses `GROQ_API_KEY`, or reuses `BOUGH_MODEL_API_KEY` when `BOUGH_MODEL_URL` points at `api.groq.com`.

```dotenv
BOUGH_MODEL_BACKEND=http
BOUGH_MODEL_URL=https://api.groq.com/openai/v1/chat/completions
BOUGH_MODEL_NAME=qwen/qwen3.8-27b
BOUGH_MODEL_API_KEY=your-groq-key
BOUGH_STT_BACKEND=groq        # Speak button -> Groq Whisper
BOUGH_TTS_BACKEND=groq        # or elevenlabs to keep ElevenLabs as the voice
```

- **Speak button:** click to record, click again to send (recording stops on its own after 2 minutes). The transcribed text is sent as a normal message, and if text-to-speech is on the reply plays automatically. The browser asks for microphone permission; it works on `http://127.0.0.1` or HTTPS.
- **Orpheus voice:** `canopylabs/orpheus-v1-english` with voices `autumn`, `diana`, `hannah`, `austin`, `daniel` and `troy` (`BOUGH_GROQ_TTS_VOICE`). Groq accepts 200 characters per request, so Bough splits replies at sentence breaks and joins the pieces into one WAV file. Accept the model's terms once in the [Groq console](https://console.groq.com/playground) first, otherwise Groq answers HTTP 400.
- **Whisper:** `whisper-large-v3-turbo` by default (`BOUGH_GROQ_STT_MODEL`); Groq bills each clip as at least 10 seconds and the free tier caps uploads at 25 MB.
- `<think>` blocks, links, tables and code blocks are removed before anything is spoken, with either voice provider.

### Optional web fetch

The web-fetch integration is disabled by default. Build the Rust CLI in `tools/web_fetch` with `cargo build --release`, then configure the executable and wrapper using absolute paths in `.env`. Set `BOUGH_WEB_FETCH_ALLOWED_HOSTS` explicitly before enabling `BOUGH_WEB_FETCH_ENABLED`. See `.env.example` for the relevant settings.

With web fetch on, a URL pasted into an ordinary message is fetched too (not only `/fetch <url>`), when its host is allowlisted; `BOUGH_WEB_FETCH_AUTO=false` limits fetching to `/fetch`. The tool accepts `text/html`, `text/plain` and `application/json`. Example allowlist: `BOUGH_WEB_FETCH_ALLOWED_HOSTS=api.github.com,en.wikipedia.org,docs.python.org`. Rebuild after pulling changes to the tool: `cd tools/web_fetch && cargo build --release`.

### Code sandbox

Off by default. With `BOUGH_SANDBOX_BACKEND=docker`, an explicit request such as "run this code", "what does this print?" or "use python to ..." runs code in a throwaway container: the user's own fenced block as written, or otherwise the block the model writes for the request, after which the model answers from the real stdout/stderr/exit code. Python, JavaScript (Node) and sh are supported (`BOUGH_SANDBOX_LANGUAGES`).

Each run is a fresh `docker run --rm` under gVisor (`runsc`, a user-space kernel between the code and the host), with no network, a read-only root filesystem, 32 MB tmpfs for `/work` and `/tmp`, user `nobody`, all capabilities dropped, `no-new-privileges`, 64 processes, `BOUGH_SANDBOX_MEMORY_MB` (256), `BOUGH_SANDBOX_CPUS` (1), `BOUGH_SANDBOX_TIMEOUT_S` (10 s) and `BOUGH_SANDBOX_OUTPUT_CHARS` (2500 per stream; floods are stopped at once). Images are never pulled at run time. Only signed-in users (Supabase auth) may run code, `BOUGH_SANDBOX_RUNS_PER_HOUR` (30) each, `BOUGH_SANDBOX_CONCURRENCY` (2) containers at a time in total. `BOUGH_SANDBOX_ALLOW_ANONYMOUS=true` lifts the sign-in rule; do not use it on a public site.

Setup on Debian/Kali:

```sh
sudo apt install -y docker.io && sudo systemctl enable --now docker
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list
sudo apt update && sudo apt install -y runsc
sudo runsc install && sudo systemctl restart docker
sudo docker pull python:3.12-slim && sudo docker pull node:22-slim
sudo docker run --rm --runtime runsc python:3.12-slim uname -r    # prints a "-gvisor" kernel
```

Bough needs access to the Docker daemon. Membership in the `docker` group is equivalent to root on that machine, so run Bough as a dedicated user rather than your own account (`sudo useradd -r -m bough && sudo usermod -aG docker bough`), or point `BOUGH_SANDBOX_DOCKER` at a small script that runs `sudo docker` with a sudoers rule limited to that script. Running strangers' code on a personal machine that also holds your files and keys carries risk even with gVisor; a separate VM or machine for Bough is the safer production setup.

Real-container test: `BOUGH_TEST_DOCKER=docker python -m pytest tests/test_sandbox.py`.

## Project layout

```text
backend/              FastAPI app, orchestration, agents, models, and memory
frontend/             Browser dashboard served by FastAPI
supabase/             Supabase database migrations
tests/                Automated tests
tools/                Supporting tools, including web fetch
.env.example          Configuration reference
requirements.txt      Backend dependencies
requirements-dev.txt  Development dependencies
```

## API overview

- `GET /api/config` — application configuration description.
- `GET /api/sessions/{sid}` — session state.
- `POST /api/sessions/{sid}/chat` — submit a message with a JSON `text` field.
- `POST /api/sessions/{sid}/cancel` — cancel the active request.
- `PUT /api/sessions/{sid}/settings` — update memory-enabled state.
- `POST /api/sessions/{sid}/feedback` — submit cycle feedback.
- `POST /api/sessions/{sid}/cycles/{cid}/save-summary` — save a cycle summary.
- `GET /api/memories`, `POST /api/memories`, `DELETE /api/memories/{mid}` — manage memory.
- `GET /api/adaptation` — inspect stored adaptation values.
- `GET /api/me` — inspect authentication identity.
- `POST /api/tts` — read text aloud (JSON `text`, optional `voice_id`); returns MP3 (ElevenLabs) or WAV (Groq). Needs `BOUGH_TTS_BACKEND`.
- `POST /api/transcribe` — raw audio body (`Content-Type: audio/webm`, `audio/ogg`, `audio/mp4`, `audio/wav`, ...) to `{"text"}` via Groq Whisper. Needs `BOUGH_STT_BACKEND=groq`.
- `WS /ws/{client_sid}` — chat and backend events over WebSocket.

## Safety and limitations

This is a prototype. Keep the default loopback binding for local development, especially when authentication is off. Do not expose an unauthenticated instance to a network. The launcher refuses non-loopback binding unless explicitly overridden; that override is not a production deployment configuration.

Text-to-speech sends the reply text (not your memories) to ElevenLabs or Groq when you press Listen. The Speak button sends your recording to Groq.

HTTP model mode sends request context to the configured model endpoint. Use a provider and storage configuration appropriate for your data, and keep credentials out of version control.
