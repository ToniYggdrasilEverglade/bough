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

## Quick start

Requires Python 3.10 or newer. Run commands from the repository root.

```sh
git clone https://github.com/ToniBoughEverglade/bough.git
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

### Optional web fetch

The web-fetch integration is disabled by default. Build the Rust CLI in `tools/web_fetch` with `cargo build --release`, then configure the executable and wrapper using absolute paths in `.env`. Set `BOUGH_WEB_FETCH_ALLOWED_HOSTS` explicitly before enabling `BOUGH_WEB_FETCH_ENABLED`. See `.env.example` for the relevant settings.

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
- `WS /ws/{client_sid}` — chat and backend events over WebSocket.

## Safety and limitations

This is a prototype. Keep the default loopback binding for local development, especially when authentication is off. Do not expose an unauthenticated instance to a network. The launcher refuses non-loopback binding unless explicitly overridden; that override is not a production deployment configuration.

HTTP model mode sends request context to the configured model endpoint. Use a provider and storage configuration appropriate for your data, and keep credentials out of version control.
