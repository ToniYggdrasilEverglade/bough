"""Fixes carried over from the local brain_ai / nested bough copies that never reached the repo."""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.memory.base import Owner
from backend.memory.seed import SeedStore
from backend.memory.supabase_store import SupabaseStore
from backend.models.http_model import HttpChatModel


class FakeResponse:
    headers = {"content-range": "0/0"}

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, size=-1):
        return self.body


def test_legacy_brain_env_names_still_work():
    s = Settings.from_env({"BRAIN_MODEL_NAME": "legacy-model", "BRAIN_MAX_MODEL_CALLS": "2"}, load_env_file=False)
    assert s.model_name == "legacy-model"
    assert s.max_model_calls == 2


def test_bough_env_name_wins_over_legacy():
    s = Settings.from_env({"BRAIN_MODEL_NAME": "old", "BOUGH_MODEL_NAME": "new"}, load_env_file=False)
    assert s.model_name == "new"


def test_legacy_database_is_reused_when_no_bough_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "brain.sqlite3").write_bytes(b"")
    assert Settings.from_env({}, load_env_file=False).db_path == "data/brain.sqlite3"
    (tmp_path / "data" / "bough.sqlite3").write_bytes(b"")
    assert Settings.from_env({}, load_env_file=False).db_path == "data/bough.sqlite3"
    assert Settings.from_env({"BOUGH_DB_PATH": "x.sqlite3"}, load_env_file=False).db_path == "x.sqlite3"


@pytest.mark.parametrize("url", ["https://user:secret@project.supabase.co", "https://project.supabase.co?token=secret",
                                 "https://project.supabase.co#frag"])
def test_supabase_url_cannot_carry_credentials(url):
    with pytest.raises(ValueError):
        Settings(auth_mode="supabase", supabase_url=url, supabase_publishable_key="sb_publishable_x")


def test_seed_pins_are_counted_once():
    store = SeedStore()
    out = store.search("zzzz unrelated", limit=0, domain="always")
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids))


def test_frontend_loads_ui_refresh_assets():
    s = Settings(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"))
    c = TestClient(create_app(s))
    html = c.get("/").text
    assert 'src="theme.js' in html and 'href="professional.css' in html and 'id="theme-toggle"' in html
    assert 'id="mic"' in html            # voice input kept
    assert c.get("/theme.js").status_code == 200
    assert c.get("/professional.css").status_code == 200


def test_http_model_uses_bearer_token(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["Authorization"] = req.headers.get("Authorization")
        return FakeResponse(b'{"choices": [{"message": {"content": "{\\"ok\\": true}"}}]}')

    monkeypatch.setattr("backend.models.http_model.urllib.request.urlopen", fake_urlopen)
    model = HttpChatModel("https://example.test/v1/chat/completions", "demo", api_key="demo-token-123")
    assert model._post({"model": "demo", "messages": [], "max_completion_tokens": 32}, timeout=1.0)[0] == '{"ok": true}'
    assert seen["Authorization"] == "Bearer demo-token-123"


def test_supabase_store_uses_bearer_token(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["Authorization"] = req.headers.get("Authorization")
        return FakeResponse(b"[]")

    monkeypatch.setattr("backend.memory.supabase_store.urllib.request.urlopen", fake_urlopen)
    store = SupabaseStore("https://example.supabase.co", "public-key")
    rows, _ = store._request(Owner("user-1", token="demo-user-token"), "GET", "memories", params={"select": "id"})
    assert rows == []
    assert seen["Authorization"] == "Bearer demo-user-token"
