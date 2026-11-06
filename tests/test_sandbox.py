"""Code sandbox: explicit run requests only, signed-in users only, locked-down docker run."""
import asyncio
import io
import json
import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

import backend.models.http_model as hm
from backend import sandbox as sb
from backend.app import create_app
from backend.config import Settings

GROQ = "https://api.groq.com/openai/v1/chat/completions"


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok(content):
    return Resp(json.dumps({"choices": [{"message": {"role": "assistant", "content": content},
                                         "finish_reason": "stop"}]}).encode())


def settings(**kw):
    return Settings(db_path=os.path.join(tempfile.mkdtemp(), "t.sqlite3"), model_backend="http", model_url=GROQ,
                    model_name="qwen/qwen3.8-27b", model_api_key="gsk_test", **kw)


def data_of(body):
    return json.loads(body["messages"][-1]["content"])["data"]


def chat(c, text):
    return c.post("/api/sessions/s1/chat", json={"text": text}).json()


def test_run_requests_are_explicit():
    for t in ("run this code", "can you execute it", "what does this print?", "use python to compute 2**64",
              "run this python script", "test my code", "do it in the sandbox"):
        assert sb.wants_run(t), t
    for t in ("write a python function", "how do I run a marathon", "explain this code", "who runs github"):
        assert not sb.wants_run(t), t


def test_first_runnable_block():
    text = "intro\n```text\nnope\n```\n```py\nprint(1)\n```\n```js\nconsole.log(2)\n```"
    assert sb.first_runnable_block(text, ("python", "javascript")) == ("python", "print(1)\n")
    assert sb.first_runnable_block(text, ("javascript",)) == ("javascript", "console.log(2)\n")
    assert sb.first_runnable_block("no code", ("python",)) is None


def test_docker_args_are_locked_down():
    args = sb.Sandbox(Settings(sandbox_backend="docker")).docker_args(sb.LANGUAGES["python"], "bough-sbx-x")
    joined = " ".join(args)
    for flag in ("--rm", "--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--user 65534:65534", "--pids-limit 64", "--memory 256m", "--memory-swap 256m", "--runtime runsc",
                 "--pull never", "--log-driver none"):
        assert flag in joined, flag
    assert "-v" not in args and "--privileged" not in args and "--volume" not in joined


def test_rate_limit_per_user():
    rl = sb.RateLimiter(2)
    assert rl.check("a", 0) is None and rl.check("a", 1) is None
    assert rl.check("a", 2) is not None
    assert rl.check("b", 2) is None            # other users unaffected
    assert rl.check("a", 3601) is None         # window slides


def test_disabled_and_bad_language():
    off = asyncio.run(sb.Sandbox(Settings()).run("u", "python", "print(1)"))
    assert off["error"]["code"] == "DISABLED"
    bad = asyncio.run(sb.Sandbox(Settings(sandbox_backend="docker")).run("u", "ruby", "puts 1"))
    assert bad["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def fake_run(calls, stdout="5050\n"):
    async def run(self, user, language, code):
        calls.append((user, language, code))
        text = json.dumps({"language": language, "exit_code": 0, "timed_out": False, "stdout": stdout, "stderr": ""})
        return {"version": 1, "tool": "sandbox", "ok": True, "error": None,
                "data": {"final_url": None, "status": None, "content_type": "application/json",
                         "title": f"Sandbox run ({language})", "text": text, "chars": len(text), "truncated": False},
                "run": {"language": language, "exit_code": 0, "timed_out": False}}
    return run


def test_model_code_is_run_then_answered_from_real_output(monkeypatch):
    calls, bodies = [], []

    def model(req, timeout):
        b = json.loads(req.data)
        bodies.append(b)
        if len(bodies) == 1:
            return ok("```python\nprint(sum(range(101)))\n```")
        return ok("The sum is **5050** (sandbox output).")

    monkeypatch.setattr(hm.urllib.request, "urlopen", model)
    monkeypatch.setattr(sb.Sandbox, "run", fake_run(calls))
    final = chat(TestClient(create_app(settings(sandbox_backend="docker", sandbox_allow_anonymous=True))),
                 "use python to add the numbers 1 to 100 and run it")
    assert calls == [("anonymous", "python", "print(sum(range(101)))\n")]
    assert "code_runner" in data_of(bodies[0]) and "5050" in data_of(bodies[1])["tool_result"]["data"]["text"]
    assert data_of(bodies[1])["ran_code"]["source"] == "model"
    assert "5050" in json.dumps(final)


def test_users_own_code_runs_without_a_model_rewrite(monkeypatch):
    calls, bodies = [], []
    monkeypatch.setattr(hm.urllib.request, "urlopen",
                        lambda req, timeout: (bodies.append(json.loads(req.data)), ok("It prints 3."))[1])
    monkeypatch.setattr(sb.Sandbox, "run", fake_run(calls, "3\n"))
    chat(TestClient(create_app(settings(sandbox_backend="docker", sandbox_allow_anonymous=True))),
         "run this code\n```python\nprint(1+2)\n```")
    assert calls == [("anonymous", "python", "print(1+2)\n")] and len(bodies) == 1
    assert data_of(bodies[0])["ran_code"] == {"language": "python", "code": "print(1+2)\n", "source": "user"}


def test_anonymous_users_cannot_run_code(monkeypatch):
    calls, bodies = [], []
    monkeypatch.setattr(hm.urllib.request, "urlopen",
                        lambda req, timeout: (bodies.append(json.loads(req.data)), ok("Sign in to run code."))[1])
    monkeypatch.setattr(sb.Sandbox, "run", fake_run(calls))
    chat(TestClient(create_app(settings(sandbox_backend="docker"))), "run this code\n```python\nprint(1)\n```")
    assert calls == [] and data_of(bodies[0])["tool_result"]["error"]["code"] == "SIGN_IN_REQUIRED"


def test_sandbox_off_never_offers_or_runs(monkeypatch):
    calls, bodies = [], []
    monkeypatch.setattr(hm.urllib.request, "urlopen",
                        lambda req, timeout: (bodies.append(json.loads(req.data)), ok("```python\nprint(1)\n```"))[1])
    monkeypatch.setattr(sb.Sandbox, "run", fake_run(calls))
    chat(TestClient(create_app(settings())), "run this code\n```python\nprint(1)\n```")
    assert calls == [] and len(bodies) == 1 and "code_runner" not in data_of(bodies[0])


DOCKER = os.environ.get("BOUGH_TEST_DOCKER")


@pytest.mark.skipif(not DOCKER or not shutil.which(DOCKER), reason="set BOUGH_TEST_DOCKER=<docker cmd> to run")
def test_real_container_isolation():
    box = sb.Sandbox(Settings(sandbox_backend="docker", sandbox_docker=shutil.which(DOCKER), sandbox_timeout_s=6))

    def run(lang, code):
        r = asyncio.run(box.run("u", lang, code))
        assert r["ok"], r
        return json.loads(r["data"]["text"])

    assert run("python", "print(6*7)")["stdout"] == "42\n"
    assert "no network" in run("python", "import socket\ntry:\n socket.create_connection(('1.1.1.1',80),2)\n"
                                         " print('OPEN')\nexcept OSError: print('no network')")["stdout"]
    assert "uid=65534" in run("bash", "id")["stdout"]
    assert run("python", "while True: pass")["timed_out"]
    assert run("python", "while True: print('A'*1000)")["output_truncated"]


def test_only_signed_in_users_get_a_sandbox_identity():
    from backend.memory.base import LOCAL, Owner
    from backend.orchestrator import Orchestrator

    class S:
        owner = LOCAL

    o = Orchestrator.__new__(Orchestrator)
    o.settings = Settings(auth_mode="supabase", supabase_url="https://x.supabase.co", supabase_publishable_key="k")
    assert o._sandbox_user(S()) is None
    S.owner = Owner("user-123", token="t")
    assert o._sandbox_user(S()) == "user-123"
    o.settings = Settings()
    assert o._sandbox_user(S()) is None
    o.settings = Settings(sandbox_allow_anonymous=True)
    assert o._sandbox_user(S()) == "anonymous"
