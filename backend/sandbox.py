"""Run user or model code in a throwaway, locked-down container (Docker + gVisor by default).

Each run is a fresh `docker run --rm` with: gVisor (`--runtime=runsc`, a user-space kernel, so container code never
talks to the host kernel directly), no network, a read-only root filesystem, a small tmpfs as the only writable
place, an unprivileged user (nobody), every Linux capability dropped, no-new-privileges, and limits on memory,
CPU, process count, open files, file size, wall-clock time and captured output. Images are never pulled at run
time (`--pull never`), so a typo cannot fetch an arbitrary image.

Who may run code is decided by the orchestrator (signed-in users only unless BOUGH_SANDBOX_ALLOW_ANONYMOUS);
this module enforces a per-user hourly cap and a global concurrency limit.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

log = logging.getLogger("bough.sandbox")

MAX_CODE_CHARS = 20_000


@dataclass(frozen=True)
class Language:
    name: str
    image_setting: str      # Settings attribute holding the image
    filename: str
    command: str            # run inside the container after the code is written to /work/<filename>


LANGUAGES: Dict[str, Language] = {
    "python": Language("python", "sandbox_python_image", "main.py", "python3 -I -B /work/main.py"),
    "javascript": Language("javascript", "sandbox_node_image", "main.js", "node /work/main.js"),
    "bash": Language("bash", "sandbox_python_image", "main.sh", "sh /work/main.sh"),
}
ALIASES = {"py": "python", "python3": "python", "js": "javascript", "node": "javascript", "nodejs": "javascript",
           "sh": "bash", "shell": "bash", "zsh": "bash"}

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)
# Explicit requests only: Bough never runs code on its own initiative.
RUN_REQUEST = re.compile(
    r"\b(?:run|execute|exec|test)\s+(?:it|this|that|the\s+(?:code|script|program|snippet)|my\s+(?:code|script)|"
    r"(?:this|that|the|a|some)?\s*(?:python|javascript|js|node|bash|shell)?\s*(?:code|script|program|snippet))\b"
    r"|\bwhat\s+(?:does|will|would)\s+(?:this|that|it|the\s+code)\s+(?:print|output|return|show)\b"
    r"|\b(?:use|with|write\s+and\s+run)\s+(?:python|code|the\s+sandbox)\s+to\b"
    r"|\b(?:in|using)\s+the\s+sandbox\b",
    re.I,
)


def canonical_language(tag: str) -> Optional[str]:
    t = (tag or "").strip().lower()
    t = ALIASES.get(t, t)
    return t if t in LANGUAGES else None


def wants_run(text: str) -> bool:
    return isinstance(text, str) and bool(RUN_REQUEST.search(text))


def first_runnable_block(text: str, allowed: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    """(language, code) of the first fenced block in a supported, enabled language."""
    if not isinstance(text, str):
        return None
    for m in _FENCE.finditer(text):
        lang = canonical_language(m.group(1))
        code = m.group(2)
        if lang and lang in allowed and code.strip():
            return lang, code
    return None


def failed(code: str, message: str) -> Dict[str, Any]:
    return {"version": 1, "tool": "sandbox", "ok": False, "data": None, "error": {"code": code, "message": message}}


class RateLimiter:
    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self._runs: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

    def check(self, user: str, now: Optional[float] = None) -> Optional[int]:
        """None if allowed (and records the run), else seconds until the next run is allowed."""
        now = time.monotonic() if now is None else now
        q = self._runs[user]
        while q and now - q[0] >= 3600:
            q.popleft()
        if len(q) >= self.per_hour:
            return int(3600 - (now - q[0])) + 1
        q.append(now)
        return None


class Sandbox:
    def __init__(self, settings):
        self.s = settings
        self.limiter = RateLimiter(settings.sandbox_runs_per_hour)
        self._slots = asyncio.Semaphore(settings.sandbox_concurrency)

    @property
    def enabled(self) -> bool:
        return self.s.sandbox_backend == "docker"

    @property
    def languages(self) -> Tuple[str, ...]:
        return tuple(self.s.sandbox_languages)

    def docker_args(self, lang: Language, name: str) -> List[str]:
        s = self.s
        args = [s.sandbox_docker, "run", "--rm", "-i", "--name", name, "--pull", "never",
                "--network", "none", "--read-only",
                "--tmpfs", "/work:rw,nosuid,nodev,size=32m,mode=1777",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m,mode=1777",
                "--workdir", "/work", "--user", "65534:65534", "--env", "HOME=/tmp",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "64", "--memory", f"{s.sandbox_memory_mb}m", "--memory-swap", f"{s.sandbox_memory_mb}m",
                "--cpus", str(s.sandbox_cpus), "--ulimit", "nofile=256:256", "--ulimit", "fsize=16777216:16777216",
                "--log-driver", "none", "--stop-timeout", "1"]
        if s.sandbox_runtime:
            args += ["--runtime", s.sandbox_runtime]
        script = f"head -c {MAX_CODE_CHARS * 4} > /work/{lang.filename} && exec {lang.command}"
        return args + [getattr(s, lang.image_setting), "sh", "-c", script]

    async def run(self, user: str, language: str, code: str) -> Dict[str, Any]:
        if not self.enabled:
            return failed("DISABLED", "The code sandbox is off on this server (BOUGH_SANDBOX_BACKEND=off).")
        lang_name = canonical_language(language)
        if lang_name is None or lang_name not in self.languages:
            return failed("UNSUPPORTED_LANGUAGE", f"Supported languages: {', '.join(self.languages)}.")
        if len(code) > MAX_CODE_CHARS:
            return failed("CODE_TOO_LARGE", f"Code is longer than {MAX_CODE_CHARS} characters.")
        wait = self.limiter.check(user)
        if wait is not None:
            return failed("RATE_LIMITED", f"Sandbox limit of {self.s.sandbox_runs_per_hour} runs per hour reached; "
                                          f"try again in about {max(1, wait // 60)} min.")
        lang = LANGUAGES[lang_name]
        name = f"bough-sbx-{uuid.uuid4().hex[:16]}"
        async with self._slots:
            return await self._run_container(lang, name, code)

    async def _run_container(self, lang: Language, name: str, code: str) -> Dict[str, Any]:
        s = self.s
        cap = s.sandbox_output_chars
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.docker_args(lang, name), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except (OSError, ValueError) as e:
            return failed("UNAVAILABLE", f"Cannot start docker ({type(e).__name__}). Is Docker installed and running?")

        out: Dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        overflow = {"hit": False}
        flooded = asyncio.Event()

        async def pump(stream, key):
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return
                room = cap * 4 - len(out[key])
                if room > 0:
                    out[key] += chunk[:room]
                if len(out[key]) >= cap * 4 and not overflow["hit"]:
                    overflow["hit"] = True
                    flooded.set()                # enough output: stop the container now instead of at the timeout

        timed_out = False
        try:
            proc.stdin.write(code.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
            work = asyncio.ensure_future(asyncio.gather(pump(proc.stdout, "stdout"), pump(proc.stderr, "stderr"),
                                                        proc.wait()))
            stop = asyncio.ensure_future(flooded.wait())
            done, _ = await asyncio.wait({work, stop}, timeout=s.sandbox_timeout_s,
                                         return_when=asyncio.FIRST_COMPLETED)
            stop.cancel()
            if work not in done:
                timed_out = not flooded.is_set()
                await self._kill(name, proc)
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except asyncio.CancelledError:
            await self._kill(name, proc)
            raise
        finally:
            if proc.returncode is None:
                await self._kill(name, proc)

        stdout = out["stdout"].decode("utf-8", errors="replace")
        stderr = out["stderr"].decode("utf-8", errors="replace")
        infra = _docker_problem(stderr) if proc.returncode in (125, 126, 127) else None
        if infra:
            return failed("UNAVAILABLE", infra)
        stopped_for_output = overflow["hit"] and not timed_out and proc.returncode not in (0, None) and flooded.is_set()
        result = {"language": lang.name, "exit_code": None if (timed_out or stopped_for_output) else proc.returncode,
                  "timed_out": timed_out, "duration_s": round(time.monotonic() - started, 2),
                  "stdout": stdout[:cap], "stderr": stderr[:cap],
                  "output_truncated": overflow["hit"] or len(stdout) > cap or len(stderr) > cap,
                  "limits": {"timeout_s": s.sandbox_timeout_s, "memory_mb": s.sandbox_memory_mb, "network": "none"}}
        if stopped_for_output:
            result["note"] = f"stopped early: output passed the {cap}-character limit"
        elif proc.returncode == 137 and not timed_out:
            result["note"] = "killed (exit 137): most likely the memory limit"
        text = json.dumps(result, ensure_ascii=False, indent=1)
        return {"version": 1, "tool": "sandbox", "ok": True, "error": None,
                "data": {"final_url": None, "status": None, "content_type": "application/json",
                         "title": f"Sandbox run ({lang.name})", "text": text, "chars": len(text), "truncated": False},
                "run": {"language": lang.name, "exit_code": result["exit_code"], "timed_out": timed_out}}

    async def _kill(self, name: str, proc) -> None:
        try:
            k = await asyncio.create_subprocess_exec(self.s.sandbox_docker, "rm", "-f", name,
                                                     stdout=asyncio.subprocess.DEVNULL,
                                                     stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(k.wait(), timeout=10)
        except (OSError, asyncio.TimeoutError):
            log.warning("could not remove sandbox container %s", name)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass


def _docker_problem(stderr: str) -> Optional[str]:
    low = stderr.lower()
    if "unknown or invalid runtime" in low or "runtime" in low and "not found" in low:
        return "Docker has no 'runsc' runtime. Install gVisor (see README: Code sandbox) or set BOUGH_SANDBOX_RUNTIME."
    if "no such image" in low or "pull access denied" in low or "unable to find image" in low:
        return "Sandbox image is not pulled. Run the docker pull commands from README: Code sandbox."
    if "permission denied" in low and "docker" in low:
        return "Bough cannot reach the Docker daemon (permission denied on the docker socket)."
    if "cannot connect to the docker daemon" in low:
        return "The Docker daemon is not running."
    return None
