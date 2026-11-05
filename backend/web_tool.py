"""Bridge to the existing, trusted Rust-tool Python wrapper. No new HTTP endpoint."""
from __future__ import annotations
import asyncio
import importlib.util
import sys
from pathlib import Path


def failed(code, message):
    return {"version":1, "ok":False, "data":None, "error":{"code":code,"message":message}}


class WebTool:
    def __init__(self, settings):
        self.settings = settings
        self._module = None
        self._slots = asyncio.Semaphore(2)

    async def fetch(self, url):
        s = self.settings
        if not s.web_fetch_enabled or not s.web_fetch_allowed_hosts:
            return failed("PERMISSION_DENIED", "Web fetching is disabled or the hostname allowlist is empty.")
        if not s.web_fetch_executable or not s.web_fetch_wrapper:
            return failed("CONFIG_ERROR", "Configure both absolute web-fetch executable and wrapper paths.")
        exe, wrapper = Path(s.web_fetch_executable), Path(s.web_fetch_wrapper)
        if not exe.is_absolute() or not wrapper.is_absolute():
            return failed("CONFIG_ERROR", "Web-fetch paths must be absolute.")
        async with self._slots:
            try:
                if self._module is None:
                    name = "_bough_trusted_web_fetch_wrapper"
                    spec = importlib.util.spec_from_file_location(name, wrapper)
                    if spec is None or spec.loader is None:
                        return failed("CONFIG_ERROR", "Cannot load the trusted web-fetch wrapper.")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    try:
                        spec.loader.exec_module(module)
                    except BaseException:
                        sys.modules.pop(name, None)
                        raise
                    self._module = module
                config = self._module.WebFetchConfig(
                    executable=exe, enabled=True, allowed_hosts=tuple(s.web_fetch_allowed_hosts))
                return await self._module.dispatch_web_action({"kind":"web_fetch", "url":url}, config)
            except (OSError, ValueError, ImportError, AttributeError, RuntimeError):
                return failed("CONFIG_ERROR", "Cannot load or configure the trusted web-fetch tool. Check server paths and settings.")
