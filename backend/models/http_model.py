"""Adapter for OpenAI-compatible chat-completions endpoints."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from ..errors import ModelOutputError, ModelTimeout, ModelUnavailable
from .base import ModelAdapter
from .prompts import TEXT_TASKS, build_messages


MAX_RESPONSE_BYTES = 1_000_000
# Per-task output budgets; keep planning separate from answer generation.
# Defaults only; Settings overrides these (BOUGH_MODEL_PLAN_TOKENS / _REVIEW_TOKENS / _REPLY_TOKENS).
TASK_COMPLETION_TOKENS = {
    "plan": 1500,
    "review": 600,
    "verbalize": 6000,
    "root": 6000,
    "repair": 6000,
}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)

# Groq-hosted models with the server-side browser_search tool (https://console.groq.com/docs/browser-search).
BROWSER_SEARCH_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b"}
# Only an explicit request turns web search on; Bough never searches on its own initiative.
WEB_REQUEST = re.compile(
    r"\b(?:use|search|check|browse|look\s+(?:it\s+|this\s+|that\s+)?up\s+(?:on|in)?|google|find\s+(?:out\s+)?(?:on|from))"
    r"\b[^.?!\n]{0,40}?\b(?:the\s+)?(?:web|internet|online|google|net)\b"
    r"|\b(?:web|internet|online)\s+search\b|\bsearch\s+(?:the\s+)?(?:web|internet|online)\b|\bgoogle\s+it\b",
    re.I,
)
NO_TOOLS_NOTE = ("No tools are available in this request: do not call any tool (no web search, browser, or code "
                 "execution). Answer directly from your own knowledge in the required output format; if the user "
                 "asked for live web results, say briefly that web search is unavailable here and answer anyway.")


class ModelText(str):
    """A text reply that also records which provider-side tools ran (e.g. Groq browser_search)."""
    executed_tools: List[Dict[str, Any]] = []

    def __new__(cls, value: str, executed_tools: Optional[List[Dict[str, Any]]] = None):
        obj = super().__new__(cls, value)
        obj.executed_tools = list(executed_tools or [])
        return obj


class ToolUseRejected(Exception):
    """Groq 400 tool_use_failed: the model called a tool the request did not offer."""

    def __init__(self, message: str, tool: str):
        super().__init__(message)
        self.tool = tool


def wants_web(data: Dict[str, Any]) -> bool:
    req = data.get("request") if isinstance(data, dict) else None
    return isinstance(req, str) and bool(WEB_REQUEST.search(req))


def extract_json(text: str) -> Any:
    if not isinstance(text, str):
        raise ModelOutputError("model response content must be text")

    t = _FENCE.sub("", text.strip())

    try:
        return json.loads(t)
    except ValueError:
        pass

    a, b = t.find("{"), t.rfind("}")

    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except ValueError:
            pass

    raise ModelOutputError("model output was not valid JSON")


class HttpChatModel(ModelAdapter):
    is_mock = False

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        json_mode: bool = True,
        completion_tokens: Optional[Dict[str, int]] = None,
        web_search: str = "auto",
        web_search_timeout_s: float = 60.0,
    ):
        self.url = url
        self.model = model
        self._api_key = api_key
        self.json_mode = json_mode
        self.web_search = web_search
        self.web_search_timeout_s = web_search_timeout_s
        self.completion_tokens = dict(TASK_COMPLETION_TOKENS, **(completion_tokens or {}))
        self.name = f"http:{model}"

    @property
    def supports_web_search(self) -> bool:
        return (self.web_search == "auto" and urlsplit(self.url).hostname == "api.groq.com"
                and self.model in BROWSER_SEARCH_MODELS)

    def uses_web_search(self, task: str, data: Dict[str, Any]) -> bool:
        return task == "root" and self.supports_web_search and wants_web(data)   # one search per reply, max

    def timeout_for(self, task: str, data: Dict[str, Any], default: float) -> float:
        """Gateway hook: a browser session takes longer than a plain completion."""
        return max(default, self.web_search_timeout_s) if self.uses_web_search(task, data) else default

    async def run(
        self,
        task: str,
        data: Dict[str, Any],
        *,
        timeout: float,
    ) -> Any:
        now = datetime.now().astimezone()

        clock_message = {
            "role": "system",
            "content": (
                "Runtime clock information supplied by the application:\n"
                "Current server-local date and time: "
                f"{now.isoformat(timespec='seconds')}\n"
                f"Timezone abbreviation: {now.tzname()}\n"
                f"Readable local time: {now.strftime('%I:%M:%S %p')}\n"
                "Use this timestamp when answering current date or time "
                "questions. This is the backend computer's local time, "
                "not necessarily the user's time if they are elsewhere. "
                "It was sampled immediately before this model request, "
                "so a small response delay is expected. "
                "Continue following the task's required JSON output format."
            ),
        }

        messages = list(build_messages(task, data))

        # Place clock context after existing system instructions.
        insert_at = 0

        while (
            insert_at < len(messages)
            and messages[insert_at].get("role") == "system"
        ):
            insert_at += 1

        messages.insert(insert_at, clock_message)

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": False,
            "max_completion_tokens": self.completion_tokens.get(task, 1024),
        }

        if self.json_mode and task not in TEXT_TASKS:
            body["response_format"] = {"type": "json_object"}

        searching = self.uses_web_search(task, data)
        if searching:
            # Groq runs the search server-side; the reply comes back as ordinary text plus executed_tools.
            body["tools"] = [{"type": "browser_search"}]
            body["tool_choice"] = "required"
            body["reasoning_effort"] = "low"   # Groq's recommendation for browser search
            timeout = max(timeout, self.web_search_timeout_s)


        print(
            f"[request config] model={body['model']} "
            f"task={task} "
            f"max_completion_tokens={body['max_completion_tokens']}",
            flush=True,
        )

        text_task = task in TEXT_TASKS
        try:
            content, tools = await asyncio.to_thread(self._post, body, timeout, text_task)
        except ToolUseRejected as first:
            # The model reached for a tool nobody offered (gpt-oss does this for "use the web"). Ask once more
            # with an explicit no-tools instruction instead of failing the whole reply.
            if "tools" in body:
                raise ModelOutputError(str(first)) from None
            retry = dict(body, messages=messages[:insert_at + 1] + [{"role": "system", "content": NO_TOOLS_NOTE}]
                         + messages[insert_at + 1:])
            try:
                content, tools = await asyncio.to_thread(self._post, retry, timeout, text_task)
            except ToolUseRejected as again:
                raise ModelOutputError(f"model kept calling a tool Bough did not offer ({again.tool})") from None
        if text_task:
            return ModelText(content, tools)   # free-form reply; the branches inspect it, nothing parses it
        return extract_json(content)

    def _post(self, body: Dict[str, Any], timeout: float, allow_truncated: bool = False) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Bough/1.0",
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body = dict(body)
        body.pop("max_tokens", None)

        print(
            "[OUTGOING MODEL REQUEST] "
            f"model={body.get('model')} "
            f"max_completion_tokens={body.get('max_completion_tokens')}",
            flush=True,
        )
        
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES + 1)

        except urllib.error.HTTPError as e:
            raw_error = e.read(8192).decode(
                "utf-8",
                errors="replace",
            )

            provider_code = None
            try:
                payload = json.loads(raw_error)

                if isinstance(payload, dict):
                    error = payload.get("error", payload)
                else:
                    error = payload

                if isinstance(error, dict):
                    detail = str(error.get("message", "Model endpoint error"))
                    provider_code = error.get("code")
                else:
                    detail = str(error)

            except ValueError:
                detail = raw_error

            if self._api_key:
                detail = detail.replace(self._api_key, "***")

            message = (
                f"HTTP {e.code} from model endpoint: {detail[:2000]}"
            )

            print(f"[model error] {message}", flush=True)
            if e.code == 400 and provider_code == "tool_use_failed":
                tool = "unknown tool"
                try:
                    fg = json.loads(error.get("failed_generation") or "{}") if isinstance(error, dict) else {}
                    tool = str(fg.get("name") or tool)[:60]
                except (ValueError, AttributeError):
                    pass
                raise ToolUseRejected(f"model tried to call a tool Bough did not offer ({tool})", tool) from None
            if e.code == 400 and (
                provider_code == "json_validate_failed"
                or "failed to generate json" in detail.lower()
            ):
                raise ModelOutputError(message) from None
            raise ModelUnavailable(message) from None

        except TimeoutError:
            raise ModelTimeout("model endpoint timed out") from None

        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)

            if isinstance(reason, TimeoutError):
                raise ModelTimeout(
                    "model endpoint timed out"
                ) from None

            raise ModelUnavailable(
                "cannot reach model endpoint "
                f"({type(reason).__name__})"
            ) from None

        if len(raw) > MAX_RESPONSE_BYTES:
            raise ModelOutputError("model response too large")

        try:
            choice = json.loads(raw)["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelOutputError(
                "unexpected response envelope from model endpoint"
            ) from None

        if not isinstance(content, str):
            raise ModelOutputError(
                "model response content was missing or not text"
            )

        if choice.get("finish_reason") == "length":
            if not allow_truncated:
                raise ModelOutputError("model output reached its completion-token limit")
            # Free-form replies keep what was produced; the branches see the cut (e.g. an unclosed fence).
            content = content.rstrip() + "\n\n[reply cut off at the completion-token limit]"

        executed = choice.get("message", {}).get("executed_tools") or []
        return content, [t for t in executed if isinstance(t, dict)] if isinstance(executed, list) else []