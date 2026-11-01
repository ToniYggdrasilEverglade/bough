"""Adapter for OpenAI-compatible chat-completions endpoints."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional

from ..errors import ModelOutputError, ModelTimeout, ModelUnavailable
from .base import ModelAdapter
from .prompts import build_messages


MAX_RESPONSE_BYTES = 1_000_000
# Per-task output budgets; keep planning separate from answer generation.
# Defaults only; Settings overrides these (BRAIN_MODEL_PLAN_TOKENS / _REVIEW_TOKENS / _REPLY_TOKENS).
TASK_COMPLETION_TOKENS = {
    "plan": 1500,
    "review": 600,
    "verbalize": 6000,
}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


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
    ):
        self.url = url
        self.model = model
        self._api_key = api_key
        self.json_mode = json_mode
        self.completion_tokens = dict(TASK_COMPLETION_TOKENS, **(completion_tokens or {}))
        self.name = f"http:{model}"

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

        if self.json_mode:
            body["response_format"] = {"type": "json_object"}


        print(
            f"[request config] model={body['model']} "
            f"task={task} "
            f"max_completion_tokens={body['max_completion_tokens']}",
            flush=True,
        )

        content = await asyncio.to_thread(self._post, body, timeout)
        return extract_json(content)

    def _post(self, body: Dict[str, Any], timeout: float) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "BrainAI/1.0",
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

        if choice.get("finish_reason") == "length":
            raise ModelOutputError("model output reached its completion-token limit")

        if not isinstance(content, str):
            raise ModelOutputError(
                "model response content was missing or not text"
            )

        return content