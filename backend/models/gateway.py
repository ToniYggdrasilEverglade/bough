"""Model gateway with terminal and file diagnostics."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..context import CycleContext
from ..errors import (
    ModelFailure,
    ModelOutputError,
    ModelTimeout,
    ModelUnavailable,
    redact,
)
from ..schemas import EventType
from .base import ModelAdapter


def make_logger() -> logging.Logger:
    logger = logging.getLogger("bough.model_debug")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [model] %(message)s"
    )

    terminal = logging.StreamHandler()
    terminal.setFormatter(formatter)
    logger.addHandler(terminal)

    try:
        log_dir = Path("data")
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / "model_debug.log",
            maxBytes=2_000_000,
            backupCount=2,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning(
            "File logging unavailable (%s); terminal logging remains active",
            type(exc).__name__,
        )

    return logger


logger = make_logger()


class ModelGateway:
    def __init__(self, adapter: ModelAdapter, settings: Settings):
        self.adapter = adapter
        self.settings = settings

        logger.info(
            "Gateway initialized: model=%s mock=%s timeout=%ss",
            self._safe(adapter.name),
            adapter.is_mock,
            settings.model_timeout_s,
        )

    def _safe(self, value: Any) -> str:
        return redact(str(value), [self.settings.model_api_key])

    async def call(
        self,
        ctx: CycleContext,
        task: str,
        data: dict,
        validator: Callable[[Any], Any],
    ) -> Any:
        s = self.settings
        code, detail = "unknown", ""

        logger.info("Gateway call entered: task=%s", self._safe(task))

        for attempt in range(1, s.max_retries + 2):
            if not ctx.budget.try_consume():
                info = {
                    "limit": "max_model_calls",
                    "value": ctx.budget.limit,
                    "task": task,
                }
                ctx.limits_hit.append(info)

                logger.error(
                    "Budget exhausted: task=%s used=%s limit=%s",
                    self._safe(task),
                    ctx.budget.used,
                    ctx.budget.limit,
                )

                await ctx.emit(
                    "model_gateway",
                    EventType.LIMIT_REACHED,
                    info,
                )

                raise ModelFailure(
                    "budget_exceeded",
                    f"model call budget of {ctx.budget.limit} reached",
                )

            t0 = time.monotonic()

            logger.info(
                "START task=%s attempt=%s model=%s",
                self._safe(task),
                attempt,
                self._safe(self.adapter.name),
            )

            try:
                raw = await asyncio.wait_for(
                    self.adapter.run(
                        task,
                        data,
                        timeout=s.model_timeout_s,
                    ),
                    timeout=s.model_timeout_s,
                )
                result = validator(raw)

            except asyncio.CancelledError:
                logger.warning(
                    "CANCELLED task=%s attempt=%s",
                    self._safe(task),
                    attempt,
                )
                raise

            except (asyncio.TimeoutError, ModelTimeout):
                code = "timeout"
                detail = f"no answer within {s.model_timeout_s}s"

            except ModelUnavailable as exc:
                code = "unavailable"
                detail = self._safe(exc)

            except ModelOutputError as exc:
                code = "malformed_output"
                detail = self._safe(exc)

            except Exception as exc:
                detail = self._safe(
                    f"{type(exc).__name__}: {exc}"
                )

                logger.error(
                    "UNEXPECTED task=%s attempt=%s\n%s",
                    self._safe(task),
                    attempt,
                    self._safe(traceback.format_exc()),
                )

                await self._log(
                    ctx,
                    task,
                    attempt,
                    t0,
                    False,
                    "unexpected_error",
                    detail,
                )
                raise

            else:
                await self._log(
                    ctx, task, attempt, t0, True, None, ""
                )
                return result

            await self._log(
                ctx,
                task,
                attempt,
                t0,
                False,
                code,
                detail,
            )

            if code == "unavailable":
                break

        logger.error(
            "Giving up: task=%s code=%s detail=%s",
            self._safe(task),
            code,
            self._safe(detail),
        )
        raise ModelFailure(code, detail)

    async def _log(
        self,
        ctx,
        task,
        attempt,
        t0,
        ok,
        code,
        detail,
    ):
        duration_ms = int((time.monotonic() - t0) * 1000)
        safe_detail = self._safe(detail)

        log_method = logger.info if ok else logger.error
        log_method(
            "%s task=%s attempt=%s duration_ms=%s "
            "code=%s detail=%s",
            "SUCCESS" if ok else "FAILED",
            self._safe(task),
            attempt,
            duration_ms,
            code,
            safe_detail,
        )

        await ctx.emit(
            "model_gateway",
            EventType.MODEL_CALL,
            {
                "task": task,
                "attempt": attempt,
                "ok": ok,
                "error_code": code,
                "detail": safe_detail,
                "model": self.adapter.name,
                "is_mock": self.adapter.is_mock,
                "duration_ms": duration_ms,
                "calls_used": ctx.budget.used,
                "calls_limit": ctx.budget.limit,
            },
        )