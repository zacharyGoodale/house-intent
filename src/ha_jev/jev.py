"""Jev client (spec §5.1): the JevClient protocol, answer types, and the TypeSafe SDK implementation."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ha_jev.config import JevSettings


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def p(self, option: str) -> float:
        return float(self.probabilities.get(option, 1.0 if option == self.choice and not self.probabilities else 0.0))

    def to_dict(self) -> dict:
        return {"type": "choice", "choice": self.choice, "confidence": self.confidence, "probabilities": dict(self.probabilities)}


@dataclass(frozen=True)
class NoulAnswer:
    p: float

    def to_dict(self) -> dict:
        return {"type": "noul", "noul": self.p}


Answer = ChoiceAnswer | NoulAnswer


@dataclass
class JevResponse:
    answers: dict[str, Answer]
    model: str
    request_id: str | None
    input_tokens: int
    latency_ms: float
    attempts: int


class JevError(Exception):
    def __init__(self, message: str, *, status: int | None = None, body: Any = None, attempts: int = 1,
                 latency_ms: float = 0.0) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.attempts = attempts
        self.latency_ms = latency_ms


class JevClient(Protocol):
    async def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse: ...


_RETRYABLE_STATUS = {408, 429, 529}


class TypeSafeJevClient:
    """One Jev request per call. Retries once on 408/429/529/5xx/timeout/connection errors within the
    overall deadline; never retries 401/422. The API key is read from TYPESAFE_API_KEY on first use."""

    def __init__(self, settings: JevSettings, api_key: str | None = None) -> None:
        self._settings = settings
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

            key = self._api_key or os.environ.get("TYPESAFE_API_KEY")
            if not key:
                raise JevError("TYPESAFE_API_KEY is not set")
            self._client = AsyncTypeSafeClient(
                api_key=key,
                model=self._settings.model,
                retry=RetryPolicy(max_retries=0),
                timeout=self._settings.attempt_timeout_s,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse:
        from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

        client = self._get_client()
        start = time.perf_counter()
        deadline = start + self._settings.timeout_s
        attempts = 0
        last: Exception | None = None
        while attempts <= self._settings.max_retries:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.05:
                break
            attempts += 1
            try:
                async with asyncio.timeout(min(remaining, self._settings.attempt_timeout_s)):
                    resp = await client.system_one(state=state, questions=questions)
            except TypeSafeAPIError as e:
                last = e
                status = getattr(e, "status", None)
                if status in _RETRYABLE_STATUS or (status is not None and status >= 500):
                    continue
                raise JevError(f"jev http {status}", status=status, body=getattr(e, "body", None),
                               attempts=attempts, latency_ms=(time.perf_counter() - start) * 1000) from e
            except (TypeSafeAPIConnectionError, TimeoutError) as e:
                last = e
                continue
            latency = (time.perf_counter() - start) * 1000
            answers: dict[str, Answer] = {}
            for qid, a in resp.answers.items():
                if getattr(a, "type", None) == "choice":
                    answers[qid] = ChoiceAnswer(a.choice, float(a.confidence), {k: float(v) for k, v in a.probabilities.items()})
                elif getattr(a, "type", None) == "noul":
                    answers[qid] = NoulAnswer(float(a.noul))
            try:
                request_id = resp.request_id
            except Exception:  # noqa: BLE001 - header may be absent
                request_id = None
            return JevResponse(answers=answers, model=resp.model, request_id=request_id,
                               input_tokens=int(resp.usage.input_tokens or 0), latency_ms=latency, attempts=attempts)
        status = getattr(last, "status", None)
        raise JevError(f"jev unavailable: {type(last).__name__ if last else 'deadline'}", status=status,
                       attempts=attempts, latency_ms=(time.perf_counter() - start) * 1000)
