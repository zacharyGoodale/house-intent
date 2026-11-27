"""Luna client (spec §8): LunaClient protocol, OpenAILuna (Responses API), NullLuna for --no-llm."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ha_jev.config import LunaSettings

INSTRUCTIONS = (
    "You are the voice of a home assistant. {persona}Answer in at most two short sentences of plain text suitable "
    "for text-to-speech: no lists, markdown or emoji. You cannot control devices or read sensors yourself, and never "
    "say that you did. Only if the user asks for something the home can do, say briefly what to ask for instead; "
    "otherwise don't bring up the home's devices or modes. "
    "You don't have live information such as news, prices or weather away from home; say so briefly instead of "
    "guessing. You can't save notes yourself; things to remember are saved only when the user starts with "
    "'remember that'. {household}{home}{memory}{speaker}Local time: {now}. The user is in the {speaker_area}."
)
DISABLED_TEXT = "[llm disabled]"


@dataclass
class LunaReply:
    text: str
    response_id: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    cached_tokens: int = 0
    cost_usd: float | None = None  # adapter override; None uses the Luna price table
    fallback_error: str | None = None


class LunaError(Exception):
    pass


@dataclass(frozen=True)
class LunaProfile:
    """Per-request context from profiles.py and memory.py: household notes first (the cacheable prefix is shared
    by everyone), then who is speaking and their own notes."""
    memory: str = ""  # household notes block, fenced as data
    speaker: str = ""  # the person sentence plus that person's own notes; each block ends with a space


class LunaClient(Protocol):
    async def answer(self, utterance: str, *, now: datetime, speaker_area: str | None,
                     previous_response_id: str | None, home: str | None = None,
                     profile: LunaProfile | None = None) -> LunaReply: ...


def _sentence(text: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return (text if text[-1] in ".!?" else text + ".") + " "


def build_instructions(now: datetime, speaker_area: str | None, settings: LunaSettings | None = None,
                       home: str | None = None, profile: LunaProfile | None = None) -> str:
    """`home` is the catalog's summary of what the home can do (Catalog.luna_summary)."""
    settings = settings or LunaSettings()
    profile = profile or LunaProfile()
    return INSTRUCTIONS.format(persona=_sentence(settings.persona), household=_sentence(settings.household),
                               home=_sentence(home), memory=profile.memory, speaker=profile.speaker,
                               now=now.strftime("%A %Y-%m-%d %H:%M"), speaker_area=speaker_area or "home")


class NullLuna:
    async def answer(self, utterance: str, *, now: datetime, speaker_area: str | None,
                     previous_response_id: str | None, home: str | None = None,
                     profile: LunaProfile | None = None) -> LunaReply:
        return LunaReply(text=DISABLED_TEXT, response_id=None, input_tokens=0, output_tokens=0, latency_ms=0.0,
                         model="disabled")


class OpenAILuna:
    """gpt-6-luna via the Responses API with reasoning effort none. The key is read from OPENAI_API_KEY on first use."""

    def __init__(self, settings: LunaSettings, api_key: str | None = None, client: Any = None) -> None:
        self._settings = settings
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LunaError("OPENAI_API_KEY is not set")
            self._client = AsyncOpenAI(api_key=key, timeout=self._settings.timeout_s, max_retries=self._settings.max_retries)
        return self._client

    async def create(self, **kwargs: Any) -> Any:
        return await self._get_client().responses.create(**kwargs)

    async def answer(self, utterance: str, *, now: datetime, speaker_area: str | None,
                     previous_response_id: str | None, home: str | None = None,
                     profile: LunaProfile | None = None) -> LunaReply:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "instructions": build_instructions(now, speaker_area, self._settings, home, profile),
            "input": utterance,
            "reasoning": {"effort": self._settings.effort},
            "max_output_tokens": self._settings.max_output_tokens,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        start = time.perf_counter()
        try:
            # One deadline for the whole call, SDK retries included (§13.1/13.2; review S-4).
            async with asyncio.timeout(self._settings.timeout_s):
                resp = await self.create(**kwargs)
        except LunaError:
            raise
        except TimeoutError as e:
            raise LunaError(f"luna timed out after {self._settings.timeout_s}s") from e
        except Exception as e:  # noqa: BLE001 - any SDK/network failure becomes R_LUNA_DOWN
            raise LunaError(f"luna failed: {type(e).__name__}") from e
        latency = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        cached = 0
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        text = (getattr(resp, "output_text", "") or "").strip()
        if not text:
            raise LunaError("luna returned no text")
        return LunaReply(text=text, response_id=getattr(resp, "id", None),
                         input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                         output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                         latency_ms=latency, model=getattr(resp, "model", self._settings.model), cached_tokens=cached)


JUDGE_INSTRUCTIONS = (
    "You grade one reply from a home voice assistant against a rubric. Respond with only a JSON object "
    '{"pass": true or false, "why": "<one short sentence>"}.'
)


class LunaJudge:
    """Eval-only rubric judge (spec §11.3). Its cost is reported separately from the product's."""

    def __init__(self, luna: OpenAILuna) -> None:
        self._luna = luna

    async def __call__(self, rubric: str, reply: str) -> tuple[bool, str, dict]:
        import json

        try:
            resp = await self._luna.create(
                model=self._luna._settings.model, instructions=JUDGE_INSTRUCTIONS,
                input=f"Rubric: {rubric}\nReply: {reply}", reasoning={"effort": self._luna._settings.effort},
                max_output_tokens=120)
        except Exception as e:  # noqa: BLE001
            return False, f"judge error: {type(e).__name__}", {}
        usage = getattr(resp, "usage", None)
        u = {"input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
             "output_tokens": int(getattr(usage, "output_tokens", 0) or 0)}
        text = (getattr(resp, "output_text", "") or "").strip()
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            verdict = json.loads(text[start:end])
            return bool(verdict.get("pass")), str(verdict.get("why", "")), u
        except (ValueError, json.JSONDecodeError):
            return False, "judge returned no JSON", u
