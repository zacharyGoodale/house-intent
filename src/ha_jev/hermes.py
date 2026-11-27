"""Private Hermes Responses endpoint, with Luna failover and isolated response chains."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from ha_jev.config import REPO_ROOT, HermesSettings, LunaSettings, Settings
from ha_jev.luna import LunaClient, LunaError, LunaProfile, LunaReply, NullLuna, OpenAILuna, _sentence


def build_hermes_instructions(now: datetime, speaker_area: str | None, voice: LunaSettings,
                              home: str | None, profile: LunaProfile | None) -> str:
    profile = profile or LunaProfile()
    return (
        "Reply briefly in plain text suitable for speech. "
        "For a temperature question without an explicit time, give today's forecast high and low and "
        "call out any chance of rain, however small. Use live forecast data; report missing data as unavailable. "
        "Explicit current-temperature requests should still get the current temperature. "
        "For Jev alarms and timers, use your Home Assistant tools: discover the Jev next alarm "
        "and next timer timestamp sensors; their schedules attributes contain "
        "IDs, labels, UTC times, owners and playback targets. Call ha_jev.manage_schedule with "
        "kind=alarm or timer, and op=create, update or delete. "
        "Read the sensors first and use id to update or delete; ask which one if ambiguous. "
        "For create supply seconds or a future ISO at timestamp with timezone and an area_id or device_id "
        "for the alert. Updates accept seconds, at, hour/minute/tomorrow, or new_label and retain the ID "
        "and playback target. The HA tool must authenticate as the schedule owner. "
        "The native HA tool supports plain service calls without return_response. After a successful call, "
        "reread the sensor schedules to verify the change before confirming; report tool or permission errors. "
        + _sentence(voice.household) + _sentence(home) + profile.memory + profile.speaker
        + f"Local time: {now.strftime('%A %Y-%m-%d %H:%M')}. The user is in the {speaker_area or 'home'}."
    )


class HermesClient:
    sticky_conversation = True
    def __init__(self, settings: HermesSettings, voice: LunaSettings, *, transport: Any = None) -> None:
        self.settings = settings
        self.voice = voice
        self.transport = transport

    def _key(self) -> str:
        key = os.environ.get("HERMES_API_KEY", "").strip()
        if key:
            return key
        path = Path(os.environ.get("HERMES_API_KEY_FILE") or self.settings.api_key_file).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            key = path.read_text().strip()
        except OSError as exc:
            raise LunaError("Hermes key unavailable; set HERMES_API_KEY or HERMES_API_KEY_FILE") from exc
        if not key:
            raise LunaError("Hermes key is empty")
        return key

    async def answer(self, utterance: str, *, now: datetime, speaker_area: str | None,
                     previous_response_id: str | None, home: str | None = None,
                     profile: LunaProfile | None = None) -> LunaReply:
        base = (os.environ.get("HERMES_BASE_URL") or self.settings.base_url).rstrip("/")
        if not base:
            raise LunaError("Hermes base_url is not configured")
        instructions = build_hermes_instructions(now, speaker_area, self.voice, home, profile)
        payload: dict[str, Any] = {
            "model": self.settings.model, "input": utterance, "instructions": instructions, "store": True,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        start = time.perf_counter()
        key = self._key()
        try:
            async with asyncio.timeout(self.settings.timeout_s):
                async with httpx.AsyncClient(timeout=self.settings.timeout_s, transport=self.transport) as client:
                    response = await client.post(base + "/responses", json=payload,
                                                 headers={"Authorization": "Bearer " + key})
                    response.raise_for_status()
                    data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LunaError(f"Hermes returned HTTP {exc.response.status_code}") from exc
        except Exception as exc:
            # Never include response bodies, URLs or request headers in an error/log.
            raise LunaError(f"Hermes failed: {type(exc).__name__}") from exc
        try:
            if data.get("status") not in (None, "completed"):
                raise ValueError("incomplete response")
            text = "".join(
                part["text"] for item in data.get("output", [])
                if item.get("type") == "message" and item.get("role") == "assistant"
                and item.get("phase") != "commentary"
                for part in item.get("content", []) if part.get("type") == "output_text"
            ).strip()
            if not text:
                raise ValueError("empty answer")
            usage = data.get("usage") or {}
            tokens_in = int(usage.get("input_tokens", 0))
            tokens_out = int(usage.get("output_tokens", 0))
            cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens")
                         or usage.get("cache_read_tokens") or 0)
            cost = (max(0, tokens_in - cached) * self.settings.input_per_m
                    + cached * self.settings.cached_per_m + tokens_out * self.settings.output_per_m) / 1e6
            runtime = data.get("runtime") or {}
            return LunaReply(text, data.get("id"), tokens_in, tokens_out,
                             (time.perf_counter() - start) * 1000,
                             "hermes/" + (runtime.get("model") or data.get("model") or self.settings.model),
                             cached_tokens=cached, cost_usd=cost)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise LunaError("Hermes returned an invalid or empty answer") from exc


class HermesWithLuna:
    """Try Hermes on each turn, then Luna; never send an ID to the wrong provider."""
    sticky_conversation = True

    def __init__(self, hermes: LunaClient, luna: LunaClient) -> None:
        self.hermes = hermes
        self.luna = luna
        self._chains: dict[str, dict[str, str]] = {}

    async def answer(self, utterance: str, *, previous_response_id: str | None, **context: Any) -> LunaReply:
        chain = dict(self._chains.get(previous_response_id or "", {}))
        start = time.perf_counter()
        fallback_error = None
        try:
            reply = await self.hermes.answer(utterance, previous_response_id=chain.get("hermes"), **context)
            provider = "hermes"
        except LunaError as hermes_error:
            fallback_error = str(hermes_error)
            try:
                reply = await self.luna.answer(utterance, previous_response_id=chain.get("luna"), **context)
                provider = "luna"
            except LunaError as exc:
                raise LunaError("Hermes and Luna are both unavailable") from exc
        if reply.response_id:
            chain[provider] = reply.response_id
            response_id = provider + ":" + reply.response_id
            self._chains[response_id] = chain
        else:
            response_id = previous_response_id
        return replace(reply, response_id=response_id, latency_ms=(time.perf_counter() - start) * 1000,
                       fallback_error=fallback_error)


def make_fallback(settings: Settings, backend: str, no_llm: bool = False) -> LunaClient:
    if no_llm:
        return NullLuna()
    luna = OpenAILuna(settings.luna)
    if backend == "luna":
        return luna
    return HermesWithLuna(HermesClient(settings.hermes, settings.luna), luna)
