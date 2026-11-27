"""A small interactive terminal interface for Core."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from ha_jev.models import Result


class CommandCore(Protocol):
    async def handle(self, text: str, speaker_area: str | None = None, session_id: str | None = None,
                     person: str | None = None) -> Result: ...


def format_result(result: Result) -> list[str]:
    lines = [f"[{result.outcome}] {result.reply}"]
    if result.debug.luna:
        trace = result.debug.luna
        label = " • Hermes unavailable, used Luna" if trace.error else ""
        lines.append(f"  {trace.model} • {trace.latency_ms:.0f} ms{label}")
    lines.extend(f"  ✓ {call.describe()}" for call in result.service_calls)
    if result.clarify_options:
        lines.append(f"  Options: {', '.join(result.clarify_options)}")
    if result.debug.failed_calls:
        lines.append("  Some Home Assistant calls failed; see the request log for details.")
    return lines


async def run(core: CommandCore, *, ha_name: str, area: str | None = None, person: str | None = None,
              fallback_name: str = "luna", reader: Callable[[str], str] = input, writer: Callable[[str], None] = print) -> None:
    """Keep one Core and session alive while the user enters commands."""
    writer(f"ha-jev • {ha_name} • area: {area or 'any'} • person: {person or 'unknown'} • answers: {fallback_name}")
    writer("Type a home command. /help for controls; /quit to exit.")
    while True:
        try:
            command = (await asyncio.to_thread(reader, "home> ")).strip()
        except (EOFError, KeyboardInterrupt):
            writer("")
            return
        if not command:
            continue
        if command in ("/quit", "/exit"):
            return
        if command == "/help":
            writer("Enter a natural-language command or a reply to a clarification. /quit exits.")
            continue
        try:
            result = await core.handle(command, speaker_area=area, session_id="terminal", person=person)
        except Exception as exc:  # Keep the session usable after a command error.
            writer(f"[error] {type(exc).__name__}: {exc}")
            continue
        for line in format_result(result):
            writer(line)
