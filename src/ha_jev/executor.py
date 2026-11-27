"""execute(calls, ha) behind safety.guard (spec §6.1 layer 3, §13.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ha_jev.catalog import Catalog
from ha_jev.config import SafetyConfig
from ha_jev.ha.interface import HAClient
from ha_jev.models import ServiceCall
from ha_jev.safety import guard


@dataclass
class ExecutionReport:
    done: list[ServiceCall] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)


async def execute(calls: Sequence[ServiceCall], ha: HAClient, *, catalog: Catalog, safety: SafetyConfig) -> ExecutionReport:
    """Guard the whole list first (raises SafetyViolation, executing nothing), then run the calls in order,
    stopping at the first HA failure."""
    guard(calls, cfg=safety, controllable=catalog.controllable, lookup=catalog.lookup, table=catalog.caps,
          proxies={twin: speaker for speaker, twin in catalog.music_twins.items()},
          proxy_services=catalog.music_play_services())
    report = ExecutionReport()
    for call in calls:
        try:
            await ha.call_service(call.domain, call.service, list(call.entity_ids), dict(call.data))
        except Exception as e:  # noqa: BLE001 - any HA failure stops execution
            report.failed.append({"call": call.to_dict(), "error": f"{type(e).__name__}: {e}"})
            break
        report.done.append(call)
    return report
