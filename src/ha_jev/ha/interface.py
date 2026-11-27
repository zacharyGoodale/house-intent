"""HA access interface (spec §9): the HAClient protocol and registry/state value types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class HAError(Exception):
    pass


@dataclass(frozen=True)
class Area:
    id: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    area_id: str | None = None
    manufacturer: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class EntityEntry:
    entity_id: str
    name: str
    area_id: str | None = None  # effective area: the entity's own, else its device's
    device_id: str | None = None
    device_class: str | None = None
    features: frozenset[str] = frozenset()
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    exposed: bool = False
    disabled: bool = False
    hidden: bool = False
    platform: str | None = None
    aliases: tuple[str, ...] = ()
    # The entity's own definition when HA exposes it (script/automation config): used by the decision-6
    # check that refuses routines which reach a lock. None means unknown.
    config: Mapping[str, Any] | None = None
    # The integration's own id for the device (a speaker's hardware id): pairs a speaker with its Music Assistant twin.
    unique_id: str | None = None

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0]


@dataclass(frozen=True)
class Registry:
    areas: Mapping[str, Area]
    devices: Mapping[str, Device]
    entities: Mapping[str, EntityEntry]


@dataclass(frozen=True)
class EntityState:
    entity_id: str
    state: str | None
    attributes: Mapping[str, Any] = field(default_factory=dict)


class HAClient(Protocol):
    async def get_registry(self) -> Registry: ...

    async def get_states(self) -> dict[str, EntityState]: ...

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_ids: Sequence[str],
        data: Mapping[str, Any],
        return_response: bool = False,
    ) -> dict | None: ...


def effective_area(entity_area: str | None, device_id: str | None, devices: Mapping[str, Device]) -> str | None:
    """An entity's area_id overrides its device's (§9.2)."""
    if entity_area:
        return entity_area
    if device_id and device_id in devices:
        return devices[device_id].area_id
    return None
