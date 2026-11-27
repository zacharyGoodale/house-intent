"""StubHA (spec §9.1): the demo registry plus in-memory state, with overlay application (§11.2).

This module simulates Home Assistant's own service semantics, so (unlike the router) it knows
HA's per-domain services.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ha_jev.ha.interface import Area, Device, EntityEntry, EntityState, HAError, Registry, effective_area
from ha_jev.models import ReadCall, ServiceCall

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "fixtures" / "ha_snapshot.json"
STUB_STATE_PATH = REPO_ROOT / "fixtures" / "stub_state.json"

_FIXTURE_CACHE: dict[Path, Any] = {}


class OverlayError(Exception):
    pass


def _load_json(path: Path) -> Any:
    if path not in _FIXTURE_CACHE:
        _FIXTURE_CACHE[path] = json.loads(path.read_text())
    return copy.deepcopy(_FIXTURE_CACHE[path])


_SERVICES: dict[str, set[str]] = {
    "light": {"turn_on", "turn_off", "toggle"},
    "switch": {"turn_on", "turn_off", "toggle"},
    "automation": {"turn_on", "turn_off", "toggle", "trigger"},
    "cover": {"open_cover", "close_cover", "stop_cover", "set_cover_position", "toggle"},
    "media_player": {"media_play", "media_pause", "media_stop", "media_next_track", "media_previous_track",
                     "volume_set", "volume_mute", "volume_up", "volume_down", "select_source", "turn_on",
                     "turn_off", "play_media", "repeat_set", "shuffle_set"},
    "fan": {"turn_on", "turn_off", "toggle", "set_percentage", "oscillate", "set_direction", "set_preset_mode"},
    "climate": {"turn_on", "turn_off", "set_temperature", "set_hvac_mode"},
    "scene": {"turn_on"},
    "script": {"turn_on", "turn_off"},
    "todo": {"add_item", "get_items"},
    "weather": {"get_forecasts"},
    "lock": {"lock", "unlock", "open"},
    "alarm_control_panel": {"alarm_arm_away", "alarm_arm_home", "alarm_disarm"},
    "button": {"press"},
    "number": {"set_value"},
    "select": {"select_option"},
    "homeassistant": {"turn_on", "turn_off", "toggle"},
    "calendar": {"get_events", "create_event"},
    "music_assistant": {"search", "get_library", "play_media"},
}
_READ_SERVICES = {("weather", "get_forecasts"), ("todo", "get_items"), ("calendar", "get_events"),
                  ("music_assistant", "search"), ("music_assistant", "get_library")}
# Services whose targets belong to another domain: Music Assistant plays on its media_player entities.
_TARGET_DOMAIN = {("music_assistant", "play_media"): "media_player"}
_SEARCH_KEYS = {"artist": "artists", "album": "albums", "track": "tracks", "playlist": "playlists", "radio": "radio",
                "podcast": "podcasts", "audiobook": "audiobooks"}


def _words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold().replace("&", " and ")
    return [w for w in re.findall(r"[\w+]+", s) if w != "the"]


def _entry_from_fixture(e: Mapping[str, Any]) -> EntityEntry:
    return EntityEntry(
        entity_id=e["entity_id"],
        name=(e.get("friendly_name") or e.get("name") or e.get("original_name") or e["entity_id"]).strip(),
        area_id=e.get("area_id"),
        device_id=e.get("device_id"),
        device_class=e.get("device_class"),
        features=frozenset(e.get("features") or ()),
        capabilities=dict(e.get("capabilities") or {}),
        exposed=bool(e.get("exposed", False)),
        disabled=bool(e.get("disabled_by")),
        hidden=bool(e.get("hidden_by")),
        platform=e.get("platform"),
        aliases=tuple(e.get("aliases") or ()),
        config=e.get("config"),
        unique_id=e.get("unique_id"),
    )


def _entry_from_overlay(e: Mapping[str, Any], devices: Mapping[str, Device]) -> EntityEntry:
    if "entity_id" not in e or "friendly_name" not in e:
        raise OverlayError(f"overlay entity needs entity_id and friendly_name: {e}")
    device_id = e.get("device_id")
    if device_id and device_id not in devices:
        raise OverlayError(f"overlay entity {e['entity_id']} has unknown device_id {device_id}")
    return EntityEntry(
        entity_id=e["entity_id"],
        name=str(e["friendly_name"]).strip(),
        area_id=effective_area(e.get("area_id"), device_id, devices),
        device_id=device_id,
        device_class=e.get("device_class"),
        features=frozenset(e.get("features") or ()),
        capabilities=dict(e.get("capabilities") or {}),
        exposed=bool(e.get("exposed", True)),
        disabled=bool(e.get("disabled_by")),
        hidden=bool(e.get("hidden_by")),
        platform=e.get("platform", "overlay"),
        aliases=tuple(e.get("aliases") or ()),
        config=e.get("config"),
        unique_id=e.get("unique_id"),
    )


class StubHA:
    def __init__(
        self,
        registry: Registry,
        states: dict[str, EntityState],
        todo_items: dict[str, list[dict]],
        forecasts: dict[str, dict[str, list[dict]]],
        now: datetime,
        overlay_scope: dict[str, list[str]] | None = None,
        calendar_events: dict[str, list[dict]] | None = None,
        music_library: list[dict] | None = None,
    ) -> None:
        self._registry = registry
        self._states = states
        self._todo = todo_items
        self._forecasts = forecasts
        self._events = calendar_events or {}
        self._music = music_library or []  # MA media items; `in_library` marks the account's own (get_library)
        self.now = now
        self.overlay_scope = overlay_scope or {"include": [], "exclude": []}
        self.calls: list[ServiceCall] = []
        self.attempted_calls: list[ServiceCall] = []
        self.reads: list[ReadCall] = []
        self.fail_on: set[str] = set()

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def default(
        cls,
        overlays: Sequence[Mapping[str, Any]] = (),
        initial_state: Mapping[str, Any] | None = None,
        now: datetime | str | None = None,
        fixture_path: Path = FIXTURE_PATH,
        stub_state_path: Path = STUB_STATE_PATH,
    ) -> StubHA:
        fixture = _load_json(fixture_path)
        stub = _load_json(stub_state_path)
        areas = {a["id"]: Area(a["id"], a["name"], tuple(a.get("aliases") or ())) for a in fixture["areas"]}
        devices = {d["id"]: Device(d["id"], d.get("name") or d["id"], d.get("area_id"), d.get("manufacturer"), d.get("model"))
                   for d in fixture["devices"]}
        entities: dict[str, EntityEntry] = {}
        states: dict[str, EntityState] = {}
        for e in fixture["entities"]:
            entities[e["entity_id"]] = _entry_from_fixture(e)
            states[e["entity_id"]] = EntityState(e["entity_id"], e.get("state"), dict(e.get("attributes") or {}))
        for add in stub.get("registry_additions", []):
            entities[add["entity_id"]] = _entry_from_overlay({"exposed": False, **add}, devices)
        for eid, s in stub["states"].items():
            prev = states.get(eid)
            attrs = dict(prev.attributes) if prev else {}
            attrs.update(s.get("attributes") or {})
            states[eid] = EntityState(eid, s.get("state"), attrs)
        todo = {k: [{"summary": i, "status": "needs_action"} for i in v] for k, v in stub.get("todo_items", {}).items()}
        forecasts = stub.get("forecasts", {})
        events = {k: list(v) for k, v in (stub.get("calendar_events") or {}).items()}
        library = list(stub.get("music_library") or [])
        scope = {"include": [], "exclude": []}

        for ov in overlays:
            cls._apply_overlay(ov, areas, devices, entities, states, scope)
            for eid, evs in (ov.get("calendar_events") or {}).items():
                events.setdefault(eid, []).extend(evs)
            library += list(ov.get("music_library") or [])

        for eid, s in (initial_state or {}).items():
            if not isinstance(s, Mapping):
                raise OverlayError(f"initial_state for {eid} must be an object")
            if "items" in s:
                todo[eid] = [i if isinstance(i, dict) else {"summary": i, "status": "needs_action"} for i in s["items"]]
            if "state" in s or "attributes" in s:
                prev = states.get(eid)
                attrs = dict(prev.attributes) if prev else {}
                attrs.update(s.get("attributes") or {})
                states[eid] = EntityState(eid, s.get("state", prev.state if prev else None), attrs)

        clock = now or stub.get("clock")
        when = datetime.fromisoformat(clock) if isinstance(clock, str) else clock
        registry = Registry(areas=areas, devices=devices, entities=entities)
        return cls(registry, states, todo, forecasts, when, scope, events, library)

    @staticmethod
    def _apply_overlay(ov, areas, devices, entities, states, scope) -> None:
        if not isinstance(ov, Mapping):
            raise OverlayError("overlay must be a JSON object")
        for a in ov.get("areas", []) or []:
            if "id" not in a or "name" not in a:
                raise OverlayError(f"overlay area needs id and name: {a}")
            areas[a["id"]] = Area(a["id"], a["name"], tuple(a.get("aliases") or ()))
        for d in ov.get("devices", []) or []:
            if "id" not in d:
                raise OverlayError(f"overlay device needs id: {d}")
            if d.get("area_id") and d["area_id"] not in areas:
                raise OverlayError(f"overlay device {d['id']} has unknown area_id {d['area_id']}")
            devices[d["id"]] = Device(d["id"], d.get("name") or d["id"], d.get("area_id"), d.get("manufacturer"), d.get("model"))
        for e in ov.get("entities", []) or []:
            if e.get("area_id") and e["area_id"] not in areas:
                raise OverlayError(f"overlay entity {e.get('entity_id')} has unknown area_id {e['area_id']}")
            entry = _entry_from_overlay(e, devices)
            entities[entry.entity_id] = entry
            attrs = {"friendly_name": entry.name}
            caps = entry.capabilities
            for k in ("supported_color_modes", "min_color_temp_kelvin", "max_color_temp_kelvin", "source_list"):
                if k in caps:
                    attrs[k] = caps[k]
            if entry.device_class:
                attrs["device_class"] = entry.device_class
            states[entry.entity_id] = EntityState(entry.entity_id, e.get("state"), attrs)
        for eid in ov.get("remove", []) or []:
            entities.pop(eid, None)
            states.pop(eid, None)
        for eid, s in (ov.get("states") or {}).items():
            prev = states.get(eid)
            attrs = dict(prev.attributes) if prev else {}
            attrs.update(s.get("attributes") or {})
            states[eid] = EntityState(eid, s.get("state", prev.state if prev else None), attrs)
        sc = ov.get("scope") or {}
        scope["include"] += list(sc.get("include") or [])
        scope["exclude"] += list(sc.get("exclude") or [])

    # ---- HAClient -----------------------------------------------------------------------------
    async def get_registry(self) -> Registry:
        return self._registry

    async def get_states(self) -> dict[str, EntityState]:
        return {k: EntityState(v.entity_id, v.state, copy.deepcopy(dict(v.attributes))) for k, v in self._states.items()}

    async def call_service(self, domain, service, entity_ids, data, return_response=False):
        entity_ids = list(entity_ids)
        data = dict(data or {})
        call = ServiceCall(domain, service, tuple(entity_ids), data)
        if (domain, service) not in _READ_SERVICES:
            self.attempted_calls.append(call)  # recorded before validation, so the eval sees every attempt (S-5)
        if service not in _SERVICES.get(domain, set()):
            raise HAError(f"unknown service {domain}.{service}")
        target_domain = _TARGET_DOMAIN.get((domain, service), domain)
        for eid in entity_ids:
            if eid not in self._registry.entities or not eid.startswith(target_domain + "."):
                raise HAError(f"unknown entity {eid} for {domain}.{service}")
        if domain == "music_assistant" and self.config_entry_id_sync(domain) != data.get("config_entry_id", "") \
                and service != "play_media":
            raise HAError("music_assistant: unknown config_entry_id")
        if (domain, service) in _READ_SERVICES:
            self.reads.append(ReadCall("service", domain, service, tuple(entity_ids)))
            if f"{domain}.{service}" in self.fail_on or any(e in self.fail_on for e in entity_ids):
                raise HAError(f"injected failure for {domain}.{service}")
            return self._read(domain, service, entity_ids, data)
        if f"{domain}.{service}" in self.fail_on or any(e in self.fail_on for e in entity_ids):
            raise HAError(f"injected failure for {domain}.{service}")
        for eid in entity_ids:
            self._apply(domain, service, eid, data)
        self.calls.append(call)
        return {} if return_response else None

    def config_entry_id_sync(self, integration: str) -> str | None:
        """A config entry exists for an integration that has entities in the registry (as in HA)."""
        if any(e.platform == integration for e in self._registry.entities.values()):
            return f"stub_{integration}"
        return None

    async def config_entry_id(self, integration: str) -> str | None:
        return self.config_entry_id_sync(integration)

    # ---- effects ------------------------------------------------------------------------------
    def _read(self, domain: str, service: str, entity_ids: list[str], data: dict) -> dict:
        if domain == "weather":
            kind = data.get("type", "daily")
            return {eid: {"forecast": copy.deepcopy(self._forecasts.get(eid, {}).get(kind, []))} for eid in entity_ids}
        if domain == "calendar":
            return {eid: {"events": self._events_between(eid, data)} for eid in entity_ids}
        if domain == "music_assistant" and service == "search":
            return self._search(data)
        if domain == "music_assistant":
            return self._library(data)
        return {eid: {"items": copy.deepcopy(self._todo.get(eid, []))} for eid in entity_ids}

    def _events_between(self, eid: str, data: dict) -> list[dict]:
        """HA's get_events: events overlapping [start, end); all-day events are local dates."""
        start = datetime.fromisoformat(data["start_date_time"])
        end = datetime.fromisoformat(data["end_date_time"])
        tz = self.now.tzinfo

        def when(v: str) -> datetime:
            d = datetime.fromisoformat(v)
            return d.replace(tzinfo=tz) if d.tzinfo is None else d

        out = [e for e in self._events.get(eid, []) if when(e["start"]) < end and when(e["end"]) > start]
        return copy.deepcopy(sorted(out, key=lambda e: when(e["start"])))

    def _search(self, data: dict) -> dict:
        """MA's search, roughly: an item matches when every query word is in its name (or, for a typed search, in
        its name plus artists); exact names first, then fixture order."""
        query = _words(str(data.get("name", "")))
        artist = _words(str(data.get("artist") or ""))
        types = data.get("media_type") or list(_SEARCH_KEYS)
        types = [types] if isinstance(types, str) else list(types)
        limit = int(data.get("limit", 5))
        out: dict[str, list] = {k: [] for k in _SEARCH_KEYS.values()}
        for t in types:
            hits = []
            for i, item in enumerate(self._music):
                if item.get("media_type") != t:
                    continue
                name = _words(item["name"])
                by = [w for a in item.get("artists") or [] for w in _words(a.get("name", ""))]
                if artist and not set(artist) <= set(by):
                    continue
                if query and set(query) <= set(name):
                    rank = 0 if name == query else 1
                elif query and len(types) == 1 and t != "artist" and set(query) <= set(name) | set(by):
                    rank = 2
                else:
                    continue
                hits.append((rank, i, item))
            out[_SEARCH_KEYS[t]] = [self._item(x) for _, _, x in sorted(hits, key=lambda h: h[:2])[:limit]]
        return out

    def _library(self, data: dict) -> dict:
        query = _words(str(data.get("search") or ""))
        t = data.get("media_type")
        items = [x for x in self._music if x.get("in_library") and x.get("media_type") == t
                 and (not query or set(query) <= set(_words(x["name"])))]
        return {"items": [self._item(x) for x in items[: int(data.get("limit", 25))]], "media_type": t}

    @staticmethod
    def _item(x: dict) -> dict:
        return {"media_type": x["media_type"], "uri": x["uri"], "name": x["name"], "version": "", "image": None,
                "artists": [{"media_type": "artist", "uri": a.get("uri", ""), "name": a["name"], "version": "",
                             "image": None} for a in x.get("artists") or []]}

    def _set(self, eid: str, state: str | None = None, **attrs: Any) -> None:
        prev = self._states.get(eid) or EntityState(eid, None, {})
        a = dict(prev.attributes)
        a.update(attrs)
        self._states[eid] = EntityState(eid, prev.state if state is None else state, a)

    def _apply(self, domain: str, service: str, eid: str, data: dict) -> None:
        cur = self._states.get(eid) or EntityState(eid, None, {})
        a = cur.attributes
        if domain == "light":
            members = [m for m in (a.get("entity_id") or []) if m in self._registry.entities]
            if service == "toggle":
                service = "turn_off" if cur.state == "on" else "turn_on"
            if service == "turn_off":
                self._set(eid, "off")
            else:
                self._light_on(eid, data)
            for m in members:
                self._apply(domain, service, m, data)
        elif domain in ("switch", "automation", "fan"):
            if service == "set_percentage":
                pct = int(data.get("percentage", 0))
                self._set(eid, "on" if pct > 0 else "off", percentage=pct)
                return
            if service == "toggle":
                service = "turn_off" if cur.state == "on" else "turn_on"
            if service in ("turn_on", "turn_off"):
                self._set(eid, "on" if service == "turn_on" else "off")
        elif domain == "climate":
            if service == "set_temperature":
                self._set(eid, None, temperature=data["temperature"])
            elif service == "set_hvac_mode":
                self._set(eid, data["hvac_mode"])
            elif service == "turn_off":
                self._set(eid, "off")
            elif service == "turn_on":
                self._set(eid, "cool" if cur.state == "off" else cur.state)
        elif domain == "number" and service == "set_value":
            self._set(eid, str(data["value"]))
        elif domain == "select" and service == "select_option":
            self._set(eid, data["option"])
        elif domain == "cover":
            if service == "open_cover":
                self._set(eid, "open", current_position=100)
            elif service == "close_cover":
                self._set(eid, "closed", current_position=0)
            elif service == "set_cover_position":
                pos = int(data.get("position", 0))
                self._set(eid, "open" if pos > 0 else "closed", current_position=pos)
        elif domain == "media_player":
            if service == "media_play":
                self._set(eid, "playing")
            elif service in ("media_pause", "media_stop"):
                self._set(eid, "paused")
            elif service == "volume_set":
                self._set(eid, None, volume_level=float(data.get("volume_level", 0)))
            elif service == "volume_mute":
                self._set(eid, None, is_volume_muted=bool(data.get("is_volume_muted")))
            elif service == "select_source":
                self._set(eid, "on" if cur.state in (None, "off") else None, source=data.get("source"))
            elif service == "turn_on":
                self._set(eid, "on")
            elif service == "turn_off":
                self._set(eid, "off")
            elif service == "repeat_set":
                self._set(eid, None, repeat=data.get("repeat"))
        elif domain == "music_assistant" and service == "play_media":
            self._set(eid, "playing", media_content_id=data.get("media_id"))
        elif domain == "todo" and service == "add_item":
            self._todo.setdefault(eid, []).append({"summary": data.get("item"), "status": "needs_action"})
        # scene, script, play_media, lock, etc.: recorded only

    def _light_on(self, eid: str, data: dict) -> None:
        cur = self._states.get(eid) or EntityState(eid, None, {})
        a = dict(cur.attributes)
        onoff = list(a.get("supported_color_modes") or []) == ["onoff"]
        if not onoff:
            if a.get("brightness") is None or cur.state != "on":
                a.setdefault("brightness", 255)
                if a.get("brightness") in (None, 0):
                    a["brightness"] = 255
            a.setdefault("color_temp_kelvin", 4000)
            if a.get("color_temp_kelvin") is None:
                a["color_temp_kelvin"] = 4000
            if "brightness_pct" in data:
                a["brightness"] = round(float(data["brightness_pct"]) * 2.55)
            if "brightness_step_pct" in data:
                a["brightness"] = max(0, min(255, round(a["brightness"] + float(data["brightness_step_pct"]) * 2.55)))
            if "color_temp_kelvin" in data:
                a["color_temp_kelvin"] = data["color_temp_kelvin"]
                a.pop("color_name", None)
            if "color_name" in data:
                a["color_name"] = data["color_name"]
        self._states[eid] = EntityState(eid, "on", a)
