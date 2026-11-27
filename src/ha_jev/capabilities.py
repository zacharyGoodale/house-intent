"""The per-domain capability table (spec §4.2): the only domain-specific code.

Each row says what a domain is (its kind, device type and flags), which action Choice it uses, and for every
action option the HA service, the feature it needs, the value it takes and the reply it gives. Call building is
selected by the row's `builder`: `template` builds calls from the action definitions alone, so a new simple
domain (a fan, a valve) is one row, and can even be added from `config/capabilities.toml` without code
(decision 9). `light` and `media` name the two builders with behaviour that data can't express (per-entity
kelvin clamps, proportional master volume); they live in `actions.py` and are chosen here, by row.

Nothing else in the package compares against a domain name: policy, catalog, resolve and questions ask the
table (`CapabilityTable`) instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

_EMPTY: Mapping[str, Any] = MappingProxyType({})

# Special feature names understood by `supports`: "dimmable" = a light with a non-onoff colour mode.
DIMMABLE = "dimmable"


@dataclass(frozen=True)
class ActionDef:
    """One option of a domain's action Choice."""

    service: str | None = None  # the HA service; None when the option is handled elsewhere (see `delegate`)
    feature: str | None = None  # the entity feature it needs, or DIMMABLE
    description: str | None = None  # criteria text in the action Choice
    reply: str = "R_DONE"  # reply template key (replies.py)
    data: Mapping[str, Any] = _EMPTY  # service data; "$value" is the value, "$fraction" is value / 100
    value: str | None = None  # "percent" when the action needs a 0-100 value (a number or a level word)
    toggle: bool = False  # resolved per entity from its state: on -> turn_off, otherwise turn_on
    all_when_unspecified: bool = False  # a bare command acts on the whole house set ("louder": master volume)
    delegate: str | None = None  # "scene_choice": answered by the scene Choice (play a named podcast or playlist script)
    turns_on: bool | None = None  # counts as switching the device on (high-stakes rule); default: service turn_on

    @property
    def switches_on(self) -> bool:
        if self.turns_on is not None:
            return self.turns_on
        return self.service == "turn_on" or self.toggle


@dataclass(frozen=True)
class DomainCaps:
    domain: str
    controllable: bool
    kind: str
    action_question: str | None = None
    actions: Mapping[str, ActionDef] = _EMPTY
    query_attributes: tuple[str, ...] = ()
    builder: str = "template"  # template | light | media (actions.py)
    action_instructions: str | None = None  # "{u}" is replaced by the utterance reference
    kinds_by_device_class: Mapping[str, str] = _EMPTY  # descriptor/reply noun from the device_class (D9-6)
    target_option: bool = False  # listed in the `target` Choice
    mention: bool = False  # gets a `mentioned__<id>` noul
    scene_like: bool = False  # listed in the `scene` Choice
    # Among scene-like rows, a higher value hides a lower one with the same name in the same room (§3.1: a script
    # wraps the scene of the same name, so the script is the candidate).
    name_priority: int = 0
    groupable: bool = False  # groups detected by the `entity_id` attribute
    device_type: str | None = None  # `device_type` Choice option this domain belongs to
    device_type_description: str | None = None
    in_everything: bool = False  # part of "turn everything off" (device_type any)
    list_like: bool = False
    role: str | None = None  # "weather" | "sun" | "media": what a domain is for, so policy never names domains
    area_sweep: str = "all"  # what a room plural addresses: "all" entities of the domain, or "music_speakers"
    implicit_target: bool = False  # bare comfort requests can use the only device without recent context
    unspecified: str = "ask"  # a bare command: "room" | "only_or_room" | "active" | "ask"
    # Decisions 5/6: the security name rule (safety.denied_name_pattern) applies to this domain's names.
    security_name_rule: bool = False
    # The entity physically opens something (covers): refused unless its device_class is on the safety allowlist
    # and its area does not look like a garage or gate (safety.toml).
    opening: bool = False
    # Joins room and house sweeps ("turn off all the switches"). Automations do not: disabling a routine by
    # sweeping is never what "all the switches" means (review S-1). They stay reachable by name.
    sweepable: bool = True
    # Where the entity's definition comes from, for the decision-6 check: "config" (HA's `<domain>/config`
    # WebSocket command, or the fixture's `config`), "state_members" (the state's `entity_id` attribute:
    # the entities a scene sets), or None.
    definition_source: str | None = None


def _a(service: str | None = None, feature: str | None = None, description: str | None = None, **kw: Any) -> ActionDef:
    if "data" in kw:
        kw["data"] = MappingProxyType(dict(kw["data"]))
    return ActionDef(service, feature, description, **kw)


_ONOFF = {
    "turn_on": _a("turn_on", reply="R_ON"),
    "turn_off": _a("turn_off", reply="R_OFF"),
    "toggle": _a("turn_on", toggle=True, reply="R_TOGGLE"),
}
_LIGHT_ACTIONS = {
    "turn_on": _a("turn_on", reply="R_ON"),
    "turn_off": _a("turn_off", reply="R_OFF"),
    "toggle": _a("toggle", reply="R_TOGGLE"),
    "set_brightness": _a("turn_on", DIMMABLE, "set a specific brightness, as a number or a word like half or full",
                         reply="R_BRIGHTNESS", value="percent"),
    "brighten": _a("turn_on", DIMMABLE, "brighter, relative", reply="R_BRIGHTEN"),
    "dim": _a("turn_on", DIMMABLE, "dimmer, relative", reply="R_DIM"),
    "set_color": _a("turn_on", DIMMABLE, {"what": "a named colour, or white",
                                          "not_for": "a colour word that is only part of a device's name"},
                    reply="R_COLOR"),
    "set_color_temp": _a("turn_on", DIMMABLE, "a specific colour temperature: kelvin, or a named white such as warm "
                                               "white, cool white or daylight", reply="R_COLOR_TEMP"),
    "warmer": _a("turn_on", DIMMABLE, {"what": "warmer, more yellow, relative to now",
                                       "not_for": "a named white such as 'warm white', or a kelvin value"},
                 reply="R_WARMER"),
    "cooler": _a("turn_on", DIMMABLE, {"what": "cooler, bluer, more daylight, relative to now",
                                       "not_for": "a named white such as 'cool white', or a kelvin value"},
                 reply="R_COOLER"),
}
_COVER_ACTIONS = {
    "open_cover": _a("open_cover", "open", "open, raise, lift", reply="R_COVER_OPEN"),
    "close_cover": _a("close_cover", "close", "close, draw, lower", reply="R_COVER_CLOSE"),
    "stop_cover": _a("stop_cover", "stop", reply="R_COVER_STOP"),
    "set_cover_position": _a("set_cover_position", "set_position",
                             "a specific position, percent or fraction such as halfway", reply="R_COVER_POSITION",
                             value="percent", data={"position": "$value"}),
}
_MEDIA_ACTIONS = {
    "media_play": _a("media_play", "play", "play, resume", reply="R_MEDIA_PLAY"),
    "media_pause": _a("media_pause", "pause", "pause, stop the music", reply="R_MEDIA_PAUSE"),
    "media_next_track": _a("media_next_track", "next_track", reply="R_MEDIA_NEXT"),
    "media_previous_track": _a("media_previous_track", "previous_track", reply="R_MEDIA_PREV"),
    "volume_set": _a("volume_set", "volume_set", "set a specific volume", reply="R_MEDIA_VOLUME", value="percent",
                     all_when_unspecified=True),
    "volume_up": _a("volume_set", "volume_set", "louder, turn it up", reply="R_MEDIA_UP", all_when_unspecified=True),
    "volume_down": _a("volume_set", "volume_set", "quieter, turn it down", reply="R_MEDIA_DOWN",
                      all_when_unspecified=True),
    "volume_mute": _a("volume_mute", "volume_mute", reply="R_MEDIA_MUTE", all_when_unspecified=True),
    "volume_unmute": _a("volume_mute", "volume_mute", reply="R_MEDIA_UNMUTE", all_when_unspecified=True),
    "select_source": _a("select_source", "select_source", "open an app or switch input", reply="R_MEDIA_SOURCE"),
    "turn_on": _a("turn_on", "turn_on", reply="R_MEDIA_ON"),
    "turn_off": _a("turn_off", "turn_off", reply="R_MEDIA_OFF"),
    "play_content": _a(None, None, "play a specific song, artist, playlist, station or podcast", delegate="scene_choice",
                       all_when_unspecified=True),
}
_CLIMATE_ACTIONS = {
    "turn_on": _a("turn_on", "turn_on", "turn on the AC or heating, resuming the last mode", reply="R_ON"),
    "turn_off": _a("set_hvac_mode", None, "turn off the AC, heater or thermostat", reply="R_OFF",
                   data={"hvac_mode": "off"}),
    "set_temperature": _a("set_temperature", "target_temperature",
                          "set the thermostat or AC to a specific temperature in degrees"),
    "warmer": _a("set_temperature", "target_temperature",
                 "raise the target temperature: warmer, turn the temperature up, too cold; optional number of degrees",
                 reply="R_WARMER"),
    "cooler": _a("set_temperature", "target_temperature",
                 "lower the target temperature: cooler, turn the temperature down, too hot; optional number of degrees",
                 reply="R_COOLER"),
    "heat": _a("set_hvac_mode", None, "switch to heating mode", data={"hvac_mode": "heat"}),
    "cool": _a("set_hvac_mode", None, "switch to cooling mode", data={"hvac_mode": "cool"}),
    "fan_only": _a("set_hvac_mode", None, "fan only without heating or cooling", data={"hvac_mode": "fan_only"}),
}
_COVER_KINDS = {"awning": "awning", "blind": "blinds", "curtain": "curtains", "shade": "shades",
                "shutter": "shutters", "window": "window", "garage": "garage door", "gate": "gate", "door": "door"}

BASE_ROWS: tuple[DomainCaps, ...] = (
    DomainCaps("light", True, "light", "light_action", _LIGHT_ACTIONS, ("on_off", "brightness", "color"),
               builder="light", action_instructions="What should happen to the light(s) in {u}? Brighter, dimmer, warmer "
                                                  "or cooler said about a room, with no other device named, is about "
                                                  "its lights.",
               target_option=True, mention=True, groupable=True, device_type="light",
               device_type_description="lights, lamps, bulbs, and their colour and brightness: brighter, dimmer, "
                                       "warmer or cooler white", in_everything=True, unspecified="room"),
    DomainCaps("switch", True, "switch", "switch_action", _ONOFF, ("on_off",),
               action_instructions="What should happen to the switch or routine in {u}?",
               target_option=True, mention=True, device_type="switch",
               device_type_description="plugs and appliances, do-not-disturb, automations and routines",
               in_everything=True, security_name_rule=True),
    DomainCaps("automation", True, "routine", "switch_action", _ONOFF, ("on_off",),
               action_instructions="What should happen to the switch or routine in {u}?",
               target_option=True, mention=True, device_type="switch", security_name_rule=True, sweepable=False,
               definition_source="config"),
    DomainCaps("cover", True, "cover", "cover_action", _COVER_ACTIONS, ("position", "open_closed"),
               action_instructions="What should happen to the blinds, shades, curtains or other covers in {u}?",
               kinds_by_device_class=MappingProxyType(_COVER_KINDS), target_option=True, mention=True,
               device_type="cover", device_type_description="blinds, shades, curtains, awnings",
               unspecified="only_or_room", security_name_rule=True, opening=True),
    DomainCaps("media_player", True, "speaker", "media_action", _MEDIA_ACTIONS, ("playing", "volume", "on_off"),
               builder="media", action_instructions="What should happen to the speaker or TV in {u}?",
               target_option=True, mention=True, device_type="media_player",
               device_type_description="speakers, TV, music, volume", role="media", area_sweep="music_speakers",
               unspecified="active"),
    DomainCaps("climate", True, "thermostat", "temperature_action", _CLIMATE_ACTIONS,
               ("on_off", "temperature", "target_temperature", "hvac_mode"), builder="climate",
               action_instructions="What should happen to the thermostat in {u}? Bare warmer/cooler or too hot/cold "
                                   "means room temperature; explicit lights or warm/cool white means light colour. "
                                   "Questions about current temperature or settings are device_query, not commands.",
               target_option=True, mention=True, device_type="thermostat",
               device_type_description="indoor room temperature, thermostat, AC, air conditioning, heater, "
                                       "heating/cooling, too hot/cold, warmer/cooler (not light colour or outdoor "
                                       "weather)",
               unspecified="only_or_room", implicit_target=True),
    DomainCaps("scene", True, "scene", None, {"turn_on": _a("turn_on", reply="R_SCENE")}, (), scene_like=True,
               security_name_rule=True, definition_source="state_members"),
    DomainCaps("script", True, "script", None, {"turn_on": _a("turn_on", reply="R_SCENE")}, (), scene_like=True,
               name_priority=1, security_name_rule=True, definition_source="config"),
    DomainCaps("todo", True, "list", None, {"add_item": _a("add_item", reply="R_ADDED")}, ("items",), list_like=True),
    DomainCaps("lock", False, "lock (status only)", None, _EMPTY, ("locked",), target_option=True, device_type="other"),
    DomainCaps("alarm_control_panel", False, "lock (status only)", None, _EMPTY, ("locked",), target_option=True,
               device_type="other"),
    DomainCaps("binary_sensor", False, "sensor", None, _EMPTY, ("open_closed", "motion", "on_off"), target_option=True,
               device_type="other"),
    DomainCaps("sensor", False, "sensor", None, _EMPTY, ("next_alarm", "next_timer", "power", "value"),
               target_option=True, device_type="other"),
    DomainCaps("weather", False, "weather", None, _EMPTY, ("weather",), role="weather"),
    DomainCaps("sun", False, "sun", None, _EMPTY, ("sunrise", "sunset"), role="sun"),
    DomainCaps("person", False, "person", None, _EMPTY, ("presence",), target_option=True, device_type="other"),
)

NEVER_ALLOWED_DOMAINS = frozenset({"homeassistant", "lock", "alarm_control_panel"})
OTHER_DEVICE_TYPE_DESCRIPTION = "door, lock, motion, power use, sensors"
BUILDERS = ("template", "light", "media", "number", "climate")
READ_SERVICES_BY_ROLE = {"weather": ("get_forecasts",), "list": ("get_items",)}

WEATHER_FORECAST_SERVICE = ("weather", "get_forecasts")
TODO_ITEMS_SERVICE = ("todo", "get_items")
TODO_ADD_SERVICE = ("todo", "add_item")
# Calendars are read-only and never candidates (no row here, so Catalog.build skips them; personal-context.md §5.1).
CALENDAR_EVENTS_SERVICE = ("calendar", "get_events")
# Music (docs/usage.md). The domain is the configured integration ([music] integration in
# scope.toml); these are its services. `play_media` targets that integration's media players, the "twins" of the
# speakers in scope, so it is a proxy call: allowed only on a twin whose speaker is controllable (safety.guard).
MUSIC_SEARCH = "search"
MUSIC_LIBRARY = "get_library"
MUSIC_PLAY = "play_media"
MUSIC_PLAY_ROLE = "media"  # the role of the domain whose players MUSIC_PLAY targets

# Switches that are never swept up by area/house scope unless named (do-not-disturb toggles).
AREA_SWEEP_EXCLUDE_NAME = re.compile(r"(?i)\bdo not disturb\b|\bdnd\b")


class CapabilityError(ValueError):
    pass


class CapabilityTable:
    """The table plus everything derived from it. `extend` adds rows (e.g. from config/capabilities.toml)."""

    def __init__(self, rows: Iterable[DomainCaps]) -> None:
        self.rows: dict[str, DomainCaps] = {}
        for r in rows:
            self.rows[r.domain] = r
        self.service_allowlist: frozenset[tuple[str, str]] = frozenset(
            (d, a.service) for d, c in self.rows.items() if c.controllable for a in c.actions.values()
            if a.service and d not in NEVER_ALLOWED_DOMAINS)
        self.service_vocabulary: frozenset[str] = frozenset(
            {a.service for c in self.rows.values() for a in c.actions.values() if a.service}
            | {"get_forecasts", "get_items", "turn_on", "toggle", "get_events", "search", "get_library", "play_media"})
        dt: dict[str, tuple[str, ...]] = {}
        for d, c in self.rows.items():
            if c.device_type:
                dt[c.device_type] = dt.get(c.device_type, ()) + (d,)
        self.device_type_domains: dict[str, tuple[str, ...]] = dt
        self.everything_domains: tuple[str, ...] = tuple(d for d, c in self.rows.items() if c.in_everything)

    def extend(self, rows: Iterable[DomainCaps]) -> CapabilityTable:
        rows = list(rows)
        for r in rows:
            if r.domain in NEVER_ALLOWED_DOMAINS and r.controllable:
                raise CapabilityError(f"domain {r.domain} can never be controllable")
        return CapabilityTable([*self.rows.values(), *rows])

    def only(self, domains: Iterable[str]) -> CapabilityTable:
        """The rows for these domains: a home is asked about, and can act on, only the kinds of device it has."""
        keep = set(domains)
        return CapabilityTable(r for d, r in self.rows.items() if d in keep)

    # ---- lookups ------------------------------------------------------------------------------
    def caps_for(self, domain: str) -> DomainCaps | None:
        return self.rows.get(domain)

    def __contains__(self, domain: str) -> bool:
        return domain in self.rows

    def action_questions(self) -> dict[str, DomainCaps]:
        """Action question id -> the first row that uses it (rows sharing a question share its options)."""
        out: dict[str, DomainCaps] = {}
        for c in self.rows.values():
            if c.action_question and c.action_question not in out:
                out[c.action_question] = c
        return out

    def device_types(self) -> dict[str, str]:
        """device_type option -> description, in table order, then other / any / none."""
        out: dict[str, str] = {}
        for c in self.rows.values():
            if c.device_type and c.device_type != "other" and c.device_type not in out:
                out[c.device_type] = c.device_type_description or c.kind
        out["other"] = OTHER_DEVICE_TYPE_DESCRIPTION
        out["any"] = "everything, all devices"
        out["none"] = "no device"
        return out

    def domains_for_device_type(self, device_type: str | None) -> tuple[str, ...]:
        if device_type == "any":
            return self.everything_domains
        return self.device_type_domains.get(device_type or "", ())

    def action(self, domain: str, action: str) -> ActionDef | None:
        c = self.rows.get(domain)
        return c.actions.get(action) if c else None

    def supports(self, domain: str, action: str, features: frozenset[str] | set[str], onoff_only: bool) -> bool:
        a = self.action(domain, action)
        if a is None or (a.service is None and a.delegate is None):
            return False
        if a.feature is None:
            return True
        if a.feature == DIMMABLE:
            return not onoff_only
        return a.feature in features

    def kind_for(self, domain: str, features: frozenset[str] | set[str], device_class: str | None = None) -> str:
        c = self.rows.get(domain)
        if c is None:
            return domain
        if self.is_tv(domain, features, device_class):
            return "TV"
        if device_class and device_class in c.kinds_by_device_class:
            return c.kinds_by_device_class[device_class]
        return c.kind

    def is_music_speaker(self, domain: str, device_class: str | None, features: frozenset[str] | set[str]) -> bool:
        """A media player that plays music (review D9-1): not a TV, and either a declared speaker that can play,
        or any player with play plus volume_set (HomePod, Cast and AirPlay players often have no device_class)."""
        c = self.rows.get(domain)
        if c is None or c.role != "media" or "play" not in features:
            return False
        if self.is_tv(domain, features, device_class):
            return False
        return device_class == "speaker" or "volume_set" in features

    def is_tv(self, domain: str, features: frozenset[str] | set[str], device_class: str | None = None) -> bool:
        """A TV: device_class tv, or a player with inputs (select_source) that is neither a declared speaker nor a
        queue-based music player. HA's speaker integrations report select_source as soon as the speaker has
        saved favourites (their titles become its source_list), and Music Assistant players list sources too; both
        have a play queue (media_enqueue), which a TV or streaming box doesn't. So "select_source" alone never
        makes a speaker a TV (docs/usage.md)."""
        c = self.rows.get(domain)
        if not c or c.role != "media":
            return False
        if device_class == "tv":
            return True
        if device_class == "speaker" or "media_enqueue" in features:
            return False
        return "select_source" in features

    def is_onoff_only(self, domain: str, capabilities: Mapping[str, Any]) -> bool:
        c = self.rows.get(domain)
        return bool(c and c.groupable) and list((capabilities or {}).get("supported_color_modes") or []) == ["onoff"]

    def with_role(self, role: str) -> tuple[str, ...]:
        return tuple(d for d, c in self.rows.items() if c.role == role)

    # ---- safety -------------------------------------------------------------------------------
    def name_rule_applies(self, domain: str) -> bool:
        if domain == PRESET_DOMAIN:
            return True
        c = self.rows.get(domain)
        return bool(c and c.security_name_rule)

    def is_opening(self, domain: str) -> bool:
        c = self.rows.get(domain)
        return bool(c and c.opening)

    def definition_source(self, domain: str) -> str | None:
        c = self.rows.get(domain)
        return c.definition_source if c else None

    def definition_command(self, domain: str) -> str | None:
        """HA's WebSocket command that returns this domain's entity definition, e.g. for scripts."""
        return f"{domain}/config" if self.definition_source(domain) == "config" else None

    def callable_entity(self, domain: str, service: str) -> str | None:
        """A service that names an entity of its own domain (HA scripts: `script.<object_id>` runs that script)."""
        c = self.rows.get(domain)
        if c is None or c.definition_source != "config" or not c.scene_like:
            return None
        if service in {a.service for a in c.actions.values()} | {"turn_off", "toggle", "reload"}:
            return None
        return f"{domain}.{service}"


PRESET_DOMAIN = "preset"
DEFAULT = CapabilityTable(BASE_ROWS)
TABLE = DEFAULT.rows  # the default rows, for code that only needs to read them
SERVICE_ALLOWLIST = DEFAULT.service_allowlist
SERVICE_VOCABULARY = DEFAULT.service_vocabulary


# ---- rows from config (decision 9: a new domain without code) ---------------------------------------
_ROW_FIELDS = {f for f in DomainCaps.__dataclass_fields__} - {"actions", "kinds_by_device_class", "query_attributes"}
_ACTION_FIELDS = set(ActionDef.__dataclass_fields__)


def rows_from_config(raw: Iterable[Mapping[str, Any]]) -> list[DomainCaps]:
    """Parse `[[domains]]` rows from config/capabilities.toml. Unknown keys and builders are config errors."""
    rows = []
    for r in raw:
        r = dict(r)
        unknown = set(r) - _ROW_FIELDS - {"actions", "kinds_by_device_class", "query_attributes"}
        if unknown or "domain" not in r or "kind" not in r:
            raise CapabilityError(f"bad capability row {r.get('domain')!r}: unknown {sorted(unknown)} or missing domain/kind")
        actions = {}
        for name, a in (r.pop("actions", None) or {}).items():
            a = dict(a or {})
            bad = set(a) - _ACTION_FIELDS
            if bad:
                raise CapabilityError(f"bad action {name!r} in {r['domain']!r}: unknown {sorted(bad)}")
            actions[name] = _a(**a)
        row = DomainCaps(domain=r.pop("domain"), controllable=bool(r.pop("controllable", True)), kind=r.pop("kind"),
                         actions=MappingProxyType(actions),
                         kinds_by_device_class=MappingProxyType(dict(r.pop("kinds_by_device_class", None) or {})),
                         query_attributes=tuple(r.pop("query_attributes", None) or ()), **r)
        if row.builder not in BUILDERS:
            raise CapabilityError(f"unknown builder {row.builder!r} in {row.domain!r}")
        if row.controllable and row.actions and not row.action_question:
            row = replace(row, action_question=f"{row.domain}_action")
        rows.append(row)
    return rows


# ---- module-level helpers (default table) ---------------------------------------------------------
def caps_for(domain: str) -> DomainCaps | None:
    return DEFAULT.caps_for(domain)


def domain_of(entity_id: str) -> str:
    return entity_id.split(".", 1)[0]


def is_timestamp_kind(domain: str, device_class: str | None, name: str, word: str) -> bool:
    """Next-alarm / next-timer sensors: timestamp sensors whose name contains the word (§4.2)."""
    return domain == "sensor" and device_class == "timestamp" and word in name.casefold()


def is_power_sensor(domain: str, device_class: str | None) -> bool:
    return domain == "sensor" and device_class == "power"


def is_motion_sensor(domain: str, device_class: str | None) -> bool:
    return domain == "binary_sensor" and device_class in ("motion", "occupancy")


def is_door_sensor(domain: str, device_class: str | None) -> bool:
    return domain == "binary_sensor" and device_class in ("door", "window", "opening", "garage_door")


def is_cover(domain: str) -> bool:
    return domain == "cover"


def is_lock_like(domain: str) -> bool:
    return domain in ("lock", "alarm_control_panel")


# Feature bitmasks (HA's supported_features) for RealHA's registry mapping.
FEATURE_BITS: dict[str, dict[int, str]] = {
    "light": {4: "effect", 8: "flash", 32: "transition"},
    "cover": {1: "open", 2: "close", 4: "set_position", 8: "stop", 16: "open_tilt", 32: "close_tilt",
              64: "stop_tilt", 128: "set_tilt_position"},
    "media_player": {1: "pause", 2: "seek", 4: "volume_set", 8: "volume_mute", 16: "previous_track",
                     32: "next_track", 128: "turn_on", 256: "turn_off", 512: "play_media", 1024: "volume_step",
                     2048: "select_source", 4096: "stop", 8192: "clear_playlist", 16384: "play",
                     32768: "shuffle_set", 65536: "select_sound_mode", 131072: "browse_media",
                     262144: "repeat_set", 524288: "grouping", 1048576: "media_announce",
                     2097152: "media_enqueue", 4194304: "search_media"},
    "climate": {1: "target_temperature", 2: "target_temperature_range", 8: "fan_mode", 128: "turn_off", 256: "turn_on"},
    "fan": {1: "set_speed", 2: "oscillate", 4: "direction", 8: "preset_mode", 16: "turn_off", 32: "turn_on"},
    "todo": {1: "create_item", 2: "delete_item", 4: "update_item", 8: "move_item", 16: "set_due_date_on_item",
             32: "set_due_datetime_on_item", 64: "set_description_on_item"},
    "weather": {1: "forecast_daily", 2: "forecast_hourly", 4: "forecast_twice_daily"},
}


def features_from_bits(domain: str, bits: int) -> list[str]:
    table = FEATURE_BITS.get(domain, {})
    return [name for bit, name in sorted(table.items()) if bits & bit]


# Static state attributes that describe capabilities (copied into EntityEntry.capabilities by RealHA).
CAPABILITY_ATTRIBUTES = ("supported_color_modes", "min_color_temp_kelvin", "max_color_temp_kelvin",
                         "source_list", "effect_list", "min", "max", "step", "unit_of_measurement", "options",
                         "min_temp", "max_temp", "target_temp_step", "temperature_unit", "hvac_modes")
