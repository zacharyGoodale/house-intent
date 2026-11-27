"""Settings from config/{scope,safety,ha_jev}.toml plus env; the opt-in dev env loader (spec §10)."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from ha_jev.capabilities import DEFAULT as DEFAULT_TABLE
from ha_jev.capabilities import CapabilityError, CapabilityTable, rows_from_config
from ha_jev.models import ServiceCall

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
DEV_ENV_WHITELIST = ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "HA_URL", "HA_TOKEN", "HA_JEV_ALLOW_REAL",
                     "HERMES_API_KEY", "HERMES_API_KEY_FILE", "HERMES_BASE_URL")
CALENDAR_SHARES = ("household", "titles_private", "private")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class JevSettings:
    model: str = "jev-latest"
    timeout_s: float = 2.0
    attempt_timeout_s: float = 1.2
    max_retries: int = 1
    keepalive_s: float = 0


@dataclass(frozen=True)
class LunaSettings:
    model: str = "gpt-6-luna"
    effort: str = "none"
    max_output_tokens: int = 120
    timeout_s: float = 5.0
    max_retries: int = 1
    persona: str = "Be warm, friendly and a little casual, like a helpful housemate."
    household: str = ""  # who lives here and where; sent to Luna with every general question


@dataclass(frozen=True)
class HermesSettings:
    relaxed_streaming: bool = False  # shared config with the deployed streaming bridge
    base_url: str = ""
    api_key_file: str = ".env.hermes-key"
    model: str = "hermes-agent"
    timeout_s: float = 300.0
    conversation_idle_s: float = 30.0
    # Configure actual provider prices; zero means unpriced, not free.
    input_per_m: float = 0.0
    cached_per_m: float = 0.0
    output_per_m: float = 0.0


@dataclass(frozen=True)
class PolicySettings:
    intent_floor: float = 0.50
    general_route: float = 0.60
    needs_llm_route: float = 0.70
    security_refuse: float = 0.50
    compound: float = 0.60
    compound_nosplit: float = 0.80
    act: float = 0.70
    high_stakes_act: float = 0.85
    clarify_floor: float = 0.45
    none_confident: float = 0.60
    noul_yes: float = 0.50
    number_setting: float = 0.60
    clarify_min_p: float = 0.15
    clarify_max_options: int = 3
    step_pct: float = 20
    volume_step_pct: float = 10
    kelvin_step: float = 500
    pending_ttl_s: float = 60
    recent_ttl_s: float = 300
    prune_threshold: int = 60
    prune_keep: int = 40
    max_request_tokens: int = 12000
    # Review S3: a light group stands in for its members only when all of them are in the requested rooms.
    # false = the dashboard's rule (a room group drives all its members, wherever they are).
    strict_group_areas: bool = True
    # A room reading with no known room, when `unknown_room` is at least this, is a room this home doesn't have.
    unknown_room: float = 0.50
    # An utterance with "it"/"them", a recent target and no room named: room, house and bare readings are this
    # much less likely than the model says (the pronoun points back).
    anaphor_room_weight: float = 0.2
    # Round 2: a whole-home quantifier ("all", "every", "everything", "any", "everywhere") with no room named
    # counts as this much whole_home evidence.
    quantifier_home: float = 0.9


@dataclass(frozen=True)
class PriceSettings:
    jev_input_per_m: float = 0.042
    luna_input_per_m: float = 0.10
    luna_cached_per_m: float = 0.01
    luna_output_per_m: float = 0.50


@dataclass(frozen=True)
class LogSettings:
    dir: str = "logs"
    enabled: bool = False


@dataclass(frozen=True)
class RealHASettings:
    entity_id_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplySettings:
    llm_phrasing: bool = False


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    aliases: tuple[str, ...]
    calls: tuple[ServiceCall, ...]

    @property
    def key(self) -> str:
        return f"preset.{self.id}"


@dataclass(frozen=True)
class Defaults:
    list_entity: str | None = None  # TOML key `list`
    transport_player: dict[str, str] = field(default_factory=dict)
    high_stakes: list[str] = field(default_factory=list)
    weather: str | None = None  # the weather entity to read when there are several (review D9-5)
    speakers: list[str] = field(default_factory=list)  # extra music speakers the feature rule misses (D9-1)
    # D9-9: switches that never join a room or house sweep (globs), besides the do-not-disturb name pattern,
    # and alarm / timer sensors whose names don't say "alarm" / "timer".
    sweep_exclude: list[str] = field(default_factory=list)
    alarm_sensors: list[str] = field(default_factory=list)
    timer_sensors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MusicConfig:
    """`[music]` in scope.toml: free-form music through a search-and-play integration (Music Assistant).
    Unset `integration` means music is off, and "play <anything>" stays R_PLAY (OQ 29)."""
    integration: str | None = None  # the HA integration domain that owns the search, library and play services
    # speaker entity -> that integration's player entity, for when registry pairing (unique_id, then name) can't tell
    players: dict[str, str] = field(default_factory=dict)
    search_limit: int = 5


@dataclass(frozen=True)
class ScopeConfig:
    include_exposed: bool = True
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    defaults: Defaults = field(default_factory=Defaults)
    presets: list[Preset] = field(default_factory=list)
    music: MusicConfig = field(default_factory=MusicConfig)


@dataclass(frozen=True)
class SafetyConfig:
    denied_domains: list[str] = field(default_factory=list)
    denied_cover_device_classes: list[str] = field(default_factory=list)
    allowed_cover_device_classes: list[str] = field(default_factory=list)
    denied_entities: list[str] = field(default_factory=list)
    denied_name_pattern: str = r"(?i)\b(un)?lock(s|ed|ing)?\b"
    security_area_pattern: str = r"(?i)\bgarage\b|\bgate\b"
    allowed_entities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Person:
    """One household member (config/people.toml). The id arrives with the request, e.g. from a per-person wake word."""
    id: str
    name: str
    temperature_unit: str | None = None  # "C" | "F"; None = the weather entity's own unit
    calendars: list[str] = field(default_factory=list)  # calendar entities that are this person's
    # Who may hear this person's calendar (personal-context.md §4.2): "household" (anyone at home hears titles),
    # "titles_private" (others hear busy times only) or "private" (only this person).
    calendar_share: str = "household"
    # --- profiles (profiles.py): how the speaker is known. `agents` are the HA conversation agent ids that mean
    # this person (one per wake word pipeline, speaker-id.md §3.3); `wake_word` is documentation for that setup.
    wake_word: str | None = None
    agents: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemorySettings:
    """Per-person and household memory (memory.py). The store lives outside the repo, in `dir`."""
    enabled: bool = True
    dir: str = ""  # "" = $HA_JEV_DATA_DIR, else ~/.local/share/ha-jev
    max_fact_chars: int = 200
    max_facts_per_owner: int = 100
    undo_window_s: float = 600  # "forget that" reaches back this far for the speaker's last fact
    read_max: int = 8  # "what do you remember?" reads at most this many, newest first


@dataclass(frozen=True)
class Settings:
    jev: JevSettings
    luna: LunaSettings
    policy: PolicySettings
    prices: PriceSettings
    log: LogSettings
    ha_real: RealHASettings
    replies: ReplySettings
    scope: ScopeConfig
    safety: SafetyConfig
    config_dir: Path
    capabilities: CapabilityTable = DEFAULT_TABLE
    people: dict[str, Person] = field(default_factory=dict)
    memory: MemorySettings = field(default_factory=MemorySettings)
    hermes: HermesSettings = field(default_factory=HermesSettings)

    def person(self, person_id: str | None) -> Person | None:
        return self.people.get(person_id) if person_id else None

    def with_scope_extension(self, include: list[str], exclude: list[str]) -> Settings:
        """Overlay scope: extend (not replace) the checked-in allowlist for one case (§11.2)."""
        if not include and not exclude:
            return self
        scope = replace(self.scope, include=[*self.scope.include, *include], exclude=[*self.scope.exclude, *exclude])
        return replace(self, scope=scope)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"missing config file: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from e


def _build(cls: type, data: dict[str, Any] | None, where: str) -> Any:
    data = data or {}
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown keys in [{where}]: {sorted(unknown)}")
    return cls(**data)


def _parse_scope(raw: dict[str, Any]) -> ScopeConfig:
    raw = dict(raw)
    raw_defaults = dict(raw.pop("defaults", None) or {})
    if "list" in raw_defaults:
        raw_defaults["list_entity"] = raw_defaults.pop("list")
    defaults = _build(Defaults, raw_defaults, "defaults")
    presets = []
    for p in raw.pop("presets", []) or []:
        try:
            calls = tuple(
                ServiceCall(c["domain"], c["service"], tuple(c.get("entity_ids") or ()), c.get("data") or {})
                for c in p.get("calls", [])
            )
            presets.append(Preset(p["id"], p["name"], tuple(p.get("aliases") or ()), calls))
        except KeyError as e:
            raise ConfigError(f"preset missing field {e}") from e
    aliases = {str(k): list(v) for k, v in (raw.pop("aliases", None) or {}).items()}
    music = _build(MusicConfig, raw.pop("music", None), "music")
    scope = _build(ScopeConfig, raw, "scope")
    return replace(scope, defaults=defaults, presets=presets, aliases=aliases, music=music)


def _parse_people(raw: dict[str, Any]) -> dict[str, Person]:
    people = {}
    for pid, data in (raw.get("people") or {}).items():
        if pid == "household":  # reserved: the owner of shared memory facts
            raise ConfigError("people.household: \"household\" is reserved")
        member = _build(Person, {"id": pid, **data}, f"people.{pid}")
        if member.temperature_unit not in (None, "C", "F"):
            raise ConfigError(f"people.{pid}.temperature_unit must be \"C\" or \"F\"")
        if member.calendar_share not in CALENDAR_SHARES:
            raise ConfigError(f"people.{pid}.calendar_share must be one of {', '.join(CALENDAR_SHARES)}")
        people[pid] = member
    claimed: dict[str, str] = {}
    for member in people.values():  # one agent means one person (profiles.py)
        for agent in member.agents:
            if claimed.setdefault(agent, member.id) != member.id:
                raise ConfigError(f"agent {agent!r} is listed for both {claimed[agent]!r} and {member.id!r}")
    return people


def _set_dotted(settings: Settings, key: str, value: Any) -> Settings:
    parts = key.split(".")
    if parts[:3] == ["scope", "defaults", "list"]:
        parts[2] = "list_entity"
    section_alias = {"ha": "ha_real"}
    head = section_alias.get(parts[0], parts[0])
    if head == "ha_real" and len(parts) > 1 and parts[1] == "real":
        parts = parts[1:]

    def set_in(obj: Any, path: list[str]) -> Any:
        name = path[0]
        if is_dataclass(obj):
            if name not in {f.name for f in fields(obj)}:
                raise ConfigError(f"unknown config key: {key}")
            current = getattr(obj, name)
            new = copy.deepcopy(value) if len(path) == 1 else set_in(current, path[1:])
            return replace(obj, **{name: new})
        if isinstance(obj, dict):
            new_dict = dict(obj)
            new_dict[name] = copy.deepcopy(value) if len(path) == 1 else set_in(obj.get(name, {}), path[1:])
            return new_dict
        raise ConfigError(f"cannot set config key: {key}")

    if head not in {f.name for f in fields(Settings)} or head == "config_dir":
        raise ConfigError(f"unknown config key: {key}")
    rest = parts[1:]
    if not rest:
        raise ConfigError(f"config key needs a field: {key}")
    return replace(settings, **{head: set_in(getattr(settings, head), rest)})


def load_settings(config_dir: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    from ha_jev.safety import validate_preset  # local import: safety depends on config types

    directory = Path(config_dir or os.environ.get("HA_JEV_CONFIG_DIR") or DEFAULT_CONFIG_DIR)
    main = _read_toml(directory / "ha_jev.toml")
    scope = _parse_scope(_read_toml(directory / "scope.toml"))
    safety = _build(SafetyConfig, _read_toml(directory / "safety.toml"), "safety")
    table = DEFAULT_TABLE
    extra = directory / "capabilities.toml"
    if extra.is_file():  # decision 9: a new domain is one row, in config
        try:
            table = DEFAULT_TABLE.extend(rows_from_config(_read_toml(extra).get("domains") or []))
        except CapabilityError as e:
            raise ConfigError(f"capabilities.toml: {e}") from e
    people_file = directory / "people.toml"
    people = _parse_people(_read_toml(people_file)) if people_file.is_file() else {}
    ha = main.get("ha") or {}
    settings = Settings(
        jev=_build(JevSettings, main.get("jev"), "jev"),
        luna=_build(LunaSettings, main.get("luna"), "luna"),
        policy=_build(PolicySettings, main.get("policy"), "policy"),
        prices=_build(PriceSettings, main.get("prices"), "prices"),
        log=_build(LogSettings, main.get("log"), "log"),
        ha_real=_build(RealHASettings, ha.get("real"), "ha.real"),
        replies=_build(ReplySettings, main.get("replies"), "replies"),
        scope=scope,
        safety=safety,
        config_dir=directory,
        capabilities=table,
        people=people,
        memory=_build(MemorySettings, main.get("memory"), "memory"),
        hermes=_build(HermesSettings, main.get("hermes"), "hermes"),
    )
    for key, value in (overrides or {}).items():
        settings = _set_dotted(settings, key, value)
    for preset in settings.scope.presets:
        problem = validate_preset(preset, settings.safety, settings.capabilities)
        if problem:
            raise ConfigError(f"preset {preset.id!r} rejected: {problem}")
    return settings


def load_dev_env(path: str | Path | None = None) -> list[str]:
    """Opt-in: read KEY=VALUE lines from $HA_JEV_DEV_ENV, taking only the whitelisted keys.

    Keys are set only when absent from the environment. Values are never logged or written anywhere.
    Returns the names that were set.
    """
    target = path or os.environ.get("HA_JEV_DEV_ENV")
    if not target:
        return []
    p = Path(os.path.expanduser(str(target)))
    if not p.is_file():
        return []
    loaded: list[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in DEV_ENV_WHITELIST or os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            os.environ[key] = value
            loaded.append(key)
    return loaded
