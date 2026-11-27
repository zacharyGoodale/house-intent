"""Safety layers (spec §6.1): denylist evaluation, the lock-keyword pre-gate, and the executor guard.

Nothing a model says can bypass `guard`: every call list (including presets and completed
clarifications) passes through it immediately before execution.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Set
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ha_jev import capabilities as caps
from ha_jev.models import ServiceCall

if TYPE_CHECKING:
    from ha_jev.config import Preset, SafetyConfig

# Layer 1 keywords (spec §5.3 K, amended for decision 9 / review D9-4). "garage" and "gate" alone are rooms or
# light names ("turn on the garage lights"); they refuse only with an actuating verb, or as "garage door".
_OPEN_VERBS = r"(?:open|opening|close|closing|shut|raise|lift|lower|up|down|toggle|trigger|press|operate)"
LOCK_KEYWORDS = re.compile(
    r"\b(un)?lock(s|ing)?\b|\bdead ?bolt\b|\bdisarm\b|\barm the alarm\b|\blet me in\b|\bgarage doors?\b"
    rf"|\b{_OPEN_VERBS}\b.*\b(garage|gates?)\b|\b(garage|gates?)\b.*\b{_OPEN_VERBS}\b",
    re.IGNORECASE,
)


class SafetyViolation(Exception):
    def __init__(self, message: str, *, denylist: bool) -> None:
        super().__init__(message)
        self.denylist = denylist


@dataclass(frozen=True)
class EntityFacts:
    """What the denylist needs to know about an entity."""

    name: str = ""
    device_class: str | None = None
    area_name: str | None = None
    # The entity's own definition (script sequence, scene members, automation actions) touches a denied
    # domain or entity (decision 6), as found by `config_touches_denied` at catalog build.
    touches_denied: bool = False


Lookup = Callable[[str], EntityFacts | None]


def has_lock_keyword(text: str) -> bool:
    return bool(LOCK_KEYWORDS.search(text))


def pre_gate(text: str, question_form: bool) -> bool:
    """Layer 1: True means refuse before calling Jev."""
    return has_lock_keyword(text) and not question_form


def is_denied_entity(entity_id: str, cfg: SafetyConfig, facts: EntityFacts | None = None,
                     table: caps.CapabilityTable | None = None) -> bool:
    """Decisions 3, 5 and 6. Hard rules (denied domain, denied entity, a definition that touches a denied
    entity, a denied cover device_class) always apply. The heuristics (a cover with an unknown device_class,
    a cover in a garage- or gate-like area, the security name rule) can be waived per entity with
    `allowed_entities` in safety.toml: that is how the user "decides otherwise" (decision 5)."""
    table = table or caps.DEFAULT
    domain = caps.domain_of(entity_id)
    if domain in cfg.denied_domains or domain in caps.NEVER_ALLOWED_DOMAINS:
        return True
    if entity_id in cfg.denied_entities:
        return True
    facts = facts or EntityFacts()
    if facts.touches_denied:
        return True
    opening = table.is_opening(domain)
    if opening and facts.device_class in cfg.denied_cover_device_classes:
        return True
    if entity_id in cfg.allowed_entities:
        return False
    if opening:
        if facts.device_class not in cfg.allowed_cover_device_classes:
            return True
        if facts.area_name and re.search(cfg.security_area_pattern, facts.area_name):
            return True
    if table.name_rule_applies(domain) and name_matches_security(entity_id, facts.name, cfg):
        return True
    return False


# ---- definitions (decision 6: "any script, scene or automation whose purpose is to lock/unlock") ---------
# Keys whose subtree only *tests* state; reading a lock in a condition is not actuating it.
_CONDITION_KEYS = frozenset({"condition", "conditions", "if", "while", "until", "value_template", "wait_template",
                             "trigger", "triggers", "variables"})
_SERVICE_KEYS = frozenset({"action", "service"})
_ENTITY_KEYS = frozenset({"entity_id", "entity_ids", "scene"})
_REF = re.compile(r"\b([a-z_]+)\.([a-z0-9_]+)\b")


@dataclass
class DefinitionRefs:
    domains: set[str]
    entities: set[str]


def definition_refs(definition: Any, table: caps.CapabilityTable | None = None) -> DefinitionRefs:
    """The service domains and entity ids a definition acts on. Templated service names are scanned as
    text, so `{{ 'lock.' ~ 'unlock' }}` still counts. Condition and trigger subtrees are skipped."""
    table = table or caps.DEFAULT
    refs = DefinitionRefs(set(), set())

    def add_services(text: str) -> None:
        for domain, service in _REF.findall(text):
            refs.domains.add(domain)
            target = table.callable_entity(domain, service)
            if target:
                refs.entities.add(target)
        if "{{" in text or "{%" in text:  # a template: count any domain name mentioned in it
            refs.domains.update(re.findall(r"[a-z_]+", text))

    def add_entities(value: Any) -> None:
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                refs.entities.update(f"{d}.{o}" for d, o in _REF.findall(item))

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if k in _CONDITION_KEYS:
                    continue
                if k in _SERVICE_KEYS and isinstance(v, str):
                    add_services(v)
                elif k == "domain" and isinstance(v, str):
                    refs.domains.add(v)
                elif k in _ENTITY_KEYS:
                    add_entities(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            add_entities(node)

    walk(definition)
    return refs


def definitions_touching_denied(definitions: Mapping[str, Any], denied: Set[str], cfg: SafetyConfig,
                                table: caps.CapabilityTable | None = None) -> set[str]:
    """Fixpoint: the entities whose definition acts on a denied domain or entity, directly or through another
    definition (a script that runs a script that unlocks)."""
    refs = {eid: definition_refs(d, table) for eid, d in definitions.items() if d is not None}
    bad_domains = set(cfg.denied_domains) | caps.NEVER_ALLOWED_DOMAINS
    out: set[str] = set()
    changed = True
    while changed:
        changed = False
        for eid, r in refs.items():
            if eid in out:
                continue
            if r.domains & bad_domains or r.entities & (set(denied) | out) \
                    or any(caps.domain_of(e) in bad_domains for e in r.entities):
                out.add(eid)
                changed = True
    return out


def name_matches_security(entity_id: str, name: str | None, cfg: SafetyConfig) -> bool:
    """The denied_name_pattern on the entity_id's object id (underscores as spaces) or the friendly name."""
    pattern = re.compile(cfg.denied_name_pattern)
    object_id = entity_id.split(".", 1)[-1].replace("_", " ")
    return bool(pattern.search(object_id) or (name and pattern.search(name)))


def _has_kelvin_key(data: Any) -> bool:
    if isinstance(data, Mapping):
        return any(k == "kelvin" or _has_kelvin_key(v) for k, v in data.items())
    if isinstance(data, list):
        return any(_has_kelvin_key(v) for v in data)
    return False


def call_problem(call: ServiceCall, cfg: SafetyConfig, lookup: Lookup | None = None,
                 table: caps.CapabilityTable | None = None) -> tuple[str, bool] | None:
    """Static checks on one call. Returns (reason, is_denylist_hit) or None."""
    table = table or caps.DEFAULT
    if call.domain in caps.NEVER_ALLOWED_DOMAINS or call.domain in cfg.denied_domains:
        return (f"domain {call.domain} is never allowed", True)
    for eid in call.entity_ids:
        if is_denied_entity(eid, cfg, lookup(eid) if lookup else None, table):
            return (f"{eid} is denied", True)
    if (call.domain, call.service) not in table.service_allowlist:
        return (f"service {call.domain}.{call.service} is not allowlisted", False)
    if _has_kelvin_key(call.data):
        return ("`kelvin` is banned; use color_temp_kelvin", False)
    if not call.entity_ids:
        return ("call has no targets", False)
    for eid in call.entity_ids:
        if caps.domain_of(eid) != call.domain:
            return (f"{eid} does not match call domain {call.domain}", False)
    return None


def is_denied_call(call: ServiceCall, cfg: SafetyConfig, lookup: Lookup | None = None,
                   table: caps.CapabilityTable | None = None) -> bool:
    """For the eval's independent safety check: any denylist hit, lock.* or alarm_control_panel.*."""
    problem = call_problem(call, cfg, lookup, table)
    return bool(problem and problem[1])


READ_SERVICES = frozenset({caps.WEATHER_FORECAST_SERVICE, caps.TODO_ITEMS_SERVICE, caps.CALENDAR_EVENTS_SERVICE})


def music_reads(integration: str | None) -> frozenset[tuple[str, str]]:
    """The music integration's read-only services (search, library), when music is configured."""
    if not integration or integration in caps.NEVER_ALLOWED_DOMAINS:
        return frozenset()
    return frozenset({(integration, caps.MUSIC_SEARCH), (integration, caps.MUSIC_LIBRARY)})


def guard_read(domain: str, service: str | None, extra: Set[tuple[str, str]] = frozenset()) -> None:
    """Reads may only use the read-only services (weather forecasts, todo items, calendar events, and the music
    integration's search and library when configured)."""
    if (domain, service) not in READ_SERVICES and (domain, service) not in extra:
        raise SafetyViolation(f"read {domain}.{service} is not allowed", denylist=domain in caps.NEVER_ALLOWED_DOMAINS)


def validate_preset(preset: Preset, cfg: SafetyConfig, table: caps.CapabilityTable | None = None) -> str | None:
    if name_matches_security(f"{caps.PRESET_DOMAIN}.{preset.id}", preset.name, cfg) or any(
            re.search(cfg.denied_name_pattern, a) for a in preset.aliases):
        return f"preset name {preset.name!r} matches the security name rule"
    for call in preset.calls:
        problem = call_problem(call, cfg, None, table)
        if problem:
            return problem[0]
    return None


def _proxy_problem(call: ServiceCall, cfg: SafetyConfig, controllable: Set[str], proxies: Mapping[str, str],
                   lookup: Lookup | None, table: caps.CapabilityTable, target_domains: Set[str]) -> tuple[str, bool] | None:
    """A proxy call (the music integration's play_media on a speaker's twin player): every target must be a
    known twin of the target domain, and the speaker it stands for must be controllable and not denied."""
    if call.domain in caps.NEVER_ALLOWED_DOMAINS or call.domain in cfg.denied_domains:
        return (f"domain {call.domain} is never allowed", True)
    if _has_kelvin_key(call.data):
        return ("`kelvin` is banned; use color_temp_kelvin", False)
    if not call.entity_ids:
        return ("call has no targets", False)
    for eid in call.entity_ids:
        speaker = proxies.get(eid)
        if speaker is None or caps.domain_of(eid) not in target_domains:
            return (f"{eid} is not a player {call.domain}.{call.service} may target", False)
        for e in (eid, speaker):
            if is_denied_entity(e, cfg, lookup(e) if lookup else None, table):
                return (f"{e} is denied", True)
        if speaker not in controllable:
            return (f"{speaker} is not controllable", False)
    return None


def guard(calls: Iterable[ServiceCall], *, cfg: SafetyConfig, controllable: Set[str], lookup: Lookup | None = None,
          table: caps.CapabilityTable | None = None, proxies: Mapping[str, str] | None = None,
          proxy_services: Set[tuple[str, str]] = frozenset()) -> None:
    """Layer 3. Raises SafetyViolation if any call is not permitted; checks the whole list first.
    `proxy_services` (the music play service) may target only the `proxies` (twin player -> speaker)."""
    table = table or caps.DEFAULT
    target_domains = set(table.with_role(caps.MUSIC_PLAY_ROLE))
    for call in calls:
        if (call.domain, call.service) in proxy_services:
            problem = _proxy_problem(call, cfg, controllable, proxies or {}, lookup, table, target_domains)
            if problem:
                raise SafetyViolation(problem[0], denylist=problem[1])
            continue
        problem = call_problem(call, cfg, lookup, table)
        if problem:
            raise SafetyViolation(problem[0], denylist=problem[1])
        for eid in call.entity_ids:
            if eid not in controllable:
                raise SafetyViolation(f"{eid} is not controllable", denylist=False)
