"""Catalog.build(registry, states, scope, safety): candidates built only from registry data plus config (§3, §4)."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ha_jev import capabilities as caps
from ha_jev.config import Preset, SafetyConfig, ScopeConfig
from ha_jev.ha.interface import EntityState, Registry
from ha_jev.music import pair_players
from ha_jev.safety import EntityFacts, definitions_touching_denied, is_denied_entity


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


@dataclass(frozen=True)
class AreaInfo:
    id: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogEntity:
    entity_id: str
    domain: str
    name: str
    kind: str
    area_id: str | None
    aliases: tuple[str, ...]
    members: tuple[str, ...] | None  # None: not a group
    onoff_only: bool
    features: frozenset[str]
    device_class: str | None
    capabilities: Mapping[str, Any]
    device_id: str | None
    controllable: bool
    denied: bool
    descriptor: str
    touches_denied: bool = False

    @property
    def is_group(self) -> bool:
        return self.members is not None


@dataclass
class Catalog:
    areas: dict[str, AreaInfo]
    entities: dict[str, CatalogEntity]  # the scope: everything queryable
    controllable: frozenset[str]
    presets: dict[str, Preset]
    preset_descriptors: dict[str, str]
    preset_areas: dict[str, str | None]
    scene_options: list[str]
    target_options: list[str]
    mention_options: list[str]
    sources: list[str]
    todo_lists: list[str]
    music_speakers: list[str]
    defaults: Any
    registry_names: dict[str, str] = field(default_factory=dict)
    caps: caps.CapabilityTable = caps.DEFAULT
    registry_areas: dict[str, str | None] = field(default_factory=dict)
    # Music (music.py): speaker in scope -> its player in the music integration, which is never a candidate.
    music_twins: dict[str, str] = field(default_factory=dict)
    music_integration: str | None = None
    music_limit: int = 5
    # Scenes of the integrations in scope.toml `qualified_scene_platforms` -> that integration: they are offered
    # only when the utterance names it ("Hue scene relax"), and the rest only when it names none.
    qualified_scenes: Mapping[str, str] = field(default_factory=dict)

    def scene_options_for(self, utterance: str) -> list[str]:
        said = {p for p in set(self.qualified_scenes.values())
                if re.search(rf"\b{re.escape(p)}\s+scenes?\b", utterance, re.IGNORECASE)}
        if said:
            return [key for key in self.scene_options if self.qualified_scenes.get(key) in said]
        return [key for key in self.scene_options if key not in self.qualified_scenes]

    def music_play_services(self) -> frozenset[tuple[str, str]]:
        """The proxy service the executor may send to a twin player (safety.guard)."""
        if not self.music_integration or not self.music_twins:
            return frozenset()
        return frozenset({(self.music_integration, caps.MUSIC_PLAY)})

    # ---- lookups ------------------------------------------------------------------------------
    def descriptor(self, key: str) -> str:
        if key in self.preset_descriptors:
            return self.preset_descriptors[key]
        if key in self.entities:
            return self.entities[key].descriptor
        if key in self.areas:
            return self.areas[key].name
        return key

    def name(self, key: str) -> str:
        if key in self.presets:
            return self.presets[key].name
        if key in self.entities:
            return self.entities[key].name
        if key in self.areas:
            return self.areas[key].name
        return self.registry_names.get(key, key)

    def registry_area(self, entity_id: str) -> str | None:
        """The area of any registry entity, in scope or not (group members can be out of scope)."""
        e = self.entities.get(entity_id)
        if e is not None:
            return e.area_id
        return self.registry_areas.get(entity_id)

    def area_of(self, key: str) -> str | None:
        if key in self.preset_areas:
            return self.preset_areas[key]
        e = self.entities.get(key)
        return e.area_id if e else None

    def lookup(self, entity_id: str) -> EntityFacts | None:
        e = self.entities.get(entity_id)
        if e is None:
            return None
        area = self.areas.get(e.area_id or "")
        return EntityFacts(name=e.name, device_class=e.device_class, area_name=area.name if area else None,
                           touches_denied=e.touches_denied)

    def area_id_for(self, name_or_id: str | None) -> str | None:
        if not name_or_id:
            return None
        if name_or_id in self.areas:
            return name_or_id
        n = normalize_name(name_or_id)
        for a in self.areas.values():
            if normalize_name(a.name) == n or n in (normalize_name(x) for x in a.aliases):
                return a.id
        return None

    def luna_summary(self) -> str:
        """One line for Luna on what this home can do, so it can point the user at a command it can't run itself.
        Built from the catalog, so a new room, device kind or mode shows up with no code change."""
        modes = {self.name(k) for k in self.scene_options}
        counts: dict[str, int] = {}
        for e in self.entities.values():
            device_type = self.caps.caps_for(e.domain).device_type  # devices, not modes, lists or sensors
            if e.controllable and not e.denied and device_type and device_type != "other":
                counts[e.kind] = counts.get(e.kind, 0) + 1
        kinds = [_plural(k) if n > 1 else f"the {k}" for k, n in sorted(counts.items())]
        can = []
        if kinds:
            rooms = sorted(a.name for a in self.areas.values())
            can.append(f"control {_join(kinds)}" + (f" in the {_join(rooms)}" if rooms else ""))
        if modes:
            can.append(f"run modes and scenes such as {_join(sorted(modes))}")
        if self.todo_lists:
            can.append("add to and read lists")
        if any(e.domain in self.caps.with_role("weather") for e in self.entities.values()):
            can.append("give the local weather")
        return f"By voice, this home can {_join(can)}." if can else ""

    # ---- build --------------------------------------------------------------------------------
    @classmethod
    def build(cls, registry: Registry, states: Mapping[str, EntityState], scope: ScopeConfig,
              safety: SafetyConfig, table: caps.CapabilityTable | None = None) -> Catalog:
        table = table or caps.DEFAULT
        def matches(eid: str, patterns: list[str]) -> bool:
            return any(fnmatch.fnmatchcase(eid, p) for p in patterns)

        # A music-integration player that shadows a speaker is never a candidate ("pause the <speaker>" stays one
        # entity); only the music executor uses it (music.md §3.5).
        twins = pair_players(registry, scope.music, table)
        in_scope: list[str] = []
        for eid, e in registry.entities.items():
            picked = (scope.include_exposed and e.exposed) or matches(eid, scope.include)
            if not picked or matches(eid, scope.exclude) or e.disabled or e.hidden or eid in twins.values():
                continue
            if table.caps_for(e.domain) is None:
                continue
            in_scope.append(eid)

        areas: dict[str, AreaInfo] = {}
        for a in registry.areas.values():
            extra = list(scope.aliases.get(f"area:{a.id}", []))
            areas[a.id] = AreaInfo(a.id, a.name.strip(), tuple(dict.fromkeys([*a.aliases, *extra, *a.floor_names])))

        registry_names = {eid: e.name for eid, e in registry.entities.items()}
        touching = _definitions_touching_denied(registry, states, areas, safety, table)

        entities: dict[str, CatalogEntity] = {}
        for eid in sorted(in_scope):
            e = registry.entities[eid]
            dcaps = table.caps_for(e.domain)
            assert dcaps is not None
            st = states.get(eid)
            attrs = dict(st.attributes) if st else {}
            capabilities = dict(e.capabilities or {})
            for k in caps.CAPABILITY_ATTRIBUTES:
                if k not in capabilities and k in attrs:
                    capabilities[k] = attrs[k]
            device_class = e.device_class or attrs.get("device_class")
            aliases = list(e.aliases)
            for pattern, names in scope.aliases.items():
                if not pattern.startswith("area:") and fnmatch.fnmatchcase(eid, pattern):
                    aliases.extend(names)
            aliases = list(dict.fromkeys(aliases))
            members = None
            if dcaps.groupable and isinstance(attrs.get("entity_id"), list):
                members = tuple(attrs["entity_id"])
            onoff = table.is_onoff_only(e.domain, capabilities)
            area_name = areas[e.area_id].name if e.area_id in areas else None
            denied = is_denied_entity(eid, safety, EntityFacts(e.name, device_class, area_name, eid in touching), table)
            controllable = dcaps.controllable and not denied
            kind = table.kind_for(e.domain, e.features, device_class)
            area_name = areas[e.area_id].name if e.area_id in areas else "no room"
            desc = f"{e.name.strip()} — {kind}, {area_name}"
            if aliases:
                desc += "; also: " + ", ".join(aliases)
            if members:
                desc += "; group of: " + ", ".join(registry_names.get(m, m) for m in members)
            if onoff:
                desc += "; on/off only"
            entities[eid] = CatalogEntity(
                entity_id=eid, domain=e.domain, name=e.name.strip(), kind=kind,
                area_id=e.area_id if e.area_id in areas else None, aliases=tuple(aliases), members=members,
                onoff_only=onoff, features=frozenset(e.features), device_class=device_class,
                capabilities=capabilities, device_id=e.device_id, controllable=controllable, denied=denied,
                descriptor=desc, touches_denied=eid in touching,
            )

        controllable = frozenset(eid for eid, ce in entities.items() if ce.controllable)

        # Scenes and scripts: dedupe by (area, name); a script beats a scene with the same name (§3.1, §3.4).
        # A script hides a scene of the same name in the same room, or anywhere when the script has no room
        # (§3.1, review D9-2: the key is per room, not per house). Equal names of equal priority dedupe by id.
        def prio(ce: CatalogEntity) -> int:
            return table.caps_for(ce.domain).name_priority

        scene_like_all = [entities[eid] for eid in sorted(entities) if table.caps_for(entities[eid].domain).scene_like]
        top = max((prio(ce) for ce in scene_like_all), default=0)
        wrappers = {(ce.area_id, normalize_name(ce.name)) for ce in scene_like_all if prio(ce) == top}
        wrappers_no_area = {n for a, n in wrappers if a is None}
        qualified = {eid: registry.entities[eid].platform for eid in entities
                     if table.caps_for(entities[eid].domain).scene_like
                     and registry.entities[eid].platform in scope.qualified_scene_platforms}
        seen: dict[tuple[str | None, str, bool], str] = {}
        scene_like: list[str] = []
        for ce in scene_like_all:
            key = (ce.area_id, normalize_name(ce.name))
            if ce.entity_id not in qualified and prio(ce) < top and (key in wrappers or key[1] in wrappers_no_area):
                continue
            namespaced_key = (*key, qualified.get(ce.entity_id))
            if prio(ce) < top and namespaced_key in seen:
                continue
            seen.setdefault(namespaced_key, ce.entity_id)
            scene_like.append(ce.entity_id)

        presets: dict[str, Preset] = {}
        preset_desc: dict[str, str] = {}
        preset_areas: dict[str, str | None] = {}
        for p in scope.presets:
            targets = [t for c in p.calls for t in c.entity_ids]
            target_areas = {entities[t].area_id for t in targets if t in entities}
            area = target_areas.pop() if len(target_areas) == 1 else None
            presets[p.key] = p
            preset_areas[p.key] = area
            d = f"{p.name} — preset, {areas[area].name if area in areas else 'no room'}"
            if p.aliases:
                d += "; also: " + ", ".join(p.aliases)
            preset_desc[p.key] = d

        target_options = [eid for eid, ce in entities.items() if table.caps_for(ce.domain).target_option]
        mention_options = [eid for eid, ce in entities.items()
                           if table.caps_for(ce.domain).mention and ce.controllable]
        sources: list[str] = []
        for ce in entities.values():
            for s in ce.capabilities.get("source_list") or []:
                if s not in sources:
                    sources.append(s)
        todo_lists = [eid for eid, ce in entities.items() if table.caps_for(ce.domain).list_like]
        music = [eid for eid, ce in entities.items()
                 if ce.controllable and (table.is_music_speaker(ce.domain, ce.device_class, ce.features)
                                         or eid in set(getattr(scope.defaults, "speakers", None) or []))]

        # Only the domains this home has: no thermostat question for a home without one, and no service for a
        # domain with nothing in scope.
        present = {ce.domain for ce in entities.values()} | {c.domain for p in presets.values() for c in p.calls}
        return cls(
            areas=areas, entities=entities, controllable=controllable, presets=presets,
            preset_descriptors=preset_desc, preset_areas=preset_areas,
            scene_options=[*scene_like, *presets.keys()], target_options=target_options,
            mention_options=mention_options, sources=sources, todo_lists=todo_lists, music_speakers=music,
            defaults=scope.defaults, registry_names=registry_names, caps=table.only(present),
            registry_areas={eid: (e.area_id if e.area_id in areas else None) for eid, e in registry.entities.items()},
            music_twins={s: t for s, t in twins.items() if s in set(music)},
            music_integration=scope.music.integration, music_limit=scope.music.search_limit,
            qualified_scenes=qualified,
        )


def _definition(entry: Any, state: EntityState | None, table: caps.CapabilityTable) -> Any:
    source = table.definition_source(entry.domain)
    if source == "config":
        return entry.config
    if source == "state_members":
        members = (state.attributes if state else {}).get("entity_id")
        return {"entity_id": list(members)} if isinstance(members, list) else None
    return None


def _definitions_touching_denied(registry: Registry, states: Mapping[str, EntityState], areas: Mapping[str, AreaInfo],
                                 safety: SafetyConfig, table: caps.CapabilityTable) -> set[str]:
    """Decision 6 in code: every registry entity (in scope or not) gets its own denial first, then any script,
    scene or automation whose definition reaches a denied one is denied too. Unknown definitions are skipped."""
    denied: set[str] = set()
    for eid, e in registry.entities.items():
        st = states.get(eid)
        dc = e.device_class or (st.attributes.get("device_class") if st else None)
        area = areas[e.area_id].name if e.area_id in areas else None
        if is_denied_entity(eid, safety, EntityFacts(e.name, dc, area), table):
            denied.add(eid)
    definitions = {eid: _definition(e, states.get(eid), table) for eid, e in registry.entities.items()}
    return definitions_touching_denied(definitions, denied, safety, table)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _plural(word: str) -> str:
    if word.endswith("s"):
        return word
    return word + ("es" if word.endswith(("ch", "sh", "x")) else "s")
