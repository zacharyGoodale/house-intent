"""Target resolution helpers: areas, groups, default players, query sources, and candidate pruning (§3.4, §3.5, §6.3)."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping, Sequence

from ha_jev import capabilities as caps
from ha_jev.catalog import Catalog, CatalogEntity, normalize_name
from ha_jev.ha.interface import EntityState


def _domain_entities(cat: Catalog, domains: Iterable[str], *, controllable: bool = True) -> list[CatalogEntity]:
    ds = set(domains)
    return [e for e in cat.entities.values() if e.domain in ds and (e.controllable or not controllable)]


def apply_group_rule(cat: Catalog, entity_ids: Iterable[str], area_ids: Sequence[str] | None = None,
                     strict: bool = True) -> list[str]:
    """Remove any light that is a member of a group in the same set (§3.4).

    With `strict` (review S3) and an area scope, a group stands in for its members only if every member is in
    the requested areas or has no area; otherwise the group is dropped and its in-area members, which are in
    the set already, are addressed directly. A group whose member list is empty or unknown is kept as a light."""
    ids = set(entity_ids)
    if strict and area_ids is not None:
        wanted = set(area_ids)
        for eid in list(ids):
            e = cat.entities.get(eid)
            if e is not None and e.members and any(cat.registry_area(m) not in wanted | {None} for m in e.members):
                ids.discard(eid)
    covered: set[str] = set()
    for eid in ids:
        e = cat.entities.get(eid)
        if e is not None and e.members:
            covered.update(e.members)
    return sorted(ids - covered)


def groupable_domains(cat: Catalog) -> list[str]:
    return [d for d, c in cat.caps.rows.items() if c.groupable]


def area_light_targets(cat: Catalog, area_ids: Sequence[str], strict: bool = True) -> list[str]:
    lights = [e.entity_id for e in _domain_entities(cat, groupable_domains(cat)) if e.area_id in set(area_ids)]
    return apply_group_rule(cat, lights, area_ids, strict)


def house_light_targets(cat: Catalog) -> list[str]:
    return apply_group_rule(cat, [e.entity_id for e in _domain_entities(cat, groupable_domains(cat))])


def expand_groups(cat: Catalog, entity_ids: Iterable[str], area_ids: Sequence[str] | None = None) -> list[str]:
    """For queries: replace each group with its member lights (members may be outside scope). With areas, only
    the members in those areas are reported (review S3)."""
    out: list[str] = []
    for eid in entity_ids:
        e = cat.entities.get(eid)
        if e is not None and e.members is not None and e.members:
            for m in e.members:
                if area_ids is not None and cat.registry_area(m) not in set(area_ids):
                    continue
                if m not in out:
                    out.append(m)
        elif eid not in out:
            out.append(eid)
    return out


def area_domain_targets(cat: Catalog, domains: Sequence[str], area_ids: Sequence[str] | None) -> list[str]:
    """In-scope controllable entities of the given domains in the areas (None = whole house), without DND-style switches."""
    out = []
    for e in _domain_entities(cat, domains):
        if area_ids is not None and e.area_id not in set(area_ids):
            continue
        if not cat.caps.caps_for(e.domain).sweepable:
            continue
        if caps.AREA_SWEEP_EXCLUDE_NAME.search(e.name):
            continue
        if any(fnmatch.fnmatchcase(e.entity_id, p) for p in (getattr(cat.defaults, "sweep_exclude", None) or [])):
            continue  # listed in scope.toml defaults.sweep_exclude (D9-9)
        out.append(e.entity_id)
    return sorted(out)


def music_speakers_in(cat: Catalog, area_ids: Sequence[str] | None) -> list[str]:
    if area_ids is None:
        return list(cat.music_speakers)
    return [m for m in cat.music_speakers if cat.entities[m].area_id in set(area_ids)]


def tvs_in(cat: Catalog, area_ids: Sequence[str] | None) -> list[str]:
    return sorted(e.entity_id for e in cat.entities.values()
                  if e.controllable and cat.caps.is_tv(e.domain, e.features, e.device_class)
                  and (area_ids is None or e.area_id in set(area_ids)))


def playing_players(cat: Catalog, states: Mapping[str, EntityState]) -> list[str]:
    cands = [*cat.music_speakers, *tvs_in(cat, None)]
    return [c for c in cands if (states.get(c) and states[c].state == "playing")]


def select_source_players(cat: Catalog) -> list[str]:
    """Players that open apps or inputs: the TVs when there are any (a speaker's source list is only its
    favourites, which "open <app>" never means), else every player with select_source."""
    players = sorted(e.entity_id for e in cat.entities.values() if e.controllable and "select_source" in e.features)
    tvs = [p for p in players if p in set(tvs_in(cat, None))]
    return tvs or players


# ---- pruning (§3.5) ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")
_FILLER = {"the", "a", "an", "on", "off", "to", "in", "of", "my", "at", "and", "up", "down", "turn", "set", "please"}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.casefold()))


def word_candidates(cat: Catalog, utterance: str, domains: Iterable[str] | None, area_ids: Sequence[str] | None) -> list[str]:
    """Controllable devices whose name or alias contains a word the user said ("the lamp"), in the given areas
    when any match there, else anywhere. Room names are not device words. Groups are left out."""
    room_words = set().union(*(_tokens(a.name) | set().union(*(_tokens(x) for x in a.aliases))
                               for a in cat.areas.values())) if cat.areas else set()
    said = _tokens(utterance) - _FILLER - room_words
    if not said:
        return []
    ds = set(domains) if domains else None
    hits = [e for e in cat.entities.values() if e.controllable and not e.is_group and (ds is None or e.domain in ds)
            and any(said & (_tokens(n) - _FILLER) for n in (e.name, *e.aliases))]
    if area_ids:
        here = [e for e in hits if e.area_id in set(area_ids)]
        hits = here or hits
    return sorted(e.entity_id for e in hits)


def _room_words(cat: Catalog) -> set[str]:
    """Words that name a place: area names and aliases, plus the names of light groups (room groups)."""
    words: set[str] = set()
    for a in cat.areas.values():
        words |= _tokens(a.name)
        for x in a.aliases:
            words |= _tokens(x)
    for e in cat.entities.values():
        if e.is_group:
            words |= _tokens(e.name)
    return words


def room_words(cat: Catalog) -> set[str]:
    return _room_words(cat)


def scene_area(cat: Catalog, key: str) -> str | None:
    """A scene's room: HA's area, else the one room whose distinctive name word is in the scene's name
    ("<Room>: Wind down" scripts often have no area in HA). Names that mention several rooms get none."""
    area = cat.area_of(key)
    if area is not None:
        return area
    words = _tokens(cat.name(key))
    hits = [a.id for a in cat.areas.values()
            if words & ((_tokens(a.name) | set().union(*(_tokens(x) for x in a.aliases))) - {"room"} - _FILLER)]
    return hits[0] if len(hits) == 1 else None


def scene_distinguished(cat: Catalog, key: str, utterance: str) -> bool:
    """The utterance shares a word with the scene's names or aliases that isn't a place word or part of the name
    it shares with its siblings (e.g. an alias like "<device> evening")."""
    rooms = _room_words(cat)
    core = _tokens(cat.name(key)) - rooms - _FILLER
    e = cat.entities.get(key)
    aliases = cat.presets[key].aliases if key in cat.presets else (e.aliases if e is not None else ())
    words = set().union(*(_tokens(n) for n in (cat.name(key), *aliases)))
    return bool((words & _tokens(utterance)) - core - rooms - _FILLER - {"mode", "scene"})


def scene_siblings(cat: Catalog, chosen: str) -> list[str]:
    """Scene-like options in other rooms whose names equal the chosen one's once place words are removed ("Study
    Reading", "Hall Reading"): the same scene in another room, which the scene Choice cannot tell apart by name.
    Only scenes with a known room count."""
    rooms = _room_words(cat)

    def core(key: str) -> frozenset[str]:
        return frozenset(_tokens(cat.name(key)) - rooms - _FILLER)

    mine = core(chosen)
    home = scene_area(cat, chosen)
    if not mine or home is None:  # a scene with no room: the scene Choice (which sees the speaker's room) decides
        return []
    return sorted(o for o in cat.scene_options
                  if o != chosen and core(o) == mine and scene_area(cat, o) not in (None, home)
                  and not (o in cat.entities and cat.entities[o].denied))


def plural_kind_said(cat: Catalog, utterance: str) -> bool:
    """The utterance uses a device kind in the plural ("lights", "lamps", "speakers")."""
    kinds: set[str] = set()
    for row in cat.caps.rows.values():
        kinds |= _tokens(row.kind)
        kinds |= _tokens(str(row.device_type_description or ""))
    plurals = {k for k in kinds if k.endswith("s")} | {k + "s" for k in kinds if not k.endswith("s")}
    return bool(_tokens(utterance) & plurals)


def word_candidates_any(cat: Catalog, utterance: str) -> list[str]:
    """Any in-scope entity (read-only ones too) whose name or alias contains a non-place word the user said."""
    said = _tokens(utterance) - _FILLER - _room_words(cat)
    if not said:
        return []
    return sorted(e.entity_id for e in cat.entities.values()
                  if any(said & (_tokens(n) - _FILLER) for n in (e.name, *e.aliases)))


def alias_sharers(cat: Catalog, chosen: str, utterance: str) -> list[str]:
    """Other devices of the same domain that share an alias the user said in full: one config alias for a set."""
    e = cat.entities.get(chosen)
    if e is None:
        return []
    said = _tokens(utterance)
    spoken = [a for a in e.aliases if (t := _tokens(a) - _FILLER) and t <= said]
    if not spoken:
        return []
    keys = {normalize_name(a) for a in spoken}
    return sorted(o.entity_id for o in cat.entities.values()
                  if o.entity_id != chosen and o.domain == e.domain and o.controllable
                  and keys & {normalize_name(a) for a in o.aliases})


def generic_words(cat: Catalog) -> set[str]:
    """Words that describe a kind of device or content ("music", "speaker", "lights"), from the capability table."""
    words: set[str] = set()
    for row in cat.caps.rows.values():
        for text in (row.kind, row.device_type_description or ""):
            for t in _tokens(text):
                words |= {t, t.rstrip("s"), t + "s"}
    return words


def named_by_utterance(cat: Catalog, entity_id: str, utterance: str) -> bool | None:
    """Does the utterance name this device? True: a full name or alias is said, or a distinctive word of it is.
    False: the only shared words are generic ("pause the music" and a player called "Deck 5 (Music)").
    None: no shared words at all (a paraphrase or STT noise), so the model's judgment stands."""
    e = cat.entities.get(entity_id)
    if e is None:
        return None
    said = _tokens(utterance)
    rooms = _room_words(cat) - {t for a in e.aliases for t in _tokens(a)}  # saying the room is not naming a device
    names = [_tokens(n) - _FILLER - rooms for n in (e.name, *e.aliases)]
    if any(n and n <= said for n in names):
        return True
    shared = set().union(*names) & said if names else set()
    if not shared:
        return None
    return bool(shared - generic_words(cat))


def subsumed_by(cat: Catalog, entity_id: str, candidates: Iterable[str], utterance: str) -> str | None:
    """The candidate whose fully said name contains every word the utterance shares with this device's names:
    "<player> do not disturb" said in full subsumes "<player>". None if there is no such longer match."""
    e = cat.entities.get(entity_id)
    if e is None:
        return None
    said = _tokens(utterance)
    mine = set().union(*(_tokens(n) for n in (e.name, *e.aliases))) & said - _FILLER
    if not mine:
        return None
    for c in candidates:
        if c == entity_id or (o := cat.entities.get(c)) is None:
            continue
        for n in (o.name, *o.aliases):
            toks = _tokens(n) - _FILLER
            if toks <= said and mine < toks:
                return c
    return None


def name_siblings(cat: Catalog, chosen: str, utterance: str) -> list[str]:
    """Devices that fit the words the user said as well as the chosen one does (review: ambiguous singular).

    If the utterance used only part of the chosen device's name ("the desk speaker" for "Study Desk Speaker"),
    every other controllable device of the same domain whose name or alias also contains all of those words is a
    sibling. The full name, or a distinguishing word or number ("strip 2"), leaves no siblings."""
    e = cat.entities.get(chosen)
    if e is None:
        return []
    said = _tokens(utterance) - _FILLER
    best: set[str] = set()
    full = False
    for n in (e.name, *e.aliases):
        toks = _tokens(n) - _FILLER
        matched = toks & said
        if len(matched) > len(best):
            best = matched
        if toks and toks <= said:
            full = True
    if not best or full:
        return []
    out = []
    for other in cat.entities.values():
        if other.entity_id == chosen or other.domain != e.domain or not other.controllable or other.is_group:
            continue
        if any(best <= (_tokens(n) - _FILLER) for n in (other.name, *other.aliases)):
            out.append(other.entity_id)
    return sorted(out)


def rank_options(cat: Catalog, options: Sequence[str], utterance: str, speaker_area: str | None) -> list[str]:
    """Rank by (1) area named in the utterance, (2) area is speaker_area, (3) token overlap; ties by id."""
    text = normalize_name(utterance)
    named_areas = {a.id for a in cat.areas.values()
                   if any(re.search(rf"\b{re.escape(normalize_name(n))}\b", text) for n in (a.name, *a.aliases))}
    utoks = _tokens(utterance)

    def key(opt: str):
        area = cat.area_of(opt)
        names = [cat.name(opt)]
        e = cat.entities.get(opt)
        if e is not None:
            names += list(e.aliases)
        elif opt in cat.presets:
            names += list(cat.presets[opt].aliases)
        overlap = len(utoks & set().union(*(_tokens(n) for n in names)))
        return (0 if area in named_areas else 1, 0 if (speaker_area and area == speaker_area) else 1, -overlap, opt)

    return sorted(options, key=key)
