#!/usr/bin/env python3
"""Static validator for eval/golden.jsonl and eval/overlays/*.json (SPEC §11).

It checks cases against the fixture, config/scope.toml, config/safety.toml and the
SPEC rules. It never imports ha_jev, so it can run before the implementation exists.

    python3 eval/validate_golden.py [--golden eval/golden.jsonl] [-q]

Exit code 0 = valid, 3 = invalid (matches `ha-jev eval`'s invalid-golden code).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tomllib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ha_jev.config import load_settings

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "demo" / "ha_snapshot.json"
OVERLAY_DIR = ROOT / "eval" / "overlays"

TAGS = set("""light_named light_area light_house light_brightness light_color light_color_temp light_relative onoff_only
group_vs_bulb multi_target switch automation cover media_transport media_volume media_master_volume media_source
media_default_player sonos_duplicate mode_script preset scene scene_duplicate status weather sun list_add list_read
timer general compound followup clarify safety adversarial nevermind unknown_device speaker_area stt_noise implied
unseen_entity memory memory_remember memory_forget memory_read memory_safety person_known person_anonymous
person_units music calendar person_specific""".split())
OUTCOMES = {"executed", "clarify", "refused", "answered", "unsupported", "error"}
MODES = {"chime_success", "chime_error", "speech"}
# SPEC §8: allowed response modes per outcome.
# Music replies that name a guessed match or a miss are spoken (OQ 104), so executed and unsupported allow speech.
MODE_FOR = {"executed": {"chime_success", "speech"}, "clarify": {"speech"}, "answered": {"speech", "chime_success"},
            "refused": {"speech", "chime_error"}, "unsupported": {"chime_error", "speech"}, "error": {"chime_error"}}
FIELDS_REQ = {"id", "utterance", "tags", "expected_outcome", "expected_response_mode", "expected_calls", "requires_llm"}
FIELDS_ALL = FIELDS_REQ | {"person", "memory", "speaker_area", "history", "initial_state", "overlay", "now", "config_overrides",
                           "expected_clarify_options", "reply", "acceptable", "source", "notes"}
STUB_STATE = ROOT / "demo" / "stub_state.json"
# Music Assistant plays on its own players, the twins of the speakers (music.md §3.5): targets are checked against
# the stub's (and overlays') music_assistant players instead of the voice scope.
MUSIC_DOMAIN = "music_assistant"
MUSIC_PLAY_KEYS = {"media_id", "media_type", "enqueue", "radio_mode"}
ALT_KEYS = {"expected_outcome", "expected_response_mode", "expected_calls", "expected_clarify_options", "reply"}
REPLY_KEYS = {"must_mention", "must_not_mention", "max_words", "rubric"}

# SPEC §4.2 service allowlist and the data keys each service may carry.
SERVICES = {
    "light": {"turn_on": {"brightness_pct", "brightness_step_pct", "color_temp_kelvin", "color_name", "transition"},
              "turn_off": {"transition"}, "toggle": set()},
    "switch": {"turn_on": set(), "turn_off": set()},
    "automation": {"turn_on": set(), "turn_off": set()},
    "cover": {"open_cover": set(), "close_cover": set(), "stop_cover": set(), "set_cover_position": {"position"}},
    "media_player": {"media_play": set(), "media_pause": set(), "media_next_track": set(), "media_previous_track": set(),
                     "volume_set": {"volume_level"}, "volume_mute": {"is_volume_muted"}, "select_source": {"source"},
                     "turn_on": set(), "turn_off": set()},
    "scene": {"turn_on": {"transition"}},
    "script": {"turn_on": set()},
    "todo": {"add_item": {"item"}},
}
MEDIA_FEATURE = {"media_play": "play", "media_pause": "pause", "media_next_track": "next_track",
                 "media_previous_track": "previous_track", "volume_set": "volume_set", "volume_mute": "volume_mute",
                 "select_source": "select_source", "turn_on": "turn_on", "turn_off": "turn_off"}
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink"}
# stub_state.json group members (SPEC §9.1)
STUB_GROUPS = {
    "light.common_lights": ["light.ceiling_light", "light.paper_lantern", "light.ribbon_lamp", "light.alder", "light.backlight_left",
                        "light.backlight_right", "light.pendant_upper", "light.pendant_lower"],
    "light.bedroom_group": ["light.bedside_globe", "light.bookcase_light", "light.fern_lamp"],
    "light.bathroom_group": [],
}
SPEAKER_AREAS_PROD = {"living_room", "bedroom"}


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


class World:
    """Registry + scope for one overlay combination."""

    def __init__(self, overlays: list[dict]):
        fx = json.loads(FIXTURE.read_text())
        scope_cfg = tomllib.loads((ROOT / "demo" / "config" / "scope.toml").read_text())
        # The demo's safety.toml over the built-in rules, as the router sees them.
        self.safety = asdict(load_settings(ROOT / "demo" / "config").safety)
        self.areas = {a["id"]: a for a in fx["areas"]}
        self.entities = {e["entity_id"]: dict(e) for e in fx["entities"]}
        # sun.sun is added by stub_state.json (SPEC §9.1)
        self.entities.setdefault("sun.sun", {"entity_id": "sun.sun", "domain": "sun", "friendly_name": "Sun", "area_id": None,
                                             "exposed": False, "features": [], "capabilities": {}, "hidden_by": None,
                                             "disabled_by": None, "device_class": None})
        for add in json.loads(STUB_STATE.read_text()).get("registry_additions", []):
            if add["entity_id"] not in self.entities:
                self.entities[add["entity_id"]] = {"domain": add["entity_id"].split(".")[0], "features": [],
                                                   "capabilities": {}, "hidden_by": None, "disabled_by": None,
                                                   "device_class": None, **add}
        self.groups = dict(STUB_GROUPS)
        include, exclude = list(scope_cfg.get("include", [])), list(scope_cfg.get("exclude", []))
        self.presets = {p["id"]: p for p in scope_cfg.get("presets", [])}
        for ov in overlays:
            for a in ov.get("areas", []):
                self.areas[a["id"]] = a
            for e in ov.get("entities", []):
                d = {"domain": e["entity_id"].split(".")[0], "exposed": True, "platform": "overlay", "area_id": None,
                     "device_id": None, "device_class": None, "features": [], "capabilities": {}, "aliases": [],
                     "disabled_by": None, "hidden_by": None, "state": None}
                d.update(e)
                self.entities[e["entity_id"]] = d
            for eid in ov.get("remove", []):
                self.entities.pop(eid, None)
            for eid, st in ov.get("states", {}).items():
                members = (st.get("attributes") or {}).get("entity_id")
                if isinstance(members, list):
                    self.groups[eid] = members
            include += ov.get("scope", {}).get("include", [])
            exclude += ov.get("scope", {}).get("exclude", [])
        m = lambda eid, pats: any(fnmatch.fnmatchcase(eid, p) for p in pats)
        scope = set()
        for eid, e in self.entities.items():
            if (scope_cfg.get("include_exposed", True) and e.get("exposed")) or m(eid, include):
                if not m(eid, exclude) and not e.get("disabled_by") and not e.get("hidden_by"):
                    scope.add(eid)
        # script-over-scene and scene dedupe (SPEC §3.1, §3.4)
        script_names = {norm_name(self.entities[s]["friendly_name"]) for s in scope if s.startswith("script.")}
        seen: dict[tuple, str] = {}
        self.deduped = set()
        for eid in sorted(x for x in scope if x.startswith("scene.")):
            e = self.entities[eid]
            key = (e.get("area_id"), norm_name(e["friendly_name"]))
            if key[1] in script_names or key in seen:
                self.deduped.add(eid)
            else:
                seen[key] = eid
        self.scope = scope - self.deduped
        self.controllable = {e for e in self.scope if not self.denied(e)
                             and e.split(".")[0] in SERVICES}

    def denied(self, eid: str) -> bool:
        s, e = self.safety, self.entities.get(eid, {})
        dom = eid.split(".")[0]
        if dom in s["denied_domains"] or eid in s["denied_entities"]:
            return True
        if dom == "cover" and e.get("device_class") in s["denied_cover_device_classes"]:
            return True
        if dom in ("script", "scene", "automation"):
            pat = re.compile(s["denied_name_pattern"])
            return bool(pat.search(eid) or pat.search(e.get("friendly_name") or ""))
        return False

    def onoff_only(self, eid: str) -> bool:
        return (self.entities[eid].get("capabilities") or {}).get("supported_color_modes") == ["onoff"]


def load_overlay(name: str, errs: list[str], fixture_text: str, base_areas: set[str]) -> dict | None:
    p = OVERLAY_DIR / f"{name}.json"
    if not p.exists():
        errs.append(f"overlay {name}: file missing")
        return None
    ov = json.loads(p.read_text())
    unknown = set(ov) - {"description", "areas", "devices", "entities", "states", "remove", "scope",
                         "calendar_events", "music_library"}
    if unknown:
        errs.append(f"overlay {name}: unknown keys {unknown}")
    areas = base_areas | {a["id"] for a in ov.get("areas", [])}
    low = fixture_text.lower()
    for a in ov.get("areas", []):
        if a["id"] in base_areas:
            errs.append(f"overlay {name}: area {a['id']} already in fixture")
        if f'"{a["name"].lower()}"' in low:
            errs.append(f"overlay {name}: area name {a['name']!r} appears in the fixture")
    for d in ov.get("devices", []):
        if d.get("area_id") and d["area_id"] not in areas:
            errs.append(f"overlay {name}: device {d['id']} unknown area {d['area_id']}")
    for e in ov.get("entities", []):
        if not {"entity_id", "friendly_name"} <= set(e):
            errs.append(f"overlay {name}: entity missing entity_id/friendly_name")
            continue
        if e.get("area_id") and e["area_id"] not in areas:
            errs.append(f"overlay {name}: {e['entity_id']} unknown area {e['area_id']}")
        if e["entity_id"].lower() in low:
            errs.append(f"overlay {name}: entity_id {e['entity_id']} already in the fixture")
        if e["friendly_name"].lower() in low:
            errs.append(f"overlay {name}: name {e['friendly_name']!r} appears in the fixture")
    return ov


def check_calls(where: str, calls, w: World, errs: list[str], safety_case: bool):
    if not isinstance(calls, list):
        errs.append(f"{where}: expected_calls must be a list")
        return
    for c in calls:
        if set(c) != {"domain", "service", "entity_ids", "data"}:
            errs.append(f"{where}: call keys {sorted(c)}")
            continue
        dom, svc, ids, data = c["domain"], c["service"], c["entity_ids"], c["data"]
        tag = f"{where}: {dom}.{svc} {ids}"
        if dom == MUSIC_DOMAIN:
            if svc != "play_media" or len(ids) != 1 or set(data) - MUSIC_PLAY_KEYS or "media_id" not in data:
                errs.append(f"{tag}: music calls are play_media on one player with {sorted(MUSIC_PLAY_KEYS)}")
            for eid in ids:
                if (w.entities.get(eid) or {}).get("platform") != MUSIC_DOMAIN:
                    errs.append(f"{tag}: {eid} is not a {MUSIC_DOMAIN} player")
            continue
        if dom in ("lock", "alarm_control_panel", "homeassistant") or dom not in SERVICES or svc not in SERVICES[dom]:
            errs.append(f"{tag}: service not allowed")
            continue
        if not ids or ids != sorted(ids) or len(set(ids)) != len(ids):
            errs.append(f"{tag}: entity_ids must be non-empty, sorted, unique")
        if "kelvin" in data:
            errs.append(f"{tag}: legacy 'kelvin' key (decision 8)")
        extra = set(data) - SERVICES[dom][svc]
        if extra:
            errs.append(f"{tag}: unexpected data keys {extra}")
        for eid in ids:
            if eid not in w.entities:
                errs.append(f"{tag}: {eid} not in fixture/overlay")
                continue
            if eid.split(".")[0] != dom:
                errs.append(f"{tag}: {eid} domain mismatch")
            if w.denied(eid):
                errs.append(f"{tag}: {eid} is DENIED (safety)")
            if eid in w.deduped:
                errs.append(f"{tag}: {eid} is a deduped scene (script or duplicate wins)")
            elif eid not in w.controllable:
                errs.append(f"{tag}: {eid} not in controllable scope")
            e = w.entities[eid]
            if dom == "media_player" and MEDIA_FEATURE[svc] not in e.get("features", []):
                errs.append(f"{tag}: {eid} lacks feature {MEDIA_FEATURE[svc]}")
            if dom == "light" and w.onoff_only(eid) and set(data) - {"transition"}:
                errs.append(f"{tag}: {eid} is on/off only but gets {sorted(data)}")
            if dom == "light" and "color_temp_kelvin" in data and isinstance(data["color_temp_kelvin"], int):
                caps = e.get("capabilities") or {}
                lo, hi = caps.get("min_color_temp_kelvin", 0), caps.get("max_color_temp_kelvin", 10**6)
                if not lo <= data["color_temp_kelvin"] <= hi:
                    errs.append(f"{tag}: {data['color_temp_kelvin']} K outside {eid} range {lo}-{hi}")
            if svc == "select_source" and data.get("source") not in (e.get("capabilities") or {}).get("source_list", []):
                errs.append(f"{tag}: source {data.get('source')!r} not in {eid} source_list")
        # group-vs-bulb: a call must never carry a group and one of its members (double toggle)
        for g in ids:
            for mbr in w.groups.get(g, []):
                if mbr in ids:
                    errs.append(f"{tag}: group {g} and its member {mbr} in the same call")
        for k, lo, hi in (("brightness_pct", 0, 100), ("brightness_step_pct", -100, 100), ("position", 0, 100),
                          ("volume_level", 0, 1)):
            v = data.get(k)
            if isinstance(v, (int, float)) and not lo <= v <= hi:
                errs.append(f"{tag}: {k}={v} out of range")
        if "color_name" in data and data["color_name"] not in COLORS:
            errs.append(f"{tag}: color_name {data['color_name']!r} not a §6.4 colour")
    if safety_case:
        for c in calls:
            if any(w.denied(e) for e in c.get("entity_ids", [])):
                errs.append(f"{where}: safety case expects a denied call")


def check_expectation(where: str, exp: dict, w: World, errs: list[str], safety_case: bool):
    o, m = exp.get("expected_outcome"), exp.get("expected_response_mode")
    if o not in OUTCOMES:
        errs.append(f"{where}: bad outcome {o}")
    if m not in MODES:
        errs.append(f"{where}: bad response_mode {m}")
    if o in MODE_FOR and m not in MODE_FOR[o]:
        errs.append(f"{where}: response_mode {m} invalid for outcome {o} (SPEC §8)")
    calls = exp.get("expected_calls", [])
    if calls and o not in ("executed", "error"):
        errs.append(f"{where}: calls given for outcome {o}")
    if o == "executed" and not calls:
        errs.append(f"{where}: executed with no calls")
    check_calls(where, calls, w, errs, safety_case)
    opts = exp.get("expected_clarify_options")
    if opts is not None:
        if o != "clarify":
            errs.append(f"{where}: clarify options on a non-clarify outcome")
        for eid in opts:
            if "." not in eid or "://" in eid:  # a person id or a music URI, not an entity
                continue
            if eid not in w.scope and not eid.startswith("preset."):
                errs.append(f"{where}: clarify option {eid} not in scope")
    r = exp.get("reply")
    if r is not None:
        if not isinstance(r, dict) or set(r) - REPLY_KEYS:
            errs.append(f"{where}: bad reply {r}")
        elif m != "speech":
            errs.append(f"{where}: reply constraints on a chime case (SPEC §11.3)")
    if safety_case and o == "refused" and m != "speech":
        errs.append(f"{where}: lock refusal must be speech (OQ 33)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    fixture_text = FIXTURE.read_text()
    base_areas = {a["id"] for a in json.loads(fixture_text)["areas"]}
    errs: list[str] = []
    overlays: dict[str, dict] = {}
    for p in sorted(OVERLAY_DIR.glob("*.json")):
        ov = load_overlay(p.stem, errs, fixture_text, base_areas)
        if ov is not None:
            overlays[p.stem] = ov
    worlds: dict[tuple, World] = {}
    cases, ids = [], Counter()
    for n, line in enumerate(Path(args.golden).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as ex:
            errs.append(f"line {n}: bad JSON: {ex}")
            continue
        cases.append(c)
        cid = c.get("id", f"line{n}")
        ids[cid] += 1
        missing, extra = FIELDS_REQ - set(c), set(c) - FIELDS_ALL
        if missing or extra:
            errs.append(f"{cid}: missing {missing or ''} unknown {extra or ''}")
            continue
        tags = c["tags"]
        if not tags or set(tags) - TAGS:
            errs.append(f"{cid}: bad tags {set(tags) - TAGS}")
            continue
        if not re.fullmatch(rf"{re.escape(tags[0])}-\d{{3}}", cid):
            errs.append(f"{cid}: id must be <primary tag>-<nnn>")
        if len(set(tags)) != len(tags):
            errs.append(f"{cid}: duplicate tags")
        ov = c.get("overlay")
        ov_names = [] if ov is None else ([ov] if isinstance(ov, str) else list(ov))
        for name in ov_names:
            if name not in overlays:
                errs.append(f"{cid}: unknown overlay {name}")
        if "unseen_entity" in tags and not ov_names:
            errs.append(f"{cid}: unseen_entity case without overlay")
        key = tuple(ov_names)
        if key not in worlds:
            worlds[key] = World([overlays[x] for x in ov_names if x in overlays])
        w = worlds[key]
        sa = c.get("speaker_area")
        if sa is not None and sa not in w.areas:
            errs.append(f"{cid}: unknown speaker_area {sa}")
        if not isinstance(c.get("history", []), list) or not all(isinstance(h, str) for h in c.get("history", [])):
            errs.append(f"{cid}: history must be a list of strings")
        for eid, st in (c.get("initial_state") or {}).items():
            if eid not in w.entities:
                errs.append(f"{cid}: initial_state entity {eid} unknown")
            elif not isinstance(st, dict) or not ({"state"} <= set(st) or "items" in st):
                errs.append(f"{cid}: initial_state {eid} needs state/attributes or items")
        safety_case = "safety" in tags
        check_expectation(cid, c, w, errs, safety_case)
        for i, a in enumerate(c.get("acceptable") or []):
            if set(a) - ALT_KEYS:
                errs.append(f"{cid} alt{i}: unknown keys {set(a) - ALT_KEYS}")
            merged = {k: c.get(k) for k in ALT_KEYS} | a
            check_expectation(f"{cid} alt{i}", merged, w, errs, safety_case)
            if merged == {k: c.get(k) for k in ALT_KEYS}:
                errs.append(f"{cid} alt{i}: identical to primary")
        if c["requires_llm"] and not (c["expected_outcome"] == "answered" and c["expected_response_mode"] == "speech"):
            errs.append(f"{cid}: requires_llm cases must be answered/speech")
        if not c["requires_llm"] and "general" in tags and tags[0] == "general":
            errs.append(f"{cid}: general case without requires_llm")
    for cid, k in ids.items():
        if k > 1:
            errs.append(f"{cid}: duplicate id")
    if not args.quiet:
        prim = Counter(c["tags"][0] for c in cases)
        every = Counter(t for c in cases for t in c["tags"])
        print(f"{len(cases)} cases, {sum(c['requires_llm'] for c in cases)} require the LLM, "
              f"{sum(bool(c.get('acceptable')) for c in cases)} with alternatives, {len(overlays)} overlays")
        print("tag                     primary  any")
        for t in sorted(TAGS, key=lambda t: (-every[t], t)):
            print(f"  {t:22} {prim[t]:5}  {every[t]:5}")
        print("outcomes:", dict(Counter(c["expected_outcome"] for c in cases)))
        print("speaker_area:", dict(Counter(str(c.get("speaker_area")) for c in cases)))
    for e in errs:
        print("ERROR", e, file=sys.stderr)
    print("INVALID" if errs else "OK", f"({len(errs)} errors)")
    return 3 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
