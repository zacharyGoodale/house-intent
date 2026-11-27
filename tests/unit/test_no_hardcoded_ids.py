"""Decision 9 / spec §14 review criterion 2: no entity ids, area names or friendly names from this house in src/.

Entity ids may appear only in config/*.toml, fixtures, eval and tests. The one allowed pattern in src/ is a
`<domain>.<service>` string whose service is in the capability table's service vocabulary."""

import json
import re
from pathlib import Path

from ha_jev.capabilities import SERVICE_VOCABULARY

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
ID_PATTERN = re.compile(
    r"\b(light|switch|cover|media_player|scene|script|automation|lock|sensor|binary_sensor|weather|todo|sun|person|button)"
    r"\.([a-z0-9_]+)\b")


def _sources():
    return sorted(SRC.rglob("*.py"))


def test_no_entity_id_literals_in_src():
    offenders = []
    for path in _sources():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for m in ID_PATTERN.finditer(line):
                if m.group(2) in SERVICE_VOCABULARY:
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {m.group(0)}")
    assert not offenders, "\n".join(offenders)


def test_no_area_or_friendly_names_from_the_fixture_in_src():
    fixture = json.loads((ROOT / "demo" / "ha_snapshot.json").read_text())
    names = {a["name"] for a in fixture["areas"]}
    names |= {e["friendly_name"] for e in fixture["entities"] if e.get("friendly_name")}
    names = {n.strip() for n in names if len(n.strip()) >= 3}
    offenders = []
    for path in _sources():
        text = path.read_text()
        for n in sorted(names):
            for m in re.finditer(rf"(?<![\w.]){re.escape(n)}(?![\w])", text, re.IGNORECASE):
                lineno = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {n!r}")
    assert not offenders, "\n".join(offenders)


def test_the_id_scan_itself_catches_a_planted_id(tmp_path):
    planted = "TARGET = 'light.alder'\nOK = 'light.turn_on'\n"
    hits = [m.group(0) for m in ID_PATTERN.finditer(planted) if m.group(2) not in SERVICE_VOCABULARY]
    assert hits == ["light.alder"]


# ---- broadened (standards review D9-T) ---------------------------------------------------------------
import tomllib  # noqa: E402

GENERIC_WORDS = {"room", "light", "lights", "lamp", "home", "front", "mode", "scene", "list", "shopping", "music",
                 "speaker", "alarm", "sunrise", "routine", "timer", "white", "evening", "night", "morning", "day",
                 "wake", "news", "podcast", "movie", "bed", "blinds", "shades", "coffee", "machine", "mirror", "temperature", "thermostat", "air conditioning", "room temperature",
                 "heater"}
MORE_DOMAINS = ("fan|climate|input_boolean|input_select|alarm_control_panel|remote|vacuum|valve|humidifier|"
                "device_tracker|camera|number|select|update")
ID_PATTERN_WIDE = re.compile(
    rf"\b(light|switch|cover|media_player|scene|script|automation|lock|sensor|binary_sensor|weather|todo|sun|person|"
    rf"button|{MORE_DOMAINS})\.([a-z0-9_]+)\b")
CONCAT = re.compile(rf"[\"']\b(light|switch|cover|media_player|scene|script|automation|lock|{MORE_DOMAINS})\.[\"']\s*\+"
                    rf"|f[\"'](light|switch|cover|media_player|scene|script|automation|lock|{MORE_DOMAINS})\.\{{")


def _house_vocabulary() -> set[str]:
    """Names the router must never know: config aliases (keys and values), presets, area ids and names (and their
    distinctive words), fixture friendly names and every seed overlay's names."""
    scope = tomllib.loads((ROOT / "demo" / "config" / "scope.toml").read_text())
    out: set[str] = set()
    for key, values in (scope.get("aliases") or {}).items():
        out.update(values)
        if key.startswith("area:"):
            out.add(key.split(":", 1)[1])
    for p in scope.get("presets") or []:
        out.add(p["name"])
        out.update(p.get("aliases") or [])
        out.add(p["id"])
    fixture = json.loads((ROOT / "demo" / "ha_snapshot.json").read_text())
    for a in fixture["areas"]:
        out.update({a["id"], a["name"]})
        out.update(w for w in re.findall(r"[A-Za-z]{4,}", a["name"]) if w.casefold() not in GENERIC_WORDS)
    for ov in sorted((ROOT / "eval" / "overlays").glob("*.json")):
        data = json.loads(ov.read_text())
        out.update(a["name"] for a in data.get("areas", []))
        out.update(a["id"] for a in data.get("areas", []))
        out.update(e["friendly_name"] for e in data.get("entities", []))
    return {n.strip() for n in out if len(n.strip()) >= 3 and n.strip().casefold() not in GENERIC_WORDS}


def test_no_aliases_presets_area_ids_or_overlay_names_in_src():
    names = _house_vocabulary()
    offenders = []
    for path in _sources():
        text = path.read_text()
        for n in sorted(names):
            for m in re.finditer(rf"(?<![\w.]){re.escape(n)}(?![\w])", text, re.IGNORECASE):
                lineno = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {n!r}")
    assert not offenders, "\n".join(offenders)


def test_no_entity_ids_in_more_domains_or_built_by_concatenation():
    offenders = []
    for path in _sources():
        if path.name == "capabilities.py":  # the table: domain names and service vocabulary live there
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for m in ID_PATTERN_WIDE.finditer(line):
                if m.group(2) in SERVICE_VOCABULARY or m.group(2) in {"py", "toml", "json"}:
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {m.group(0)}")
            if CONCAT.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: built id: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_the_wide_scan_catches_planted_names():
    planted = "X = 'fan.bedroom'\nY = 'light.' + name\nZ = f'script.{x}'\n"
    assert ID_PATTERN_WIDE.search(planted.splitlines()[0])
    assert CONCAT.search(planted.splitlines()[1]) and CONCAT.search(planted.splitlines()[2])
    assert "movie mode" in {n.casefold() for n in _house_vocabulary()}
    assert "living_room" in _house_vocabulary()
