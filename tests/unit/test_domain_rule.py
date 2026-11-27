"""Spec §4.1 rule 3 / §14 criterion 2 (standards review ST-1): the capability table is the only module that
branches on a domain name. This scan fails on a comparison against a domain-name literal anywhere else."""

import re
from pathlib import Path

from ha_jev.capabilities import FEATURE_BITS, TABLE

SRC = Path(__file__).resolve().parents[2] / "src" / "ha_jev"
# capabilities.py is the table; ha/stub.py simulates Home Assistant itself (open question 55).
EXEMPT = {"capabilities.py", "stub.py"}
DOMAINS = sorted(set(TABLE) | set(FEATURE_BITS) | {"fan", "climate", "vacuum", "button", "input_boolean", "remote",
                                                    "valve", "humidifier", "water_heater", "homeassistant"})
_LIT = "|".join(DOMAINS)
COMPARISON = re.compile(
    rf"(==|!=|\bin\b|\bnot in\b)\s*[\(\[\{{]?\s*(?:[\"'](?:{_LIT})[\"']\s*,?\s*)+|[\"'](?:{_LIT})[\"']\s*(==|!=)")


def test_no_domain_name_comparisons_outside_the_capability_table():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name in EXEMPT:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if COMPARISON.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_the_scan_catches_a_planted_comparison():
    assert COMPARISON.search('if e.domain == "light":')
    assert COMPARISON.search('if d in ("cover", "switch"):')
    assert not COMPARISON.search('if row.groupable:')
