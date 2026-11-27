"""Golden set loading and validation (spec §11.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ha_jev.models import Outcome, ResponseMode

TAGS = frozenset("""light_named light_area light_house light_brightness light_color light_color_temp light_relative
onoff_only group_vs_bulb multi_target switch automation cover media_transport media_volume media_master_volume
media_source media_default_player sonos_duplicate mode_script preset scene scene_duplicate status weather sun list_add
list_read timer general compound followup clarify safety adversarial nevermind unknown_device speaker_area stt_noise
implied unseen_entity
memory memory_remember memory_forget memory_read memory_safety person_known person_anonymous person_units
music calendar person_specific""".split())
EXPECT_KEYS = ("expected_outcome", "expected_response_mode", "expected_calls", "expected_clarify_options", "reply")
REQUIRED = ("id", "utterance", "tags", "expected_outcome", "expected_response_mode", "expected_calls", "requires_llm")


class GoldenError(Exception):
    pass


@dataclass
class GoldenCase:
    id: str
    utterance: str
    tags: list[str]
    expected: dict[str, Any]
    acceptable: list[dict[str, Any]]
    requires_llm: bool
    speaker_area: str | None = None
    person: str | None = None  # config/people.toml id of the speaker
    memory: list[dict[str, Any]] = field(default_factory=list)  # seed facts: {text, owner ("household" | person id)}
    history: list[str] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    overlay: list[str] = field(default_factory=list)
    now: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    notes: str = ""

    @property
    def primary_tag(self) -> str:
        return self.tags[0]

    def alternatives(self) -> list[dict[str, Any]]:
        """The primary expectation plus each acceptable alternative with missing keys inherited."""
        out = [dict(self.expected)]
        for alt in self.acceptable:
            merged = dict(self.expected)
            merged.update({k: v for k, v in alt.items() if k in EXPECT_KEYS})
            out.append(merged)
        return out


def _check_expectation(where: str, exp: dict[str, Any], partial: bool) -> None:
    if "expected_outcome" in exp:
        try:
            Outcome(exp["expected_outcome"])
        except ValueError as e:
            raise GoldenError(f"{where}: bad expected_outcome {exp['expected_outcome']!r}") from e
    if "expected_response_mode" in exp:
        try:
            ResponseMode(exp["expected_response_mode"])
        except ValueError as e:
            raise GoldenError(f"{where}: bad expected_response_mode {exp['expected_response_mode']!r}") from e
    calls = exp.get("expected_calls")
    if calls is not None or not partial:
        if not isinstance(calls, list):
            raise GoldenError(f"{where}: expected_calls must be a list")
        for c in calls:
            if not isinstance(c, dict) or not {"domain", "service", "entity_ids"} <= set(c):
                raise GoldenError(f"{where}: bad call {c!r}")
    opts = exp.get("expected_clarify_options")
    if opts is not None and not isinstance(opts, list):
        raise GoldenError(f"{where}: expected_clarify_options must be a list or null")
    reply = exp.get("reply")
    if reply is not None and not isinstance(reply, dict):
        raise GoldenError(f"{where}: reply must be an object or null")


def parse_case(obj: dict[str, Any], where: str) -> GoldenCase:
    for k in REQUIRED:
        if k not in obj:
            raise GoldenError(f"{where}: missing field {k!r}")
    tags = obj["tags"]
    if not isinstance(tags, list) or not tags:
        raise GoldenError(f"{where}: tags must be a non-empty list")
    unknown = [t for t in tags if t not in TAGS]
    if unknown:
        raise GoldenError(f"{where}: unknown tags {unknown}")
    expected = {k: obj.get(k) for k in EXPECT_KEYS}
    _check_expectation(where, expected, partial=False)
    acceptable = obj.get("acceptable") or []
    if not isinstance(acceptable, list):
        raise GoldenError(f"{where}: acceptable must be a list")
    for i, alt in enumerate(acceptable):
        if not isinstance(alt, dict) or set(alt) - set(EXPECT_KEYS):
            raise GoldenError(f"{where}: acceptable[{i}] has unknown keys")
        _check_expectation(f"{where} acceptable[{i}]", alt, partial=True)
    overlay = obj.get("overlay")
    overlays = [] if overlay is None else ([overlay] if isinstance(overlay, str) else list(overlay))
    history = obj.get("history") or []
    if not isinstance(history, list) or not all(isinstance(h, str) for h in history):
        raise GoldenError(f"{where}: history must be a list of strings")
    memory = obj.get("memory") or []
    if not isinstance(memory, list) or not all(isinstance(m, dict) and {"text", "owner"} <= set(m) for m in memory):
        raise GoldenError(f"{where}: memory must be a list of {{text, owner}} objects")
    if not isinstance(obj["requires_llm"], bool):
        raise GoldenError(f"{where}: requires_llm must be a boolean")
    return GoldenCase(
        id=str(obj["id"]), utterance=str(obj["utterance"]), tags=list(tags), expected=expected,
        acceptable=list(acceptable), requires_llm=obj["requires_llm"], speaker_area=obj.get("speaker_area"),
        person=obj.get("person"), memory=list(memory), history=list(history), initial_state=dict(obj.get("initial_state") or {}), overlay=overlays,
        now=obj.get("now"), config_overrides=dict(obj.get("config_overrides") or {}),
        source=obj.get("source") or "", notes=obj.get("notes") or "",
    )


def load_golden(path: str | Path) -> list[GoldenCase]:
    p = Path(path)
    if not p.is_file():
        raise GoldenError(f"golden file not found: {p}")
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise GoldenError(f"{p.name}:{lineno}: invalid JSON: {e}") from e
        case = parse_case(obj, f"{p.name}:{lineno}")
        if case.id in seen:
            raise GoldenError(f"{p.name}:{lineno}: duplicate id {case.id}")
        seen.add(case.id)
        cases.append(case)
    return cases


def select(cases: list[GoldenCase], *, tags: list[str] | None = None, ids: list[str] | None = None,
           limit: int | None = None) -> list[GoldenCase]:
    out = cases
    if tags:
        out = [c for c in out if set(tags) & set(c.tags)]
    if ids:
        out = [c for c in out if c.id in set(ids)]
    if limit is not None:
        out = out[:limit]
    return out
