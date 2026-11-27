"""Case scoring (spec §11.3, §11.4)."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ha_jev.evaluation.golden import GoldenCase

Judge = Callable[[str, str], tuple[bool, str]]
DEFAULT_MAX_WORDS = 45


@dataclass
class CaseScore:
    passed: bool
    failures: list[str] = field(default_factory=list)
    unjudged: bool = False
    calls_ok: bool | None = None  # primary/alt calls matched, for the calls-accuracy section


def _num_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-6
    return a == b


def _value_ok(exp: Any, act: Any) -> bool:
    if isinstance(exp, dict) and set(exp) == {"$range"}:
        lo, hi = exp["$range"]
        return isinstance(act, (int, float)) and not isinstance(act, bool) and lo - 1e-9 <= act <= hi + 1e-9
    if isinstance(exp, dict) and set(exp) == {"$any"}:
        return any(_value_ok(v, act) for v in exp["$any"])
    if isinstance(exp, dict) and isinstance(act, dict):
        return data_subset(exp, act)
    if isinstance(exp, list) and isinstance(act, list):
        return len(exp) == len(act) and all(_value_ok(e, a) for e, a in zip(exp, act, strict=True))
    return _num_equal(exp, act)


def data_subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(k in actual and _value_ok(v, actual[k]) for k, v in (expected or {}).items())


def _canonical(calls: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for c in calls:
        key = (c["domain"], c["service"], json.dumps(c.get("data") or {}, sort_keys=True))
        if key not in merged:
            merged[key] = {"domain": c["domain"], "service": c["service"], "entity_ids": set(), "data": c.get("data") or {}}
            order.append(key)
        merged[key]["entity_ids"].update(c.get("entity_ids") or [])
    return [merged[k] for k in order]


def calls_match(expected: list[dict], actual: list[dict]) -> tuple[bool, str]:
    exp = _canonical(expected)
    act = _canonical(actual)
    if len(exp) != len(act):
        return False, f"calls: expected {len(exp)} call(s), got {len(act)}: {_brief(act)}"
    used = [False] * len(act)

    def ok(e: dict, a: dict) -> bool:
        return (e["domain"] == a["domain"] and e["service"] == a["service"] and e["entity_ids"] == a["entity_ids"]
                and data_subset(e["data"], a["data"]))

    def backtrack(i: int) -> bool:
        if i == len(exp):
            return True
        for j, a in enumerate(act):
            if not used[j] and ok(exp[i], a):
                used[j] = True
                if backtrack(i + 1):
                    return True
                used[j] = False
        return False

    if backtrack(0):
        return True, ""
    return False, f"calls: expected {_brief(exp)}, got {_brief(act)}"


def _brief(calls: list[dict]) -> str:
    return "; ".join(f"{c['domain']}.{c['service']} {','.join(sorted(c['entity_ids']))} "
                     f"{json.dumps(c.get('data') or {}, sort_keys=True)}" for c in calls) or "none"


def normalize_reply(text: str) -> str:
    t = unicodedata.normalize("NFKD", (text or "").casefold())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def reply_ok(spec: dict[str, Any] | None, reply: str, response_mode: str,
             judge: Judge | None = None) -> tuple[bool, list[str], bool]:
    """Returns (ok, reasons, unjudged)."""
    if spec is None:
        return True, [], False
    norm = normalize_reply(reply)
    reasons: list[str] = []
    for item in spec.get("must_mention") or []:
        options = item if isinstance(item, list) else [item]
        if not any(normalize_reply(o) in norm for o in options):
            reasons.append(f"reply: must mention {options!r}")
    for bad in spec.get("must_not_mention") or []:
        if normalize_reply(bad) and normalize_reply(bad) in norm:
            reasons.append(f"reply: must not mention {bad!r}")
    max_words = spec.get("max_words")
    if max_words is None and response_mode == "speech":
        max_words = DEFAULT_MAX_WORDS
    if max_words is not None and len(norm.split()) > max_words:
        reasons.append(f"reply: {len(norm.split())} words > {max_words}")
    unjudged = False
    rubric = spec.get("rubric")
    if rubric:
        if judge is None:
            unjudged = True
        else:
            passed, why = judge(rubric, reply)
            if not passed:
                reasons.append(f"reply rubric: {why}")
    return not reasons, reasons, unjudged


def _score_one(exp: dict[str, Any], actual: dict[str, Any], judge: Judge | None) -> tuple[list[str], bool, bool]:
    failures: list[str] = []
    if actual["outcome"] != exp["expected_outcome"]:
        failures.append(f"outcome: expected {exp['expected_outcome']}, got {actual['outcome']}")
    if actual["response_mode"] != exp["expected_response_mode"]:
        failures.append(f"response_mode: expected {exp['expected_response_mode']}, got {actual['response_mode']}")
    ok_calls, why = calls_match(exp.get("expected_calls") or [], actual.get("service_calls") or [])
    if not ok_calls:
        failures.append(why)
    wanted = exp.get("expected_clarify_options")
    if wanted:
        missing = [o for o in wanted if o not in (actual.get("clarify_options") or [])]
        if missing:
            failures.append(f"clarify_options: missing {missing}, got {actual.get('clarify_options')}")
    rok, reasons, unjudged = reply_ok(exp.get("reply"), actual.get("reply") or "", actual["response_mode"], judge)
    failures += reasons
    return failures, unjudged, ok_calls


def score_case(case: GoldenCase, actual: dict[str, Any], judge: Judge | None = None) -> CaseScore:
    first: tuple[list[str], bool, bool] | None = None
    for exp in case.alternatives():
        failures, unjudged, ok_calls = _score_one(exp, actual, judge)
        if first is None:
            first = (failures, unjudged, ok_calls)
        if not failures:
            return CaseScore(True, [], unjudged, ok_calls)
    assert first is not None
    return CaseScore(False, first[0], first[1], first[2])
