"""Eval runner (spec §11.5): one fresh StubHA + Catalog + Core per case, bounded concurrency."""

from __future__ import annotations

import asyncio
import csv
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ha_jev import capabilities as caps
from ha_jev.config import Settings, load_settings
from ha_jev.core import Core
from ha_jev.evaluation.golden import GoldenCase
from ha_jev.evaluation.overlay import load_overlays
from ha_jev.evaluation.scoring import score_case
from ha_jev.ha.stub import StubHA
from ha_jev.jev import JevClient
from ha_jev.luna import LunaClient
from ha_jev.memory import HOUSEHOLD, Fact, MemoryStore
from ha_jev.models import ServiceCall
from ha_jev.safety import is_denied_call

AsyncJudge = Callable[[str, str], Awaitable[tuple[bool, str, dict]]]  # (rubric, reply) -> (pass, why, usage)


@dataclass
class CaseRecord:
    id: str
    tags: list[str]
    utterance: str
    passed: bool
    skipped: bool
    failures: list[str]
    expected: dict[str, Any]
    actual: dict[str, Any] | None
    unjudged: bool = False
    calls_ok: bool | None = None
    safety_hits: list[str] = field(default_factory=list)
    history_results: list[dict] = field(default_factory=list)
    judge_usage: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"id": self.id, "tags": self.tags, "passed": self.passed, "skipped": self.skipped,
                "failures": self.failures, "expected": self.expected, "actual": self.actual,
                "unjudged": self.unjudged, "safety_hits": self.safety_hits,
                "history": self.history_results}


@dataclass
class RunResult:
    records: list[CaseRecord]
    out_dir: Path | None
    flags: dict[str, Any]
    total_cases: int
    wall_s: float

    @property
    def safety_failed_ids(self) -> list[str]:
        ids = [r.id for r in self.records if r.safety_hits]
        ids += [r.id for r in self.records if not r.skipped and not r.passed and "safety" in r.tags]
        return sorted(set(ids))


def _expected_view(case: GoldenCase) -> dict[str, Any]:
    return {**case.expected, "acceptable": case.acceptable, "requires_llm": case.requires_llm}


async def run_case(case: GoldenCase, *, base_settings_loader: Callable[[dict], Settings], overlay_dir: Path,
                   jev: JevClient, luna: LunaClient, judge: AsyncJudge | None) -> CaseRecord:
    settings = base_settings_loader(case.config_overrides)
    base_safety = base_settings_loader({}).safety  # a case's overrides never weaken the eval's own check (S-5)
    overlays = load_overlays(case.overlay, overlay_dir) if case.overlay else []
    ha = StubHA.default(overlays=overlays, initial_state=case.initial_state, now=case.now)
    settings = settings.with_scope_extension(**ha.overlay_scope)
    seeds = [Fact(f"seed{i}", m["text"], m["owner"], "2026-09-01T12:00:00-04:00",
                  None if m["owner"] == HOUSEHOLD else m["owner"]) for i, m in enumerate(case.memory)]
    core = await Core.create(settings, ha, jev, luna, clock=lambda: ha.now, memory=MemoryStore(None, seeds))
    history = []
    for turn in case.history:
        r = await core.handle(turn, speaker_area=case.speaker_area, session_id=case.id, person=case.person)
        history.append(r.to_dict())
    result = await core.handle(case.utterance, speaker_area=case.speaker_area, session_id=case.id, person=case.person)
    actual = result.to_dict()

    hits = []
    for call in ha.attempted_calls:
        if call.domain in caps.NEVER_ALLOWED_DOMAINS or is_denied_call(call, base_safety, core.catalog.lookup, core.catalog.caps):
            hits.append(call.describe())

    verdicts: dict[str, tuple[bool, str]] = {}
    usage: list[dict] = []
    if judge is not None:
        for exp in case.alternatives():
            rubric = (exp.get("reply") or {}).get("rubric")
            if rubric and rubric not in verdicts:
                ok, why, u = await judge(rubric, actual["reply"])
                verdicts[rubric] = (ok, why)
                usage.append(u)
    sync_judge = (lambda rubric, reply: verdicts[rubric]) if judge is not None else None
    score = score_case(case, actual, sync_judge)
    return CaseRecord(id=case.id, tags=case.tags, utterance=case.utterance, passed=score.passed, skipped=False,
                      failures=score.failures + [f"SAFETY: {h}" for h in hits], expected=_expected_view(case),
                      actual=actual, unjudged=score.unjudged, calls_ok=score.calls_ok, safety_hits=hits,
                      history_results=history, judge_usage=usage)


async def run_eval(cases: list[GoldenCase], *, jev: JevClient, luna: LunaClient, no_llm: bool,
                   overlay_dir: Path, concurrency: int = 4, judge: AsyncJudge | None = None,
                   config_dir: Path | None = None, total_cases: int | None = None,
                   flags: dict[str, Any] | None = None) -> RunResult:
    cache: dict[str, Settings] = {}

    def loader(overrides: dict) -> Settings:
        key = json.dumps(overrides, sort_keys=True)
        if key not in cache:
            cache[key] = load_settings(config_dir=config_dir, overrides=overrides)
        return cache[key]

    sem = asyncio.Semaphore(max(1, concurrency))
    start = time.perf_counter()

    async def one(case: GoldenCase) -> CaseRecord:
        if no_llm and case.requires_llm:
            return CaseRecord(case.id, case.tags, case.utterance, False, True, [], _expected_view(case), None)
        async with sem:
            try:
                return await run_case(case, base_settings_loader=loader, overlay_dir=overlay_dir, jev=jev,
                                      luna=luna, judge=None if no_llm else judge)
            except Exception as e:  # noqa: BLE001 - one broken case must not stop the run
                return CaseRecord(case.id, case.tags, case.utterance, False, False,
                                  [f"runner error: {type(e).__name__}: {e}"], _expected_view(case), None)

    records = await asyncio.gather(*(one(c) for c in cases))
    return RunResult(list(records), None, flags or {}, total_cases if total_cases is not None else len(cases),
                     time.perf_counter() - start)


def write_outputs(run: RunResult, out_root: Path, report_text: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in run.records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
    (out / "report.md").write_text(report_text, encoding="utf-8")
    with (out / "calibration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "primary_tag", "route", "confidence", "passed"])
        for r in run.records:
            if r.skipped or r.actual is None:
                continue
            d = r.actual["debug"]
            w.writerow([r.id, r.tags[0], d["route"], "" if d["confidence"] is None else d["confidence"], r.passed])
    run.out_dir = out
    return out


def calls_of(record: CaseRecord) -> list[ServiceCall]:
    if not record.actual:
        return []
    return [ServiceCall.from_dict(c) for c in record.actual["service_calls"]]
