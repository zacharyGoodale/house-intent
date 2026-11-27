"""Eval runner, report and CLI over a tiny fixture golden set (the real set is owned by eval/)."""

import json
from pathlib import Path

import pytest

from ha_jev import cli
from ha_jev.evaluation.golden import GoldenError, load_golden, parse_case, select
from ha_jev.evaluation.report import build_report
from ha_jev.evaluation.runner import run_eval, write_outputs
from ha_jev.config import load_settings
from ha_jev.luna import NullLuna
from ha_jev.models import ServiceCall
from tests.fakes import FakeJev, FakeLuna

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "eval"

ANSWERS = {
    "turn off the ceiling light": {"intent": "device_control", "scope": "one_named", "target": "light.ceiling_light",
                                   "light_action": "turn_off"},
    "when's sunset": {"intent": "device_query", "query_attribute": "sunset"},
    "what's the capital of france": {"intent": "general"},
    "turn on the office lights": {"intent": "device_control", "scope": "area", "device_type": "light",
                                  "area__office": 0.9, "light_action": "turn_on"},
    "turn on alder": {"intent": "device_control", "scope": "one_named", "target": "light.alder", "light_action": "turn_on"},
    "turn it up": {"intent": "device_control", "scope": "recent", "light_action": "brighten"},
    "kitchen lights off": {"intent": "device_control", "scope": "area", "device_type": "light", "area__kitchen": 0.9,
                           "light_action": "turn_off"},
}


def fake_jev():
    return FakeJev(lambda state, questions: ANSWERS.get(state["utterance"], {"intent": "nevermind"}))


def test_loader_skips_comments_and_blank_lines_and_validates():
    cases = load_golden(FIX / "golden.jsonl")
    assert [c.id for c in cases][:2] == ["light_named-001", "safety-001"]
    assert cases[4].overlay == ["office_room"]
    assert [c.id for c in select(cases, tags=["safety"])] == ["safety-001"]
    assert len(select(cases, limit=2)) == 2
    with pytest.raises(GoldenError):
        parse_case({"id": "x", "utterance": "u", "tags": ["nope"], "expected_outcome": "executed",
                    "expected_response_mode": "chime_success", "expected_calls": [], "requires_llm": False}, "t")
    with pytest.raises(GoldenError):
        parse_case({"id": "x", "utterance": "u", "tags": ["switch"], "expected_outcome": "done",
                    "expected_response_mode": "chime_success", "expected_calls": [], "requires_llm": False}, "t")


async def test_no_llm_run_scores_skips_and_reports(tmp_path):
    cases = load_golden(FIX / "golden.jsonl")
    run = await run_eval(cases, jev=fake_jev(), luna=NullLuna(), no_llm=True, overlay_dir=FIX / "overlays",
                         concurrency=2, flags={"no_llm": True})
    by_id = {r.id: r for r in run.records}
    assert by_id["general-001"].skipped
    assert by_id["light_named-001"].passed
    assert by_id["safety-001"].passed
    assert by_id["status-001"].passed, by_id["status-001"].failures
    assert by_id["unseen_entity-001"].passed, by_id["unseen_entity-001"].failures
    assert by_id["followup-001"].passed, by_id["followup-001"].failures
    assert not by_id["light_area-001"].passed
    assert run.safety_failed_ids == []
    report = build_report(run, load_settings().prices)
    assert "**SAFETY: PASS**" in report
    assert "Accuracy: **5/6" in report
    assert "Generalization (unseen_entity): 1/1" in report
    assert "light_area-001" in report.split("## Failures")[1]
    assert "jev-fake" in report
    out = write_outputs(run, tmp_path, report)
    lines = (out / "results.jsonl").read_text().splitlines()
    assert len(lines) == 7 and json.loads(lines[0])["id"] == "light_named-001"
    assert (out / "calibration.csv").read_text().startswith("id,primary_tag,route,confidence,passed")
    assert (out / "report.md").read_text() == report


async def test_llm_run_uses_luna_and_judge():
    cases = select(load_golden(FIX / "golden.jsonl"), ids=["general-001"])

    async def judge(rubric, reply):
        return ("paris" in reply.lower(), "checked", {"input_tokens": 50, "output_tokens": 10})

    run = await run_eval(cases, jev=fake_jev(), luna=FakeLuna("Paris is the capital of France."), no_llm=False,
                         overlay_dir=FIX / "overlays", judge=judge)
    assert run.records[0].passed and not run.records[0].unjudged
    report = build_report(run, load_settings().prices)
    assert "Judge (separate): 1 calls" in report


async def test_a_denied_call_in_any_case_fails_safety(monkeypatch):
    import ha_jev.executor as executor

    async def leaky_execute(calls, ha, *, catalog, safety):  # simulate a broken guard
        await ha.call_service("lock", "unlock", ["lock.entry_lock"], {})
        return executor.ExecutionReport(done=[ServiceCall("lock", "unlock", ("lock.entry_lock",), {})])

    monkeypatch.setattr("ha_jev.core.execute", leaky_execute)
    cases = select(load_golden(FIX / "golden.jsonl"), ids=["light_named-001"])
    run = await run_eval(cases, jev=fake_jev(), luna=NullLuna(), no_llm=True, overlay_dir=FIX / "overlays")
    assert run.safety_failed_ids == ["light_named-001"]
    assert "**SAFETY: FAIL**" in build_report(run, load_settings().prices)


def test_cli_eval_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("ha_jev.jev.TypeSafeJevClient", lambda settings: _ClosableFake())
    rc = cli.main(["eval", "--no-llm", "--golden", str(FIX / "golden.jsonl"), "--out", str(tmp_path), "--limit", "3"])
    assert rc == 0
    assert "SAFETY: PASS" in capsys.readouterr().out
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n')
    assert cli.main(["eval", "--no-llm", "--golden", str(bad), "--out", str(tmp_path)]) == 3


def test_cli_refuses_real_ha_without_opt_in(monkeypatch, capsys):
    monkeypatch.delenv("HA_JEV_ALLOW_REAL", raising=False)
    assert cli.main(["ask", "hi", "--ha", "real", "--no-llm"]) == 2


def test_cli_ask_prints_calls(monkeypatch, capsys):
    monkeypatch.setattr("ha_jev.jev.TypeSafeJevClient", lambda settings: _ClosableFake())
    monkeypatch.setenv("HA_JEV_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))
    monkeypatch.setattr("ha_jev.obslog.RequestLog.write", lambda self, rec: None)
    assert cli.main(["ask", "turn off the ceiling light", "--area", "living_room", "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert "outcome: executed" in out and "light.turn_off light.ceiling_light {}" in out


class _ClosableFake:
    def __init__(self):
        self._jev = fake_jev()

    async def ask(self, state, questions):
        return await self._jev.ask(state, questions)

    async def aclose(self):
        return None


async def test_a_denied_call_hha_rejects_as_unknown_still_fails_safety(monkeypatch):
    """Review S-5: a denied call StubHA rejects (unknown service or entity) must still be seen by the eval."""
    import ha_jev.executor as executor

    async def leaky_execute(calls, ha, *, catalog, safety):
        try:
            await ha.call_service("alarm_control_panel", "alarm_arm_night", ["alarm_control_panel.nope"], {})
        except Exception:  # noqa: BLE001
            pass
        return executor.ExecutionReport(done=list(calls))

    monkeypatch.setattr("ha_jev.core.execute", leaky_execute)
    cases = select(load_golden(FIX / "golden.jsonl"), ids=["light_named-001"])
    run = await run_eval(cases, jev=fake_jev(), luna=NullLuna(), no_llm=True, overlay_dir=FIX / "overlays")
    assert run.safety_failed_ids == ["light_named-001"]


async def test_a_case_cannot_weaken_its_own_safety_check(monkeypatch):
    """Review S-5: config_overrides that empty the denylist do not change what the eval counts as denied."""
    import dataclasses

    import ha_jev.executor as executor

    async def leaky_execute(calls, ha, *, catalog, safety):
        await ha.call_service("script", "turn_on", ["script.toggle_entry_lock"], {})
        return executor.ExecutionReport(done=list(calls))

    monkeypatch.setattr("ha_jev.core.execute", leaky_execute)
    case = select(load_golden(FIX / "golden.jsonl"), ids=["light_named-001"])[0]
    case = dataclasses.replace(case, config_overrides={"safety.denied_entities": [],
                                                       "safety.denied_name_pattern": "(?!x)x"})
    run = await run_eval([case], jev=fake_jev(), luna=NullLuna(), no_llm=True, overlay_dir=FIX / "overlays")
    assert run.safety_failed_ids == ["light_named-001"]
