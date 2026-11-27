"""Routing evals through Core; Hermes records handoffs without booking anything."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.ha.interface import Registry
from ha_jev.hermes import HermesWithLuna
from ha_jev.models import Outcome
from tests.fakes import FakeJev, FakeLuna
from tests.unit.test_hermes_session import FakeHermes

CASES = json.loads((Path(__file__).resolve().parents[2] / "eval/reservation-routing.json").read_text())


async def evaluate(case, jev):
    ha = StubHA.default()
    hermes, luna = FakeHermes("Reservation agent reply."), FakeLuna()
    core = await Core.create(load_settings(), ha, jev, HermesWithLuna(hermes, luna), clock=lambda: ha.now)
    result = await core.handle(case["utterance"], person="riley")
    return result, ha, hermes, luna


def assert_handoff(case, result, ha, hermes, luna):
    assert result.debug.route in [case["route"], *case.get("acceptable_routes", [])], result.to_dict()
    assert result.outcome == Outcome.ANSWERED
    assert result.debug.reads == []
    assert ha.calls == []
    assert [c["utterance"] for c in hermes.calls] == [case["utterance"]]
    assert luna.calls == []


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
async def test_reservation_handoff(case):
    jev = FakeJev({"intent": case["intent"], "calendar_whose": "speaker", "calendar_when": "today",
                   "calendar_ask": "list", "is_compound": 1.0})
    assert_handoff(case, *await evaluate(case, jev))


async def evaluate_schedule(calendar_state, jev):
    ha = StubHA.default()
    if calendar_state == "missing":
        registry = await ha.get_registry()
        ha = StubHA(Registry(registry.areas, registry.devices,
                            {k: v for k, v in registry.entities.items() if v.domain != "calendar"}),
                    {k: v for k, v in (await ha.get_states()).items() if not k.startswith("calendar.")},
                    todo_items={}, forecasts={}, now=ha.now)
    elif calendar_state == "unavailable":
        ha = StubHA(await ha.get_registry(),
                    {k: replace(v, state="unavailable") if k.startswith("calendar.") else v
                     for k, v in (await ha.get_states()).items()},
                    todo_items={}, forecasts={}, now=ha.now)
    hermes, luna = FakeHermes("Schedule reply."), FakeLuna()
    core = await Core.create(load_settings(), ha, jev, HermesWithLuna(hermes, luna), clock=lambda: ha.now)
    result = await core.handle("What's on my schedule today?", person="riley")
    return result, hermes


@pytest.mark.parametrize("calendar_state", ["missing", "unavailable", "available"])
async def test_generic_schedule_uses_only_available_ha_calendars(calendar_state):
    def classify(state, questions):
        available = "calendar" in questions["intent"]["criteria"]
        return {"intent": "calendar" if available else "general", "calendar_whose": "speaker",
                "calendar_when": "today", "calendar_ask": "list"}
    jev = FakeJev(classify)
    result, hermes = await evaluate_schedule(calendar_state, jev)
    assert result.outcome == Outcome.ANSWERED
    if calendar_state != "available":
        assert result.debug.reads == []
        assert len(hermes.calls) == 1
        assert "calendar_whose" not in jev.calls[0][1]
    else:
        assert result.debug.route == "calendar/list"
        assert len(result.debug.reads) == 1
        assert hermes.calls == []


async def test_reservation_keywords_do_not_bypass_lock_refusal():
    case = {"utterance": "Check my Resy reservations and unlock the front door"}
    result, ha, hermes, luna = await evaluate(case, FakeJev({"intent": "reservations"}))
    assert result.outcome == Outcome.REFUSED
    assert ha.calls == hermes.calls == luna.calls == []
