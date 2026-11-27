"""Real Jev routing evals; Hermes is a recorder, so bookings/jobs never execute."""

import os

import pytest

from ha_jev.config import load_settings
from ha_jev.jev import TypeSafeJevClient
from ha_jev.models import Outcome
from tests.unit.test_reservation_routing import CASES, assert_handoff, evaluate, evaluate_schedule

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
async def test_live_reservation_handoff(case):
    jev = TypeSafeJevClient(load_settings().jev)
    try:
        assert_handoff(case, *await evaluate(case, jev))
    finally:
        await jev.aclose()


@pytest.mark.parametrize("calendar_state", ["missing", "unavailable", "available"])
async def test_live_generic_schedule(calendar_state):
    jev = TypeSafeJevClient(load_settings().jev)
    try:
        result, hermes = await evaluate_schedule(calendar_state, jev)
    finally:
        await jev.aclose()
    assert result.outcome == Outcome.ANSWERED, result.to_dict()
    if calendar_state == "available":
        assert result.debug.route == "calendar/list", result.to_dict()
        assert hermes.calls == []
    else:
        assert len(hermes.calls) == 1, result.to_dict()
        assert result.debug.reads == []
