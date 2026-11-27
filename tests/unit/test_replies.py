"""Spec §8, decision 7 (standards review ST-4): the response_mode table, every cell."""

import pytest

from ha_jev.models import Outcome, ResponseMode
from ha_jev.replies import render_template, response_mode

CELLS = [
    (Outcome.EXECUTED, "R_ON", ResponseMode.CHIME_SUCCESS),
    (Outcome.EXECUTED, "R_COMPOUND", ResponseMode.CHIME_SUCCESS),
    (Outcome.ANSWERED, "Q_ON_OFF", ResponseMode.SPEECH),  # a status query
    (Outcome.ANSWERED, "W_NOW", ResponseMode.SPEECH),  # weather
    (Outcome.ANSWERED, "R_LIST_READ", ResponseMode.SPEECH),  # list read
    (Outcome.ANSWERED, "R_LUNA", ResponseMode.SPEECH),  # Luna
    (Outcome.ANSWERED, "R_OK", ResponseMode.CHIME_SUCCESS),  # nevermind, a declined confirm
    (Outcome.CLARIFY, "R_CONFIRM", ResponseMode.SPEECH),
    (Outcome.CLARIFY, "R_WHAT", ResponseMode.SPEECH),
    (Outcome.REFUSED, "R_LOCK", ResponseMode.SPEECH),  # decision 3 is spoken (OQ 33)
    (Outcome.REFUSED, "R_WHAT", ResponseMode.CHIME_ERROR),  # refused for another reason
    (Outcome.UNSUPPORTED, "R_NO_DEVICE", ResponseMode.CHIME_ERROR),
    (Outcome.UNSUPPORTED, "R_TIMERS", ResponseMode.CHIME_ERROR),
    (Outcome.ERROR, "R_JEV_DOWN", ResponseMode.CHIME_ERROR),
    (Outcome.ERROR, "R_HA_FAIL", ResponseMode.CHIME_ERROR),
]


@pytest.mark.parametrize("outcome,key,mode", CELLS, ids=[f"{o}-{k}" for o, k, _ in CELLS])
def test_response_mode_table(outcome, key, mode):
    assert response_mode(outcome, key) == mode


def test_times_are_worded_by_replies_not_policy():
    from datetime import datetime
    now = datetime.fromisoformat("2037-09-22T18:00:00-04:00")
    at = datetime.fromisoformat("2037-09-23T06:52:00-04:00")
    assert render_template("Q_SUN", {"event": "Sunrise", "at": at, "now": now}) == "Sunrise is at 6:52 AM tomorrow."
    assert render_template("Q_NO_ALARM", {"kind": "alarm"}) == "You don't have an alarm set."
    assert render_template("Q_NO_ALARM", {"kind": "timer"}) == "You don't have a timer set."
