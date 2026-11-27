"""Error paths from the standards review: ST-2 (Jev error bodies are logged), ST-3 (render bugs are not reported as
HA outages), S-4 (one Luna deadline including retries), ST-6 / core (a failed pruning retry is recorded)."""

import asyncio
import time

import pytest

from ha_jev.config import LunaSettings, load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.jev import JevError
from ha_jev.luna import LunaError, NullLuna, OpenAILuna
from ha_jev.models import Outcome
from tests.fakes import FakeJev


async def _core(jev):
    ha = StubHA.default()
    return await Core.create(load_settings(), ha, jev, NullLuna(), clock=lambda: ha.now), ha


async def test_st2_a_jev_422_body_reaches_the_trace():
    core, _ = await _core(FakeJev(error=JevError("jev http 422", status=422, body={"detail": "criteria too long"})))
    r = await core.handle("turn on alder", speaker_area="living_room")
    assert r.outcome == Outcome.ERROR and r.reply.startswith("Sorry, I can't reach")
    assert "criteria too long" in r.debug.jev[0].error


async def test_st3_a_render_bug_is_r_what_with_a_traceback_not_an_ha_failure(monkeypatch):
    core, _ = await _core(FakeJev({"intent": "weather", "weather_when": "now"}))
    monkeypatch.setattr("ha_jev.core.render_weather", lambda *a, **k: {}["boom"])
    r = await core.handle("what's the weather", speaker_area="living_room")
    assert r.outcome == Outcome.ERROR
    assert r.reply == "Sorry, what would you like me to do?"
    assert "KeyError" in (r.debug.error or "")


async def test_st3_a_bad_forecast_payload_is_still_an_ha_failure():
    core, ha = await _core(FakeJev({"intent": "weather", "weather_when": "tomorrow"}))
    ha._forecasts = {}  # HA answered, but with no forecast rows
    r = await core.handle("weather tomorrow", speaker_area="living_room")
    assert r.outcome == Outcome.ERROR and r.reply.startswith("Sorry,") and "didn't respond" in r.reply


async def test_s4_a_hung_luna_call_ends_at_the_deadline_including_retries():
    class Hung:
        class responses:  # noqa: N801 - mimics the SDK shape
            @staticmethod
            async def create(**kw):
                await asyncio.sleep(10)

    luna = OpenAILuna(LunaSettings(timeout_s=0.2, max_retries=1), client=Hung())
    start = time.perf_counter()
    from datetime import datetime
    with pytest.raises(LunaError):
        await luna.answer("hi", now=datetime.now(), speaker_area=None, previous_response_id=None)
    assert time.perf_counter() - start < 0.6


async def test_a_failed_pruning_retry_is_recorded():
    """§3.5 pruning: when the retry request fails, core keeps the first answers and records the error."""
    many = {"entities": [{"entity_id": f"light.extra_{i}", "friendly_name": f"Extra {i}", "area_id": "kitchen"}
                         for i in range(70)]}
    ha = StubHA.default(overlays=[many])
    settings = load_settings()
    calls = {"n": 0}

    class FirstOkThenDown:
        async def ask(self, state, questions):
            calls["n"] += 1
            if calls["n"] == 1:
                return await FakeJev({"intent": "device_control", "scope": "one_named", "target": "none"}).ask(
                    state, questions)
            raise JevError("jev unavailable: timeout")

    core = await Core.create(settings, ha, FirstOkThenDown(), NullLuna(), clock=lambda: ha.now)
    r = await core.handle("turn on extra 65", speaker_area="kitchen")
    assert calls["n"] == 2  # decision 1: exactly one extra request, the documented pruning retry
    assert r.debug.error and "pruning retry" in r.debug.error
