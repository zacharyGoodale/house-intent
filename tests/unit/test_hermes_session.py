from dataclasses import replace

from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.hermes import HermesWithLuna
from ha_jev.models import Outcome
from ha_jev.session import SessionStore
from tests.fakes import FakeJev, FakeLuna


class FakeHermes(FakeLuna):
    async def answer(self, *args, **kwargs):
        return replace(await super().answer(*args, **kwargs), model="hermes/test")


async def setup_core():
    tick = [0.0]
    sessions = SessionStore(clock=lambda: tick[0])
    ha = StubHA.default()
    jev = FakeJev({"intent": "general"})
    hermes = FakeHermes("Agent reply.")
    luna = FakeLuna()
    core = await Core.create(load_settings(), ha, jev, HermesWithLuna(hermes, luna),
                             clock=lambda: ha.now, sessions=sessions)
    return core, jev, hermes, luna, tick, ha


async def test_sticky_turns_bypass_jev_and_renew_until_exact_idle_deadline():
    core, jev, hermes, _, tick, ha = await setup_core()
    await core.handle("What's the capital of France?", session_id="s", person="riley")
    assert len(jev.calls) == 1
    tick[0] = 29
    result = await core.handle("turn off the lights", session_id="s", person="riley")
    assert result.debug.route == "hermes/conversation"
    assert result.debug.jev == []
    assert len(jev.calls) == 1
    assert hermes.calls[-1]["previous_response_id"] == "resp_1"
    assert ha.calls == []  # Hermes, rather than Core, owns the action in this route.
    tick[0] = 58
    await core.handle("and the kitchen too", session_id="s", person="riley")
    assert len(jev.calls) == 1
    tick[0] = 88  # exactly 30 seconds after the last response
    await core.handle("What's the capital of Italy?", session_id="s", person="riley")
    assert len(jev.calls) == 2
    assert hermes.calls[-1]["previous_response_id"] is None


async def test_new_session_and_different_speaker_do_not_inherit_sticky_history():
    core, jev, hermes, _, _, _ = await setup_core()
    await core.handle("What's the capital of France?", session_id="s", person="riley")
    await core.handle("What's the capital of Italy?", session_id="other", person="riley")
    assert hermes.calls[-1]["previous_response_id"] is None
    await core.handle("What's the capital of Spain?", session_id="s", person="morgan")
    assert len(jev.calls) == 3
    assert hermes.calls[-1]["previous_response_id"] is None


async def test_failover_ends_sticky_routing_and_both_down_returns_error():
    core, jev, hermes, luna, _, _ = await setup_core()
    await core.handle("What's the capital of France?", session_id="s")
    hermes.error = True
    reply = await core.handle("Why?", session_id="s")
    assert reply.debug.luna.error
    assert core.sessions.hermes_response("s", None) is None
    luna.error = True
    reply = await core.handle("What's the capital of Italy?", session_id="s")
    assert len(jev.calls) == 2
    assert reply.outcome == Outcome.ERROR


async def test_no_session_cannot_start_sticky_routing():
    core, jev, hermes, _, _, _ = await setup_core()
    await core.handle("What's the capital of France?")
    await core.handle("What's the capital of Italy?")
    assert len(jev.calls) == 2
    assert hermes.calls[-1]["previous_response_id"] is None


async def test_device_command_returns_to_jev_after_idle_timeout():
    core, jev, _, _, tick, ha = await setup_core()
    await core.handle("What's the capital of France?", session_id="s")
    tick[0] = 30
    jev._answers = {"intent": "device_control", "scope": "one_named", "target": "light.alder",
                    "light_action": "turn_off"}
    result = await core.handle("turn off Alder", session_id="s")
    assert result.outcome == Outcome.EXECUTED
    assert len(jev.calls) == 2
    assert result.debug.luna is None
    assert [call.describe() for call in ha.calls] == ["light.turn_off light.alder {}"]


async def test_idle_window_starts_after_slow_hermes_reply():
    core, jev, hermes, _, tick, _ = await setup_core()
    original = hermes.answer

    async def slow_reply(*args, **kwargs):
        tick[0] += 40
        return await original(*args, **kwargs)

    hermes.answer = slow_reply
    await core.handle("What's the capital of France?", session_id="s")
    tick[0] += 29
    reply = await core.handle("Why?", session_id="s")
    assert reply.debug.route == "hermes/conversation"
    assert len(jev.calls) == 1
