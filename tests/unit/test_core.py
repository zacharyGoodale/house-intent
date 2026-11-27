"""Core.handle end to end over StubHA with FakeJev/FakeLuna."""

import json

import pytest

from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.luna import NullLuna
from ha_jev.models import Outcome, Result, ResponseMode
from ha_jev.obslog import RequestLog
from tests.fakes import FakeJev, FakeLuna, jev_down


async def make_core(jev, luna=None, *, initial_state=None, overrides=None, overlays=(), log=None):
    ha = StubHA.default(initial_state=initial_state, overlays=overlays)
    settings = load_settings(overrides=overrides).with_scope_extension(**ha.overlay_scope)
    core = await Core.create(settings, ha, jev, luna or NullLuna(), clock=lambda: ha.now, log=log)
    return core, ha


def check_invariants(r: Result):
    if r.service_calls:
        assert r.outcome in (Outcome.EXECUTED, Outcome.ERROR)
    if r.outcome == Outcome.EXECUTED:
        assert r.service_calls and not r.debug.failed_calls
    assert r.reply
    assert Result.from_dict(json.loads(json.dumps(r.to_dict()))).to_dict() == r.to_dict()


async def test_named_light_executes_with_success_chime():
    jev = FakeJev({"intent": "device_control", "scope": "one_named", "target": "light.alder", "light_action": "turn_off"})
    core, ha = await make_core(jev)
    r = await core.handle("turn off alder", speaker_area="living_room")
    check_invariants(r)
    assert r.outcome == Outcome.EXECUTED and r.response_mode == ResponseMode.CHIME_SUCCESS
    assert r.reply == "Turning off Alder."
    assert [c.describe() for c in ha.calls] == ["light.turn_off light.alder {}"]
    assert len(jev.calls) == 1  # decision 1
    assert r.debug.jev[0].input_tokens == 5000
    assert r.debug.cost_usd["jev"] == pytest.approx(5000 * 0.042 / 1e6)


async def test_speaker_area_accepts_name_or_id():
    jev = FakeJev({"intent": "device_control", "scope": "unspecified", "device_type": "light", "light_action": "turn_on"})
    core, ha = await make_core(jev)
    r = await core.handle("lights on", speaker_area="Bedroom")
    assert [c.describe() for c in r.service_calls] == ["light.turn_on light.bedroom_group {}"]
    assert jev.calls[0][0]["speaker_area"] == "Bedroom"


async def test_jev_down_never_acts_and_never_calls_luna():
    luna = FakeLuna()
    core, ha = await make_core(jev_down(), luna)
    r = await core.handle("turn off alder")
    check_invariants(r)
    assert r.outcome == Outcome.ERROR and r.response_mode == ResponseMode.CHIME_ERROR
    assert r.reply == "Sorry, I can't reach my decision service right now."
    assert ha.calls == [] and luna.calls == []


async def test_lock_pre_gate_refuses_without_calling_jev():
    jev = FakeJev({"intent": "device_control"})
    core, ha = await make_core(jev)
    r = await core.handle("unlock the front door")
    check_invariants(r)
    assert r.outcome == Outcome.REFUSED and r.response_mode == ResponseMode.SPEECH
    assert "can't lock or unlock" in r.reply
    assert jev.calls == [] and ha.calls == []


async def test_lock_status_question_passes_the_pre_gate():
    jev = FakeJev({"intent": "device_query", "query_attribute": "locked", "scope": "one_named", "target": "lock.entry_lock"})
    core, _ = await make_core(jev)
    r = await core.handle("is the front door locked?")
    assert r.outcome == Outcome.ANSWERED and r.reply == "The Entry Door is locked."
    assert r.response_mode == ResponseMode.SPEECH


@pytest.mark.parametrize("when,expected", [
    ("now", "It's 66°F and partly cloudy, humidity 62%."),
    ("tomorrow", "Tomorrow: rain, high 63, low 52, 80% chance of rain."),
    ("today", "Today: partly cloudy, high 70, low 55, 10% chance of rain."),
    ("next_hours", "Next few hours: partly cloudy, cloudy and rain, 57–64°F, up to 60% chance of rain."),
    ("tonight", "Tonight: cloudy, around 61, up to 60% chance of rain."),
    ("this_week", "Tuesday partly cloudy 70; Wednesday rain 63; Thursday sunny 72; Friday cloudy 68; Saturday sunny 74."),
])
async def test_weather_from_ha(when, expected):
    core, ha = await make_core(FakeJev({"intent": "weather", "weather_when": when}))
    r = await core.handle("weather?")
    assert r.outcome == Outcome.ANSWERED and r.response_mode == ResponseMode.SPEECH
    assert r.reply == expected
    assert ha.calls == []


async def test_weather_read_failure_is_ha_fail():
    core, ha = await make_core(FakeJev({"intent": "weather", "weather_when": "tomorrow"}))
    ha.fail_on.add("weather.get_forecasts")
    r = await core.handle("weather tomorrow")
    assert r.outcome == Outcome.ERROR and r.reply.startswith("Sorry, Forecast Home didn't respond")


async def test_sun_and_alarm_queries():
    core, _ = await make_core(FakeJev({"intent": "device_query", "query_attribute": "sunset"}))
    assert (await core.handle("when's sunset")).reply == "Sunset is at 7:02 PM."
    core, _ = await make_core(FakeJev({"intent": "device_query", "query_attribute": "sunrise"}))
    assert (await core.handle("when's sunrise")).reply == "Sunrise is at 6:52 AM tomorrow."
    core, _ = await make_core(FakeJev({"intent": "device_query", "query_attribute": "next_alarm"}))
    assert (await core.handle("when's my next alarm")).reply == "Your next alarm is tomorrow at 7:00 AM."


async def test_list_add_then_read():
    jev = FakeJev(lambda s, q: {"intent": "list_add"} if s["utterance"].startswith("add") else {"intent": "list_read"})
    core, ha = await make_core(jev)
    r = await core.handle("add oat milk to the shopping list")
    assert r.outcome == Outcome.EXECUTED and r.reply == "Added oat milk to your shopping list."
    r = await core.handle("what's on my shopping list")
    assert r.reply == "Your shopping list has eggs, coffee beans and oat milk."


async def test_general_question_goes_to_luna_with_follow_up_id():
    luna = FakeLuna("Paris.")
    core, _ = await make_core(FakeJev({"intent": "general"}), luna)
    r = await core.handle("what's the capital of france", speaker_area="bedroom", session_id="s1")
    assert r.outcome == Outcome.ANSWERED and r.response_mode == ResponseMode.SPEECH and r.reply == "Paris."
    assert r.debug.luna and r.debug.luna.cost_usd > 0
    await core.handle("and of italy", session_id="s1")
    assert luna.calls[1]["previous_response_id"] == "resp_1"
    assert luna.calls[0]["speaker_area"] == "Bedroom"


async def test_no_llm_mode_uses_null_luna():
    core, _ = await make_core(FakeJev({"intent": "general"}))
    r = await core.handle("tell me a joke")
    assert r.reply == "[llm disabled]"


async def test_luna_down():
    core, _ = await make_core(FakeJev({"intent": "general"}), FakeLuna(error=True))
    r = await core.handle("tell me a joke")
    assert r.outcome == Outcome.ERROR and r.reply == "Sorry, I couldn't answer that right now."
    assert r.response_mode == ResponseMode.CHIME_ERROR


async def test_partial_ha_failure_reports_error_with_done_calls():
    jev = FakeJev({"intent": "device_control", "is_compound": 0.9,
                   "p0__intent": "device_control", "p0__scope": "one_named", "p0__target": "light.alder",
                   "p0__light_action": "turn_off",
                   "p1__intent": "device_control", "p1__scope": "one_named", "p1__target": "media_player.demo_bedroom_speaker",
                   "p1__media_action": "media_pause"})
    core, ha = await make_core(jev)
    ha.fail_on.add("media_player.demo_bedroom_speaker")
    r = await core.handle("turn off alder and pause the bedroom speaker")
    check_invariants(r)
    assert r.outcome == Outcome.ERROR
    assert [c.describe() for c in r.service_calls] == ["light.turn_off light.alder {}"]
    assert r.debug.failed_calls and r.reply == "Sorry, Bedroom Speaker didn't respond."


async def test_clarify_then_pick_completes_across_turns():
    def answers(state, questions):
        if "pending_clarification" in state:
            return {"intent": "clarify_reply", "clarify_pick": "scene.demo_bedroom_relax"}
        return {"intent": "activate_scene", "scene": {"scene.demo_common_relax": 0.5, "scene.demo_bedroom_relax": 0.45, "none": 0.05}}
    core, ha = await make_core(FakeJev(answers))
    r1 = await core.handle("Hue scene relax", session_id="s")
    assert r1.outcome == Outcome.CLARIFY and r1.response_mode == ResponseMode.SPEECH
    assert r1.clarify_options == ["scene.demo_common_relax", "scene.demo_bedroom_relax"]
    assert r1.reply == "Which one: Common Area Relax or Bedroom Relax?"
    r2 = await core.handle("the bedroom one", session_id="s")
    assert r2.outcome == Outcome.EXECUTED
    assert [c.describe() for c in ha.calls] == ["scene.turn_on scene.demo_bedroom_relax {}"]
    r3 = await core.handle("the bedroom one", session_id="s")  # pending was cleared: a fresh request, which asks again
    assert r3.outcome == Outcome.CLARIFY and len(ha.calls) == 1


async def test_confirm_high_stakes_then_yes():
    def answers(state, questions):
        if "pending_clarification" in state:
            return {"intent": "clarify_reply", "clarify_affirm": 0.95}
        return {"intent": "device_control", "scope": "one_named", "target": ("switch.demo_coffee_maker", 0.8),
                "switch_action": "turn_on"}
    core, ha = await make_core(FakeJev(answers))
    r1 = await core.handle("coffee machine on", session_id="k")
    assert r1.outcome == Outcome.CLARIFY and r1.reply == "Do you want me to turn on Espresso Machine?"
    assert ha.calls == []
    r2 = await core.handle("yes", session_id="k")
    assert [c.describe() for c in r2.service_calls] == ["switch.turn_on switch.demo_coffee_maker {}"]


async def test_follow_up_uses_recent_targets_from_session():
    def answers(state, questions):
        if state["utterance"] == "turn it up":
            return {"intent": "device_control", "scope": "recent", "light_action": "brighten"}
        return {"intent": "device_control", "scope": "one_named", "target": "light.alder", "light_action": "turn_on"}
    core, ha = await make_core(FakeJev(answers))
    await core.handle("turn on alder", session_id="f")
    r = await core.handle("turn it up", session_id="f")
    assert [c.describe() for c in r.service_calls] == ['light.turn_on light.alder {"brightness_step_pct": 20}']


async def test_empty_utterance():
    core, _ = await make_core(FakeJev())
    r = await core.handle("  hey jev ")
    assert r.outcome == Outcome.UNSUPPORTED and r.reply == "Sorry, I didn't catch that."


async def test_too_many_parts_asks_one_at_a_time_without_jev():
    jev = FakeJev()
    core, _ = await make_core(jev)
    r = await core.handle("turn on the lights and close the blinds and pause the music and mute the tv")
    assert r.reply.startswith("Sorry, I couldn't do all of that") and jev.calls == []


async def test_missing_answer_is_an_error():
    class Broken(FakeJev):
        async def ask(self, state, questions):
            resp = await super().ask(state, questions)
            resp.answers.pop("scope")
            return resp
    core, ha = await make_core(Broken({"intent": "device_control"}))
    r = await core.handle("turn off alder")
    assert r.outcome == Outcome.ERROR and "jev_missing_answer" in r.debug.error and ha.calls == []


async def test_request_log_is_written_and_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-value-123")
    log = RequestLog(tmp_path)
    jev = FakeJev({"intent": "device_control", "scope": "one_named", "target": "light.alder", "light_action": "turn_off"})
    core, _ = await make_core(jev, log=log)
    await core.handle("turn off alder ts-secret-value-123 Authorization: Bearer abc.def")
    files = list(tmp_path.glob("requests-*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text()
    rec = json.loads(line.splitlines()[0])
    assert rec["route"] and rec["jev"][0]["answers"]["intent"]["choice"] == "device_control"
    assert "ts-secret-value-123" not in line and "abc.def" not in line


async def test_luna_gets_a_summary_of_what_the_home_can_do():
    luna = FakeLuna()
    core, _ = await make_core(FakeJev({"intent": "general"}), luna)
    await core.handle("what's a good dinner tonight", speaker_area="bedroom")
    home = luna.calls[0]["home"]
    assert home.startswith("By voice, this home can control ")
    assert "lights" in home and "switches" in home and "Bedroom" in home
    assert "lock" not in home.lower()  # denied devices are never advertised
