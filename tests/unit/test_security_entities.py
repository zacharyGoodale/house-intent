"""Decisions 5 and 6: security-type entities are refused by default, detected generically (adversarial review S1,
S2, S8; standards review D9-3, D9-4, S-1, S-3). Every repro in the reviews has a regression test here."""

import pytest

from ha_jev.catalog import Catalog
from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.luna import NullLuna
from ha_jev.models import Outcome
from tests.fakes import FakeJev

COVER_FEATURES = ["open", "close", "stop", "set_position"]

# Adversarial S1 overlay: a gate, a roll-up door and a door strike with no device_class, all in the front area.
FRONT_OPENERS = {
    "entities": [
        {"entity_id": "cover.side_gate", "friendly_name": "Side Gate", "area_id": "front", "features": COVER_FEATURES},
        {"entity_id": "cover.roll_up_door", "friendly_name": "Roll-up door", "area_id": "front", "features": COVER_FEATURES},
        {"entity_id": "switch.door_strike", "friendly_name": "Door strike", "area_id": "front"},
    ],
    "states": {"cover.side_gate": {"state": "closed", "attributes": {"current_position": 0}},
               "cover.roll_up_door": {"state": "closed", "attributes": {"current_position": 0}},
               "switch.door_strike": {"state": "off"}},
}


async def build(overlays=(), overrides=None):
    ha = StubHA.default(overlays=overlays)
    s = load_settings(overrides=overrides).with_scope_extension(**ha.overlay_scope)
    cat = Catalog.build(await ha.get_registry(), await ha.get_states(), s.scope, s.safety)
    return cat, s, ha


async def core_for(answers, overlays=(), overrides=None):
    ha = StubHA.default(overlays=overlays)
    s = load_settings(overrides=overrides).with_scope_extension(**ha.overlay_scope)
    core = await Core.create(s, ha, FakeJev(answers), NullLuna(), clock=lambda: ha.now)
    return core, ha


# ---- S1: covers and switches that open things ------------------------------------------------
async def test_gate_roll_up_door_and_door_strike_without_device_class_are_not_controllable():
    cat, _, _ = await build([FRONT_OPENERS])
    for eid in ("cover.side_gate", "cover.roll_up_door", "switch.door_strike"):
        assert eid in cat.entities, eid  # still queryable
        assert eid not in cat.controllable, eid
        assert cat.entities[eid].denied, eid
    assert "cover.bedroom_shade" in cat.controllable  # device_class shade stays controllable


GARAGE_ROOM = {
    "areas": [{"id": "garage", "name": "Garage"}],
    "entities": [
        {"entity_id": "light.workbench", "friendly_name": "Workbench light", "area_id": "garage"},
        {"entity_id": "switch.garage_lights", "friendly_name": "Garage lights", "area_id": "garage"},
        {"entity_id": "cover.garage_roller", "friendly_name": "Roller", "area_id": "garage", "device_class": "shutter",
         "features": COVER_FEATURES},
        {"entity_id": "scene.garage_party", "friendly_name": "Garage Party", "area_id": "garage"},
    ],
}
WORDS = {"entities": [
    {"entity_id": "script.reset_clock", "friendly_name": "Reset clock", "exposed": True},
    {"entity_id": "scene.block_party", "friendly_name": "Block Party", "area_id": "living_room"},
    {"entity_id": "scene.tailgate_party", "friendly_name": "Tailgate Party", "area_id": "living_room"},
    {"entity_id": "scene.sherlock", "friendly_name": "Sherlock", "area_id": "living_room"},
    {"entity_id": "scene.outdoor_dinner", "friendly_name": "Outdoor dinner", "area_id": "living_room"},
    {"entity_id": "script.unlatch_back_door", "friendly_name": "Unlatch back door", "exposed": True},
    {"entity_id": "script.front_door_release", "friendly_name": "Front door release", "exposed": True},
    {"entity_id": "switch.gate_opener", "friendly_name": "Gate opener", "area_id": "front"},
]}


async def test_word_boundaries_keep_clock_block_tailgate_outdoor_controllable():
    cat, _, _ = await build([WORDS])
    for eid in ("script.reset_clock", "scene.block_party", "scene.tailgate_party", "scene.sherlock",
                "scene.outdoor_dinner"):
        assert eid in cat.controllable, eid


async def test_unlatch_release_and_opener_names_are_denied():
    cat, _, _ = await build([WORDS])
    for eid in ("script.unlatch_back_door", "script.front_door_release", "switch.gate_opener",
                "script.toggle_entry_lock"):
        assert eid not in cat.controllable, eid


async def test_a_garage_room_keeps_its_lights_but_not_its_cover():
    cat, _, _ = await build([GARAGE_ROOM])
    assert "light.workbench" in cat.controllable
    assert "switch.garage_lights" in cat.controllable
    assert "scene.garage_party" in cat.controllable
    assert "cover.garage_roller" not in cat.controllable  # a "shutter" in a garage-like area


async def test_allowed_entities_waive_the_heuristics_but_never_a_denied_device_class():
    garage_class = {"entities": [{"entity_id": "cover.real_garage", "friendly_name": "Door", "area_id": "front",
                                  "device_class": "garage", "features": COVER_FEATURES}]}
    cat, _, _ = await build([FRONT_OPENERS, garage_class], overrides={
        "safety.allowed_entities": ["cover.side_gate", "cover.real_garage", "script.toggle_entry_lock"]})
    assert "cover.side_gate" in cat.controllable
    assert "cover.real_garage" not in cat.controllable
    assert "script.toggle_entry_lock" not in cat.controllable  # denied_entities always wins


# ---- D9-4: the pre-gate refuses garage/gate only with an actuating verb ---------------------------
@pytest.mark.parametrize("text,refuse", [
    ("turn on the garage lights", False), ("turn off the garage", False), ("garage party", False),
    ("set the gate lights to 50%", False), ("turn on the clock", False), ("block party", False),
    ("open the garage", True), ("close the gate", True), ("raise the garage door", True), ("garage door up", True),
    ("put the gate down", True), ("open the side gate", True), ("unlock the front door", True), ("let me in", True),
])
def test_pre_gate_garage_and_gate_need_an_actuating_verb(text, refuse):
    from ha_jev.extract import extract
    from ha_jev.safety import pre_gate
    x = extract(text)
    assert pre_gate(x.text, x.question_form) is refuse


# ---- S1 repros through Core --------------------------------------------------------------------
CONTROL = {"intent": "device_control"}


async def test_s1_house_cover_sweep_only_opens_the_blinds():
    core, ha = await core_for({**CONTROL, "scope": "house", "device_type": "cover", "cover_action": "open_cover"},
                              [FRONT_OPENERS])
    r = await core.handle("raise all the shades in the house", speaker_area="living_room")
    assert [c.describe() for c in r.service_calls] == ["cover.open_cover cover.bedroom_shade {}"]


async def test_s1_room_sweep_never_actuates_a_door_strike():
    core, ha = await core_for({**CONTROL, "scope": "area", "device_type": "any", "area__front": 0.95,
                               "light_action": "turn_on", "switch_action": "turn_on"}, [FRONT_OPENERS])
    r = await core.handle("turn on everything in the front", speaker_area="living_room")
    assert all("door_strike" not in c.describe() for c in ha.attempted_calls)
    assert r.outcome != Outcome.EXECUTED


async def test_s1_area_cover_clarification_never_offers_a_gate():
    core, ha = await core_for({**CONTROL, "scope": "area", "device_type": "cover", "area__front": 0.95,
                               "cover_action": ("open_cover", 0.6)}, [FRONT_OPENERS])
    r = await core.handle("open all covers in the front", speaker_area="living_room")
    assert not any(o in r.clarify_options for o in ("cover.side_gate", "cover.roll_up_door"))
    assert ha.attempted_calls == []


async def test_s1_named_gate_is_refused_with_the_lock_explanation():
    core, ha = await core_for({**CONTROL, "scope": "one_named", "target": "cover.side_gate",
                               "cover_action": "open_cover"}, [FRONT_OPENERS])
    r = await core.handle("raise the side entrance", speaker_area="living_room")
    assert r.outcome == Outcome.REFUSED and r.reply.startswith("For safety")
    assert ha.attempted_calls == []


# ---- S2: plural sweeps never turn a high-stakes device on --------------------------------------
@pytest.mark.parametrize("text,answers", [
    ("turn on all the plugs", {"scope": "area", "device_type": "switch", "switch_action": "turn_on"}),
    ("turn on all the switches", {"scope": "house", "device_type": "switch", "switch_action": "turn_on"}),
    ("turn on all the appliances", {"scope": "area", "device_type": "switch", "switch_action": "turn_on"}),
    ("turn on the kitchen plugs", {"scope": "area", "device_type": "switch", "area__kitchen": 0.95,
                                   "switch_action": "turn_on"}),
    ("toggle the kitchen switches", {"scope": "area", "device_type": "switch", "area__kitchen": 0.95,
                                     "switch_action": "toggle"}),
    ("power up the kitchen", {"scope": "area", "device_type": "any", "area__kitchen": 0.95,
                              "switch_action": "turn_on", "light_action": "turn_on"}),
])
async def test_s2_plural_sweeps_never_turn_on_the_high_stakes_device(text, answers):
    core, ha = await core_for({**CONTROL, **answers})
    r = await core.handle(text, speaker_area="kitchen", session_id="s")
    assert all("switch.demo_coffee_maker" not in c.entity_ids for c in ha.attempted_calls), r.reply


async def test_s2_a_room_whose_only_plug_is_high_stakes_asks_first():
    core, ha = await core_for({**CONTROL, "scope": "area", "device_type": "switch", "area__kitchen": 0.95,
                               "switch_action": "turn_on"},
                              overrides={"scope.defaults.high_stakes": ["switch.demo_coffee_maker",
                                                                        "switch.amber_lamp_plug"]})
    r = await core.handle("turn on the kitchen plugs", speaker_area="kitchen", session_id="s")
    assert ha.attempted_calls == []
    assert r.outcome == Outcome.CLARIFY


async def test_s2_named_high_stakes_device_still_works_when_confident():
    core, ha = await core_for({**CONTROL, "scope": "one_named", "target": "switch.demo_coffee_maker",
                               "switch_action": "turn_on"})
    r = await core.handle("turn on the espresso machine", speaker_area="kitchen")
    assert [c.describe() for c in r.service_calls] == ["switch.turn_on switch.demo_coffee_maker {}"]


# ---- S-1: switch sweeps never touch automations ------------------------------------------------
@pytest.mark.parametrize("scope,extra", [("house", {}), ("area", {"area__bedroom": 0.95})])
async def test_switch_sweeps_leave_automations_alone(scope, extra):
    core, ha = await core_for({**CONTROL, "scope": scope, "device_type": "switch", "switch_action": "turn_off", **extra})
    await core.handle("turn off all the switches", speaker_area="bedroom")
    assert all(c.domain != "automation" for c in ha.attempted_calls)


async def test_a_named_automation_is_still_reachable():
    core, ha = await core_for({**CONTROL, "scope": "one_named", "target": "automation.demo_morning_scene",
                               "switch_action": "turn_off"})
    r = await core.handle("disable the sunrise routine", speaker_area="bedroom")
    assert [c.describe() for c in r.service_calls] == ["automation.turn_off automation.demo_morning_scene {}"]


# ---- S8: scripts, scenes and automations whose definition touches a lock (decision 6) ----------
def _script(eid, name, sequence, **kw):
    return {"entity_id": eid, "friendly_name": name, "exposed": True, "config": {"alias": name, "sequence": sequence}, **kw}


DEFINITIONS = {
    "entities": [
        _script("script.welcome_home", "Welcome home",
                [{"action": "light.turn_on", "target": {"entity_id": "light.alder"}},
                 {"action": "lock.unlock", "target": {"entity_id": "lock.entry_lock"}}]),
        _script("script.arrive", "Arrive", [{"action": "script.turn_on", "target": {"entity_id": "script.welcome_home"}}]),
        _script("script.chain_direct", "Chain", [{"action": "script.toggle_entry_lock"}]),
        _script("script.templated", "Templated", [{"service": "{{ 'lock.' ~ 'unlock' }}", "entity_id": "lock.entry_lock"}]),
        _script("script.device_action", "Device step", [{"device_id": "abc", "domain": "lock", "type": "unlock"}]),
        _script("script.open_side", "Morning air", [{"action": "cover.open_cover", "target": {"entity_id": "cover.side_gate"}}]),
        _script("script.reading", "Reading",
                [{"if": [{"condition": "state", "entity_id": "lock.entry_lock", "state": "locked"}],
                  "then": [{"action": "light.turn_on", "target": {"entity_id": "light.alder"}}]}]),
        {"entity_id": "automation.arrival", "friendly_name": "Arrival", "exposed": True,
         "config": {"triggers": [{"trigger": "zone"}], "actions": [{"action": "lock.unlock", "entity_id": "lock.entry_lock"}]}},
        {"entity_id": "scene.goodnight", "friendly_name": "Goodnight", "area_id": "bedroom"},
        {"entity_id": "scene.cozy", "friendly_name": "Cozy", "area_id": "bedroom"},
    ],
    "states": {"scene.goodnight": {"state": "unknown", "attributes": {"entity_id": ["light.bedroom_group", "lock.entry_lock"]}},
               "scene.cozy": {"state": "unknown", "attributes": {"entity_id": ["light.bedroom_group"]}}},
}


async def test_s8_definitions_that_reach_a_lock_are_denied_at_catalog_build():
    cat, _, _ = await build([FRONT_OPENERS, DEFINITIONS])
    for eid in ("script.welcome_home", "script.arrive", "script.chain_direct", "script.templated",
                "script.device_action", "script.open_side", "automation.arrival", "scene.goodnight"):
        assert eid in cat.entities, eid
        assert eid not in cat.controllable, eid
    assert "scene.cozy" in cat.controllable
    assert "script.reading" in cat.controllable  # reading a lock's state in a condition is not actuating it


async def test_s8_run_welcome_home_is_refused_and_nothing_runs():
    core, ha = await core_for({"intent": "activate_scene", "scene": "script.welcome_home"}, [DEFINITIONS])
    r = await core.handle("run welcome home", speaker_area="living_room")
    assert r.outcome == Outcome.REFUSED and ha.attempted_calls == []


async def test_s8_the_guard_rejects_a_definition_denied_script_as_a_denylist_hit():
    from ha_jev.models import ServiceCall
    from ha_jev.safety import SafetyViolation, guard
    cat, s, _ = await build([DEFINITIONS])
    with pytest.raises(SafetyViolation) as e:
        guard([ServiceCall("script", "turn_on", ("script.welcome_home",), {})], cfg=s.safety,
              controllable=cat.controllable | {"script.welcome_home"}, lookup=cat.lookup)
    assert e.value.denylist


async def test_real_probability_on_a_lock_target_refuses_even_when_none_is_on_top():
    """safety-009 shape: "buzz my friend in", security noul low, target none 0.66 but the lock at 0.25."""
    core, ha = await core_for({"intent": ("device_control", 0.69), "security": 0.37, "device_type": "other",
                               "scope": {"one_named": 0.48, "unspecified": 0.47},
                               "target": {"none": 0.66, "lock.entry_lock": 0.25, "binary_sensor.entry_contact": 0.05}})
    r = await core.handle("buzz my friend in", speaker_area="living_room")
    assert r.outcome == Outcome.REFUSED and r.reply.startswith("For safety") and ha.attempted_calls == []


async def test_real_probability_on_an_unlock_script_refuses_a_scene_request():
    core, ha = await core_for({"intent": "activate_scene",
                               "scene": {"scene.demo_common_relax": 0.6, "script.unlock_back_door": 0.3, "none": 0.1}},
                              [{"entities": [{"entity_id": "script.unlock_back_door", "friendly_name": "Back door",
                                              "exposed": True}]}])
    r = await core.handle("the back door one", speaker_area="living_room")
    assert r.outcome == Outcome.REFUSED and ha.attempted_calls == []


async def test_s11_a_yes_that_adds_a_lock_request_refuses_everything():
    import time as _t

    from ha_jev.models import PlanItem, ServiceCall
    from ha_jev.session import Pending, SessionSnapshot
    from tests.unit.helpers import decide_for
    item = PlanItem((ServiceCall("light", "turn_on", ("light.alder",), {}),), "R_ON", {"names": "Alder"}, ("light.alder",))
    s = SessionSnapshot(pending=Pending("confirm", "Do you want me to turn on Alder?", {"light.alder": "Alder"},
                                        {"light.alder": item}, _t.time() + 60))
    d, _ = await decide_for("yes and also the front door", {"intent": "clarify_reply", "clarify_affirm": 0.95,
                                                             "security": 0.8}, session=s)
    assert d.outcome == Outcome.REFUSED and d.reply_key == "R_LOCK" and d.calls == ()


async def test_s11_a_missing_jev_answer_reports_the_decision_service_not_a_question():
    class Partial:
        async def ask(self, state, questions):
            from ha_jev.jev import JevResponse
            full = await FakeJev({"intent": "device_control"}).ask(state, questions)
            answers = {k: v for k, v in full.answers.items() if k != "security"}
            return JevResponse(answers=answers, model="x", request_id=None, input_tokens=1, latency_ms=1, attempts=1)

    ha = StubHA.default()
    core = await Core.create(load_settings(), ha, Partial(), NullLuna(), clock=lambda: ha.now)
    r = await core.handle("turn on alder", speaker_area="living_room")
    assert r.outcome == Outcome.ERROR and r.reply == "Sorry, I can't reach my decision service right now."
