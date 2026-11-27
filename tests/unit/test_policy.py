"""Table-driven policy tests over spec §6.2–6.9, through decide() with FakeJev answers on the fixture catalog."""

import time

import pytest

from ha_jev.models import ClarifySpec, Outcome, PlanItem, ServiceCall
from ha_jev.policy import Pending, SessionSnapshot
from tests.unit.helpers import calls_of, decide_for

# Review S3: the Common Area group also drives the kitchen Pendant, so the living room addresses its own lights.
LIVING_SET = [
    "light.alder", "light.backlight_left", "light.backlight_right", "light.ceiling_light",
    "light.corner_lamp", "light.paper_lantern", "light.reading_lamp", "light.ribbon_lamp",
    "light.ring_lamp",
]
LIVING_DIMMABLE = [
    "light.alder", "light.backlight_left", "light.backlight_right", "light.ceiling_light",
    "light.paper_lantern", "light.ribbon_lamp",
]
SPEAKERS = [
    "media_player.demo_bedroom_speaker", "media_player.demo_soundbar", "media_player.demo_stereo",
]

CONTROL = {"intent": "device_control"}


async def test_named_light_off():
    d, _ = await decide_for("turn off the ceiling light", {**CONTROL, "scope": "one_named",
                            "target": "light.ceiling_light", "light_action": "turn_off"})
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("light", "turn_off", ["light.ceiling_light"], {})]
    assert d.reply_key == "R_OFF"
    assert d.confidence == 1.0


async def test_area_lights_use_group_rule():
    d, _ = await decide_for("living room lights off", {**CONTROL, "scope": "area", "device_type": "light",
                            "area__living_room": 0.95, "light_action": "turn_off"}, speaker_area="bedroom")
    assert calls_of(d) == [("light", "turn_off", LIVING_SET, {})]
    assert d.reply_params["names"] == "the Living Room lights"
    assert d.confidence == 0.95


async def test_area_brightness_absolute_with_plain_turn_on_for_plug_lamps():
    d, _ = await decide_for("dim the living room to 30%", {**CONTROL, "scope": "area", "device_type": "light",
                            "area__living_room": 0.99, "light_action": "set_brightness", "number_is_setting__n0": 0.98})
    assert calls_of(d) == [
        ("light", "turn_on", LIVING_DIMMABLE, {"brightness_pct": 30}),
        ("light", "turn_on", ["light.corner_lamp", "light.reading_lamp", "light.ring_lamp"], {}),
    ]


async def test_brighter_in_the_room_i_am_in():
    d, _ = await decide_for("brighter", {**CONTROL, "scope": "unspecified", "device_type": "light",
                            "light_action": "brighten"}, speaker_area="bedroom")
    assert calls_of(d) == [("light", "turn_on", ["light.bedroom_group"], {"brightness_step_pct": 20})]


async def test_unspecified_light_without_speaker_area_asks_which_room():
    d, _ = await decide_for("lights on", {**CONTROL, "scope": "unspecified", "device_type": "light",
                            "light_action": "turn_on"})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_WHICH_ROOM"
    assert d.clarify and set(d.clarify.options) <= {"living_room", "kitchen", "bedroom", "bathroom", "front"}


async def test_kelvin_on_dimmable_members_only():
    d, _ = await decide_for("set the kitchen lights to 4000 kelvin", {**CONTROL, "scope": "area",
                            "device_type": "light", "area__kitchen": 0.99, "light_action": "set_color_temp",
                            "number_is_setting__n0": 0.99})
    assert calls_of(d) == [("light", "turn_on", ["light.pendant_lower", "light.pendant_upper"], {"color_temp_kelvin": 4000})]


async def test_warmer_steps_from_current_kelvin():
    d, _ = await decide_for("make the pendant warmer", {**CONTROL, "scope": "several_named",
                            "mentioned__light.pendant_upper": 0.9, "mentioned__light.pendant_lower": 0.9,
                            "light_action": "warmer"},
                            initial_state={"light.pendant_upper": {"state": "on", "attributes": {"color_temp_kelvin": 3000}}})
    assert calls_of(d) == [("light", "turn_on", ["light.pendant_lower"], {"color_temp_kelvin": 3500}),
                           ("light", "turn_on", ["light.pendant_upper"], {"color_temp_kelvin": 2500})]


async def test_brightness_on_onoff_only_named_lamp_is_unsupported():
    d, _ = await decide_for("dim the ring lamp to 30", {**CONTROL, "scope": "one_named",
                            "target": "light.ring_lamp", "light_action": "set_brightness",
                            "number_is_setting__n0": 0.9})
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_CANT" and d.calls == ()


async def test_brightness_out_of_range():
    d, _ = await decide_for("set alder to 150%", {**CONTROL, "scope": "one_named", "target": "light.alder",
                            "light_action": "set_brightness", "number_is_setting__n0": 0.9})
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_RANGE"


async def test_brightness_zero_turns_off():
    d, _ = await decide_for("set alder to 0%", {**CONTROL, "scope": "one_named", "target": "light.alder",
                            "light_action": "set_brightness", "number_is_setting__n0": 0.9})
    assert calls_of(d) == [("light", "turn_off", ["light.alder"], {})]


async def test_brightness_without_value_asks_level():
    d, _ = await decide_for("set alder brightness", {**CONTROL, "scope": "one_named", "target": "light.alder",
                            "light_action": "set_brightness"})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_WHAT_LEVEL"


async def test_several_named_lights():
    d, _ = await decide_for("turn on alder and the paper lantern", {**CONTROL, "scope": "several_named",
                            "mentioned__light.alder": 0.95, "mentioned__light.paper_lantern": 0.9, "light_action": "turn_on"})
    assert calls_of(d) == [("light", "turn_on", ["light.alder", "light.paper_lantern"], {})]
    assert d.confidence == 0.9


async def test_named_group_with_its_area_mentioned_becomes_area_scope():
    d, _ = await decide_for("turn on the living room lights", {**CONTROL, "scope": "one_named",
                            "target": "light.common_lights", "area__living_room": 0.9, "light_action": "turn_on"})
    assert calls_of(d) == [("light", "turn_on", LIVING_SET, {})]


async def test_toggle_set_turns_off_when_any_member_is_on():
    d, _ = await decide_for("toggle the living room lights", {**CONTROL, "scope": "area", "device_type": "light",
                            "area__living_room": 0.9, "light_action": "toggle"},
                            initial_state={"light.corner_lamp": {"state": "on"}})
    assert calls_of(d) == [("light", "turn_off", LIVING_SET, {})]


async def test_master_volume_up_is_proportional_volume_set():
    d, _ = await decide_for("louder", {**CONTROL, "scope": "unspecified", "device_type": "media_player",
                            "media_action": "volume_up"}, speaker_area="living_room")
    assert calls_of(d) == [("media_player", "volume_set", SPEAKERS, {"volume_level": 0.4})]


async def test_proportional_volume_set():
    d, _ = await decide_for("set the volume to 30 everywhere", {**CONTROL, "scope": "house", "device_type": "media_player",
                            "media_action": "volume_set", "number_is_setting__n0": 0.95},
                            initial_state={"media_player.demo_soundbar": {"attributes": {"volume_level": 0.5}},
                                           "media_player.demo_stereo": {"attributes": {"volume_level": 0.25}},
                                           "media_player.demo_bedroom_speaker": {"attributes": {"volume_level": 0.0}}})
    assert sorted(calls_of(d)) == sorted([
        ("media_player", "volume_set", ["media_player.demo_soundbar"], {"volume_level": 0.3}),
        ("media_player", "volume_set", ["media_player.demo_stereo"], {"volume_level": 0.15}),
        ("media_player", "volume_set", ["media_player.demo_bedroom_speaker"], {"volume_level": 0.0}),
    ])


async def test_named_speaker_volume():
    d, _ = await decide_for("set the soundbar volume to 25", {**CONTROL, "scope": "one_named", "target": "media_player.demo_soundbar",
                            "media_action": "volume_set", "number_is_setting__n0": 0.95})
    assert calls_of(d) == [("media_player", "volume_set", ["media_player.demo_soundbar"], {"volume_level": 0.25})]


async def test_pause_with_nothing_playing_uses_room_default_player():
    d, _ = await decide_for("pause", {**CONTROL, "scope": "unspecified", "device_type": "media_player",
                            "media_action": "media_pause"}, speaker_area="bedroom")
    assert calls_of(d) == [("media_player", "media_pause", ["media_player.demo_bedroom_speaker"], {})]


async def test_pause_targets_the_single_playing_player():
    d, _ = await decide_for("pause", {**CONTROL, "scope": "unspecified", "device_type": "media_player",
                            "media_action": "media_pause"}, speaker_area="bedroom",
                            initial_state={"media_player.demo_soundbar": {"state": "playing"}})
    assert calls_of(d) == [("media_player", "media_pause", ["media_player.demo_soundbar"], {})]


async def test_transport_on_echo_is_unsupported():
    d, _ = await decide_for("pause the bedroom radio", {**CONTROL, "scope": "one_named",
                            "target": "media_player.demo_bedroom_radio", "media_action": "media_pause"})
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_CANT"


async def test_select_source_verbatim():
    d, _ = await decide_for("open netflix on the apple tv", {**CONTROL, "scope": "one_named",
                            "target": "media_player.demo_tv", "media_action": "select_source",
                            "source": "Netflix"})
    assert calls_of(d) == [("media_player", "select_source", ["media_player.demo_tv"], {"source": "Netflix"})]


async def test_select_source_defaults_to_the_only_tv():
    d, _ = await decide_for("put on youtube", {**CONTROL, "scope": "unspecified", "device_type": "media_player",
                            "media_action": "select_source", "source": "YouTube"})
    assert calls_of(d) == [("media_player", "select_source", ["media_player.demo_tv"], {"source": "YouTube"})]


async def test_play_content_runs_a_matching_script():
    d, _ = await decide_for("play morning brief", {**CONTROL, "scope": "unspecified", "device_type": "media_player",
                            "media_action": "play_content", "scene": "script.play_morning_brief"})
    assert calls_of(d) == [("script", "turn_on", ["script.play_morning_brief"], {})]


async def test_play_arbitrary_music_is_unsupported():
    d, _ = await decide_for("play some jazz", {**CONTROL, "scope": "unspecified", "device_type": "media_player",
                            "media_action": "play_content"})
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_PLAY"


async def test_blinds_halfway():
    d, _ = await decide_for("close the blinds halfway", {**CONTROL, "scope": "unspecified", "device_type": "cover",
                            "cover_action": "set_cover_position", "level_word": "half"})
    assert calls_of(d) == [("cover", "set_cover_position", ["cover.bedroom_shade"], {"position": 50})]


async def test_automation_toggle_resolves_from_state():
    d, _ = await decide_for("toggle the sunrise alarm", {**CONTROL, "scope": "one_named",
                            "target": "automation.demo_morning_scene", "switch_action": "toggle"})
    assert calls_of(d) == [("automation", "turn_off", ["automation.demo_morning_scene"], {})]


async def test_everything_on_never_includes_high_stakes():
    d, _ = await decide_for("turn everything on", {**CONTROL, "scope": "house", "device_type": "any",
                            "light_action": "turn_on", "switch_action": "turn_on"})
    ids = {e for c in d.calls for e in c.entity_ids}
    assert "switch.demo_coffee_maker" not in ids
    assert "light.common_lights" in ids
    assert not any("do_not_disturb" in e for e in ids)


async def test_high_stakes_turn_on_below_threshold_confirms():
    d, _ = await decide_for("turn on the espresso machine", {**CONTROL, "scope": "one_named",
                            "target": ("switch.demo_coffee_maker", 0.8), "switch_action": "turn_on"})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_CONFIRM"
    assert d.clarify.kind == "confirm"
    assert list(d.clarify.options) == ["switch.demo_coffee_maker"]
    assert d.calls == ()


async def test_mid_confidence_target_confirms_and_low_goes_to_what():
    d, _ = await decide_for("turn on the lamp", {**CONTROL, "scope": "one_named",
                            "target": {"light.alder": 0.6, "none": 0.4}, "light_action": "turn_on"})
    assert d.outcome == Outcome.CLARIFY
    d, _ = await decide_for("turn on the lamp", {**CONTROL, "scope": "one_named",
                            "target": {"light.alder": 0.3, "light.paper_lantern": 0.29, "light.bedside_globe": 0.2, "none": 0.21},
                            "light_action": "turn_on"})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_WHICH_DEVICE"  # which lamp (round 2)
    assert "light.alder" not in d.clarify.options  # not named by any word said


async def test_two_plausible_targets_pick():
    d, _ = await decide_for("turn on the backlight", {**CONTROL, "scope": "one_named",
                            "target": {"light.backlight_left": 0.5, "light.backlight_right": 0.45, "none": 0.05},
                            "light_action": "turn_on"})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_WHICH_DEVICE"
    assert list(d.clarify.options) == ["light.backlight_left", "light.backlight_right"]
    assert d.clarify.plan["light.backlight_right"].calls == (ServiceCall("light", "turn_on", ("light.backlight_right",), {}),)


async def test_target_none_confident_is_no_device():
    d, _ = await decide_for("turn on the jacuzzi", {**CONTROL, "scope": "one_named", "target": "none",
                            "light_action": "turn_on"})
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_NO_DEVICE"


async def test_queryable_only_target_cannot_be_controlled():
    d, _ = await decide_for("turn on the motion sensor", {**CONTROL, "scope": "one_named",
                            "target": "binary_sensor.demo_bedroom_motion", "switch_action": "turn_on"})
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_CANT"


# ---- scenes -----------------------------------------------------------------------------------
async def test_mode_script():
    d, _ = await decide_for("movie mode", {"intent": "activate_scene", "scene": "script.demo_movie_mode"})
    assert calls_of(d) == [("script", "turn_on", ["script.demo_movie_mode"], {})]
    assert d.reply_key == "R_SCENE" and d.reply_params["name"] == "Living: Film"


async def test_preset_calls_emitted_verbatim():
    d, _ = await decide_for("kitchen daytime", {"intent": "activate_scene", "scene": "preset.kitchen_day"})
    assert calls_of(d) == [("light", "turn_on", ["light.pendant_lower", "light.pendant_upper"],
                            {"color_temp_kelvin": 3000, "brightness_pct": 85, "transition": 0.5})]


async def test_ambiguous_scene_uses_speaker_area():
    probs = {"scene.demo_common_relax": 0.5, "scene.demo_bedroom_relax": 0.45, "none": 0.05}
    d, _ = await decide_for("Hue scene relax", {"intent": "activate_scene", "scene": probs}, speaker_area="bedroom")
    assert calls_of(d) == [("scene", "turn_on", ["scene.demo_bedroom_relax"], {})]


async def test_ambiguous_scene_with_area_noul_filters():
    probs = {"scene.demo_common_relax": 0.5, "scene.demo_bedroom_relax": 0.45, "none": 0.05}
    d, _ = await decide_for("Hue scene relax in the bedroom", {"intent": "activate_scene", "scene": probs, "area__bedroom": 0.9},
                            speaker_area="living_room")
    assert calls_of(d) == [("scene", "turn_on", ["scene.demo_bedroom_relax"], {})]


async def test_ambiguous_scene_without_room_picks():
    probs = {"scene.demo_common_relax": 0.5, "scene.demo_bedroom_relax": 0.45, "none": 0.05}
    d, _ = await decide_for("Hue scene relax", {"intent": "activate_scene", "scene": probs})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_WHICH_SCENE"
    assert list(d.clarify.options) == ["scene.demo_common_relax", "scene.demo_bedroom_relax"]


async def test_unknown_scene():
    d, _ = await decide_for("party mode", {"intent": "activate_scene", "scene": "none"})
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_NO_SCENE"


# ---- routing, safety --------------------------------------------------------------------------
async def test_security_noul_refuses():
    d, _ = await decide_for("open the front door", {**CONTROL, "security": 0.97})
    assert d.outcome == Outcome.REFUSED and d.reply_key == "R_LOCK" and d.calls == ()


async def test_denied_target_refuses_even_when_security_noul_is_low():
    d, _ = await decide_for("switch the front door", {**CONTROL, "security": 0.0, "scope": "one_named",
                            "target": "lock.entry_lock", "switch_action": "turn_on"})
    assert d.outcome == Outcome.REFUSED and d.reply_key == "R_LOCK" and d.calls == ()


async def test_denied_script_as_scene_refuses():
    d, _ = await decide_for("run the door script", {"intent": "activate_scene", "scene": "script.toggle_entry_lock"},
                            overrides={"scope.include": ["script.*"]})
    assert d.outcome == Outcome.REFUSED and d.reply_key == "R_LOCK" and d.calls == ()


async def test_general_goes_to_luna():
    d, _ = await decide_for("what's the capital of france", {"intent": "general"})
    assert d.use_luna and d.outcome == Outcome.ANSWERED


async def test_unclear_intent_without_llm_need_asks_what():
    d, _ = await decide_for("hmm", {"intent": ("device_control", 0.4), "needs_llm": 0.2})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_WHAT"
    d, _ = await decide_for("hmm", {"intent": ("device_control", 0.4), "needs_llm": 0.8})
    assert d.use_luna


async def test_timers_and_nevermind():
    d, _ = await decide_for("set a timer for 5 minutes", {"intent": "timer_alarm_set"})
    assert d.schedule == {"op": "set", "kind": "timer", "label": "", "seconds": 300}
    assert d.outcome == Outcome.ANSWERED
    d, _ = await decide_for("never mind", {"intent": "nevermind"})
    assert d.outcome == Outcome.ANSWERED and d.reply_key == "R_OK"


async def test_list_add_defaults_to_shopping_list():
    d, _ = await decide_for("add oat milk to the shopping list", {"intent": "list_add"})
    assert calls_of(d) == [("todo", "add_item", ["todo.shopping_list"], {"item": "oat milk"})]
    assert d.reply_key == "R_ADDED"


async def test_list_add_without_item_asks():
    d, _ = await decide_for("add something", {"intent": "list_add"})
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_WHAT_ITEM"


# ---- compound ---------------------------------------------------------------------------------
async def test_compound_executes_all_parts():
    d, _ = await decide_for("turn off the lights and pause the music", {
        "intent": "device_control", "is_compound": 0.95,
        "p0__intent": "device_control", "p0__scope": "unspecified", "p0__device_type": "light", "p0__light_action": "turn_off",
        "p1__intent": "device_control", "p1__scope": "unspecified", "p1__device_type": "media_player",
        "p1__media_action": "media_pause"}, speaker_area="bedroom")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("light", "turn_off", ["light.bedroom_group"], {}),
                           ("media_player", "media_pause", ["media_player.demo_bedroom_speaker"], {})]


async def test_compound_with_an_unclear_part_does_nothing():
    d, _ = await decide_for("turn off the lights and tell me a joke", {
        "intent": "device_control", "is_compound": 0.95,
        "p0__intent": "device_control", "p0__scope": "unspecified", "p0__device_type": "light", "p0__light_action": "turn_off",
        "p1__intent": "general"}, speaker_area="bedroom")
    assert d.outcome == Outcome.CLARIFY and d.reply_key == "R_ONE_AT_A_TIME" and d.calls == ()
    assert d.clarify is None


async def test_unsplit_compound_asks_one_at_a_time():
    d, _ = await decide_for("lights off music off", {**CONTROL, "is_compound": 0.9})
    assert d.reply_key == "R_ONE_AT_A_TIME"


# ---- session: recent targets and clarification replies ----------------------------------------
async def test_followup_uses_recent_targets():
    s = SessionSnapshot(recent_targets=["light.alder"])
    d, _ = await decide_for("turn it up", {**CONTROL, "scope": "recent", "light_action": "brighten"}, session=s)
    assert calls_of(d) == [("light", "turn_on", ["light.alder"], {"brightness_step_pct": 20})]


def _pending(kind, options):
    plan = {o: PlanItem((ServiceCall("scene", "turn_on", (o,), {}),), "R_SCENE", {"name": o}, (o,)) for o in options}
    spec = ClarifySpec(kind, "Which one: A or B?", {o: o for o in options}, plan)
    return Pending(kind=kind, question=spec.question, options=spec.options, plan=spec.plan, expires=time.time() + 60)


async def test_pick_reply_completes_the_plan():
    s = SessionSnapshot(pending=_pending("pick", ["scene.demo_common_relax", "scene.demo_bedroom_relax"]))
    d, req = await decide_for("the second one", {"intent": "clarify_reply", "clarify_pick": "scene.demo_bedroom_relax"}, session=s)
    assert "clarify_pick" in req.questions
    assert calls_of(d) == [("scene", "turn_on", ["scene.demo_bedroom_relax"], {})]


async def test_pick_all():
    s = SessionSnapshot(pending=_pending("pick", ["scene.demo_common_relax", "scene.demo_bedroom_relax"]))
    d, _ = await decide_for("both", {"intent": "clarify_reply", "clarify_pick": "all"}, session=s)
    assert calls_of(d) == [("scene", "turn_on", ["scene.demo_bedroom_relax", "scene.demo_common_relax"], {})]


async def test_declined_confirm_is_okay():
    s = SessionSnapshot(pending=_pending("confirm", ["scene.demo_common_relax"]))
    d, _ = await decide_for("no", {"intent": "clarify_reply", "clarify_affirm": 0.1}, session=s)
    assert d.outcome == Outcome.ANSWERED and d.reply_key == "R_OK" and d.calls == ()


async def test_new_command_while_pending_drops_pending():
    s = SessionSnapshot(pending=_pending("confirm", ["scene.demo_common_relax"]))
    d, _ = await decide_for("turn off alder", {**CONTROL, "scope": "one_named", "target": "light.alder",
                            "light_action": "turn_off"}, session=s)
    assert calls_of(d) == [("light", "turn_off", ["light.alder"], {})]


# ---- queries ----------------------------------------------------------------------------------
QUERY = {"intent": "device_query"}


@pytest.mark.parametrize("answers,key", [
    ({**QUERY, "query_attribute": "locked", "scope": "one_named", "target": "lock.entry_lock"}, "Q_LOCKED"),
    ({**QUERY, "query_attribute": "locked", "scope": "one_named", "target": "binary_sensor.entry_contact"}, "Q_LOCKED"),
    ({**QUERY, "query_attribute": "open_closed", "scope": "one_named", "target": "binary_sensor.entry_contact"}, "Q_CLOSED"),
    ({**QUERY, "query_attribute": "on_off", "scope": "one_named", "target": "automation.demo_morning_scene"}, "Q_ON_OFF"),
    ({**QUERY, "query_attribute": "position", "scope": "unspecified", "device_type": "cover"}, "Q_CLOSED"),
    ({**QUERY, "query_attribute": "sunset"}, "Q_SUN"),
    ({**QUERY, "query_attribute": "next_alarm"}, "Q_NEXT_ALARM"),
    ({**QUERY, "query_attribute": "next_timer"}, "Q_NO_ALARM"),
    ({**QUERY, "query_attribute": "motion", "scope": "area", "area__bedroom": 0.9}, "Q_MOTION_NO"),
    ({**QUERY, "query_attribute": "playing", "scope": "one_named", "target": "media_player.demo_stereo"}, "Q_NOTHING_PLAYING"),
    ({**QUERY, "query_attribute": "on_off", "scope": "area", "device_type": "light", "area__living_room": 0.9}, "Q_AREA_ALL_OFF"),
    ({**QUERY, "query_attribute": "color", "scope": "one_named", "target": "light.alder"}, "R_CANT"),
])
async def test_queries_answer_without_calls(answers, key):
    d, _ = await decide_for("question?", answers, speaker_area="bedroom")
    assert d.calls == ()
    assert d.reply_key == key
    assert d.outcome in (Outcome.ANSWERED, Outcome.UNSUPPORTED)


async def test_weather_reads_forecast():
    d, _ = await decide_for("will it rain tomorrow", {"intent": "weather", "weather_when": "tomorrow"})
    assert d.outcome == Outcome.ANSWERED and d.reply_key == "W_DAY"
    assert d.reads[0].service == "get_forecasts"


async def test_weather_elsewhere_goes_to_luna():
    d, _ = await decide_for("weather in paris", {"intent": "weather", "weather_when": "elsewhere"})
    assert d.use_luna


async def test_room_sweep_turns_high_stakes_off_but_never_on():
    d, _ = await decide_for("turn off the kitchen", {**CONTROL, "scope": "area", "device_type": "any",
                            "area__kitchen": 0.9, "light_action": "turn_off", "switch_action": "turn_off"})
    assert "switch.demo_coffee_maker" in {e for c in d.calls for e in c.entity_ids}
    d, _ = await decide_for("toggle everything in the kitchen", {**CONTROL, "scope": "area", "device_type": "any",
                            "area__kitchen": 0.9, "light_action": "toggle", "switch_action": "toggle"},
                            initial_state={"switch.demo_coffee_maker": {"state": "on"}})
    assert ("switch", "turn_off", ["switch.demo_coffee_maker"], {}) in calls_of(d)
    d, _ = await decide_for("toggle everything in the kitchen", {**CONTROL, "scope": "area", "device_type": "any",
                            "area__kitchen": 0.9, "light_action": "toggle", "switch_action": "toggle"})
    assert "switch.demo_coffee_maker" not in {e for c in d.calls for e in c.entity_ids}


async def test_s3_living_room_colour_never_reaches_the_kitchen():
    d, _ = await decide_for("turn the living room lights red", {**CONTROL, "scope": "area", "device_type": "light",
                            "area__living_room": 0.95, "light_action": "set_color", "color": "red"},
                            speaker_area="kitchen")
    ids = {e for c in d.calls for e in c.entity_ids}
    assert ids == set(LIVING_DIMMABLE)


async def test_s3_area_query_reports_only_lights_in_that_room():
    d, _ = await decide_for("are any lights on in the living room?", {"intent": "device_query", "scope": "area",
                            "device_type": "light", "area__living_room": 0.95, "query_attribute": "on_off"},
                            speaker_area="kitchen",
                            initial_state={"light.pendant_upper": {"state": "on"}, "light.pendant_lower": {"state": "on"}})
    assert d.reply_key == "Q_AREA_ALL_OFF"


async def test_s3_house_scope_still_uses_the_group():
    d, _ = await decide_for("turn off all the lights", {**CONTROL, "scope": "house", "device_type": "light",
                            "light_action": "turn_off"}, speaker_area="living_room")
    ids = {e for c in d.calls for e in c.entity_ids}
    assert "light.common_lights" in ids and "light.alder" not in ids


def _light_confirm(ids):
    item = PlanItem((ServiceCall("light", "turn_on", tuple(ids), {}),), "R_ON", {"names": "the lights"}, tuple(ids))
    return Pending(kind="confirm", question="Do you want me to turn on the lights?", options={"living_room": "Living Room"},
                   plan={"living_room": item}, expires=time.time() + 60)


async def test_answering_a_confirm_with_another_device_moves_the_action_to_it():
    s = SessionSnapshot(pending=_light_confirm(["light.alder", "light.paper_lantern"]))
    d, _ = await decide_for("the paper lantern one", {"intent": {"clarify_reply": 0.55, "device_control": 0.45},
                                              "clarify_affirm": 0.37, "target": "light.paper_lantern"}, session=s)
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("light", "turn_on", ["light.paper_lantern"], {})]


async def test_a_correction_never_moves_the_action_to_another_domain_or_a_denied_device():
    s = SessionSnapshot(pending=_light_confirm(["light.alder"]))
    for target in ("switch.demo_coffee_maker", "lock.entry_lock"):
        d, _ = await decide_for("the other one", {"intent": "clarify_reply", "clarify_affirm": 0.1, "target": target},
                                session=s)
        assert d.calls == () and d.outcome != Outcome.EXECUTED


async def test_a_room_question_keeps_every_room_and_speaks_the_list_only_when_short():
    d, _ = await decide_for("turn off the lights", {**CONTROL, "scope": "area", "device_type": "light",
                                                    "light_action": "turn_off"})
    assert d.reply_key == "R_WHICH_ROOM"
    assert {"living_room", "kitchen", "bedroom", "bathroom"} <= set(d.clarify.options)
    from ha_jev.replies import render_template
    assert render_template(d.reply_key, d.reply_params) == "Which room?"


async def test_naming_an_offered_room_answers_a_room_question():
    first, _ = await decide_for("turn off the lights", {**CONTROL, "scope": "area", "device_type": "light",
                                                        "light_action": "turn_off"})
    c = first.clarify
    s = SessionSnapshot(pending=Pending(c.kind, c.question, c.options, c.plan, time.time() + 60))
    d, _ = await decide_for("the kitchen", {"intent": {"clarify_reply": 0.48, "general": 0.36, "activate_scene": 0.16},
                                            "clarify_pick": {"none": 0.6, "bathroom": 0.4}, "area__kitchen": 0.97},
                            session=s)
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("light", "turn_off", ["light.kitchen_amber_lamp", "light.pendant_lower",
                                                  "light.pendant_upper"], {})]
