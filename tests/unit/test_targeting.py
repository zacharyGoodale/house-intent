"""Targeting and confidence (adversarial review S4, S5, S6, S9; standards review S-2, D9-5, D9-8).

Answers are given as probability maps, the shape live Jev returns, so the tests exercise how policy combines
probabilities rather than a single `confidence` per Choice."""

from ha_jev.models import Outcome
from ha_jev.policy import SessionSnapshot
from ha_jev.replies import render_template
from tests.unit.helpers import calls_of, decide_for

CONTROL = {"intent": "device_control"}


# ---- S5: scope readings that give the same calls add up ----------------------------------------
async def test_bedroom_blinds_area_and_named_readings_are_the_same_plan():
    d, _ = await decide_for("close the bedroom blinds halfway", {
        **CONTROL, "scope": {"area": 0.55, "one_named": 0.45}, "target": {"cover.bedroom_shade": 0.98, "none": 0.02},
        "device_type": "cover", "area__bedroom": 0.97, "cover_action": "set_cover_position",
        "level_word": "half"}, speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("cover", "set_cover_position", ["cover.bedroom_shade"], {"position": 50})]
    assert d.confidence >= 0.97


async def test_room_group_named_or_swept_is_one_plan():
    d, _ = await decide_for("turn off the bedroom light", {
        **CONTROL, "scope": {"one_named": 0.61, "area": 0.39}, "target": {"light.bedroom_group": 0.69, "none": 0.31},
        "device_type": "light", "area__bedroom": 0.9, "light_action": "turn_off"}, speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("light", "turn_off", ["light.bedroom_group"], {})]


async def test_different_plans_do_not_add_up():
    d, _ = await decide_for("turn off the fern light", {
        **CONTROL, "scope": {"one_named": 0.55, "area": 0.45}, "target": {"light.fern_lamp": 0.5, "none": 0.5},
        "device_type": "light", "light_action": "turn_off"}, speaker_area="bedroom")
    assert d.outcome == Outcome.CLARIFY  # 0.55 is not enough to act; the fern light is what it offers
    assert d.clarify and list(d.clarify.plan) == ["light.fern_lamp"]


async def test_a_confidently_named_device_outweighs_a_room_reading():
    """Review S5: the target Choice says a single device is named, which contradicts the room reading."""
    d, _ = await decide_for("turn off the fern light", {
        **CONTROL, "scope": {"one_named": 0.55, "area": 0.45}, "target": {"light.fern_lamp": 0.99, "none": 0.01},
        "device_type": "light", "light_action": "turn_off"}, speaker_area="bedroom")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("light", "turn_off", ["light.fern_lamp"], {})]


async def test_a_room_that_is_not_in_this_home_is_not_the_speakers_room():
    d, _ = await decide_for("turn on the garden lights", {
        **CONTROL, "scope": "area", "device_type": "light", "unknown_room": 0.95, "light_action": "turn_on"},
        speaker_area="living_room")
    assert d.outcome == Outcome.UNSUPPORTED and d.calls == ()


async def test_action_readings_that_give_the_same_calls_add_up():
    # For a set of lights with one on, a toggle resolves to turn_off: toggle and turn_off are the same plan.
    d, _ = await decide_for("switch the living room lights", {
        **CONTROL, "scope": "area", "device_type": "light", "area__living_room": 0.95,
        "light_action": {"toggle": 0.5, "turn_off": 0.45, "none": 0.05}}, speaker_area="living_room",
        initial_state={"light.alder": {"state": "on"}})
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d)[0][1] == "turn_off"
    assert d.confidence >= 0.9


# ---- S4: a confident named device is not widened to the room ------------------------------------
async def test_named_speaker_is_not_widened_to_every_speaker_in_the_room():
    d, _ = await decide_for("set the stereo volume to 20", {
        **CONTROL, "scope": {"area": 0.63, "one_named": 0.37}, "target": {"media_player.demo_stereo": 0.99},
        "device_type": "media_player", "media_action": "volume_set", "number_is_setting__n0": 0.95},
        speaker_area="living_room")
    targets = [e for c in d.calls for e in c.entity_ids] or list(d.targets)
    assert targets == ["media_player.demo_stereo"]


async def test_named_light_is_not_widened_to_the_room():
    d, _ = await decide_for("turn on the backlight 1", {
        **CONTROL, "scope": {"area": 0.6, "one_named": 0.4}, "target": {"light.backlight_left": 0.97, "none": 0.03},
        "device_type": "light", "light_action": "turn_on"}, speaker_area="living_room")
    targets = [e for c in d.calls for e in c.entity_ids] or list(d.targets)
    assert targets == ["light.backlight_left"]


async def test_a_weak_named_reading_does_not_override_the_room():
    d, _ = await decide_for("turn off the living room lights", {
        **CONTROL, "scope": {"area": 0.9, "one_named": 0.1}, "target": {"light.alder": 0.5, "none": 0.5},
        "device_type": "light", "area__living_room": 0.95, "light_action": "turn_off"}, speaker_area="bedroom")
    assert "light.alder" in calls_of(d)[0][2] and len(calls_of(d)[0][2]) > 1


async def test_an_anaphor_with_nothing_recent_asks_instead_of_sweeping():
    d, _ = await decide_for("toggle it", {**CONTROL, "scope": "area", "device_type": "light",
                                          "light_action": "toggle"}, speaker_area="living_room")
    assert d.outcome == Outcome.CLARIFY and d.calls == ()


# ---- S6: a room without that kind of device falls back to the house rule -----------------------
async def test_open_the_blinds_from_a_room_without_blinds_uses_the_only_blind():
    d, _ = await decide_for("open the blinds", {
        **CONTROL, "scope": {"area": 0.89, "one_named": 0.11}, "target": {"cover.bedroom_shade": 0.59, "none": 0.41},
        "device_type": "cover", "cover_action": "open_cover"}, speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("cover", "open_cover", ["cover.bedroom_shade"], {})]


# ---- S9 / D9-8: never "Which one: ?" ------------------------------------------------------------
def test_which_prompts_never_render_an_empty_option_list():
    for key in ("R_WHICH_DEVICE", "R_WHICH_SCENE"):
        assert "?" in render_template(key, {}) and ": ?" not in render_template(key, {})


async def test_which_device_without_candidates_offers_the_likely_targets():
    d, _ = await decide_for("toggle the side thing", {
        **CONTROL, "scope": "unspecified", "device_type": "switch", "switch_action": "toggle",
        "target": {"switch.living_room_corner lamp": 0.3, "switch.reading_lamp_plug": 0.3, "none": 0.4}}, speaker_area="living_room")
    assert d.outcome == Outcome.CLARIFY
    assert d.reply_key != "R_WHICH_DEVICE" or (d.clarify and d.clarify.options)


async def test_bare_pause_in_a_room_with_one_speaker_uses_it():
    den = {"areas": [{"id": "den", "name": "Den"}],
           "entities": [{"entity_id": "media_player.den_speaker", "friendly_name": "Den speaker", "area_id": "den",
                         "features": ["play", "pause", "volume_set", "next_track", "previous_track"]}],
           "states": {"media_player.den_speaker": {"state": "idle", "attributes": {"volume_level": 0.3}}}}
    d, _ = await decide_for("pause", {**CONTROL, "scope": "unspecified", "device_type": "media_player",
                                      "media_action": "media_pause"}, speaker_area="den", overlays=[den])
    assert calls_of(d) == [("media_player", "media_pause", ["media_player.den_speaker"], {})]


# ---- S-2: status of several devices -------------------------------------------------------------
async def test_status_of_two_named_lights_answers_for_both():
    d, _ = await decide_for("are alder and the paper lantern on", {
        "intent": "device_query", "scope": "several_named", "mentioned__light.alder": 0.9,
        "mentioned__light.paper_lantern": 0.9, "query_attribute": "on_off"},
        initial_state={"light.alder": {"state": "on"}})
    text = render_template(d.reply_key, d.reply_params)
    assert "Alder" in text and "Paper Lantern" in text


async def test_status_after_a_multi_target_command_covers_every_target():
    s = SessionSnapshot(recent_targets=["light.alder", "light.paper_lantern"])
    d, _ = await decide_for("are they on?", {"intent": "device_query", "scope": "recent", "query_attribute": "on_off"},
                            session=s, initial_state={"light.paper_lantern": {"state": "on"}})
    text = render_template(d.reply_key, d.reply_params)
    assert "Alder" in text and "Paper Lantern" in text


# ---- D9-5: which weather entity -----------------------------------------------------------------
SECOND_WEATHER = {"entities": [{"entity_id": "weather.accuweather", "friendly_name": "AccuWeather", "exposed": True}],
                  "states": {"weather.accuweather": {"state": "sunny", "attributes": {"temperature": 90}}}}


async def test_a_second_weather_entity_does_not_change_the_answer_source():
    d, _ = await decide_for("what's the weather", {"intent": "weather", "weather_when": "now"},
                            overlays=[SECOND_WEATHER], overrides={"scope.include": ["weather.forecast_home", "sun.sun"]})
    assert d.reply_params["entity"] == "weather.forecast_home"


async def test_the_weather_default_is_config():
    d, _ = await decide_for("what's the weather", {"intent": "weather", "weather_when": "now"},
                            overlays=[SECOND_WEATHER], overrides={"scope.defaults.weather": "weather.accuweather"})
    assert d.reply_params["entity"] == "weather.accuweather"


async def test_the_named_reading_uses_the_target_given_that_a_device_is_named():
    """'Bedroom volume to 20': target none 0.65, Bedroom Speaker 0.27. If a device is named it is the Bedroom Speaker (0.77), which is
    also what the room reading does, so the two readings add up."""
    d, _ = await decide_for("bedroom volume to 20", {
        **CONTROL, "scope": {"one_named": 0.69, "area": 0.25, "unspecified": 0.06},
        "target": {"none": 0.65, "media_player.demo_bedroom_speaker": 0.27, "media_player.demo_bedroom_radio": 0.08},
        "device_type": "media_player", "area__bedroom": 0.95, "media_action": "volume_set",
        "number_is_setting__n0": 0.96}, speaker_area="living_room")
    assert calls_of(d) == [("media_player", "volume_set", ["media_player.demo_bedroom_speaker"], {"volume_level": 0.2})]


async def test_a_weak_unknown_room_does_not_block_the_speakers_room():
    d, _ = await decide_for("turn off the lights", {
        **CONTROL, "scope": "area", "device_type": "light", "unknown_room": 0.3, "light_action": "turn_off"},
        speaker_area="bedroom")
    assert calls_of(d) == [("light", "turn_off", ["light.bedroom_group"], {})]


async def test_status_query_uses_the_evidence_weighted_scope():
    d, _ = await decide_for("what's playing in the living room?", {
        "intent": "device_query", "scope": {"one_named": 0.48, "unspecified": 0.29, "area": 0.23},
        "target": {"none": 0.84, "media_player.demo_tv": 0.13, "media_player.demo_stereo": 0.03},
        "device_type": "media_player", "area__living_room": 0.98, "query_attribute": "playing"},
        speaker_area="living_room")
    assert d.outcome == Outcome.ANSWERED


# ---- a partial name that fits two devices is a question, not a guess -----------------------------
async def test_a_partial_name_shared_by_two_devices_asks_which():
    d, _ = await decide_for("set the volume on the radio to 30", {
        **CONTROL, "scope": "one_named", "target": {"media_player.demo_bedroom_radio": 1.0},
        "media_action": "volume_set", "number_is_setting__n0": 0.95}, speaker_area="bedroom")
    assert d.outcome == Outcome.CLARIFY
    assert set(d.clarify.options) == {"media_player.demo_bedroom_radio", "media_player.reading_alcove_radio"}


async def test_the_full_name_or_a_distinct_number_is_not_ambiguous():
    for text, target in (("set the volume on the bedroom radio to 30", "media_player.demo_bedroom_radio"),
                         ("turn on backlight 2", "light.backlight_right")):
        d, _ = await decide_for(text, {**CONTROL, "scope": "one_named", "target": target,
                                       "media_action": "volume_set", "light_action": "turn_on",
                                       "number_is_setting__n0": 0.95 if "volume" in text else 0.0},
                                speaker_area="bedroom")
        assert d.outcome == Outcome.EXECUTED, text


async def test_intent_readings_that_run_the_same_script_add_up():
    """'Put on the news podcast': device_control (play_content -> the scene Choice) and activate_scene both run
    the same script, so the intent split does not matter (review S5)."""
    d, _ = await decide_for("put on the news podcast", {
        "intent": {"device_control": 0.6, "activate_scene": 0.39, "general": 0.01}, "scope": "unspecified",
        "device_type": "media_player", "media_action": "play_content",
        "scene": {"script.play_morning_brief": 0.99, "none": 0.01}}, speaker_area="bedroom")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("script", "turn_on", ["script.play_morning_brief"], {})]


async def test_it_with_something_recent_means_the_recent_device():
    s = SessionSnapshot(recent_targets=["light.alder"])
    d, _ = await decide_for("make it warmer", {
        **CONTROL, "scope": {"area": 0.34, "recent": 0.34, "unspecified": 0.30, "one_named": 0.02},
        "target": {"none": 0.53, "light.alder": 0.39, "light.common_lights": 0.08},
        "device_type": "light", "light_action": "warmer"}, speaker_area="living_room", session=s)
    assert d.outcome == Outcome.EXECUTED
    assert [e for c in d.calls for e in c.entity_ids] == ["light.alder"]


async def test_one_lamp_described_only_by_its_kind_asks_which_lamp():
    d, _ = await decide_for("turn on the lamp", {
        **CONTROL, "scope": {"one_named": 0.7, "area": 0.3}, "device_type": "light", "light_action": "turn_on",
        "target": {"none": 0.84, "light.common_lights": 0.11, "light.paper_lantern": 0.05}}, speaker_area="living_room")
    assert d.outcome == Outcome.CLARIFY and d.calls == ()
    assert d.clarify and "light.paper_lantern" in d.clarify.options and len(d.clarify.options) >= 2


async def test_a_device_word_this_home_does_not_have_is_not_found():
    d, _ = await decide_for("turn on the fan", {
        **CONTROL, "scope": "one_named", "device_type": "switch", "switch_action": "turn_on",
        "target": {"none": 0.9, "switch.demo_coffee_maker": 0.1}}, speaker_area="bedroom")
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_NO_DEVICE"


async def test_the_bedroom_lamp_with_two_bedroom_lamps_asks_which_not_the_room_group():
    d, _ = await decide_for("turn on the bedroom lamp", {
        **CONTROL, "scope": {"one_named": 0.61, "area": 0.38}, "device_type": "light", "light_action": "turn_on",
        "area__bedroom": 0.9, "target": {"none": 0.67, "light.bedroom_group": 0.33}}, speaker_area="bedroom")
    assert d.outcome == Outcome.CLARIFY
    assert {"light.bookcase_light", "light.bedside_globe"} <= set(d.clarify.options)


# ---- the same scene name in several rooms -------------------------------------------------------------
async def test_a_scene_name_shared_by_rooms_asks_which_room_when_nothing_decides():
    d, _ = await decide_for("Hue scene rolling hills", {"intent": "activate_scene",
                                              "scene": {"scene.demo_common_rolling_hills": 0.9, "none": 0.1}})
    assert d.outcome == Outcome.CLARIFY
    assert {"scene.demo_common_rolling_hills", "scene.demo_bedroom_rolling_hills"} <= set(d.clarify.options)


async def test_the_speakers_room_decides_between_same_named_scenes():
    d, _ = await decide_for("Hue scene rolling hills", {"intent": "activate_scene",
                                              "scene": {"scene.demo_common_rolling_hills": 0.9, "none": 0.1}},
                            speaker_area="bedroom")
    assert calls_of(d) == [("scene", "turn_on", ["scene.demo_bedroom_rolling_hills"], {})]


async def test_a_longer_scene_name_is_not_a_sibling():
    d, _ = await decide_for("Hue scene relax", {"intent": "activate_scene", "scene": {"scene.demo_bedroom_relax": 0.9, "none": 0.1}},
                            speaker_area="bedroom")
    assert calls_of(d) == [("scene", "turn_on", ["scene.demo_bedroom_relax"], {})]


async def test_a_plural_kind_word_means_all_the_name_siblings():
    d, _ = await decide_for("turn on the backlight lights", {
        **CONTROL, "scope": "one_named", "target": {"light.backlight_left": 0.8, "light.backlight_right": 0.2},
        "light_action": "turn_on"}, speaker_area="living_room")
    assert calls_of(d) == [("light", "turn_on", ["light.backlight_left", "light.backlight_right"], {})]


async def test_s10_is_the_door_open_reads_the_door_sensor_not_the_blind():
    d, _ = await decide_for("is the door open?", {
        "intent": "device_query", "scope": "one_named", "target": {"none": 0.79, "binary_sensor.entry_contact": 0.14,
                                                                   "cover.bedroom_shade": 0.07},
        "query_attribute": "open_closed"}, speaker_area="bedroom")
    assert d.outcome == Outcome.ANSWERED
    assert d.reads[0].entity_ids == ("binary_sensor.entry_contact",)


async def test_a_default_room_without_the_device_and_no_candidates_is_not_found():
    d, _ = await decide_for("turn on the fan", {
        **CONTROL, "scope": {"area": 0.56, "one_named": 0.42, "unspecified": 0.02}, "device_type": "switch",
        "switch_action": "turn_on", "target": {"none": 0.95, "switch.demo_coffee_maker": 0.05}},
        speaker_area="bedroom")
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_NO_DEVICE"


async def test_an_alias_shared_by_several_devices_means_all_of_them():
    d, _ = await decide_for("turn on the kitchen pendant", {
        **CONTROL, "scope": "one_named", "target": {"light.pendant_lower": 0.55, "light.pendant_upper": 0.45},
        "light_action": "turn_on"}, speaker_area="living_room")
    assert calls_of(d) == [("light", "turn_on", ["light.pendant_lower", "light.pendant_upper"], {})]
