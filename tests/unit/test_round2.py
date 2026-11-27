"""Round-2 holdout failure classes (orchestrator): general fixes, one test per class shape."""

from ha_jev.models import Outcome
from ha_jev.policy import SessionSnapshot
from tests.unit.helpers import calls_of, decide_for

CONTROL = {"intent": "device_control"}


# ---- class 1: whole-home plurals execute house-wide ------------------------------------------------
async def test_all_the_lights_goes_house_wide_even_when_scope_says_area():
    d, _ = await decide_for("set all the lights to 50%", {
        **CONTROL, "scope": {"area": 0.97, "house": 0.03}, "whole_home": 0.95, "device_type": "light",
        "light_action": "set_brightness", "number_is_setting__n0": 0.97}, speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    ids = {e for c in d.calls for e in c.entity_ids}
    assert {"light.common_lights", "light.bedroom_group", "light.bathroom_group"} <= ids


async def test_everything_off_split_between_room_and_house_executes_house_wide():
    d, _ = await decide_for("turn off everything", {
        **CONTROL, "scope": {"area": 0.51, "house": 0.49}, "whole_home": 0.9, "device_type": "any",
        "light_action": "turn_off", "switch_action": "turn_off"}, speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert "light.bedroom_group" in {e for c in d.calls for e in c.entity_ids}


async def test_the_lights_without_all_stays_in_the_speakers_room():
    d, _ = await decide_for("turn off the lights", {
        **CONTROL, "scope": "area", "whole_home": 0.1, "device_type": "light", "light_action": "turn_off"},
        speaker_area="bedroom")
    assert calls_of(d) == [("light", "turn_off", ["light.bedroom_group"], {})]


async def test_a_named_room_beats_whole_home():
    d, _ = await decide_for("turn off all the bedroom lights", {
        **CONTROL, "scope": "area", "whole_home": 0.8, "area__bedroom": 0.95, "device_type": "light",
        "light_action": "turn_off"}, speaker_area="living_room")
    assert calls_of(d) == [("light", "turn_off", ["light.bedroom_group"], {})]


async def test_are_any_lights_on_asks_about_the_whole_home():
    d, _ = await decide_for("are any lights on?", {
        "intent": "device_query", "scope": {"area": 0.99, "house": 0.01}, "whole_home": 0.9, "device_type": "light",
        "query_attribute": "on_off"}, speaker_area="bedroom", initial_state={"light.alder": {"state": "on"}})
    assert d.reply_key == "Q_HOUSE_SOME_ON"


async def test_a_quantifier_word_carries_whole_home_when_the_noul_is_unsure():
    d, _ = await decide_for("switch all the lamps on", {
        **CONTROL, "scope": {"area": 0.9, "house": 0.1}, "whole_home": 0.4, "device_type": "light",
        "light_action": "turn_on"}, speaker_area="bedroom")
    assert d.outcome == Outcome.EXECUTED
    assert "light.common_lights" in {e for c in d.calls for e in c.entity_ids}


def test_quantifier_extraction():
    from ha_jev.extract import extract
    assert extract("turn off all the lights").quantifier and extract("pause everything").quantifier
    assert not extract("turn off the lights").quantifier and not extract("hallway lamp on").quantifier


# ---- class 2: bare and room-qualified transport ------------------------------------------------------
MEDIA = {**CONTROL, "device_type": "media_player"}


async def test_pause_the_music_pauses_whatever_is_playing_not_a_player_named_music():
    d, _ = await decide_for("pause the music", {
        **MEDIA, "scope": {"one_named": 0.51, "area": 0.39, "unspecified": 0.1},
        "target": {"media_player.demo_stereo": 0.64, "none": 0.36}, "media_action": "media_pause"},
        speaker_area="living_room", initial_state={"media_player.demo_bedroom_speaker": {"state": "playing"}})
    assert calls_of(d) == [("media_player", "media_pause", ["media_player.demo_bedroom_speaker"], {})]


async def test_bare_pause_with_nothing_playing_uses_the_rooms_default_player():
    d, _ = await decide_for("pause", {**MEDIA, "scope": {"area": 0.6, "unspecified": 0.4},
                                      "media_action": "media_pause"}, speaker_area="bedroom")
    assert calls_of(d) == [("media_player", "media_pause", ["media_player.demo_bedroom_speaker"], {})]


async def test_transport_in_a_named_room_uses_that_rooms_player():
    d, _ = await decide_for("resume music in the bedroom", {
        **MEDIA, "scope": {"one_named": 0.62, "area": 0.36, "unspecified": 0.02}, "area__bedroom": 0.99,
        "target": {"none": 0.81, "media_player.demo_bedroom_speaker": 0.09, "media_player.demo_stereo": 0.1},
        "media_action": "media_play"}, speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("media_player", "media_play", ["media_player.demo_bedroom_speaker"], {})]


async def test_an_action_the_device_cannot_do_is_not_a_competing_reading():
    d, _ = await decide_for("skip on the bedroom speaker", {
        **MEDIA, "scope": "one_named", "target": "media_player.demo_bedroom_speaker",
        "media_action": {"media_next_track": 0.5, "turn_off": 0.35, "none": 0.15}},
        speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("media_player", "media_next_track", ["media_player.demo_bedroom_speaker"], {})]


# ---- class 5: a pronoun with no referent never acts on everything ---------------------------------------
async def test_set_it_to_50_with_nothing_recent_asks():
    d, _ = await decide_for("set it to 50", {
        **MEDIA, "scope": {"unspecified": 0.7, "recent": 0.0, "area": 0.3}, "media_action": "volume_set",
        "number_is_setting__n0": 0.95}, speaker_area="living_room")
    assert d.outcome == Outcome.CLARIFY and d.calls == ()


async def test_set_it_to_50_with_a_recent_speaker_sets_that_speaker():
    d, _ = await decide_for("set it to 50", {
        **MEDIA, "scope": {"recent": 0.6, "unspecified": 0.4}, "media_action": "volume_set",
        "number_is_setting__n0": 0.95}, speaker_area="living_room",
        session=SessionSnapshot(recent_targets=["media_player.demo_soundbar"]))
    assert calls_of(d) == [("media_player", "volume_set", ["media_player.demo_soundbar"], {"volume_level": 0.5})]


# ---- class 3: named device + colour-temperature word; a named group set to a preset ----------------------
async def test_a_named_bulb_set_to_a_named_white_executes():
    d, _ = await decide_for("make the paper lantern warm white", {
        **CONTROL, "scope": "one_named", "target": "light.paper_lantern", "device_type": "light",
        "light_action": {"set_color_temp": 0.5, "set_color": 0.3, "warmer": 0.19, "none": 0.01},
        "color": "warm_white"}, speaker_area="living_room")
    # set_color_temp (no number, a named white) and set_color both mean 2700 K: they add up
    assert calls_of(d) == [("light", "turn_on", ["light.paper_lantern"], {"color_temp_kelvin": 2700})]
    assert d.outcome == Outcome.EXECUTED


async def test_a_reading_that_finds_nothing_to_do_does_not_compete_with_a_preset():
    d, _ = await decide_for("set the kitchen pendants to kitchen evening", {
        "intent": {"activate_scene": 0.5, "device_control": 0.5}, "scope": "one_named",
        "target": {"light.pendant_lower": 0.6, "none": 0.4}, "device_type": "light", "light_action": "none",
        "scene": {"preset.kitchen_evening": 1.0}}, speaker_area="kitchen")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d)[0][2] == ["light.pendant_lower", "light.pendant_upper"]


async def test_a_nothing_to_do_reading_never_inflates_a_general_question():
    d, _ = await decide_for("what is kitchen evening", {
        "intent": {"general": 0.5, "activate_scene": 0.3, "device_control": 0.2}, "needs_llm": 0.2,
        "scope": "one_named", "light_action": "none", "scene": {"preset.kitchen_evening": 1.0}},
        speaker_area="kitchen")
    assert d.outcome != Outcome.EXECUTED


# ---- class 6: an out-of-catalog device type -------------------------------------------------------------
async def test_a_device_type_the_home_does_not_have_is_unsupported():
    d, _ = await decide_for("set the thermostat to 70", {
        **CONTROL, "scope": {"area": 0.52, "one_named": 0.45, "unspecified": 0.03}, "device_type": "other",
        "target": {"none": 1.0}}, speaker_area="living_room")
    assert d.outcome == Outcome.UNSUPPORTED and d.reply_key == "R_NO_DEVICE"


# ---- class 7: a switch named after a device, said in full -------------------------------------------------
async def test_the_full_name_of_a_do_not_disturb_switch_beats_the_player_it_is_named_after():
    d, _ = await decide_for("enable do not disturb on the stereo", {
        **CONTROL, "scope": "one_named", "device_type": "media_player",
        "target": {"switch.demo_stereo_do_not_disturb": 0.55, "media_player.demo_stereo": 0.45},
        "switch_action": "turn_on", "media_action": "volume_mute"}, speaker_area="living_room")
    assert calls_of(d) == [("switch", "turn_on", ["switch.demo_stereo_do_not_disturb"], {})]


async def test_a_name_inside_another_said_name_is_not_a_second_mention():
    d, _ = await decide_for("turn off the bedroom radio do not disturb", {
        **CONTROL, "scope": {"one_named": 0.65, "several_named": 0.35},
        "target": {"switch.demo_bedroom_radio_do_not_disturb": 0.95, "media_player.demo_bedroom_radio": 0.05},
        "mentioned__switch.demo_bedroom_radio_do_not_disturb": 0.95, "mentioned__media_player.demo_bedroom_radio": 0.8,
        "switch_action": "turn_off", "media_action": "turn_off"}, speaker_area="bedroom")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("switch", "turn_off", ["switch.demo_bedroom_radio_do_not_disturb"], {})]


async def test_a_negated_command_with_action_none_never_executes_the_other_reading():
    d, _ = await decide_for("don't turn off the kitchen lights", {
        **CONTROL, "scope": "area", "device_type": "light", "area__kitchen": 0.99,
        "light_action": {"none": 0.67, "turn_on": 0.30, "turn_off": 0.03}, "scene": {"none": 0.78}},
        speaker_area="living_room")
    assert d.outcome != Outcome.EXECUTED and d.calls == ()


async def test_an_exact_source_name_launches_it():
    d, _ = await decide_for("put on netflix", {
        **CONTROL, "scope": {"one_named": 0.6, "unspecified": 0.4}, "device_type": "media_player",
        "target": {"none": 0.55, "media_player.demo_tv": 0.45},
        "media_action": {"select_source": 0.6, "none": 0.25, "play_content": 0.15}, "source": "Netflix",
        "scene": {"none": 0.9}}, speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert calls_of(d) == [("media_player", "select_source", ["media_player.demo_tv"],
                            {"source": "Netflix"})]


async def test_a_bare_transport_verb_takes_its_device_type_from_the_only_confident_action():
    d, _ = await decide_for("pause", {
        **CONTROL, "scope": "unspecified", "device_type": {"none": 0.8, "media_player": 0.16, "switch": 0.04},
        "media_action": {"media_pause": 0.95, "none": 0.05}, "switch_action": {"none": 0.87, "turn_off": 0.13}},
        speaker_area="living_room")
    assert calls_of(d) == [("media_player", "media_pause", ["media_player.demo_stereo"], {})]


async def test_two_confident_action_choices_leave_the_device_type_open():
    d, _ = await decide_for("turn it off", {
        **CONTROL, "scope": "unspecified", "device_type": "none",
        "media_action": {"turn_off": 0.9, "none": 0.1}, "light_action": {"turn_off": 0.9, "none": 0.1}},
        speaker_area="living_room")
    assert d.calls == ()


# ---- class 5b: a bare mode name shared by rooms --------------------------------------------------------
async def test_a_mode_name_shared_by_rooms_asks_without_a_speaker_area():
    d, _ = await decide_for("wind down", {"intent": "activate_scene",
                                          "scene": {"script.demo_living_wind_down": 0.8, "script.demo_bedroom_wind_down": 0.2}})
    assert d.outcome == Outcome.CLARIFY
    assert {"script.demo_living_wind_down", "script.demo_bedroom_wind_down"} <= set(d.clarify.options)


async def test_the_speakers_room_picks_the_mode_named_for_it():
    d, _ = await decide_for("wind down", {"intent": "activate_scene",
                                          "scene": {"script.demo_living_wind_down": 0.8, "script.demo_bedroom_wind_down": 0.2}},
                            speaker_area="bedroom")
    assert calls_of(d) == [("script", "turn_on", ["script.demo_bedroom_wind_down"], {})]


async def test_a_distinguishing_word_said_keeps_the_chosen_scene():
    d, _ = await decide_for("pendant evening mode", {"intent": "activate_scene",
                                                   "scene": {"preset.kitchen_evening": 0.9, "script.demo_living_evening": 0.1}},
                            speaker_area="living_room")
    assert calls_of(d)[0][2] == ["light.pendant_lower", "light.pendant_upper"]


async def test_turn_it_down_with_nothing_recent_is_the_master_volume():
    # "it" in "turn it down" is idiom: a relative whole-house action needs no recent device ("set it to 50" does)
    d, _ = await decide_for("turn it down", {**MEDIA, "scope": "unspecified", "media_action": "volume_down"},
                            speaker_area="living_room")
    assert d.outcome == Outcome.EXECUTED
    assert {e for c in d.calls for e in c.entity_ids} == {
        "media_player.demo_bedroom_speaker", "media_player.demo_soundbar", "media_player.demo_stereo"}
