"""Decision 9: new rooms, groups, scenes and speakers are routable from registry data alone (spec §4, §14)."""

import hashlib
from pathlib import Path

import pytest

from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.luna import NullLuna
from ha_jev.models import Outcome
from tests.fakes import FakeJev

SRC = Path(__file__).resolve().parents[2] / "src"
BEDROOM_SPEAKER_FEATURES = ["pause", "seek", "volume_set", "volume_mute", "previous_track", "next_track", "play_media", "stop",
                  "play", "shuffle_set", "browse_media", "repeat_set", "grouping"]

OFFICE_ROOM = {
    "areas": [{"id": "office", "name": "Office", "aliases": []}],
    "devices": [{"id": "dev_office_group", "name": "Office", "area_id": "office"}],
    "entities": [
        {"entity_id": "light.desk_lamp", "friendly_name": "Desk Lamp", "area_id": "office",
         "capabilities": {"supported_color_modes": ["color_temp", "xy"], "min_color_temp_kelvin": 2000, "max_color_temp_kelvin": 6535}},
        {"entity_id": "light.office", "friendly_name": "Office", "area_id": "office", "device_id": "dev_office_group",
         "capabilities": {"supported_color_modes": ["color_temp", "xy"]}},
        {"entity_id": "scene.office_focus", "friendly_name": "Office Focus", "area_id": "office"},
        {"entity_id": "script.party_mode", "friendly_name": "Party mode", "exposed": False},
    ],
    "states": {"light.office": {"state": "off", "attributes": {"entity_id": ["light.desk_lamp"]}}},
    "scope": {"include": ["script.party_mode"], "exclude": []},
}
NEW_SCENE_BEDROOM = {"entities": [{"entity_id": "scene.bedroom_movie_night", "friendly_name": "Bedroom Movie Night",
                                   "area_id": "bedroom"}]}
KITCHEN_SPEAKER = {"entities": [{"entity_id": "media_player.kitchen", "friendly_name": "Kitchen Speaker", "area_id": "kitchen",
                                 "device_class": "speaker", "features": BEDROOM_SPEAKER_FEATURES}],
                   "states": {"media_player.kitchen": {"state": "idle", "attributes": {"volume_level": 0.3}}}}
NEW_GROUP_KITCHEN = {"entities": [{"entity_id": "light.kitchen_island", "friendly_name": "Island", "area_id": "kitchen",
                                   "capabilities": {"supported_color_modes": ["color_temp", "xy"]}}],
                     "states": {"light.kitchen_island": {"state": "off",
                                                         "attributes": {"entity_id": ["light.pendant_upper", "light.pendant_lower"]}}}}
UNLOCK_BACK = {"entities": [{"entity_id": "script.unlock_back_door", "friendly_name": "Unlock back door", "exposed": False}],
               "scope": {"include": ["script.unlock_back_door"]}}

CASES = [
    # (overlay, utterance, speaker_area, fake answers, expected calls, expected question ids/options)
    (OFFICE_ROOM, "turn on the office lights", "living_room",
     {"intent": "device_control", "scope": "area", "device_type": "light", "area__office": 0.95, "light_action": "turn_on"},
     ["light.turn_on light.office {}"], {"area__office", "mentioned__light.desk_lamp"}),
    (OFFICE_ROOM, "office focus", "living_room", {"intent": "activate_scene", "scene": "scene.office_focus"},
     ["scene.turn_on scene.office_focus {}"], {"area__office"}),
    (OFFICE_ROOM, "party mode", "living_room", {"intent": "activate_scene", "scene": "script.party_mode"},
     ["script.turn_on script.party_mode {}"], set()),
    (NEW_SCENE_BEDROOM, "movie night in the bedroom", "bedroom",
     {"intent": "activate_scene", "scene": "scene.bedroom_movie_night"}, ["scene.turn_on scene.bedroom_movie_night {}"], set()),
    (KITCHEN_SPEAKER, "pause the kitchen speaker", "living_room",
     {"intent": "device_control", "scope": "one_named", "target": "media_player.kitchen", "media_action": "media_pause"},
     ["media_player.media_pause media_player.kitchen {}"], {"mentioned__media_player.kitchen"}),
    (KITCHEN_SPEAKER, "louder", "living_room",
     {"intent": "device_control", "scope": "unspecified", "device_type": "media_player", "media_action": "volume_up"},
     ['media_player.volume_set media_player.demo_bedroom_speaker,media_player.demo_soundbar,media_player.demo_stereo,media_player.kitchen {"volume_level": 0.4}'],
     set()),
    (NEW_GROUP_KITCHEN, "kitchen lights off", "living_room",
     {"intent": "device_control", "scope": "area", "device_type": "light", "area__kitchen": 0.95, "light_action": "turn_off"},
     ["light.turn_off light.kitchen_amber_lamp,light.kitchen_island {}"], {"mentioned__light.kitchen_island"}),
]


def _src_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(SRC.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize("overlay,text,area,answers,expected,question_ids", CASES)
async def test_seed_overlays_are_routable_without_code_changes(overlay, text, area, answers, expected, question_ids):
    before = _src_digest()
    ha = StubHA.default(overlays=[overlay])
    settings = load_settings().with_scope_extension(**ha.overlay_scope)
    jev = FakeJev(answers)
    core = await Core.create(settings, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle(text, speaker_area=area)
    asked = set(jev.calls[0][1])
    assert question_ids <= asked
    for eid in [c.split()[1] for c in expected]:
        for one in eid.split(","):
            if one.startswith("scene.") or one.startswith("script."):
                assert one in jev.calls[0][1]["scene"]["criteria"]
    assert r.outcome == Outcome.EXECUTED, r.reply
    assert [c.describe() for c in r.service_calls] == expected
    assert _src_digest() == before


async def test_new_script_named_unlock_is_refused_by_the_name_rule():
    ha = StubHA.default(overlays=[UNLOCK_BACK])
    settings = load_settings().with_scope_extension(**ha.overlay_scope)
    core = await Core.create(settings, ha, FakeJev({"intent": "activate_scene", "scene": "script.unlock_back_door"}),
                             NullLuna(), clock=lambda: ha.now)
    assert "script.unlock_back_door" in core.catalog.entities
    assert "script.unlock_back_door" not in core.catalog.controllable
    r = await core.handle("run the back door script")
    assert r.outcome == Outcome.REFUSED and ha.attempted_calls == []


# ---- review item: a new domain is one table row, added from config alone ----------------------------
FAN_ROW = '''
[[domains]]
domain = "fan"
kind = "fan"
device_type = "fan"
device_type_description = "fans, ceiling fans"
action_question = "fan_action"
action_instructions = "What should happen to the fan(s) in {u}?"
query_attributes = ["on_off"]
target_option = true
mention = true
[domains.actions.turn_on]
service = "turn_on"
reply = "R_ON"
[domains.actions.turn_off]
service = "turn_off"
reply = "R_OFF"
[domains.actions.set_speed]
service = "set_percentage"
feature = "set_speed"
description = "set a specific speed, as a percent or a word like half or full"
value = "percent"
data = { percentage = "$value" }
'''
FAN_OVERLAY = {"entities": [{"entity_id": "fan.ceiling_fan", "friendly_name": "Ceiling fan", "area_id": "bedroom",
                             "features": ["set_speed", "oscillate"]}],
               "states": {"fan.ceiling_fan": {"state": "off", "attributes": {"percentage": 0}}}}


@pytest.fixture
def fan_config(tmp_path):
    import shutil
    cfg = tmp_path / "config"
    shutil.copytree(SRC.parent / "config", cfg)
    (cfg / "capabilities.toml").write_text(FAN_ROW)
    return cfg


@pytest.mark.parametrize("text,answers,expected", [
    ("set the ceiling fan to 50%", {"intent": "device_control", "scope": "one_named", "target": "fan.ceiling_fan",
                                    "fan_action": "set_speed", "number_is_setting__n0": 0.95},
     ['fan.set_percentage fan.ceiling_fan {"percentage": 50}']),
    ("turn off the bedroom fans", {"intent": "device_control", "scope": "area", "device_type": "fan",
                                   "area__bedroom": 0.95, "fan_action": "turn_off"},
     ["fan.turn_off fan.ceiling_fan {}"]),
])
async def test_a_fan_domain_added_from_config_and_an_overlay_needs_no_code(fan_config, text, answers, expected):
    before = _src_digest()
    ha = StubHA.default(overlays=[FAN_OVERLAY])
    settings = load_settings(config_dir=fan_config).with_scope_extension(**ha.overlay_scope)
    jev = FakeJev(answers)
    core = await Core.create(settings, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle(text, speaker_area="bedroom")
    questions = jev.calls[0][1]
    assert "fan_action" in questions and "set_speed" in questions["fan_action"]["criteria"]
    assert "fan" in questions["device_type"]["criteria"]
    assert "fan.ceiling_fan" in questions["target"]["criteria"]
    assert r.outcome == Outcome.EXECUTED, r.reply
    assert [c.describe() for c in r.service_calls] == expected
    assert _src_digest() == before


async def test_a_config_row_cannot_make_a_lock_controllable(fan_config):
    from ha_jev.config import ConfigError
    (fan_config / "capabilities.toml").write_text(
        '[[domains]]\ndomain = "lock"\nkind = "lock"\n[domains.actions.unlock]\nservice = "unlock"\n')
    with pytest.raises(ConfigError):
        load_settings(config_dir=fan_config)


# ---- review failure shapes (standards review D9-1, D9-2, D9-4, D9-6, D9-10) -------------------------
SUN_ROOM = {  # D9-1: a two-word room with a speaker that has no device_class (HomePod/Cast/AirPlay style)
    "areas": [{"id": "sun_room", "name": "Sun Room"}],
    "entities": [{"entity_id": "media_player.homepod", "friendly_name": "HomePod", "area_id": "sun_room",
                  "features": ["play", "pause", "volume_set", "next_track", "previous_track"]}],
    "states": {"media_player.homepod": {"state": "playing",
                                        "attributes": {"volume_level": 0.3, "media_title": "Blue in Green"}}}}
FOCUS_COLLISION = {  # D9-2: a Den scene and an Office script share a name
    "areas": [{"id": "den", "name": "Den"}, {"id": "office", "name": "Office"}],
    "entities": [{"entity_id": "scene.den_focus", "friendly_name": "Focus", "area_id": "den"},
                 {"entity_id": "script.office_focus", "friendly_name": "Focus", "area_id": "office", "exposed": True}]}
AWNING = {  # D9-6: a cover that is not a blind
    "entities": [{"entity_id": "cover.patio_awning", "friendly_name": "Patio Awning", "area_id": "front",
                  "device_class": "awning", "features": ["open", "close", "stop", "set_position"]}],
    "states": {"cover.patio_awning": {"state": "open", "attributes": {"current_position": 40}}}}
DEN_GROUP_NO_MEMBERS = {  # D9-10: a light group whose state has no entity_id attribute
    "areas": [{"id": "den", "name": "Den"}],
    "entities": [{"entity_id": "light.den_ceiling", "friendly_name": "Den Ceiling", "area_id": "den"},
                 {"entity_id": "light.den_bulb", "friendly_name": "Den bulb", "area_id": "den"}]}
GARAGE = {  # D9-4: a Garage room keeps its lights
    "areas": [{"id": "garage", "name": "Garage"}],
    "entities": [{"entity_id": "light.workbench", "friendly_name": "Workbench", "area_id": "garage"}],
    "states": {"light.workbench": {"state": "off"}}}


async def _run(overlay, text, area, answers):
    ha = StubHA.default(overlays=[overlay])
    settings = load_settings().with_scope_extension(**ha.overlay_scope)
    jev = FakeJev(answers)
    core = await Core.create(settings, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle(text, speaker_area=area)
    return r, core, jev, ha


async def test_d9_1_a_speaker_without_device_class_is_a_music_speaker():
    r, core, _, _ = await _run(SUN_ROOM, "turn up the music in the sun room", "living_room",
                               {"intent": "device_control", "scope": "area", "device_type": "media_player",
                                "area__sun_room": 0.95, "media_action": "volume_up"})
    assert "media_player.homepod" in core.catalog.music_speakers
    assert [c.describe() for c in r.service_calls] == ['media_player.volume_set media_player.homepod {"volume_level": 0.4}']


async def test_d9_1_bare_pause_finds_the_playing_speaker():
    r, *_ = await _run(SUN_ROOM, "pause", "living_room", {"intent": "device_control", "scope": "unspecified",
                                                          "device_type": "media_player", "media_action": "media_pause"})
    assert [c.describe() for c in r.service_calls] == ["media_player.media_pause media_player.homepod {}"]


async def test_d9_1_whats_playing_finds_the_playing_speaker():
    r, *_ = await _run(SUN_ROOM, "what's playing", "living_room", {"intent": "device_query", "scope": "unspecified",
                                                                   "device_type": "media_player",
                                                                   "query_attribute": "playing"})
    assert "Blue in Green" in r.reply


async def test_d9_2_a_scene_is_hidden_only_by_a_script_of_the_same_name_in_its_own_room():
    r, core, jev, _ = await _run(FOCUS_COLLISION, "focus in the den", "living_room",
                                 {"intent": "activate_scene", "scene": "scene.den_focus", "area__den": 0.95})
    assert "scene.den_focus" in jev.calls[0][1]["scene"]["criteria"]
    assert "script.office_focus" in jev.calls[0][1]["scene"]["criteria"]
    assert [c.describe() for c in r.service_calls] == ["scene.turn_on scene.den_focus {}"]


async def test_d9_6_an_awning_is_described_and_reported_as_itself():
    r, core, *_ = await _run(AWNING, "is the awning open", "living_room",
                             {"intent": "device_query", "scope": "one_named", "target": "cover.patio_awning",
                              "query_attribute": "position"})
    assert "awning" in core.catalog.entities["cover.patio_awning"].descriptor
    assert "blinds" not in core.catalog.entities["cover.patio_awning"].descriptor
    assert r.reply == "The Patio Awning is 40% open."


async def test_d9_10_a_group_without_a_member_list_is_a_plain_light():
    """HA's convention: no entity_id attribute means HA doesn't say what the group drives, so a room sweep
    addresses it and the bulb (both off is harmless; documented rather than guessed)."""
    r, *_ = await _run(DEN_GROUP_NO_MEMBERS, "turn off the den lights", "living_room",
                       {"intent": "device_control", "scope": "area", "device_type": "light", "area__den": 0.95,
                        "light_action": "turn_off"})
    assert [c.describe() for c in r.service_calls] == ["light.turn_off light.den_bulb,light.den_ceiling {}"]


async def test_d9_4_garage_lights_are_reachable_by_voice():
    r, *_ = await _run(GARAGE, "turn on the garage lights", "living_room",
                       {"intent": "device_control", "scope": "area", "device_type": "light", "area__garage": 0.95,
                        "light_action": "turn_on"})
    assert r.debug.route != "pre_gate/lock"
    assert [c.describe() for c in r.service_calls] == ["light.turn_on light.workbench {}"]


# ---- D9-9: sweep exclusion and alarm sensors are config, not only integration naming -----------------
QUIET = {"entities": [{"entity_id": "switch.quiet_mode", "friendly_name": "Quiet mode", "area_id": "kitchen"},
                      {"entity_id": "sensor.wake_up", "friendly_name": "Wake up", "device_class": "timestamp",
                       "exposed": True}],
         "states": {"switch.quiet_mode": {"state": "off"},
                    "sensor.wake_up": {"state": "2037-09-23T06:30:00-04:00", "attributes": {"device_class": "timestamp"}}}}


async def test_d9_9_a_listed_switch_never_joins_a_sweep():
    ha = StubHA.default(overlays=[QUIET])
    settings = load_settings(overrides={"scope.defaults.sweep_exclude": ["switch.quiet_mode"]}).with_scope_extension(
        **ha.overlay_scope)
    core = await Core.create(settings, ha, FakeJev({"intent": "device_control", "scope": "area", "device_type": "switch",
                                                    "area__kitchen": 0.95, "switch_action": "turn_on"}),
                             NullLuna(), clock=lambda: ha.now)
    r = await core.handle("turn on the kitchen switches", speaker_area="kitchen")
    assert all("switch.quiet_mode" not in c.entity_ids for c in ha.attempted_calls), r.reply


async def test_d9_9_a_listed_alarm_sensor_answers_next_alarm():
    ha = StubHA.default(overlays=[QUIET])
    settings = load_settings(overrides={"scope.defaults.alarm_sensors": ["sensor.wake_up"]}).with_scope_extension(
        **ha.overlay_scope)
    core = await Core.create(settings, ha, FakeJev({"intent": "device_query", "query_attribute": "next_alarm"}),
                             NullLuna(), clock=lambda: ha.now)
    r = await core.handle("when's my next alarm", speaker_area="bedroom")
    assert "6:30 AM" in r.reply
