from ha_jev.catalog import Catalog
from ha_jev.config import load_settings
from ha_jev.ha.stub import StubHA
from ha_jev.resolve import area_light_targets, house_light_targets


async def build(overlays=(), overrides=None, initial_state=None):
    ha = StubHA.default(overlays=overlays, initial_state=initial_state)
    settings = load_settings(overrides=overrides).with_scope_extension(**ha.overlay_scope)
    return Catalog.build(await ha.get_registry(), await ha.get_states(), settings.scope, settings.safety)


async def test_scope_includes_exposed_plus_include_minus_exclude_hidden_disabled():
    cat = await build()
    assert "light.alder" in cat.entities  # exposed
    assert "script.demo_movie_mode" in cat.entities  # include
    assert "media_player.demo_soundbar_duplicate" not in cat.entities  # exclude
    assert "scene.demo_bedroom_imported_scene_2" not in cat.entities  # exclude glob
    assert "switch.corner_lamp_plug" not in cat.entities  # hidden
    assert "light.ring_lamp_duplicate" not in cat.entities  # not exposed, not included
    assert "script.toggle_entry_lock" not in cat.entities
    assert "sun.sun" in cat.entities


async def test_denied_entity_is_queryable_but_not_controllable():
    cat = await build()
    assert "lock.entry_lock" in cat.entities
    assert cat.entities["lock.entry_lock"].denied
    assert "lock.entry_lock" not in cat.controllable
    assert "lock.entry_lock" in cat.target_options


async def test_denylisted_script_in_scope_is_still_not_controllable():
    cat = await build(overrides={"scope.include": ["script.*"]})
    assert "script.toggle_entry_lock" in cat.entities
    assert "script.toggle_entry_lock" not in cat.controllable
    assert cat.entities["script.toggle_entry_lock"].denied


async def test_alias_globs_match_both_blind_ids():
    alternate_blind = {"entities": [{"entity_id": "cover.bedroom_shade_demo_02", "friendly_name": "Bedroom Shade",
                                "area_id": "bedroom", "features": ["open", "close", "set_position", "stop"],
                                "device_class": "shade"}],
                  "remove": ["cover.bedroom_shade"]}
    for overlays in ((), (alternate_blind,)):
        cat = await build(overlays=overlays)
        blinds = [e for e in cat.entities.values() if e.domain == "cover"]
        assert len(blinds) == 1
        assert "blinds" in blinds[0].aliases and "shades" in blinds[0].aliases


LIVING_MEMBERS = [
    "light.alder", "light.backlight_left", "light.backlight_right", "light.ceiling_light",
    "light.corner_lamp", "light.paper_lantern", "light.reading_lamp", "light.ribbon_lamp",
    "light.ring_lamp",
]


async def test_a_group_with_members_outside_the_room_is_replaced_by_its_in_room_members():
    """Review S3: the Common Area group also drives the kitchen Pendant, so "living room lights" must not use it."""
    cat = await build()
    assert area_light_targets(cat, ["living_room"]) == LIVING_MEMBERS
    assert area_light_targets(cat, ["living_room", "kitchen"]) == [
        "light.common_lights", "light.corner_lamp", "light.kitchen_amber_lamp",
        "light.reading_lamp", "light.ring_lamp"]


async def test_the_dashboard_group_rule_is_one_config_switch_away():
    cat = await build()
    assert area_light_targets(cat, ["living_room"], strict=False) == [
        "light.common_lights", "light.corner_lamp", "light.reading_lamp", "light.ring_lamp"]


async def test_group_rule_gives_the_spec_area_sets():
    cat = await build()
    assert area_light_targets(cat, ["bedroom"]) == ["light.bedroom_group"]
    assert area_light_targets(cat, ["bathroom"]) == ["light.bathroom_group"]
    assert area_light_targets(cat, ["kitchen"]) == [
        "light.kitchen_amber_lamp", "light.pendant_lower", "light.pendant_upper"]
    house = house_light_targets(cat)
    assert "light.mirror_strip" not in house  # covered by the bathroom group
    assert "light.alder" not in house and "light.common_lights" in house


async def test_scene_dedupe_and_script_over_scene():
    cat = await build()
    assert "scene.demo_common_default" in cat.scene_options
    assert "scene.demo_common_default_2" not in cat.scene_options
    assert "scene.demo_living_wind_down" not in cat.scene_options
    assert "script.demo_living_wind_down" in cat.scene_options
    assert "preset.kitchen_day" in cat.scene_options
    assert len(cat.scene_options) == 37


async def test_music_speakers_and_sources():
    cat = await build()
    assert cat.music_speakers == ["media_player.demo_bedroom_speaker", "media_player.demo_soundbar", "media_player.demo_stereo"]
    assert "App Store" in cat.sources and "Netflix" in cat.sources


async def test_descriptors_are_built_from_registry_and_config():
    cat = await build()
    assert cat.descriptor("light.corner_lamp") == "corner lamp — light, Living Room; on/off only"
    assert cat.descriptor("media_player.demo_tv") == "Demo TV — TV, Living Room; also: TV, Apple TV"
    assert cat.descriptor("light.bedroom_group") == (
        "Bedroom — light, Bedroom; group of: Globe Lamp, Bookcase Lamp, Fern Light")
    assert cat.descriptor("script.demo_movie_mode") == "Living: Film — script, no room; also: movie mode, film mode"
    assert cat.descriptor("preset.kitchen_day").startswith("Kitchen daytime — preset, Kitchen")


async def test_area_aliases_come_from_scope():
    cat = await build()
    assert cat.areas["living_room"].aliases == ("lounge", "downstairs")


async def test_target_and_mention_options():
    cat = await build()
    assert "light.alder" in cat.target_options
    assert "binary_sensor.entry_contact" in cat.target_options
    assert "scene.demo_common_relax" not in cat.target_options
    assert "weather.forecast_home" not in cat.target_options
    assert "light.alder" in cat.mention_options
    assert "lock.entry_lock" not in cat.mention_options
    assert 35 <= len(cat.target_options) <= 45


async def test_new_room_group_and_scene_from_registry_only():
    office = {"areas": [{"id": "office", "name": "Office", "aliases": []}],
              "entities": [{"entity_id": "light.desk_lamp", "friendly_name": "Desk Lamp", "area_id": "office",
                            "capabilities": {"supported_color_modes": ["color_temp"]}},
                           {"entity_id": "light.office", "friendly_name": "Office", "area_id": "office",
                            "capabilities": {"supported_color_modes": ["color_temp"]}},
                           {"entity_id": "scene.office_focus", "friendly_name": "Office Focus", "area_id": "office"}],
              "states": {"light.office": {"state": "off", "attributes": {"entity_id": ["light.desk_lamp"]}}}}
    cat = await build(overlays=[office])
    assert "office" in cat.areas
    assert area_light_targets(cat, ["office"]) == ["light.office"]
    assert "scene.office_focus" in cat.scene_options
