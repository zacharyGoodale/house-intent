import pytest

from ha_jev.ha.interface import HAError
from ha_jev.ha.stub import OverlayError, StubHA


async def test_registry_comes_from_the_fixture_plus_sun():
    ha = StubHA.default()
    reg = await ha.get_registry()
    assert {"living_room", "kitchen", "bedroom", "front", "bathroom"} <= set(reg.areas)
    assert reg.entities["light.alder"].area_id == "living_room"
    assert "sun.sun" in reg.entities
    assert reg.entities["media_player.demo_tv"].features >= {"select_source", "turn_on"}


async def test_default_live_state_matches_spec():
    ha = StubHA.default()
    st = await ha.get_states()
    assert st["light.alder"].state == "off"
    assert st["light.bedroom_group"].attributes["entity_id"] == ["light.bedside_globe", "light.bookcase_light", "light.fern_lamp"]
    assert st["automation.demo_morning_scene"].state == "on"
    assert st["cover.bedroom_shade"].attributes["current_position"] == 0
    assert st["media_player.demo_bedroom_speaker"].attributes["volume_level"] == 0.30
    assert st["media_player.demo_tv"].state == "off"
    assert st["lock.entry_lock"].state == "locked"
    assert st["weather.forecast_home"].attributes["humidity"] == 62
    assert ha.now.isoformat() == "2037-09-22T18:00:00-04:00"


async def test_light_effects():
    ha = StubHA.default()
    await ha.call_service("light", "turn_on", ["light.alder"], {"brightness_pct": 20})
    st = (await ha.get_states())["light.alder"]
    assert st.state == "on" and st.attributes["brightness"] == 51
    await ha.call_service("light", "turn_on", ["light.alder"], {"brightness_step_pct": -100})
    assert (await ha.get_states())["light.alder"].attributes["brightness"] == 0
    await ha.call_service("light", "turn_on", ["light.alder"], {"color_temp_kelvin": 3000})
    assert (await ha.get_states())["light.alder"].attributes["color_temp_kelvin"] == 3000
    await ha.call_service("light", "toggle", ["light.alder"], {})
    assert (await ha.get_states())["light.alder"].state == "off"


async def test_turning_on_a_light_defaults_brightness_and_kelvin():
    ha = StubHA.default()
    await ha.call_service("light", "turn_on", ["light.ceiling_light"], {})
    a = (await ha.get_states())["light.ceiling_light"].attributes
    assert a["brightness"] == 255 and a["color_temp_kelvin"] == 4000


async def test_cover_media_todo_effects_and_records():
    ha = StubHA.default()
    await ha.call_service("cover", "set_cover_position", ["cover.bedroom_shade"], {"position": 50})
    await ha.call_service("media_player", "volume_set", ["media_player.demo_bedroom_speaker"], {"volume_level": 0.25})
    await ha.call_service("media_player", "select_source", ["media_player.demo_tv"], {"source": "Netflix"})
    await ha.call_service("todo", "add_item", ["todo.shopping_list"], {"item": "oat milk"})
    await ha.call_service("script", "turn_on", ["script.demo_movie_mode"], {})
    st = await ha.get_states()
    assert st["cover.bedroom_shade"].state == "open"
    assert st["cover.bedroom_shade"].attributes["current_position"] == 50
    assert st["media_player.demo_bedroom_speaker"].attributes["volume_level"] == 0.25
    assert st["media_player.demo_tv"].attributes["source"] == "Netflix"
    items = await ha.call_service("todo", "get_items", ["todo.shopping_list"], {}, return_response=True)
    assert [i["summary"] for i in items["todo.shopping_list"]["items"]] == ["eggs", "coffee beans", "oat milk"]
    assert len(ha.calls) == 5 and ha.calls[-1].domain == "script"
    assert len(ha.reads) == 1


async def test_forecast_read_returns_ha_shape():
    ha = StubHA.default()
    resp = await ha.call_service("weather", "get_forecasts", ["weather.forecast_home"], {"type": "daily"}, return_response=True)
    assert resp["weather.forecast_home"]["forecast"][1]["condition"] == "rainy"
    assert ha.calls == []


async def test_unknown_entity_or_service_raises():
    ha = StubHA.default()
    with pytest.raises(HAError):
        await ha.call_service("light", "turn_on", ["light.nope"], {})
    with pytest.raises(HAError):
        await ha.call_service("light", "explode", ["light.alder"], {})


async def test_fail_on_hook():
    ha = StubHA.default()
    ha.fail_on.add("light.alder")
    with pytest.raises(HAError):
        await ha.call_service("light", "turn_on", ["light.alder"], {})
    assert ha.calls == []
    assert len(ha.attempted_calls) == 1


OFFICE = {
    "areas": [{"id": "office", "name": "Office", "aliases": []}],
    "devices": [{"id": "dev_office_group", "name": "Office", "area_id": "office"}],
    "entities": [
        {"entity_id": "light.desk_lamp", "friendly_name": "Desk Lamp", "area_id": "office",
         "capabilities": {"supported_color_modes": ["color_temp", "xy"]}},
        {"entity_id": "light.office", "friendly_name": "Office", "device_id": "dev_office_group",
         "capabilities": {"supported_color_modes": ["color_temp", "xy"]}},
        {"entity_id": "script.party_mode", "friendly_name": "Party mode", "exposed": False},
    ],
    "states": {"light.office": {"state": "off", "attributes": {"entity_id": ["light.desk_lamp"]}}},
    "remove": ["light.alder"],
    "scope": {"include": ["script.party_mode"], "exclude": []},
}


async def test_overlay_merges_registry_and_state_and_returns_scope():
    ha = StubHA.default(overlays=[OFFICE])
    reg = await ha.get_registry()
    assert reg.areas["office"].name == "Office"
    assert reg.entities["light.office"].area_id == "office"  # from its device
    assert reg.entities["script.party_mode"].exposed is False
    assert reg.entities["light.desk_lamp"].exposed is True
    assert "light.alder" not in reg.entities
    st = await ha.get_states()
    assert st["light.office"].attributes["entity_id"] == ["light.desk_lamp"]
    assert ha.overlay_scope == {"include": ["script.party_mode"], "exclude": []}


def test_overlay_with_unknown_area_is_rejected():
    bad = {"entities": [{"entity_id": "light.x", "friendly_name": "X", "area_id": "attic"}]}
    with pytest.raises(OverlayError):
        StubHA.default(overlays=[bad])


async def test_initial_state_merges_over_stub_and_sets_todo_items():
    ha = StubHA.default(initial_state={
        "light.alder": {"state": "on", "attributes": {"brightness": 128}},
        "todo.shopping_list": {"items": ["milk"]},
    })
    st = await ha.get_states()
    assert st["light.alder"].state == "on"
    assert st["light.alder"].attributes["brightness"] == 128
    assert st["light.alder"].attributes["friendly_name"] == "Alder"
    items = await ha.call_service("todo", "get_items", ["todo.shopping_list"], {}, return_response=True)
    assert [i["summary"] for i in items["todo.shopping_list"]["items"]] == ["milk"]
