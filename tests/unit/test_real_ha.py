"""RealHA against the fake HA WebSocket server (never the real house)."""

import asyncio
import logging

import pytest

from ha_jev.ha.interface import HAError
from ha_jev.ha.real import RealHA
from tests.fakes import FakeHAServer

REGISTRY = {
    "areas": [{"area_id": "den", "name": "Den", "aliases": ["snug"]}, {"area_id": "hall", "name": "Hall", "aliases": []}],
    "devices": [{"id": "d1", "name": "Lamp device", "name_by_user": None, "area_id": "den", "manufacturer": "Signify",
                 "model": "Bulb", "disabled_by": None}],
    "entities": [
        {"entity_id": "light.lamp", "area_id": None, "device_id": "d1", "name": None, "original_name": "Lamp",
         "platform": "hue", "disabled_by": None, "hidden_by": None, "entity_category": None},
        {"entity_id": "light.override", "area_id": "hall", "device_id": "d1", "name": "Override", "original_name": None,
         "platform": "hue", "disabled_by": None, "hidden_by": None, "entity_category": None},
        {"entity_id": "switch.diag", "area_id": None, "device_id": "d1", "name": None, "original_name": "Diag",
         "platform": "x", "disabled_by": None, "hidden_by": None, "entity_category": "diagnostic"},
        {"entity_id": "media_player.tv", "area_id": "den", "device_id": None, "name": None, "original_name": "TV",
         "platform": "apple_tv", "disabled_by": None, "hidden_by": "user", "entity_category": None},
    ],
}
STATES = [
    {"entity_id": "light.lamp", "state": "off", "attributes": {"friendly_name": "Den Lamp", "supported_features": 44,
                                                               "supported_color_modes": ["color_temp", "xy"],
                                                               "min_color_temp_kelvin": 2000}},
    {"entity_id": "light.override", "state": "on", "attributes": {"friendly_name": "Override"}},
    {"entity_id": "media_player.tv", "state": "off", "attributes": {"friendly_name": "TV", "supported_features": 2048 | 128,
                                                                    "source_list": ["Netflix"]}},
]
EXPOSED = {"light.lamp": {"conversation": True}, "light.override": {"conversation": False}}


@pytest.fixture
async def server():
    s = FakeHAServer(registry=REGISTRY, states=STATES, exposed=EXPOSED)
    url = await s.start()
    s.url = url
    yield s
    await s.stop()


async def test_auth_and_registry_mapping(server):
    ha = RealHA(server.url, "good-token")
    await ha.connect()
    try:
        reg = await ha.get_registry()
        assert reg.areas["den"].aliases == ("snug",)
        lamp = reg.entities["light.lamp"]
        assert lamp.area_id == "den"  # from its device
        assert lamp.name == "Den Lamp"
        assert lamp.exposed is True
        assert lamp.features == {"effect", "flash", "transition"}
        assert lamp.capabilities["supported_color_modes"] == ["color_temp", "xy"]
        assert reg.entities["light.override"].area_id == "hall"  # entity area overrides device area
        assert reg.entities["light.override"].exposed is False
        assert "switch.diag" not in reg.entities  # entity_category skipped
        assert reg.entities["media_player.tv"].hidden is True
        assert {"select_source", "turn_on"} <= reg.entities["media_player.tv"].features
        states = await ha.get_states()
        assert states["light.override"].state == "on"
        types = [m["type"] for m in server.received]
        assert types[0] == "auth"
        assert {"config/area_registry/list", "config/device_registry/list", "config/entity_registry/list",
                "homeassistant/expose_entity/list", "get_states", "subscribe_events"} <= set(types)
        ids = [m["id"] for m in server.received if "id" in m]
        assert ids == sorted(ids) and len(ids) == len(set(ids))
    finally:
        await ha.close()


async def test_bad_token_is_rejected(server):
    ha = RealHA(server.url, "wrong")
    with pytest.raises(HAError):
        await ha.connect()


async def test_call_service_shape_and_response(server):
    ha = RealHA(server.url, "good-token")
    await ha.connect()
    try:
        await ha.call_service("light", "turn_on", ["light.lamp"], {"brightness_pct": 20})
        msg = [m for m in server.received if m["type"] == "call_service"][-1]
        assert msg["domain"] == "light" and msg["service"] == "turn_on"
        assert msg["service_data"] == {"brightness_pct": 20}
        assert msg["target"] == {"entity_id": ["light.lamp"]}
        assert msg["return_response"] is False
        resp = await ha.call_service("weather", "get_forecasts", ["weather.x"], {"type": "daily"}, return_response=True)
        assert resp == {"weather.x": {"echo": {"type": "daily"}}}
        with pytest.raises(HAError):
            await ha.call_service("fail", "x", ["fail.y"], {})
    finally:
        await ha.close()


async def test_call_service_timeout():
    s = FakeHAServer(registry=REGISTRY, states=STATES, exposed=EXPOSED, service_delay=0.5)
    url = await s.start()
    ha = RealHA(url, "good-token", timeout=0.1)
    await ha.connect()
    try:
        with pytest.raises(HAError):
            await ha.call_service("light", "turn_on", ["light.lamp"], {})
    finally:
        await ha.close()
        await s.stop()


async def test_state_changed_events_update_the_cache_and_registry_events_notify(server):
    changed = asyncio.Event()
    ha = RealHA(server.url, "good-token", on_registry_change=lambda: changed.set())
    await ha.connect()
    try:
        await server.push_event("state_changed", {"entity_id": "light.lamp",
                                                  "new_state": {"entity_id": "light.lamp", "state": "on", "attributes": {}}})
        for _ in range(50):
            if (await ha.get_states())["light.lamp"].state == "on":
                break
            await asyncio.sleep(0.01)
        assert (await ha.get_states())["light.lamp"].state == "on"
        await server.push_event("entity_registry_updated", {"action": "create", "entity_id": "light.new"})
        await asyncio.wait_for(changed.wait(), 1.0)
    finally:
        await ha.close()


async def test_entity_id_map_translates_config_ids_to_live_ids(server):
    ha = RealHA(server.url, "good-token", entity_id_map={"light.config_lamp": "light.lamp"})
    await ha.connect()
    try:
        reg = await ha.get_registry()
        assert "light.config_lamp" in reg.entities and "light.lamp" not in reg.entities
        assert "light.config_lamp" in await ha.get_states()
        await ha.call_service("light", "turn_off", ["light.config_lamp"], {})
        msg = [m for m in server.received if m["type"] == "call_service"][-1]
        assert msg["target"] == {"entity_id": ["light.lamp"]}
    finally:
        await ha.close()


async def test_token_never_appears_in_logs(server, caplog):
    caplog.set_level(logging.DEBUG, logger="ha_jev")
    ha = RealHA(server.url, "good-token")
    await ha.connect()
    await ha.call_service("light", "turn_on", ["light.lamp"], {})
    await ha.close()
    assert caplog.records  # debug logging happened
    assert "good-token" not in caplog.text


async def test_script_and_automation_definitions_are_fetched_for_the_decision_6_check():
    registry = {"areas": [], "devices": [], "entities": [
        {"entity_id": "script.welcome_home", "area_id": None, "device_id": None, "name": None,
         "original_name": "Welcome home", "platform": "script", "disabled_by": None, "hidden_by": None,
         "entity_category": None},
        {"entity_id": "script.mystery", "area_id": None, "device_id": None, "name": None, "original_name": "Mystery",
         "platform": "script", "disabled_by": None, "hidden_by": None, "entity_category": None},
        {"entity_id": "automation.arrival", "area_id": None, "device_id": None, "name": None,
         "original_name": "Arrival", "platform": "automation", "disabled_by": None, "hidden_by": None,
         "entity_category": None},
    ]}
    configs = {"script.welcome_home": {"sequence": [{"action": "lock.unlock", "target": {"entity_id": "lock.entry_lock"}}]},
               "automation.arrival": {"actions": [{"action": "light.turn_on", "entity_id": "light.hall"}]}}
    s = FakeHAServer(registry=registry, states=[], exposed={}, configs=configs)
    url = await s.start()
    ha = RealHA(url, "good-token")
    try:
        await ha.connect()
        reg = await ha.get_registry()
        assert reg.entities["script.welcome_home"].config == configs["script.welcome_home"]
        assert reg.entities["automation.arrival"].config == configs["automation.arrival"]
        assert reg.entities["script.mystery"].config is None  # unknown: fetch failed, logged, not fatal
        asked = {(m["type"], m.get("entity_id")) for m in s.received}
        assert ("script/config", "script.welcome_home") in asked
    finally:
        await ha.close()
        await s.stop()


async def test_s8_core_rebuilds_its_catalog_when_the_registry_changes(server):
    from ha_jev.config import load_settings
    from ha_jev.core import Core
    from ha_jev.luna import NullLuna
    from tests.fakes import FakeJev
    ha = RealHA(server.url, "good-token")
    await ha.connect()
    try:
        core = await Core.create(load_settings(), ha, FakeJev({}), NullLuna())
        assert "light.new" not in core.catalog.entities
        server.registry["entities"].append({"entity_id": "light.new", "area_id": "den", "device_id": None, "name": "New",
                                            "original_name": None, "platform": "hue", "disabled_by": None,
                                            "hidden_by": None, "entity_category": None})
        server.exposed["light.new"] = {"conversation": True}
        await server.push_event("entity_registry_updated", {"action": "create", "entity_id": "light.new"})
        for _ in range(100):
            if "light.new" in core.catalog.entities:
                break
            await asyncio.sleep(0.01)
        assert "light.new" in core.catalog.entities
    finally:
        await ha.close()


async def test_s8_a_failed_reload_does_not_rebuild_from_stale_data(server):
    calls = []
    ha = RealHA(server.url, "good-token", on_registry_change=lambda: calls.append(1))
    await ha.connect()
    try:
        server.fail_types = {"config/entity_registry/list"}
        await server.push_event("entity_registry_updated", {"action": "create", "entity_id": "light.x"})
        await asyncio.sleep(0.2)
        assert calls == []
    finally:
        await ha.close()


async def test_st6_an_expose_list_failure_is_fatal_not_an_empty_scope(server):
    server.fail_types = {"homeassistant/expose_entity/list"}
    ha = RealHA(server.url, "good-token")
    with pytest.raises(HAError):
        await ha.connect()
    await ha.close()


async def test_response_keys_are_translated_back_to_config_ids(server):
    ha = RealHA(server.url, "good-token", entity_id_map={"weather.home": "weather.home_live_2"})
    await ha.connect()
    try:
        resp = await ha.call_service("weather", "get_forecasts", ["weather.home"], {"type": "daily"},
                                     return_response=True)
        sent = [m for m in server.received if m["type"] == "call_service"][-1]
        assert sent["target"] == {"entity_id": ["weather.home_live_2"]}
        assert resp == {"weather.home": {"echo": {"type": "daily"}}}
    finally:
        await ha.close()
