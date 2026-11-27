"""The router is not the demo home's: a different house, with no config files at all, works from HA alone.

The second home has two floors, three bedrooms, an office and a thermostat, and no covers or speakers. Everything
it knows comes from its registry: areas, HA floors and Assist exposure."""

import json

import pytest

from ha_jev import config
from ha_jev.catalog import Catalog
from ha_jev.config import ConfigError, load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.luna import NullLuna
from tests.fakes import FakeJev

UP, DOWN = "upstairs", "ground"
ROOMS = {"primary_bedroom": ("Primary Bedroom", UP), "kids_bedroom": ("Kids Bedroom", UP),
         "guest_bedroom": ("Guest Bedroom", UP), "office": ("Office", DOWN), "kitchen": ("Kitchen", DOWN)}
LIGHT_CAPS = {"supported_color_modes": ["brightness"]}


def _snapshot() -> dict:
    lights = [{"entity_id": f"light.{rid}", "friendly_name": f"{name} Light", "area_id": rid, "exposed": True,
               "capabilities": LIGHT_CAPS, "state": "on", "attributes": {"supported_color_modes": ["brightness"]}}
              for rid, (name, _) in ROOMS.items()]
    thermostat = {"entity_id": "climate.hallway", "friendly_name": "Hallway Thermostat", "area_id": "office",
                  "exposed": True, "capabilities": {"hvac_modes": ["off", "heat"], "min_temp": 7, "max_temp": 30},
                  "state": "heat", "attributes": {"temperature": 20, "temperature_unit": "°C", "min_temp": 7,
                                                  "max_temp": 30, "hvac_modes": ["off", "heat"],
                                                  "supported_features": 1}}  # target temperature only
    return {
        "floors": [{"floor_id": UP, "name": "Upstairs", "aliases": ["second floor"]},
                   {"floor_id": DOWN, "name": "Ground Floor", "aliases": ["downstairs"]}],
        "areas": [{"id": rid, "name": name, "aliases": [], "floor_id": floor} for rid, (name, floor) in ROOMS.items()],
        "devices": [],
        "entities": [*lights, thermostat],
    }


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A StubHA of the second home and its settings, from no config directory at all."""
    (tmp_path / "snapshot.json").write_text(json.dumps(_snapshot()))
    (tmp_path / "state.json").write_text(json.dumps({"clock": "2026-01-05T19:00:00+00:00", "states": {}}))
    monkeypatch.delenv("HA_JEV_CONFIG_DIR", raising=False)
    monkeypatch.setattr(config, "USER_CONFIG_DIR", tmp_path / "nowhere")
    ha = StubHA.default(fixture_path=tmp_path / "snapshot.json", stub_state_path=tmp_path / "state.json")
    return ha, load_settings()


def test_no_config_directory_runs_on_safe_defaults(home):
    _, settings = home
    assert settings.config_dir is None
    assert settings.people == {} and settings.scope.include_exposed
    assert {"lock", "alarm_control_panel"} <= set(settings.safety.denied_domains)
    assert "garage" in settings.safety.denied_cover_device_classes
    assert "climate" in settings.capabilities  # thermostats are built in, not demo config


def test_a_named_config_directory_that_is_missing_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_JEV_CONFIG_DIR", str(tmp_path / "typo"))
    with pytest.raises(ConfigError, match="not found"):
        load_settings()


def test_a_config_directory_may_hold_only_the_files_it_changes(tmp_path, monkeypatch):
    (tmp_path / "people.toml").write_text('[people.sam]\nname = "Sam"\n')
    monkeypatch.setenv("HA_JEV_CONFIG_DIR", str(tmp_path))
    s = load_settings()
    assert list(s.people) == ["sam"] and "lock" in s.safety.denied_domains


async def test_floors_name_every_room_on_them_and_absent_kinds_are_not_asked(home):
    ha, s = home
    cat = Catalog.build(await ha.get_registry(), await ha.get_states(), s.scope, s.safety, s.capabilities)
    assert "Upstairs" in cat.areas["kids_bedroom"].aliases and "second floor" in cat.areas["guest_bedroom"].aliases
    assert "downstairs" in cat.areas["office"].aliases and "Upstairs" not in cat.areas["office"].aliases
    asked = cat.caps.action_questions()
    assert "light_action" in asked and "temperature_action" in asked
    assert "cover_action" not in asked and "media_action" not in asked  # this home has no blinds or speakers


async def test_upstairs_lights_reach_all_three_bedrooms_and_nothing_else(home):
    ha, s = home
    jev = FakeJev({"intent": "device_control", "scope": "area", "device_type": "light", "light_action": "turn_off",
                   **{f"area__{rid}": 0.95 for rid, (_, floor) in ROOMS.items() if floor == UP}})
    core = await Core.create(s, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle("turn off the upstairs lights", speaker_area="kitchen")
    targets = sorted(e for c in r.service_calls for e in c.entity_ids)
    assert targets == ["light.guest_bedroom", "light.kids_bedroom", "light.primary_bedroom"]
    assert "Upstairs" in json.dumps(jev.calls[0][0])  # Jev sees the floor name with each upstairs room


async def test_one_bedroom_of_several_is_just_that_room(home):
    ha, s = home
    jev = FakeJev({"intent": "device_control", "scope": "area", "device_type": "light", "light_action": "turn_off",
                   "area__kids_bedroom": 0.95})
    core = await Core.create(s, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle("turn off the lights in the kids bedroom", speaker_area="primary_bedroom")
    assert [c.describe() for c in r.service_calls] == ["light.turn_off light.kids_bedroom {}"]


async def test_thermostat_without_turn_on_support_is_not_sent_turn_on(home):
    ha, s = home
    jev = FakeJev({"intent": "device_control", "scope": "one_named", "target": "climate.hallway",
                   "device_type": "thermostat", "temperature_action": "turn_on"})
    core = await Core.create(s, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle("turn on the hallway thermostat", speaker_area="office")
    assert r.service_calls == [] and ha.calls == []
    jev = FakeJev({"intent": "device_control", "scope": "one_named", "target": "climate.hallway",
                   "device_type": "thermostat", "temperature_action": "warmer"})
    core = await Core.create(s, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle("make the hallway thermostat warmer", speaker_area="office")
    assert [c.describe() for c in r.service_calls] == ['climate.set_temperature climate.hallway {"temperature": 21}']


async def test_a_home_without_handoffs_has_no_reservation_intent_or_keyword_route(home):
    ha, s = home
    jev = FakeJev({"intent": "general"})
    core = await Core.create(s, ha, jev, NullLuna(), clock=lambda: ha.now)
    r = await core.handle("book a table at a restaurant", speaker_area="kitchen")
    assert "reservations" not in jev.calls[0][1]["intent"]["criteria"]
    assert not r.debug.route.startswith("hermes/")


def test_a_handoff_needs_a_new_intent_name(tmp_path, monkeypatch):
    (tmp_path / "ha_jev.toml").write_text('[[handoffs]]\nid = "weather"\nwhat = "x"\n')
    monkeypatch.setenv("HA_JEV_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError, match="new lower_snake_case intent"):
        load_settings()
