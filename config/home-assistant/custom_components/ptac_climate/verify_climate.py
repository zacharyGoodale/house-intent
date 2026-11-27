"""Run in the HA container against an isolated HA instance; no hardware calls.

  docker exec homeassistant python /config/custom_components/ptac_climate/verify_climate.py
"""
import asyncio
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from homeassistant.core import HomeAssistant
from homeassistant import loader, config_entries
from homeassistant.bootstrap import async_load_base_functionality
from homeassistant.setup import async_setup_component
from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM, METRIC_SYSTEM
from homeassistant.components.climate import DATA_COMPONENT
from homeassistant.components.alexa import CONFIG_SCHEMA
from homeassistant.components.alexa.entities import ENTITY_ADAPTERS
from homeassistant.core import State
from homeassistant.components.alexa.smart_home import AlexaConfig, async_handle_message

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ptac_adapter", ROOT / "climate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

async def verify():
    with TemporaryDirectory() as config_dir:
        hass = HomeAssistant(config_dir)
        loader.async_setup(hass)
        hass.config_entries = config_entries.ConfigEntries(hass, {})
        assert await async_load_base_functionality(hass)
        hass.config.units = US_CUSTOMARY_SYSTEM
        assert await async_setup_component(hass, "climate", {})
        calls = []
        async def capture(call):
            calls.append((call.domain, call.service, dict(call.data)))
            data = call.data
            if call.domain == "select":
                hass.states.async_set(data["entity_id"], data["option"], {"options": list(module.MODES.values())})
                hass.states.async_set("switch.example_bedroom_ptac_power", "off" if data["option"] == "Off" else "on")
            elif call.domain == "number":
                hass.states.async_set(data["entity_id"], str(data["value"]), {"min": 61, "max": 88, "step": 1, "unit_of_measurement": "°F"})
            elif call.domain == "switch":
                hass.states.async_set(data["entity_id"], "on" if call.service == "turn_on" else "off")
        for domain, service in [("select", "select_option"), ("number", "set_value"), ("switch", "turn_on"), ("switch", "turn_off")]:
            hass.services.async_register(domain, service, capture)
        hass.states.async_set("switch.example_bedroom_ptac_power", "off")
        hass.states.async_set("select.example_bedroom_ptac_operating_mode", "Off", {"options": list(module.MODES.values())})
        hass.states.async_set("number.example_bedroom_ptac_target_temperature", "68", {"min": 61, "max": 88, "step": 1, "unit_of_measurement": "°F"})
        entity = module.PtacClimate("Bedroom thermostat", "example_bedroom_ptac")
        entity.entity_id = "climate.bedroom_demo_thermostat"
        await hass.data[DATA_COMPONENT].async_add_entities([entity])
        cfg = CONFIG_SCHEMA({"alexa": {"smart_home": {"filter": {"include_entities": [entity.entity_id]}}}})["alexa"]["smart_home"]
        config = AlexaConfig(hass, cfg)
        await config.async_initialize()
        async def directive(namespace, name, payload, endpoint=entity.entity_id):
            value = {"directive": {"header": {"namespace": namespace, "name": name, "payloadVersion": "3", "messageId": "test", "correlationToken": "test"}, "payload": payload}}
            if namespace != "Alexa.Discovery":
                value["directive"]["endpoint"] = {"endpointId": endpoint.replace(".", "#"), "scope": {"type": "BearerToken", "token": "test-only"}}
            response = await async_handle_message(hass, config, value)
            await hass.async_block_till_done()
            return response
        discovery = await directive("Alexa.Discovery", "Discover", {})
        endpoints = discovery["event"]["payload"]["endpoints"]
        assert len(endpoints) == 1
        assert "Alexa.PowerController" not in {c["interface"] for c in endpoints[0]["capabilities"]}, "PTAC thermostat must use mode Off, without a separate power toggle"
        thermostat = next(c for c in endpoints[0]["capabilities"] if c["interface"] == "Alexa.ThermostatController")
        assert {"HEAT", "COOL", "OFF"} <= set(thermostat["configuration"]["supportedModes"])
        other = State("climate.unrelated_thermostat", "off", {"hvac_modes": ["off", "heat"], "supported_features": 385})
        other_capabilities = ENTITY_ADAPTERS["climate"](hass, config, other)
        assert "Alexa.PowerController" in {c.name() for c in other_capabilities.interfaces()}
        assert entity.current_temperature is None
        for alexa_mode, option in [("HEAT", "Heat"), ("COOL", "Cool"), ("OFF", "Off")]:
            response = await directive("Alexa.ThermostatController", "SetThermostatMode", {"thermostatMode": alexa_mode})
            assert response["event"]["header"]["name"] == "Response", response
            assert calls[-1] == ("select", "select_option", {"entity_id": entity.mode, "option": option}), calls[-1]
            assert hass.states.get(entity.entity_id).state == alexa_mode.lower()
        await directive("Alexa.ThermostatController", "SetTargetTemperature", {"targetSetpoint": {"value": 70, "scale": "FAHRENHEIT"}})
        assert calls[-1][2]["value"] == 70
        await directive("Alexa.ThermostatController", "AdjustTargetTemperature", {"targetSetpointDelta": {"value": 2, "scale": "FAHRENHEIT"}})
        assert calls[-1][2]["value"] == 72
        count = len(calls)
        response = await directive("Alexa.ThermostatController", "SetTargetTemperature", {"targetSetpoint": {"value": 95, "scale": "FAHRENHEIT"}})
        assert response["event"]["payload"]["type"] == "TEMPERATURE_VALUE_OUT_OF_RANGE"
        response = await directive("Alexa.ThermostatController", "SetThermostatMode", {"thermostatMode": "AUTO"})
        assert response["event"]["header"]["name"] == "ErrorResponse"
        hass.states.async_set("lock.test_lock", "locked")
        response = await directive("Alexa", "ReportState", {}, "lock.test_lock")
        assert response["event"]["payload"]["type"] == "NO_SUCH_ENDPOINT"
        assert len(calls) == count
        await entity.async_turn_on()
        assert calls[-1][:2] == ("switch", "turn_on")
        await entity.async_turn_off()
        assert calls[-1][:2] == ("switch", "turn_off")
        hass.states.async_set(entity.mode, "unavailable")
        await hass.async_block_till_done()
        assert not entity.available
        hass.states.async_set(entity.mode, "Heat")
        hass.states.async_set(entity.power, "on")
        await hass.async_block_till_done()
        assert entity.available and entity.hvac_mode == "heat"
        # HA converts the native F target for metric installations; no double conversion.
        hass.config.units = METRIC_SYSTEM
        entity.async_write_ha_state()
        state = hass.states.get(entity.entity_id)
        assert abs(state.attributes["temperature"] - 22.2) < 0.1, state
        added = []
        await module.async_setup_platform(hass, {"name": "Living Room thermostat", "entity_prefix": "example_living_room_ptac"}, added.extend)
        assert not added
        for entity_id, value in [("switch.example_living_room_ptac_power", "off"), ("select.example_living_room_ptac_operating_mode", "Off"), ("number.example_living_room_ptac_target_temperature", "68")]:
            hass.states.async_set(entity_id, value)
        await hass.async_block_till_done()
        assert len(added) == 1, added
        hass.states.async_set("select.example_living_room_ptac_operating_mode", "Heat")
        await hass.async_block_till_done()
        assert len(added) == 1
        print(json.dumps({"result": "passed", "checks": ["native Alexa discovery heat/cool/off without duplicate power toggle", "unrelated thermostat retains power capability", "native Alexa mode-to-selector routing", "Fahrenheit target and increment", "invalid modes and temperature rejected", "lock excluded", "power delegates to remembered ESPHome mode", "source availability and state synchronization", "metric conversion", "future room appears once sources exist"], "endpoint": endpoints[0]}, indent=2))
        await hass.async_stop(force=True)

asyncio.run(verify())
