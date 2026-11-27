"""PTAC adapter: delegate commands; never run a thermostat control loop."""

import math

import voluptuous as vol

from homeassistant.components.alexa.entities import ClimateCapabilities, ENTITY_ADAPTERS
from homeassistant.components.climate import (
    PLATFORM_SCHEMA as CLIMATE_SCHEMA,
    ClimateEntity,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.const import STATE_OFF, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_state_change_event

PLATFORM_SCHEMA = CLIMATE_SCHEMA.extend({
    vol.Required("name"): cv.string,
    vol.Required("entity_prefix"): cv.slug,
})

MODES = {
    HVACMode.OFF: "Off",
    HVACMode.COOL: "Cool",
    HVACMode.HEAT: "Heat",
    HVACMode.FAN_ONLY: "Fan only",
}


_MODE_ONLY_ENTITIES = set()


class PtacAlexaCapabilities(ClimateCapabilities):
    """Keep Heat/Cool/Off as the sole on/off control for PTAC thermostats."""

    def interfaces(self):
        for capability in super().interfaces():
            if (self.entity.entity_id in _MODE_ONLY_ENTITIES
                    and capability.name() == "Alexa.PowerController"):
                continue
            yield capability


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Wait for real source entities, including a PTAC installed in the future."""
    entity = PtacClimate(config["name"], config["entity_prefix"])
    remove_listener = None
    added = False

    @callback
    def add_when_present(event=None):
        nonlocal remove_listener, added
        if added:
            return True
        if all(hass.states.get(entity_id) is not None for entity_id in entity.sources):
            if remove_listener is not None:
                remove_listener()
            added = True
            async_add_entities([entity])
            return True
        return False

    if not add_when_present():
        remove_listener = async_track_state_change_event(hass, entity.sources, add_when_present)


class PtacClimate(ClimateEntity):
    """Last-commanded PTAC state, with native Fahrenheit temperatures."""

    _attr_should_poll = False
    _attr_temperature_unit = UnitOfTemperature.FAHRENHEIT
    _attr_min_temp = 61
    _attr_max_temp = 88
    _attr_hvac_modes = list(MODES)
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.TURN_ON
        | ClimateEntityFeature.TURN_OFF
    )

    def __init__(self, name, prefix):
        self._attr_name = name
        self._attr_unique_id = f"{prefix}_climate"
        self.power = f"switch.{prefix}_power"
        self.mode = f"select.{prefix}_operating_mode"
        self.temperature = f"number.{prefix}_target_temperature"
        self.sources = [self.power, self.mode, self.temperature]

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        # HA otherwise advertises a second power toggle whenever Off is supported.
        # Scope the adapter to entities created here; other climates stay unchanged.
        _MODE_ONLY_ENTITIES.add(self.entity_id)
        ENTITY_ADAPTERS["climate"] = PtacAlexaCapabilities
        self.async_on_remove(lambda: _MODE_ONLY_ENTITIES.discard(self.entity_id))
        self.async_on_remove(async_track_state_change_event(self.hass, self.sources, self._changed))

    @callback
    def _changed(self, event):
        self.async_write_ha_state()

    @property
    def available(self):
        return all(
            (state := self.hass.states.get(entity_id)) is not None
            and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)
            for entity_id in self.sources
        ) and self.hvac_mode is not None and self.target_temperature is not None

    @property
    def precision(self):
        return 1 if self.hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT else 0.1

    @property
    def target_temperature_step(self):
        return 1 if self.hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT else 0.5

    @property
    def extra_state_attributes(self):
        # Climate state values use HA display units, which voice clients must know.
        return {"temperature_unit": self.hass.config.units.temperature_unit}

    @property
    def hvac_mode(self):
        power = self.hass.states.get(self.power)
        if power is not None and power.state == STATE_OFF:
            return HVACMode.OFF
        state = self.hass.states.get(self.mode)
        return next((mode for mode, option in MODES.items() if state and state.state == option), None)

    @property
    def target_temperature(self):
        state = self.hass.states.get(self.temperature)
        try:
            value = float(state.state) if state else None
            return value if value is not None and math.isfinite(value) else None
        except ValueError:
            return None

    async def async_set_hvac_mode(self, hvac_mode):
        if hvac_mode not in MODES:
            raise ServiceValidationError("PTAC mode must be off, cool, heat, or fan_only")
        await self.hass.services.async_call(
            "select", "select_option", {"entity_id": self.mode, "option": MODES[hvac_mode]},
            blocking=True, context=self._context,
        )

    async def async_set_temperature(self, **kwargs):
        value = kwargs.get("temperature")
        if value is not None:
            if not math.isfinite(value) or not self.min_temp <= value <= self.max_temp:
                raise ServiceValidationError("PTAC target must be between 61 and 88 Fahrenheit")
            await self.hass.services.async_call(
                "number", "set_value", {"entity_id": self.temperature, "value": round(value)},
                blocking=True, context=self._context,
            )
        if "hvac_mode" in kwargs:
            await self.async_set_hvac_mode(kwargs["hvac_mode"])

    async def async_turn_on(self):
        # ESPHome remembers the last non-off mode; do not force cooling here.
        await self.hass.services.async_call(
            "switch", "turn_on", {"entity_id": self.power}, blocking=True, context=self._context,
        )

    async def async_turn_off(self):
        await self.hass.services.async_call(
            "switch", "turn_off", {"entity_id": self.power}, blocking=True, context=self._context,
        )
