"""Configure TLS and separate core/speech credentials."""
import voluptuous as vol
from homeassistant.config_entries import ConfigFlow
from homeassistant.helpers import selector
from . import DOMAIN
from .client import tls_context

FIELDS = ("core_url", "core_key", "core_ca_file", "speech_url", "speech_key", "speech_ca_file")
SCHEMA = vol.Schema({
    vol.Required("core_url", description={"suggested_value": "https://bridge.example.test:8444"}): str,
    vol.Required("core_key"): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),
    vol.Optional("core_ca_file", default=""): str,
    vol.Required("speech_url", description={"suggested_value": "https://speech.example.test:8443"}): str,
    vol.Required("speech_key"): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),
    vol.Optional("speech_ca_file", default=""): str,
})


class JevConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            try:
                for prefix in ("core", "speech"):
                    await self.hass.async_add_executor_job(tls_context, user_input[prefix + "_url"], user_input.get(prefix + "_ca_file", ""))
                    if len(user_input[prefix + "_key"]) < 32:
                        raise ValueError("Key too short")
            except (ValueError, OSError):
                errors["base"] = "invalid_config"
            else:
                return self.async_create_entry(title="Jev Voice", data=user_input)
        return self.async_show_form(step_id="user", data_schema=SCHEMA, errors=errors)
