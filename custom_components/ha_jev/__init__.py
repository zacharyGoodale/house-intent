"""HA platform adapters for the remote Jev core and Mac mini speech service."""
from homeassistant.const import Platform
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from .client import VoiceClient, ServiceError

DOMAIN = "ha_jev"
PLATFORMS = [Platform.CONVERSATION, Platform.STT, Platform.TTS]


async def async_setup(hass, config):
    from .schedules import async_setup as setup_schedules
    await setup_schedules(hass)
    return True


async def async_setup_entry(hass, entry):
    session = async_get_clientsession(hass)
    data = entry.data
    try:
        core = await hass.async_add_executor_job(VoiceClient, session, data["core_url"], data["core_key"], data.get("core_ca_file", ""))
        speech = await hass.async_add_executor_job(VoiceClient, session, data["speech_url"], data["speech_key"], data.get("speech_ca_file", ""))
        await core.request("/v1/models")
        models = await speech.request("/v1/models")
        if not {"stt", "tts"}.issubset({m["id"] for m in models["data"]}):
            raise ValueError("Missing speech models")
    except (ServiceError, ValueError, OSError, KeyError, TypeError) as err:
        raise ConfigEntryNotReady("Core or speech service unavailable; check TLS, keys and URLs") from err
    entry.runtime_data = {"core": core, "speech": speech}
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass, entry):
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
