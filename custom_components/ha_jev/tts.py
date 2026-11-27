"""Nonstreaming Mac mini speech; HA handles media URLs and resampling."""
from homeassistant.components.tts import TextToSpeechEntity
from homeassistant.exceptions import HomeAssistantError
from .client import ServiceError


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([JevTTS(entry)])


class JevTTS(TextToSpeechEntity):
    _attr_default_language = "en"
    _attr_supported_languages = ["en"]
    _attr_supported_options = []

    def __init__(self, entry):
        self.client = entry.runtime_data["speech"]
        self._attr_name = "Mac mini TTS"
        self._attr_unique_id = entry.entry_id + "-tts"

    async def async_get_tts_audio(self, message, language, options):
        try:
            if language != "en" or options:
                raise ServiceError("Unsupported language or voice option")
            return "wav", await self.client.speak(message)
        except ServiceError as err:
            raise HomeAssistantError(str(err)) from err
