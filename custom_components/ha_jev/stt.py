"""Assist raw PCM to Mac mini WAV HTTP transcription."""
from homeassistant.components import stt
from .client import ServiceError


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([JevSTT(entry)])


class JevSTT(stt.SpeechToTextEntity):
    def __init__(self, entry):
        self.client = entry.runtime_data["speech"]
        self._attr_name = "Mac mini Parakeet"
        self._attr_unique_id = entry.entry_id + "-stt"

    @property
    def supported_languages(self):
        return ["en"]

    @property
    def supported_formats(self):
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self):
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self):
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self):
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self):
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(self, metadata, stream):
        try:
            if not self.check_metadata(metadata):
                raise ServiceError("Unsupported audio metadata")
            parts = []
            size = 0
            async for chunk in stream:
                size += len(chunk)
                if size > 60 * 16000 * 2:
                    raise ServiceError("Utterance exceeds 60 seconds")
                parts.append(chunk)
            text = await self.client.transcribe(b"".join(parts))
            return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)
        except ServiceError:
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
