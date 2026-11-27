"""HA platform behavior with minimal interfaces matching pinned 2026.8.3 source."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2] / "custom_components/ha_jev"


@pytest.fixture
def modules(monkeypatch):
    def module(name, **values):
        item = ModuleType(name)
        item.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, item)
        return item
    class Entity:
        pass
    class ConversationEntity(Entity):
        async def async_added_to_hass(self): pass
        async def async_will_remove_from_hass(self): pass
    class IntentResponse:
        def __init__(self, language): self.language = language
        def async_set_error(self, code, message): self.error = (code, message)
        def async_set_speech(self, speech, extra_data=None): self.speech, self.extra_data = speech, extra_data
    module("homeassistant")
    module("homeassistant.components")
    module("homeassistant.helpers")
    module("homeassistant.components.conversation", ConversationEntity=ConversationEntity,
           ConversationEntityFeature=SimpleNamespace(CONTROL=1),
           ConversationResult=lambda response, cid, more=False: SimpleNamespace(response=response, conversation_id=cid, continue_conversation=more))
    enums = {name: SimpleNamespace(**{value: value for value in values}) for name, values in {
        "AudioFormats": ["WAV"], "AudioCodecs": ["PCM"], "AudioBitRates": ["BITRATE_16"],
        "AudioSampleRates": ["SAMPLERATE_16000"], "AudioChannels": ["CHANNEL_MONO"],
        "SpeechResultState": ["SUCCESS", "ERROR"]}.items()}
    module("homeassistant.components.stt", SpeechToTextEntity=Entity,
           SpeechResult=lambda text, state: SimpleNamespace(text=text, state=state), **enums)
    module("homeassistant.components.tts", TextToSpeechEntity=Entity)
    module("homeassistant.exceptions", HomeAssistantError=RuntimeError)
    module("homeassistant.helpers.intent", IntentResponse=IntentResponse,
           IntentResponseType=SimpleNamespace(ACTION_DONE="action_done", QUERY_ANSWER="query_answer"),
           IntentResponseErrorCode=SimpleNamespace(FAILED_TO_HANDLE="failed"))
    module("homeassistant.helpers.device_registry", async_get=lambda hass: SimpleNamespace(async_get=lambda device_id: SimpleNamespace(area_id="kitchen")))
    module("_test_voice_package", __path__=[str(ROOT)])
    loaded = {}
    for name in ("client", "conversation", "stt", "tts"):
        spec = importlib.util.spec_from_file_location("_test_voice_package." + name, ROOT / (name + ".py"))
        item = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, item)
        spec.loader.exec_module(item)
        loaded[name] = item
    return loaded


async def test_conversation_identity_area_and_chime_metadata(modules):
    class Remote:
        def __init__(self): self.calls = []
        async def request(self, path, **kwargs):
            self.calls.append((path, kwargs))
            return {"request_id":"req", "outcome":"clarify", "response_mode":"speech", "reply":"Which room?"}
    remote = Remote()
    entry = SimpleNamespace(entry_id="entry", runtime_data={"core": remote})
    entity = modules["conversation"].JevConversation(entry)
    entity.hass = object()
    user = SimpleNamespace(text="turn it on", context=SimpleNamespace(user_id="user1"),
                           conversation_id="shared", device_id=None, language="en")
    result = await entity.async_process(user)
    assert remote.calls[0][1]["json"] == {"text":"turn it on", "user_id":"user1", "conversation_id":"shared", "speaker_area":None}
    assert result.continue_conversation
    assert result.response.extra_data["ha_jev"]["version"] == 1
    user.device_id = "microphone"
    await entity.async_process(user)
    assert remote.calls[-1][1]["json"]["speaker_area"] == "kitchen"
    user.context.user_id = None
    result = await entity.async_process(user)
    assert result.response.error[0] == "failed"
    assert len(remote.calls) == 2


async def test_stt_audio_and_tts_failure_translation(modules):
    class Remote:
        async def transcribe(self, pcm):
            assert pcm == b"\x00\x00" * 2
            return "hello"
        async def speak(self, message):
            raise modules["client"].ServiceError("unavailable")
    entry = SimpleNamespace(entry_id="entry", runtime_data={"speech": Remote()})
    entity = modules["stt"].JevSTT(entry)
    entity.check_metadata = lambda metadata: True
    async def stream():
        yield b"\x00\x00"
        yield b"\x00\x00"
    assert (await entity.async_process_audio_stream(None, stream())).text == "hello"
    async def oversized():
        yield b"x" * (60 * 16000 * 2 + 1)
    assert (await entity.async_process_audio_stream(None, oversized())).state == "ERROR"
    with pytest.raises(RuntimeError, match="unavailable"):
        await modules["tts"].JevTTS(entry).async_get_tts_audio("hello", "en", {})
