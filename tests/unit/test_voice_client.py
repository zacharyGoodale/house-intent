"""Run the dependency-free HA transport module without installing HA."""
import importlib.util
import io
from pathlib import Path
import wave

import pytest

SPEC = importlib.util.spec_from_file_location("voice_client", Path(__file__).parents[2] / "custom_components/ha_jev/client.py")
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


def test_pcm_wrapped_once_as_wav_and_limits():
    pcm = b"\x00\x01" * 16000
    with wave.open(io.BytesIO(client.pcm_wav(pcm)), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.readframes(16000) == pcm
    for invalid in (b"", b"x", b"xx" * (60 * 16000 + 1)):
        with pytest.raises(client.ServiceError):
            client.pcm_wav(invalid)


@pytest.mark.parametrize("url", ["http://speech.example.test", "https://key@speech.example.test", "https://speech.example.test?key=x"])
def test_tls_mandatory(url):
    with pytest.raises(ValueError):
        client.tls_context(url)


class Content:
    def __init__(self, body):
        self.body = body
    async def read(self, limit):
        body, self.body = self.body[:limit], self.body[limit:]
        return body
    async def iter_chunked(self, size):
        while self.body:
            chunk, self.body = self.body[:size], self.body[size:]
            yield chunk


class Response:
    def __init__(self, body, status=200, content_type="audio/wav"):
        self.content = Content(body)
        self.status = status
        self.content_type = content_type
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def read(self):
        return await self.content.read(999999)


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        return next(self.responses)


def wav(rate=24000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x00\x00" * rate)
    return output.getvalue()


async def test_synthesis_contract_and_validation():
    audio = wav()
    session = Session([Response(audio)])
    remote = client.VoiceClient(session, "https://speech.example.test:8443", "x" * 32)
    assert await remote.speak("Done.") == audio
    options = session.requests[0][1]
    assert options["json"] == {"input": "Done.", "model": "tts", "voice": "default", "response_format": "wav"}
    assert options["headers"]["Authorization"] == "Bearer " + "x" * 32
    assert options["allow_redirects"] is False
    for invalid in (wav(16000), audio[:-100], b"invalid", b"x" * (6 * 1024 * 1024 + 1)):
        remote = client.VoiceClient(Session([Response(invalid)]), "https://speech.example.test", "x" * 32)
        with pytest.raises(client.ServiceError):
            await remote.speak("Done.")


async def test_no_retry_accepted_errors_and_invalid_transcript():
    session = Session([Response(b"", status=503)])
    remote = client.VoiceClient(session, "https://speech.example.test", "x" * 32)
    with pytest.raises(client.ServiceError):
        await remote.speak("Done.")
    assert len(session.requests) == 1
    remote = client.VoiceClient(Session([Response(b'{"text": null}', content_type="application/json")]), "https://speech.example.test", "x" * 32)
    with pytest.raises(client.ServiceError):
        await remote.transcribe(b"\x00\x00" * 16000)
