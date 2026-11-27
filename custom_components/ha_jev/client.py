"""Small TLS-only client; no dependency on the Jev runtime inside HA."""
import asyncio
import io
import json as json_module
import ssl
import wave
from urllib.parse import urlsplit

import aiohttp


class ServiceError(Exception):
    """A remote voice service failed (never contains credentials)."""


def tls_context(url, ca_file=""):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("A credential-free HTTPS base URL is required")
    return ssl.create_default_context(cafile=ca_file or None)


def pcm_wav(pcm):
    if not pcm or len(pcm) % 2 or len(pcm) > 60 * 16000 * 2:
        raise ServiceError("Audio must be PCM16 mono, at most 60 seconds")
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return target.getvalue()


class VoiceClient:
    def __init__(self, session, url, key, ca_file="", timeout=350):
        if not key or len(key) < 32:
            raise ValueError("Use a random API key with at least 32 characters")
        self.session = session
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {key}"}
        self.ssl = tls_context(url, ca_file)
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def request(self, path, *, json=None, form_factory=None, binary=False):
        # Retry only rejected busy requests. Never retry an accepted command on timeout.
        for attempt in range(4):
            data = form_factory() if form_factory else None
            method = "GET" if json is None and data is None else "POST"
            try:
                async with self.session.request(method, self.url + path, headers=self.headers,
                        ssl=self.ssl, timeout=self.timeout, json=json, data=data,
                        allow_redirects=False) as response:
                    if response.status == 429 and attempt < 3:
                        await response.read()
                        await asyncio.sleep(attempt + 1)
                        continue
                    if response.status != 200:
                        raise ServiceError(f"Voice service returned HTTP {response.status}")
                    if binary:
                        if response.content_type != "audio/wav":
                            raise ServiceError("Expected WAV audio")
                        body = await response.content.read(6 * 1024 * 1024 + 1)
                        # read(n) is not readexactly; continue through the body with a cap.
                        chunks = [body]
                        size = len(body)
                        if size > 6 * 1024 * 1024:
                            raise ServiceError("Speech response too large")
                        async for chunk in response.content.iter_chunked(65536):
                            size += len(chunk)
                            if size > 6 * 1024 * 1024:
                                raise ServiceError("Speech response too large")
                            chunks.append(chunk)
                        audio = b"".join(chunks)
                        with wave.open(io.BytesIO(audio), "rb") as wav:
                            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 24000, "NONE"):
                                raise ServiceError("Unexpected speech audio format")
                            frames = wav.getnframes()
                            if not 0 < frames <= 120 * 24000 or len(wav.readframes(frames)) != frames * 2:
                                raise ServiceError("Speech WAV truncated or too long")
                        return audio
                    chunks = []
                    size = 0
                    async for chunk in response.content.iter_chunked(65536):
                        size += len(chunk)
                        if size > 65536:
                            raise ServiceError("JSON response too large")
                        chunks.append(chunk)
                    return json_module.loads(b"".join(chunks))
            except (aiohttp.ClientError, TimeoutError, ValueError, wave.Error, EOFError) as err:
                raise ServiceError("Voice service unavailable or invalid response") from err
        raise ServiceError("Voice service is busy")

    async def transcribe(self, pcm):
        audio = pcm_wav(pcm)
        def form():
            data = aiohttp.FormData()
            data.add_field("file", audio, filename="audio.wav", content_type="audio/wav")
            data.add_field("model", "stt")
            return data
        result = await self.request("/v1/audio/transcriptions", form_factory=form)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ServiceError("Invalid transcription")
        return result["text"]

    async def speak(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ServiceError("Speech must contain 1–1000 characters")
        return await self.request("/v1/audio/speech", json={"input": text, "voice": "default", "model": "tts", "response_format": "wav"}, binary=True)
