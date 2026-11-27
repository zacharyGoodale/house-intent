"""Offline trusted bridge tests with actual HTTP and a recording Core stub."""
import asyncio
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from ha_jev.server import UserRuntime, create_app

KEY = "test-bridge-key-" * 3


class CoreStub:
    def __init__(self):
        self.calls = []
    async def handle(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return SimpleNamespace(request_id="req", outcome="executed", response_mode="chime_success", reply="Done.")


@pytest.fixture
async def bridge():
    first, second = CoreStub(), CoreStub()
    users = {"first": UserRuntime(first, "person1"), "second": UserRuntime(second)}
    client = TestClient(TestServer(create_app(users, KEY)))
    await client.start_server()
    yield client, users
    await client.close()


def payload(user="first", **extra):
    return {"text": "turn on the light", "user_id": user, "conversation_id": "same", "speaker_area": None, **extra}


async def test_auth_and_strict_identity(bridge):
    client, users = bridge
    assert (await client.post("/v1/conversation", json=payload())).status == 401
    headers = {"Authorization": "Bearer " + KEY}
    assert (await client.post("/v1/conversation", json=payload("unknown"), headers=headers)).status == 403
    assert (await client.post("/v1/conversation", json=payload(person="person2"), headers=headers)).status == 400
    assert not users["first"].core.calls


async def test_partition_identity_and_no_implicit_area(bridge):
    client, users = bridge
    headers = {"Authorization": "Bearer " + KEY}
    for user in ("first", "second"):
        response = await client.post("/v1/conversation", json=payload(user), headers=headers)
        assert response.status == 200
        assert (await response.json())["response_mode"] == "chime_success"
    assert users["first"].core.calls[0][1] == {"speaker_area": None, "session_id": "same", "person": "person1"}
    assert users["second"].core.calls[0][1]["person"] is None


async def test_busy_never_executes_and_input_validation(bridge):
    client, users = bridge
    headers = {"Authorization": "Bearer " + KEY}
    await users["first"].gate.acquire()
    try:
        response = await client.post("/v1/conversation", json=payload(), headers=headers)
        assert response.status == 429
        assert response.headers["Retry-After"] == "1"
        assert not users["first"].core.calls
    finally:
        users["first"].gate.release()
    for invalid in (payload(text=""), payload(conversation_id=""), payload(speaker_area=3)):
        assert (await client.post("/v1/conversation", json=invalid, headers=headers)).status == 400


async def test_device_context_is_forwarded_and_validated(bridge):
    client, users = bridge
    headers = {"Authorization": "Bearer " + KEY}
    response = await client.post("/v1/conversation", json=payload(device_id="nabu"), headers=headers)
    assert response.status == 200
    assert users["first"].core.calls[-1][1]["device_id"] == "nabu"
    for device in ("", 12, "x" * 129):
        assert (await client.post("/v1/conversation", json=payload(device_id=device), headers=headers)).status == 400
