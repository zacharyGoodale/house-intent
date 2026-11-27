from datetime import datetime

import httpx
import pytest

from ha_jev.cli import build_parser
from ha_jev.config import HermesSettings, LunaSettings, load_settings
from ha_jev.hermes import HermesClient, HermesWithLuna, build_hermes_instructions, make_fallback
from ha_jev.luna import LunaError, LunaProfile, NullLuna
from tests.fakes import FakeLuna

NOW = datetime.fromisoformat("2037-10-03T12:00:00-04:00")
CONTEXT = dict(now=NOW, speaker_area="Den", home="Can control lights.",
               profile=LunaProfile(memory="Household fact. ", speaker="Personal fact. "))


async def test_hermes_sends_context_auth_and_chain_and_only_returns_final(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "test-private-key")
    requests = []

    def respond(request):
        import json
        requests.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-private-key"
        return httpx.Response(200, json={"id": "resp_hermes", "status": "completed", "output": [
            {"type": "message", "role": "assistant", "phase": "commentary",
             "content": [{"type": "output_text", "text": "Searching..."}]},
            {"type": "function_call", "name": "search"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Final answer."}]},
        ], "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 40},
            "runtime": {"model": "actual-model"}})

    client = HermesClient(HermesSettings(base_url="https://example.test/v1", input_per_m=1,
                                       cached_per_m=0.1, output_per_m=2), LunaSettings(),
                          transport=httpx.MockTransport(respond))
    reply = await client.answer("Question", previous_response_id="previous", **CONTEXT)
    assert reply.text == "Final answer."
    assert reply.model == "hermes/actual-model"
    assert reply.cost_usd == pytest.approx(0.000084)
    assert requests[0]["previous_response_id"] == "previous"
    assert "reasoning" not in requests[0]  # use Hermes' configured default
    for expected in ("Household fact.", "Personal fact.", "Den", "2037-10-03 12:00", "Can control lights."):
        assert expected in requests[0]["instructions"]


@pytest.mark.parametrize("status,body", [(401, {}), (200, {"status": "failed"}),
                                         (200, {"output": []})])
async def test_hermes_failures_are_redacted(monkeypatch, status, body):
    monkeypatch.setenv("HERMES_API_KEY", "test-private-key")
    client = HermesClient(HermesSettings(base_url="https://example.test/v1"), LunaSettings(),
                          transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body)))
    with pytest.raises(LunaError) as err:
        await client.answer("Question", previous_response_id=None, **CONTEXT)
    assert "test-private-key" not in str(err.value)


async def test_failover_keeps_backend_chains_and_conversations_separate():
    hermes, luna = FakeLuna("Hermes answer"), FakeLuna("Luna answer")
    client = HermesWithLuna(hermes, luna)
    first = await client.answer("First", previous_response_id=None, **CONTEXT)
    hermes.error = True
    second = await client.answer("Follow-up", previous_response_id=first.response_id, **CONTEXT)
    assert second.text == "Luna answer"
    assert second.fallback_error == "luna down"
    assert luna.calls[-1]["previous_response_id"] is None
    third = await client.answer("Again", previous_response_id=second.response_id, **CONTEXT)
    assert luna.calls[-1]["previous_response_id"] == "resp_1"
    hermes.error = False
    await client.answer("Recovered", previous_response_id=third.response_id, **CONTEXT)
    assert hermes.calls[-1]["previous_response_id"] == "resp_1"
    await client.answer("Other conversation", previous_response_id=None, **CONTEXT)
    assert hermes.calls[-1]["previous_response_id"] is None


def test_backend_flags_and_disabled_mode():
    for command in (["tui"], ["ask", "hello"]):
        assert build_parser().parse_args(command).fallback == "hermes"
        assert build_parser().parse_args([*command, "--fallback", "luna"]).fallback == "luna"
    assert isinstance(make_fallback(load_settings(), "hermes", True), NullLuna)


async def test_both_backends_down():
    client = HermesWithLuna(FakeLuna(error=True), FakeLuna(error=True))
    with pytest.raises(LunaError, match="both unavailable"):
        await client.answer("Hello", previous_response_id=None, **CONTEXT)


def test_hermes_prompt_leaves_behavior_to_soul():
    text = build_hermes_instructions(NOW, "Den", LunaSettings(persona="CUSTOM LUNA PERSONA"),
                                     None, LunaProfile())
    assert text.startswith("Reply briefly in plain text suitable for speech.")
    assert "CUSTOM LUNA PERSONA" not in text
    assert "cannot" not in text
    assert "Do not" not in text


def test_agent_request_budget_is_separate_from_conversation_idle_window():
    # Tool-using turns may take minutes; silence after the reply still expires in 30s.
    for settings in (HermesSettings(), load_settings().hermes):
        assert settings.timeout_s == 300
        assert settings.conversation_idle_s == 30


async def test_slow_tool_turn_finishes_with_extended_request_budget(monkeypatch):
    import asyncio
    from dataclasses import replace

    monkeypatch.setenv("HERMES_API_KEY", "test-private-key")

    async def respond(request):
        await asyncio.sleep(0.03)
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Finished."}]}]})

    client = HermesClient(HermesSettings(base_url="https://example.test/v1", timeout_s=0.01),
                          LunaSettings(), transport=httpx.MockTransport(respond))
    with pytest.raises(LunaError, match="TimeoutError"):
        await client.answer("Slow tool task", previous_response_id=None, **CONTEXT)
    client.settings = replace(client.settings, timeout_s=0.1)
    reply = await client.answer("Slow tool task", previous_response_id=None, **CONTEXT)
    assert reply.text == "Finished."
