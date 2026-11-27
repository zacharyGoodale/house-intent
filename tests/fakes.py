"""Test doubles: FakeJev, FakeLuna, and a fake Home Assistant WebSocket server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from ha_jev.jev import Answer, ChoiceAnswer, JevError, JevResponse, NoulAnswer
from ha_jev.luna import LunaError, LunaReply

AnswerSpec = Any  # str | float | dict[str, float] | ChoiceAnswer | NoulAnswer


def _to_answer(q: Mapping[str, Any], value: AnswerSpec) -> Answer:
    if isinstance(value, (ChoiceAnswer, NoulAnswer)):
        return value
    if q["type"] == "noul":
        return NoulAnswer(float(value))
    options = list(q["criteria"])
    if isinstance(value, str):
        probs = {o: (1.0 if o == value else 0.0) for o in options}
        if value not in probs:
            probs[value] = 1.0  # adversarial: an option that was never offered
        return ChoiceAnswer(value, 1.0, probs)
    if isinstance(value, tuple):  # (choice, confidence)
        choice, conf = value
        probs = {o: 0.0 for o in options}
        probs[choice] = conf
        rest = [o for o in options if o != choice]
        if rest:
            probs[rest[-1]] = round(1.0 - conf, 6)
        return ChoiceAnswer(choice, conf, probs)
    if isinstance(value, Mapping):
        probs = {o: 0.0 for o in options}
        probs.update({k: float(v) for k, v in value.items()})
        choice = max(probs, key=lambda k: probs[k])
        return ChoiceAnswer(choice, probs[choice], probs)
    raise TypeError(f"bad fake answer {value!r}")


def default_answer(q: Mapping[str, Any]) -> Answer:
    if q["type"] == "noul":
        return NoulAnswer(0.0)
    options = list(q["criteria"])
    choice = "none" if "none" in options else options[0]
    return ChoiceAnswer(choice, 1.0, {o: (1.0 if o == choice else 0.0) for o in options})


class FakeJev:
    """FakeJev(answers={...}) fills unlisted Choices with `none` (or the first option) at 1.0 and unlisted
    Nouls with 0.0. FakeJev(fn) computes answers from (state, questions). `error` makes every call fail."""

    def __init__(self, answers: Mapping[str, AnswerSpec] | Callable[[dict, dict], Mapping[str, AnswerSpec]] | None = None,
                 *, error: Exception | None = None, model: str = "jev-fake", input_tokens: int = 5000) -> None:
        self._answers = answers or {}
        self._error = error
        self.calls: list[tuple[dict, dict]] = []
        self.model = model
        self.input_tokens = input_tokens

    async def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse:
        self.calls.append((state, questions))
        if self._error is not None:
            raise self._error
        spec = self._answers(state, questions) if callable(self._answers) else self._answers
        answers: dict[str, Answer] = {}
        for qid, q in questions.items():
            answers[qid] = _to_answer(q, spec[qid]) if qid in spec else default_answer(q)
        for qid, value in spec.items():  # adversarial extras for ids never asked are ignored by design
            if qid not in answers and isinstance(value, (ChoiceAnswer, NoulAnswer)):
                answers[qid] = value
        return JevResponse(answers=answers, model=self.model, request_id="req_fake",
                           input_tokens=self.input_tokens, latency_ms=1.0, attempts=1)


def jev_down() -> FakeJev:
    return FakeJev(error=JevError("jev unavailable: timeout", status=None))


class FakeLuna:
    def __init__(self, text: str = "Paris is the capital of France.", *, error: bool = False) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    async def answer(self, utterance: str, *, now: datetime, speaker_area: str | None,
                     previous_response_id: str | None, home: str | None = None, profile=None) -> LunaReply:
        self.calls.append({"utterance": utterance, "speaker_area": speaker_area, "previous_response_id": previous_response_id,
                           "home": home, "profile": profile})
        if self.error:
            raise LunaError("luna down")
        return LunaReply(text=self.text, response_id=f"resp_{len(self.calls)}", input_tokens=300, output_tokens=20,
                         cached_tokens=0, latency_ms=5.0, model="gpt-6-luna")


class FakeHAServer:
    """A tiny fake of HA's WebSocket API for RealHA tests."""

    def __init__(self, token: str = "good-token", *, registry: dict | None = None, states: list | None = None,
                 exposed: dict | None = None, service_delay: float = 0.0, configs: dict | None = None) -> None:
        self.token = token
        self.configs = configs or {}  # entity_id -> script/automation config, served by `<domain>/config`
        self.fail_types: set[str] = set()  # command types answered with an error
        self.registry = registry or {"areas": [], "devices": [], "entities": []}
        self.states = states or []
        self.exposed = exposed or {}
        self.service_delay = service_delay
        self.received: list[dict] = []
        self.subscriptions: dict[str, int] = {}
        self._conns: list[Any] = []
        self._server: Any = None
        self.port: int = 0

    async def start(self) -> str:
        import websockets

        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{self.port}"

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def push_event(self, event_type: str, data: dict) -> None:
        for ws in list(self._conns):
            sub_id = self.subscriptions.get(event_type)
            if sub_id is not None:
                await ws.send(json.dumps({"id": sub_id, "type": "event",
                                          "event": {"event_type": event_type, "data": data}}))

    async def _handle(self, ws) -> None:
        await ws.send(json.dumps({"type": "auth_required", "ha_version": "2026.9.0"}))
        msg = json.loads(await ws.recv())
        self.received.append(msg)
        if msg.get("type") != "auth" or msg.get("access_token") != self.token:
            await ws.send(json.dumps({"type": "auth_invalid", "message": "Invalid access token"}))
            await ws.close()
            return
        await ws.send(json.dumps({"type": "auth_ok", "ha_version": "2026.9.0"}))
        self._conns.append(ws)
        try:
            async for raw in ws:
                m = json.loads(raw)
                self.received.append(m)
                await self._reply(ws, m)
        except Exception:  # noqa: BLE001 - connection closed
            pass
        finally:
            if ws in self._conns:
                self._conns.remove(ws)

    async def _reply(self, ws, m: dict) -> None:
        t = m.get("type")
        result: Any = None
        if t in self.fail_types:
            await ws.send(json.dumps({"id": m["id"], "type": "result", "success": False,
                                      "error": {"code": "unknown_error", "message": "injected"}}))
            return
        if t == "config/area_registry/list":
            result = self.registry["areas"]
        elif t == "config/device_registry/list":
            result = self.registry["devices"]
        elif t == "config/entity_registry/list":
            result = self.registry["entities"]
        elif t == "homeassistant/expose_entity/list":
            result = {"exposed_entities": self.exposed}
        elif t == "get_states":
            result = self.states
        elif t in ("script/config", "automation/config"):
            cfg = self.configs.get(m.get("entity_id"))
            if cfg is None:
                await ws.send(json.dumps({"id": m["id"], "type": "result", "success": False,
                                          "error": {"code": "not_found", "message": "Entity not found"}}))
                return
            result = {"config": cfg}
        elif t == "subscribe_events":
            self.subscriptions[m.get("event_type")] = m["id"]
        elif t == "ping":
            await ws.send(json.dumps({"id": m["id"], "type": "pong"}))
            return
        elif t == "call_service":
            if self.service_delay:
                await asyncio.sleep(self.service_delay)
            if m.get("domain") == "fail":
                await ws.send(json.dumps({"id": m["id"], "type": "result", "success": False,
                                          "error": {"code": "not_found", "message": "Service not found"}}))
                return
            targets = (m.get("target") or {}).get("entity_id") or []
            response = {eid: {"echo": m.get("service_data")} for eid in targets}  # HA keys responses by entity_id
            result = {"context": {"id": "ctx"}, "response": response if m.get("return_response") else None}
        await ws.send(json.dumps({"id": m["id"], "type": "result", "success": True, "result": result}))
