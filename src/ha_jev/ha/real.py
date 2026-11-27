"""RealHA (spec §9.2): Home Assistant over its WebSocket API. Built and tested against a fake server only;
not wired to the house in this phase (the CLI refuses --ha real without HA_JEV_ALLOW_REAL=1, HA_URL, HA_TOKEN).

Entity ids always come from the live registry. `entity_id_map` maps an id as written in config to the live id
(for the rare case a glob will not do); RealHA exposes the config id and translates back on calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ha_jev import capabilities as caps
from ha_jev.ha.interface import Area, Device, EntityEntry, EntityState, HAError, Registry
from ha_jev.obslog import redact

log = logging.getLogger("ha_jev.ha.real")

REGISTRY_EVENTS = ("area_registry_updated", "device_registry_updated", "entity_registry_updated")
EXPOSE_COMMAND = "homeassistant/expose_entity/list"  # verify against HA source when wiring (spec §9.2)


def _ws_url(url: str) -> str:
    url = url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    return url + "/api/websocket"


class RealHA:
    def __init__(self, url: str, token: str, *, entity_id_map: Mapping[str, str] | None = None, timeout: float = 3.0,
                 ping_interval: float = 30.0, backoff: Sequence[float] = (1, 2, 5),
                 on_registry_change: Callable[[], Any] | None = None, ca_file: str | None = None) -> None:
        self._url = _ws_url(url)
        self._ssl = ssl.create_default_context(cafile=ca_file) if ca_file else None
        self._token = token
        self._to_live = dict(entity_id_map or {})
        self._from_live = {v: k for k, v in self._to_live.items()}
        self._timeout = timeout
        self._ping_interval = ping_interval
        self._backoff = tuple(backoff)
        self._on_registry_change = on_registry_change
        self._ws: Any = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._subs: dict[int, str] = {}
        self._reader: asyncio.Task | None = None
        self._pinger: asyncio.Task | None = None
        self._closing = False
        self._registry: Registry | None = None
        self._states: dict[str, EntityState] = {}
        self._raw_registry: dict[str, list] = {}

    # ---- ids ----------------------------------------------------------------------------------
    def _cfg(self, live_id: str) -> str:
        return self._from_live.get(live_id, live_id)

    def _live(self, cfg_id: str) -> str:
        return self._to_live.get(cfg_id, cfg_id)

    # ---- connection ---------------------------------------------------------------------------
    async def connect(self) -> None:
        import websockets

        try:
            options = {"ssl": self._ssl} if self._ssl and self._url.startswith("wss://") else {}
            self._ws = await asyncio.wait_for(websockets.connect(self._url, max_size=None, **options), self._timeout)
            first = json.loads(await asyncio.wait_for(self._ws.recv(), self._timeout))
            if first.get("type") != "auth_required":
                raise HAError(f"unexpected handshake message: {first.get('type')}")
            await self._ws.send(json.dumps({"type": "auth", "access_token": self._token}))
            log.debug("sent auth %s", redact(json.dumps({"type": "auth", "access_token": self._token})))
            reply = json.loads(await asyncio.wait_for(self._ws.recv(), self._timeout))
        except HAError:
            raise
        except Exception as e:  # noqa: BLE001
            raise HAError(f"cannot connect to Home Assistant: {type(e).__name__}") from e
        if reply.get("type") != "auth_ok":
            await self._ws.close()
            raise HAError("Home Assistant rejected the token (auth_invalid)")
        log.debug("auth ok, ha_version=%s", reply.get("ha_version"))
        self._reader = asyncio.create_task(self._read_loop())
        if self._ping_interval > 0:
            self._pinger = asyncio.create_task(self._ping_loop())
        await self._load_all()
        await self._subscribe("state_changed")
        for ev in REGISTRY_EVENTS:
            await self._subscribe(ev)

    async def close(self) -> None:
        self._closing = True
        for task in (self._pinger, self._reader):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._fail_pending(HAError("connection closed"))

    def _fail_pending(self, err: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(err)
        self._pending.clear()

    async def _read_loop(self) -> None:
        while True:
            try:
                async for raw in self._ws:
                    self._dispatch(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("websocket read error: %s", type(e).__name__)
            if self._closing:
                return
            self._fail_pending(HAError("connection lost"))
            await self._reconnect()
            return

    async def _reconnect(self) -> None:
        delays = list(self._backoff) or [5]
        attempt = 0
        while not self._closing:
            await asyncio.sleep(delays[min(attempt, len(delays) - 1)])
            attempt += 1
            try:
                await self.connect()
                log.debug("reconnected")
                return
            except HAError as e:
                log.debug("reconnect failed: %s", e)

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(self._ping_interval)
            with contextlib.suppress(HAError):
                await self._command({"type": "ping"})

    def _dispatch(self, msg: dict) -> None:
        mid = msg.get("id")
        if msg.get("type") == "event" and mid in self._subs:
            self._on_event(msg.get("event") or {})
            return
        fut = self._pending.pop(mid, None) if mid is not None else None
        if fut is None or fut.done():
            return
        if msg.get("type") == "pong":
            fut.set_result(None)
        elif msg.get("success"):
            fut.set_result(msg.get("result"))
        else:
            err = msg.get("error") or {}
            fut.set_exception(HAError(f"{err.get('code', 'error')}: {err.get('message', '')}"))

    async def _command(self, payload: dict, on_id: Callable[[int], None] | None = None) -> Any:
        if self._ws is None:
            raise HAError("not connected")
        mid = self._next_id
        self._next_id += 1
        if on_id is not None:
            on_id(mid)  # the id this command really uses (a subscription's events carry it)
        msg = {"id": mid, **payload}
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        log.debug("send %s", redact(json.dumps(msg)))
        try:
            await self._ws.send(json.dumps(msg))
            return await asyncio.wait_for(fut, self._timeout)
        except TimeoutError as e:
            self._pending.pop(mid, None)
            raise HAError(f"timeout waiting for {payload.get('type')}") from e
        except HAError:
            raise
        except Exception as e:  # noqa: BLE001
            self._pending.pop(mid, None)
            raise HAError(f"websocket error: {type(e).__name__}") from e

    async def _subscribe(self, event_type: str) -> None:
        await self._command({"type": "subscribe_events", "event_type": event_type},
                            on_id=lambda mid: self._subs.__setitem__(mid, event_type))

    def _on_event(self, event: dict) -> None:
        etype = event.get("event_type")
        data = event.get("data") or {}
        if etype == "state_changed":
            new = data.get("new_state")
            eid = data.get("entity_id")
            if not eid:
                return
            if new is None:
                self._states.pop(self._cfg(eid), None)
            else:
                self._states[self._cfg(eid)] = EntityState(self._cfg(eid), new.get("state"), dict(new.get("attributes") or {}))
        elif etype in REGISTRY_EVENTS:
            asyncio.get_running_loop().create_task(self._registry_changed())

    def set_registry_listener(self, callback: Callable[[], Any]) -> None:
        """Core registers `refresh_catalog` here, so a registry change rebuilds the catalog (§4.1 rule 1)."""
        self._on_registry_change = callback

    async def _registry_changed(self) -> None:
        try:
            await self._load_all()
        except HAError as e:
            log.warning("registry reload failed (%s); keeping the previous catalog", e)
            return  # never rebuild from stale data (review S-8)
        if self._on_registry_change is not None:
            res = self._on_registry_change()
            if asyncio.iscoroutine(res):
                await res

    # ---- registry and states ------------------------------------------------------------------
    async def _load_all(self) -> None:
        areas = await self._command({"type": "config/area_registry/list"}) or []
        devices = await self._command({"type": "config/device_registry/list"}) or []
        entities = await self._command({"type": "config/entity_registry/list"}) or []
        try:
            exposed_raw = await self._command({"type": EXPOSE_COMMAND}) or {}
        except HAError as e:
            # Without the expose list every Assist-exposed entity would silently drop out of scope (ST-6).
            raise HAError(f"cannot read Assist exposure ({EXPOSE_COMMAND}): {e}") from e
        states = await self._command({"type": "get_states"}) or []
        self._states = {self._cfg(s["entity_id"]): EntityState(self._cfg(s["entity_id"]), s.get("state"),
                                                               dict(s.get("attributes") or {})) for s in states}
        exposed_map = exposed_raw.get("exposed_entities", exposed_raw) if isinstance(exposed_raw, dict) else {}
        configs = await self._load_definitions(entities)
        self._registry = self._build_registry(areas, devices, entities, exposed_map, configs)

    async def _load_definitions(self, entities: list) -> dict[str, Any]:
        """Script and automation configs (decision 6: refuse routines that reach a lock). A failed fetch leaves
        the definition unknown: logged, not fatal."""
        out: dict[str, Any] = {}
        for e in entities:
            live_id = e["entity_id"]
            command = caps.DEFAULT.definition_command(caps.domain_of(live_id))
            if command is None or e.get("entity_category"):
                continue
            try:
                result = await self._command({"type": command, "entity_id": live_id})
            except HAError as err:
                log.warning("definition of %s unavailable (%s); only the name rule protects it", live_id, err)
                continue
            if isinstance(result, dict) and isinstance(result.get("config"), dict):
                out[self._cfg(live_id)] = result["config"]
        return out

    def _build_registry(self, areas: list, devices: list, entities: list, exposed: Mapping[str, Any],
                        configs: Mapping[str, Any] | None = None) -> Registry:
        area_map = {a.get("area_id") or a.get("id"): Area(a.get("area_id") or a.get("id"), a.get("name", ""),
                                                          tuple(a.get("aliases") or ())) for a in areas}
        dev_map = {d["id"]: Device(d["id"], d.get("name_by_user") or d.get("name") or d["id"], d.get("area_id"),
                                   d.get("manufacturer"), d.get("model")) for d in devices}
        ents: dict[str, EntityEntry] = {}
        for e in entities:
            if e.get("entity_category"):
                continue
            live_id = e["entity_id"]
            eid = self._cfg(live_id)
            domain = live_id.split(".", 1)[0]
            st = self._states.get(eid)
            attrs = dict(st.attributes) if st else {}
            device = dev_map.get(e.get("device_id") or "")
            area = e.get("area_id") or (device.area_id if device else None)
            name = attrs.get("friendly_name") or e.get("name") or e.get("original_name") or live_id
            bits = attrs.get("supported_features") or e.get("supported_features") or 0
            capabilities = dict(e.get("capabilities") or {})
            for k in caps.CAPABILITY_ATTRIBUTES:
                if k in attrs and k not in capabilities:
                    capabilities[k] = attrs[k]
            exp = exposed.get(live_id)
            is_exposed = bool(exp.get("conversation")) if isinstance(exp, dict) else bool(exp)
            ents[eid] = EntityEntry(
                entity_id=eid, name=str(name).strip(), area_id=area, device_id=e.get("device_id"),
                device_class=e.get("device_class") or e.get("original_device_class") or attrs.get("device_class"),
                features=frozenset(caps.features_from_bits(domain, int(bits))), capabilities=capabilities,
                exposed=is_exposed, disabled=bool(e.get("disabled_by")), hidden=bool(e.get("hidden_by")),
                platform=e.get("platform"), aliases=tuple(e.get("aliases") or ()),
                config=(configs or {}).get(eid), unique_id=e.get("unique_id"),
            )
        return Registry(areas=area_map, devices=dev_map, entities=ents)

    async def get_registry(self) -> Registry:
        if self._registry is None:
            await self._load_all()
        if self._registry is None:
            raise HAError("registry unavailable")
        return self._registry

    async def get_states(self) -> dict[str, EntityState]:
        return dict(self._states)

    async def current_user(self) -> dict:
        """Return the authenticated token owner for the trusted conversation bridge."""
        result = await self._command({"type": "auth/current_user"})
        if not isinstance(result, dict) or not result.get("id"):
            raise HAError("Home Assistant did not return the authenticated user")
        return result

    async def config_entry_id(self, integration: str) -> str | None:
        """The first loaded config entry of an integration (the music search needs it), via `config_entries/get`."""
        entries = await self._command({"type": "config_entries/get", "domain": integration}) or []
        loaded = [e for e in entries if isinstance(e, dict) and e.get("domain") == integration
                  and e.get("state", "loaded") == "loaded"]
        return str(loaded[0]["entry_id"]) if loaded else None

    async def call_service(self, domain: str, service: str, entity_ids: Sequence[str], data: Mapping[str, Any],
                           return_response: bool = False) -> dict | None:
        payload = {"type": "call_service", "domain": domain, "service": service, "service_data": dict(data or {}),
                   "return_response": return_response}
        if entity_ids:  # services without a target (music search) must not get an empty one
            payload["target"] = {"entity_id": [self._live(e) for e in entity_ids]}
        result = await self._command(payload)
        if return_response:
            resp = (result or {}).get("response")
            if isinstance(resp, dict):
                return {self._cfg(k): v for k, v in resp.items()}
            return resp
        return None
