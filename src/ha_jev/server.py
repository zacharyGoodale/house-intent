"""Trusted HA-only HTTP bridge. Every user's Core uses that user's HA token."""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web


@dataclass
class UserRuntime:
    core: object
    person: str | None = None
    gate: asyncio.Lock | None = None

    def __post_init__(self):
        self.gate = asyncio.Lock()


def create_app(users: dict[str, UserRuntime], api_key: str) -> web.Application:
    """Only the trusted HA custom component may assert user_id; never expose to browsers."""
    if len(api_key) < 32:
        raise ValueError("HA_JEV_BRIDGE_KEY must contain at least 32 characters")

    @web.middleware
    async def authenticate(request, handler):
        if not hmac.compare_digest(request.headers.get("Authorization", "").encode("utf-8"), ("Bearer " + api_key).encode("utf-8")):
            raise web.HTTPUnauthorized()
        return await handler(request)

    app = web.Application(middlewares=[authenticate], client_max_size=16384)

    async def models(request):
        return web.json_response({"data": [{"id": "jev", "loaded": bool(users)}], "contract_version": 1})

    async def converse(request):
        try:
            data = await request.json()
        except (ValueError, UnicodeError):
            raise web.HTTPBadRequest(text="Invalid JSON") from None
        if not isinstance(data, dict) or set(data) - {"text", "user_id", "conversation_id", "speaker_area", "device_id"}:
            raise web.HTTPBadRequest(text="Invalid conversation fields")
        text = data.get("text")
        uid = data.get("user_id")
        cid = data.get("conversation_id")
        area = data.get("speaker_area")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise web.HTTPBadRequest(text="Text must contain 1–4000 characters")
        if not isinstance(uid, str) or uid not in users:
            raise web.HTTPForbidden(text="Home Assistant user is not configured")
        if not isinstance(cid, str) or not cid or len(cid) > 128:
            raise web.HTTPBadRequest(text="Conversation ID required, at most 128 characters")
        if area is not None and (not isinstance(area, str) or len(area) > 128):
            raise web.HTTPBadRequest(text="Invalid speaker area")
        device = data.get("device_id")
        if device is not None and (not isinstance(device, str) or not device or len(device) > 128):
            raise web.HTTPBadRequest(text="Invalid device ID")
        runtime = users[uid]
        # Each user's entire core is independent. Same-user sessions serialized to avoid
        # competing pending confirmations; browser cannot select person or token.
        if runtime.gate.locked():
            raise web.HTTPTooManyRequests(headers={"Retry-After": "1"})
        async with runtime.gate:
            result = await runtime.core.handle(text, speaker_area=area, session_id=cid, person=runtime.person, **({"device_id": device} if device else {}))
        return web.json_response({"contract_version": 1, "request_id": result.request_id,
            "outcome": str(result.outcome), "response_mode": str(result.response_mode), "reply": result.reply})

    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/conversation", converse)
    return app


async def build_app(args):
    from ha_jev.config import load_settings
    from ha_jev.core import Core
    from ha_jev.ha.real import RealHA
    from ha_jev.hermes import make_fallback
    from ha_jev.jev import TypeSafeJevClient
    from ha_jev.memory import MemoryStore

    key = os.environ.get("HA_JEV_BRIDGE_KEY", "")
    if len(key) < 32:
        raise ValueError("Set HA_JEV_BRIDGE_KEY to a random secret of at least 32 characters")
    url = os.environ.get("HA_URL", "")
    parsed = urlsplit(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}):
        raise ValueError("HA_URL must use HTTPS (HTTP allowed only on loopback)")
    settings = load_settings(args.config_dir)
    identities = json.loads(Path(args.users_file).read_text())
    if not isinstance(identities, dict) or not identities:
        raise ValueError("Users file must map HA user IDs to token_env and optional person")
    users, resources = {}, []
    memory = MemoryStore.from_settings(settings.memory)
    try:
        for uid, config in identities.items():
            if not isinstance(config, dict) or set(config) - {"token_env", "person", "allow_hermes"}:
                raise ValueError("Invalid user configuration")
            token = os.environ.get(config.get("token_env", ""), "")
            if not token:
                raise ValueError("Missing mapped HA user's token environment variable")
            person = config.get("person")
            if person is not None and settings.person(person) is None:
                raise ValueError("Mapped person is absent from people.toml")
            ha = RealHA(url, token, entity_id_map=settings.ha_real.entity_id_map,
                        ca_file=os.environ.get("HA_CA_FILE") or None)
            resources.append(ha)
            await ha.connect()
            current = await ha.current_user()
            if current.get("id") != uid:
                raise ValueError("Mapped HA user ID does not match token owner")
            if args.fallback == "hermes" and (config.get("allow_hermes") is not True or not current.get("is_admin")):
                raise ValueError("Hermes requires explicit allow_hermes and an administrator token; its tools do not inherit caller permissions")
            jev = TypeSafeJevClient(settings.jev)
            resources.append(jev)
            fallback = make_fallback(settings, args.fallback, args.no_llm)
            # Persistent memory shared by configured person policy; no private identity from caller.
            core = await Core.create(settings, ha, jev, fallback,
                                     memory=memory)
            users[uid] = UserRuntime(core, person)
        app = create_app(users, key)
    except BaseException:
        for resource in reversed(resources):
            await (resource.aclose() if hasattr(resource, "aclose") else resource.close())
        raise

    async def close(app):
        for resource in reversed(resources):
            await (resource.aclose() if hasattr(resource, "aclose") else resource.close())
    app.on_cleanup.append(close)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users-file", required=True, help="JSON HA user ID -> token_env, optional person")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--fallback", choices=["luna", "hermes"], default="luna")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    # TLS reverse proxy is the only LAN entry point. Never bind plaintext bearer API to LAN.
    web.run_app(build_app(args), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
