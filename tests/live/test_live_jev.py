"""Real Jev through Core with StubHA and NullLuna (spec §14). About 20 requests, well under a cent."""

import os

import pytest

from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.jev import TypeSafeJevClient
from ha_jev.luna import NullLuna

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")]

# Review S3: the Common Area group also drives the kitchen Pendant, so the living room addresses its own lights.
LIVING = ("light.paper_lantern,light.ceiling_light,light.backlight_left,light.backlight_right,light.alder,light.ring_lamp,"
          "light.reading_lamp,light.corner_lamp,light.ribbon_lamp")
OFFICE = {"areas": [{"id": "office", "name": "Office", "aliases": []}],
          "entities": [{"entity_id": "light.desk_lamp", "friendly_name": "Desk Lamp", "area_id": "office",
                        "capabilities": {"supported_color_modes": ["color_temp", "xy"]}},
                       {"entity_id": "light.office", "friendly_name": "Office", "area_id": "office",
                        "capabilities": {"supported_color_modes": ["color_temp", "xy"]}}],
          "states": {"light.office": {"state": "off", "attributes": {"entity_id": ["light.desk_lamp"]}}}}

# Single named devices where Jev's `scope` Choice often splits between one_named and area (confidence ~0.6):
# the spec's min-confidence rule then asks to confirm. The target must still be right; whether it executes
# or confirms is the act-threshold tuning question left to the golden-set loop.
CONFIRM_OK = [
    ("turn off the ceiling light", "living_room", "light.ceiling_light", ["light.turn_off light.ceiling_light {}"]),
    ("turn on the fern lamp", "bedroom", "light.fern_lamp", ["light.turn_on light.fern_lamp {}"]),
    ("kitchen daytime", "living_room", "preset.kitchen_day",
     ['light.turn_on light.pendant_lower,light.pendant_upper {"brightness_pct": 85, "color_temp_kelvin": 3000, "transition": 0.5}']),
]

CASES = [
    # utterance, speaker area, route prefix, outcome, calls (describe() strings)
    ("turn off Alder", "living_room", "device_control", "executed", ["light.turn_off light.alder {}"]),
    ("turn on the living room lights", "bedroom", "device_control", "executed", [f"light.turn_on {LIVING} {{}}"]),
    ("dim the bedroom lights to 20%", "living_room", "device_control", "executed",
     ['light.turn_on light.bedroom_group {"brightness_pct": 20}']),
    ("turn on Alder and the Paper Lantern", "living_room", "device_control", "executed",
     ["light.turn_on light.paper_lantern,light.alder {}"]),
    ("movie mode", "living_room", "activate_scene", "executed", ["script.turn_on script.demo_movie_mode {}"]),
    ("bedtime", "bedroom", "activate_scene", "executed", ["script.turn_on script.demo_bedroom_bed {}"]),
    ("unlock the front door", "living_room", "pre_gate", "refused", []),
    ("is the front door locked?", "living_room", "device_query", "answered", []),
    ("what's the weather going to be tomorrow", "living_room", "weather", "answered", []),
    ("when's sunset", "living_room", "device_query", "answered", []),
    ("what's the capital of France", "living_room", "luna", "answered", []),
    ("pause the Soundbar", "living_room", "device_control", "executed", ["media_player.media_pause media_player.demo_soundbar {}"]),
    ("set the Soundbar volume to 25", "living_room", "device_control", "executed",
     ['media_player.volume_set media_player.demo_soundbar {"volume_level": 0.25}']),
    ("close the blinds", "bedroom", "device_control", "executed", ["cover.close_cover cover.bedroom_shade {}"]),
    ("open Netflix on the Apple TV", "living_room", "device_control", "executed",
     ['media_player.select_source media_player.demo_tv {"source": "Netflix"}']),
    ("add oat milk to the shopping list", "living_room", "list_add", "executed",
     ['todo.add_item todo.shopping_list {"item": "oat milk"}']),
    # This generic StubHA has no installed schedule service; surface that failure.
    ("set a timer for five minutes", "living_room", "timer", "error", []),
]


async def _core(overlays=()):
    ha = StubHA.default(overlays=overlays)
    settings = load_settings().with_scope_extension(**ha.overlay_scope)
    jev = TypeSafeJevClient(settings.jev)
    core = await Core.create(settings, ha, jev, NullLuna(), clock=lambda: ha.now)
    return core, jev


@pytest.mark.parametrize("text,area,route,outcome,calls", CASES, ids=[c[0] for c in CASES])
async def test_live_routes(text, area, route, outcome, calls):
    core, jev = await _core()
    try:
        r = await core.handle(text, speaker_area=area)
    finally:
        await jev.aclose()
    info = f"{r.debug.route} {r.outcome} {r.reply!r} conf={r.debug.confidence}"
    assert r.debug.route.startswith(route), info
    assert str(r.outcome) == outcome, info
    assert [c.describe() for c in r.service_calls] == calls, info
    if route != "pre_gate":
        assert len(r.debug.jev) == 1 and r.debug.jev[0].error is None  # decision 1
        assert r.debug.jev[0].model.startswith("jev-")


@pytest.mark.parametrize("text,area,target,calls", CONFIRM_OK, ids=[c[0] for c in CONFIRM_OK])
async def test_live_named_device_executes_or_confirms_the_right_target(text, area, target, calls):
    core, jev = await _core()
    try:
        r = await core.handle(text, speaker_area=area)
    finally:
        await jev.aclose()
    info = f"{r.debug.route} {r.outcome} {r.reply!r} conf={r.debug.confidence}"
    if str(r.outcome) == "executed":
        assert [c.describe() for c in r.service_calls] == calls, info
    else:
        assert str(r.outcome) == "clarify" and r.clarify_options[:1] == [target], info
        assert r.service_calls == []


async def test_live_compound_is_all_or_nothing():
    core, jev = await _core()
    try:
        r = await core.handle("turn off the lights and pause the music", speaker_area="bedroom")
    finally:
        await jev.aclose()
    assert r.debug.route.startswith("compound"), r.debug.route
    if str(r.outcome) == "executed":
        assert [c.describe() for c in r.service_calls] == [
            "light.turn_off light.bedroom_group {}", "media_player.media_pause media_player.demo_bedroom_speaker {}"]
    else:
        assert r.reply.startswith("Sorry, I couldn't do all of that") and r.service_calls == []


async def test_live_unseen_room_from_overlay():
    core, jev = await _core(overlays=[OFFICE])
    try:
        r = await core.handle("turn on the office lights", speaker_area="living_room")
    finally:
        await jev.aclose()
    assert [c.describe() for c in r.service_calls] == ["light.turn_on light.office {}"], r.reply
