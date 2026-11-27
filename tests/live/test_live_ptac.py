"""Real language classification, with all appliance writes confined to StubHA."""
import os
import pytest
from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.jev import TypeSafeJevClient
from ha_jev.luna import NullLuna
from tests.unit.test_ptac import OVERLAY, CASES, QUERY_CASES

pytestmark = [pytest.mark.live, pytest.mark.skipif(not os.environ.get('TYPESAFE_API_KEY'), reason='TYPESAFE_API_KEY not set')]

@pytest.mark.parametrize('text,target,qid,action,domain,service,data', CASES)
async def test_live_ptac(text, target, qid, action, domain, service, data):
    settings = load_settings()
    ha = StubHA.default(overlays=[OVERLAY])
    jev = TypeSafeJevClient(settings.jev)
    try:
        core = await Core.create(settings, ha, jev, NullLuna(), clock=lambda: ha.now)
        result = await core.handle(text, speaker_area='bedroom')
    finally:
        await jev.aclose()
    assert str(result.outcome) == 'executed', f'{result.reply} {result.debug}'
    assert [(c.domain, c.service, c.entity_ids, dict(c.data)) for c in result.service_calls] == [
        (domain, service, (target,), data)]

@pytest.mark.parametrize('text,attribute', QUERY_CASES)
async def test_live_ptac_query(text, attribute):
    settings = load_settings()
    ha = StubHA.default(overlays=[OVERLAY])
    jev = TypeSafeJevClient(settings.jev)
    try:
        core = await Core.create(settings, ha, jev, NullLuna(), clock=lambda: ha.now)
        result = await core.handle(text, speaker_area='bedroom')
    finally:
        await jev.aclose()
    assert str(result.outcome) == 'answered', f'{result.reply} {result.debug}'
    assert not result.service_calls
    assert result.debug.route == f'device_query/{attribute}', result.debug

@pytest.mark.parametrize('text,prefix', [
    ('what is the temperature outside', 'weather/'),
    ('how hot will it be tomorrow', 'weather/'),
    ('set the bedroom lights to warm white', 'device_control/'),
])
async def test_live_temperature_does_not_capture_weather_or_light_colour(text, prefix):
    settings = load_settings()
    ha = StubHA.default(overlays=[OVERLAY])
    jev = TypeSafeJevClient(settings.jev)
    try:
        core = await Core.create(settings, ha, jev, NullLuna(), clock=lambda: ha.now)
        result = await core.handle(text, speaker_area='bedroom')
    finally:
        await jev.aclose()
    assert result.debug.route.startswith(prefix), f'{result.reply} {result.debug}'
    assert not any(call.domain == 'climate' for call in result.service_calls)
