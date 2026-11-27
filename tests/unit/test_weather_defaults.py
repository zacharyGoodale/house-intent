import pytest

from ha_jev.replies import render_weather
from tests.fakes import FakeJev
from tests.unit.test_core import make_core


@pytest.mark.parametrize('text', ["what's the temperature?", "what's the temp outside?", 'how hot is it?'])
@pytest.mark.parametrize('classified_when', ['now', 'none', 'today'])
async def test_bare_temperature_gives_the_reading_now_then_daily_high_low_and_rain(text, classified_when):
    core, ha = await make_core(FakeJev({'intent': 'weather', 'weather_when': classified_when}))
    result = await core.handle(text)
    assert result.debug.route == 'weather/today'
    assert result.reply == "It's 66°F now. Today: partly cloudy, high 70, low 55, 10% chance of rain."
    assert result.debug.reads[0].data['type'] == 'daily'


@pytest.mark.parametrize('text,when,route', [
    ("what's the temperature right now?", 'now', 'weather/now'),
    ("what's the current temperature?", 'now', 'weather/now'),
    ("how cold is it at the moment?", 'now', 'weather/now'),
    ("what's the temperature tomorrow?", 'tomorrow', 'weather/tomorrow'),
    ("what's the humidity?", 'now', 'weather/now'),
])
async def test_explicit_time_and_other_weather_queries_keep_their_scope(text, when, route):
    core, _ = await make_core(FakeJev({'intent': 'weather', 'weather_when': when}))
    assert (await core.handle(text)).debug.route == route


@pytest.mark.parametrize('probability,phrase', [(0.5, '0.5% chance of rain'), (0, '0% chance of rain'), (None, 'rain chance unavailable')])
async def test_daily_rain_chance_preserves_small_probabilities_and_unknowns(probability, phrase):
    core, ha = await make_core(FakeJev())
    now = ha.now
    forecast = [{'datetime': now.isoformat(), 'temperature': 70, 'templow': 55,
                 'condition': 'partlycloudy', 'precipitation_probability': probability}]
    reply = render_weather('W_DAY', {'when': 'today'}, {'temperature_unit': '°F'}, 'partlycloudy', forecast, now)
    assert phrase in reply


@pytest.mark.parametrize('condition,amount,phrase', [
    ('partlycloudy', 0, 'no rain forecast'),
    ('partlycloudy', 0.1, 'rain is forecast'),
    ('rainy', None, 'rain is forecast'),
    ('snowy', 1, 'precipitation is forecast'),
])
async def test_daily_rain_without_percentage_uses_available_forecast(condition, amount, phrase):
    _, ha = await make_core(FakeJev())
    forecast = [{'datetime': ha.now.isoformat(), 'temperature': 70, 'templow': 55,
                 'condition': condition, 'precipitation': amount}]
    reply = render_weather('W_DAY', {'when': 'today'}, {}, condition, forecast, ha.now)
    assert phrase in reply and '0%' not in reply


async def test_an_explicit_today_gets_the_forecast_without_the_reading_now():
    core, _ = await make_core(FakeJev({'intent': 'weather', 'weather_when': 'today'}))
    result = await core.handle("what's the temperature today?")
    assert result.reply == 'Today: partly cloudy, high 70, low 55, 10% chance of rain.'
