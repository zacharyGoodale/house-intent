"""Unified thermostat controls and indoor temperature queries use HA climate."""
import pytest

from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.luna import NullLuna
from tests.fakes import FakeJev

THERMOSTAT = 'climate.bedroom_demo_thermostat'
TEMP = POWER = MODE = THERMOSTAT
FAN = 'select.example_bedroom_ptac_fan_speed'
OVERLAY = {
    'entities': [{'entity_id': eid, 'friendly_name': name, 'area_id': 'bedroom', 'exposed': False}
                 for eid, name in [(THERMOSTAT, 'Bedroom thermostat'), (FAN, 'Bedroom PTAC Fan Speed')]],
    'states': {
        THERMOSTAT: {'state': 'cool', 'attributes': {
            'temperature': 68, 'current_temperature': None, 'temperature_unit': '°F',
            'min_temp': 61, 'max_temp': 88, 'target_temp_step': 1,
            'hvac_modes': ['off', 'cool', 'heat', 'fan_only'], 'supported_features': 385}},
        FAN: {'state': 'Auto', 'attributes': {'options': ['Auto', 'Low', 'High']}},
    },
}
CASES = [
    ('turn on the A/C', THERMOSTAT, 'temperature_action', 'turn_on', 'climate', 'turn_on', {}),
    ('turn off the air conditioner', THERMOSTAT, 'temperature_action', 'turn_off', 'climate', 'set_hvac_mode', {'hvac_mode': 'off'}),
    ('make it warmer', THERMOSTAT, 'temperature_action', 'warmer', 'climate', 'set_temperature', {'temperature': 69}),
    ('make it cooler', THERMOSTAT, 'temperature_action', 'cooler', 'climate', 'set_temperature', {'temperature': 67}),
    ("it's too hot in here", THERMOSTAT, 'temperature_action', 'cooler', 'climate', 'set_temperature', {'temperature': 67}),
    ("it's too cold in here", THERMOSTAT, 'temperature_action', 'warmer', 'climate', 'set_temperature', {'temperature': 69}),
    ('turn the temperature down by 3 degrees', THERMOSTAT, 'temperature_action', 'cooler', 'climate', 'set_temperature', {'temperature': 65}),
    ('set the AC to 72', THERMOSTAT, 'temperature_action', 'set_temperature', 'climate', 'set_temperature', {'temperature': 72}),
    ('switch the AC to heat mode', THERMOSTAT, 'temperature_action', 'heat', 'climate', 'set_hvac_mode', {'hvac_mode': 'heat'}),
    ('set the air conditioner to cool mode', THERMOSTAT, 'temperature_action', 'cool', 'climate', 'set_hvac_mode', {'hvac_mode': 'cool'}),
    ('set the AC fan to low', FAN, 'hvac_setting_action', 'low', 'select', 'select_option', {'option': 'Low'}),
    ('set the AC fan to high', FAN, 'hvac_setting_action', 'high', 'select', 'select_option', {'option': 'High'}),
    ('set the AC fan to auto', FAN, 'hvac_setting_action', 'auto', 'select', 'select_option', {'option': 'Auto'}),
    ('put the AC in fan only mode', THERMOSTAT, 'temperature_action', 'fan_only', 'climate', 'set_hvac_mode', {'hvac_mode': 'fan_only'}),
]

async def run(text, target, qid, action, *, initial_state=None):
    ha = StubHA.default(overlays=[OVERLAY], initial_state=initial_state)
    settings = load_settings()
    answers = {'intent': 'device_control', 'scope': 'one_named', 'target': target, qid: action,
               'number_is_setting__n0': .99}
    if text in ('make it warmer', 'make it cooler', "it's too hot in here", "it's too cold in here"):
        answers.update(scope='unspecified', target='none', device_type='thermostat')
    core = await Core.create(settings, ha, FakeJev(answers), NullLuna(), clock=lambda: ha.now)
    return await core.handle(text, speaker_area='bedroom'), ha

@pytest.mark.parametrize('text,target,qid,action,domain,service,data', CASES)
async def test_ptac_controls(text, target, qid, action, domain, service, data):
    result, ha = await run(text, target, qid, action)
    assert str(result.outcome) == 'executed', result.reply
    assert [(c.domain, c.service, c.entity_ids, dict(c.data)) for c in result.service_calls] == [
        (domain, service, (target,), data)]
    if service == 'set_temperature':
        assert (await ha.get_states())[target].attributes['temperature'] == data['temperature']
        assert (await ha.get_states())[target].state == 'cool'

@pytest.mark.parametrize('state,action,value', [('88', 'warmer', 88), ('61', 'cooler', 61)])
async def test_temperature_clamps_relative_changes(state, action, value):
    result, _ = await run(f'make it {action}', TEMP, 'temperature_action', action,
                          initial_state={TEMP: {'attributes': {'temperature': state}}})
    assert dict(result.service_calls[0].data) == {'temperature': value}

@pytest.mark.parametrize('state', ['unknown', 'unavailable', 'nan'])
async def test_temperature_never_guesses_missing_state(state):
    result, _ = await run('make it warmer', TEMP, 'temperature_action', 'warmer',
                          initial_state={TEMP: {'attributes': {'temperature': state}}})
    assert not result.service_calls

@pytest.mark.parametrize('text', ['set the thermostat to 100', 'set the thermostat to 50', 'set the thermostat to 70%'])
async def test_invalid_temperature_refuses(text):
    result, _ = await run(text, TEMP, 'temperature_action', 'set_temperature')
    assert not result.service_calls

async def test_mode_cannot_be_sent_to_fan_speed_select():
    result, _ = await run('set the fan speed to heat', FAN, 'hvac_setting_action', 'heat')
    assert not result.service_calls

@pytest.mark.parametrize('text,value', [('set the thermostat to 20 degrees C', 68),
                                       ('set the thermostat to 72 °F', 72),
                                       ('set the thermostat to 22 celsius', 72)])
async def test_explicit_temperature_units(text, value):
    result, _ = await run(text, TEMP, 'temperature_action', 'set_temperature')
    assert dict(result.service_calls[0].data) == {'temperature': value}

QUERY_CASES = [
    ("what's the temperature", 'temperature'),
    ("how hot is it in here", 'temperature'),
    ("what's the room temperature in the bedroom", 'temperature'),
    ("what is the AC set to", 'target_temperature'),
    ("what's the thermostat target", 'target_temperature'),
    ("is the AC heating or cooling", 'hvac_mode'),
    ("is the heater on", 'on_off'),
]

@pytest.mark.parametrize('text,attribute', QUERY_CASES)
async def test_temperature_queries_are_read_only(text, attribute):
    ha = StubHA.default(overlays=[OVERLAY])
    jev = FakeJev({'intent': 'device_query', 'device_type': 'thermostat',
                  'scope': 'one_named', 'target': THERMOSTAT, 'query_attribute': attribute})
    core = await Core.create(load_settings(), ha, jev, NullLuna(), clock=lambda: ha.now)
    result = await core.handle(text, speaker_area='bedroom')
    assert str(result.outcome) == 'answered', result.reply
    assert not result.service_calls
    if attribute == 'temperature':
        assert 'don’t have a room temperature reading' in result.reply
        assert '68' not in result.reply
    elif attribute == 'target_temperature':
        assert '68' in result.reply and '°F' in result.reply
    elif attribute == 'hvac_mode':
        assert 'cool' in result.reply

async def test_indoor_temperature_with_actual_reading():
    ha = StubHA.default(overlays=[OVERLAY], initial_state={TEMP: {'attributes': {'current_temperature': 71}}})
    core = await Core.create(load_settings(), ha, FakeJev({'intent': 'device_query', 'scope': 'one_named',
        'target': THERMOSTAT, 'query_attribute': 'temperature'}), NullLuna(), clock=lambda: ha.now)
    result = await core.handle("what is the temperature in here", speaker_area='bedroom')
    assert '71' in result.reply and '°F' in result.reply and not result.service_calls

async def test_legacy_controls_are_never_voice_candidates_even_when_exposed():
    legacy = ['switch.example_bedroom_ptac_power', 'number.example_bedroom_ptac_target_temperature', 'select.example_bedroom_ptac_operating_mode']
    ha = StubHA.default(overlays=[OVERLAY, {'entities': [
        {'entity_id': eid, 'friendly_name': eid, 'area_id': 'bedroom', 'exposed': True} for eid in legacy]}])
    jev = FakeJev({'intent': 'general'})
    core = await Core.create(load_settings(), ha, jev, NullLuna(), clock=lambda: ha.now)
    assert THERMOSTAT in core.catalog.entities
    assert not set(legacy) & core.catalog.entities.keys()

async def test_unsupported_thermostat_mode_is_refused():
    result, _ = await run('switch to heat', THERMOSTAT, 'temperature_action', 'heat',
                          initial_state={THERMOSTAT: {'attributes': {'hvac_modes': ['off', 'cool']}}})
    assert not result.service_calls

async def test_temperature_query_uses_speaker_room_with_two_thermostats():
    other = 'climate.living_room_demo_thermostat'
    overlay = {'entities': [{'entity_id': other, 'friendly_name': 'Living Room thermostat', 'area_id': 'living_room'}],
               'states': {other: {'state': 'heat', 'attributes': {'temperature': 74, 'temperature_unit': '°F'}}}}
    ha = StubHA.default(overlays=[OVERLAY, overlay])
    core = await Core.create(load_settings(), ha, FakeJev({'intent': 'device_query', 'scope': 'unspecified',
        'device_type': 'thermostat', 'query_attribute': 'target_temperature'}), NullLuna(), clock=lambda: ha.now)
    result = await core.handle('what is it set to', speaker_area='living_room')
    assert '74' in result.reply and 'Living Room' in result.reply and not result.service_calls
