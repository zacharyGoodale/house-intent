"""HomeKit cleanup preserves unrelated selections and avoids whole-domain exposure."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('homekit_ptac', Path(__file__).parents[2] / 'scripts/configure_ptac_homekit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_migration_replaces_legacy_controls_and_preserves_unrelated():
    old = ['script.evening', 'lock.entry_lock', 'switch.espresso',
           'switch.example_bedroom_ptac_power', 'button.example_bedroom_ptac_temperature_up',
           'button.example_bedroom_ptac_temperature_down', 'number.example_living_room_ptac_target_temperature']
    result = module.thermostat_entities(old)
    assert set(result) == {'script.evening', 'lock.entry_lock', 'switch.espresso', *module.THERMOSTATS}
    assert not set(result) & module.LEGACY
    assert result == module.thermostat_entities(result)


def test_future_room_is_not_submitted_to_ha_until_it_exists():
    result = module.thermostat_entities(['switch.example_bedroom_ptac_power'], ['climate.bedroom_demo_thermostat'])
    assert result == ['climate.bedroom_demo_thermostat']
    assert {entity.split('.', 1)[0] for entity in result} == {'climate'}


def test_flow_defaults_preserve_existing_options():
    assert module.defaults({'data_schema': [{'name': 'devices', 'default': ['existing']},
                                            {'name': 'unselected'}]}) == {'devices': ['existing']}
