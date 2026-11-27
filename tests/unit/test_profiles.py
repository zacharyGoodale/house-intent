"""Per-person profiles (profiles.py, SPEC §17): who is speaking, units, and the Luna person sentence."""

import shutil
from pathlib import Path

import pytest

from ha_jev.config import ConfigError, load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.luna import LunaProfile, NullLuna, build_instructions
from ha_jev.profiles import speaker_for, speaker_sentence, temperature_converter
from ha_jev.replies import render_weather
from tests.fakes import FakeJev
from tests.unit.helpers import NOW

ROOT = Path(__file__).resolve().parents[2]


def test_speaker_from_person_id_or_conversation_agent():
    people = load_settings().people
    assert speaker_for(people, person_id="morgan").name == "Morgan"
    assert speaker_for(people, agent_id=people["riley"].agents[0]).id == "riley"
    assert speaker_for(people, agent_id="conversation.somebody_else") is None
    assert speaker_for(people) is None and speaker_for(people, person_id="nobody") is None


def _config_copy(tmp_path, people_toml: str):
    for name in ("scope.toml", "safety.toml", "ha_jev.toml"):
        shutil.copy(ROOT / "config" / name, tmp_path / name)
    (tmp_path / "people.toml").write_text(people_toml)


def test_an_agent_maps_to_one_person_and_household_is_reserved(tmp_path):
    _config_copy(tmp_path, '[people.a]\nname = "A"\nagents = ["x"]\n[people.b]\nname = "B"\nagents = ["x"]\n')
    with pytest.raises(ConfigError):
        load_settings(config_dir=tmp_path)
    _config_copy(tmp_path, '[people.household]\nname = "H"\n')
    with pytest.raises(ConfigError):
        load_settings(config_dir=tmp_path)


def test_speaker_sentence():
    people = load_settings().people
    assert speaker_sentence(people["morgan"], people) == "You are talking to Morgan. Give temperatures in Celsius."
    assert "can't tell" in speaker_sentence(None, people)
    assert speaker_sentence(None, {}) == ""


def test_household_text_no_longer_claims_the_speaker_is_unknown():
    assert "can't tell" not in load_settings().luna.household


def test_profile_blocks_sit_after_home_and_before_the_clock():
    text = build_instructions(NOW, "Den", None, "Home.", LunaProfile(memory="MEM. ", speaker="SPK. "))
    assert text.index("Home.") < text.index("MEM.") < text.index("SPK.") < text.index("Local time")
    assert "  " not in build_instructions(NOW, None, None, None, LunaProfile())


@pytest.mark.parametrize("have,want,value,out,label", [
    ("°F", "C", 66, 18.89, "°C"), ("°C", "F", 20, 68, "°F"), ("°F", "F", 66, 66, "°F"), ("°F", None, 66, 66, "°F"),
    ("K", "C", 300, 300, "K"),
])
def test_temperature_converter(have, want, value, out, label):
    conv, unit = temperature_converter(have, want)
    assert round(conv(value), 2) == out and unit == label


def test_weather_is_converted_for_the_speaker():
    attrs = {"temperature": 66, "humidity": 62, "temperature_unit": "°F"}
    assert render_weather("W_NOW", {}, attrs, "partlycloudy", None, NOW, "C") == \
        "It's 19°C and partly cloudy, humidity 62%."
    assert render_weather("W_NOW", {}, attrs, "partlycloudy", None, NOW) == "It's 66°F and partly cloudy, humidity 62%."
    fc = [{"datetime": "2037-09-23T12:00:00-04:00", "condition": "rainy", "temperature": 68, "templow": 50,
           "precipitation_probability": 80}]
    assert render_weather("W_DAY", {"when": "tomorrow"}, attrs, None, fc, NOW, "C") == \
        "Tomorrow: rain, high 20, low 10, 80% chance of rain."


@pytest.mark.parametrize("person,expected", [("morgan", "°C"), ("riley", "°F"), (None, "°F")])
async def test_core_weather_uses_the_speakers_unit(person, expected):
    ha = StubHA.default()
    core = await Core.create(load_settings(), ha, FakeJev({"intent": "weather", "weather_when": "now"}), NullLuna(),
                             clock=lambda: ha.now)
    r = await core.handle("what's the weather", person=person)
    assert expected in r.reply
