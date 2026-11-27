"""Luna instructions: persona, household profile and the home summary come from config and the catalog."""

from datetime import datetime

from ha_jev.config import LunaSettings
from ha_jev.luna import build_instructions

NOW = datetime.fromisoformat("2037-09-22T18:00:00-04:00")


def test_persona_household_and_home_are_included_as_sentences():
    settings = LunaSettings(persona="Be cheerful", household="Two people live here, in Springfield")
    text = build_instructions(NOW, "Den", settings, "By voice, this home can control lights.")
    assert "Be cheerful. " in text
    assert "Two people live here, in Springfield. " in text
    assert "By voice, this home can control lights. Local time: Tuesday 2037-09-22 18:00. The user is in the Den." in text


def test_empty_profile_leaves_no_gaps():
    text = build_instructions(NOW, None, LunaSettings(persona="", household=""), None)
    assert "  " not in text
    assert text.endswith("The user is in the home.")
