import pytest

from ha_jev.extract import extract, normalize


@pytest.mark.parametrize("raw,expected", [
    ("  hey jev, turn   on the lights please ", "turn on the lights"),
    ("Hi Jev turn off Alder thanks", "turn off Alder"),
    ("what's the weather?", "what's the weather?"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_numbers_with_units_and_ids():
    x = extract("dim the bedroom lights to 20%")
    assert x.numbers == [{"id": "n0", "text": "20%", "value": 20, "unit": "percent"}]


def test_kelvin_and_bare_numbers():
    x = extract("set the kitchen lights to 4000 kelvin")
    assert x.numbers[0]["value"] == 4000 and x.numbers[0]["unit"] == "kelvin"
    x = extract("set lamp 2 to 3000k")
    assert [(n["value"], n["unit"]) for n in x.numbers] == [(2, None), (3000, "kelvin")]


def test_number_words():
    x = extract("set the volume to twenty five percent")
    assert x.numbers[0]["value"] == 25 and x.numbers[0]["unit"] == "percent"
    assert extract("set the bedroom to one hundred percent").numbers[0]["value"] == 100
    assert extract("volume thirty").numbers[0]["value"] == 30


def test_at_most_three_numbers():
    assert len(extract("1 2 3 4 5").numbers) == 3


def test_decimal_numbers():
    assert extract("set it to 2.5 percent").numbers[0]["value"] == 2.5


@pytest.mark.parametrize("text,parts", [
    ("turn off the lights and pause the music", ["turn off the lights", "pause the music"]),
    ("turn off the lights, then close the blinds", ["turn off the lights", "close the blinds"]),
    ("turn off the kitchen and bedroom lights", []),
    ("turn on alder and the paper lantern", []),
    ("dim the lights and then play morning brief", ["dim the lights", "play morning brief"]),
])
def test_compound_split(text, parts):
    assert extract(text).parts == parts


def test_more_than_three_parts_is_flagged():
    x = extract("turn on the lights and close the blinds and pause the music and mute the tv")
    assert x.too_many_parts


@pytest.mark.parametrize("text,hit", [
    ("unlock the front door", True),
    ("lock the door", True),
    ("let me in", True),
    ("open the garage", True),
    ("disarm the alarm", True),
    ("is the front door locked?", False),
    ("is it unlocked", False),
    ("turn on the lights", False),
])
def test_lock_keywords(text, hit):
    assert extract(text).lock_keyword is hit


@pytest.mark.parametrize("text,q", [
    ("is the front door locked", True),
    ("what's playing on the stereo", True),
    ("when's sunset", True),
    ("the lights are on?", True),
    ("turn on the lights", False),
])
def test_question_form(text, q):
    assert extract(text).question_form is q


def test_list_item_keeps_original_case():
    x = extract("add Oat Milk to the shopping list")
    assert x.list_item == {"item": "Oat Milk", "list": "shopping"}
    assert extract("put eggs on my list").list_item == {"item": "eggs", "list": ""}
    assert extract("turn on the lights").list_item is None


def test_empty():
    assert extract("  hey jev  ").text == ""
