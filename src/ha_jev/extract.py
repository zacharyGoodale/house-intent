"""Code-side extraction before Jev (§5.3): normalize, numbers, compound split, list item, question form, lock keywords."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ha_jev.safety import has_lock_keyword

_WAKE = re.compile(r"^(?:(?:hey|hi|ok|okay)\s+(?:jev|nabu))\b[\s,.!]*", re.IGNORECASE)
_TRAILING = re.compile(r"[\s,]*\b(?:please|thanks|thank you)\b[\s.!]*$", re.IGNORECASE)

_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,5}(?:\.\d+)?)\s*(%|percent\b|per cent\b|degrees?\s*(?:celsius|fahrenheit|[cf])\b|°\s*[cf]\b|celsius\b|fahrenheit\b|degrees?\b|°|kelvin\b|k\b)?", re.IGNORECASE)

_UNITS = {"%": "percent", "percent": "percent", "per cent": "percent", "degree": "degrees", "degrees": "degrees",
          "°": "degrees", "celsius": "C", "fahrenheit": "F", "°c": "C", "°f": "F", "° c": "C", "° f": "F",
          "degree c": "C", "degrees c": "C", "degree f": "F", "degrees f": "F",
          "degree celsius": "C", "degrees celsius": "C", "degree fahrenheit": "F", "degrees fahrenheit": "F", "kelvin": "kelvin", "k": "kelvin"}

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORD_NUMBER = re.compile(
    r"\b(?:(?:one|a)\s+hundred|(?:" + "|".join(_TENS) + r")(?:[\s-]+(?:" + "|".join(k for k in _ONES if _ONES[k] < 10 and k != "zero") + r"))?|"
    + "|".join(sorted(_ONES, key=len, reverse=True)) + r")\b"
    r"(\s*(?:%|percent\b|per cent\b|degrees?\s*(?:celsius|fahrenheit|[cf])\b|celsius\b|fahrenheit\b|degrees?\b|kelvin\b))?",
    re.IGNORECASE,
)

SPLIT_VERBS = {"turn", "switch", "set", "dim", "brighten", "open", "close", "draw", "pause", "play", "resume", "stop",
               "skip", "mute", "unmute", "start", "activate", "put", "make", "raise", "lower", "lock", "unlock", "add",
               "change", "shut"}
_SEPARATOR = re.compile(r"\s*,?\s+(?:and then|then|and also|and|also)\s+|\s*;\s*", re.IGNORECASE)

_QUESTION_START = re.compile(
    r"^(?:is|are|was|were|did|do|does|has|have|what|what's|whats|when|when's|whens|where|which|who|how|will)\b",
    re.IGNORECASE)

_LIST_ITEM = re.compile(
    r"^(?:please\s+)?(?:add|put)\s+(?P<item>.+?)\s+(?:to|on|onto)\s+(?:the\s+|my\s+)?(?P<list>[\w\s-]*?)\s*list\b",
    re.IGNORECASE)

# A pronoun standing for a device ("toggle it", "turn them off"): with nothing recent it must not become a
# room sweep (review S4).
_ANAPHOR = re.compile(r"\b(?:it|them|those|these|that one|this one)\b", re.IGNORECASE)

# A whole-home quantifier over devices ("all the lights", "everything", "any lights", "everywhere"). Policy uses it
# only when no room is named (round 2, class 1; spec §5.4 house: "all devices of a kind in the whole home").
_QUANTIFIER = re.compile(r"\b(?:all|every|everything|everywhere|any|whole (?:house|home|place))\b", re.IGNORECASE)

MAX_PARTS = 3
MAX_NUMBERS = 3


@dataclass
class Extraction:
    original: str
    text: str
    numbers: list[dict] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    part_numbers: list[list[int]] = field(default_factory=list)  # indices into numbers, per part
    too_many_parts: bool = False
    lock_keyword: bool = False
    question_form: bool = False
    list_item: dict | None = None
    anaphor: bool = False
    quantifier: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    t = _WAKE.sub("", t).strip()
    t = _TRAILING.sub("", t).strip()
    t = t.rstrip(".!").strip()
    return t


def _word_value(words: str) -> int:
    w = re.sub(r"[\s-]+", " ", words.casefold()).strip()
    if w.endswith("hundred"):
        return 100
    total = 0
    for tok in w.split():
        total += _TENS.get(tok, 0) + _ONES.get(tok, 0)
    return total


def _numbers(text: str) -> list[tuple[int, dict]]:
    found: list[tuple[int, int, dict]] = []
    for m in _NUMBER.finditer(text):
        raw = m.group(1)
        value: float | int = float(raw)
        if value.is_integer():
            value = int(value)
        unit = _UNITS.get((m.group(2) or "").strip().casefold()) if m.group(2) else None
        found.append((m.start(), m.end(), {"text": m.group(0).strip(), "value": value, "unit": unit}))
    for m in _WORD_NUMBER.finditer(text):
        if any(s <= m.start() < e for s, e, _ in found):
            continue
        num_text = m.group(0)[: len(m.group(0)) - len(m.group(1) or "")]
        unit_raw = (m.group(1) or "").strip().casefold()
        unit = _UNITS.get(unit_raw) if unit_raw else None
        found.append((m.start(), m.end(), {"text": m.group(0).strip(), "value": _word_value(num_text), "unit": unit}))
    found.sort(key=lambda t: t[0])
    out = []
    for i, (start, _end, n) in enumerate(found[:MAX_NUMBERS]):
        out.append((start, {"id": f"n{i}", **n}))
    return out


def _split(text: str) -> tuple[list[tuple[int, str]], bool]:
    pieces: list[tuple[int, str]] = []
    start = 0
    for m in _SEPARATOR.finditer(text):
        rest = text[m.end():]
        first = re.match(r"[a-z]+", rest, re.IGNORECASE)
        if not first or first.group(0).casefold() not in SPLIT_VERBS:
            continue
        pieces.append((start, text[start:m.start()].strip()))
        start = m.end()
    pieces.append((start, text[start:].strip()))
    pieces = [(s, p) for s, p in pieces if p]
    return pieces, len(pieces) > MAX_PARTS


def is_question(text: str) -> bool:
    return bool(_QUESTION_START.match(text)) or text.endswith("?")


def extract(original: str) -> Extraction:
    text = normalize(original)
    x = Extraction(original=original, text=text)
    if not text:
        return x
    located = _numbers(text)
    x.numbers = [n for _, n in located]
    pieces, too_many = _split(text)
    x.too_many_parts = too_many
    if len(pieces) >= 2 and not too_many:
        x.parts = [p for _, p in pieces]
        starts = [s for s, _ in pieces] + [len(text) + 1]
        x.part_numbers = [[i for i, (pos, _n) in enumerate(located) if starts[k] <= pos < starts[k + 1]]
                          for k in range(len(pieces))]
    x.lock_keyword = has_lock_keyword(text)
    x.question_form = is_question(text)
    x.anaphor = bool(_ANAPHOR.search(text))
    x.quantifier = bool(_QUANTIFIER.search(text))
    m = _LIST_ITEM.match(text)
    if m:
        x.list_item = {"item": m.group("item").strip(), "list": m.group("list").strip()}
    return x
