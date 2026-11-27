"""Per-person profiles (SPEC §17): who is speaking, and the typed preferences that change what we say.

The speaker is a configured person id (config/people.toml) or None. It is known by convention, not proven:
each person has their own wake word, pipeline and HA conversation agent (docs/usage.md), so the
HA side sends either the person id or the agent id and `speaker_for` maps it. Identity never reaches the
safety guard or Jev's state; it only picks preferences, the Luna person sentence and memory scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ha_jev.config import Person

UNITS = {"°F": "F", "F": "F", "°C": "C", "C": "C"}


def speaker_for(people: Mapping[str, Person], *, person_id: str | None = None,
                agent_id: str | None = None) -> Person | None:
    """The configured person for an explicit id, else for the conversation agent that took the request."""
    if person_id:
        return people.get(person_id)
    if agent_id:
        for member in people.values():
            if agent_id in member.agents:
                return member
    return None


def speaker_sentence(speaker: Person | None, people: Mapping[str, Person]) -> str:
    """One sentence for Luna built from typed fields only, never free text (personal-context.md §3.2)."""
    if not people:
        return ""
    if speaker is None:
        return "You can't tell which of them is speaking, and it may be a guest."
    unit = {"C": " Give temperatures in Celsius.", "F": " Give temperatures in Fahrenheit."}.get(
        speaker.temperature_unit or "", "")
    return f"You are talking to {speaker.name}.{unit}"


# ---- temperature units ------------------------------------------------------------------------
def temperature_converter(entity_unit: str | None, wanted: str | None) -> tuple[Callable[[float], float], str]:
    """(convert, unit label) from the weather entity's unit to the speaker's. No preference, or an unknown
    entity unit, leaves values as they are."""
    have = UNITS.get((entity_unit or "").strip())
    label = entity_unit or ""
    if wanted is None or have is None or have == wanted:
        return (lambda v: v), label
    if wanted == "C":
        return (lambda v: (v - 32) * 5 / 9), "°C"
    return (lambda v: v * 9 / 5 + 32), "°F"
