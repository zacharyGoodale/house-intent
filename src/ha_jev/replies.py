"""Reply templates (spec §8) and the response_mode mapping (decision 7)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ha_jev.models import Outcome, ResponseMode
from ha_jev.profiles import temperature_converter

TEMPLATES: dict[str, str] = {
    "R_ON": "Turning on {names}.",
    "R_OFF": "Turning off {names}.",
    "R_TOGGLE": "Toggling {names}.",
    "R_BRIGHTNESS": "Setting {names} to {v}%.",
    "R_BRIGHTEN": "Brightening {names}.",
    "R_DIM": "Dimming {names}.",
    "R_COLOR": "Setting {names} to {color}.",
    "R_COLOR_TEMP": "Setting {names} to {k} kelvin.",
    "R_WARMER": "Making {names} warmer.",
    "R_COOLER": "Making {names} cooler.",
    "R_COVER_OPEN": "Opening {name}.",
    "R_COVER_CLOSE": "Closing {name}.",
    "R_COVER_STOP": "Stopping {name}.",
    "R_COVER_POSITION": "Setting {name} to {v}%.",
    "R_MEDIA_PLAY": "Resuming {name}.",
    "R_MEDIA_PAUSE": "Pausing {name}.",
    "R_MEDIA_NEXT": "Skipping.",
    "R_MEDIA_PREV": "Going back.",
    "R_MEDIA_VOLUME": "Volume {v}% on {names}.",
    "R_MEDIA_UP": "Turning {names} up.",
    "R_MEDIA_DOWN": "Turning {names} down.",
    "R_MEDIA_MUTE": "Muting {names}.",
    "R_MEDIA_UNMUTE": "Unmuting {names}.",
    "R_MEDIA_SOURCE": "Opening {source}.",
    "R_MEDIA_ON": "Turning on {name}.",
    "R_MEDIA_OFF": "Turning off {name}.",
    "R_SCENE": "{name}.",
    "R_ADDED": "Added {item} to your {list}.",
    "R_LOCK": "For safety, I can't lock or unlock doors by voice. Use the lock's app or keypad.",
    "R_NO_DEVICE": "I couldn't find that device.",
    "R_NO_SCENE": "I couldn't find that scene.",
    "R_CANT": "{name} can't do that.",
    "R_RANGE": "That's out of range.",
    "R_COLOR_UNKNOWN": "I don't know that colour.",
    "R_PLAY": "I can't play specific music yet.",
    "R_TIMERS": "Tell me which timer or alarm to change and the new time.",
    "R_NO_SCHEDULES": "I can't set timers or alarms here yet; that needs the House Intent Home Assistant integration.",
    "R_SCHEDULE": "Done.",
    "R_EMPTY": "Sorry, I didn't catch that.",
    "R_WHICH_DEVICE": "Which one: {options}?",
    "R_WHICH_SCENE": "Which one: {options}?",
    "R_DONE": "Done: {names}.",
    "R_WHICH_ROOM": "Which room: {options}?",
    "R_CONFIRM": "Do you want me to {phrase}?",
    "R_WHAT": "Sorry, what would you like me to do?",
    "R_WHAT_LEVEL": "To what level?",
    "R_WHAT_ITEM": "What should I add?",
    "R_ONE_AT_A_TIME": "Sorry, I couldn't do all of that. Please ask one thing at a time.",
    "R_JEV_DOWN": "Sorry, I can't reach my decision service right now.",
    "R_LUNA_DOWN": "Sorry, I couldn't answer that right now.",
    "R_HA_FAIL": "Sorry, {names} didn't respond.",
    "R_OK": "Okay.",
    # music (music.py) and calendars (calendars.py)
    "R_PLAYING": "Playing {what}.",
    "R_PLAYING_SAY": "Playing {what}.",
    "R_PLAY_NOT_FOUND": "I couldn't find {what}.",
    "R_PLAY_ONE_ROOM": "I can only play music in one room at a time for now.",
    "R_NO_SPEAKER": "There's no speaker in the {area}.",
    "R_WHICH_MUSIC": "Which one: {options}?",
    "R_WHOSE_CALENDAR": "Whose calendar: {options}?",
    "R_NO_CALENDAR": "I don't have a calendar for {name}.",
    "R_CAL_WRITE": "I can't change calendars yet.",
    "R_CAL_WHEN": "I can check today, tomorrow, a day this week, or next week.",
    # status queries (§6.6)
    "Q_TEMPERATURE": "The room temperature at {name} is {value} {unit}.",
    "Q_TARGET_TEMPERATURE": "The {name} is set to {value} {unit}.",
    "Q_ROOM_TEMP_UNKNOWN": "I don’t have a room temperature reading for {name}; its target setting is not a measured temperature.",
    "Q_HVAC_MODE": "The {name} is in {mode} mode.",
    "Q_ON_OFF": "The {name} is {state}.",
    "Q_AREA_ALL_OFF": "All the {area} lights are off.",
    "Q_AREA_ALL_ON": "All the {area} lights are on.",
    "Q_AREA_SOME_ON": "On in the {area}: {names}.",
    "Q_HOUSE_ALL_OFF": "All the lights are off.",
    "Q_HOUSE_SOME_ON": "On right now: {names}.",
    "Q_BRIGHTNESS": "The {name} is at {pct}%.",
    "Q_LIGHT_OFF": "The {name} is off.",
    "Q_LOCKED": "The {name} is locked.",
    "Q_UNLOCKED": "The {name} is unlocked.",
    "Q_JAMMED": "The {name} is jammed.",
    "Q_OPEN": "The {name} is open.",
    "Q_CLOSED": "The {name} is closed.",
    "Q_COVER_PCT": "The {name} is {n}% open.",
    "Q_MOTION_YES": "There's motion in the {area}.",
    "Q_MOTION_NO": "No motion in the {area}.",
    "Q_POWER": "The {name} is drawing {w} watts.",
    "Q_NEXT_ALARM": "Your next {kind} is {day} at {time}.",
    "Q_NO_ALARM": "You don't have {article} {kind} set.",
    "Q_SUN": "{event} is at {time}{suffix}.",
    "Q_PLAYING": "The {name} is playing {title}{by}.",
    "Q_PAUSED": "The {name} is paused.",
    "Q_NOTHING_PLAYING": "Nothing is playing on the {name}.",
    "Q_VOLUME": "The {name} is at {pct}% volume.",
    "Q_UNKNOWN": "I don't know the {name}'s state right now.",
    # --- memory (memory.py, SPEC §17) ---
    "R_REMEMBERED": "I'll remember that {fact}.",
    "R_FORGOT": "Okay, I've forgotten that.",
    "R_MEMORY_LIST": "{text}",
    "R_MEMORY_EMPTY": "I'm not remembering anything yet.",
    "R_MEMORY_WHAT": "What should I remember?",
    "R_MEMORY_WHO": "I can't tell who's speaking, so I can't remember that for you.",
    "R_MEMORY_SECRET": "I can't remember passwords, codes or other secrets.",
    "R_MEMORY_REFUSE": "Sorry, I can't remember that.",
    "R_MEMORY_REMINDER": "I can't set reminders yet.",
    "R_MEMORY_TOO_LONG": "That's too long for me to remember. Try a shorter sentence.",
    "R_MEMORY_FULL": "My memory is full. Ask me to forget something first.",
    "R_MEMORY_NOT_FOUND": "I couldn't find that in what I remember.",
    "R_MEMORY_ONE": "I can only forget one thing at a time.",
    "R_MEMORY_FAIL": "Sorry, I couldn't update my memory.",
    "R_MEMORY_OFF": "My memory is turned off.",
}

# Answers that are acknowledged with the success chime, and refusals that must be spoken so the user knows why.
CHIME_ANSWERS = {"R_OK", "R_FORGOT"}
SPOKEN_REFUSALS = {"R_LOCK", "R_MEMORY_WHO", "R_MEMORY_SECRET", "R_MEMORY_REFUSE", "R_MEMORY_TOO_LONG",
                   "R_MEMORY_FULL"}

CONFIRM_PHRASES = {
    "R_ON": "turn on {names}", "R_OFF": "turn off {names}", "R_TOGGLE": "toggle {names}",
    "R_BRIGHTNESS": "set {names} to {v}%", "R_BRIGHTEN": "brighten {names}", "R_DIM": "dim {names}",
    "R_COLOR": "set {names} to {color}", "R_COLOR_TEMP": "set {names} to {k} kelvin",
    "R_WARMER": "make {names} warmer", "R_COOLER": "make {names} cooler",
    "R_COVER_OPEN": "open {name}", "R_COVER_CLOSE": "close {name}", "R_COVER_STOP": "stop {name}",
    "R_COVER_POSITION": "set {name} to {v}%",
    "R_MEDIA_PLAY": "resume {name}", "R_MEDIA_PAUSE": "pause {name}", "R_MEDIA_NEXT": "skip",
    "R_MEDIA_PREV": "go back", "R_MEDIA_VOLUME": "set the volume to {v}% on {names}",
    "R_MEDIA_UP": "turn {names} up", "R_MEDIA_DOWN": "turn {names} down", "R_MEDIA_MUTE": "mute {names}",
    "R_MEDIA_UNMUTE": "unmute {names}", "R_MEDIA_SOURCE": "open {source}", "R_MEDIA_ON": "turn on {name}",
    "R_MEDIA_OFF": "turn off {name}", "R_SCENE": "activate {name}", "R_ADDED": "add {item} to your {list}",
}

CONDITIONS = {"partlycloudy": "partly cloudy", "clear-night": "clear", "rainy": "rain", "pouring": "heavy rain",
              "lightning": "thunderstorms", "lightning-rainy": "thunderstorms", "snowy-rainy": "sleet",
              "exceptional": "severe weather"}


class NoData(ValueError):
    """HA answered a read, but without the data a reply needs (no forecast rows): reported as R_HA_FAIL."""


class _Default(dict):
    def __missing__(self, key: str) -> str:
        return ""


def join_names(names: Sequence[str], conj: str = "and") -> str:
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} {conj} {names[-1]}"


# Which-prompts with nothing to offer (review S9): never "Which one: ?".
_NO_OPTIONS = {"R_WHICH_DEVICE": "Which device do you mean?", "R_WHICH_SCENE": "Which scene do you mean?",
               "R_WHICH_ROOM": "Which room?"}


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def _day_word(dt: datetime, now: datetime) -> str:
    delta = (dt.date() - now.date()).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return dt.strftime("%A")


def _as_dt(v: Any) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def _time_params(key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Wording for times lives here, not in policy (standards review, divergent change)."""
    if key in ("Q_SUN", "Q_NEXT_ALARM") and "at" in params:
        at, now = _as_dt(params["at"]), _as_dt(params["now"])
        day = _day_word(at, now)
        params["time"] = _fmt_time(at)
        params["day"] = day
        params["suffix"] = "" if day == "today" else (" tomorrow" if day == "tomorrow" else f" on {day}")
    if key == "Q_NO_ALARM":
        params["article"] = "an" if str(params.get("kind", "x"))[:1] in "aeiou" else "a"
    return params


def render_template(key: str, params: Mapping[str, Any] | None = None) -> str:
    params = dict(params or {})
    if key in ("R_COMPOUND", "Q_MULTI"):
        return " ".join(render_template(k, p) for k, p in params.get("parts", []))
    if key in _NO_OPTIONS and not params.get("options"):
        return _NO_OPTIONS[key]
    params = _time_params(key, params)
    if key == "Q_PLAYING":
        params["by"] = f" by {params['artist']}" if params.get("artist") else ""
    template = TEMPLATES.get(key, TEMPLATES["R_WHAT"])
    text = template.format_map(_Default(params))
    return text[0].upper() + text[1:] if text else text


def confirm_phrase(key: str, params: Mapping[str, Any]) -> str:
    template = CONFIRM_PHRASES.get(key, "do that")
    return template.format_map(_Default(dict(params)))


# Music replies that are spoken although they would chime (OQ 104): a guessed match says what it chose, so a wrong
# guess is obvious, and a miss says what wasn't found.
SPOKEN = {"R_PLAYING_SAY", "R_PLAY_NOT_FOUND"}


def response_mode(outcome: Outcome, reply_key: str) -> ResponseMode:
    if reply_key in SPOKEN and outcome in (Outcome.EXECUTED, Outcome.UNSUPPORTED):
        return ResponseMode.SPEECH
    if outcome == Outcome.EXECUTED:
        return ResponseMode.CHIME_SUCCESS
    if outcome == Outcome.ANSWERED:
        return ResponseMode.CHIME_SUCCESS if reply_key in CHIME_ANSWERS else ResponseMode.SPEECH
    if outcome == Outcome.CLARIFY:
        return ResponseMode.SPEECH
    if outcome == Outcome.REFUSED and reply_key in SPOKEN_REFUSALS:
        return ResponseMode.SPEECH
    return ResponseMode.CHIME_ERROR


# ---- weather and list replies that need reads -------------------------------------------------
def _cond(c: str | None) -> str:
    c = c or "unknown"
    return CONDITIONS.get(c, c.replace("-", " "))


def _num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "?"
    return str(int(round(f)))


def _dt(value: str, tz: Any) -> datetime:
    d = datetime.fromisoformat(value)
    return d.astimezone(tz) if tz and d.tzinfo else d


def render_weather(key: str, params: Mapping[str, Any], state_attrs: Mapping[str, Any], state: str | None,
                   forecast: list[dict] | None, now: datetime, temperature_unit: str | None = None) -> str:
    """`temperature_unit` is the speaker's preference ("C" | "F", profiles.py); None keeps the entity's unit."""
    conv, unit = temperature_converter(state_attrs.get("temperature_unit", ""), temperature_unit)

    def _t(v: Any) -> str:  # a temperature, in the speaker's unit
        try:
            return _num(conv(float(v)))
        except (TypeError, ValueError):
            return "?"

    if key == "W_NOW":
        return (f"It's {_t(state_attrs.get('temperature'))}{unit} and {_cond(state)}, "
                f"humidity {_num(state_attrs.get('humidity'))}%.")
    fc = forecast or []
    when = params.get("when")
    if key == "W_NEXT_HOURS":
        horizon = [f for f in fc if now < _dt(f["datetime"], now.tzinfo) <= now.replace() + _hours(6)]
        if not horizon:
            raise NoData("no hourly forecast")
        conds = list(dict.fromkeys(_cond(f.get("condition")) for f in horizon))
        temps = [float(f["temperature"]) for f in horizon if f.get("temperature") is not None]  # _t converts
        p = max(int(f.get("precipitation_probability") or 0) for f in horizon)
        return (f"Next few hours: {join_names(conds)}, {_t(min(temps))}–{_t(max(temps))}{unit}, "
                f"up to {p}% chance of rain.")
    if key == "W_DAY":
        target = now.date() if when == "today" else (now + _hours(24)).date()
        for f in fc:
            if _dt(f["datetime"], now.tzinfo).date() == target:
                word = "Today" if when == "today" else "Tomorrow"
                probability = f.get("precipitation_probability")
                precipitation = f.get("precipitation")
                if probability is not None:
                    rain = f"{float(probability):g}% chance of rain"
                elif f.get("condition") in ("rainy", "pouring", "lightning-rainy", "snowy-rainy") or (precipitation is not None and float(precipitation) > 0):
                    rain = "precipitation is forecast" if f.get("condition") == "snowy" else "rain is forecast"
                else:
                    rain = "no rain forecast" if precipitation is not None else "rain chance unavailable"
                current = (f"It's {_t(state_attrs['temperature'])}{unit} now. "
                           if params.get("with_now") and state_attrs.get("temperature") is not None else "")
                return (f"{current}{word}: {_cond(f.get('condition'))}, high {_t(f.get('temperature'))}, "
                        f"low {_t(f.get('templow'))}, {rain}.")
        raise NoData("no daily forecast for that day")
    if key == "W_TONIGHT":
        rows = [f for f in fc if now <= _dt(f["datetime"], now.tzinfo) and _dt(f["datetime"], now.tzinfo).date() == now.date()]
        if not rows:
            raise NoData("no hourly forecast tonight")
        majority = Counter(_cond(f.get("condition")) for f in rows).most_common(1)[0][0]
        avg = sum(float(f["temperature"]) for f in rows) / len(rows)
        p = max(int(f.get("precipitation_probability") or 0) for f in rows)
        return f"Tonight: {majority}, around {_t(avg)}, up to {p}% chance of rain."
    if key == "W_WEEK":
        rows = [f for f in fc if _dt(f["datetime"], now.tzinfo).date() >= now.date()][:5]
        if not rows:
            raise NoData("no daily forecast")
        parts = [f"{_dt(f['datetime'], now.tzinfo).strftime('%A')} {_cond(f.get('condition'))} {_t(f.get('temperature'))}"
                 for f in rows]
        return "; ".join(parts) + "."
    raise ValueError(f"unknown weather key {key}")


def _hours(n: int):
    from datetime import timedelta

    return timedelta(hours=n)


def render_list(list_name: str, items: list[dict]) -> str:
    open_items = [i.get("summary", "") for i in items if i.get("status", "needs_action") == "needs_action"]
    name = list_name.lower()
    if not open_items:
        return f"Your {name} is empty."
    return f"Your {name} has {join_names(open_items)}."
