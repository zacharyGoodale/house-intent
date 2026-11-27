"""Calendar questions: "what's on my calendar today", "am I free at 3", "what's Morgan doing tomorrow"
(docs/usage.md, phase A plus the per-person wake word as the speaker).

Jev gets a `calendar` intent and three gated Choices in the same single request (decision 1): whose calendar,
which time and what kind of question. `decide` (pure, called from policy) resolves the owner from the known
speaker and config/people.toml, applies the owner's sharing flag and plans a `calendar.get_events` read; core
performs it (through a short cache) and `render` answers from a template. No model ever sees event text: calendars
are out of the voice scope, and titles are sanitised and spoken as data (personal-context.md §5).
"""

from __future__ import annotations

import fnmatch
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ha_jev import capabilities as caps
from ha_jev.jev import ChoiceAnswer
from ha_jev.models import ClarifySpec, Decision, Outcome, PlanItem, ReadCall
from ha_jev.replies import join_names

if TYPE_CHECKING:
    from ha_jev.catalog import Catalog
    from ha_jev.config import Person, PolicySettings
    from ha_jev.questions import AnswerView

REPLY_KEY = "CAL_ANSWER"
INTENT = "calendar"
WHOSE, WHEN, ASK = "calendar_whose", "calendar_when", "calendar_ask"
MAX_SPOKEN = 6  # events read out, then "and N more"
TITLE_MAX = 80
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


# ---- questions ---------------------------------------------------------------------------------
def with_calendars(people: Mapping[str, Person]) -> dict[str, Person]:
    return {pid: p for pid, p in people.items() if p.calendars}


def intent_option() -> dict[str, Any]:
    return {"what": "Ask about someone's calendar or schedule: what's on, appointments, meetings, plans, what someone "
                    "is doing on a day, when the next event is, whether someone is free or busy at a time.",
            "not_for": "restaurants, dining reservations, booking tables, Resy, Amooseboosh (Amos Bush), reservation jobs; "
                       "alarms, timers, sunrise or sunset, the weather, lists, or the state of a device"}


def questions(u: str, people: Mapping[str, Person]) -> dict[str, dict]:
    """The calendar Choices, only when someone has a calendar configured. Names come from config (decision 9)."""
    owners = with_calendars(people)
    if not owners:
        return {}
    whose: dict[str, Any] = {"speaker": "the person speaking: my, me, I, am I"}
    for pid, p in owners.items():
        whose[pid] = f"{p.name}'s calendar: {p.name}, him, her, them"
    whose["everyone"] = "everyone together: our, us, we, both of us, the household"
    whose["none"] = "no one in particular: 'the calendar', 'anything on today'"
    return {
        WHOSE: {"type": "choice", "instructions": f"Whose calendar does {u} ask about?", "criteria": whose},
        WHEN: {"type": "choice", "instructions": f"Which time does the calendar question in {u} ask about?",
               "criteria": {"today": "today, this morning, this afternoon, later today", "tonight": None,
                            "tomorrow": None, "this_week": "this week, the next few days",
                            "weekend": "this weekend, Saturday and Sunday together",
                            "next": "the next event, whenever it is", "other": "a named day or date, or next week",
                            "none": "no time given"}},
        ASK: {"type": "choice", "instructions": f"What does {u} want to know about the calendar?",
              "criteria": {"list": "what's on, what someone is doing, the schedule",
                           "next": "the next or upcoming event", "count": "how busy, how many things",
                           "free_at": "whether someone is free or busy at, before or after a time",
                           "change": "add, move or cancel an event or appointment",
                           "other": "something else about the events"}},
    }


# ---- times -------------------------------------------------------------------------------------
_TIME = re.compile(r"\b(?:at|by|around|from|until|till|before|after)\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?"
                   r"\s*(?P<ap>a\.?m\.?|p\.?m\.?)?(?![\w:])|\b(?P<h2>\d{1,2})(?::(?P<m2>\d{2}))?\s*(?P<ap2>a\.?m\.?|p\.?m\.?)"
                   r"|\b(?P<word>noon|midday|midnight)\b", re.IGNORECASE)
_WORD_HOURS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
               "ten": 10, "eleven": 11, "twelve": 12}
_TIME_WORD = re.compile(r"\b(?:at|by|around)\s+(?P<w>" + "|".join(_WORD_HOURS) + r")(?:\s+o'?clock)?"
                        r"(?:\s*(?P<ap>a\.?m\.?|p\.?m\.?|in the (?:morning|afternoon|evening)|tonight))?\b", re.IGNORECASE)


def parse_time(text: str) -> tuple[int, int] | None:
    """A clock time said in the utterance, as (hour, minute). With no am/pm, 1-7 is afternoon (a household
    rarely asks about 3 in the morning) and 8-11 is morning."""
    m = _TIME.search(text)
    hour = minute = None
    ap = None
    if m and m.group("word"):
        return (0, 0) if m.group("word").casefold() == "midnight" else (12, 0)
    if m:
        hour = int(m.group("h") or m.group("h2"))
        minute = int(m.group("m") or m.group("m2") or 0)
        ap = (m.group("ap") or m.group("ap2") or "").casefold()
    else:
        w = _TIME_WORD.search(text)
        if not w:
            return None
        hour, minute = _WORD_HOURS[w.group("w").casefold()], 0
        ap = (w.group("ap") or "").casefold()
        ap = "a" if "morning" in ap else ("p" if ap and ap[0] in "ipt" else ap)
    if hour is None or hour > 23 or minute > 59:
        return None
    if ap.startswith("p") and hour < 12:
        hour += 12
    elif ap.startswith("a") and hour == 12:
        hour = 0
    elif not ap and 1 <= hour <= 7:
        hour += 12
    return hour, minute


def _day_start(d: date, tz: Any) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=tz)


def window(when: str, text: str, now: datetime) -> tuple[datetime, datetime, str] | None:
    """(start, end, spoken label) for a calendar_when answer, or None for a day that can't be read."""
    tz = now.tzinfo
    today = _day_start(now.date(), tz)
    low = text.casefold()
    if when in ("today", "none"):
        return today, today + timedelta(days=1), "today"
    if when == "tonight":
        return max(now, today + timedelta(hours=17)), today + timedelta(days=1), "tonight"
    if when == "tomorrow":
        return today + timedelta(days=1), today + timedelta(days=2), "tomorrow"
    if when == "this_week":
        return now, today + timedelta(days=7 - now.weekday()), "this week"
    if when == "weekend":
        sat = today + timedelta(days=(5 - now.weekday()) % 7)
        if now.weekday() == 6:
            sat = today - timedelta(days=1)
        return max(now, sat), sat + timedelta(days=2), "this weekend"
    if when == "next":
        return now, now + timedelta(days=7), "this week"
    if "next week" in low:
        start = today + timedelta(days=7 - now.weekday())
        return start, start + timedelta(days=7), "next week"
    for i, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", low):
            ahead = (i - now.weekday()) % 7
            if ahead == 0 and re.search(rf"\bnext {name}\b", low):
                ahead = 7
            day = today + timedelta(days=ahead)
            label = "today" if ahead == 0 else ("tomorrow" if ahead == 1 else f"on {name.capitalize()}")
            return day, day + timedelta(days=1), label
    if re.search(r"\btomorrow\b", low):
        return window("tomorrow", text, now)
    if re.search(r"\btonight\b", low):
        return window("tonight", text, now)
    if re.search(r"\btoday\b", low):
        return window("today", text, now)
    return None


# ---- policy (pure) -----------------------------------------------------------------------------
def _p(a: ChoiceAnswer) -> float:
    return float(a.probabilities.get(a.choice, 0.0)) if a.probabilities else float(a.confidence)


def entities_for(member: Person, cat: Catalog) -> tuple[str, ...]:
    """The person's calendar entities; globs in people.toml match the registry (calendars are never in scope)."""
    known = set(cat.registry_names)
    out: list[str] = []
    for pattern in member.calendars:
        hits = sorted(e for e in known if fnmatch.fnmatchcase(e, pattern)) if any(c in pattern for c in "*?[") \
            else [pattern]
        out += [h for h in hits if h not in out]
    return tuple(out)


@dataclass(frozen=True)
class _Owner:
    id: str
    name: str
    share: str
    entities: tuple[str, ...]
    self: bool  # the owner is the known speaker (or said who they are in a follow-up)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "share": self.share, "entities": list(self.entities),
                "self": self.self}


def _answer(params: dict[str, Any], owners: Sequence[_Owner], route: str, conf: float | None) -> Decision:
    params = {**params, "owners": [o.as_dict() for o in owners]}
    ids = tuple(e for o in owners if o.share != "private" or o.self for e in o.entities)
    reads = (ReadCall("service", *caps.CALENDAR_EVENTS_SERVICE, ids,
                      {"start_date_time": params["start"], "end_date_time": params["end"]}),) if ids else ()
    return Decision(outcome=Outcome.ANSWERED, reads=reads, reply_key=REPLY_KEY, reply_params=params, route=route,
                    rule="8", confidence=conf)


def decide(cat: Catalog, text: str, av: AnswerView, pol: PolicySettings, now: datetime,
           people: Mapping[str, Person], speaker: Person | None, intent_p: float) -> Decision:
    owners_cfg = with_calendars(people)
    whose = av.choice(WHOSE)
    when = av.choice(WHEN)
    ask = av.choice(ASK)
    conf = min(intent_p, _p(whose), _p(ask))
    route = f"calendar/{ask.choice}"
    if ask.choice == "change":
        return Decision(outcome=Outcome.UNSUPPORTED, reply_key="R_CAL_WRITE", route=route, rule="8", confidence=conf)
    span = window(when.choice, text, now)
    if span is None:
        return Decision(outcome=Outcome.ANSWERED, reply_key="R_CAL_WHEN", route=route, rule="8", confidence=conf)
    start, end, label = span
    at = parse_time(text) if ask.choice == "free_at" else None
    params: dict[str, Any] = {"ask": ask.choice, "when": when.choice, "label": label, "now": now.isoformat(),
                              "start": start.isoformat(), "end": end.isoformat(),
                              "at": list(at) if at else None}

    def owner(pid: str, is_self: bool) -> _Owner:
        p = people[pid]
        return _Owner(pid, p.name, p.calendar_share, entities_for(p, cat), is_self)

    me = speaker.id if speaker is not None else None
    w = whose.choice
    if w in people and w not in owners_cfg:
        return Decision(outcome=Outcome.ANSWERED, reply_key="R_NO_CALENDAR", reply_params={"name": people[w].name},
                        route=route, rule="8", confidence=conf)
    if w in owners_cfg:
        return _answer(params, [owner(w, w == me)], route, conf)
    if w == "everyone" or (w == "none" and me is None):
        return _answer(params, [owner(pid, pid == me) for pid in owners_cfg], route, conf)
    # "my calendar": the known speaker, else ask who (OQ 118).
    if me is not None:
        if me not in owners_cfg:
            return Decision(outcome=Outcome.ANSWERED, reply_key="R_NO_CALENDAR", reply_params={"name": "you"},
                            route=route, rule="8", confidence=conf)
        return _answer(params, [owner(me, True)], route, conf)
    options = {pid: p.name for pid, p in owners_cfg.items()}
    plan = {pid: PlanItem((), REPLY_KEY, {**params, "owners": [owner(pid, True).as_dict()]}, ())
            for pid in owners_cfg}
    names = join_names([f"{p.name}'s" for p in owners_cfg.values()], "or")
    spec = ClarifySpec("pick", f"Whose calendar: {names}?", options, plan)
    return Decision(outcome=Outcome.CLARIFY, reply_key="R_WHOSE_CALENDAR", reply_params={"options": names},
                    clarify=spec, route="calendar/whose", rule="8", confidence=conf)


def is_plan(items: Sequence[PlanItem]) -> bool:
    return bool(items) and all(it.reply_key == REPLY_KEY and not it.calls for it in items)


def completed(item: PlanItem, conf: float | None) -> Decision:
    """A "whose calendar?" answered in the next turn ("Morgan"): read that calendar. A named owner is
    disambiguation, not authentication (personal-context.md §4.1), so the sharing flag still applies."""
    params = dict(item.reply_params)
    owners = [_Owner(o["id"], o["name"], o["share"], tuple(o["entities"]), o["share"] == "household")
              for o in params.pop("owners")]
    return _answer(params, owners, "clarify_reply/calendar", conf)


# ---- reading and replies -----------------------------------------------------------------------
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean_title(title: Any) -> str:
    """Event text is untrusted (anyone can send an invitation): strip links, addresses, numbers and control
    characters, and cap the length. It is only ever spoken, never sent to a model or parsed into actions."""
    t = _CONTROL.sub(" ", str(title or ""))
    t = _URL.sub("", t)
    t = _EMAIL.sub("", t)
    t = _PHONE.sub("", t)
    t = re.sub(r"[<>{}\[\]`*_#|\\]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -:;,")
    if len(t) > TITLE_MAX:
        t = t[:TITLE_MAX].rsplit(" ", 1)[0] + "…"
    return t or "an untitled event"


@dataclass(frozen=True)
class Event:
    title: str
    start: datetime
    end: datetime
    all_day: bool


def _when_value(v: Any, tz: Any) -> tuple[datetime, bool]:
    s = v.get("dateTime") or v.get("date") if isinstance(v, Mapping) else v
    s = str(s)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        d = date.fromisoformat(s)
        return datetime(d.year, d.month, d.day, tzinfo=tz), True
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz), False


def events_from(result: Any, entities: Sequence[str], tz: Any) -> list[Event]:
    out: list[Event] = []
    if not isinstance(result, Mapping):
        return out
    for eid in entities:
        for raw in ((result.get(eid) or {}).get("events") or []):
            try:
                start, all_day = _when_value(raw.get("start"), tz)
                end, _ = _when_value(raw.get("end") or raw.get("start"), tz)
            except (TypeError, ValueError):
                continue
            out.append(Event(clean_title(raw.get("summary")), start, end, all_day))
    return sorted(out, key=lambda e: (not e.all_day, e.start, e.title))


def fmt_time(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"{h} {'AM' if dt.hour < 12 else 'PM'}" if dt.minute == 0 else f"{h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def _day_word(d: date, now: datetime) -> str:
    delta = (d - now.date()).days
    return "today" if delta == 0 else "tomorrow" if delta == 1 else f"on {d.strftime('%A')}"


def _phrase(e: Event, now: datetime, multi_day: bool) -> str:
    day = f" {_day_word(e.start.date(), now)}" if multi_day else ""
    if e.all_day:
        return f"{e.title}{day}, all day" if multi_day else f"{e.title}, all day"
    return f"{e.title}{day} at {fmt_time(e.start)}"


def _sub(o: Mapping[str, Any]) -> dict[str, str]:
    if o["self"]:
        return {"has": "you have", "Has": "You have", "poss": "your", "is": "you're", "does_not": "you don't"}
    n = o["name"]
    return {"has": f"{n} has", "Has": f"{n} has", "poss": f"{n}'s", "is": f"{n} is", "does_not": f"{n} doesn't"}


def _things(n: int) -> str:
    return "one thing" if n == 1 else f"{n} things"


def _one_owner(o: Mapping[str, Any], evs: list[Event], p: Mapping[str, Any], now: datetime) -> str:
    s = _sub(o)
    label = p["label"]
    titles = o["share"] == "household" or o["self"]
    if o["share"] == "private" and not o["self"]:
        return f"Sorry, {o['name']}'s calendar is private."
    ask = p["ask"]
    start, end = datetime.fromisoformat(p["start"]), datetime.fromisoformat(p["end"])
    multi_day = (end - start) > timedelta(days=1) or start.date() != (end - timedelta(seconds=1)).date()
    if ask == "next":
        upcoming = [e for e in evs if not e.all_day and e.start >= now] or [e for e in evs if e.start >= now]
        if not upcoming:
            return f"{s['does_not'].capitalize()} have anything coming up {label}."
        e = upcoming[0]
        when = f"{_day_word(e.start.date(), now)} at {fmt_time(e.start)}" if not e.all_day \
            else f"{_day_word(e.start.date(), now)}, all day"
        what = f"{e.title} " if titles else ""
        return f"{s['poss'].capitalize()} next event is {what}{when}."
    if ask == "free_at" and p.get("at"):
        h, m = p["at"]
        at = datetime(start.year, start.month, start.day, h, m, tzinfo=now.tzinfo)
        clash = [e for e in evs if not e.all_day and e.start < at + timedelta(hours=1) and e.end > at]
        day = "" if label == "today" else f" {label}"
        if not clash:
            return f"Yes, {s['is']} free at {fmt_time(at)}{day}."
        e = clash[0]
        if not titles:
            return f"No, {s['is']} busy at {fmt_time(at)}{day}."
        return f"No, {s['has']} {e.title} at {fmt_time(e.start)}{day}."
    if not evs:
        return f"Nothing on {s['poss']} calendar {label}."
    first = next((e for e in evs if not e.all_day), None)
    if ask in ("count", "free_at") or not titles:
        tail = f", the first at {fmt_time(first.start)}" if first is not None else ""
        return f"{s['Has']} {_things(len(evs))} {label}{tail}."
    said = [_phrase(e, now, multi_day) for e in evs[:MAX_SPOKEN]]
    more = len(evs) - len(said)
    listed = join_names(said) if not more else ", ".join(said) + f" and {more} more"
    return f"{s['Has']} {_things(len(evs))} {label}: {listed}."


def render(params: Mapping[str, Any], result: Any, now: datetime) -> str:
    owners = params["owners"]
    parts = []
    for o in owners:
        evs = events_from(result, o["entities"], now.tzinfo) if (o["share"] != "private" or o["self"]) else []
        parts.append(_one_owner(o, evs, params, now))
    return " ".join(parts)


# ---- a short cache (personal-context.md §2.3) --------------------------------------------------
class EventCache:
    """Recent `calendar.get_events` results by (entities, window), for `ttl_s`. HA's own sync is ~15 minutes,
    so a few minutes more of staleness costs nothing and a repeated question skips the read."""

    def __init__(self, ttl_s: float = 300.0, clock: Any = time.monotonic) -> None:
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: dict[tuple, tuple[float, Any]] = {}

    @staticmethod
    def key(rc: ReadCall) -> tuple | None:
        if (rc.domain, rc.service) != caps.CALENDAR_EVENTS_SERVICE:
            return None
        return (rc.entity_ids, rc.data.get("start_date_time"), rc.data.get("end_date_time"))

    def get(self, rc: ReadCall) -> Any:
        k = self.key(rc)
        hit = self._data.get(k) if k else None
        if hit is None or self._clock() - hit[0] > self.ttl_s:
            return None
        return hit[1]

    def put(self, rc: ReadCall, value: Any) -> None:
        k = self.key(rc)
        if k is not None and isinstance(value, Mapping):
            self._data[k] = (self._clock(), value)
