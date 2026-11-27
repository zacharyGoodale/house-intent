"""build_request(ctx) -> (state, questions) for the ONE Jev request (§5), and AnswerView for reading answers.

Fixed question text follows spec §5.4, with the amendments recorded there (review S5, S7, D9-7): `scope` uses
structured criteria (what / examples / not_for) because its options overlap, examples never use this house's
names, and the device_type and action Choices are generated from the capability table.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ha_jev import calendars, music
from ha_jev import capabilities as caps
from ha_jev import memory
from ha_jev.catalog import Catalog
from ha_jev.config import Handoff, Person, PolicySettings
from ha_jev.extract import Extraction
from ha_jev.jev import Answer, ChoiceAnswer, NoulAnswer
from ha_jev.resolve import rank_options

U = "`utterance`"


@dataclass
class RequestContext:
    extraction: Extraction
    catalog: Catalog
    speaker_area: str | None  # area id
    now: datetime
    recent_targets: list[str] = field(default_factory=list)
    pending: Mapping[str, Any] | None = None  # {kind, question, options{id: descriptor}}
    policy: PolicySettings = field(default_factory=PolicySettings)
    memory_options: Mapping[str, str] | None = None  # memory.forget_options: asked only for "forget <something>"
    people: Mapping[str, Person] = field(default_factory=dict)  # config/people.toml, for the calendar questions
    handoffs: Sequence[Handoff] = ()  # config [[handoffs]]: extra intents for the Hermes agent


@dataclass
class BuiltRequest:
    state: dict
    questions: dict[str, dict]
    overflow: dict[str, list[str]] = field(default_factory=dict)  # choice id -> options pruned away, ranked


# ---- fixed text (§5.4) ------------------------------------------------------------------------
def _intent(u: str, pending: bool, calendar: bool = False, handoffs: Sequence[Handoff] = ()) -> dict:
    criteria: dict[str, Any] = {
        "device_control": "Change a device in this home: lights, plugs or switches, air conditioning (AC), thermostats, room temperature, blinds, speakers or TV (play, pause, volume, app), do-not-disturb, automations, locks. Includes implied requests such as 'it's too dark in here' or 'louder', and any request that names a device from `home.devices`, even when that name contains a word like 'play' or 'mode'.",
        "device_query": "Ask about the current state of something in this home without changing it: whether a light or routine is on, whether a door is locked or open, blind position, what is playing, motion, power use, next alarm or timer, sunrise or sunset time. Includes indoor or room temperature, thermostat target/setpoint, heating/cooling mode, and whether the AC is on. Bare temperature inquiries mean indoor temperature; outdoor temperature belongs to weather.",
        "activate_scene": "Activate a named scene, mood, mode, routine or preset, such as 'reading mode', 'game night' or 'evening in the study'. A name from `home.scenes` said on its own, or with a room, is this.",
        "list_add": "Add an item to a shopping or to-do list.",
        "list_read": "Read out a shopping or to-do list.",
        "weather": "Outdoor weather, outdoor temperature, humidity or forecast at home: now, today, tonight, tomorrow, the next few hours, this week, umbrella. Not indoor room temperature, thermostat settings, heating/cooling, or comfort commands; those are device_query or device_control.",
        "timer_alarm_set": {"what": "Create, list, query, update, reschedule, rename or delete a timer, alarm or reminder.",
                            "not_for": "restaurant bookings or reservation jobs; turning a routine or device from `home.devices` on or off, even if its name contains 'alarm' or 'timer'"},
        "general": "Anything else: facts, maths, advice, jokes, chit-chat, weather somewhere else, questions not about this home's devices. Never a name from `home.scenes` or `home.devices`.",
        "nevermind": "Cancel or dismiss: never mind, cancel that, forget it.",
    }
    criteria.update(memory.INTENT_CRITERIA)  # --- memory intents (memory.py, SPEC §17)
    for h in handoffs:  # config [[handoffs]]: intents the Hermes agent owns
        criteria[h.id] = {"what": h.what, "not_for": h.not_for} if h.not_for else h.what
    if calendar:  # calendars.py: only when someone has a calendar in config/people.toml
        criteria[calendars.INTENT] = calendars.intent_option()
    if pending:
        criteria["clarify_reply"] = "A short answer to `pending_clarification.question`, such as 'the second one', 'yes', 'both'."
    return {"type": "choice",
            "instructions": f"What is the user asking the home voice assistant to do in {u}? Judge the whole request. If `pending_clarification` is present and {u} answers it, choose clarify_reply.",
            "criteria": criteria}


def _noul(text: Any) -> dict:  # a string, or a structured object (TypeSafe: question / not_for ...)
    return {"type": "noul", "instructions": text}


def _device_type(u: str, table: caps.CapabilityTable) -> dict:
    return {"type": "choice", "instructions": f"Which kind of device does {u} act on or ask about?",
            "criteria": table.device_types()}


def _scope(u: str, has_recent: bool) -> dict:
    """Structured criteria (TypeSafe Choice guide): these options overlap for a device that is the only one of its
    kind in a room, a room group, or a device near the speaker, so each says what it is not for (review S5)."""
    criteria: dict[str, Any] = {
        "one_named": {
            "what": "One specific device, picked out by its own name, an alias, or a description that fits only it. "
                    "A room word may be part of its name, and it may be the only device of its kind in its room.",
            "examples": ["turn off the floor lamp", "pause the soundbar", "set the reading light to 40%",
                         "close the study blinds"],
            "not_for": "several devices of one kind, such as 'the study lights'"},
        "several_named": {
            "what": "Two or more specific devices, each picked out by name",
            "examples": ["turn on the floor lamp and the reading light"]},
        "area": {
            "what": "Every device of one kind in one or more named rooms, or around the speaker ('in here', "
                    "'the lights' with no room named)",
            "examples": ["turn off the study lights", "lights off in here", "dim the hall and study lights",
                         "turn off the lights", "study volume to 30"],
            "not_for": "one device called by its own name; one device described only by its kind ('the lamp', "
                       "'the speaker'); 'all the lights' or 'everything' with no room named"},
        "house": {
            "what": "Every device of one kind, or every device, in the whole home or everywhere",
            "examples": ["turn off all the lights", "turn everything off", "every light in the house",
                         "music off everywhere"],
            "not_for": "a request that names a room or says 'in here'"},
    }
    if has_recent:
        criteria["recent"] = {"what": "Refers back to the device(s) in `recent_targets`",
                              "examples": ["turn it off", "make them brighter", "and that one too"]}
    criteria["unspecified"] = {"what": "No device and no room is named; the kind of device is only implied",
                               "examples": ["louder", "pause", "brighter", "next song"]}
    instructions = {
        "question": f"Which devices does {u} refer to?",
        "focus": "How far the request reaches. 'All', 'every' and 'everything' with no room named mean the whole "
                 "home. 'The lights' with no room named means the lights around the speaker."}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _area_noul(u: str, name: str, aliases: tuple[str, ...]) -> dict:
    also = f"; also called: {', '.join(aliases)}" if aliases else ""
    return _noul(f"Does {u} mention the {name} room as a place the request is about (for example 'in the {name}' or '{name} lights'){also}? A room the request excludes ('not the {name}') does not count.")


def _choice(instructions: Any, criteria: dict[str, Any]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _action_choice(u: str, row: caps.DomainCaps) -> dict:
    """An action Choice, generated from the capability row (decision 9: a new domain is one row)."""
    instructions = (row.action_instructions or f"What should happen to the {row.kind} in {{u}}?").replace("{u}", u)
    criteria: dict[str, Any] = {name: a.description for name, a in row.actions.items()}
    criteria["none"] = None
    return _choice(instructions, criteria)


# What a status question can ask about, in Choice order; each is offered only when a domain this home has can
# answer it (its capability row's `query_attributes`). "other" is always offered.
QUERY_ATTRIBUTES: dict[str, str | None] = {
    "on_off": None, "brightness": None, "color": None, "position": None,
    "playing": "what is playing or on: the show, song, app or channel on a TV or speaker",
    "volume": None, "locked": None, "open_closed": None, "motion": None, "power": None, "next_alarm": None,
    "next_timer": None, "sunrise": None, "sunset": None, "items": None,
    "temperature": "actual current indoor or room temperature: how hot/cold is it in here, what's the temperature; not the thermostat target and not outside weather",
    "target_temperature": "thermostat target or setpoint: what is the AC or thermostat set to, what temperature did I set",
    "hvac_mode": "thermostat operating mode: heating, cooling, fan-only or off; is the AC heating or cooling",
}


def _query_attribute(u: str, table: caps.CapabilityTable) -> dict:
    answerable = {a for row in table.rows.values() for a in row.query_attributes}
    criteria = {o: d for o, d in QUERY_ATTRIBUTES.items() if o in answerable}
    criteria["other"] = None
    return _choice(f"What does {u} ask about?", criteria)


def _color(u: str) -> dict:
    opts = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "warm_white", "cool_white", "other", "none"]
    return _choice(f"Which colour does {u} ask for?", {o: None for o in opts})


def _level_word(u: str) -> dict:
    return _choice(f"Which level word does {u} use?", {
        "minimum": "lowest, barely on", "quarter": "a quarter", "low": "low, dim", "half": "half, halfway, medium",
        "three_quarters": "three quarters", "high": "bright, high", "maximum": "full, max, fully", "none": None})


def _number_noul(i: int) -> dict:
    return _noul(f"Is `numbers[{i}]` the value the user wants a device set to or changed by (brightness, volume, blind position, colour temperature, thermostat temperature), rather than part of a device's name such as the number in 'Lamp 2'?")


def _weather_when(u: str) -> dict:
    return _choice(f"Which time does the weather question in {u} ask about?", {
        "now": "explicit current conditions: now, currently, current temperature; humidity",
        "next_hours": "the next few hours, this afternoon, later",
        "today": "today, or a temperature question with no explicit time: give today's high and low plus rain chance",
        "tonight": None,
        "tomorrow": None, "this_week": None, "elsewhere": "a place other than home", "none": None})


PART_QUESTION_IDS = ("intent", "device_type", "scope", "target", "scene", "color", "level_word")


def _is_action_question(qid: str) -> bool:
    return qid.endswith("_action")


# ---- building ---------------------------------------------------------------------------------
def _fmt_now(now: datetime) -> str:
    return now.strftime("%A %Y-%m-%d %H:%M")


def build_state(ctx: RequestContext) -> dict:
    cat = ctx.catalog
    x = ctx.extraction
    state: dict[str, Any] = {"utterance": " ".join(x.original.split()) or x.text}  # §5.3: the original text
    if ctx.speaker_area and ctx.speaker_area in cat.areas:
        state["speaker_area"] = cat.areas[ctx.speaker_area].name
    state["now"] = _fmt_now(ctx.now)
    if x.numbers:
        state["numbers"] = x.numbers
    if x.parts:
        state["parts"] = x.parts
    if ctx.recent_targets:
        state["recent_targets"] = [{"entity_id": t, "name": cat.name(t)} for t in ctx.recent_targets]
    if ctx.pending:
        state["pending_clarification"] = {"question": ctx.pending["question"], "options": dict(ctx.pending["options"])}
    state["home"] = home_directory(cat)
    return state


def home_directory(cat: Catalog) -> dict[str, list[str]]:
    """The names this home knows, from the registry and config (decision 9), so the intent question can tell a
    bare scene name ("storybook") from chit-chat and a device name containing "play" from a request to play."""
    def names(ids: list[str]) -> list[str]:
        out: list[str] = []
        for i in ids:
            e = cat.entities.get(i)
            extra = list(e.aliases) if e is not None else list(cat.presets[i].aliases) if i in cat.presets else []
            for n in (cat.name(i), *extra):
                if i in cat.qualified_scenes:
                    n = f"{cat.qualified_scenes[i].capitalize()} scene {n}"
                if n and n not in out:
                    out.append(n)
        return out

    devices = [t for t in cat.target_options if t in cat.controllable]
    rooms = [a.name + (f" (also: {', '.join(a.aliases)})" if a.aliases else "") for a in cat.areas.values()]
    return {"rooms": rooms, "devices": names(devices),
            "scenes": names(list(cat.scene_options))}


def mention_label(cat: Catalog, eid: str) -> str:
    """The `mentioned__` descriptor: the device's own names only. Group membership and room are left out, since
    a group whose member list contains the named lamp otherwise reads as 'named' too (see open-questions.md)."""
    e = cat.entities[eid]
    names = [e.name, *e.aliases]
    quoted = " or ".join(f"'{n}'" for n in dict.fromkeys(names))
    return f"the {e.kind} {quoted}"


def _estimate_tokens(state: dict, questions: dict) -> float:
    return len(json.dumps({"state": state, "questions": questions}, ensure_ascii=False)) / 3.4


def build_request(ctx: RequestContext, restrict: Mapping[str, list[str]] | None = None) -> BuiltRequest:
    """Build the single request. `restrict` (used for the §3.5 pruning retry) replaces a Choice's options."""
    cat = ctx.catalog
    x = ctx.extraction
    pol = ctx.policy
    state = build_state(ctx)
    overflow: dict[str, list[str]] = {}

    def entity_options(qid: str, options: list[str]) -> list[str]:
        if restrict and qid in restrict:
            return [key for key in restrict[qid] if key in options]
        if len(options) > pol.prune_threshold:
            ranked = rank_options(cat, options, x.text, ctx.speaker_area)
            overflow[qid] = ranked[pol.prune_keep:]
            return ranked[: pol.prune_keep]
        return list(options)

    target_opts = entity_options("target", cat.target_options)
    # Include each part's eligible scenes before pruning; qualification applies per action.
    eligible_scenes = set().union(*(cat.scene_options_for(text) for text in (x.text, *x.parts)))
    scene_opts = entity_options("scene", [key for key in cat.scene_options if key in eligible_scenes])

    def assemble(target_opts: list[str], scene_opts: list[str], mention_opts: list[str]) -> dict[str, dict]:
        q: dict[str, dict] = {}
        pending = ctx.pending is not None
        q["intent"] = _intent(U, pending, bool(calendars.with_calendars(ctx.people)), ctx.handoffs)
        q["needs_llm"] = _noul(f"Would a good reply to {U} require composing an answer from general knowledge or conversation, rather than operating or reading a device, a list, or the home weather forecast?")
        q["security"] = _noul(f"Does {U} ask to lock, unlock, open, arm or disarm anything that secures the home (door lock, deadbolt, alarm system, garage door, gate), or to let someone in, directly or through a scene, script or routine? Asking whether something is locked is not asking to change it. Lights, scenes or routines that only have a word like 'alarm' in their name, such as 'alarm lights', are not security devices.")
        q["unknown_room"] = _noul({
            "question": f"Does {U} name a room, floor or other place in the home that is not listed in `home.rooms`?",
            "not_for": ["'here', 'in here' or no place at all",
                        "a word that is part of a device or scene name in `home.devices` or `home.scenes`"]})
        q["whole_home"] = _noul({
            "question": f"Is {U} meant for the whole home rather than only the room the speaker is in? Words like 'all', 'every', 'everything', 'any' or 'everywhere' applied to devices, with no room named, mean the whole home. `speaker_area` is only where the speaker is; it does not limit 'all'.",
            "not_for": ["a named room or 'here'", "'the lights' or 'the music' without all/every/any"]})
        q["is_compound"] = _noul(f"Does {U} ask for two or more different actions, such as 'turn off the lights and pause the music'? One action on several devices ('turn off the hall and study lights') is not compound.")

        def per_utterance(u: str, prefix: str, number_ids: list[int], text: str) -> None:
            q[prefix + "device_type"] = _device_type(u, cat.caps)
            q[prefix + "scope"] = _scope(u, bool(ctx.recent_targets))
            for a in cat.areas.values():
                q[f"{prefix}area__{a.id}"] = _area_noul(u, a.name, a.aliases)
            q[prefix + "target"] = _choice(
                {"question": f"Which single listed device is {u} about?",
                 "none_when": "no specific device is named, several are, or only a room is named",
                 "not_for": "a generic word such as 'music', 'the lights', 'the speaker' or 'volume' that describes a "
                            "kind of device or content; a device is named only by its own name, alias or a "
                            "description that fits it alone, even if other device names contain that word"},
                {**{t: cat.descriptor(t) for t in target_opts}, "none": "no single listed device"})
            allowed_scenes = set(cat.scene_options_for(text))
            q[prefix + "scene"] = _choice(
                f"Which scene, mode, routine or preset does {u} ask to activate? The room may be said separately ('reading mode in the study').",
                {**{s: cat.descriptor(s) for s in scene_opts if s in allowed_scenes}, "none": "no listed scene fits"})
            for qid, row in cat.caps.action_questions().items():
                q[prefix + qid] = _action_choice(u, row)
            q[prefix + "color"] = _color(u)
            q[prefix + "level_word"] = _level_word(u)
            for i in number_ids:
                q[f"{prefix}number_is_setting__n{i}"] = _number_noul(i)

        per_utterance(U, "", list(range(len(x.numbers))), x.text)
        for eid in mention_opts:
            q[f"mentioned__{eid}"] = _noul(f"Does {U} name {mention_label(cat, eid)} as one of the devices to act on?")
        q["query_attribute"] = _query_attribute(U, cat.caps)
        if cat.sources:
            q["source"] = _choice(f"Which app or source does {U} ask to open on the TV?",
                                  {**{s: None for s in cat.sources}, "none": None})
        if cat.todo_lists:
            q["todo_list"] = _choice(f"Which list does {U} refer to?",
                                     {**{t: cat.descriptor(t) for t in cat.todo_lists}, "none": "no list named"})
        q["weather_when"] = _weather_when(U)
        if ctx.memory_options:  # --- memory (memory.py): which note "forget …" means
            q["memory_item"] = memory.item_question(U, ctx.memory_options)
        # Gated additions (music.py, calendars.py): play spans only when the utterance asks to play something
        # nameable, and the calendar Choices only when someone has a calendar.
        q.update(music.questions(U, music.parse(x.text, cat)))
        q.update(calendars.questions(U, ctx.people))
        if ctx.pending and ctx.pending.get("kind") == "pick":
            q["clarify_pick"] = _choice(
                "Which of the options in `pending_clarification.options` does `utterance` choose?",
                {**dict(ctx.pending["options"]), "all": "all of them, both", "none": "none of them, or `utterance` is not an answer"})
        if ctx.pending and ctx.pending.get("kind") == "confirm":
            q["clarify_affirm"] = _noul("Does `utterance` say yes, or go ahead, to `pending_clarification.question`?")
        for i, _part in enumerate(x.parts):
            u = f"`parts[{i}]`"
            prefix = f"p{i}__"
            q[prefix + "intent"] = _intent(u, pending, handoffs=ctx.handoffs)
            nums = x.part_numbers[i] if i < len(x.part_numbers) else []
            per_utterance(u, prefix, nums, _part)
        return q

    mention_opts = [m for m in cat.mention_options if m in set(target_opts)]
    questions = assemble(target_opts, scene_opts, mention_opts)
    if _estimate_tokens(state, questions) > pol.max_request_tokens and not restrict:
        keep = pol.prune_keep
        if len(target_opts) > keep:
            ranked = rank_options(cat, target_opts, x.text, ctx.speaker_area)
            overflow["target"] = ranked[keep:] + overflow.get("target", [])
            target_opts = ranked[:keep]
        if len(scene_opts) > keep:
            ranked = rank_options(cat, scene_opts, x.text, ctx.speaker_area)
            overflow["scene"] = ranked[keep:] + overflow.get("scene", [])
            scene_opts = ranked[:keep]
        mention_opts = [m for m in cat.mention_options if m in set(target_opts)]
        questions = assemble(target_opts, scene_opts, mention_opts)
    return BuiltRequest(state=state, questions=questions, overflow=overflow)


def build_retry_request(ctx: RequestContext, qid: str, options: list[str]) -> BuiltRequest:
    """§3.5: a second request with only that Choice over the next options."""
    state = build_state(ctx)
    full = build_request(ctx, restrict={qid: options})
    return BuiltRequest(state=state, questions={qid: full.questions[qid]})


# ---- reading ----------------------------------------------------------------------------------
class MissingAnswer(Exception):
    pass


class AnswerView:
    """Reads answers by unprefixed id. A part view maps the part-level ids onto `p<i>__...`, and falls
    through to the whole-utterance answers for global questions (security, needs_llm, ...)."""

    def __init__(self, answers: Mapping[str, Answer], prefix: str = "", base: AnswerView | None = None) -> None:
        self._answers = answers
        self._prefix = prefix
        self._base = base

    @property
    def is_part(self) -> bool:
        return bool(self._prefix)

    def _key(self, qid: str) -> str:
        if not self._prefix:
            return qid
        head = qid.split("__", 1)[0] if "__" in qid else qid
        if qid in PART_QUESTION_IDS or _is_action_question(qid) or head in ("area", "number_is_setting"):
            return self._prefix + qid
        return qid

    def has(self, qid: str) -> bool:
        return self._key(qid) in self._answers

    def raw(self, qid: str) -> Answer:
        key = self._key(qid)
        if key not in self._answers:
            raise MissingAnswer(key)
        return self._answers[key]

    def choice(self, qid: str) -> ChoiceAnswer:
        a = self.raw(qid)
        if not isinstance(a, ChoiceAnswer):
            raise MissingAnswer(f"{qid} is not a choice")
        return a

    def noul(self, qid: str) -> float:
        a = self.raw(qid)
        if not isinstance(a, NoulAnswer):
            raise MissingAnswer(f"{qid} is not a noul")
        return a.p

    def for_part(self, i: int) -> AnswerView:
        return AnswerView(self._answers, prefix=f"p{i}__", base=self)


def action_question_ids(table: caps.CapabilityTable = caps.DEFAULT) -> tuple[str, ...]:
    return tuple(table.action_questions())
