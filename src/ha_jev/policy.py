"""decide(ctx, answers) -> Decision (spec §6). Pure: no I/O and no clock reads (ctx.now is given).

The first rule that fires wins (§6.2). Everything domain-specific comes from the capability table on the
catalog (`cat.caps`): which action Choice a domain uses, what a room plural addresses, what a bare command
means, and how calls are built (`actions.py`). This module never compares against a domain name.

Decision confidence (§6, amended for review S5) is the minimum over the probabilities actually read: the intent's
probability, the **combined** probability of the scope readings that lead to the same service calls, the target
evidence, the combined probability of the action options that lead to the same calls, and each argument used.
Choice answers are read through their `probabilities`, not their flatness statistic `confidence`, and uncertainty
between readings that do the same thing is ignored (TypeSafe: "ignore uncertainty on unused branches").
"""

from __future__ import annotations

import re
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ha_jev import actions, calendars, music
from ha_jev import capabilities as caps
from ha_jev import memory as memory_intents
from ha_jev.catalog import Catalog, normalize_name
from ha_jev.config import Person, PolicySettings
from ha_jev.extract import Extraction
from ha_jev.ha.interface import EntityState
from ha_jev.jev import ChoiceAnswer
from ha_jev.memory import MemoryView
from ha_jev.models import ClarifySpec, Decision, Outcome, PlanItem, ReadCall, ServiceCall, canonical_calls
from ha_jev.questions import AnswerView, MissingAnswer
from ha_jev.replies import confirm_phrase, join_names, render_template
from ha_jev.resolve import (
    area_domain_targets,
    area_light_targets,
    expand_groups,
    house_light_targets,
    music_speakers_in,
    name_siblings,
    named_by_utterance,
    room_words,
    subsumed_by,
    playing_players,
    alias_sharers,
    plural_kind_said,
    scene_area,
    scene_distinguished,
    scene_siblings,
    word_candidates_any,
    word_candidates,
    select_source_players,
    tvs_in,
)
from ha_jev.session import Pending, SessionSnapshot

__all__ = ["DecisionContext", "Pending", "SessionSnapshot", "decide"]

LEVELS = {"minimum": 1, "quarter": 25, "low": 20, "half": 50, "three_quarters": 75, "high": 80, "maximum": 100}
ONOFF_ACTIONS = {"turn_on", "turn_off", "toggle"}
NAME_CHOICES = ("scene", "source")  # Choice ids whose picks are catalog names (for exact-name evidence)


@dataclass
class DecisionContext:
    utterance: str
    extraction: Extraction
    catalog: Catalog
    states: Mapping[str, EntityState]
    speaker_area: str | None  # area id
    session: SessionSnapshot
    policy: PolicySettings
    now: datetime
    memory: MemoryView | None = None  # memory.py: the speaker and the facts they can see (None = memory off)
    people: Mapping[str, Person] = field(default_factory=dict)  # config/people.toml
    person: Person | None = None  # the known speaker (a per-person wake word), else None


class Stop(Exception):
    """A terminal Decision raised from deep inside a handler."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(decision.reply_key)
        self.decision = decision


@dataclass
class Alt:
    id: str
    p: float
    item: PlanItem


def _freeze(v: Any) -> Any:
    if isinstance(v, Mapping):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def call_signature(calls: Sequence[ServiceCall]) -> tuple:
    """What a plan does, for adding up readings that do the same thing."""
    return tuple((c.domain, c.service, c.entity_ids, _freeze(c.data)) for c in canonical_calls(calls))


@dataclass
class Plan:
    calls: list[ServiceCall]
    reply_key: str
    params: dict[str, Any]
    confs: list[float]
    targets: list[str]
    route: str
    high_stakes_on: bool = False
    force_confirm: bool = False  # only a spoken yes may run it (a sweep that found only high-stakes devices)
    force_pick: bool = False  # a partial name that fits several devices: always ask which
    alts: list[Alt] = field(default_factory=list)
    alt_kind: str = "device"  # device | scene | room
    option_id: str | None = None  # this plan's own option id for a clarification

    @property
    def confidence(self) -> float:
        return min(self.confs) if self.confs else 1.0

    def item(self) -> PlanItem:
        return PlanItem(tuple(canonical_calls(self.calls)), self.reply_key, dict(self.params), tuple(self.targets))


def p_of(a: ChoiceAnswer) -> float:
    """The probability of the chosen option (falls back to confidence when no distribution came back)."""
    if a.probabilities:
        return float(a.probabilities.get(a.choice, 0.0))
    return float(a.confidence)


def readings(a: ChoiceAnswer, min_p: float) -> list[tuple[str, float]]:
    """The options worth considering: the chosen one, plus any other with p >= min_p, most likely first."""
    probs = dict(a.probabilities) or {a.choice: a.confidence}
    probs.setdefault(a.choice, a.confidence)
    out = [(o, float(p)) for o, p in probs.items() if o == a.choice or p >= min_p]
    return sorted(out, key=lambda kv: (kv[0] != a.choice, -kv[1], kv[0]))


@dataclass
class Turn:
    ctx: DecisionContext
    av: AnswerView
    numbers: list[int]
    intent_p: float

    @property
    def cat(self) -> Catalog:
        return self.ctx.catalog

    @property
    def table(self) -> caps.CapabilityTable:
        return self.ctx.catalog.caps

    @property
    def pol(self) -> PolicySettings:
        return self.ctx.policy

    def number(self) -> actions.Value | None:
        for i in self.numbers:
            qid = f"number_is_setting__n{i}"
            if self.av.has(qid):
                p = self.av.noul(qid)
                if p >= self.pol.number_setting:
                    n = self.ctx.extraction.numbers[i]
                    return actions.Value(float(n["value"]), n.get("unit"), p)
        return None

    def value(self) -> actions.Value | None:
        v = self.number()
        if v is not None:
            return v
        if self.av.has("level_word"):
            lw = self.av.choice("level_word")
            if lw.choice in LEVELS:
                return actions.Value(float(LEVELS[lw.choice]), "level", p_of(lw))
        return None

    def answer(self, qid: str) -> ChoiceAnswer | None:
        return self.av.choice(qid) if self.av.has(qid) else None

    def device_type(self) -> str | None:
        """The device_type answer, or, when it says none, the kind whose action Choice alone has a confident
        non-none answer ("pause": media_action pause 0.95; round 2, class 2). Two confident kinds stay open."""
        if not self.av.has("device_type"):
            return None
        dt = self.av.choice("device_type").choice
        if dt != "none":
            return dt
        kinds = set()
        for qid, row in self.table.action_questions().items():
            if self.av.has(qid):
                a = self.av.choice(qid)
                if a.choice != "none" and p_of(a) >= self.pol.act and row.device_type:
                    kinds.add(row.device_type)
        return kinds.pop() if len(kinds) == 1 else dt


# ---- decision constructors --------------------------------------------------------------------
def _d(outcome: Outcome, key: str, route: str, rule: str, *, params: Mapping[str, Any] | None = None,
       confidence: float | None = None, **kw: Any) -> Decision:
    return Decision(outcome=outcome, reply_key=key, reply_params=dict(params or {}), route=route, rule=rule,
                    confidence=confidence, **kw)


def refuse_lock(route: str, rule: str, confidence: float | None = None) -> Decision:
    return _d(Outcome.REFUSED, "R_LOCK", route, rule, confidence=confidence)


def _luna(route: str, rule: str, confidence: float | None = None) -> Decision:
    return _d(Outcome.ANSWERED, "R_LUNA", route, rule, confidence=confidence, use_luna=True)


def _clarify(key: str, route: str, rule: str, confidence: float | None = None, **params: Any) -> Decision:
    return _d(Outcome.CLARIFY, key, route, rule, params=params, confidence=confidence)


def _unsupported(key: str, route: str, rule: str = "8", **params: Any) -> Decision:
    return _d(Outcome.UNSUPPORTED, key, route, rule, params=params)


def _stop_unsupported(key: str, route: str, **params: Any) -> Stop:
    return Stop(_unsupported(key, route, **params))


def _stop_clarify(key: str, route: str, **params: Any) -> Stop:
    return Stop(_clarify(key, route, "8", **params))


# ---- entry point ------------------------------------------------------------------------------
def decide(ctx: DecisionContext, av: AnswerView) -> Decision:
    try:
        return _decide(ctx, av)
    except Stop as s:
        return s.decision


def _decide(ctx: DecisionContext, av: AnswerView) -> Decision:
    pol = ctx.policy
    x = ctx.extraction
    intent = av.choice("intent")
    intent_p = p_of(intent)

    # 4 before 3 (review S11): a reply that also asks for a lock action ("yes, and unlock it") refuses
    # everything; it never completes the pending plan and drops the lock clause.
    security = av.noul("security")
    pending = ctx.session.pending
    acting = ("device_control", "activate_scene") + (("clarify_reply",) if pending is not None else ())
    if security >= pol.security_refuse and intent.choice in acting:
        return refuse_lock("policy/security", "4", confidence=security)

    # Resy owns reservation schedules, not HA calendars or local timers.
    if re.search(r"\b(?:resy|amooseboosh|amos\s+bush|amuse[\s-]+bouche|reservations?|restaurants?)\b",
                 x.text, re.IGNORECASE):
        return _luna("hermes/reservations", "reservations", 1.0)

    # 3: a pending clarification is answered, or dropped. A room question is also answered by naming one of the
    # offered rooms, even when the reply is short enough that the intent is unsure ("the study").
    if pending is not None and intent.choice == "clarify_reply":
        if intent_p >= pol.intent_floor:
            return _resolve_pending(ctx, av, pending)
        named = [a for a in pending.options if av.has(f"area__{a}") and av.noul(f"area__{a}") >= pol.noul_yes]
        if pending.kind == "pick" and len(named) == 1 and named[0] in pending.plan:
            return _complete(pending, named, av.noul(f"area__{named[0]}"))

    # 5, 6: unclear intent, general questions.
    needs_llm = av.noul("needs_llm")
    if intent_p < pol.intent_floor:
        if needs_llm >= pol.needs_llm_route:
            return _luna("luna/low_intent", "5", intent_p)
        return _clarify("R_WHAT", "clarify/low_intent", "5", intent_p)
    if intent.choice == "reservations":
        return _luna("hermes/reservations", "reservations", intent_p)
    if intent.choice == "general":
        if intent_p >= pol.general_route or needs_llm >= pol.needs_llm_route:
            return _luna("luna/general", "6", intent_p)
        return _clarify("R_WHAT", "clarify/general_unsure", "6", intent_p)

    # 7: compound commands.
    is_compound = av.noul("is_compound")
    if len(x.parts) >= 2 and is_compound >= pol.compound:
        return _compound(ctx, av)
    if len(x.parts) < 2 and is_compound >= pol.compound_nosplit:
        return _clarify("R_ONE_AT_A_TIME", "compound/unsplit", "7", is_compound)

    # 8, 9: dispatch.
    turn = Turn(ctx, av, list(range(len(x.numbers))), intent_p)
    handler = _INTENT_HANDLERS.get(intent.choice)
    if handler is None:  # clarify_reply without a pending clarification
        return _clarify("R_WHAT", "clarify/no_pending", "8", intent_p)
    return handler(turn)


def _timers(turn: Turn) -> Decision:
    from ha_jev.schedules import parse
    operation = parse(turn.ctx.extraction.text)
    if operation is None:
        return _luna("hermes/schedules", "8", turn.intent_p)
    return Decision(Outcome.ANSWERED, reply_key="R_OK", route="timer", rule="8",
                    confidence=turn.intent_p, schedule=operation)


def _nevermind(turn: Turn) -> Decision:
    return _d(Outcome.ANSWERED, "R_OK", "nevermind", "8", confidence=turn.intent_p)


# ---- band and clarification (§6.2 step 9, §6.9) ----------------------------------------------
def _executed(plan: Plan, rule: str = "9") -> Decision:
    calls = canonical_calls(plan.calls)
    if not calls:
        return _unsupported("R_CANT", plan.route, rule, name="That")
    return Decision(outcome=Outcome.EXECUTED, calls=calls, reply_key=plan.reply_key, reply_params=dict(plan.params),
                    route=plan.route, rule=rule, confidence=plan.confidence, targets=tuple(plan.targets))


def _band(turn: Turn, plan: Plan) -> Decision:
    pol = turn.pol
    conf = plan.confidence
    threshold = pol.high_stakes_act if plan.high_stakes_on else pol.act
    if plan.force_confirm:
        return _clarify_from(turn, plan, force_confirm=True)
    if plan.force_pick:
        return _pick(turn, plan.alts, "R_WHICH_DEVICE", plan.route, conf)
    if conf >= threshold:
        return _executed(plan)
    if plan.high_stakes_on and conf >= pol.act:
        return _clarify_from(turn, plan, force_confirm=True)
    if conf >= pol.clarify_floor:
        return _clarify_from(turn, plan)
    if turn.av.noul("needs_llm") >= pol.needs_llm_route:
        return _luna("luna/low_confidence", "9", conf)
    return _clarify("R_WHAT", plan.route, "9", conf)


def _pick(turn: Turn, alts: list[Alt], key: str, route: str, confidence: float | None) -> Decision:
    cat = turn.cat
    offered = alts[: turn.pol.clarify_max_options]
    options = {a.id: cat.descriptor(a.id) for a in offered}
    params = {"options": join_names([cat.name(a.id) for a in offered], "or")}
    spec = ClarifySpec("pick", render_template(key, params), options, {a.id: a.item for a in offered})
    targets = tuple(a.id for a in offered if a.id in cat.entities)
    return Decision(outcome=Outcome.CLARIFY, reply_key=key, reply_params=params, clarify=spec, route=route,
                    rule="9", confidence=confidence, targets=targets)


def _confirm(turn: Turn, best: Alt, route: str, confidence: float | None) -> Decision:
    cat = turn.cat
    params = {"phrase": confirm_phrase(best.item.reply_key, best.item.reply_params)}
    spec = ClarifySpec("confirm", render_template("R_CONFIRM", params), {best.id: cat.descriptor(best.id)},
                       {best.id: best.item})
    return Decision(outcome=Outcome.CLARIFY, reply_key="R_CONFIRM", reply_params=params, clarify=spec,
                    route=route, rule="9", confidence=confidence, targets=tuple(best.item.targets))


def _clarify_from(turn: Turn, plan: Plan, force_confirm: bool = False) -> Decision:
    pol = turn.pol
    alts = sorted((a for a in plan.alts if a.p >= pol.clarify_min_p), key=lambda a: -a.p)
    own = plan.option_id or (plan.targets[0] if plan.targets else "plan")
    if not alts or force_confirm:
        alts = [Alt(own, plan.confidence, plan.item())]
    if len(alts) >= 2:
        key = {"scene": "R_WHICH_SCENE", "room": "R_WHICH_ROOM"}.get(plan.alt_kind, "R_WHICH_DEVICE")
        return _pick(turn, alts, key, plan.route, plan.confidence)
    return _confirm(turn, alts[0], plan.route, plan.confidence)


def _word_ids(turn: Turn) -> list[str]:
    device_type = turn.device_type()
    domains = turn.table.domains_for_device_type(device_type) or None
    areas, _ = _area_answers(turn)
    areas = areas or ([turn.ctx.speaker_area] if turn.ctx.speaker_area else [])
    return word_candidates(turn.cat, turn.ctx.extraction.text, domains, areas)


def _candidate_alts(turn: Turn, confs: Sequence[float]) -> list[Alt]:
    """Devices whose names contain the words said, most likely first by the target Choice (never denied ones)."""
    ids = _word_ids(turn)
    probs = turn.av.choice("target").probabilities if turn.av.has("target") else {}
    out = []
    for i in sorted(ids, key=lambda i: (-probs.get(i, 0.0), i)):
        try:
            out.append(Alt(i, probs.get(i, 0.0), _build_plan(turn, [i], list(confs), "one", None,
                                                             route="device_control/one_named").item()))
        except Stop:
            continue
    return out


def _which_device(turn: Turn, route: str, confs: Sequence[float] = ()) -> Stop:
    """R_WHICH_DEVICE with real options (review S9): the likely targets, then devices whose names contain the
    words said, else R_WHAT. Never "Which one: ?"."""
    worded = _candidate_alts(turn, confs)
    alts = []
    if len(worded) >= 2:
        alts = worded  # the words said fit several devices: offer those, not unrelated guesses
    elif turn.av.has("target"):
        alts = _target_alts(turn, turn.av.choice("target").probabilities, list(confs))
    if len(alts) < 2:
        seen = {a.id for a in alts}
        alts += [a for a in worded if a.id not in seen]
    if len(alts) >= 2:
        return Stop(_pick(turn, alts, "R_WHICH_DEVICE", route, min(a.p for a in alts[:2])))
    if len(alts) == 1:
        return Stop(_confirm(turn, alts[0], route, alts[0].p))
    return _stop_clarify("R_WHAT", route)


def _room_clarify(turn: Turn, device_type: str | None, confs: list[float], route: str) -> Stop:
    """R_WHICH_ROOM: offer the areas that have targets for this device type, most likely first."""
    cat = turn.cat
    scored = []
    for a in cat.areas.values():
        try:
            plan = _plan_for_area(turn, device_type, [a.id], list(confs), "area")
        except Stop:
            continue
        p = turn.av.noul(f"area__{a.id}") if turn.av.has(f"area__{a.id}") else 0.0
        scored.append((p, a.id, plan))
    scored.sort(key=lambda t: (-t[0], t[1]))
    if not scored:
        return _stop_clarify("R_WHAT", route)
    # Every candidate room stays answerable; the list is spoken only when it is short (so no room is left out).
    options = {aid: cat.areas[aid].name for _, aid, _ in scored}
    spoken = list(options.values()) if len(options) <= turn.pol.clarify_max_options else []
    params = {"options": join_names(spoken, "or")}
    spec = ClarifySpec("pick", render_template("R_WHICH_ROOM", params), options,
                       {aid: plan.item() for _, aid, plan in scored})
    return Stop(Decision(outcome=Outcome.CLARIFY, reply_key="R_WHICH_ROOM", reply_params=params, clarify=spec,
                         route=route, rule="8", confidence=min(confs) if confs else None))


def _resolve_pending(ctx: DecisionContext, av: AnswerView, pending: Pending) -> Decision:
    pol = ctx.policy
    chosen: list[str] = []
    conf: float | None = None
    if pending.kind == "pick" and av.has("clarify_pick"):
        cp = av.choice("clarify_pick")
        conf = p_of(cp)
        if cp.choice != "none" and conf >= pol.act:
            if cp.choice == "all":
                chosen = list(pending.options)
            elif cp.choice in pending.plan:
                chosen = [cp.choice]
    elif pending.kind == "confirm" and av.has("clarify_affirm"):
        conf = av.noul("clarify_affirm")
        if conf >= pol.noul_yes:
            chosen = list(pending.plan)[:1]
    if not chosen and pending.kind == "pick":
        named = [a for a in pending.options if av.has(f"area__{a}") and av.noul(f"area__{a}") >= pol.noul_yes]
        if len(named) == 1 and named[0] in pending.plan:
            chosen, conf = named, av.noul(f"area__{named[0]}")
    if not chosen:
        corrected = _correction(ctx, av, pending)
        if corrected is not None:
            return corrected
        return _d(Outcome.ANSWERED, "R_OK", "clarify_reply/declined", "3", confidence=conf)
    return _complete(pending, chosen, conf)


def _complete(pending: Pending, chosen: list[str], conf: float | None) -> Decision:
    items = [pending.plan[c] for c in chosen]
    if calendars.is_plan(items):  # hook: "whose calendar?" answered; a read, not calls
        return calendars.completed(items[0], conf)
    calls = canonical_calls(c for it in items for c in it.calls)
    if len(items) == 1:
        key, params = items[0].reply_key, dict(items[0].reply_params)
    else:
        key, params = "R_COMPOUND", {"parts": [[it.reply_key, dict(it.reply_params)] for it in items]}
    targets = tuple(t for it in items for t in it.targets)
    return Decision(outcome=Outcome.EXECUTED, calls=calls, reply_key=key, reply_params=params,
                    route="clarify_reply/completed", rule="3", confidence=conf, targets=targets)


def _correction(ctx: DecisionContext, av: AnswerView, pending: Pending) -> Decision | None:
    """"The reading one" in reply to "Do you want me to turn on the Study lights?": not a yes, but a clear
    device of the same domain. The pending action moves to it (same service and data), through the normal guard.
    Never to another domain, and never to a denied or uncontrollable device."""
    if not av.has("target") or len(pending.plan) != 1:
        return None
    tc = av.choice("target")
    tp = p_of(tc)
    e = ctx.catalog.entities.get(tc.choice)
    if e is None or tp < ctx.policy.act or e.denied or not e.controllable:
        return None
    item = next(iter(pending.plan.values()))
    if not item.calls or any(c.domain != e.domain for c in item.calls):
        return None
    calls = canonical_calls(ServiceCall(c.domain, c.service, (e.entity_id,), c.data) for c in item.calls)
    params = {k: v for k, v in item.reply_params.items() if k not in ("names", "name")}
    params.update(names=e.name, name=e.name)
    return Decision(outcome=Outcome.EXECUTED, calls=calls, reply_key=item.reply_key, reply_params=params,
                    route="clarify_reply/corrected", rule="3", confidence=tp,
                    targets=(e.entity_id,))


# ---- compound (§6.8) --------------------------------------------------------------------------
def _compound(ctx: DecisionContext, av: AnswerView) -> Decision:
    pol = ctx.policy
    x = ctx.extraction
    one_at_a_time = _clarify("R_ONE_AT_A_TIME", "compound", "7")
    plans: list[Plan] = []
    for i in range(len(x.parts)):
        pav = av.for_part(i)
        try:
            pi = pav.choice("intent")
        except MissingAnswer:
            return one_at_a_time
        if pi.choice not in ("device_control", "activate_scene"):
            return one_at_a_time
        numbers = x.part_numbers[i] if i < len(x.part_numbers) else []
        pt = Turn(ctx, pav, numbers, p_of(pi))
        try:
            plan = _control(pt) if pi.choice == "device_control" else _scene(pt)
        except Stop as s:
            if s.decision.reply_key == "R_LOCK":
                return s.decision
            return one_at_a_time
        threshold = pol.high_stakes_act if plan.high_stakes_on else pol.act
        if plan.confidence < threshold or not plan.calls or plan.force_confirm or plan.force_pick:
            return one_at_a_time
        plans.append(plan)
    calls = canonical_calls(c for p in plans for c in p.calls)
    return Decision(outcome=Outcome.EXECUTED, calls=calls, reply_key="R_COMPOUND",
                    reply_params={"parts": [[p.reply_key, dict(p.params)] for p in plans]},
                    route="compound", rule="7", confidence=min(p.confidence for p in plans),
                    targets=tuple(t for p in plans for t in p.targets))


# ---- marginalization (review S5) --------------------------------------------------------------
@dataclass
class _Reading:
    option: str
    p: float
    signature: tuple | None = None  # what the reading does, when it produced calls
    stop: Stop | None = None
    value: Any = None  # the Plan or built calls behind the signature

    def group_key(self) -> tuple:
        if self.stop is None:
            return ("do", self.signature)
        d = self.stop.decision
        return ("stop", d.outcome, d.reply_key)


def _choose(results: list[_Reading], min_p: float, exact_name: bool = False) -> tuple[_Reading, float]:
    """Group readings by what they would do and return the most probable group's first reading and its mass.
    A lock refusal with any real mass wins: refusing is always safe (decision 3)."""
    groups: dict[tuple, list[_Reading]] = {}
    for r in results:
        groups.setdefault(r.group_key(), []).append(r)
    mass = {k: sum(r.p for r in rs) for k, rs in groups.items()}
    for k, rs in groups.items():
        if k[0] == "stop" and k[2] == "R_LOCK" and mass[k] >= min_p:
            return rs[0], mass[k]
    # A reading the device can't do at all ("turn off" a speaker without power control) is not a competing
    # interpretation: condition on the feasible ones (round 2, class 2/3).
    # When the utterance says a catalog name in full (a scene, script, preset or source; round 2, class 4), a
    # reading that finds nothing to do ("what would you like me to do?") is explained by that name, not a
    # competitor. Without such a name, "none" is a real answer ("don't turn off ...") and stays. Masses are renormalized against the whole distribution (1.0), so
    # probability on options that weren't read (say, a general question) is never handed to the survivors.
    drop_keys = ("R_CANT", "R_WHAT", "R_WHAT_LEVEL") if exact_name else ("R_CANT",)
    dropped = [k for k in groups if k[0] == "stop" and k[2] in drop_keys]
    if dropped and any(k[0] == "do" for k in groups):
        gone = sum(mass[k] for k in dropped)
        for k in dropped:
            del groups[k]
        mass = {k: min(1.0, v / max(1e-9, 1.0 - gone)) for k, v in mass.items() if k in groups}
    order = {k: i for i, k in enumerate(groups)}  # the chosen option's group first on ties
    best = max(groups, key=lambda k: (mass[k], -order[k]))
    # Within a group every reading does the same thing; keep the one with the strongest evidence of its own.
    member = max(groups[best], key=lambda r: getattr(r.value, "confidence", 0.0) if r.stop is None else 0.0)
    return member, mass[best]


# ---- device control (§6.3, §6.4) --------------------------------------------------------------
def _area_answers(turn: Turn) -> tuple[list[str], list[float]]:
    areas, confs = [], []
    for a in turn.cat.areas:
        qid = f"area__{a}"
        if turn.av.has(qid):
            p = turn.av.noul(qid)
            if p >= turn.pol.noul_yes:
                areas.append(a)
                confs.append(p)
    return areas, confs


def exact_catalog_name(turn: Turn) -> bool:
    """The utterance says, in full, the name or an alias of the scene Choice's pick or the source Choice's pick
    (room words may be left out, since the room is often said apart)."""
    said = normalize_name(turn.ctx.extraction.text)
    words = set(re.findall(r"[a-z0-9+]+", said))
    for qid in NAME_CHOICES:
        if not turn.av.has(qid):
            continue
        a = turn.av.choice(qid)
        if a.choice == "none" or p_of(a) < turn.pol.act:
            continue
        names = [turn.cat.name(a.choice), *_aliases(turn.cat, a.choice)]
        for n in names:
            toks = set(re.findall(r"[a-z0-9+]+", normalize_name(n))) - room_words(turn.cat)
            if toks and toks <= words:
                return True
    return False


def _aliases(cat: Catalog, key: str) -> tuple[str, ...]:
    if key in cat.presets:
        return tuple(cat.presets[key].aliases)
    e = cat.entities.get(key)
    return e.aliases if e is not None else ()


def _mentioned(turn: Turn) -> list[tuple[str, float]]:
    chosen: list[tuple[str, float]] = []
    if turn.av.is_part:
        return chosen
    for eid in turn.cat.mention_options:
        qid = f"mentioned__{eid}"
        if turn.av.has(qid):
            p = turn.av.noul(qid)
            if p >= turn.pol.noul_yes:
                chosen.append((eid, p))
    ids = [e for e, _ in chosen]
    return [(e, p) for e, p in chosen if subsumed_by(turn.cat, e, ids, turn.ctx.extraction.text) is None]


def target_probs(turn: Turn) -> dict[str, float]:
    """The target Choice with devices the utterance only mentions by a generic word ("the music") moved to none
    (round 2, class 2): such a device is not named, so its probability is evidence of no named device."""
    tc = turn.av.choice("target")
    probs = dict(tc.probabilities or {tc.choice: 1.0})
    text = turn.ctx.extraction.text
    for o in list(probs):
        if o in turn.cat.entities and named_by_utterance(turn.cat, o, text) is False:
            probs["none"] = probs.get("none", 0.0) + probs.pop(o)
    # Longest match (round 2, class 7): a device whose words said all sit inside another candidate's fully said
    # name ("<speaker>" inside "<speaker> do not disturb") is that other device.
    for o in list(probs):
        if o in turn.cat.entities:
            host = subsumed_by(turn.cat, o, [c for c in probs if c in turn.cat.entities], text)
            if host is not None:
                probs[host] = probs.get(host, 0.0) + probs.pop(o)
    return probs


def scope_posterior(turn: Turn, scope: ChoiceAnswer) -> dict[str, float]:
    """The scope Choice, reweighted by the other answers in the same request (review S4, S5).

    The `target` Choice is asked to pick none when no single device is named, so P(a single non-group device is
    named) is evidence *against* the room, house and bare readings, and P(target is not none) is evidence *for*
    the named reading. Each reading's probability is multiplied by its consistency with that evidence and the
    result renormalized. A confident named device is then never widened to its room, and a room request whose
    target is none is not lost to an impossible named reading."""
    probs = dict(scope.probabilities) or {scope.choice: 1.0}
    if not turn.av.has("target"):
        return probs
    tprobs = target_probs(turn)
    p_device = sum(p for o, p in tprobs.items() if (e := turn.cat.entities.get(o)) is not None and not e.is_group)
    p_any = sum(p for o, p in tprobs.items() if o != "none")
    several = len(_mentioned(turn)) >= 2
    if p_any < 1.0 and len(_word_ids(turn)) >= 2:
        p_any = 1.0  # "the lamp" with several lamps: one device is meant, only which is open (a question, not none)
    weights = {"one_named": p_any, "several_named": 1.0 if several else p_any,
               "area": 1.0 - p_device, "house": 1.0 - p_device, "unspecified": 1.0 - p_device}
    if turn.ctx.extraction.anaphor and turn.ctx.session.recent_targets and not _area_answers(turn)[0]:
        # "make it warmer" with a device just used and no room named: the pronoun points back (review S4).
        for o in ("area", "house", "unspecified"):
            weights[o] *= turn.pol.anaphor_room_weight
    raw = {o: p * weights.get(o, 1.0) for o, p in probs.items()}
    if not _area_answers(turn)[0]:
        # Round 2, class 1: the scope Choice reads "all the lights" as the speaker's room. With no room named, the
        # room-or-bare mass moves to the home by the whole_home noul, or by a whole-home quantifier word.
        w = turn.av.noul("whole_home") if turn.av.has("whole_home") else 0.0
        if turn.ctx.extraction.quantifier:
            w = max(w, turn.pol.quantifier_home)
        if w >= turn.pol.noul_yes:  # only ever widens; an explicit house reading is never narrowed by it
            moved = (raw.get("area", 0.0) + raw.get("unspecified", 0.0)) * w
            raw["house"] = raw.get("house", 0.0) + moved
            for o in ("area", "unspecified"):
                if o in raw:
                    raw[o] *= 1.0 - w
    total = sum(raw.values())
    if total <= 1e-9:
        return probs
    return {o: v / total for o, v in raw.items()}


def _denied_mass(turn: Turn, qid: str) -> float:
    """Probability a Choice puts on denied entities (a lock, an unlock script). Any real mass refuses (decision 3):
    the refusal is safe, and a request whose most likely device is a lock is never quietly redirected."""
    if not turn.av.has(qid):
        return 0.0
    a = turn.av.choice(qid)
    probs = a.probabilities or {a.choice: 1.0}
    return sum(p for o, p in probs.items() if (e := turn.cat.entities.get(o)) is not None and e.denied)


def _no_such_device(turn: Turn) -> bool:
    """A control request for a kind of device this home can't control ("the thermostat", round 2, class 6): the
    device_type is one with no controllable domain (or none), the target Choice is confident nothing is named,
    and no device's name contains the words said."""
    if not turn.av.has("device_type") or not turn.av.has("target"):
        return False
    dt = turn.av.choice("device_type").choice
    if dt in ("none", "any") or turn.ctx.extraction.anaphor or turn.ctx.session.recent_targets:
        return False
    controllable = any((row := turn.table.caps_for(d)) is not None and row.controllable
                       for d in turn.table.domains_for_device_type(dt))
    tprobs = target_probs(turn)
    return (not controllable and dt != "any" and tprobs.get("none", 0.0) >= turn.pol.none_confident
            and not _word_ids(turn) and not _area_answers(turn)[0])


def _control(turn: Turn) -> Plan:
    """Build a plan for every plausible scope reading, add up the readings that do the same thing, and keep the
    most probable (review S5)."""
    if _denied_mass(turn, "target") >= turn.pol.clarify_min_p:
        raise Stop(refuse_lock("device_control/denied_target", "4"))
    if _no_such_device(turn):
        raise _stop_unsupported("R_NO_DEVICE", "device_control/no_such_device")
    scope = turn.av.choice("scope")
    post = scope_posterior(turn, scope)
    top = max(post, key=lambda o: (post[o], o == scope.choice))
    results = []
    for option, p in sorted(post.items(), key=lambda kv: (kv[0] != top, -kv[1], kv[0])):
        if option != top and p < turn.pol.clarify_min_p:
            continue
        try:
            plan = _control_for(turn, option, [turn.intent_p])
            sig = ("confirm",) + call_signature(plan.calls) if plan.force_confirm else call_signature(plan.calls)
            results.append(_Reading(option, p, signature=sig, value=plan))
        except Stop as s:
            results.append(_Reading(option, p, stop=s))
    chosen, mass = _choose(results, turn.pol.clarify_min_p)
    if chosen.stop is not None:
        raise chosen.stop
    plan: Plan = chosen.value
    plan.confs.append(mass)
    return plan


def _control_for(turn: Turn, s: str, confs: list[float]) -> Plan:
    av, cat, ctx = turn.av, turn.cat, turn.ctx
    if s == "several_named":
        chosen = _mentioned(turn)
        if len(chosen) >= 2:
            return _build_plan(turn, [e for e, _ in chosen], confs + [p for _, p in chosen], "named", None,
                               route="device_control/several_named")
        s = "one_named"
    if s == "one_named":
        return _control_one(turn, confs)
    if s == "recent":
        recent = [r for r in ctx.session.recent_targets if r in cat.entities]
        if not recent:
            raise _which_device(turn, "device_control/recent", confs)
        for r in recent:
            if cat.entities[r].denied:
                raise Stop(refuse_lock("device_control/recent", "4"))
        return _build_plan(turn, recent, confs, "recent", None, route="device_control/recent")
    device_type = turn.device_type()
    row = turn.table.caps_for((turn.table.domains_for_device_type(device_type) or ("",))[0])
    unresolved_pronoun = ctx.extraction.anaphor and not ctx.session.recent_targets and not (row and row.implicit_target)
    if s == "area":
        areas, area_confs = _area_answers(turn)
        if not areas:
            if unresolved_pronoun:
                raise _which_device(turn, "device_control/area", confs)  # "toggle it" with nothing recent (S4)
            if av.has("unknown_room") and av.noul("unknown_room") >= turn.pol.unknown_room:
                raise _stop_unsupported("R_NO_DEVICE", "device_control/area")  # a room this home doesn't have
            row = turn.table.caps_for((turn.table.domains_for_device_type(device_type) or ("",))[0])
            if row is not None and row.unspecified == "active":
                # "pause" with no room named: the playing player or the room's default (spec §6.3), not every
                # speaker in the room (round 2, class 2).
                return _plan_unspecified(turn, device_type, confs)
            if ctx.speaker_area not in cat.areas:
                raise _room_clarify(turn, device_type, confs, "device_control/area")
            try:
                return _plan_for_area(turn, device_type, [ctx.speaker_area], confs, "area")
            except Stop as stop:
                if stop.decision.reply_key != "R_NO_DEVICE":
                    raise
                # The room was only the default and has no such device: use the house rule (review S6), unless
                # that can only ask "what?", in which case the device isn't here: say so.
                try:
                    return _plan_unspecified(turn, device_type, confs)
                except Stop as fallback:
                    if fallback.decision.reply_key == "R_WHAT":
                        raise stop from None
                    raise
        return _plan_for_area(turn, device_type, areas, confs + area_confs, "area")
    if s == "house":
        if unresolved_pronoun:
            raise _which_device(turn, "device_control/house", confs)
        return _plan_for_area(turn, device_type, None, confs, "house")
    if unresolved_pronoun:
        raise _which_device(turn, "device_control/unspecified", confs)  # "set it to 50" with nothing recent (class 5)
    return _plan_unspecified(turn, device_type, confs)


def named_target(turn: Turn) -> tuple[str | None, float]:
    """The target under the one_named reading: conditional on a device being named (review S5). The scope
    posterior already carries P(target is not none), so here the best device is read from the rest of the
    distribution. Returns (None, p_none) when no device stands out."""
    all_probs = target_probs(turn)
    top = max(all_probs, key=lambda o: (all_probs[o], o))
    probs = {o: p for o, p in all_probs.items() if o in turn.cat.entities}
    rest = sum(probs.values())
    if not probs or rest <= 1e-9:
        return None, all_probs.get("none", 0.0)
    best = max(probs, key=lambda o: (probs[o], o))
    cond = probs[best] / rest
    if top == best:
        return best, probs[best]
    if cond >= turn.pol.act and probs[best] >= turn.pol.clarify_min_p:  # a real, not a residual, second choice
        return best, cond
    return None, all_probs.get("none", 0.0)


def _only_device_in_named_room(turn: Turn) -> tuple[str | None, float]:
    """"Resume music in the study", "the study speaker": no target stands out, but a room is named and holds
    exactly one device of the requested kind that can do the action (round 2, class 2)."""
    areas, aconfs = _area_answers(turn)
    if not areas or not turn.av.has("device_type"):
        return None, 0.0
    cat, table = turn.cat, turn.table
    cands: list[str] = []
    for d in table.domains_for_device_type(turn.device_type()):
        row = table.caps_for(d)
        if row is None or not row.action_question or not turn.av.has(row.action_question):
            continue
        action = turn.av.choice(row.action_question).choice
        pool = music_speakers_in(cat, areas) if row.area_sweep == "music_speakers" else area_domain_targets(cat, [d], areas)
        cands += [i for i in pool if (e := cat.entities[i]) is not None and not e.is_group
                  and table.supports(e.domain, action, e.features, e.onoff_only)]
    return (cands[0], min(aconfs)) if len(cands) == 1 else (None, 0.0)


def _control_one(turn: Turn, confs: list[float]) -> Plan:
    av, cat, pol = turn.av, turn.cat, turn.pol
    tc = av.choice("target")
    route = "device_control/one_named"
    chosen, tp = named_target(turn)
    if chosen is None:
        chosen, tp = _only_device_in_named_room(turn)
    words = _word_ids(turn)
    if chosen is not None and len(words) >= 2 and chosen not in words \
            and named_by_utterance(cat, chosen, turn.ctx.extraction.text) is not True:
        chosen = None  # a guess the words don't support, while the words said fit several devices: ask
    if chosen is None:
        if _candidate_alts(turn, confs):  # a kind word the home has ("the lamp"): ask which one
            raise _which_device(turn, route, confs)
        if tc.choice not in ("none", *cat.entities) or tc.probabilities.get("none", 0.0) >= pol.none_confident:
            raise _stop_unsupported("R_NO_DEVICE", route)
        raise _which_device(turn, route, confs)
    e = cat.entities[chosen]
    if e.denied:
        raise Stop(refuse_lock(route, "4"))
    if e.is_group and e.area_id and av.has(f"area__{e.area_id}"):
        p_area = av.noul(f"area__{e.area_id}")
        if p_area >= pol.noul_yes:
            return _plan_for_area(turn, None, [e.area_id], confs + [tp, p_area], "area", domains=[e.domain])
    if not e.controllable:
        raise _stop_unsupported("R_CANT", route, name=e.name)
    co_named = alias_sharers(cat, e.entity_id, turn.ctx.extraction.text)
    if co_named:  # one alias the user gave several devices (a glob in scope.toml) means all of them
        return _build_plan(turn, [e.entity_id, *co_named], confs + [max(tp, _mass(tc, [e.entity_id, *co_named]))],
                           "named", None, route=route)
    siblings = _undecided_siblings(turn, e.entity_id)
    if siblings and plural_kind_said(cat, turn.ctx.extraction.text):
        # "the hue play lights": the plural means every device the words fit
        return _build_plan(turn, [e.entity_id, *siblings], confs + [tp], "named", None, route=route)
    plan = _build_plan(turn, [e.entity_id], confs + [tp], "one", None, route=route)
    plan.option_id = e.entity_id
    plan.alts = _target_alts(turn, tc.probabilities, confs) or []
    if siblings:
        alts = [Alt(e.entity_id, tp, plan.item())]
        for sib in siblings:
            try:
                alts.append(Alt(sib, 0.0, _build_plan(turn, [sib], confs, "one", None, route=route).item()))
            except Stop:
                continue
        if len(alts) >= 2:
            plan.alts, plan.force_pick = alts, True
    return plan


def _mass(a: ChoiceAnswer, options: Sequence[str]) -> float:
    return sum(a.probabilities.get(o, 0.0) for o in options)


def _undecided_siblings(turn: Turn, chosen: str) -> list[str]:
    """Name siblings (resolve.name_siblings) that neither a mentioned room nor the speaker's room rules out."""
    cat = turn.cat
    siblings = name_siblings(cat, chosen, turn.ctx.extraction.text)
    if not siblings:
        return []
    group = [chosen, *siblings]
    areas, _ = _area_answers(turn)
    for scope_areas in (areas, [turn.ctx.speaker_area] if turn.ctx.speaker_area else []):
        if scope_areas:
            here = [g for g in group if cat.entities[g].area_id in scope_areas]
            if len(here) == 1:
                return [] if here[0] == chosen else siblings
    return siblings


def _target_alts(turn: Turn, probabilities: Mapping[str, float], confs: list[float]) -> list[Alt]:
    cat, pol = turn.cat, turn.pol
    out = []
    for opt, p in sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0])):
        if opt == "none" or p < pol.clarify_min_p or opt not in cat.entities:
            continue
        e = cat.entities[opt]
        if e.denied or not e.controllable:
            continue
        try:
            plan = _build_plan(turn, [opt], confs + [p], "one", None, route="device_control/one_named")
        except Stop:
            continue
        out.append(Alt(opt, p, plan.item()))
    return out


def _plan_for_area(turn: Turn, device_type: str | None, areas: list[str] | None, confs: list[float], scope_kind: str,
                   domains: Sequence[str] | None = None) -> Plan:
    cat, table = turn.cat, turn.table
    route = f"device_control/{scope_kind}"
    if domains is None:
        if device_type in (None, "none", "other"):
            raise _stop_clarify("R_WHAT", route)
        domains = table.domains_for_device_type(device_type)
    targets: list[str] = []
    for d in domains:
        dc = table.caps_for(d)
        if dc is None:
            continue
        if dc.groupable:
            targets += (area_light_targets(cat, areas, turn.pol.strict_group_areas) if areas is not None
                        else house_light_targets(cat))
        elif dc.area_sweep == "music_speakers":
            targets += music_speakers_in(cat, areas)
        else:
            targets += area_domain_targets(cat, [d], areas)
    label = None
    if domains and all((c := table.caps_for(d)) is not None and c.groupable for d in domains):
        label = "all the lights" if areas is None else f"the {join_names([cat.areas[a].name for a in areas])} lights"
    # A room or house plural is a sweep: it never turns a high-stakes device on (review S2, OQ 28).
    plan = _build_plan(turn, list(dict.fromkeys(targets)), confs, scope_kind, areas, route=route, label=label,
                       sweep=True)
    plan.option_id = areas[0] if areas else None
    plan.alt_kind = "room"
    return plan


def _plan_unspecified(turn: Turn, device_type: str | None, confs: list[float]) -> Plan:
    cat, ctx, table = turn.cat, turn.ctx, turn.table
    route = "device_control/unspecified"
    domains = table.domains_for_device_type(device_type)
    if device_type in ("any", "none", "other", None) or not domains:
        raise _stop_clarify("R_WHAT", route)
    row = table.caps_for(domains[0])
    rule = row.unspecified
    if rule == "room":
        if ctx.speaker_area in cat.areas:
            return _plan_for_area(turn, device_type, [ctx.speaker_area], confs, "area")
        raise _room_clarify(turn, device_type, confs, route)
    action_id = ""
    if row.action_question and turn.av.has(row.action_question):
        action_id = turn.av.choice(row.action_question).choice
    action = table.action(domains[0], action_id)

    def capable(ids: Sequence[str]) -> list[str]:
        return [i for i in ids if (e := cat.entities.get(i)) is not None and e.controllable
                and (action is None or table.supports(e.domain, action_id, e.features, e.onoff_only))]

    pool = music_speakers_in(cat, None) if row.area_sweep == "music_speakers" else area_domain_targets(cat, domains, None)
    in_house = capable(pool)
    in_room = [i for i in in_house if cat.entities[i].area_id == ctx.speaker_area] if ctx.speaker_area else []
    if rule == "active":
        if action is not None and action.all_when_unspecified:
            targets = in_house  # "louder": the master volume (OQ 25)
        elif action is not None and action.feature and not in_house:
            targets = capable(select_source_players(cat))  # e.g. "open Netflix": the one player that can
            if len(targets) != 1:
                raise _which_device(turn, route, confs)
        else:
            playing = [p for p in playing_players(cat, ctx.states) if capable([p])]
            default = (cat.defaults.transport_player or {}).get(ctx.speaker_area or "")
            if len(playing) == 1:
                targets = playing
            elif default in cat.controllable:
                targets = [default]
            elif len(in_room) == 1:
                targets = in_room  # a new room with one speaker (review D9-8)
            else:
                raise _which_device(turn, route, confs)
        if not targets:
            raise _which_device(turn, route, confs)
        return _build_plan(turn, targets, confs, "unspecified", None, route=route)
    if rule == "only_or_room":
        targets = in_house if len(in_house) == 1 else in_room
        if not targets:
            raise _which_device(turn, route, confs)
        return _build_plan(turn, targets, confs, "unspecified", None, route=route)
    raise _which_device(turn, route, confs)


def _names(turn: Turn, ids: Sequence[str]) -> str:
    return join_names([turn.cat.name(i) for i in ids])


def _is_high_stakes(turn: Turn, entity_id: str) -> bool:
    return entity_id in set(turn.cat.defaults.high_stakes or [])


def _turns_on(turn: Turn, call: ServiceCall, entity_id: str) -> bool:
    if call.service == "turn_on":
        return True
    if call.service == "toggle":
        st = turn.ctx.states.get(entity_id)
        return not (st is not None and st.state == "on")
    return False


@dataclass
class _Built:
    built: actions.Built
    action_p: float
    skipped_high_stakes: list[str]
    high_stakes_on: bool
    only_high_stakes: bool = False  # a sweep that found nothing but high-stakes devices (review S2)


def _build_input(turn: Turn, targets: list[str], action_id: str, scope_kind: str, areas: list[str] | None,
                 label: str | None) -> actions.BuildInput:
    cat, table, pol = turn.cat, turn.table, turn.pol
    ents = [cat.entities[t] for t in targets]

    def target(e: Any) -> actions.Target:
        return actions.Target(e.entity_id, e.domain, e.name, e.features, e.onoff_only, e.capabilities)

    def extra(what: str) -> list[actions.Target]:
        if what == "tvs_in_scope":
            return [target(cat.entities[t]) for t in tvs_in(cat, areas)]
        if what == "source_players":
            return [target(cat.entities[t]) for t in select_source_players(cat)]
        return []

    domain = ents[0].domain
    return actions.BuildInput(
        table=table, domain=domain, action_id=action_id, action=table.action(domain, action_id),
        targets=[target(e) for e in ents], scope_kind=scope_kind, label=label,
        names=lambda ids: _names(turn, ids), value=turn.value, number=turn.number, answer=turn.answer,
        state=lambda eid: turn.ctx.states.get(eid), step_pct=pol.step_pct, volume_step_pct=pol.volume_step_pct,
        kelvin_step=pol.kelvin_step, extra_targets=extra, sources=cat.sources)


def _build_question(turn: Turn, question: str, targets: list[str], scope_kind: str, areas: list[str] | None,
                    label: str | None, sweep: bool, route: str) -> _Built:
    """Marginalize the action Choice: build each plausible action option, add up those giving the same calls."""
    cat = turn.cat
    answer = turn.av.choice(question)
    by_domain: dict[str, list[str]] = {}
    for t in targets:
        by_domain.setdefault(cat.entities[t].domain, []).append(t)
    results: list[_Reading] = []
    for option, p in readings(answer, turn.pol.clarify_min_p):
        try:
            b = _build_option(turn, by_domain, option, scope_kind, areas, label, sweep, route)
            sig = ("confirm",) + call_signature(b.built.calls) if b.only_high_stakes else call_signature(b.built.calls)
            results.append(_Reading(option, p, signature=sig, value=b))
        except Stop as s:
            results.append(_Reading(option, p, stop=s))
    chosen, mass = _choose(results, turn.pol.clarify_min_p, exact_catalog_name(turn))
    if chosen.stop is not None:
        raise chosen.stop
    out: _Built = chosen.value
    out.action_p = mass
    return out


def _build_option(turn: Turn, by_domain: dict[str, list[str]], option: str, scope_kind: str, areas: list[str] | None,
                  label: str | None, sweep: bool, route: str) -> _Built:
    table = turn.table
    if option == "none":
        raise _stop_clarify("R_WHAT", route)
    parts: list[actions.Built] = []
    skipped: list[str] = []
    high = False
    for domain, ids in by_domain.items():
        action = table.action(domain, option)
        row = table.caps_for(domain)
        if action is None or row is None:
            raise _stop_clarify("R_WHAT", route)
        if action.delegate == "scene_choice":
            parts.append(_delegate_scene(turn, route))
            continue
        try:
            built = actions.build(_build_input(turn, ids, option, scope_kind, areas, label), row.builder)
        except actions.ActionStop as a:
            if a.kind == "clarify":
                raise _stop_clarify(a.key, route, **a.params) from None
            raise _stop_unsupported(a.key, route, **a.params) from None
        calls = []
        for c in built.calls:
            keep = []
            for eid in c.entity_ids:
                if _is_high_stakes(turn, eid) and _turns_on(turn, c, eid):
                    if sweep:
                        skipped.append(eid)  # a room or house sweep never turns a high-stakes device on (S2)
                        continue
                    high = True
                keep.append(eid)
            if keep:
                calls.append(ServiceCall(c.domain, c.service, tuple(keep), c.data))
        built.targets = [t for t in built.targets if t not in skipped]
        built.calls = calls
        parts.append(built)
    parts = [b for b in parts if b.calls]
    if not parts:
        if skipped:
            first = turn.cat.entities[skipped[0]]
            b = actions.Built([ServiceCall(first.domain, "turn_on", tuple(skipped), {})], "R_ON",
                              {"names": _names(turn, skipped)}, [], list(skipped))
            return _Built(b, 1.0, skipped, True, only_high_stakes=True)
        raise _stop_unsupported("R_NO_DEVICE", route)
    if len(parts) == 1:
        merged = parts[0]
    else:
        acted = [t for b in parts for t in b.targets]
        if len({b.key for b in parts}) == 1:
            merged = actions.Built([c for b in parts for c in b.calls], parts[0].key,
                                   {**parts[0].params, "names": label or _names(turn, acted)},
                                   [c for b in parts for c in b.confs], acted)
        else:
            merged = actions.Built([c for b in parts for c in b.calls], "R_COMPOUND",
                                   {"parts": [[b.key, b.params] for b in parts]},
                                   [c for b in parts for c in b.confs], acted)
    return _Built(merged, 1.0, skipped, high)


def _delegate_scene(turn: Turn, route: str) -> actions.Built:
    """play_content: "play <a named script>" runs the matching scene or script (§6.4), else R_PLAY."""
    sc = turn.answer("scene")
    if sc is not None and sc.choice != "none" and p_of(sc) >= turn.pol.act and sc.choice in turn.cat.scene_options \
            and music.names_scene(turn.cat, sc.choice, turn.ctx.extraction.text):  # "play <another show>" isn't it
        plan = _activation(turn, sc.choice, [p_of(sc)], route)
        return actions.Built(plan.calls, plan.reply_key, plan.params, [p_of(sc)], plan.targets)
    raise _stop_unsupported("R_PLAY", route)


def _build_plan(turn: Turn, targets: list[str], confs: list[float], scope_kind: str, areas: list[str] | None, *,
                route: str, label: str | None = None, sweep: bool = False) -> Plan:
    cat, table = turn.cat, turn.table
    if not targets:
        raise _stop_unsupported("R_NO_DEVICE", route)
    for t in targets:
        if t in cat.entities and cat.entities[t].denied:
            raise Stop(refuse_lock(route, "4"))
    by_question: dict[str, list[str]] = {}
    for t in targets:
        e = cat.entities.get(t)
        if e is None or not e.controllable:
            continue
        row = table.caps_for(e.domain)
        if row is None or not row.action_question:
            continue
        by_question.setdefault(row.action_question, []).append(t)
    if not by_question:
        first = cat.entities.get(targets[0])
        raise _stop_unsupported("R_CANT", route, name=first.name if first else "That")
    if len(by_question) > 1 and any(turn.av.choice(q).choice not in ONOFF_ACTIONS for q in by_question):
        raise _stop_clarify("R_WHAT", route)
    results: list[_Built] = []
    only_high: list[_Built] = []
    for q, ts in by_question.items():
        try:
            b = _build_question(turn, q, ts, scope_kind, areas, label, sweep, route)
        except Stop as s:
            if len(by_question) > 1 and s.decision.reply_key in ("R_NO_DEVICE", "R_CANT"):
                continue  # in a mixed set, unsupported members are skipped (§6.4)
            raise
        (only_high if b.only_high_stakes else results).append(b)
    if not results:
        if only_high:  # the sweep found only high-stakes devices: ask, never act (review S2)
            b = only_high[0]
            return Plan(b.built.calls, b.built.key, b.built.params, list(confs) + [b.action_p], b.built.targets,
                        route, high_stakes_on=True, force_confirm=True)
        raise _stop_unsupported("R_NO_DEVICE", route)
    calls = [c for r in results for c in r.built.calls]
    all_confs = list(confs) + [r.action_p for r in results] + [c for r in results for c in r.built.confs]
    acted = [t for r in results for t in r.built.targets]
    if len(results) == 1:
        key, params = results[0].built.key, results[0].built.params
    elif len({r.built.key for r in results}) == 1 and "names" in results[0].built.params:
        key = results[0].built.key
        params = {**results[0].built.params, "names": label or _names(turn, acted)}
    else:
        key, params = "R_COMPOUND", {"parts": [[r.built.key, r.built.params] for r in results]}
    return Plan(calls, key, params, all_confs, acted, route, high_stakes_on=any(r.high_stakes_on for r in results))


# ---- scenes (§6.5) ----------------------------------------------------------------------------
def _activation(turn: Turn, key: str, confs: list[float], route: str) -> Plan:
    cat = turn.cat
    if key in cat.hue_scenes and key not in cat.scene_options_for(turn.ctx.extraction.text):
        raise _stop_unsupported("R_NO_SCENE", route)
    if key in cat.presets:
        preset = cat.presets[key]
        calls = list(preset.calls)
        name = preset.name
        targets = [t for c in calls for t in c.entity_ids]
    else:
        e = cat.entities.get(key)
        if e is None:
            raise _stop_unsupported("R_NO_SCENE", route)
        if e.denied:
            raise Stop(refuse_lock(route, "4"))
        if not e.controllable:
            raise _stop_unsupported("R_CANT", route, name=e.name)
        action = turn.table.action(e.domain, "turn_on")
        calls = [ServiceCall(e.domain, action.service if action and action.service else "turn_on", (key,), {})]
        name = e.name
        targets = [key]
    return Plan(calls, "R_SCENE", {"name": name}, list(confs), targets, route, alt_kind="scene", option_id=key)


def _same_name_in_rooms(turn: Turn, chosen: str, confs: list[float], route: str) -> Plan:
    """The same scene name in several rooms: a mentioned room or the speaker's room decides; otherwise ask
    (§6.5). The scene Choice can't tell "Study Reading" from "Hall Reading" when the user says "reading"."""
    cat = turn.cat
    siblings = [key for key in scene_siblings(cat, chosen)
                if (key in cat.hue_scenes) == (chosen in cat.hue_scenes)]
    if siblings and scene_distinguished(cat, chosen, turn.ctx.extraction.text):
        siblings = []  # the user said a word only the chosen one has (its alias, a device name): no doubt
    if not siblings:
        return _activation(turn, chosen, confs, route)
    group = [chosen, *siblings]
    areas, area_confs = _area_answers(turn)
    for scope_areas, extra in ((areas, area_confs), ([turn.ctx.speaker_area] if turn.ctx.speaker_area else [], [])):
        if scope_areas:
            here = [g for g in group if scene_area(cat, g) in scope_areas]
            if len(here) == 1:
                return _activation(turn, here[0], confs + extra, route)
    probs = turn.av.choice("scene").probabilities
    ordered = [chosen] + sorted(siblings, key=lambda o: (-probs.get(o, 0.0), o))
    alts = [Alt(o, probs.get(o, 0.0), _activation(turn, o, confs, route).item()) for o in ordered]
    raise Stop(_pick(turn, alts, "R_WHICH_SCENE", route, min(confs)))


def _scene(turn: Turn) -> Plan:
    cat, pol, ctx, av = turn.cat, turn.pol, turn.ctx, turn.av
    route = "activate_scene"
    if _denied_mass(turn, "scene") >= pol.clarify_min_p:
        raise Stop(refuse_lock(route, "4"))
    sc = av.choice("scene")
    sp = p_of(sc)
    confs = [turn.intent_p]
    if sc.choice != "none" and sc.choice in cat.entities and cat.entities[sc.choice].denied:
        raise Stop(refuse_lock(route, "4"))
    valid = set(cat.scene_options)
    if sc.choice != "none" and sc.choice in valid and sp >= pol.act:
        return _same_name_in_rooms(turn, sc.choice, confs + [sp], route)
    if sc.choice == "none" and sp >= pol.none_confident:
        raise _stop_unsupported("R_NO_SCENE", route)
    cands = sorted(((o, p) for o, p in sc.probabilities.items()
                    if o != "none" and o in valid and p >= pol.clarify_min_p
                    and not (o in cat.entities and cat.entities[o].denied)), key=lambda kv: (-kv[1], kv[0]))
    areas, area_confs = _area_answers(turn)
    narrowed = False
    if areas:
        in_area = [(o, p) for o, p in cands if cat.area_of(o) in set(areas)]
        if in_area:
            cands = in_area
            narrowed = True
    elif ctx.speaker_area:
        here = [(o, p) for o, p in cands if cat.area_of(o) == ctx.speaker_area]
        if len(here) == 1 and here[0][1] >= pol.clarify_floor:
            return _activation(turn, here[0][0], confs + [pol.act], route)
    if narrowed and len(cands) == 1 and cands[0][1] >= pol.clarify_floor:
        return _activation(turn, cands[0][0], confs + [pol.act] + area_confs, route)
    if not cands:
        raise _stop_clarify("R_WHAT", route)
    alts = [Alt(o, p, _activation(turn, o, confs + [p], route).item()) for o, p in cands]
    best = _activation(turn, cands[0][0], confs + [cands[0][1]], route)
    best.alts = alts
    if len(cands) >= 2:
        raise Stop(_clarify_from(turn, best))
    return best


# ---- queries (§6.6) ---------------------------------------------------------------------------
def _answered(key: str, route: str, params: Mapping[str, Any], reads: Sequence[ReadCall], conf: float | None,
              targets: Sequence[str] = ()) -> Decision:
    return Decision(outcome=Outcome.ANSWERED, reply_key=key, reply_params=dict(params), reads=tuple(reads),
                    route=route, rule="8", confidence=conf, targets=tuple(targets))


def _state_read(ids: Sequence[str]) -> ReadCall:
    domains = {caps.domain_of(i) for i in ids}
    return ReadCall("state", domains.pop() if len(domains) == 1 else "mixed", None, tuple(ids))


def _parse_ts(value: Any, now: datetime) -> datetime | None:
    if not isinstance(value, str) or value in ("unknown", "unavailable", ""):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=now.tzinfo)
    return dt.astimezone(now.tzinfo) if now.tzinfo else dt


def _query(turn: Turn) -> Decision:
    cat, ctx, av = turn.cat, turn.ctx, turn.av
    qa = av.choice("query_attribute")
    attr = qa.choice
    conf = min(turn.intent_p, p_of(qa))
    route = f"device_query/{attr}"
    if attr == "items":
        return _list_read(turn)
    if attr in ("sunrise", "sunset"):
        suns = sorted(e for e, ce in cat.entities.items() if ce.domain in turn.table.with_role("sun"))
        if not suns:
            return _unsupported("R_CANT", route, name="That")
        st = ctx.states.get(suns[0])
        dt = _parse_ts((st.attributes if st else {}).get("next_rising" if attr == "sunrise" else "next_setting"), ctx.now)
        if dt is None:
            return _answered("Q_UNKNOWN", route, {"name": cat.name(suns[0])}, [_state_read(suns)], conf)
        return _answered("Q_SUN", route, {"event": attr.capitalize(), "at": dt, "now": ctx.now},
                         [_state_read(suns)], conf)
    if attr in ("next_alarm", "next_timer"):
        word = attr.split("_", 1)[1]
        listed = set(getattr(cat.defaults, f"{word}_sensors", None) or [])  # D9-9: config beats naming
        sensors = sorted(e for e, ce in cat.entities.items()
                         if e in listed or caps.is_timestamp_kind(ce.domain, ce.device_class, ce.name, word))
        best = None
        for s in sensors:
            st = ctx.states.get(s)
            dt = _parse_ts(st.state if st else None, ctx.now)
            if dt is not None and dt >= ctx.now and (best is None or dt < best):
                best = dt
        reads = [_state_read(sensors)] if sensors else []
        if best is None:
            return _answered("Q_NO_ALARM", route, {"kind": word}, reads, conf)
        return _answered("Q_NEXT_ALARM", route, {"kind": word, "at": best, "now": ctx.now}, reads, conf)
    if attr == "motion":
        areas, aconfs = _area_answers(turn)
        if not areas and ctx.speaker_area in cat.areas:
            areas = [ctx.speaker_area]
        sensors = sorted(e for e, ce in cat.entities.items()
                         if caps.is_motion_sensor(ce.domain, ce.device_class) and (not areas or ce.area_id in areas))
        if not sensors:
            return _unsupported("R_CANT", route, name="That")
        area_name = join_names([cat.areas[a].name for a in areas]) if areas else "home"
        moving = any(ctx.states.get(s) and ctx.states[s].state == "on" for s in sensors)
        return _answered("Q_MOTION_YES" if moving else "Q_MOTION_NO", route, {"area": area_name}, [_state_read(sensors)],
                         min([conf, *aconfs]))
    if attr in ("color", "other"):
        name = "That"
        if av.has("target") and av.choice("target").choice in cat.entities:
            name = cat.entities[av.choice("target").choice].name
        return _unsupported("R_CANT", route, name=name)

    targets, scope_kind, label_area, tconfs = _query_targets(turn, attr)
    conf = min([conf, *tconfs])
    if scope_kind in ("area", "house") and attr == "on_off":
        return _area_on_off(turn, targets, scope_kind, label_area, route, conf)
    if len(targets) > 1:  # several named devices, or everything touched last time (review S-2)
        parts, reads, answered = [], [], []
        for t in targets:
            d = _one_attribute(turn, attr, t, route, conf)
            if d.outcome != Outcome.ANSWERED:
                continue
            parts.append([d.reply_key, dict(d.reply_params)])
            reads += list(d.reads)
            answered.append(t)
        if not parts:
            return _unsupported("R_CANT", route, name=cat.name(targets[0]))
        return _answered("Q_MULTI", route, {"parts": parts}, reads, conf, answered)
    return _one_attribute(turn, attr, targets[0], route, conf)


def _one_attribute(turn: Turn, attr: str, target: str, route: str, conf: float) -> Decision:
    source = _attr_source(turn, attr, target)
    if source is None:
        if attr == "power":
            source, attr = target, "on_off"
        else:
            return _unsupported("R_CANT", route, name=turn.cat.name(target))
    return _render_attribute(turn, attr, source, route, conf)


def _query_targets(turn: Turn, attr: str) -> tuple[list[str], str, str | None, list[float]]:
    cat, ctx, av, pol, table = turn.cat, turn.ctx, turn.av, turn.pol, turn.table
    route = f"device_query/{attr}"
    scope = av.choice("scope")
    post = scope_posterior(turn, scope)  # the same evidence-weighted reading as for commands (review S5)
    s = max(post, key=lambda o: (post[o], o == scope.choice))
    confs = [post[s]]
    if s == "several_named":
        chosen = [e for e, _ in _mentioned(turn)]
        if chosen:
            return chosen, "named", None, confs
        s = "one_named"
    if s == "one_named":
        target, tp = named_target(turn)
        if target is not None:
            return [target], "one", None, confs + [tp]
        fallback = _only_source(turn, attr)
        if fallback:
            return [fallback], "one", None, confs
        tc = av.choice("target")
        if tc.probabilities.get("none", p_of(tc) if tc.choice == "none" else 0.0) >= pol.none_confident:
            raise _stop_unsupported("R_NO_DEVICE", route)
        raise _which_device(turn, route)
    if s == "recent":
        recent = [r for r in ctx.session.recent_targets if r in cat.entities]
        if recent:
            return recent, "named", None, confs
        raise _which_device(turn, route)
    device_type = turn.device_type()
    domains = table.domains_for_device_type(device_type)
    groupable = any((c := table.caps_for(d)) is not None and c.groupable for d in domains)
    if s in ("area", "house") or (s == "unspecified" and groupable):
        areas, aconfs = _area_answers(turn) if s == "area" else ([], [])
        if s == "area" and not areas and ctx.speaker_area in cat.areas:
            areas = [ctx.speaker_area]
        if s == "unspecified" and ctx.speaker_area in cat.areas:
            areas, s = [ctx.speaker_area], "area"
        kind = "area" if areas else "house"
        ids = []
        for d in domains or ():
            c = table.caps_for(d)
            if c is not None and c.groupable:
                base = area_light_targets(cat, areas, pol.strict_group_areas) if areas else house_light_targets(cat)
                ids += expand_groups(cat, base, areas if areas and pol.strict_group_areas else None)
            else:
                ids += [e for e, ce in cat.entities.items() if ce.domain == d and (not areas or ce.area_id in areas)]
        if not ids:
            fallback = _only_source(turn, attr)
            if fallback:
                return [fallback], "one", None, confs
            raise _which_device(turn, route)
        label = join_names([cat.areas[a].name for a in areas]) if areas else None
        if attr not in ("on_off", "temperature", "target_temperature", "hvac_mode"):
            return ids[:1], "one", label, confs + aconfs
        return ids, kind, label, confs + aconfs
    # An unqualified indoor temperature query follows the speaker's room.
    if s == "unspecified":
        here = [e for e, ce in cat.entities.items() if ce.domain in domains
                and ce.area_id == ctx.speaker_area and _serves(turn, ce, attr)]
        if len(here) == 1:
            return here, "one", None, confs
    # unspecified, non-light
    fallback = _only_source(turn, attr)
    if fallback:
        return [fallback], "one", None, confs
    if any((c := table.caps_for(d)) is not None and c.unspecified == "active" for d in domains):
        playing = playing_players(cat, ctx.states)
        if len(playing) == 1:
            return playing, "one", None, confs
        default = (cat.defaults.transport_player or {}).get(ctx.speaker_area or "")
        if default in cat.entities:
            return [default], "one", None, confs
        here = music_speakers_in(cat, [ctx.speaker_area]) if ctx.speaker_area else []
        if len(here) == 1:
            return here, "one", None, confs
    raise _which_device(turn, route)


def _serves(turn: Turn, ce: Any, attr: str) -> bool:
    if attr == "locked":
        return caps.is_lock_like(ce.domain)
    if attr in ("open_closed", "position"):
        return caps.is_door_sensor(ce.domain, ce.device_class) or (caps.is_cover(ce.domain) and not caps.is_lock_like(ce.domain))
    if attr == "power":
        return caps.is_power_sensor(ce.domain, ce.device_class)
    c = turn.table.caps_for(ce.domain)
    return c is not None and attr in c.query_attributes


def _only_source(turn: Turn, attr: str) -> str | None:
    cands = sorted(e for e, ce in turn.cat.entities.items() if _serves(turn, ce, attr))
    if attr in ("open_closed", "position"):
        covers = [c for c in cands if caps.is_cover(turn.cat.entities[c].domain)]
        if attr == "position" and len(covers) == 1:
            return covers[0]
    if len(cands) > 1:  # "is the door open?": the words said pick the door sensor over the blind (review S10)
        worded = [c for c in word_candidates_any(turn.cat, turn.ctx.extraction.text) if c in cands]
        if len(worded) == 1:
            return worded[0]
    return cands[0] if len(cands) == 1 else None


def _attr_source(turn: Turn, attr: str, target: str) -> str | None:
    cat = turn.cat
    ce = cat.entities.get(target)
    if ce is None:
        return None
    c = turn.table.caps_for(ce.domain)
    if attr in ("on_off", "brightness", "playing", "volume") and c is not None and attr in c.query_attributes:
        return target
    if _serves(turn, ce, attr):
        return target
    same_device = [e for e, x in cat.entities.items() if x.device_id and x.device_id == ce.device_id and _serves(turn, x, attr)]
    if same_device:
        return sorted(same_device)[0]
    if attr == "power":
        return None
    same_area = [e for e, x in cat.entities.items() if ce.area_id and x.area_id == ce.area_id and _serves(turn, x, attr)]
    if len(same_area) == 1:
        return same_area[0]
    return _only_source(turn, attr)


def _area_on_off(turn: Turn, ids: list[str], scope_kind: str, area_label: str | None, route: str, conf: float) -> Decision:
    ctx, cat = turn.ctx, turn.cat
    on, known = [], 0
    for i in ids:
        st = ctx.states.get(i)
        if st is None or st.state in (None, "unknown", "unavailable"):
            continue
        known += 1
        if st.state == "on":
            on.append(i)
    reads = [_state_read(ids)]
    names = join_names([cat.name(i) for i in on])
    if scope_kind == "area":
        if not on:
            return _answered("Q_AREA_ALL_OFF", route, {"area": area_label}, reads, conf, ids)
        if len(on) == known:
            return _answered("Q_AREA_ALL_ON", route, {"area": area_label}, reads, conf, ids)
        return _answered("Q_AREA_SOME_ON", route, {"area": area_label, "names": names}, reads, conf, ids)
    if not on:
        return _answered("Q_HOUSE_ALL_OFF", route, {}, reads, conf, ids)
    return _answered("Q_HOUSE_SOME_ON", route, {"names": names}, reads, conf, ids)


def _render_attribute(turn: Turn, attr: str, eid: str, route: str, conf: float) -> Decision:
    ctx, cat = turn.ctx, turn.cat
    ce = cat.entities[eid]
    st = ctx.states.get(eid)
    reads = [_state_read([eid])]
    name = ce.name
    state = st.state if st else None
    a = dict(st.attributes) if st else {}

    def ans(key: str, **params: Any) -> Decision:
        return _answered(key, route, params, reads, conf, [eid])

    if state in (None, "unknown", "unavailable"):
        return ans("Q_UNKNOWN", name=name)
    if attr in ("temperature", "target_temperature"):
        value = a.get("current_temperature" if attr == "temperature" else "temperature")
        try:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError):
            return ans("Q_ROOM_TEMP_UNKNOWN" if attr == "temperature" else "Q_UNKNOWN", name=name)
        return ans("Q_TEMPERATURE" if attr == "temperature" else "Q_TARGET_TEMPERATURE",
                   name=name, value=actions._num(value), unit=a.get("temperature_unit", "degrees"))
    if attr == "hvac_mode":
        return ans("Q_HVAC_MODE", name=name, mode={"heat": "heat", "cool": "cool", "off": "off",
                   "fan_only": "fan only", "auto": "automatic", "heat_cool": "automatic heating and cooling"}.get(state, state))
    if attr == "locked":
        return ans({"locked": "Q_LOCKED", "unlocked": "Q_UNLOCKED", "jammed": "Q_JAMMED"}.get(state, "Q_UNKNOWN"),
                   name=name)
    if attr in ("open_closed", "position"):
        if caps.is_cover(ce.domain):
            pos = a.get("current_position")
            if pos is None:
                return ans("Q_OPEN" if state == "open" else "Q_CLOSED", name=name)
            pos = int(pos)
            if pos <= 0:
                return ans("Q_CLOSED", name=name)
            if pos >= 100:
                return ans("Q_OPEN", name=name)
            return ans("Q_COVER_PCT", name=name, n=pos)
        return ans("Q_OPEN" if state in ("on", "open") else "Q_CLOSED", name=name)
    if attr == "power":
        try:
            watts = float(state)
        except ValueError:
            return ans("Q_UNKNOWN", name=name)
        return ans("Q_POWER", name=name, w=actions._num(round(watts, 1)))
    if attr == "brightness":
        if state != "on":
            return ans("Q_LIGHT_OFF", name=name)
        b = a.get("brightness")
        if b is None:
            return ans("Q_ON_OFF", name=name, state="on")
        return ans("Q_BRIGHTNESS", name=name, pct=int(round(float(b) / 2.55)))
    if attr == "playing":
        if state == "playing":
            return ans("Q_PLAYING", name=name, title=a.get("media_title") or a.get("app_name") or "something",
                       artist=a.get("media_artist"))
        if state == "paused":
            return ans("Q_PAUSED", name=name)
        return ans("Q_NOTHING_PLAYING", name=name)
    if attr == "volume":
        v = a.get("volume_level")
        if v is None:
            return ans("Q_UNKNOWN", name=name)
        return ans("Q_VOLUME", name=name, pct=int(round(float(v) * 100)))
    return ans("Q_ON_OFF", name=name, state="off" if state == "off" else "on")


# ---- weather and lists (§6.7) -----------------------------------------------------------------
def weather_entity(cat: Catalog) -> str | None:
    """`defaults.weather` if set, else the only weather entity with a forecast, else the first by id (D9-5)."""
    weathers = sorted(e for e, ce in cat.entities.items() if ce.domain in cat.caps.with_role("weather"))
    default = getattr(cat.defaults, "weather", None)
    if default in weathers:
        return default
    forecasting = [w for w in weathers if cat.entities[w].features & {"forecast_daily", "forecast_hourly"}]
    if len(forecasting) == 1:
        return forecasting[0]
    return weathers[0] if weathers else None


def _weather(turn: Turn) -> Decision:
    cat, av = turn.cat, turn.av
    when = av.choice("weather_when")
    conf = min(turn.intent_p, p_of(when))
    if when.choice == "elsewhere":
        return _luna("luna/weather_elsewhere", "8", conf)
    w = weather_entity(cat)
    if w is None:
        return _unsupported("R_CANT", "weather", name="The weather")
    period = when.choice
    text = turn.ctx.extraction.text
    if (period in ("now", "none")
            and re.search(r"\b(?:temp(?:erature)?|how (?:hot|cold|warm|cool))\b", text, re.IGNORECASE)
            and not re.search(r"\b(?:now|currently|current|at the moment)\b", text, re.IGNORECASE)):
        period = "today"
    route = f"weather/{period}"
    if period in ("now", "none"):
        return _answered("W_NOW", route, {"entity": w}, [_state_read([w])], conf)
    forecast_type = "hourly" if period in ("next_hours", "tonight") else "daily"
    key = {"next_hours": "W_NEXT_HOURS", "today": "W_DAY", "tomorrow": "W_DAY", "tonight": "W_TONIGHT",
           "this_week": "W_WEEK"}[period]
    domain, service = caps.WEATHER_FORECAST_SERVICE
    return _answered(key, route, {"entity": w, "when": period},
                     [ReadCall("service", domain, service, (w,), {"type": forecast_type})], conf)


def _pick_list(turn: Turn) -> tuple[str | None, list[float]]:
    cat, pol, av = turn.cat, turn.pol, turn.av
    if av.has("todo_list"):
        tl = av.choice("todo_list")
        if tl.choice != "none" and tl.choice in cat.todo_lists and p_of(tl) >= pol.act:
            return tl.choice, [p_of(tl)]
    default = cat.defaults.list_entity
    return (default if default in cat.entities else None), []


def _list_add(turn: Turn) -> Decision:
    x = turn.ctx.extraction
    item = (x.list_item or {}).get("item")
    if not item:
        return _clarify("R_WHAT_ITEM", "list_add", "8", turn.intent_p)
    lst, confs = _pick_list(turn)
    if lst is None:
        return _unsupported("R_NO_DEVICE", "list_add")
    conf = min([turn.intent_p, *confs])
    if conf < turn.pol.act:  # list_add is banded like every other executing path (review S11, OQ 53)
        call = ServiceCall(*caps.TODO_ADD_SERVICE, (lst,), {"item": item})
        item_plan = PlanItem((call,), "R_ADDED", {"item": item, "list": turn.cat.name(lst).lower()}, (lst,))
        if conf >= turn.pol.clarify_floor:
            return _confirm(turn, Alt(lst, conf, item_plan), "list_add", conf)
        return _clarify("R_WHAT", "list_add", "9", conf)
    domain, service = caps.TODO_ADD_SERVICE
    call = ServiceCall(domain, service, (lst,), {"item": item})
    return Decision(outcome=Outcome.EXECUTED, calls=(call,), reply_key="R_ADDED",
                    reply_params={"item": item, "list": turn.cat.name(lst).lower()}, route="list_add", rule="8",
                    confidence=conf, targets=(lst,))


def _list_read(turn: Turn) -> Decision:
    lst, confs = _pick_list(turn)
    if lst is None:
        return _unsupported("R_NO_DEVICE", "list_read")
    domain, service = caps.TODO_ITEMS_SERVICE
    return _answered("R_LIST_READ", "list_read", {"entity": lst, "list": turn.cat.name(lst)},
                     [ReadCall("service", domain, service, (lst,))], min([turn.intent_p, *confs]))


ACTING_INTENTS = {"device_control": "_control", "activate_scene": "_scene"}


def _act(turn: Turn) -> Plan:
    """device_control and activate_scene readings that run the same calls add up (review S5): "put on the news
    podcast" plays a script whether it is read as a device command or as a scene."""
    intent = turn.av.choice("intent")
    probs = {o: p for o, p in (intent.probabilities or {intent.choice: turn.intent_p}).items() if o in ACTING_INTENTS}
    probs.setdefault(intent.choice, turn.intent_p)
    results = []
    for option, p in sorted(probs.items(), key=lambda kv: (kv[0] != intent.choice, -kv[1])):
        if option != intent.choice and p < turn.pol.clarify_min_p:
            continue
        sub = Turn(turn.ctx, turn.av, turn.numbers, 1.0)
        try:
            plan = _control(sub) if option == "device_control" else _scene(sub)
            sig = ("confirm",) + call_signature(plan.calls) if plan.force_confirm else call_signature(plan.calls)
            results.append(_Reading(option, p, signature=sig, value=plan))
        except Stop as stop:
            results.append(_Reading(option, p, stop=stop))
    if len(results) == 1:  # the usual case: nothing to add up, so nothing changes
        r = results[0]
        if r.stop is not None:
            raise r.stop
        r.value.confs[0] = turn.intent_p
        return r.value
    chosen, mass = _choose(results, turn.pol.clarify_min_p, exact_catalog_name(turn))
    if chosen.option != intent.choice and chosen.stop is None and mass < turn.intent_p:
        chosen = next(r for r in results if r.option == intent.choice)  # never trade the chosen intent for less
        mass = turn.intent_p
    if chosen.stop is not None:
        raise chosen.stop
    plan: Plan = chosen.value
    plan.confs[0] = mass
    return plan


def _music(turn: Turn) -> Decision | None:
    """Hook (music.py): a free-form "play <music>" request; None leaves it to the paths below."""
    ctx = turn.ctx
    intent = turn.av.choice("intent")
    acting = sum(p for o, p in (intent.probabilities or {intent.choice: turn.intent_p}).items() if o in ACTING_INTENTS)
    return music.decide(ctx.catalog, ctx.extraction.text, turn.av, turn.pol, ctx.speaker_area,
                        max(acting, turn.intent_p))


def _calendar(turn: Turn) -> Decision:
    """Hook (calendars.py)."""
    ctx = turn.ctx
    return calendars.decide(ctx.catalog, ctx.extraction.text, turn.av, turn.pol, ctx.now, ctx.people, ctx.person,
                            turn.intent_p)


_INTENT_HANDLERS = {
    "device_control": lambda turn: _music(turn) or _band(turn, _act(turn)),
    "activate_scene": lambda turn: _music(turn) or _band(turn, _act(turn)),
    calendars.INTENT: _calendar,
    "device_query": _query,
    "weather": _weather,
    "list_add": _list_add,
    "list_read": _list_read,
    "timer_alarm_set": _timers,
    "nevermind": _nevermind,
}
# --- memory intents (memory.py, SPEC §17); "no, forget that" after a remember undoes it even as `nevermind`.
_INTENT_HANDLERS.update(memory_intents.HANDLERS)
_INTENT_HANDLERS["nevermind"] = memory_intents.undo_or(_nevermind)
