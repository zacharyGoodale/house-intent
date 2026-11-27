"""Core.create(...); await core.handle(text, speaker_area=None, session_id=None, person=None) -> Result (spec §2, §13)."""

from __future__ import annotations

import asyncio
import re
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ha_jev import calendars, music, schedules
from ha_jev import memory as mem
from ha_jev.capabilities import CALENDAR_EVENTS_SERVICE
from ha_jev.catalog import Catalog
from ha_jev.config import Person, Settings
from ha_jev.executor import execute
from ha_jev.extract import Extraction, extract, is_question
from ha_jev.ha.interface import HAClient
from ha_jev.jev import ChoiceAnswer, JevClient, JevError, JevResponse, NoulAnswer
from ha_jev.luna import LunaClient, LunaError, LunaProfile
from ha_jev.models import (
    Debug,
    Decision,
    JevTrace,
    LunaTrace,
    Outcome,
    ReadCall,
    Result,
    ServiceCall,
)
from ha_jev.obslog import RequestLog
from ha_jev.policy import DecisionContext, SessionSnapshot, decide
from ha_jev.profiles import speaker_sentence
from ha_jev.questions import AnswerView, MissingAnswer, RequestContext, build_request, build_retry_request
from ha_jev.replies import NoData, join_names, render_list, render_template, render_weather, response_mode
from ha_jev.safety import SafetyViolation, guard_read, music_reads, pre_gate
from ha_jev.session import SessionStore

PRUNABLE = ("target", "scene")


def _jev_error_text(e: JevError) -> str:
    """The error plus its HTTP body (§13.2 "log the body"; review ST-2): a 422 names the bad question."""
    if e.body in (None, "", {}):
        return str(e)
    return f"{e}: {e.body}"


def sanitize_answers(answers: dict, questions: dict[str, dict]) -> dict:
    """Keep only answers to questions we asked; a Choice naming an option we never offered becomes a
    zero-confidence `none` (or its best offered option), so no model output can introduce a new target."""
    out: dict = {}
    for qid, q in questions.items():
        a = answers.get(qid)
        if a is None:
            continue
        if q.get("type") == "choice":
            if not isinstance(a, ChoiceAnswer):
                continue
            offered = set(q["criteria"])
            probs = {k: v for k, v in a.probabilities.items() if k in offered}
            if a.choice not in offered:
                fallback = "none" if "none" in offered else (max(probs, key=probs.get) if probs else next(iter(offered)))
                a = ChoiceAnswer(fallback, 0.0, probs)
            elif len(probs) != len(a.probabilities):
                a = ChoiceAnswer(a.choice, a.confidence, probs)
        elif not isinstance(a, NoulAnswer):
            continue
        out[qid] = a
    return out


class Core:
    def __init__(self, settings: Settings, ha: HAClient, jev: JevClient, luna: LunaClient, catalog: Catalog, *,
                 clock: Callable[[], datetime] | None = None, log: RequestLog | None = None,
                 sessions: SessionStore | None = None, memory: mem.MemoryStore | None = None) -> None:
        self.settings = settings
        self.ha = ha
        self.jev = jev
        self.luna = luna
        self.catalog = catalog
        self._clock = clock or (lambda: datetime.now().astimezone())
        self.log = log
        self.sessions = sessions or SessionStore(settings.policy.pending_ttl_s, settings.policy.recent_ttl_s)
        # Memory (memory.py). Only the CLI/server pass the on-disk store; tests and the eval stay in memory.
        self.memory = memory if memory is not None else mem.MemoryStore()
        self.music_entry: str | None = None  # the music integration's config entry (its search needs it)
        self.calendar_cache = calendars.EventCache()

    async def _find_music_entry(self) -> None:
        find = getattr(self.ha, "config_entry_id", None)
        integration = self.settings.scope.music.integration
        if integration and callable(find):
            try:
                self.music_entry = await find(integration)
            except Exception:  # noqa: BLE001 - music then reports "not set up" (R_PLAY) instead of failing startup
                self.music_entry = None

    @classmethod
    async def create(cls, settings: Settings, ha: HAClient, jev: JevClient, luna: LunaClient, *,
                     clock: Callable[[], datetime] | None = None, log: RequestLog | None = None,
                     sessions: SessionStore | None = None, memory: mem.MemoryStore | None = None) -> Core:
        catalog = Catalog.build(await ha.get_registry(), await ha.get_states(), settings.scope, settings.safety,
                                settings.capabilities)
        core = cls(settings, ha, jev, luna, catalog, clock=clock, log=log, sessions=sessions, memory=memory)
        await core._find_music_entry()
        listen = getattr(ha, "set_registry_listener", None)
        if callable(listen):  # RealHA: rebuild the catalog when the registry changes (§4.1 rule 1, review S-8)
            listen(core.refresh_catalog)
        return core

    async def refresh_catalog(self) -> None:
        """Rebuild after a registry change (RealHA calls this on registry-update events)."""
        self.catalog = Catalog.build(await self.ha.get_registry(), await self.ha.get_states(),
                                     self.settings.scope, self.settings.safety, self.settings.capabilities)

    # ------------------------------------------------------------------------------------------
    async def handle(self, text: str, speaker_area: str | None = None, session_id: str | None = None,
                     person: str | None = None, device_id: str | None = None) -> Result:
        """`person` is a configured person id (config/people.toml) when the speaker is known, else None."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        request_id = uuid.uuid4().hex
        area_id = self.catalog.area_id_for(speaker_area)
        run = _Run(self, request_id, text, speaker_area, area_id, session_id, t0, timings,
                   self.settings.person(person))
        run.device_id = device_id
        try:
            return await run.go()
        except Exception as e:  # noqa: BLE001 - never crash the voice path
            run.error = f"unexpected: {type(e).__name__}: {e}"
            run.traceback = traceback.format_exc()
            decision = Decision(outcome=Outcome.ERROR, reply_key="R_WHAT", route="error/unexpected", rule="error")
            return run.finish(decision, render_template("R_WHAT"), [], [])


class _Run:
    """One request's mutable bookkeeping."""

    def __init__(self, core: Core, request_id: str, text: str, speaker_area: str | None, area_id: str | None,
                 session_id: str | None, t0: float, timings: dict[str, float], person: Person | None = None) -> None:
        self.device_id = None
        self.core = core
        self.request_id = request_id
        self.text = text
        self.speaker_area = speaker_area
        self.area_id = area_id
        self.session_id = session_id
        self.speaker = person  # the known speaker, else None
        self.t0 = t0
        self.timings = timings
        self.extraction: Extraction | None = None
        self.jev_traces: list[JevTrace] = []
        self.luna_trace: LunaTrace | None = None
        self.reads: list[ReadCall] = []
        self.failed: list[dict] = []
        self.safety_violation = False
        self.error: str | None = None
        self.traceback: str | None = None
        self.session_before: SessionSnapshot = SessionSnapshot()
        self.memory_op: dict | None = None  # the memory write this request made, for the log

    def _lap(self, name: str, start: float) -> float:
        now = time.perf_counter()
        self.timings[name] = round((now - start) * 1000, 3)
        return now

    async def go(self) -> Result:
        core = self.core
        s = core.settings
        start = time.perf_counter()
        x = extract(self.text)
        self.extraction = x
        start = self._lap("extract", start)

        if not x.text:
            return self.finish(Decision(Outcome.UNSUPPORTED, reply_key="R_EMPTY", route="empty", rule="0"), None, [], [])
        explicit_hermes = re.search(r"\bhermes\b", self.text, re.IGNORECASE) is not None
        if explicit_hermes:
            if not getattr(core.luna, "sticky_conversation", False):
                return self.finish(Decision(Outcome.UNSUPPORTED, route="hermes/unavailable"),
                                   "Hermes is not enabled for this voice service.", [], [])
            snapshot = core.sessions.snapshot(self.session_id)
            self.session_before = snapshot
            core.sessions.clear_pending(self.session_id)
            now = core._clock()
            return await self._answer_fallback(
                Decision(Outcome.ANSWERED, reply_key="R_LUNA", route="hermes/explicit",
                         rule="hermes_explicit", use_luna=True),
                snapshot, self._memory_view(now), now, start)
        if getattr(core.luna, "sticky_conversation", False) and schedules.parse(x.text) is None:
            response_id = core.sessions.hermes_response(self.session_id, self.speaker.id if self.speaker else None)
            if response_id:
                snapshot = core.sessions.snapshot(self.session_id)
                self.session_before = snapshot
                now = core._clock()
                return await self._answer_fallback(
                    Decision(Outcome.ANSWERED, reply_key="R_LUNA", route="hermes/conversation",
                             rule="hermes_session", use_luna=True),
                    snapshot, self._memory_view(now), now, start)
        # A question may carry a lock command in another part ("what's on today and unlock the door"): each
        # part gets the pre-gate on its own, so the question form of the first never shields the second.
        if pre_gate(x.text, x.question_form) or any(pre_gate(p, is_question(p)) for p in x.parts):
            return self.finish(Decision(Outcome.REFUSED, reply_key="R_LOCK", route="pre_gate/lock", rule="1"), None, [], [])
        if x.too_many_parts:
            return self.finish(Decision(Outcome.CLARIFY, reply_key="R_ONE_AT_A_TIME", route="compound/too_many", rule="0"),
                               None, [], [])

        states = await core.ha.get_states()
        now = core._clock()
        snapshot = core.sessions.snapshot(self.session_id)
        self.session_before = snapshot
        core.sessions.clear_pending(self.session_id)
        pending = None
        if snapshot.pending is not None:
            pending = {"kind": snapshot.pending.kind, "question": snapshot.pending.question,
                       "options": dict(snapshot.pending.options)}
        memory_view = self._memory_view(now)  # --- memory (memory.py)
        # people.toml may name calendars before their HA integrations are installed.
        # Offer local calendar routing only for entities that actually have usable state.
        calendar_people = {
            pid: replace(p, calendars=[eid for eid in calendars.entities_for(p, core.catalog)
                                      if eid in states and states[eid].state not in (None, "unknown", "unavailable")])
            for pid, p in s.people.items()
        }
        rctx = RequestContext(x, core.catalog, self.area_id, now, list(snapshot.recent_targets), pending, s.policy,
                              memory_options=mem.forget_options(x.text, memory_view.facts) if memory_view else None,
                              people=calendar_people)
        req = build_request(rctx)
        start = self._lap("build", start)

        try:
            resp = await self._ask(req.state, req.questions)
        except JevError as e:
            self.error = f"jev: {_jev_error_text(e)}"
            self._lap("jev", start)
            return self.finish(Decision(Outcome.ERROR, reply_key="R_JEV_DOWN", route="error/jev_down", rule="2"), None, [], [])
        answers = sanitize_answers(resp.answers, req.questions)
        # §3.5 pruning retry: one more request over the next options when the pruned Choice found nothing.
        for qid in PRUNABLE:
            rest = req.overflow.get(qid)
            ans = answers.get(qid)
            if rest and isinstance(ans, ChoiceAnswer) and (ans.choice == "none" or ans.confidence < s.policy.clarify_floor):
                retry = build_retry_request(rctx, qid, rest[: s.policy.prune_keep])
                try:
                    r2 = await self._ask(retry.state, retry.questions)
                except JevError as e:
                    self.error = f"pruning retry failed: {_jev_error_text(e)}"  # keep the first answers (ST-6)
                    break
                fixed = sanitize_answers(r2.answers, retry.questions)
                if isinstance(fixed.get(qid), ChoiceAnswer) and fixed[qid].choice != "none":
                    answers[qid] = fixed[qid]
                break
        start = self._lap("jev", start)

        dctx = DecisionContext(utterance=self.text, extraction=x, catalog=core.catalog, states=states,
                               speaker_area=self.area_id, session=snapshot, policy=s.policy, now=now,
                               memory=memory_view, people=calendar_people, person=self.speaker)
        try:
            decision = decide(dctx, AnswerView(answers))
        except MissingAnswer as e:
            self.error = f"jev_missing_answer: {e}"
            self._lap("policy", start)
            return self.finish(Decision(Outcome.ERROR, reply_key="R_JEV_DOWN", route="error/jev_missing_answer",
                                        rule="error"), None, [], [])
        if music.is_play(decision):  # hook: the search needs the integration's config entry
            decision = music.prepare(decision, core.music_entry)
        start = self._lap("policy", start)
        if decision.memory is not None:  # --- memory write (memory.py), after the pure decision
            decision = mem.apply(decision, core.memory, self.speaker, self.area_id, now)
            self._lap("memory", start)
            return self.finish(decision, None, [], [])

        if decision.schedule is not None:
            return await self._schedule(decision)
        if decision.reply_key in ("Q_NO_ALARM", "Q_NEXT_ALARM"):
            operation = {"op": "query", "kind": decision.reply_params["kind"]}
            return await self._schedule(replace(decision, schedule=operation), fallback=True)
        if decision.use_luna:
            return await self._answer_fallback(decision, snapshot, memory_view, now, start)

        # Reads (weather forecasts, todo items); state reads are logged only.
        reply_text: str | None = None
        read_results: dict[str, Any] = {}
        for rc in decision.reads:
            self.reads.append(rc)
            if rc.kind != "service":
                continue
            try:
                guard_read(rc.domain, rc.service, music_reads(core.catalog.music_integration))
            except SafetyViolation as v:
                self.safety_violation = True
                self.error = f"safety_violation: {v}"
                return self.finish(Decision(Outcome.ERROR, reply_key="R_WHAT", route="guard/read", rule="6.1.3"),
                                   None, [], [])
            try:
                result = core.calendar_cache.get(rc)
                if result is None:
                    result = await core.ha.call_service(
                        rc.domain, rc.service or "", list(rc.entity_ids), dict(rc.data), return_response=True)
                    core.calendar_cache.put(rc, result)
                read_results[f"{rc.domain}.{rc.service}"] = result
            except Exception as e:  # noqa: BLE001
                self.error = f"ha read failed: {type(e).__name__}: {e}"
                names = join_names([core.catalog.name(i) for i in rc.entity_ids]) or "Home Assistant"
                self._lap("reads", start)
                return self.finish(replace(decision, outcome=Outcome.ERROR, reply_key="R_HA_FAIL",
                                           reply_params={"names": names}), None, [], [])
        if music.is_play(decision):  # hook: choose from the search results, in code
            decision = music.resolve(decision, read_results, s.policy.clarify_max_options)
        try:
            reply_text = self._render_with_reads(decision, states, read_results, now)
        except NoData as e:  # HA answered, but without what the reply needs: an HA-side failure
            self.error = f"render failed: {e}"
            names = join_names([core.catalog.name(i) for rc in decision.reads for i in rc.entity_ids])
            return self.finish(replace(decision, outcome=Outcome.ERROR, reply_key="R_HA_FAIL",
                                       reply_params={"names": names or "the home"}), None, [], [])
        except Exception as e:  # noqa: BLE001 - a bug in rendering: R_WHAT with the traceback (ST-3, §13.2)
            self.error = f"render bug: {type(e).__name__}: {e}"
            self.traceback = traceback.format_exc()
            return self.finish(Decision(Outcome.ERROR, reply_key="R_WHAT", route="error/render", rule="error"),
                               None, [], [])
        start = self._lap("reads", start)

        # Execute behind the guard.
        done: list[ServiceCall] = []
        if decision.calls:
            try:
                report = await execute(decision.calls, core.ha, catalog=core.catalog, safety=s.safety)
            except SafetyViolation as v:
                self.safety_violation = True
                self.error = f"safety_violation: {v}"
                self._lap("execute", start)
                if v.denylist:
                    return self.finish(Decision(Outcome.REFUSED, reply_key="R_LOCK", route="guard/denylist",
                                                rule="6.1.3"), None, [], [])
                return self.finish(Decision(Outcome.ERROR, reply_key="R_WHAT", route="guard/violation", rule="6.1.3"),
                                   None, [], [])
            done = report.done
            self.failed = report.failed
            self._lap("execute", start)
            if report.failed:
                failed_ids = report.failed[0]["call"]["entity_ids"]
                names = join_names([core.catalog.name(i) for i in failed_ids])
                return self.finish(replace(decision, outcome=Outcome.ERROR, reply_key="R_HA_FAIL",
                                           reply_params={"names": names}), None, done, [])
        return self.finish(decision, reply_text, done, list(decision.clarify.options) if decision.clarify else [])

    async def _schedule(self, decision: Decision, fallback: bool = False) -> Result:
        data = dict(decision.schedule)
        if data["op"] in ("set", "create"):
            data["area_id"] = self.area_id or ""
            if self.device_id:
                data["device_id"] = self.device_id
        call = ServiceCall("ha_jev", "manage_schedule", (), data)
        try:
            action = asyncio.create_task(self.core.ha.call_service(call.domain, call.service, [], data, return_response=True))
            try:
                result = await asyncio.shield(action)
            except asyncio.CancelledError:
                while not action.done():
                    try:
                        await asyncio.shield(action)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if action.done() and not action.cancelled():
                    action.exception()
                raise
            if not isinstance(result, dict) or not isinstance(result.get("reply"), str):
                raise ValueError("Invalid scheduling response")
        except Exception as error:
            if fallback:
                return self.finish(decision, None, [], [])
            self.error = f"schedule: {type(error).__name__}: {error}"
            return self.finish(replace(decision, outcome=Outcome.ERROR),
                               "I couldn't reach the Home Assistant timer and alarm service.", [], [])
        if fallback:
            local = result.get("next_at")
            if not local or (decision.reply_key == "Q_NEXT_ALARM" and
                             datetime.fromisoformat(local) >= decision.reply_params["at"]):
                return self.finish(decision, None, [], [])
        changed = result.get("changed") is True
        return self.finish(replace(decision, reply_key="R_SCHEDULE", outcome=Outcome.EXECUTED if changed else Outcome.ANSWERED),
                           result["reply"], [call] if changed else [], [])

    async def _ask(self, state: dict, questions: dict) -> JevResponse:
        core = self.core
        try:
            resp = await core.jev.ask(state, questions)
        except JevError as e:
            self.jev_traces.append(JevTrace(core.settings.jev.model, None, 0, round(e.latency_ms, 3), e.attempts,
                                            list(questions), {}, _jev_error_text(e), 0.0))
            raise
        cost = resp.input_tokens * core.settings.prices.jev_input_per_m / 1e6
        self.jev_traces.append(JevTrace(resp.model, resp.request_id, resp.input_tokens, round(resp.latency_ms, 3),
                                        resp.attempts, list(questions), {k: v.to_dict() for k, v in resp.answers.items()},
                                        None, cost))
        return resp

    def _render_with_reads(self, decision: Decision, states, read_results: dict[str, Any], now: datetime) -> str | None:
        key = decision.reply_key
        params = decision.reply_params
        if key.startswith("W_"):
            eid = params["entity"]
            st = states.get(eid)
            forecast = None
            fr = read_results.get("weather.get_forecasts")
            if fr is not None:
                forecast = (fr.get(eid) or {}).get("forecast")
            unit = self.speaker.temperature_unit if self.speaker else None  # --- per-person unit (profiles.py)
            return render_weather(key, params, dict(st.attributes) if st else {}, st.state if st else None, forecast, now,
                                  unit)
        if key == "R_LIST_READ":
            eid = params["entity"]
            items = ((read_results.get("todo.get_items") or {}).get(eid) or {}).get("items", [])
            return render_list(params["list"], items)
        if key == calendars.REPLY_KEY:  # hook (calendars.py)
            return calendars.render(params, read_results.get(".".join(CALENDAR_EVENTS_SERVICE)), now)
        return None

    # --- profiles and memory (profiles.py, memory.py) ------------------------------------------
    def _memory_view(self, now: datetime) -> mem.MemoryView | None:
        settings = self.core.settings.memory
        if not settings.enabled:
            return None
        store = self.core.memory
        who = self.speaker.id if self.speaker else None
        counts: dict[str, int] = {}
        for f in store.all():
            counts[f.owner] = counts.get(f.owner, 0) + 1
        return mem.MemoryView(settings, self.speaker, tuple(store.visible(who)),
                              store.last_by(who, now, settings.undo_window_s), counts)

    async def _answer_fallback(self, decision: Decision, snapshot: SessionSnapshot,
                               memory_view: mem.MemoryView | None, now: datetime, start: float) -> Result:
        core = self.core
        s = core.settings
        try:
            reply = await core.luna.answer(self.extraction.text, now=now, speaker_area=core.catalog.areas[self.area_id].name
                                           if self.area_id else None,
                                           previous_response_id=snapshot.luna_response_id,
                                           home=core.catalog.luna_summary(), profile=self._luna_profile(memory_view))
        except LunaError as e:
            if getattr(core.luna, "sticky_conversation", False):
                core.sessions.clear_hermes(self.session_id)
            self.error = f"luna: {e}"
            self._lap("luna", start)
            return self.finish(replace(decision, outcome=Outcome.ERROR, reply_key="R_LUNA_DOWN", use_luna=False),
                               None, [], [])
        p = s.prices
        uncached = max(0, reply.input_tokens - reply.cached_tokens)
        cost = (uncached * p.luna_input_per_m + reply.cached_tokens * p.luna_cached_per_m
                + reply.output_tokens * p.luna_output_per_m) / 1e6
        if reply.cost_usd is not None:
            cost = reply.cost_usd
        self.luna_trace = LunaTrace(reply.model, reply.response_id, reply.input_tokens, reply.output_tokens,
                                    round(reply.latency_ms, 3), reply.fallback_error, cost, reply.cached_tokens)
        core.sessions.set_luna_response(self.session_id, reply.response_id)
        if getattr(core.luna, "sticky_conversation", False):
            if reply.model.startswith("hermes/") and reply.response_id:
                core.sessions.keep_hermes(self.session_id, self.speaker.id if self.speaker else None,
                                          s.hermes.conversation_idle_s)
            else:
                core.sessions.clear_hermes(self.session_id)
        self._lap("luna", start)
        return self.finish(decision, reply.text, [], [])

    def _luna_profile(self, view: mem.MemoryView | None) -> LunaProfile:
        household, personal = mem.luna_blocks(view.facts if view else (), self.speaker)
        sentence = speaker_sentence(self.speaker, self.core.settings.people)
        context = ""
        if getattr(self.core.luna, "sticky_conversation", False):
            context = f"Requesting HA area_id: {self.area_id or 'unknown'}; device_id: {self.device_id or 'unknown'}. "
        return LunaProfile(memory=household, speaker=(sentence + " " if sentence else "") + personal + context)

    # ------------------------------------------------------------------------------------------
    def finish(self, decision: Decision, reply_text: str | None, done: list[ServiceCall], clarify_options: list[str]) -> Result:
        core = self.core
        text = reply_text if reply_text is not None else render_template(decision.reply_key, decision.reply_params)
        self.memory_op = dict(decision.memory) if decision.memory else None
        outcome = decision.outcome
        if outcome == Outcome.EXECUTED and not done:
            outcome = Outcome.ERROR  # invariant: EXECUTED implies calls were acknowledged
        mode = response_mode(outcome, decision.reply_key)
        # Session: pending clarification and recent targets.
        if outcome == Outcome.CLARIFY and decision.clarify is not None and decision.clarify.options:
            c = decision.clarify
            core.sessions.set_pending(self.session_id, c.kind, c.question, c.options, c.plan)
            core.sessions.set_recent(self.session_id, [t for t in decision.targets if t in core.catalog.entities])
        if outcome in (Outcome.EXECUTED, Outcome.ANSWERED):  # answered queries too, so "toggle it" works (S4)
            core.sessions.set_recent(self.session_id, [t for t in decision.targets if t in core.catalog.entities])
        self.timings["total"] = round((time.perf_counter() - self.t0) * 1000, 3)
        jev_cost = sum(j.cost_usd for j in self.jev_traces)
        luna_cost = self.luna_trace.cost_usd if self.luna_trace else 0.0
        x = self.extraction.to_dict() if self.extraction else {}
        debug = Debug(route=decision.route, rule=decision.rule, confidence=decision.confidence,
                      extraction={k: x.get(k) for k in ("numbers", "parts", "lock_keyword", "question_form", "list_item")},
                      jev=self.jev_traces, luna=self.luna_trace, reads=list(self.reads), failed_calls=list(self.failed),
                      safety_violation=self.safety_violation, timings_ms=dict(self.timings),
                      cost_usd={"jev": jev_cost, "luna": luna_cost, "total": jev_cost + luna_cost}, error=self.error)
        result = Result(request_id=self.request_id, utterance=self.text, speaker_area=self.speaker_area,
                        outcome=outcome, response_mode=mode, reply=text, service_calls=list(done),
                        clarify_options=list(clarify_options) if outcome == Outcome.CLARIFY else [], debug=debug)
        if core.log is not None:
            core.log.write(self._log_record(result))
        return result

    def _log_record(self, r: Result) -> dict:
        d = r.debug
        return {
            "ts": datetime.now(timezone.utc).isoformat(), "request_id": r.request_id, "session_id": self.session_id,
            "person": self.speaker and self.speaker.id,
            "memory": dict(self.memory_op) if self.memory_op else None,
            "utterance": r.utterance, "speaker_area": r.speaker_area, "outcome": str(r.outcome),
            "response_mode": str(r.response_mode), "reply": r.reply,
            "service_calls": [c.to_dict() for c in r.service_calls], "clarify_options": r.clarify_options,
            "route": d.route, "rule": d.rule, "confidence": d.confidence, "extraction": d.extraction,
            "jev": [{"model": j.model, "request_id": j.request_id, "input_tokens": j.input_tokens,
                     "latency_ms": j.latency_ms, "attempts": j.attempts, "answers": j.answers, "error": j.error}
                    for j in d.jev],
            "luna": ({"model": d.luna.model, "response_id": d.luna.response_id,
                      "tokens": {"input": d.luna.input_tokens, "output": d.luna.output_tokens,
                                 "cached": d.luna.cached_tokens}, "latency_ms": d.luna.latency_ms} if d.luna else None),
            "reads": [rc.to_dict() for rc in d.reads], "failed_calls": d.failed_calls,
            "safety_violation": d.safety_violation, "timings_ms": d.timings_ms, "cost_usd": d.cost_usd,
            "error": d.error, "traceback": self.traceback,
        }
