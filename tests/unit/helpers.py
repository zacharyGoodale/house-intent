"""Shared helpers for unit tests: build a DecisionContext from StubHA and run decide() with FakeJev answers."""

from __future__ import annotations

from datetime import datetime

from ha_jev.catalog import Catalog
from ha_jev.config import load_settings
from ha_jev.core import sanitize_answers
from ha_jev.extract import extract
from ha_jev.ha.stub import StubHA
from ha_jev.policy import DecisionContext, SessionSnapshot, decide
from ha_jev.questions import AnswerView, RequestContext, build_request
from tests.fakes import FakeJev


async def decide_for(text: str, answers=None, *, speaker_area: str | None = None, session: SessionSnapshot | None = None,
                     initial_state=None, overrides=None, overlays=()):
    ha = StubHA.default(overlays=overlays, initial_state=initial_state)
    settings = load_settings(overrides=overrides).with_scope_extension(**ha.overlay_scope)
    states = await ha.get_states()
    cat = Catalog.build(await ha.get_registry(), states, settings.scope, settings.safety)
    x = extract(text)
    session = session or SessionSnapshot()
    pending = None
    if session.pending is not None:
        pending = {"kind": session.pending.kind, "question": session.pending.question, "options": dict(session.pending.options)}
    req = build_request(RequestContext(x, cat, speaker_area, ha.now, list(session.recent_targets), pending, settings.policy))
    jev = FakeJev(answers or {})
    resp = await jev.ask(req.state, req.questions)
    answers = sanitize_answers(resp.answers, req.questions)  # the same pipeline Core runs
    ctx = DecisionContext(utterance=text, extraction=x, catalog=cat, states=states, speaker_area=speaker_area,
                          session=session, policy=settings.policy, now=ha.now)
    return decide(ctx, AnswerView(answers)), req


def calls_of(decision):
    return [(c.domain, c.service, list(c.entity_ids), dict(c.data)) for c in decision.calls]


NOW = datetime.fromisoformat("2037-09-22T18:00:00-04:00")
