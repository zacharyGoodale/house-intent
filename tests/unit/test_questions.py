import json
import re
from pathlib import Path

import pytest
from typesafe_sdk import Choice, Noul

from ha_jev.catalog import Catalog
from ha_jev.config import load_settings
from ha_jev.extract import extract
from ha_jev.ha.stub import FIXTURE_PATH, StubHA
from ha_jev.jev import ChoiceAnswer, NoulAnswer
from ha_jev.questions import AnswerView, MissingAnswer, RequestContext, build_request

SNAPSHOT = Path(__file__).parent / "snapshots" / "questions.json"
GENERATED_PREFIXES = ("area__", "mentioned__", "number_is_setting__", "p0__", "p1__", "p2__")
GENERATED_CHOICES = {"target", "scene", "source", "todo_list", "clarify_pick"}


async def ctx_for(text, speaker_area=None, recent=(), pending=None, overlays=()):
    ha = StubHA.default(overlays=overlays)
    s = load_settings().with_scope_extension(**ha.overlay_scope)
    cat = Catalog.build(await ha.get_registry(), await ha.get_states(), s.scope, s.safety)
    return RequestContext(extraction=extract(text), catalog=cat, speaker_area=speaker_area, now=ha.now,
                          recent_targets=list(recent), pending=pending, policy=s.policy), cat


def shape(questions):
    return {qid: sorted(q["criteria"]) if q["type"] == "choice" else "noul" for qid, q in questions.items()}


async def test_snapshot_of_ids_and_option_keys():
    ctx, _ = await ctx_for("dim the bedroom lights to 20%", speaker_area="living_room")
    _, questions = build_request(ctx).state, build_request(ctx).questions
    actual = shape(questions)
    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(actual, indent=1, ensure_ascii=False, sort_keys=True))
        pytest.fail("snapshot written; inspect and re-run")
    assert actual == json.loads(SNAPSHOT.read_text())


async def test_state_shape_omits_empty_fields():
    ctx, _ = await ctx_for("dim the bedroom lights to 20%", speaker_area="living_room")
    state = build_request(ctx).state
    home = state.pop("home")
    assert state == {
        "utterance": "dim the bedroom lights to 20%",
        "speaker_area": "Living Room",
        "now": "Tuesday 2037-09-22 18:00",
        "numbers": [{"id": "n0", "text": "20%", "value": 20, "unit": "percent"}],
    }
    # the home directory: names from the registry and config, so intent can recognise a bare scene name
    assert any(r.startswith("Living Room") for r in home["rooms"]) and "Alder" in home["devices"] and "Front Door" not in home["devices"]
    assert "Hue scene Common Area Relax" in home["scenes"] and "movie mode" in home["scenes"]


async def test_every_question_passes_sdk_validation():
    ctx, _ = await ctx_for("turn off the lights and pause the music", recent=["light.alder"])
    questions = build_request(ctx).questions
    for q in questions.values():
        body = {k: v for k, v in q.items() if k != "type"}
        obj = Choice(**body) if q["type"] == "choice" else Noul(**body)
        assert obj is not None
        if q["type"] == "choice":
            assert 2 <= len(q["criteria"]) <= 255


async def test_no_fixed_question_text_contains_a_registry_name():
    fixture = json.loads(FIXTURE_PATH.read_text())
    names = {a["name"] for a in fixture["areas"]} | {e["friendly_name"] for e in fixture["entities"] if e["friendly_name"]}
    names = {n.strip() for n in names if len(n.strip()) >= 3}
    ctx, _ = await ctx_for("turn off the lights and pause the music", recent=["light.alder"])
    questions = build_request(ctx).questions
    for qid, q in questions.items():
        if qid.startswith(GENERATED_PREFIXES):
            continue
        texts = [json.dumps(q["instructions"])]
        if q["type"] == "choice" and qid not in GENERATED_CHOICES:
            texts.append(json.dumps(q["criteria"]))
        blob = " ".join(texts)
        for n in names:
            assert not re.search(rf"(?<!\w){re.escape(n)}(?!\w)", blob, re.IGNORECASE), (qid, n)


async def test_conditional_questions():
    ctx, _ = await ctx_for("turn it up")
    q = build_request(ctx).questions
    assert "recent" not in q["scope"]["criteria"]
    assert "clarify_reply" not in q["intent"]["criteria"]
    assert "clarify_pick" not in q and "clarify_affirm" not in q
    assert "source" in q and "todo_list" in q
    ctx, _ = await ctx_for("turn it up", recent=["light.alder"],
                           pending={"kind": "pick", "question": "Which one: A or B?",
                                    "options": {"scene.demo_common_relax": "Common Area Relax — scene, Living Room",
                                                "scene.demo_bedroom_relax": "Bedroom Relax — scene, Bedroom"}})
    req = build_request(ctx)
    assert "recent" in req.questions["scope"]["criteria"]
    assert "clarify_reply" in req.questions["intent"]["criteria"]
    assert set(req.questions["clarify_pick"]["criteria"]) == {"scene.demo_common_relax", "scene.demo_bedroom_relax", "all", "none"}
    assert req.state["recent_targets"] == [{"entity_id": "light.alder", "name": "Alder"}]
    assert req.state["pending_clarification"]["question"] == "Which one: A or B?"


async def test_compound_parts_get_prefixed_questions():
    ctx, _ = await ctx_for("dim the lights to 30% and pause the music")
    req = build_request(ctx)
    assert req.state["parts"] == ["dim the lights to 30%", "pause the music"]
    assert "p0__intent" in req.questions and "p1__media_action" in req.questions
    assert "p0__number_is_setting__n0" in req.questions
    assert "p1__number_is_setting__n0" not in req.questions
    assert "`parts[1]`" in req.questions["p1__intent"]["instructions"]


async def test_new_area_gets_an_area_noul():
    office = {"areas": [{"id": "office", "name": "Office", "aliases": ["study"]}],
              "entities": [{"entity_id": "light.desk_lamp", "friendly_name": "Desk Lamp", "area_id": "office"}]}
    ctx, _ = await ctx_for("turn on the office lights", overlays=[office])
    q = build_request(ctx).questions
    assert "area__office" in q
    assert "Office" in q["area__office"]["instructions"] and "study" in q["area__office"]["instructions"]
    assert "light.desk_lamp" in q["target"]["criteria"]
    assert "mentioned__light.desk_lamp" in q


def test_answer_view_reads_and_part_views():
    answers = {
        "intent": ChoiceAnswer("device_control", 0.9, {"device_control": 0.9, "general": 0.1}),
        "security": NoulAnswer(0.1),
        "p1__intent": ChoiceAnswer("activate_scene", 0.8, {"activate_scene": 0.8}),
    }
    av = AnswerView(answers)
    assert av.choice("intent").choice == "device_control"
    assert av.noul("security") == 0.1
    part = av.for_part(1)
    assert part.choice("intent").choice == "activate_scene"
    assert part.noul("security") == 0.1  # global questions fall through
    with pytest.raises(MissingAnswer):
        av.choice("scope")


async def test_pruning_kicks_in_only_for_large_choices():
    many = {"entities": [{"entity_id": f"scene.extra_{i}", "friendly_name": f"Extra {i}", "area_id": "kitchen"}
                         for i in range(65)]}
    ctx, _ = await ctx_for("activate extra 7", overlays=[many])
    req = build_request(ctx)
    assert len(req.questions["scene"]["criteria"]) == 41  # 40 kept + none
    assert "scene.extra_7" in req.questions["scene"]["criteria"]
    assert req.overflow["scene"]
    ctx, _ = await ctx_for("activate relax")
    assert build_request(ctx).overflow == {}
