"""Scorer: every §11.4 rule, plus the §11.3 reply constraints."""

from ha_jev.evaluation.golden import parse_case
from ha_jev.evaluation.scoring import calls_match, data_subset, normalize_reply, reply_ok, score_case


def call(domain, service, ids, data=None):
    return {"domain": domain, "service": service, "entity_ids": ids, "data": data or {}}


def test_data_subset_rules():
    assert data_subset({"a": 1}, {"a": 1.0000001, "b": 2})
    assert not data_subset({"a": 1}, {"a": 1.1})
    assert not data_subset({"a": 1}, {})
    assert data_subset({"v": {"$range": [0.3, 0.4]}}, {"v": 0.35})
    assert not data_subset({"v": {"$range": [0.3, 0.4]}}, {"v": 0.41})
    assert data_subset({"v": {"$any": [1, 2]}}, {"v": 2})
    assert not data_subset({"v": {"$any": [1, 2]}}, {"v": 3})
    assert data_subset({}, {"transition": 0.5})


def test_calls_merge_before_matching_and_must_match_as_multisets():
    expected = [call("light", "turn_off", ["light.a", "light.b"])]
    actual = [call("light", "turn_off", ["light.b"]), call("light", "turn_off", ["light.a"])]
    assert calls_match(expected, actual)[0]
    assert not calls_match(expected, [call("light", "turn_off", ["light.a"])])[0]
    assert not calls_match(expected, actual + [call("light", "turn_on", ["light.c"])])[0]  # unmatched actual
    assert not calls_match([], [call("light", "turn_on", ["light.c"])])[0]
    assert calls_match([], [])[0]


def test_calls_entity_sets_must_be_equal_and_data_subset():
    exp = [call("light", "turn_on", ["light.a"], {"brightness_pct": 20})]
    assert calls_match(exp, [call("light", "turn_on", ["light.a"], {"brightness_pct": 20, "transition": 1})])[0]
    assert not calls_match(exp, [call("light", "turn_on", ["light.a", "light.b"], {"brightness_pct": 20})])[0]


def test_reply_constraints():
    assert normalize_reply("Sunset is at 7:02 PM.") == "sunset is at 702 pm"
    ok, reasons, unjudged = reply_ok({"must_mention": ["locked"]}, "The Front Door is locked.", "speech")
    assert ok and not unjudged
    ok, _, _ = reply_ok({"must_mention": [["rain", "rainy"]]}, "Tomorrow: rain.", "speech")
    assert ok
    ok, reasons, _ = reply_ok({"must_not_mention": ["unlocked"]}, "The door is unlocked.", "speech")
    assert not ok and reasons
    ok, _, _ = reply_ok({}, " ".join(["word"] * 46), "speech")
    assert not ok  # default max_words 45 for speech
    ok, _, unjudged = reply_ok({"rubric": "is polite"}, "Hi.", "speech", judge=None)
    assert ok and unjudged
    ok, _, unjudged = reply_ok({"rubric": "is polite"}, "Hi.", "speech", judge=lambda rubric, text: (False, "rude"))
    assert not ok and not unjudged


def _case(**over):
    base = {"id": "x-001", "utterance": "u", "tags": ["light_named"], "expected_outcome": "executed",
            "expected_response_mode": "chime_success", "expected_calls": [call("light", "turn_on", ["light.a"])],
            "requires_llm": False}
    base.update(over)
    return parse_case(base, "test")


def _actual(outcome="executed", mode="chime_success", calls=None, clarify=None, reply="Turning on A."):
    return {"outcome": outcome, "response_mode": mode, "service_calls": calls if calls is not None else
            [call("light", "turn_on", ["light.a"])], "clarify_options": clarify or [], "reply": reply}


def test_score_case_pass_and_each_failure_reason():
    assert score_case(_case(), _actual()).passed
    assert "outcome" in score_case(_case(), _actual(outcome="clarify")).failures[0]
    assert any("response_mode" in f for f in score_case(_case(), _actual(mode="speech")).failures)
    assert any("calls" in f for f in score_case(_case(), _actual(calls=[])).failures)
    c = _case(expected_outcome="clarify", expected_response_mode="speech", expected_calls=[],
              expected_clarify_options=["scene.a", "scene.b"])
    assert score_case(c, _actual("clarify", "speech", [], ["scene.a", "scene.b", "scene.c"])).passed
    assert not score_case(c, _actual("clarify", "speech", [], ["scene.a"])).passed


def test_acceptable_alternatives_inherit_missing_keys():
    c = _case(acceptable=[{"expected_calls": [call("light", "turn_on", ["light.b"])]}])
    assert score_case(c, _actual(calls=[call("light", "turn_on", ["light.b"])])).passed
    assert not score_case(c, _actual(calls=[call("light", "turn_on", ["light.c"])])).passed


def test_must_mention_ignores_diacritics():
    assert normalize_reply("Arigatō!") == "arigato"
    ok, _, _ = reply_ok({"must_mention": [["arigato", "arigatou"]]}, "It's “arigatō” (ありがとう).", "speech")
    assert ok
