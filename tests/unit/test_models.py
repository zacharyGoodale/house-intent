import json

from ha_jev.models import (
    Debug,
    JevTrace,
    Outcome,
    ReadCall,
    ResponseMode,
    Result,
    ServiceCall,
    canonical_calls,
)


def make_result() -> Result:
    call = ServiceCall("light", "turn_off", ("light.b", "light.a"), {})
    return Result(
        request_id="abc",
        utterance="lights off",
        speaker_area="bedroom",
        outcome=Outcome.EXECUTED,
        response_mode=ResponseMode.CHIME_SUCCESS,
        reply="Turning off A and B.",
        service_calls=[call],
        clarify_options=[],
        debug=Debug(
            route="device_control/area",
            rule="9",
            confidence=0.9,
            extraction={"numbers": []},
            jev=[JevTrace("jev-1", "req_1", 100, 12.0, 1, ["intent"], {"intent": {"choice": "x"}}, None, 0.0001)],
            luna=None,
            reads=[ReadCall("state", "light", None, ("light.a",))],
            failed_calls=[],
            safety_violation=False,
            timings_ms={"total": 1.0},
            cost_usd={"jev": 0.0001, "luna": 0.0, "total": 0.0001},
        ),
    )


def test_service_call_sorts_entity_ids():
    assert ServiceCall("light", "turn_on", ("light.b", "light.a"), {}).entity_ids == ("light.a", "light.b")


def test_result_to_dict_is_json_safe_and_round_trips():
    r = make_result()
    d = r.to_dict()
    text = json.dumps(d)
    back = Result.from_dict(json.loads(text))
    assert back.to_dict() == d
    assert d["outcome"] == "executed"
    assert d["response_mode"] == "chime_success"
    assert d["service_calls"] == [
        {"domain": "light", "service": "turn_off", "entity_ids": ["light.a", "light.b"], "data": {}}
    ]


def test_canonical_calls_merge_same_domain_service_data_in_first_seen_order():
    calls = [
        ServiceCall("light", "turn_on", ("light.b",), {"brightness_pct": 20}),
        ServiceCall("light", "turn_off", ("light.c",), {}),
        ServiceCall("light", "turn_on", ("light.a",), {"brightness_pct": 20}),
        ServiceCall("light", "turn_on", ("light.a",), {"brightness_pct": 20}),
    ]
    assert canonical_calls(calls) == (
        ServiceCall("light", "turn_on", ("light.a", "light.b"), {"brightness_pct": 20}),
        ServiceCall("light", "turn_off", ("light.c",), {}),
    )
