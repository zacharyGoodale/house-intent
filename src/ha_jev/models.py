"""Output contract (spec §7) plus the small value types shared across modules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class Outcome(StrEnum):
    EXECUTED = "executed"
    CLARIFY = "clarify"
    REFUSED = "refused"
    ANSWERED = "answered"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class ResponseMode(StrEnum):
    CHIME_SUCCESS = "chime_success"
    CHIME_ERROR = "chime_error"
    SPEECH = "speech"


def _freeze(value: Any) -> Any:
    """Deep-copy JSON-ish data into plain dicts/lists (ServiceCall data must be JSON-safe)."""
    return json.loads(json.dumps(value))


@dataclass(frozen=True)
class ServiceCall:
    domain: str
    service: str
    entity_ids: tuple[str, ...]
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_ids", tuple(sorted(set(self.entity_ids))))
        object.__setattr__(self, "data", _freeze(dict(self.data)))

    def __hash__(self) -> int:
        return hash((self.domain, self.service, self.entity_ids, self.data_key()))

    def data_key(self) -> str:
        return json.dumps(self.data, sort_keys=True)

    def to_dict(self) -> dict:
        return {"domain": self.domain, "service": self.service, "entity_ids": list(self.entity_ids), "data": dict(self.data)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ServiceCall:
        return cls(d["domain"], d["service"], tuple(d.get("entity_ids") or ()), d.get("data") or {})

    def describe(self) -> str:
        return f"{self.domain}.{self.service} {','.join(self.entity_ids)} {json.dumps(dict(self.data), sort_keys=True)}"


def canonical_calls(calls: Iterable[ServiceCall]) -> tuple[ServiceCall, ...]:
    """One call per distinct (domain, service, data), entity_ids unioned and sorted, first-seen order (§6.4)."""
    order: list[tuple[str, str, str]] = []
    merged: dict[tuple[str, str, str], tuple[set[str], Mapping[str, Any]]] = {}
    for c in calls:
        key = (c.domain, c.service, c.data_key())
        if key not in merged:
            order.append(key)
            merged[key] = (set(), c.data)
        merged[key][0].update(c.entity_ids)
    return tuple(ServiceCall(k[0], k[1], tuple(merged[k][0]), merged[k][1]) for k in order)


@dataclass(frozen=True)
class ReadCall:
    kind: Literal["state", "service"]
    domain: str
    service: str | None
    entity_ids: tuple[str, ...]
    data: Mapping[str, Any] = field(default_factory=dict)  # service data, e.g. the forecast type

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "domain": self.domain, "service": self.service, "entity_ids": list(self.entity_ids)}
        if self.data:
            d["data"] = dict(self.data)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReadCall:
        return cls(d["kind"], d["domain"], d.get("service"), tuple(d.get("entity_ids") or ()), dict(d.get("data") or {}))


@dataclass(frozen=True)
class PlanItem:
    """What one clarification option would do if chosen: calls plus the reply for them."""

    calls: tuple[ServiceCall, ...]
    reply_key: str
    reply_params: Mapping[str, Any]
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClarifySpec:
    kind: Literal["pick", "confirm", "none"]
    question: str
    options: Mapping[str, str]  # id -> descriptor, in offer order
    plan: Mapping[str, PlanItem]  # id -> what to do when chosen


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    calls: tuple[ServiceCall, ...] = ()
    reads: tuple[ReadCall, ...] = ()
    reply_key: str = "R_WHAT"
    reply_params: Mapping[str, Any] = field(default_factory=dict)
    clarify: ClarifySpec | None = None
    route: str = ""
    rule: str = ""
    confidence: float | None = None
    use_luna: bool = False
    targets: tuple[str, ...] = ()  # entities acted on or offered; feeds recent_targets
    schedule: Mapping[str, Any] | None = None  # HA-owned timer/alarm operation
    memory: Mapping[str, Any] | None = None  # a memory write for core to apply (memory.py): {"op": ...}


@dataclass
class JevTrace:
    model: str
    request_id: str | None
    input_tokens: int
    latency_ms: float
    attempts: int
    question_ids: list[str]
    answers: dict[str, dict]
    error: str | None
    cost_usd: float


@dataclass
class LunaTrace:
    model: str
    response_id: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None
    cost_usd: float
    cached_tokens: int = 0


@dataclass
class Debug:
    route: str
    rule: str
    confidence: float | None
    extraction: dict
    jev: list[JevTrace]
    luna: LunaTrace | None
    reads: list[ReadCall]
    failed_calls: list[dict]
    safety_violation: bool
    timings_ms: dict[str, float]
    cost_usd: dict[str, float]
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "rule": self.rule,
            "confidence": self.confidence,
            "extraction": _freeze(self.extraction),
            "jev": [asdict(j) for j in self.jev],
            "luna": asdict(self.luna) if self.luna else None,
            "reads": [r.to_dict() for r in self.reads],
            "failed_calls": _freeze(self.failed_calls),
            "safety_violation": self.safety_violation,
            "timings_ms": dict(self.timings_ms),
            "cost_usd": dict(self.cost_usd),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Debug:
        return cls(
            route=d["route"],
            rule=d["rule"],
            confidence=d["confidence"],
            extraction=d["extraction"],
            jev=[JevTrace(**j) for j in d["jev"]],
            luna=LunaTrace(**d["luna"]) if d.get("luna") else None,
            reads=[ReadCall.from_dict(r) for r in d["reads"]],
            failed_calls=d["failed_calls"],
            safety_violation=d["safety_violation"],
            timings_ms=d["timings_ms"],
            cost_usd=d["cost_usd"],
            error=d.get("error"),
        )


@dataclass
class Result:
    request_id: str
    utterance: str
    speaker_area: str | None
    outcome: Outcome
    response_mode: ResponseMode
    reply: str
    service_calls: list[ServiceCall]
    clarify_options: list[str]
    debug: Debug

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "utterance": self.utterance,
            "speaker_area": self.speaker_area,
            "outcome": str(self.outcome),
            "response_mode": str(self.response_mode),
            "reply": self.reply,
            "service_calls": [c.to_dict() for c in self.service_calls],
            "clarify_options": list(self.clarify_options),
            "debug": self.debug.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Result:
        return cls(
            request_id=d["request_id"],
            utterance=d["utterance"],
            speaker_area=d["speaker_area"],
            outcome=Outcome(d["outcome"]),
            response_mode=ResponseMode(d["response_mode"]),
            reply=d["reply"],
            service_calls=[ServiceCall.from_dict(c) for c in d["service_calls"]],
            clarify_options=list(d["clarify_options"]),
            debug=Debug.from_dict(d["debug"]),
        )
