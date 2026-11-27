"""Call builders, chosen per capability row (`DomainCaps.builder`), never by domain name.

- `template`: builds calls from the row's `ActionDef`s alone (service, feature, value, data template, per-entity
  toggle). Switches, automations and covers use it, and so does any domain added from config.
- `light`: brightness on dimmable members with plain turn_on for on/off-only lamps in a set, colour names and
  white points, per-entity kelvin clamps, warmer/cooler from the current colour temperature (spec §6.4).
- `media`: transport, proportional master volume, mute, source selection and TV power (spec §6.4).

A builder raises `ActionStop` for a terminal answer (unsupported, clarify); policy turns it into a Decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from dataclasses import dataclass, field
from typing import Any

from ha_jev.capabilities import ActionDef, CapabilityTable
from ha_jev.models import ServiceCall

NAMED_COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink"}
WHITE_KELVIN = {"white": 4000, "warm_white": 2700, "cool_white": 5000}


class ActionStop(Exception):
    """A terminal answer from a builder: kind is "unsupported" or "clarify"."""

    def __init__(self, kind: str, key: str, **params: Any) -> None:
        super().__init__(key)
        self.kind = kind
        self.key = key
        self.params = params


@dataclass
class Value:
    value: float
    unit: str | None
    conf: float


@dataclass
class Target:
    entity_id: str
    domain: str
    name: str
    features: frozenset[str]
    onoff_only: bool
    capabilities: Mapping[str, Any]


@dataclass
class BuildInput:
    table: CapabilityTable
    domain: str
    action_id: str
    action: ActionDef
    targets: list[Target]
    scope_kind: str  # one | named | recent | area | house | unspecified
    label: str | None
    names: Callable[[Sequence[str]], str]
    value: Callable[[], Value | None]  # the setting number, else a level word
    number: Callable[[], Value | None]  # the setting number only
    answer: Callable[[str], Any]  # an argument Choice answer (color, source) or None
    state: Callable[[str], Any]  # EntityState or None
    step_pct: float = 20
    volume_step_pct: float = 10
    kelvin_step: float = 500
    extra_targets: Callable[[str], list[Target]] = lambda what: []  # "tvs_in_scope", "source_players"
    sources: Sequence[str] = ()


    @property
    def set_scope(self) -> bool:
        """An area or house set: plug lamps join a brightness set, TVs join media power."""
        return self.scope_kind in ("area", "house")


@dataclass
class Built:
    calls: list[ServiceCall]
    key: str
    params: dict[str, Any]
    confs: list[float] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)


def build(inp: BuildInput, builder: str) -> Built:
    return BUILDERS[builder](inp)


def _num(v: float) -> int | float:
    return int(v) if float(v).is_integer() else round(float(v), 2)


def _percent_value(inp: BuildInput) -> Value:
    v = inp.value()
    if v is None:
        raise ActionStop("clarify", "R_WHAT_LEVEL")
    if v.unit not in (None, "percent", "level") or not 0 <= v.value <= 100:
        raise ActionStop("unsupported", "R_RANGE")
    return v


def _fill(template: Any, value: float | None) -> Any:
    if isinstance(template, Mapping):
        return {k: _fill(v, value) for k, v in template.items()}
    if template == "$value":
        return _num(round(value or 0))
    if template == "$fraction":
        return round((value or 0) / 100, 2)
    return template


def _capable(inp: BuildInput) -> list[Target]:
    ents = [t for t in inp.targets if inp.table.supports(t.domain, inp.action_id, t.features, t.onoff_only)]
    if not ents:
        raise ActionStop("unsupported", "R_CANT", name=inp.targets[0].name)
    return ents


# ---- template ---------------------------------------------------------------------------------------
def template(inp: BuildInput) -> Built:
    a = inp.action
    if a.service is None:
        raise ActionStop("clarify", "R_WHAT")
    ents = _capable(inp)
    acted = [t.entity_id for t in ents]
    names = inp.label or inp.names(acted)
    confs: list[float] = []
    value: float | None = None
    if a.value:
        v = _percent_value(inp)
        confs.append(v.conf)
        value = v.value
    data = _fill(dict(a.data), value)
    if "hvac_mode" in data and any(data["hvac_mode"] not in t.capabilities.get("hvac_modes", ()) for t in ents):
        raise ActionStop("unsupported", "R_CANT", name=names)
    if "option" in data and any(data["option"] not in t.capabilities.get("options", ()) for t in ents):
        raise ActionStop("unsupported", "R_CANT", name=names)
    calls: list[ServiceCall] = []
    if a.toggle:
        for t in ents:
            st = inp.state(t.entity_id)
            svc = "turn_off" if (st is not None and st.state == "on") else "turn_on"
            calls.append(ServiceCall(t.domain, svc, (t.entity_id,), data))
    else:
        calls.append(ServiceCall(ents[0].domain, a.service, tuple(acted), data))
    params: dict[str, Any] = {"names": names, "name": names}
    if value is not None:
        params["v"] = _num(round(value))
    return Built(calls, a.reply, params, confs, acted)


# ---- light ------------------------------------------------------------------------------------------
def _clamp_k(t: Target, k: float) -> int:
    lo = t.capabilities.get("min_color_temp_kelvin") or 1000
    hi = t.capabilities.get("max_color_temp_kelvin") or 20000
    return int(round(max(lo, min(hi, k))))


def light(inp: BuildInput) -> Built:
    ents = inp.targets
    dimmable = [e for e in ents if not e.onoff_only]
    onoff = [e for e in ents if e.onoff_only]
    names = inp.label or inp.names([e.entity_id for e in ents])
    A = inp.action_id
    reply = inp.action.reply
    domain = ents[0].domain

    def call(service: str, es: Sequence[Target], data: Mapping[str, Any] | None = None) -> ServiceCall:
        return ServiceCall(domain, service, tuple(e.entity_id for e in es), dict(data or {}))

    def need_dimmable() -> list[Target]:
        if not dimmable:
            raise ActionStop("unsupported", "R_CANT", name=ents[0].name)
        return dimmable

    def label_for(es: Sequence[Target]) -> str:
        return inp.label or inp.names([e.entity_id for e in es])

    ids = [e.entity_id for e in ents]
    if A in ("turn_on", "turn_off"):
        return Built([call(A, ents)], reply, {"names": names}, [], ids)
    if A == "toggle":
        if len(ents) == 1:
            return Built([call("toggle", ents)], reply, {"names": names}, [], ids)
        any_on = any((st := inp.state(e.entity_id)) is not None and st.state == "on" for e in ents)
        return Built([call("turn_off" if any_on else "turn_on", ents)], "R_OFF" if any_on else "R_ON",
                     {"names": names}, [], ids)
    if A == "set_brightness":
        v = _percent_value(inp)
        if v.value == 0:
            return Built([call("turn_off", ents)], "R_OFF", {"names": names}, [v.conf], ids)
        pct = _num(v.value)
        calls = []
        if dimmable:
            calls.append(call("turn_on", dimmable, {"brightness_pct": pct}))
        if onoff and inp.set_scope:
            calls.append(call("turn_on", onoff))
        if not calls:
            raise ActionStop("unsupported", "R_CANT", name=ents[0].name)
        acted = dimmable + (onoff if inp.set_scope else [])
        return Built(calls, reply, {"names": label_for(acted), "v": pct}, [v.conf], [e.entity_id for e in acted])
    if A in ("brighten", "dim"):
        es = need_dimmable()
        v = inp.number()
        step, confs = inp.step_pct, []
        if v is not None and v.unit == "percent":
            step = v.value
            confs.append(v.conf)
        s = _num(step if A == "brighten" else -step)
        return Built([call("turn_on", es, {"brightness_step_pct": s})], reply, {"names": label_for(es)}, confs,
                     [e.entity_id for e in es])
    if A == "set_color":
        color = inp.answer("color")
        if color is None or color.choice == "none":
            raise ActionStop("clarify", "R_WHAT")
        if color.choice not in NAMED_COLORS and color.choice not in WHITE_KELVIN:
            raise ActionStop("unsupported", "R_COLOR_UNKNOWN")
        es = need_dimmable()
        data = {"color_name": color.choice} if color.choice in NAMED_COLORS else {"color_temp_kelvin": WHITE_KELVIN[color.choice]}
        return Built([call("turn_on", es, data)], reply, {"names": label_for(es), "color": color.choice.replace("_", " ")},
                     [color.probabilities.get(color.choice, color.confidence)], [e.entity_id for e in es])
    if A == "set_color_temp":
        v = inp.number()
        if v is None:  # no number: a named white ("warm white") is a colour temperature too (round 2, class 3)
            color = inp.answer("color")
            if color is not None and color.choice in WHITE_KELVIN:
                es = need_dimmable()
                k = WHITE_KELVIN[color.choice]
                return Built([call("turn_on", es, {"color_temp_kelvin": k})], "R_COLOR",
                             {"names": label_for(es), "color": color.choice.replace("_", " ")},
                             [color.probabilities.get(color.choice, color.confidence)], [e.entity_id for e in es])
            raise ActionStop("clarify", "R_WHAT_LEVEL")
        if not (v.unit == "kelvin" or (v.unit is None and v.value >= 1000)) or not 1000 <= v.value <= 20000:
            raise ActionStop("unsupported", "R_RANGE")
        es = need_dimmable()
        calls = [call("turn_on", [e], {"color_temp_kelvin": _clamp_k(e, v.value)}) for e in es]
        return Built(calls, reply, {"names": label_for(es), "k": _num(v.value)}, [v.conf], [e.entity_id for e in es])
    if A in ("warmer", "cooler"):
        es = need_dimmable()
        delta = -inp.kelvin_step if A == "warmer" else inp.kelvin_step
        calls = []
        for e in es:
            st = inp.state(e.entity_id)
            cur = (st.attributes.get("color_temp_kelvin") if st is not None else None) or 4000
            calls.append(call("turn_on", [e], {"color_temp_kelvin": _clamp_k(e, float(cur) + delta)}))
        return Built(calls, reply, {"names": label_for(es)}, [], [e.entity_id for e in es])
    raise ActionStop("clarify", "R_WHAT")


# ---- media ------------------------------------------------------------------------------------------
def media(inp: BuildInput) -> Built:
    A = inp.action_id
    reply = inp.action.reply
    ts = list(inp.targets)
    domain = ts[0].domain
    if A in ("turn_on", "turn_off") and inp.set_scope:
        ts += [t for t in inp.extra_targets("tvs_in_scope") if t.entity_id not in {x.entity_id for x in ts}]
    if A == "select_source":
        src = inp.answer("source")
        if src is None or src.choice == "none" or src.choice not in inp.sources:
            raise ActionStop("clarify", "R_WHAT")
        capable = [t for t in ts if "select_source" in t.features]
        if not capable:
            only = inp.extra_targets("source_players")
            if len(only) == 1 and inp.scope_kind != "one":
                capable = only
            else:
                raise ActionStop("unsupported", "R_CANT", name=ts[0].name)
        ids = [t.entity_id for t in capable]
        return Built([ServiceCall(domain, "select_source", tuple(ids), {"source": src.choice})], reply,
                     {"source": src.choice, "name": inp.names(ids)},
                     [src.probabilities.get(src.choice, src.confidence)], ids)
    inp_ts = BuildInput(**{**inp.__dict__, "targets": ts})
    ents = _capable(inp_ts)
    acted = [e.entity_id for e in ents]
    names = inp.names(acted)
    if A in ("media_play", "media_pause", "media_next_track", "media_previous_track"):
        return Built([ServiceCall(domain, A, tuple(acted), {})], reply, {"name": names}, [], acted)
    if A in ("turn_on", "turn_off"):
        return Built([ServiceCall(domain, A, tuple(acted), {})], reply,
                     {"name": names if len(acted) > 1 else ents[0].name}, [], acted)
    if A in ("volume_mute", "volume_unmute"):
        return Built([ServiceCall(domain, "volume_mute", tuple(acted), {"is_volume_muted": A == "volume_mute"})],
                     reply, {"names": names}, [], acted)
    if A in ("volume_set", "volume_up", "volume_down"):
        cur = {}
        for t in acted:
            st = inp.state(t)
            level = st.attributes.get("volume_level") if st is not None else None
            cur[t] = float(level) if isinstance(level, (int, float)) else 0.30
        cur_max = max(cur.values())
        confs: list[float] = []
        if A == "volume_set":
            v = _percent_value(inp)
            confs.append(v.conf)
            new_max = v.value / 100
        else:
            v = inp.number()
            step = inp.volume_step_pct
            if v is not None and v.unit == "percent":
                step = v.value
                confs.append(v.conf)
            delta = step / 100 if A == "volume_up" else -step / 100
            new_max = max(0.0, min(1.0, cur_max + delta))
        calls = []
        for t in acted:
            level = round(new_max, 2) if (len(acted) == 1 or cur_max == 0) else round(cur[t] * new_max / cur_max, 2)
            calls.append(ServiceCall(domain, "volume_set", (t,), {"volume_level": level}))
        return Built(calls, reply, {"names": names, "v": int(round(new_max * 100))}, confs, acted)
    raise ActionStop("clarify", "R_WHAT")


def number(inp: BuildInput) -> Built:
    """Set or adjust a bounded numeric control using its advertised range and step."""
    return _temperature(inp, False)


def climate(inp: BuildInput) -> Built:
    if inp.action_id not in ("set_temperature", "warmer", "cooler"):
        return template(inp)
    return _temperature(inp, True)


def _temperature(inp: BuildInput, thermostat: bool) -> Built:
    """Bound and convert temperature values before building numeric or climate calls."""
    v = inp.number()
    relative = inp.action_id in ("warmer", "cooler")
    if v is None and not relative:
        raise ActionStop("clarify", "R_WHAT_LEVEL")
    if v is not None and (v.unit not in (None, "degrees", "C", "F") or not math.isfinite(v.value)):
        raise ActionStop("unsupported", "R_RANGE")
    calls = []
    ents = _capable(inp)
    for t in ents:
        try:
            lo = float(t.capabilities["min_temp" if thermostat else "min"])
            hi = float(t.capabilities["max_temp" if thermostat else "max"])
            step = float(t.capabilities.get("target_temp_step" if thermostat else "step", 1))
            if not all(math.isfinite(x) for x in (lo, hi, step)) or hi < lo or step <= 0:
                raise ValueError
            requested = v.value if v is not None else step
            if v is not None and v.unit in ("C", "F"):
                unit = str(t.capabilities.get("temperature_unit" if thermostat else "unit_of_measurement", "")).replace("°", "")
                if unit not in ("C", "F"):
                    raise ValueError
                if unit != v.unit:
                    offset = 0 if relative else 32
                    requested = requested * 9 / 5 + offset if unit == "F" else (requested - offset) * 5 / 9
            if relative:
                st = inp.state(t.entity_id)
                current = float(st.attributes.get("temperature") if thermostat else st.state) if st is not None else float("nan")
                if not math.isfinite(current):
                    raise ValueError
                delta = requested
                if delta <= 0:
                    raise ValueError
                value = max(lo, min(hi, current + (delta if inp.action_id == "warmer" else -delta)))
            else:
                value = requested
                if not lo <= value <= hi:
                    raise ValueError
            value = max(lo, min(hi, lo + round((value - lo) / step) * step))
        except (KeyError, TypeError, ValueError):
            raise ActionStop("unsupported", "R_RANGE") from None
        calls.append(ServiceCall(t.domain, inp.action.service, (t.entity_id,), {"temperature" if thermostat else "value": _num(value)}))
    ids = [t.entity_id for t in ents]
    return Built(calls, inp.action.reply, {"names": inp.label or inp.names(ids)},
                 [v.conf] if v is not None else [], ids)


BUILDERS: dict[str, Callable[[BuildInput], Built]] = {
    "template": template, "light": light, "media": media, "number": number, "climate": climate,
}
