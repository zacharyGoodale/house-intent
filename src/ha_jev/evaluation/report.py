"""Eval report (spec §11.5): markdown text, same summary printed to stdout."""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ha_jev.config import PriceSettings
from ha_jev.evaluation.runner import RunResult
from ha_jev.models import Outcome, ResponseMode

REPO = Path(__file__).resolve().parents[3]


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config"], cwd=REPO, capture_output=True,
                               text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _lat_row(name: str, values: list[float]) -> str:
    if not values:
        return f"| {name} | – | – | – | 0 |"
    return f"| {name} | {_quantile(values, 0.5):.0f} | {_quantile(values, 0.95):.0f} | {max(values):.0f} | {len(values)} |"


def build_report(run: RunResult, prices: PriceSettings) -> str:
    recs = run.records
    ran = [r for r in recs if not r.skipped]
    skipped = [r for r in recs if r.skipped]
    passed = [r for r in ran if r.passed]
    models = Counter(j["model"] for r in ran if r.actual for j in r.actual["debug"]["jev"] if j.get("model"))
    for r in ran:
        for h in r.history_results:
            for j in h["debug"]["jev"]:
                models[j["model"]] += 1
    safety_ids = run.safety_failed_ids
    lines: list[str] = []
    lines.append("# ha-jev eval report")
    lines.append("")
    lines.append(f"- Jev model: {', '.join(f'{m} ({n})' for m, n in models.most_common()) or 'none'}")
    lines.append(f"- git: {_git_sha()}")
    lines.append(f"- flags: {' '.join(f'{k}={v}' for k, v in run.flags.items())}")
    lines.append(f"- cases: total {run.total_cases}, selected {len(recs)}, run {len(ran)}, skipped {len(skipped)}; "
                 f"wall {run.wall_s:.1f}s")
    lines.append("")
    lines.append(f"**SAFETY: {'FAIL' if safety_ids else 'PASS'}**" + (f" — {', '.join(safety_ids)}" if safety_ids else ""))
    lines.append("")
    unjudged = sum(1 for r in ran if r.unjudged)
    lines.append("## Overall")
    lines.append("")
    lines.append(f"Accuracy: **{len(passed)}/{len(ran)} = {_pct(len(passed), len(ran))}**; unjudged rubrics: {unjudged}")
    lines.append("")

    # per tag
    by_tag: dict[str, list] = defaultdict(list)
    for r in ran:
        for t in dict.fromkeys(r.tags):
            by_tag[t].append(r)
    rows = sorted(((t, len(v), sum(x.passed for x in v)) for t, v in by_tag.items()),
                  key=lambda row: (row[2] / row[1] if row[1] else 0, row[0]))
    lines.append("## Per tag")
    lines.append("")
    if "unseen_entity" in by_tag:
        v = by_tag["unseen_entity"]
        lines.append(f"**Generalization (unseen_entity): {sum(x.passed for x in v)}/{len(v)} = "
                     f"{_pct(sum(x.passed for x in v), len(v))}**")
        lines.append("")
    lines.append("| tag | n | passed | accuracy |")
    lines.append("|---|---|---|---|")
    for t, n, p in rows:
        lines.append(f"| {t} | {n} | {p} | {_pct(p, n)} |")
    lines.append("")

    # confusion matrices
    def matrix(title: str, values: list[str], exp_key: str, act_key: str) -> None:
        lines.append(f"## {title} (rows expected, columns actual)")
        lines.append("")
        counts: Counter = Counter()
        for r in ran:
            if r.actual is None:
                counts[(r.expected[exp_key], "runner_error")] += 1
                continue
            counts[(r.expected[exp_key], r.actual[act_key])] += 1
        cols = [v for v in values if any(c[1] == v for c in counts)] + (["runner_error"] if any(c[1] == "runner_error" for c in counts) else [])
        rows_ = [v for v in values if any(c[0] == v for c in counts)]
        lines.append("| expected \\ actual | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for e in rows_:
            lines.append(f"| {e} | " + " | ".join(str(counts.get((e, a), 0) or "") for a in cols) + " |")
        lines.append("")

    matrix("Outcome confusion", [o.value for o in Outcome], "expected_outcome", "outcome")
    matrix("Response mode confusion", [m.value for m in ResponseMode], "expected_response_mode", "response_mode")

    exec_ok = [r for r in ran if r.actual and r.expected["expected_outcome"] == "executed" and r.actual["outcome"] == "executed"]
    calls_ok = sum(1 for r in exec_ok if r.calls_ok)
    lines.append("## Calls accuracy")
    lines.append("")
    lines.append(f"On expected=executed cases whose outcome matched: {calls_ok}/{len(exec_ok)} = {_pct(calls_ok, len(exec_ok))}")
    lines.append("")

    # cost
    jev_reqs = jev_tokens = 0
    jev_cost = 0.0
    luna_calls = luna_in = luna_out = 0
    luna_cost = 0.0
    utterances = 0
    for r in ran:
        results = list(r.history_results) + ([r.actual] if r.actual else [])
        for a in results:
            utterances += 1
            for j in a["debug"]["jev"]:
                jev_reqs += 1
                jev_tokens += j["input_tokens"]
                jev_cost += j["cost_usd"]
            lu = a["debug"]["luna"]
            if lu and lu.get("response_id"):
                luna_calls += 1
                luna_in += lu["input_tokens"]
                luna_out += lu["output_tokens"]
                luna_cost += lu["cost_usd"]
    judge_calls = sum(len(r.judge_usage) for r in ran)
    judge_in = sum(u.get("input_tokens", 0) for r in ran for u in r.judge_usage)
    judge_out = sum(u.get("output_tokens", 0) for r in ran for u in r.judge_usage)
    judge_cost = (judge_in * prices.luna_input_per_m + judge_out * prices.luna_output_per_m) / 1e6
    lines.append("## Cost")
    lines.append("")
    lines.append(f"- Jev: {jev_reqs} requests, {jev_tokens:,} input tokens, ${jev_cost:.5f}"
                 f" (avg {jev_tokens / jev_reqs if jev_reqs else 0:,.0f} tokens/request)")
    lines.append(f"- Luna: {luna_calls} calls, {luna_in:,} in / {luna_out:,} out tokens, ${luna_cost:.5f}")
    lines.append(f"- Judge (separate): {judge_calls} calls, ${judge_cost:.5f}")
    lines.append(f"- Product $ per utterance: ${(jev_cost + luna_cost) / utterances if utterances else 0:.6f} "
                 f"over {utterances} utterances (incl. history turns)")
    lines.append("")

    # latency
    paths: dict[str, dict[str, list[float]]] = {"chime": defaultdict(list), "speech": defaultdict(list)}
    for r in ran:
        if not r.actual:
            continue
        a = r.actual
        path = "speech" if a["response_mode"] == "speech" else "chime"
        t = a["debug"]["timings_ms"]
        paths[path]["total"].append(t.get("total", 0.0))
        for j in a["debug"]["jev"]:
            paths[path]["jev"].append(j["latency_ms"])
        if a["debug"]["luna"] and a["debug"]["luna"].get("response_id"):
            paths[path]["luna"].append(a["debug"]["luna"]["latency_ms"])
    lines.append("## Latency (ms)")
    lines.append("")
    lines.append("| path / stage | p50 | p95 | max | n |")
    lines.append("|---|---|---|---|---|")
    for p in ("chime", "speech"):
        for stage in ("total", "jev", "luna"):
            lines.append(_lat_row(f"{p} / {stage}", paths[p][stage]))
    lines.append("")

    # failures
    failed = [r for r in ran if not r.passed]
    lines.append(f"## Failures ({len(failed)})")
    lines.append("")
    if failed:
        lines.append("| id | utterance | expected | actual | reasons |")
        lines.append("|---|---|---|---|---|")
        for r in sorted(failed, key=lambda x: x.id):
            exp = f"{r.expected['expected_outcome']}/{r.expected['expected_response_mode']}"
            act = f"{r.actual['outcome']}/{r.actual['response_mode']} «{r.actual['reply']}»" if r.actual else "–"
            reasons = "; ".join(r.failures).replace("|", "/")
            lines.append(f"| {r.id} | {r.utterance.replace('|', '/')} | {exp} | {act.replace('|', '/')} | {reasons} |")
    lines.append("")
    return "\n".join(lines)


def summary_numbers(run: RunResult) -> dict[str, Any]:
    ran = [r for r in run.records if not r.skipped]
    unseen = [r for r in ran if "unseen_entity" in r.tags]
    return {"run": len(ran), "passed": sum(r.passed for r in ran), "skipped": sum(r.skipped for r in run.records),
            "safety_fail": run.safety_failed_ids,
            "unseen": (sum(r.passed for r in unseen), len(unseen))}
