"""Export benchmark metrics and a Markdown report without raw tool inputs/outputs."""
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

MODELS = ('gpt-6-luna', 'gpt-6.1-sol')


def describe(rows):
    values = sorted(r['latency_ms'] / 1000 for r in rows)
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': values[math.ceil(.95 * len(values)) - 1], 'max': max(values),
            'tools': sum(len(r['tool_calls']) for r in rows),
            'with_tools': sum(bool(r['tool_calls']) for r in rows),
            'over_30s': sum(v > 30 for v in values)}


def report(source, destination):
    manifest = json.loads((source / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (source / 'results.jsonl').read_text().splitlines()]
    for row in rows:
        errors = 0
        for item in row.get('tool_items', []):
            if item.get('type') != 'function_call_output':
                continue
            value = item.get('output')
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    continue
            errors += bool(isinstance(value, dict) and value.get('error'))
        row['tool_error_count'] = errors
    scored = [r for r in rows if r['scored']]
    good = [r for r in scored if not r.get('error')]
    groups = {m: [r for r in good if r['requested_model'] == m] for m in MODELS}
    stats = {m: describe(v) for m, v in groups.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    metric_path = destination.with_suffix('.metrics.jsonl')
    fields = ('case_id', 'repeat', 'requested_model', 'latency_ms', 'http_status', 'status', 'runtime',
              'usage', 'tool_calls', 'tool_error_count', 'error')
    metric_path.write_text(''.join(json.dumps({k: r[k] for k in fields if k in r}) + '\n' for r in scored))
    luna, sol = (stats[m] for m in MODELS)
    ratio = sol['mean'] / luna['mean']
    lines = ['# Hermes Luna 6 vs Sol 6.1 latency benchmark', '',
             f"Run started {manifest['started_utc']}. Real production Hermes endpoint (version 0.21.5); low reasoning on both models.", '',
             f"Sol averaged **{sol['mean']:.2f}s**, versus **{luna['mean']:.2f}s** for Luna: "
             f"**{ratio:.2f}×** Luna’s latency ({(ratio-1)*100:.1f}% higher).", '',
             '| Model | Completed | Mean | Median | p95 | Max | Calls with tools | Total tool calls | >30s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        s = stats[m]
        lines.append(f"| `{m}` | {s['n']} | {s['mean']:.2f}s | {s['median']:.2f}s | {s['p95']:.2f}s | {s['max']:.2f}s | {s['with_tools']} | {s['tools']} | {s['over_30s']} |")
    lines += ['', '## Method', '',
              f"- Selected all {len(manifest['case_ids'])} existing `requires_llm: true` cases from `eval/golden.jsonl`; {manifest['repeats']} repeats per case/model. All are answer requests, with no requested device service calls or conversation histories.",
              '- Sent `provider: "openai-codex"`, exact model IDs, and `model_options.reasoning: {enabled: true, effort: "low"}` to `POST /v1/responses`. Verified the deployed agent factory resolves both models and low reasoning. The current Hermes default is Luna at low reasoning, and no fallback model is configured.',
              '- Used the configured Hermes tools, provider credentials, SOUL, and persistent context. Added the same instruction excluding front door locks, blinds/covers, espresso, orders, and messages to both models. The user explicitly authorized the real endpoint and home tool execution.',
              '- One request at a time; alternating model order within each pair and across repeats. Fresh response chain per request, `store: false`. One shared HTTP client with connection reuse. No dedicated warm-up: two preflight calls (Luna movie recommendation and Sol arithmetic) preceded the measured run.',
              '- Timed HTTP POST through the full non-streaming response using `perf_counter`. Includes transport, server agent setup, model generation, and server-side tools; excludes Jev, STT, and TTS. This is completion latency, not time to first token.',
              '- Client timeout was 180s to observe tails; the production ha-jev Hermes client timeout is 30s. The >30s column shows calls that would exceed that deadline.',
              '- This deployed Responses handler omits `runtime`; its `model` field echoes the request. Routing was checked separately against deployed source and agent construction, rather than treating the echoed label as proof of serving model.',
              f"- Golden SHA-256: `{manifest['golden_sha256']}`. {sum(bool(r.get('error')) for r in scored)} errors; error samples are excluded from successful-completion statistics. p95 uses nearest rank.",
              '- API override contract: [upstream Hermes API documentation](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/api-server.md).', '',
              '## Per-case measurements', '',
              'Times are seconds. Each cell lists three runs followed by their mean.', '',
              '| Case | Luna runs → mean | Sol runs → mean | Faster mean |', '|---|---|---|---|']
    wins = Counter()
    for case_id in manifest['case_ids']:
        cells = []
        means = []
        for m in MODELS:
            rr = sorted((r for r in groups[m] if r['case_id'] == case_id), key=lambda r: r['repeat'])
            vals = [r['latency_ms'] / 1000 for r in rr]
            avg = statistics.mean(vals)
            means.append(avg)
            cells.append(', '.join(f'{v:.2f}' for v in vals) + f' → **{avg:.2f}**')
        winner = 'Luna' if means[0] < means[1] else 'Sol'
        wins[winner] += 1
        lines.append(f"| `{case_id}` | {cells[0]} | {cells[1]} | {winner} |")
    lines += ['', f"Luna had the lower case mean in {wins['Luna']} cases; Sol in {wins['Sol']}."]
    lines += ['', '## Repeat and token summaries', '', '| Repeat | Luna mean | Sol mean |', '|---|---:|---:|']
    for repeat in range(1, manifest['repeats'] + 1):
        values = [statistics.mean(r['latency_ms'] / 1000 for r in groups[m] if r['repeat'] == repeat)
                  for m in MODELS]
        lines.append(f'| {repeat} | {values[0]:.2f}s | {values[1]:.2f}s |')
    lines.append('')
    for m in MODELS:
        usages = [(r.get('usage') or {}) for r in groups[m]]
        lines.append(f"- `{m}` mean reported tokens per call: {statistics.mean(u.get('input_tokens', 0) for u in usages):.0f} input, {statistics.mean(u.get('output_tokens', 0) for u in usages):.0f} output.")
    lines += ['', 'Usage is reported by Hermes across the agent turn, including repeated inputs for tool-loop iterations. Provider caching/reasoning breakdowns were not returned, so these totals do not establish uncached work or cost.', '', '## Tool use', '', '| Visible Hermes tool | Luna calls | Sol calls |', '|---|---:|---:|']
    counts = {m: Counter(t for r in groups[m] for t in r['tool_calls']) for m in MODELS}
    for tool in sorted(set(counts[MODELS[0]]) | set(counts[MODELS[1]])):
        lines.append(f'| `{tool}` | {counts[MODELS[0]][tool]} | {counts[MODELS[1]][tool]} |')
    lines += ['', f"Visible tool outputs containing an error: Luna {sum(r['tool_error_count'] for r in groups[MODELS[0]])}; Sol {sum(r['tool_error_count'] for r in groups[MODELS[1]])}. These are tool errors within completed responses, separate from HTTP/response failures.", '',
              'Code execution repeatedly returned `EnvironmentConnectionError`: the Docker command exists, but `docker version` fails. This benchmark therefore measures the current installation including broken code-execution tooling and model recovery, not healthy-tool efficiency. The server was not repaired during the run.', '',
              'Tool counts cover `function_call` items surfaced by Hermes. Provider-native browsing or other work may not appear in that list; zero visible Hermes calls does not prove an answer used no external tools.', '',
              '## Interpretation and limits', '',
              'For latency on these existing answer cases, keep Luna as the default: Sol’s lower visible tool count did not offset its longer completion times.', '',
              'This is a small repeated voice-answer workload, not a home-control or complex multi-step task suite. Three repeats help expose variability but do not establish a stable production latency distribution. Persistent Hermes context, prompt caches, backend load, and serial order can affect timings.', '',
              'The existing golden rubrics were written for the direct Luna/stub setup; some require declining live weather/news or matching a fixed fixture clock. They are not a fair pass-rate rubric for a live tool-enabled Hermes agent. No LLM quality judge was run. Inspect local raw replies before making quality claims.', '',
              f'Metrics without raw tool arguments/results: [{metric_path.name}]({metric_path.name}). Raw responses remain in the gitignored eval run directory.', '',
              '## Reproduce', '', '```sh',
              '.venv/bin/python scripts/benchmark_hermes.py \\',
              '  --base-url https://hermes.example.test/v1 \\',
              '  --repeats 3 --out eval/runs/hermes-latency',
              '.venv/bin/python scripts/report_hermes_benchmark.py \\',
              '  eval/runs/hermes-latency eval/runs/hermes-latency/report.md',
              '```', '']
    arithmetic = {m: [r for r in groups[m] if r['case_id'] == 'general-004'] for m in MODELS}
    if all(arithmetic.values()):
        lines += ['## Arithmetic recovery observation', '',
                  'For `general-004` (17 × 23), the correct answer is 391. Each model encountered the same Docker tool failure; the following counts inspect the returned answer directly, not an LLM judge.', '']
        for m in MODELS:
            hits = sum('391' in r.get('reply', '') for r in arithmetic[m])
            lines.append(f"- `{m}`: {hits}/{len(arithmetic[m])} replies contained the correct answer.")
        lines += ['', 'In this case Sol recovered from the tool failure more reliably, although the latency results do not support a speed advantage. This single case does not establish general model quality.', '']
    ha_writes = sum(r['tool_calls'].count('ha_call_service') for r in good)
    lines += [f'Visible HA service calls during the measured run: **{ha_writes}**.', '']
    destination.write_text('\n'.join(lines))
    print(json.dumps({'stats': stats, 'case_wins': dict(wins)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    report(args.source, args.destination)
