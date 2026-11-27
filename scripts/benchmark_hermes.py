"""Sequential, paired Hermes latency benchmark; uses the real Hermes endpoint; excludes lock, blinds, and espresso requests."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ha_jev.catalog import Catalog
from ha_jev.config import REPO_ROOT, load_settings
from ha_jev.evaluation.golden import load_golden
from ha_jev.ha.stub import StubHA
from ha_jev.hermes import HermesClient, build_hermes_instructions
from ha_jev.luna import LunaProfile
from ha_jev.memory import Fact, luna_blocks
from ha_jev.profiles import speaker_sentence

MODELS = ('gpt-6-luna', 'gpt-6.1-sol')


def final_text(data):
    return ''.join(p['text'] for item in data.get('output', [])
                   if item.get('type') == 'message' and item.get('role') == 'assistant'
                   and item.get('phase') != 'commentary'
                   for p in item.get('content', []) if p.get('type') == 'output_text').strip()


async def run(args):
    settings = load_settings()
    key = HermesClient(settings.hermes, settings.luna)._key()
    base = args.base_url.rstrip('/')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    golden = REPO_ROOT / 'eval/golden.jsonl'
    cases = [c for c in load_golden(golden) if c.requires_llm]
    if args.ids:
        cases = [c for c in cases if c.id in args.ids.split(',')]
    if not cases or args.repeats < 1:
        raise ValueError('Need cases and at least one repeat')
    async with httpx.AsyncClient(timeout=args.timeout, headers={'Authorization': 'Bearer ' + key}) as client:
        inventory = (await client.get(base + '/toolsets'))
        inventory.raise_for_status()
        enabled = {t for row in inventory.json()['data'] if row['enabled'] for t in row['tools']}
        manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
                    'models': MODELS, 'provider': 'openai-codex', 'reasoning_effort': 'low',
                    'repeats': args.repeats, 'case_ids': [c.id for c in cases], 'history_turns_timed': False,
                    'enabled_tools': sorted(enabled), 'excluded_devices': ['front door lock', 'blinds', 'espresso machine'], 'tool_policy': 'production Hermes permissions; device exclusions in request instructions', 'golden_sha256': hashlib.sha256(golden.read_bytes()).hexdigest(),
                    'timeout_s': args.timeout, 'timing': 'HTTP POST through full non-streaming response; no warmup; sequential paired order alternates'}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        rows = []
        with (out / 'results.jsonl').open('w') as output:
            for repeat in range(args.repeats):
                for index, case in enumerate(cases):
                    s = load_settings(overrides=case.config_overrides)
                    ha = StubHA.default(now=case.now, initial_state=case.initial_state)
                    catalog = Catalog.build(await ha.get_registry(), await ha.get_states(), s.scope, s.safety, s.capabilities)
                    speaker = s.person(case.person)
                    facts = [Fact(str(i), m['text'], m['owner'], '2037-09-01T12:00:00-04:00', None)
                             for i, m in enumerate(case.memory)]
                    shared, personal = luna_blocks(facts, speaker)
                    profile = LunaProfile(memory=shared, speaker=speaker_sentence(speaker, s.people) + ' ' + personal)
                    area = catalog.areas.get(case.speaker_area)
                    instructions = build_hermes_instructions(ha.now, area.name if area else None,
                                                             s.luna, catalog.luna_summary(), profile)
                    instructions += (' Benchmark restriction: never operate the front door lock, any blinds or covers, '
                                     'or the espresso machine. Do not place orders or send messages. '
                                     'Use supplied speaker and memory context for this case.')
                    for model in (MODELS if (repeat + index) % 2 == 0 else MODELS[::-1]):
                        previous = None
                        # Existing histories, if any, are separate untimed turns; never shared across models/repeats.
                        for utterance in [*case.history, case.utterance]:
                            payload = {'model': model, 'provider': 'openai-codex',
                                       'model_options': {'reasoning': {'enabled': True, 'effort': 'low'}},
                                       'instructions': instructions, 'input': utterance, 'store': bool(case.history)}
                            if previous:
                                payload['previous_response_id'] = previous
                            start = time.perf_counter()
                            row = {'case_id': case.id, 'repeat': repeat + 1, 'requested_model': model,
                                   'utterance': utterance, 'scored': utterance == case.utterance}
                            try:
                                response = await client.post(base + '/responses', json=payload)
                                row['latency_ms'] = round((time.perf_counter() - start) * 1000, 3)
                                row['http_status'] = response.status_code
                                response.raise_for_status()
                                data = response.json()
                                previous = data.get('id')
                                runtime = data.get('runtime') or {}
                                tools = [item['name'] for item in data.get('output', []) if item.get('type') == 'function_call']
                                row.update(status=data.get('status'), response_model=data.get('model'), runtime=runtime, reply=final_text(data),
                                           usage=data.get('usage'), tool_calls=tools,
                                           tool_items=[i for i in data.get('output', []) if i.get('type') in ('function_call', 'function_call_output')],
                                           expected_reply=case.expected.get('reply'))
                                if (runtime and runtime.get('model') != model) or data.get('model') != model:
                                    raise RuntimeError('Serving runtime differs from requested runtime')
                                if data.get('status') != 'completed' or not row['reply']:
                                    raise RuntimeError('Incomplete or empty response')
                            except (httpx.HTTPError, RuntimeError) as exc:
                                row.update(latency_ms=round((time.perf_counter() - start) * 1000, 3), error=type(exc).__name__)
                            output.write(json.dumps(row, ensure_ascii=False) + '\n')
                            output.flush()
                            rows.append(row)
                            print(case.id, repeat + 1, model, round(row['latency_ms']/1000, 2), row.get('tool_calls', []), row.get('error', ''), flush=True)
        summary = {}
        for model in MODELS:
            successful = [r for r in rows if r['scored'] and r['requested_model'] == model and not r.get('error')]
            ms = [r['latency_ms'] for r in successful]
            summary[model] = {'completed': len(ms), 'errors': sum(bool(r.get('error')) for r in rows if r['requested_model'] == model),
                              'mean_ms': statistics.mean(ms) if ms else None,
                              'median_ms': statistics.median(ms) if ms else None,
                              'min_ms': min(ms) if ms else None, 'max_ms': max(ms) if ms else None,
                              'tool_calls': sum(len(r['tool_calls']) for r in successful)}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Hermes endpoint, including /v1; tools execute live')
    parser.add_argument('--out', default='eval/runs/hermes-latency')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--ids', help='Optional comma-separated requires_llm case IDs')
    asyncio.run(run(parser.parse_args()))
