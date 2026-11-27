"""Read-only synthetic Hermes SSE probe; credentials stay in environment.

The deployed server does not enforce tools=[]/tool_choice=none. This probe asks
only for a synthetic explanation, never forwards any output to speech, and
records whether tool events occurred. Missing final phase is deliberately unsafe.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import ssl
import time

import httpx


def is_final(item):
    return (item.get('type') == 'message' and item.get('role') == 'assistant'
            and item.get('phase') == 'final_answer')


def self_test():
    for item in [dict(type='reasoning'), dict(type='function_call'),
                 dict(type='function_call_output'),
                 dict(type='message', role='assistant', phase='commentary'),
                 dict(type='message', role='assistant'),
                 dict(type='message', role='user', phase='final_answer')]:
        assert not is_final(item)
    assert is_final(dict(type='message', role='assistant', phase='final_answer'))
    print('Synthetic reasoning/tool/commentary/ambiguous-item filtering: passed')


async def probe(args):
    tls = ssl.create_default_context(cafile=os.environ.get('HERMES_CA_FILE') or None)
    payload = dict(model=os.environ.get('HERMES_MODEL', 'hermes-agent'),
                   input='Explain how rain forms in five short sentences.',
                   instructions='Answer the synthetic science question directly. Do not use tools.',
                   stream=True, store=False, tools=[], tool_choice='none')
    runs = []
    async with httpx.AsyncClient(verify=tls, timeout=120, follow_redirects=False) as client:
        for number in range(1, args.runs + 1):
            start = time.monotonic()
            run = dict(run=number, events=[], first_raw_delta_ms=None,
                       first_safe_delta_ms=None, completion_ms=None)
            items, texts, safe = {}, {}, []
            terminal = None
            data_lines = []
            try:
                async with client.stream('POST', os.environ['HERMES_BASE_URL'].rstrip('/') + '/responses',
                                         json=payload, headers={'Authorization': 'Bearer ' + os.environ['HERMES_API_KEY']}) as response:
                    run['http_status'] = response.status_code
                    run['content_type'] = response.headers.get('content-type')
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith('data:'):
                            data_lines.append(line[5:].lstrip())
                            continue
                        if line or not data_lines:
                            continue
                        raw = '\n'.join(data_lines)
                        data_lines.clear()
                        if raw == '[DONE]':
                            continue
                        event = json.loads(raw)
                        elapsed = round((time.monotonic() - start) * 1000, 3)
                        kind = event.get('type', 'unknown')
                        row = dict(elapsed_ms=elapsed, type=kind, sequence=event.get('sequence_number'),
                                   output_index=event.get('output_index'))
                        item = event.get('item')
                        if item:
                            items[item['id']] = item
                            row['item'] = {k: item.get(k) for k in ('type', 'role', 'phase', 'status')}
                        if kind == 'response.output_text.delta':
                            item_id = event.get('item_id')
                            delta = event.get('delta', '')
                            texts[item_id] = texts.get(item_id, '') + delta
                            row['delta_chars'] = len(delta)
                            row['safe_final'] = is_final(items.get(item_id, {}))
                            if run['first_raw_delta_ms'] is None:
                                run['first_raw_delta_ms'] = elapsed
                            if row['safe_final']:
                                safe.append(delta)
                                if run['first_safe_delta_ms'] is None:
                                    run['first_safe_delta_ms'] = elapsed
                        if kind in ('response.completed', 'response.failed', 'response.incomplete'):
                            terminal = event.get('response', {})
                            run['completion_ms'] = elapsed
                        run['events'].append(row)
                run['terminal_status'] = (terminal or {}).get('status')
                run['runtime'] = {k: v for k, v in (terminal or {}).get('runtime', {}).items()
                                  if k in ('model', 'provider')}
                output = (terminal or {}).get('output', [])
                answer = ''.join(p.get('text', '') for item in output
                                 if item.get('type') == 'message' and item.get('role') == 'assistant'
                                 and item.get('phase') != 'commentary'
                                 for p in item.get('content', []) if p.get('type') == 'output_text')
                run['terminal_answer'] = answer
                run['raw_delta_text'] = ''.join(texts.values())
                run['raw_matches_terminal'] = run['raw_delta_text'] == answer
                run['safe_matches_terminal'] = ''.join(safe) == answer and bool(answer)
                run['tool_items_observed'] = any(e.get('item', {}).get('type') in ('function_call', 'function_call_output') for e in run['events'])
                seq = [e['sequence'] for e in run['events'] if e['sequence'] is not None]
                run['sequence_strictly_increasing'] = bool(seq) and all(a < b for a, b in zip(seq, seq[1:]))
            except Exception as error:
                run['error_type'] = type(error).__name__
            runs.append(run)
            print(json.dumps({k: v for k, v in run.items() if k not in ('events', 'terminal_answer', 'raw_delta_text')}), flush=True)
    Path(args.output).write_text(json.dumps(dict(prompt=payload['input'], tool_disabling_enforced=False, runs=runs), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output', default='/tmp/ha-jev-hermes-stream.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        asyncio.run(probe(args))
