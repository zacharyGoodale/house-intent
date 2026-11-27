"""Check paired ordering, model overrides, isolation, and failed-sample accounting."""
import importlib.util
import json
from types import SimpleNamespace

import httpx

from ha_jev.config import REPO_ROOT
from ha_jev.evaluation.golden import load_golden

spec = importlib.util.spec_from_file_location('benchmark', REPO_ROOT / 'scripts/benchmark_hermes.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


async def test_pairs_and_failure_accounting(tmp_path, monkeypatch):
    case = next(c for c in load_golden(REPO_ROOT / 'eval/golden.jsonl') if c.id == 'general-004')
    monkeypatch.setattr(benchmark, 'load_golden', lambda _: [case])
    monkeypatch.setenv('HERMES_API_KEY', 'test-key')
    requests = []

    def respond(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': [{'enabled': True, 'tools': ['execute_code']}]})
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 3:
            return httpx.Response(503)
        return httpx.Response(200, json={
            'model': body['model'], 'status': 'completed', 'id': 'unused-chain',
            'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': '391'}]}],
            'usage': {'input_tokens': 1, 'output_tokens': 1}})

    original = httpx.AsyncClient
    monkeypatch.setattr(benchmark.httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    await benchmark.run(SimpleNamespace(base_url='https://example.test/v1', out=str(tmp_path),
                                         ids=None, repeats=3, timeout=10))
    assert [r['model'] for r in requests] == [*benchmark.MODELS, *benchmark.MODELS[::-1], *benchmark.MODELS]
    for request in requests:
        assert request['provider'] == 'openai-codex'
        assert request['model_options']['reasoning'] == {'enabled': True, 'effort': 'low'}
        assert 'previous_response_id' not in request
        assert 'front door lock' in request['instructions']
        assert 'espresso machine' in request['instructions']
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['gpt-6-luna']['completed'] == 3
    assert summary['gpt-6.1-sol']['completed'] == 2
    assert summary['gpt-6.1-sol']['errors'] == 1
    assert len((tmp_path / 'results.jsonl').read_text().splitlines()) == 6


def test_report_excludes_raw_tool_content_and_uses_nearest_rank_p95(tmp_path):
    spec = importlib.util.spec_from_file_location('report', REPO_ROOT / 'scripts/report_hermes_benchmark.py')
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    source = tmp_path / 'run'
    source.mkdir()
    manifest = {'started_utc': '2037-10-03', 'case_ids': ['general-004'], 'repeats': 3,
                'golden_sha256': 'test-hash'}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    rows = [dict(case_id='general-004', repeat=i + 1, requested_model=m, scored=True,
                 latency_ms=(i + 1) * 1000, tool_calls=['execute_code'],
                 tool_items=[{'type': 'function_call_output', 'output': '{"error":"private raw content"}'}])
            for m in benchmark.MODELS for i in range(3)]
    (source / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    destination = tmp_path / 'report.md'
    reporter.report(source, destination)
    metrics = destination.with_suffix('.metrics.jsonl').read_text()
    assert 'private raw content' not in metrics
    assert 'private raw content' not in destination.read_text()
    assert all(r['tool_error_count'] == 1 for r in map(json.loads, metrics.splitlines()))
    assert reporter.describe(rows)['p95'] == 3
