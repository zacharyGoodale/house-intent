# Golden evaluation

The cases in `golden.jsonl` use the fictional registry and states under `fixtures/`. They describe an utterance, its expected outcome and service calls, and any state, profile, or fixture overlay required by the case.

Run the offline validation with `uv run python eval/validate_golden.py`. Run selected cases through the real router with `uv run ha-jev eval --no-llm --tag safety` after setting `TYPESAFE_API_KEY`. Eval uses StubHA and stores reports in `eval/runs/`.
