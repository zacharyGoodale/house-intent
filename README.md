# House Intent

A voice-command router for Home Assistant. It turns a transcript into a spoken reply, a confirmation chime, or a guarded Home Assistant service call. Each turn uses one Jev (TypeSafe System One) request to classify intent and targets. The router checks the result before it acts. Optional Hermes and OpenAI backends answer open-ended questions.

## Try the demo

Clone the repository and run these commands from the checkout. You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

    uv sync --locked
    uv run pytest

The default test suite uses fictional fixtures and fake clients. Set `TYPESAFE_API_KEY` in your environment to try these CLI examples.

    uv run ha-jev ask "turn on the living room lights" --area living_room --no-llm
    uv run ha-jev ask "what's on my calendar today" --person riley --no-llm
    uv run ha-jev tui --ha stub --area living_room --no-llm

These commands use StubHA, an in-memory Home Assistant demo. `--no-llm` disables the optional conversational backends. The CLI still calls Jev.

## Connect Home Assistant

Use a copy of `config/` outside the checkout for your own entity scope, safety rules, and optional speaker profiles. See the [Home Assistant setup guide](docs/home-assistant.md) for real-HA access, the authenticated voice bridge, and the thermostat adapter. See the [usage guide](docs/usage.md) for routing, optional backends, and local data.

## Evaluate

    uv run python eval/validate_golden.py
    uv run ha-jev eval --no-llm --limit 20

The validator runs offline. The CLI eval calls Jev against the fictional home and writes results to the ignored `eval/runs/` directory. Live integration tests are separately marked and opt-in.

## Project layout

- `src/ha_jev/` holds the router, policy, CLI, and bridge.
- `custom_components/ha_jev/` provides Home Assistant conversation, STT, TTS, and schedules.
- `config/` and `fixtures/` supply public demo configuration and fictional data.
- `tests/` and `eval/` exercise routing, safety, and integration behavior.
