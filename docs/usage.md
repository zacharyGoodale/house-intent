# Usage and local data

## Routing

Jev selects intent and candidates from a runtime Home Assistant registry or the StubHA demo. The router resolves supported actions, checks explicit confirmation and safety policy, and makes service calls only after those checks pass. The built-in safety rules deny locks and alarm controls, and check scripts, scenes, and automations for security-sensitive targets. A `safety.toml` field replaces its default, so keep the generic rules when you override one.

StubHA runs the fictional demo home in `demo/` (registry snapshot, states and `demo/config/`) with a fixed clock. You can use `--overlay office_room` to add a demo office or `--state path/to/fixture.json` to override starting states for a CLI command.

## Optional conversational backends

Set `OPENAI_API_KEY` to enable the OpenAI fallback. Set `HERMES_BASE_URL` to your HTTPS Hermes endpoint including `/v1`, and provide `HERMES_API_KEY` or a local key file, if you use Hermes. An unset Hermes endpoint leaves the OpenAI fallback available. Use `--fallback luna` to choose the OpenAI backend directly or `--no-llm` to disable both.

Hermes can execute its own tools during an active conversation. Configure Hermes permissions on its server and grant only the tools you want it to use. Home Assistant service calls performed by the router pass through the router's safety policy.

## Speaker profiles and storage

The demo profiles are fictional. In your own `people.toml` (see the [configuration guide](configuration.md)), a profile can map a Home Assistant conversation agent to a person ID, a temperature unit, and optional calendars. Agent mappings are a routing convention. Authorization for the HTTP bridge is handled separately.

Optional answer backends may receive the utterance, current room and home capabilities, and any household profile or relevant memory you configure. Request logging is off by default. If you enable it in your private configuration, logs can contain utterances, replies, and routing data. Real memory is stored under `HA_JEV_DATA_DIR` or `~/.local/share/ha-jev`. Keep private configuration, logs, and memory outside the checkout.
