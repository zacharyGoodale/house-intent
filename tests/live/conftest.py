"""Live tests are opt-in: `uv run pytest -m live` (Jev) and `-m live_llm` (Luna). Keys come from the
environment, or from $HA_JEV_DEV_ENV via the whitelisted dev loader; tests skip without them."""

from ha_jev.config import load_dev_env

load_dev_env()
