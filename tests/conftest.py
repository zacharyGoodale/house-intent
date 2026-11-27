"""The unit tests run against the fictional demo home (demo/): its registry snapshot, stub state and config."""
import os

from ha_jev.config import DEMO_CONFIG_DIR

# Forced, not defaulted: a developer's own $HA_JEV_CONFIG_DIR must not leak into the demo-home tests.
os.environ["HA_JEV_CONFIG_DIR"] = str(DEMO_CONFIG_DIR)
