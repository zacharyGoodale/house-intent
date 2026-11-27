import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ha_jev.config import ConfigError, load_dev_env, load_settings

ROOT = Path(__file__).resolve().parents[2]


def test_loads_the_three_checked_in_files():
    s = load_settings()
    assert s.policy.act == 0.70
    assert s.jev.model == "jev-latest"
    assert s.luna.model == "gpt-6-luna"
    assert s.scope.include_exposed is True
    assert "lock" in s.safety.denied_domains
    assert [p.id for p in s.scope.presets] == ["kitchen_day", "kitchen_evening"]
    assert s.scope.defaults.transport_player["bedroom"].startswith("media_player.")


def test_dotted_overrides_replace_values():
    s = load_settings(overrides={"policy.act": 0.8, "scope.include": ["script.*"]})
    assert s.policy.act == 0.8
    assert s.scope.include == ["script.*"]


def test_unknown_override_key_is_a_config_error():
    with pytest.raises(ConfigError):
        load_settings(overrides={"policy.nope": 1})


def test_config_dir_env_override(tmp_path, monkeypatch):
    for name in ("scope.toml", "safety.toml", "ha_jev.toml"):
        shutil.copy(ROOT / "config" / name, tmp_path / name)
    text = (tmp_path / "ha_jev.toml").read_text().replace("act = 0.70", "act = 0.75")
    (tmp_path / "ha_jev.toml").write_text(text)
    monkeypatch.setenv("HA_JEV_CONFIG_DIR", str(tmp_path))
    assert load_settings().policy.act == 0.75


def _write_scope_with_preset(tmp_path, call: str):
    for name in ("safety.toml", "ha_jev.toml"):
        shutil.copy(ROOT / "config" / name, tmp_path / name)
    (tmp_path / "scope.toml").write_text(
        'include_exposed = true\ninclude = []\nexclude = []\n[[presets]]\nid = "bad"\nname = "Bad"\n'
        f"calls = [{call}]\n"
    )


def test_preset_with_lock_call_fails_at_load(tmp_path):
    _write_scope_with_preset(tmp_path, '{ domain = "lock", service = "unlock", entity_ids = ["lock.x"], data = {} }')
    with pytest.raises(ConfigError):
        load_settings(config_dir=tmp_path)


def test_preset_with_kelvin_key_fails_at_load(tmp_path):
    _write_scope_with_preset(tmp_path, '{ domain = "light", service = "turn_on", entity_ids = ["light.x"], data = { kelvin = 3000 } }')
    with pytest.raises(ConfigError):
        load_settings(config_dir=tmp_path)


def test_preset_calling_a_denylisted_script_fails_at_load(tmp_path):
    _write_scope_with_preset(tmp_path, '{ domain = "script", service = "turn_on", entity_ids = ["script.toggle_entry_lock"], data = {} }')
    with pytest.raises(ConfigError):
        load_settings(config_dir=tmp_path)


def test_dev_env_loader_takes_only_whitelisted_keys_and_never_overwrites(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TYPESAFE_API_KEY=ts-123\nOPENAI_API_KEY='sk-abc'\nHA_URL=http://example.test:8123\n"
                   "HA_TOKEN=ha-test-token\nHA_JEV_ALLOW_REAL=1\nOTHER_SECRET=nope\n# comment\n")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "already")
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    for key in ("HA_URL", "HA_TOKEN", "HA_JEV_ALLOW_REAL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HA_JEV_DEV_ENV", str(env))
    loaded = load_dev_env()
    assert os.environ["TYPESAFE_API_KEY"] == "ts-123"
    assert os.environ["OPENAI_API_KEY"] == "already"
    assert "OTHER_SECRET" not in os.environ
    assert os.environ["HA_URL"] == "http://example.test:8123"
    assert os.environ["HA_TOKEN"] == "ha-test-token"
    assert os.environ["HA_JEV_ALLOW_REAL"] == "1"
    assert loaded == ["TYPESAFE_API_KEY", "HA_URL", "HA_TOKEN", "HA_JEV_ALLOW_REAL"]


def test_dev_env_loader_is_opt_in(monkeypatch):
    monkeypatch.delenv("HA_JEV_DEV_ENV", raising=False)
    assert load_dev_env() == []


def test_no_env_file_is_tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert not [f for f in out.splitlines() if Path(f).name.startswith(".env")]


def test_people_load_from_people_toml():
    s = load_settings()
    assert s.person("morgan").temperature_unit == "C"
    assert s.person("riley").name == "Riley"
    assert s.person(None) is None and s.person("nobody") is None


def test_people_file_is_optional_and_validated(tmp_path):
    for name in ("scope.toml", "safety.toml", "ha_jev.toml"):
        shutil.copy(ROOT / "config" / name, tmp_path / name)
    assert load_settings(config_dir=tmp_path).people == {}
    (tmp_path / "people.toml").write_text('[people.x]\nname = "X"\ntemperature_unit = "K"\n')
    with pytest.raises(ConfigError):
        load_settings(config_dir=tmp_path)
