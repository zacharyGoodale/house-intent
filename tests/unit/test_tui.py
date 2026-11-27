"""Interactive terminal commands use the normal Core path and keep session context."""

from ha_jev import cli
from ha_jev.config import load_settings
from ha_jev.core import Core
from ha_jev.ha.stub import StubHA
from ha_jev.luna import NullLuna
from ha_jev.tui import run
from tests.fakes import FakeJev


async def test_tui_executes_follow_up_in_one_session():
    jev = FakeJev({"intent": "device_control", "scope": "one_named", "target": "light.alder",
                   "light_action": "turn_off"})
    ha = StubHA.default()
    core = await Core.create(load_settings(), ha, jev, NullLuna(), clock=lambda: ha.now)
    commands = iter(["turn off Alder", "/help", "/quit"])
    lines = []
    await run(core, ha_name="stub", area="living_room", reader=lambda _: next(commands), writer=lines.append)
    assert [call.describe() for call in ha.calls] == ["light.turn_off light.alder {}"]
    assert any("[executed] Turning off Alder." in line for line in lines)
    assert any("light.turn_off light.alder" in line for line in lines)
    assert len(jev.calls) == 1
    assert core.sessions.snapshot("terminal").recent_targets == ["light.alder"]


def test_tui_defaults_to_real_and_requires_live_opt_in(monkeypatch, capsys):
    monkeypatch.delenv("HA_JEV_ALLOW_REAL", raising=False)
    assert cli.build_parser().parse_args(["tui"]).ha == "real"
    assert cli.main(["tui", "--no-llm"]) == 2
    assert "HA_JEV_ALLOW_REAL=1" in capsys.readouterr().err


async def test_tui_shows_clarification_options():
    ha = StubHA.default()
    core = await Core.create(load_settings(), ha, FakeJev({"intent": "device_control", "scope": "unspecified",
                                                            "device_type": "light", "light_action": "turn_on"}),
                             NullLuna(), clock=lambda: ha.now)
    commands = iter(["lights on", "/quit"])
    lines = []
    await run(core, ha_name="stub", reader=lambda _: next(commands), writer=lines.append)
    assert any("[clarify]" in line for line in lines)
    assert not ha.calls
