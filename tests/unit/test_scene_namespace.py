"""With `qualified_scene_platforms = ["hue"]`, dashboard routines are the default and Hue scenes require their
spoken namespace. (Off by default: see test_scenes_are_offered_by_name_without_the_option.)"""

import shutil

import pytest

from ha_jev.config import DEMO_CONFIG_DIR

from ha_jev.models import Outcome
from tests.fakes import FakeJev
from tests.unit.test_core import make_core
from tests.unit.test_questions import ctx_for
from ha_jev.questions import build_request, build_retry_request


@pytest.fixture(autouse=True)
def hue_qualified(tmp_path, monkeypatch, request):
    if request.node.name == "test_scenes_are_offered_by_name_without_the_option":
        return
    cfg = tmp_path / "config"
    shutil.copytree(DEMO_CONFIG_DIR, cfg)
    scope = cfg / "scope.toml"
    scope.write_text('qualified_scene_platforms = ["hue"]\n' + scope.read_text())
    monkeypatch.setenv("HA_JEV_CONFIG_DIR", str(cfg))


@pytest.mark.parametrize("text,target", [
    ("relax", "script.common_relax"),
    ("bedtime", "script.demo_bedroom_bed"),
    ("movie", "script.demo_movie_mode"),
    ("evening", "script.demo_living_evening"),
    ("day", "script.demo_living_day"),
    ("wind down", "script.demo_living_wind_down"),
    ("Hue scene relax in the bedroom", "scene.demo_bedroom_relax"),
    ("activate the HUE SCENE Common Area Relax", "scene.demo_common_relax"),
])
async def test_scene_requests_offer_only_the_requested_namespace(text, target):
    jev = FakeJev({"intent": "activate_scene", "scene": target,
                   "area__bedroom": 1.0 if "bedroom" in text.casefold() else 0.0})
    core, ha = await make_core(jev)
    result = await core.handle(text, speaker_area="living_room")
    offered = jev.calls[0][1]["scene"]["criteria"]
    assert target in offered
    assert (set(offered) - {"none"}) <= (
        set(core.catalog.qualified_scenes) if target in core.catalog.qualified_scenes
        else set(core.catalog.scene_options) - set(core.catalog.qualified_scenes)
    )
    assert result.outcome == Outcome.EXECUTED
    assert [c.entity_ids for c in ha.calls] == [(target,)]


@pytest.mark.parametrize("text", ["relax", "relax scene", "Hue relax", "hues scene relax"])
async def test_a_model_cannot_activate_hue_without_the_qualifier(text):
    core, ha = await make_core(FakeJev({"intent": "activate_scene", "scene": "scene.demo_common_relax"}))
    result = await core.handle(text)
    assert result.outcome != Outcome.EXECUTED
    assert ha.calls == []


async def test_retry_cannot_reintroduce_an_unqualified_hue_scene():
    ctx, _ = await ctx_for("relax")
    req = build_retry_request(ctx, "scene", ["scene.demo_common_relax", "script.common_relax"])
    assert set(req.questions["scene"]["criteria"]) == {"script.common_relax", "none"}


async def test_compound_qualifier_applies_only_to_its_own_action():
    text = "activate Hue scene relax in the bedroom and activate movie mode"
    ctx, _ = await ctx_for(text)
    req = build_request(ctx)
    assert "scene.demo_bedroom_relax" in req.questions["p0__scene"]["criteria"]
    assert "script.demo_movie_mode" not in req.questions["p0__scene"]["criteria"]
    assert "script.demo_movie_mode" in req.questions["p1__scene"]["criteria"]
    assert "scene.demo_bedroom_relax" not in req.questions["p1__scene"]["criteria"]
    core, ha = await make_core(FakeJev({
        "intent": "activate_scene", "is_compound": 1.0,
        "p0__intent": "activate_scene", "p0__scene": "scene.demo_bedroom_relax", "p0__area__bedroom": 1.0,
        "p1__intent": "activate_scene", "p1__scene": "script.demo_movie_mode",
    }))
    result = await core.handle(text)
    assert result.outcome == Outcome.EXECUTED
    assert {t for c in ha.calls for t in c.entity_ids} == {"scene.demo_bedroom_relax", "script.demo_movie_mode"}


async def test_a_dashboard_script_cannot_hide_a_hue_scene_of_the_same_name():
    overlay = {"entities": [
        {"entity_id": "script.focus", "friendly_name": "Focus", "area_id": "bedroom"},
        {"entity_id": "scene.focus", "friendly_name": "Focus", "area_id": "bedroom", "platform": "hue"},
    ]}
    for text, target in (("focus", "script.focus"), ("Hue scene focus", "scene.focus")):
        core, ha = await make_core(FakeJev({"intent": "activate_scene", "scene": target}), overlays=[overlay])
        result = await core.handle(text, speaker_area="bedroom")
        assert result.outcome == Outcome.EXECUTED
        assert [c.entity_ids for c in ha.calls] == [(target,)]


async def test_scenes_are_offered_by_name_without_the_option():
    ctx, cat = await ctx_for("relax")
    assert not cat.qualified_scenes
    offered = set(build_request(ctx).questions["scene"]["criteria"])
    assert {"scene.demo_bedroom_relax", "scene.demo_common_natural_light", "script.demo_movie_mode"} <= offered
