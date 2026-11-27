"""Free-form music (music.py, docs/usage.md): span extraction, twin pairing, the TV rule, the play plan,
choosing from search results, and the proxy guard. End to end over StubHA with FakeJev."""

import pytest

from ha_jev import capabilities as caps
from ha_jev.catalog import Catalog
from ha_jev.config import MusicConfig, load_settings
from ha_jev.ha.interface import EntityEntry, Registry
from ha_jev.ha.stub import StubHA
from ha_jev.models import Outcome, ResponseMode, ServiceCall
from ha_jev.music import choose, pair_players, parse
from ha_jev.safety import SafetyViolation, guard
from tests.fakes import FakeJev
from tests.unit.test_core import check_invariants, make_core

STEREO_TWIN, BEDROOM_SPEAKER_TWIN = "media_player.demo_stereo_music", "media_player.demo_bedroom_music"


async def catalog(overrides=None, overlays=()):
    ha = StubHA.default(overlays=overlays)
    s = load_settings(overrides=overrides).with_scope_extension(**ha.overlay_scope)
    return Catalog.build(await ha.get_registry(), await ha.get_states(), s.scope, s.safety)


def spans(req):
    return [(s.text, s.hint, s.artist) for s in req.spans] if req else None


# ---- extraction --------------------------------------------------------------------------------
@pytest.mark.parametrize("text,first", [
    ("play some jazz", ("jazz", None, None)),
    ("play Taylor Swift in the bedroom", ("Taylor Swift", None, None)),
    ("play the album Blonde", ("Blonde", "album", None)),
    ("play my playlist Scenic Drive", ("Scenic Drive", "my_playlist", None)),
    ("play my Scenic Drive playlist", ("Scenic Drive", "my_playlist", None)),
    ("play Hello by Adele on the Stereo", ("Hello", None, "Adele")),
    ("could you put on some relaxing music", ("relaxing", None, None)),
    ("I want to listen to the station WDEX", ("WDEX", "radio", None)),
])
async def test_spans_are_cut_from_the_utterance(text, first):
    assert spans(parse(text, await catalog()))[0] == first


@pytest.mark.parametrize("text", ["play the Cedar Soundbar", "Put on YouTube", "Play Morning Brief", "play music", "play something",
                                  "what's playing on the Stereo?", "turn on the backlight lights"])
async def test_devices_scenes_apps_and_no_music_are_not_music(text):
    assert parse(text, await catalog()) is None  # these keep their existing paths


async def test_a_place_word_inside_a_title_stays_a_candidate():
    req = parse("play the bedroom playlist in the living room", await catalog())
    assert [s.text for s in req.spans] == ["the bedroom playlist", "the bedroom playlist in the living room"]
    assert parse("play jazz everywhere", await catalog()).everywhere


async def test_play_questions_are_asked_only_for_play_requests():
    core, _ = await make_core(FakeJev({"intent": "device_control"}))
    await core.handle("turn off the lights", speaker_area="living_room")
    assert "play_span" not in core.jev.calls[0][1]
    await core.handle("play the album Blonde", speaker_area="living_room")
    q = core.jev.calls[1][1]
    assert set(q["play_span"]["criteria"]) == {"s0", "s1", "none"} and "album" in q["content_type"]["criteria"]


# ---- twins and the TV rule ---------------------------------------------------------------------
def _entry(eid, name, platform, unique_id=None):
    return EntityEntry(eid, name, platform=platform, unique_id=unique_id, features=frozenset({"play"}))


def test_twins_pair_by_config_then_unique_id_then_name():
    reg = Registry({}, {}, {e.entity_id: e for e in [
        _entry("media_player.a", "Den", "sonos", "RINCON_1"), _entry("media_player.b", "Hall", "sonos", "RINCON_2"),
        _entry("media_player.c", "Loft", "sonos"),
        _entry("media_player.x", "Something else", "music_assistant", "RINCON_1"),
        _entry("media_player.y", "Hall", "music_assistant"), _entry("media_player.z", "Web player", "music_assistant"),
        _entry("media_player.w", "Loft two", "music_assistant")]})
    cfg = MusicConfig(integration="music_assistant", players={"media_player.c": "media_player.w"})
    assert pair_players(reg, cfg, caps.DEFAULT) == {"media_player.a": "media_player.x", "media_player.b": "media_player.y",
                                                    "media_player.c": "media_player.w"}
    assert pair_players(reg, MusicConfig(), caps.DEFAULT) == {}  # music off: no twins


async def test_twins_are_never_candidates_and_calendars_never_in_scope():
    cat = await catalog()
    assert cat.music_twins["media_player.demo_stereo"] == STEREO_TWIN and cat.music_twins["media_player.demo_bedroom_speaker"] == BEDROOM_SPEAKER_TWIN
    assert not set(cat.music_twins.values()) & set(cat.entities)
    assert not [e for e in cat.entities if e.startswith("calendar.")]


@pytest.mark.parametrize("features,device_class,tv", [
    ({"play", "volume_set", "select_source"}, None, True),                      # a streaming box with inputs
    ({"play", "volume_set", "select_source", "media_enqueue"}, None, False),   # a speaker with saved favourites
    ({"play", "volume_set", "select_source"}, "speaker", False),               # a declared speaker
    ({"play", "select_source", "media_enqueue"}, "tv", True),                  # declared TV
])
def test_select_source_alone_never_makes_a_speaker_a_tv(features, device_class, tv):
    t = caps.DEFAULT
    assert t.is_tv("media_player", features, device_class) is tv
    assert (t.kind_for("media_player", features, device_class) == "TV") is tv
    assert t.is_music_speaker("media_player", device_class, features) is (not tv)


async def test_a_speaker_whose_favourites_appear_as_sources_stays_a_music_speaker():
    ov = {"entities": [{"entity_id": "media_player.study_speaker", "friendly_name": "Study Speaker", "area_id": "bedroom",
                        "device_class": "speaker", "features": ["play", "pause", "volume_set", "select_source",
                                                                "media_enqueue", "play_media"],
                        "capabilities": {"source_list": ["Morning Jazz", "WCEX"]}}]}
    cat = await catalog(overlays=[ov])
    assert "media_player.study_speaker" in cat.music_speakers
    assert cat.entities["media_player.study_speaker"].kind == "speaker"
    assert cat.entities["media_player.demo_tv"].kind == "TV"


# ---- choosing ----------------------------------------------------------------------------------
RESULT = {"albums": [{"media_type": "album", "uri": "a:1", "name": "Blonde", "artists": [{"name": "Frank Ocean"}]}],
          "tracks": [{"media_type": "track", "uri": "t:1", "name": "Blonde", "artists": [{"name": "X"}]}],
          "artists": [{"media_type": "artist", "uri": "ar:1", "name": "Blondie"}]}


def test_choose_prefers_exact_names_and_types():
    assert choose(RESULT, "Blonde", "album", None) == ([RESULT["albums"][0]], True)
    items, exact = choose(RESULT, "Blonde", "unknown", None)
    assert items == [RESULT["albums"][0]] and exact is False  # an exact album beats the track; unknown is spoken
    assert choose({"albums": []}, "nothing", "album", None) == ([], False)
    dup = {"albums": [{"uri": "q", "name": "Greatest Hits", "artists": [{"name": "Queen"}]},
                      {"uri": "f", "name": "Greatest Hits", "artists": [{"name": "Fleetwood Mac"}]}]}
    assert [i["uri"] for i in choose(dup, "Greatest Hits", "album", None)[0]] == ["q", "f"]
    assert [i["uri"] for i in choose(dup, "Greatest Hits", "album", "Queen")[0]] == ["q"]


# ---- end to end --------------------------------------------------------------------------------
def play(span="s0", ctype="unknown", **extra):
    return FakeJev({"intent": "device_control", "media_action": "play_content", "play_span": span,
                    "content_type": ctype} | extra)


async def test_play_the_album_blonde_in_the_speakers_room():
    core, ha = await make_core(play(ctype="album"))
    r = await core.handle("play the album Blonde", speaker_area="living_room")
    check_invariants(r)
    assert r.outcome == Outcome.EXECUTED and r.response_mode == ResponseMode.CHIME_SUCCESS
    assert r.reply == "Playing Blonde by Frank Ocean."
    assert [c.to_dict() for c in ha.calls] == [{"domain": "music_assistant", "service": "play_media",
                                                 "entity_ids": [STEREO_TWIN], "data": {
        "media_id": "library://album/demo-08", "media_type": "album", "enqueue": "replace",
        "radio_mode": False}}]
    search = [rc for rc in ha.reads if rc.service == "search"]
    assert search and len(core.jev.calls) == 1  # decision 1: the search is a read, not another Jev call


async def test_play_some_jazz_is_a_spoken_genre_playlist_with_radio_mode():
    core, ha = await make_core(play(ctype="genre"))
    r = await core.handle("play some jazz", speaker_area="bedroom")
    assert r.outcome == Outcome.EXECUTED and r.response_mode == ResponseMode.SPEECH
    assert r.reply == "Playing Jazz Classics."
    call = ha.calls[0]
    assert call.entity_ids == (BEDROOM_SPEAKER_TWIN,) and call.data["radio_mode"] is True and call.data["media_type"] == "playlist"


async def test_a_named_room_beats_the_speakers_room():
    core, ha = await make_core(play(ctype="artist", area__bedroom=0.97))
    r = await core.handle("play Taylor Swift in the bedroom", speaker_area="living_room")
    assert r.reply == "Playing Taylor Swift." and r.response_mode == ResponseMode.CHIME_SUCCESS
    assert ha.calls[0].entity_ids == (BEDROOM_SPEAKER_TWIN,) and ha.calls[0].data["radio_mode"] is True


async def test_radio_my_playlist_and_discover_weekly():
    core, ha = await make_core(play(ctype="radio"))
    r = await core.handle("play WDEX", speaker_area="living_room")
    assert r.reply == "Playing WDEX 93.9 FM." and r.response_mode == ResponseMode.SPEECH  # not an exact name
    assert ha.calls[0].data["media_id"] == "library://radio/demo-39"
    core, ha = await make_core(play(ctype="my_playlist"))
    r = await core.handle("play my playlist Scenic Drive", speaker_area="living_room")
    assert r.reply == "Playing Scenic Drive." and r.response_mode == ResponseMode.CHIME_SUCCESS
    assert [rc.service for rc in ha.reads] == ["get_library"]
    core, ha = await make_core(play(ctype="my_playlist"))
    assert (await core.handle("play Weekly Sampler", speaker_area="bedroom")).reply == "Playing Weekly Sampler."


async def test_nothing_found_is_spoken_and_plays_nothing():
    core, ha = await make_core(play())
    r = await core.handle("play asdfgh", speaker_area="living_room")
    check_invariants(r)
    assert r.outcome == Outcome.UNSUPPORTED and r.response_mode == ResponseMode.SPEECH
    assert r.reply == "I couldn't find asdfgh." and ha.calls == []


async def test_same_name_albums_ask_and_the_answer_plays_it():
    def answers(state, questions):
        if "pending_clarification" in state:
            uri = next(k for k, v in questions["clarify_pick"]["criteria"].items() if "Queen" in str(v))
            return {"intent": "clarify_reply", "clarify_pick": uri}
        return {"intent": "device_control", "media_action": "play_content", "play_span": "s0", "content_type": "album"}
    core, ha = await make_core(FakeJev(answers))
    r = await core.handle("play the album Greatest Hits", speaker_area="living_room", session_id="s")
    assert r.outcome == Outcome.CLARIFY and r.reply == "Which one: Greatest Hits by Queen or Greatest Hits by Fleetwood Mac?"
    r = await core.handle("the Queen one", speaker_area="living_room", session_id="s")
    assert r.outcome == Outcome.EXECUTED and ha.calls[0].data["media_id"] == "library://album/demo-14"


async def test_a_podcast_script_still_runs_through_the_scene_path():
    core, ha = await make_core(play(intent="activate_scene", scene="script.play_morning_brief"))
    r = await core.handle("Play MORNING NEWS's Morning Brief on the Stereo", speaker_area="living_room")
    assert [c.describe() for c in ha.calls] == ["script.turn_on script.play_morning_brief {}"]


async def test_a_similar_script_the_user_did_not_name_does_not_win():
    core, ha = await make_core(play(ctype="podcast", scene="script.play_morning_brief"))
    r = await core.handle("play The Daily", speaker_area="living_room")
    assert ha.calls[0].domain == "music_assistant" and ha.calls[0].data["media_type"] == "podcast"
    assert r.reply == "Playing The Daily."


async def test_type_words_in_the_chosen_span_are_read_as_its_type():
    core, ha = await make_core(play(span="s1"))  # Jev picks the words as said: "my Scenic Drive playlist"
    r = await core.handle("play my Scenic Drive playlist", speaker_area="living_room")
    assert r.reply == "Playing Scenic Drive." and [rc.service for rc in ha.reads] == ["get_library"]


async def test_an_unsure_type_searches_everything():
    core, ha = await make_core(play(ctype=("track", 0.55)))
    r = await core.handle("play Blonde", speaker_area="living_room")
    assert r.reply == "Playing Blonde by Frank Ocean." and r.response_mode == ResponseMode.SPEECH


async def test_music_not_set_up_or_everywhere_is_unsupported():
    core, ha = await make_core(play(), overrides={"scope.music.integration": None})
    r = await core.handle("play some jazz", speaker_area="living_room")
    assert r.outcome == Outcome.UNSUPPORTED and r.reply == "I can't play specific music yet." and ha.calls == []
    core, ha = await make_core(play())
    r = await core.handle("play jazz everywhere", speaker_area="living_room")
    assert r.reply == "I can only play music in one room at a time for now." and ha.calls == []


async def test_a_failed_search_is_an_error_chime():
    core, ha = await make_core(play(ctype="genre"))
    ha.fail_on.add("music_assistant.search")
    r = await core.handle("play some jazz", speaker_area="living_room")
    assert r.outcome == Outcome.ERROR and r.response_mode == ResponseMode.CHIME_ERROR and ha.calls == []


async def test_no_room_and_no_speaker_area_asks_which_room():
    core, ha = await make_core(play(ctype="genre"))
    r = await core.handle("play some jazz")
    assert r.outcome == Outcome.CLARIFY and r.reply.startswith("Which room") and ha.calls == []


# ---- guard -------------------------------------------------------------------------------------
async def test_the_play_service_may_target_only_twins_of_controllable_speakers():
    cat = await catalog()
    s = load_settings()
    proxies = {t: sp for sp, t in cat.music_twins.items()}

    def check(eid, controllable=cat.controllable):
        guard([ServiceCall("music_assistant", "play_media", (eid,), {"media_id": "x"})], cfg=s.safety,
              controllable=controllable, lookup=cat.lookup, table=cat.caps, proxies=proxies,
              proxy_services=cat.music_play_services())

    check(STEREO_TWIN)
    for bad in ("media_player.demo_stereo", "light.alder", "media_player.demo_soundbar_duplicate"):
        with pytest.raises(SafetyViolation):
            check(bad)
    with pytest.raises(SafetyViolation):
        check(STEREO_TWIN, controllable=cat.controllable - {"media_player.demo_stereo"})
    with pytest.raises(SafetyViolation):  # without the proxy grant the service isn't allowlisted at all
        guard([ServiceCall("music_assistant", "play_media", (STEREO_TWIN,), {})], cfg=s.safety,
              controllable=cat.controllable, table=cat.caps)
