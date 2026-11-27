"""Free-form music: "play some jazz", "play the album Blonde in the study", "play WNYC" (docs/usage.md).

Jev can't write free text, so code proposes the words to search for and Jev picks among them in the same single
request (decision 1): `parse` cuts candidate spans from the utterance, `questions` asks `play_span` and
`content_type` only when there are spans (no tokens spent on other utterances), and `decide` (pure, called from
policy) resolves the room to one speaker and plans a search. Core then runs the search as a read, and `resolve`
picks a result in code and plays its URI on the speaker's Music Assistant twin. No model ever searches or plays.

Everything house-specific comes from the registry and `[music]` in scope.toml (decision 9): the integration name,
the twin pairing (config map, then registry unique_id, then the player's name) and the rooms' default players.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ha_jev import capabilities as caps
from ha_jev.jev import ChoiceAnswer
from ha_jev.models import ClarifySpec, Decision, Outcome, PlanItem, ReadCall, ServiceCall

if TYPE_CHECKING:
    from ha_jev.catalog import Catalog
    from ha_jev.config import MusicConfig, PolicySettings
    from ha_jev.ha.interface import Registry
    from ha_jev.questions import AnswerView

PLAY_KEY = "M_PLAY"  # the planned decision, until core has searched
SPAN_QUESTION = "play_span"
TYPE_QUESTION = "content_type"
CONTENT_TYPES = ("artist", "album", "track", "playlist", "my_playlist", "genre", "radio", "podcast", "unknown")
MAX_SPANS = 4

# Search media types per content type (music.md §3.2). Genres search playlists; unknown searches everything.
_SEARCH_TYPES = {"artist": ["artist"], "album": ["album"], "track": ["track"], "playlist": ["playlist"],
                 "genre": ["playlist"], "radio": ["radio"], "podcast": ["podcast"]}
_RESULT_KEYS = {"artist": "artists", "album": "albums", "track": "tracks", "playlist": "playlists",
                "genre": "playlists", "radio": "radio", "podcast": "podcasts", "my_playlist": "items"}
# Unknown type: an exact name wins in this order (music.md §3.3; deliberately not MA's tracks-first default).
_UNKNOWN_ORDER = ("artists", "albums", "playlists", "radio", "podcasts", "tracks")
_RADIO_MODE = {"artist", "genre"}  # keep playing similar music after the first tracks


# ---- twins (catalog hook) ----------------------------------------------------------------------
def pair_players(registry: Registry, cfg: MusicConfig, table: caps.CapabilityTable) -> dict[str, str]:
    """Speaker entity -> the music integration's player for the same speaker: `[music] players` first, then the
    same registry unique_id (the music server reuses the speaker's own id), then the same name. A player that pairs with
    nothing (a web player, say) stays an ordinary media player."""
    if not cfg.integration:
        return {}
    media = set(table.with_role(caps.MUSIC_PLAY_ROLE))
    players = {eid: e for eid, e in registry.entities.items() if e.domain in media and e.platform == cfg.integration}
    speakers = {eid: e for eid, e in registry.entities.items() if e.domain in media and e.platform != cfg.integration}
    out = {s: t for s, t in cfg.players.items() if t in players and s in speakers}
    taken = set(out.values())
    for sid, s in sorted(speakers.items()):
        if sid in out:
            continue
        same_id = [p for p, e in players.items() if p not in taken and s.unique_id and e.unique_id == s.unique_id]
        same_name = [p for p, e in players.items() if p not in taken and _norm(e.name) == _norm(s.name)]
        match = same_id or (same_name if len(same_name) == 1 else [])
        if match:
            out[sid] = match[0]
            taken.add(match[0])
    return out


# ---- extraction (before Jev) -------------------------------------------------------------------
_PLAY = re.compile(
    r"^(?:(?:can|could|would|will) you\s+|please\s+|go ahead and\s+|let'?s\s+|just\s+|now\s+"
    r"|i(?: want| wanna|'d like| would like)(?: to)?\s+)*"
    r"(?:play|put on|listen to|hear|shuffle|start playing|queue up|throw on|blast)\s+(?:me\s+|us\s+)?(?P<rest>.+)$",
    re.IGNORECASE)
_GENERIC = {"music", "some music", "something", "anything", "a song", "songs", "some songs", "tunes", "some tunes",
            "something good", "something nice", "some", "the music", "my music", "it", "that", "this", "again",
            "the radio", "radio", "something else", "a playlist", "a podcast", "some podcasts", "playlist",
            "a station", "station", "podcast", "album", "song", "the song", "the album", "the playlist"}
_EVERYWHERE = re.compile(r"\s+(?:everywhere|in every room|in all (?:the )?rooms|all over the (?:house|home)"
                         r"|throughout the (?:house|home)|in the whole (?:house|home))$", re.IGNORECASE)
_HERE = re.compile(r"\s+(?:in here|here|in this room)$", re.IGNORECASE)
_SOME = re.compile(r"^(?:some|a little|a bit of|some more|more|a few)\s+(?P<x>.+)$", re.IGNORECASE)
_MUSIC_SUFFIX = re.compile(r"^(?P<x>.+?)\s+(?:music|songs|tunes|tracks)$", re.IGNORECASE)
_BY = re.compile(r"^(?P<name>.+?)\s+by\s+(?P<artist>.+)$", re.IGNORECASE)
_TYPED = (  # explicit type words: (pattern, hint); `x` is the name
    (re.compile(r"^(?:my|our)\s+(?:playlist|mix)\s+(?:called\s+|named\s+)?(?P<x>.+)$", re.I), "my_playlist"),
    (re.compile(r"^(?:my|our)\s+(?P<x>.+?)\s+(?:playlist|mix)$", re.I), "my_playlist"),
    (re.compile(r"^(?:my|our)\s+(?P<x>.+)$", re.I), "my_playlist"),
    (re.compile(r"^(?:the\s+|that\s+)?(?:album|record|lp|ep)\s+(?:called\s+|named\s+)?(?P<x>.+)$", re.I), "album"),
    (re.compile(r"^(?:the\s+)?(?P<x>.+?)\s+album$", re.I), "album"),
    (re.compile(r"^(?:the\s+|that\s+)?(?:song|track|single|tune)\s+(?:called\s+|named\s+)?(?P<x>.+)$", re.I), "track"),
    (re.compile(r"^(?:the\s+)?(?:artist|band|singer|group)\s+(?:called\s+|named\s+)?(?P<x>.+)$", re.I), "artist"),
    (re.compile(r"^(?:the\s+|a\s+)?playlist\s+(?:called\s+|named\s+)?(?P<x>.+)$", re.I), "playlist"),
    (re.compile(r"^(?:the\s+)?(?P<x>.+?)\s+playlist$", re.I), "playlist"),
    (re.compile(r"^(?:the\s+)?(?:radio\s+)?station\s+(?:called\s+|named\s+)?(?P<x>.+)$", re.I), "radio"),
    (re.compile(r"^(?:the\s+)?(?P<x>.+?)\s+(?:radio\s+)?station$", re.I), "radio"),
    (re.compile(r"^(?:the\s+)?podcast\s+(?:called\s+|named\s+)?(?P<x>.+)$", re.I), "podcast"),
    (re.compile(r"^(?:the\s+)?(?P<x>.+?)\s+podcast$", re.I), "podcast"),
)


@dataclass(frozen=True)
class Span:
    text: str
    hint: str | None = None  # a content type the words said ("the album ...")
    artist: str | None = None  # "... by <artist>"

    def label(self) -> str:
        return f"'{self.text}' by '{self.artist}'" if self.artist else f"'{self.text}'"


@dataclass(frozen=True)
class PlayRequest:
    spans: tuple[Span, ...]
    everywhere: bool = False  # "... everywhere": more than one room


def _norm(s: str | None) -> str:
    """Casefold, strip diacritics, '&' -> 'and', drop punctuation and a leading 'the' (music.md §3.3)."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold().replace("&", " and ")
    s = re.sub(r"[^\w\s+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^the ", "", s)


def _places(cat: Catalog) -> list[str]:
    """Words that say where to play: room names and aliases, and the speakers' names and aliases."""
    out: set[str] = set()
    for a in cat.areas.values():
        out |= {a.name, *a.aliases}
    for s in cat.music_speakers:
        e = cat.entities[s]
        out |= {e.name, *e.aliases}
    return sorted((p for p in out if p.strip()), key=len, reverse=True)


def _cut_place(rest: str, places: Sequence[str]) -> str | None:
    """`rest` without a trailing place phrase ("... in the study", "... on the <speaker>"), else None."""
    low = rest.casefold()
    for p in places:
        pl = re.escape(p.casefold())
        m = re.search(rf"\s+(?:in|on|through|over|from)\s+(?:the\s+|my\s+)?{pl}(?:\s+(?:speakers?|room))?$", low)
        if m:
            return rest[: m.start()].strip()
    m = re.search(r"\s+in\s+the\s+[a-z]+(?:\s+room)?$", low)  # a room this home may not have ("in the attic")
    if m:
        return rest[: m.start()].strip()
    return None


def _typed(text: str) -> Span:
    text = text.strip(" ,.!?")
    for pattern, hint in _TYPED:
        m = pattern.match(text)
        if m and m.group("x").strip():
            x = m.group("x").strip()
            by = _BY.match(x) if hint in ("album", "track") else None
            if by:
                return Span(by.group("name").strip(), hint, by.group("artist").strip())
            return Span(x, hint)
    m = _SOME.match(text)
    if m:
        x = m.group("x").strip()
        suffix = _MUSIC_SUFFIX.match(x)
        return Span(suffix.group("x").strip() if suffix else x, None)
    by = _BY.match(text)
    if by:
        return Span(by.group("name").strip(), None, by.group("artist").strip())
    return Span(text, None)


def _known_names(cat: Catalog) -> set[str]:
    """Names that mean a device, scene, app or room, not music: "play the <speaker>" resumes it, "put on <app>"
    opens it, "play <script alias>" runs it. Those utterances keep their existing paths."""
    names: set[str] = set()
    for eid, e in cat.entities.items():
        names |= {_norm(e.name), *(_norm(a) for a in e.aliases)}
    for key in cat.scene_options:
        names.add(_norm(cat.name(key)))
        if key in cat.presets:
            names |= {_norm(a) for a in cat.presets[key].aliases}
    names |= {_norm(s) for s in cat.sources}
    for a in cat.areas.values():
        names |= {_norm(a.name), *(_norm(x) for x in a.aliases)}
    return {n for n in names if n}


def parse(text: str, cat: Catalog) -> PlayRequest | None:
    """Candidate spans for a play request, or None when the utterance isn't one (or names only a device, scene,
    app or room, or no music at all: "play music" stays on the transport path)."""
    m = _PLAY.match(text.strip())
    if not m:
        return None
    rest = m.group("rest").strip(" ,.!?")
    everywhere = bool(_EVERYWHERE.search(rest))
    rest = _EVERYWHERE.sub("", rest)
    rest = _HERE.sub("", rest).strip()
    cut = _cut_place(rest, _places(cat))
    main = cut or rest
    # The typed reading of the words without the place, the same words as said, and (when a place was cut) the
    # whole rest, in case the "place" is part of a title ("Midnight in the Garden").
    candidates = [_typed(main), Span(main.strip(" ,.!?"))] + ([Span(rest.strip(" ,.!?"))] if cut else [])
    spans: list[Span] = []
    known = _known_names(cat)
    for s in candidates:
        n = _norm(s.text)
        if not n or n in _GENERIC or n in known or s.text.casefold() in _GENERIC:
            continue
        if all(_norm(o.text) != n or o.artist != s.artist for o in spans):
            spans.append(s)
    spans = spans[:MAX_SPANS]
    return PlayRequest(tuple(spans), everywhere) if spans else None


def names_scene(cat: Catalog, key: str, text: str) -> bool:
    """The utterance says a scene's name or one of its aliases in full: "play <that podcast script>" runs the
    script, while "play <another show>" is a search even when the scene Choice likes a similar script."""
    said = set(_norm(text).split())
    names = [cat.name(key), *(cat.presets[key].aliases if key in cat.presets
                              else cat.entities[key].aliases if key in cat.entities else ())]
    return any((words := set(_norm(n).split())) and words <= said for n in names)


# ---- questions (same single request) -----------------------------------------------------------
def questions(u: str, req: PlayRequest | None) -> dict[str, dict]:
    """`play_span` and `content_type`, only when the utterance has play spans (decision 1: still one request)."""
    if req is None or not req.spans:
        return {}
    spans = {f"s{i}": s.label() for i, s in enumerate(req.spans)}
    spans["none"] = "no specific music is named; or the request is to resume, or names a device, scene or app"
    return {
        SPAN_QUESTION: {"type": "choice", "instructions": {
            "question": f"Which listed text in {u} is the name of the music, station, playlist or podcast to play?",
            "not_for": "words that only say where to play it (a room or a speaker), or filler around the name "
                       "such as 'some', 'my', 'the album', 'playlist' or 'music'; prefer the listed text without "
                       "them unless they are part of the name"},
            "criteria": spans},
        TYPE_QUESTION: {"type": "choice", "instructions": f"What kind of thing does {u} ask to play?", "criteria": {
            "artist": "a singer, band or composer", "album": "an album or record", "track": "a single song",
            "playlist": "a named public or editorial playlist",
            "my_playlist": "the user's own or saved playlist or mix, such as 'my ...', liked songs or a weekly "
                           "personal mix like Discover Weekly or Daily Mix",
            "genre": "a genre, mood or activity, such as jazz, something relaxing or dinner music",
            "radio": "a radio station or call sign", "podcast": "a podcast or show",
            "unknown": "can't tell"}},
    }


# ---- policy (pure) -----------------------------------------------------------------------------
def _p(a: ChoiceAnswer) -> float:
    return float(a.probabilities.get(a.choice, 0.0)) if a.probabilities else float(a.confidence)


def _decision(outcome: Outcome, key: str, route: str, conf: float | None = None, **params: Any) -> Decision:
    return Decision(outcome=outcome, reply_key=key, reply_params=params, route=route, rule="8", confidence=conf)


def _room_player(cat: Catalog, area: str) -> str | None:
    """A room's music speaker: `defaults.transport_player` for the room, else its only speaker, else the first."""
    default = (cat.defaults.transport_player or {}).get(area)
    if default in cat.music_speakers:
        return default
    here = sorted(s for s in cat.music_speakers if cat.entities[s].area_id == area)
    return here[0] if here else None


def decide(cat: Catalog, text: str, av: AnswerView, pol: PolicySettings, speaker_area: str | None,
           intent_p: float) -> Decision | None:
    """The play plan, or None when this isn't a free-form music request (the existing paths then decide: a named
    scene or script such as a podcast script, resuming a player, or R_PLAY when music isn't set up)."""
    req = parse(text, cat)
    if req is None or not av.has(SPAN_QUESTION):
        return None
    sc = av.choice("scene") if av.has("scene") else None
    if sc is not None and sc.choice in cat.scene_options and _p(sc) >= pol.act and names_scene(cat, sc.choice, text):
        return None  # "play <podcast script>": the scene path runs it (music.md §3.1)
    span_a = av.choice(SPAN_QUESTION)
    idx = int(span_a.choice[1:]) if re.fullmatch(r"s\d+", span_a.choice) else None
    if idx is None or idx >= len(req.spans) or _p(span_a) < pol.clarify_floor:
        return None
    span = req.spans[idx]
    typed = _typed(span.text)
    if span.hint is None and span.artist is None and (typed.hint or typed.artist):
        span = typed  # the words as said, with type words or "by <artist>" in them: search their typed reading
    ctype = span.hint
    if ctype is None and av.has(TYPE_QUESTION):
        ta = av.choice(TYPE_QUESTION)
        # An unsure type ("play <a name that is an album and a song>") searches everything and prefers exact
        # names in a fixed order (music.md §3.3) rather than trusting a coin flip.
        ctype = ta.choice if ta.choice in CONTENT_TYPES and _p(ta) >= pol.act else "unknown"
    ctype = ctype or "unknown"
    conf = min(intent_p, _p(span_a))
    route = f"music/{ctype}"
    if not cat.music_integration:
        return _decision(Outcome.UNSUPPORTED, "R_PLAY", route, conf)

    # Where: a named speaker, else the named room, else the speaker's room (music.md §3.1, OQ 105).
    areas = [a for a in cat.areas if av.has(f"area__{a}") and av.noul(f"area__{a}") >= pol.noul_yes]
    named = None
    if av.has("target"):
        t = av.choice("target")
        if t.choice in cat.music_speakers and _p(t) >= pol.act:
            named = t.choice
    if req.everywhere or len(areas) > 1:
        return _decision(Outcome.UNSUPPORTED, "R_PLAY_ONE_ROOM", route, conf)
    if named is not None:
        player = named
    elif areas:
        player = _room_player(cat, areas[0])
        if player is None:
            return _decision(Outcome.UNSUPPORTED, "R_NO_SPEAKER", route, conf, area=cat.areas[areas[0]].name)
    elif av.has("unknown_room") and av.noul("unknown_room") >= pol.unknown_room:
        return _decision(Outcome.UNSUPPORTED, "R_NO_DEVICE", route, conf)
    elif speaker_area in cat.areas:
        player = _room_player(cat, speaker_area)
        if player is None:
            return _decision(Outcome.UNSUPPORTED, "R_NO_SPEAKER", route, conf, area=cat.areas[speaker_area].name)
    else:
        rooms = [cat.areas[a].name for a in sorted(cat.areas) if _room_player(cat, a)]
        return Decision(outcome=Outcome.CLARIFY, reply_key="R_WHICH_ROOM", route=route, rule="8", confidence=conf,
                        reply_params={"options": " or ".join(rooms) if len(rooms) <= pol.clarify_max_options else ""})
    e = cat.entities[player]
    twin = cat.music_twins.get(player)
    if e.denied or not e.controllable or twin is None:
        return _decision(Outcome.UNSUPPORTED, "R_PLAY", route, conf)

    domain = cat.music_integration
    if ctype == "my_playlist":
        read = ReadCall("service", domain, caps.MUSIC_LIBRARY, (),
                        {"media_type": "playlist", "search": span.text, "limit": _limit(cat)})
    else:
        data: dict[str, Any] = {"name": span.text, "limit": _limit(cat)}
        if ctype in _SEARCH_TYPES:
            data["media_type"] = list(_SEARCH_TYPES[ctype])
        if span.artist:
            data["artist"] = span.artist
        read = ReadCall("service", domain, caps.MUSIC_SEARCH, (), data)
    params = {"span": span.text, "artist": span.artist, "type": ctype, "player": player, "twin": twin,
              "integration": domain, "name": e.name}
    return Decision(outcome=Outcome.EXECUTED, reads=(read,), reply_key=PLAY_KEY, reply_params=params, route=route,
                    rule="8", confidence=conf, targets=(player,))


def _limit(cat: Catalog) -> int:
    return cat.music_limit


# ---- core hooks --------------------------------------------------------------------------------
def is_play(decision: Decision) -> bool:
    return decision.reply_key == PLAY_KEY


def prepare(decision: Decision, config_entry_id: str | None) -> Decision:
    """Add the integration's config entry to the search read (known only to core), or say music isn't set up."""
    if config_entry_id is None:
        return Decision(outcome=Outcome.UNSUPPORTED, reply_key="R_PLAY", route=decision.route, rule="8",
                        confidence=decision.confidence)
    reads = tuple(ReadCall(r.kind, r.domain, r.service, r.entity_ids, {"config_entry_id": config_entry_id, **r.data})
                  for r in decision.reads)
    return Decision(outcome=decision.outcome, reads=reads, reply_key=decision.reply_key,
                    reply_params=decision.reply_params, route=decision.route, rule=decision.rule,
                    confidence=decision.confidence, targets=decision.targets)


def _artists(item: Mapping[str, Any]) -> list[str]:
    return [str(a.get("name")) for a in (item.get("artists") or []) if isinstance(a, Mapping) and a.get("name")]


def _artist_ok(item: Mapping[str, Any], artist: str | None) -> bool:
    if not artist:
        return True
    want = _norm(artist)
    return any(want in _norm(a) or _norm(a) in want for a in _artists(item))


def _items(result: Any, key: str) -> list[Mapping[str, Any]]:
    rows = result.get(key) if isinstance(result, Mapping) else None
    return [r for r in (rows or []) if isinstance(r, Mapping) and r.get("uri") and r.get("name")]


def choose(result: Any, span: str, ctype: str, artist: str | None) -> tuple[list[Mapping[str, Any]], bool]:
    """Pick in code (music.md §3.3). Returns (items, exact): one item to play, several exact same-name albums by
    different artists to ask about, or none. `exact` means an exact name match on a requested type."""
    want = _norm(span)
    if ctype in _RESULT_KEYS:
        items = [i for i in _items(result, _RESULT_KEYS[ctype]) if _artist_ok(i, artist)]
        exact = [i for i in items if _norm(str(i["name"])) == want]
        if ctype == "album" and not artist:
            by = {}
            for i in exact:
                by.setdefault(tuple(_artists(i)), i)
            if len(by) >= 2:
                return list(by.values()), True
        if exact:
            return [exact[0]], True
        return (items[:1], False) if items else ([], False)
    for key in _UNKNOWN_ORDER:
        exact = [i for i in _items(result, key) if _norm(str(i["name"])) == want and _artist_ok(i, artist)]
        if exact:
            return [exact[0]], False
    artists = _items(result, "artists")
    if artists and want and want in _norm(str(artists[0]["name"])):
        return [artists[0]], False
    for key in ("tracks", "albums", "playlists", "radio", "podcasts"):
        items = [i for i in _items(result, key) if _artist_ok(i, artist)]
        if items:
            return [items[0]], False
    return [], False


def _what(item: Mapping[str, Any]) -> str:
    names = _artists(item)
    kind = str(item.get("media_type") or "")
    if names and kind in ("album", "track"):
        return f"{item['name']} by {names[0]}"
    return str(item["name"])


def _play_call(params: Mapping[str, Any], item: Mapping[str, Any]) -> ServiceCall:
    kind = str(item.get("media_type") or "")
    data = {"media_id": item["uri"], "media_type": kind, "enqueue": "replace",
            "radio_mode": params["type"] in _RADIO_MODE}
    return ServiceCall(params["integration"], caps.MUSIC_PLAY, (params["twin"],), data)


def resolve(decision: Decision, read_results: Mapping[str, Any], max_options: int = 3) -> Decision:
    """After the search: play the chosen item on the speaker's twin, ask between same-name albums, or say what
    wasn't found. The reply is a chime on an exact match of the requested type, else spoken (OQ 104)."""
    p = decision.reply_params
    read = decision.reads[0]
    result = read_results.get(f"{read.domain}.{read.service}")
    items, exact = choose(result, p["span"], p["type"], p.get("artist"))
    base = {"route": decision.route, "rule": decision.rule, "confidence": decision.confidence}
    if not items:
        return Decision(outcome=Outcome.UNSUPPORTED, reply_key="R_PLAY_NOT_FOUND", reply_params={"what": p["span"]},
                        **base)
    if len(items) > 1:
        offered = items[:max_options]
        options = {str(i["uri"]): _what(i) for i in offered}
        plan = {str(i["uri"]): PlanItem((_play_call(p, i),), "R_PLAYING", {"what": _what(i)}, (p["player"],))
                for i in offered}
        spoken = " or ".join(options.values())
        spec = ClarifySpec("pick", f"Which one: {spoken}?", options, plan)
        return Decision(outcome=Outcome.CLARIFY, reply_key="R_WHICH_MUSIC", reply_params={"options": spoken},
                        clarify=spec, targets=(p["player"],), **base)
    item = items[0]
    return Decision(outcome=Outcome.EXECUTED, calls=(_play_call(p, item),),
                    reply_key="R_PLAYING" if exact else "R_PLAYING_SAY", reply_params={"what": _what(item)},
                    targets=(p["player"],), **base)
