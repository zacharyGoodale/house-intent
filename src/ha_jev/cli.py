"""`ha-jev ask` and `ha-jev eval` (spec §10)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from ha_jev.config import REPO_ROOT, ConfigError, load_dev_env, load_settings

EXIT_OK, EXIT_CONFIG, EXIT_SAFETY, EXIT_GOLDEN = 0, 2, 2, 3


def _load_json_file(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _overlay_arg(value: str) -> dict:
    p = Path(value)
    if not p.is_file():
        candidate = REPO_ROOT / "eval" / "overlays" / f"{value}.json"
        if candidate.is_file():
            p = candidate
    return _load_json_file(str(p))


def _real_allowed() -> bool:
    return os.environ.get("HA_JEV_ALLOW_REAL") == "1" and bool(os.environ.get("HA_URL")) and bool(os.environ.get("HA_TOKEN"))


async def _ask(args: argparse.Namespace) -> int:
    from ha_jev.core import Core
    from ha_jev.ha.stub import StubHA
    from ha_jev.jev import TypeSafeJevClient
    from ha_jev.hermes import make_fallback
    from ha_jev.memory import MemoryStore
    from ha_jev.obslog import RequestLog
    from ha_jev.profiles import speaker_for

    settings = load_settings(demo=args.ha == "stub")
    if args.ha == "real":
        if not _real_allowed():
            print("error: --ha real needs HA_JEV_ALLOW_REAL=1, HA_URL and HA_TOKEN", file=sys.stderr)
            return EXIT_CONFIG
        from ha_jev.ha.real import RealHA

        ha: Any = RealHA(os.environ["HA_URL"], os.environ["HA_TOKEN"], entity_id_map=settings.ha_real.entity_id_map)
        await ha.connect()
        clock = None
    else:
        overlays = [_overlay_arg(args.overlay)] if args.overlay else []
        initial = _load_json_file(args.state) if args.state else None
        ha = StubHA.default(overlays=overlays, initial_state=initial)
        settings = settings.with_scope_extension(**ha.overlay_scope)
        clock = (lambda: ha.now)
    jev = TypeSafeJevClient(settings.jev)
    luna = make_fallback(settings, args.fallback, args.no_llm)
    log = RequestLog(REPO_ROOT / settings.log.dir, settings.log.enabled)
    store = MemoryStore.from_settings(settings.memory) if args.ha == "real" or args.memory else MemoryStore()
    core = await Core.create(settings, ha, jev, luna, clock=clock, log=log, memory=store)
    speaker = speaker_for(settings.people, person_id=args.person, agent_id=args.agent)
    if (args.person or args.agent) and speaker is None:
        print(f"error: no person for --person {args.person!r} / --agent {args.agent!r} in people.toml", file=sys.stderr)
        return EXIT_CONFIG
    try:
        result = await core.handle(args.text, speaker_area=args.area, person=speaker.id if speaker else None)
    finally:
        await jev.aclose()
        if args.ha == "real":
            await ha.close()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return EXIT_OK
    print(f"outcome: {result.outcome}")
    print(f"response_mode: {result.response_mode}")
    print(f"reply: {result.reply}")
    for c in result.service_calls:
        print(c.describe())
    if result.clarify_options:
        print(f"clarify_options: {', '.join(result.clarify_options)}")
    d = result.debug
    jev_bits = ", ".join(f"{j.model} {j.input_tokens} tok {j.latency_ms:.0f} ms" + (f" ({j.error})" if j.error else "")
                         for j in d.jev) or "none"
    print(f"# route={d.route} rule={d.rule} confidence={d.confidence} jev=[{jev_bits}] "
          f"total={d.timings_ms.get('total', 0):.0f} ms cost=${d.cost_usd['total']:.6f}")
    return EXIT_OK


async def _tui(args: argparse.Namespace) -> int:
    from ha_jev.core import Core
    from ha_jev.ha.stub import StubHA
    from ha_jev.ha.interface import HAError
    from ha_jev.jev import TypeSafeJevClient
    from ha_jev.hermes import make_fallback
    from ha_jev.memory import MemoryStore
    from ha_jev.obslog import RequestLog
    from ha_jev.profiles import speaker_for
    from ha_jev.tui import run

    if args.ha == "real" and not _real_allowed():
        print("error: real HA needs HA_JEV_ALLOW_REAL=1, HA_URL and HA_TOKEN", file=sys.stderr)
        return EXIT_CONFIG
    settings = load_settings(demo=args.ha == "stub")
    speaker = speaker_for(settings.people, person_id=args.person, agent_id=args.agent)
    if (args.person or args.agent) and speaker is None:
        print(f"error: no person for --person {args.person!r} / --agent {args.agent!r} in people.toml", file=sys.stderr)
        return EXIT_CONFIG
    if args.ha == "real":
        from ha_jev.ha.real import RealHA

        ha: Any = RealHA(os.environ["HA_URL"], os.environ["HA_TOKEN"], entity_id_map=settings.ha_real.entity_id_map)
        clock = None
    else:
        ha = StubHA.default()
        clock = lambda: ha.now
    jev = TypeSafeJevClient(settings.jev)
    try:
        if args.ha == "real":
            await ha.connect()
        luna = make_fallback(settings, args.fallback, args.no_llm)
        log = RequestLog(REPO_ROOT / settings.log.dir, settings.log.enabled)
        store = MemoryStore.from_settings(settings.memory) if args.ha == "real" else MemoryStore()
        core = await Core.create(settings, ha, jev, luna, clock=clock, log=log, memory=store)
        await run(core, ha_name=args.ha, area=args.area, person=speaker.id if speaker else None,
                  fallback_name="disabled" if args.no_llm else args.fallback)
    except HAError as exc:
        print(f"Home Assistant error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    finally:
        await jev.aclose()
        if args.ha == "real":
            await ha.close()
    return EXIT_OK


async def _eval(args: argparse.Namespace) -> int:
    from ha_jev.evaluation.golden import GoldenError, load_golden, select
    from ha_jev.evaluation.overlay import validate_overlays
    from ha_jev.evaluation.report import build_report
    from ha_jev.evaluation.runner import run_eval, write_outputs
    from ha_jev.ha.stub import OverlayError
    from ha_jev.jev import TypeSafeJevClient
    from ha_jev.luna import LunaJudge, NullLuna, OpenAILuna

    settings = load_settings(demo=True)
    golden = Path(args.golden)
    overlay_dir = golden.parent / "overlays"
    try:
        cases = load_golden(golden)
        if overlay_dir.is_dir():
            validate_overlays(overlay_dir)
        for c in cases:
            for name in c.overlay:
                if not (overlay_dir / f"{name}.json").is_file():
                    raise GoldenError(f"{c.id}: overlay {name!r} not found in {overlay_dir}")
    except (GoldenError, OverlayError) as e:
        print(f"error: invalid golden or overlay: {e}", file=sys.stderr)
        return EXIT_GOLDEN
    selected = select(cases, tags=args.tag, ids=args.id, limit=args.limit)
    jev = TypeSafeJevClient(settings.jev)
    if args.no_llm:
        luna: Any = NullLuna()
        judge = None
    else:
        real_luna = OpenAILuna(settings.luna)
        luna, judge = real_luna, LunaJudge(real_luna)
    flags = {"no_llm": args.no_llm, "limit": args.limit, "tags": args.tag or [], "ids": args.id or [],
             "concurrency": args.concurrency, "golden": str(golden)}
    try:
        run = await run_eval(selected, jev=jev, luna=luna, no_llm=args.no_llm, overlay_dir=overlay_dir,
                             concurrency=args.concurrency, judge=judge, total_cases=len(cases), flags=flags)
    finally:
        await jev.aclose()
    report = build_report(run, settings.prices)
    out = write_outputs(run, Path(args.out), report)
    print(report)
    print(f"\nwritten to {out}")
    return EXIT_SAFETY if run.safety_failed_ids else EXIT_OK


def _memory(args: argparse.Namespace) -> int:
    """`ha-jev memory list|forget`: review and delete remembered facts (docs/usage.md)."""
    from ha_jev.memory import HOUSEHOLD, MemoryStore

    store = MemoryStore.from_settings(load_settings().memory)
    if args.action == "forget":
        if not args.id or store.remove(args.id) is None:
            print(f"error: no fact {args.id!r} in {store.path}", file=sys.stderr)
            return EXIT_CONFIG
        print(f"forgot {args.id}")
        return EXIT_OK
    facts = store.visible(args.person) if args.person else store.all()
    for f in facts:
        print(f"{f.id}  {f.owner if f.owner != HOUSEHOLD else 'household':<10} {f.created}  {f.text}")
    print(f"# {len(facts)} fact(s) in {store.path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ha-jev", description="Voice command core for Home Assistant, routed by Jev")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask", help="handle one utterance")
    a.add_argument("text")
    a.add_argument("--area", help="speaker area id (or name)")
    a.add_argument("--person", help="speaker's person id from people.toml")
    a.add_argument("--agent", help="HA conversation agent id; mapped to a person by people.toml `agents`")
    a.add_argument("--memory", action="store_true", help="use the real memory store (default: in-memory with the stub)")
    a.add_argument("--json", action="store_true", help="print Result.to_dict()")
    a.add_argument("--no-llm", action="store_true", help="disable open-ended answers")
    a.add_argument("--fallback", choices=("hermes", "luna"), default="hermes",
                   help="answer backend (default: Hermes with automatic Luna failover)")
    a.add_argument("--state", help="JSON file with initial_state to merge over the stub")
    a.add_argument("--overlay", help="overlay JSON file, or the name of one in eval/overlays")
    a.add_argument("--ha", choices=("stub", "real"), default="stub")
    t = sub.add_parser("tui", help="interactive text commands (real Home Assistant by default)")
    t.add_argument("--ha", choices=("stub", "real"), default="real")
    t.add_argument("--area", help="speaker area id (or name)")
    t.add_argument("--person", help="speaker's person id from people.toml")
    t.add_argument("--agent", help="HA conversation agent id; mapped to a person by people.toml `agents`")
    t.add_argument("--no-llm", action="store_true", help="disable open-ended answers")
    t.add_argument("--fallback", choices=("hermes", "luna"), default="hermes",
                   help="answer backend (default: Hermes with automatic Luna failover)")
    e = sub.add_parser("eval", help="run the golden set")
    e.add_argument("--no-llm", action="store_true")
    e.add_argument("--limit", type=int)
    e.add_argument("--tag", action="append")
    e.add_argument("--id", action="append")
    e.add_argument("--golden", default=str(REPO_ROOT / "eval" / "golden.jsonl"))
    e.add_argument("--out", default=str(REPO_ROOT / "eval" / "runs"))
    e.add_argument("--concurrency", type=int, default=4)
    m = sub.add_parser("memory", help="list or forget remembered facts")
    m.add_argument("action", choices=("list", "forget"))
    m.add_argument("id", nargs="?", help="fact id to forget")
    m.add_argument("--person", help="list what this person's requests see (household + own)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dev_env()
    try:
        if args.command == "ask":
            return asyncio.run(_ask(args))
        if args.command == "tui":
            return asyncio.run(_tui(args))
        if args.command == "memory":
            return _memory(args)
        return asyncio.run(_eval(args))
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
