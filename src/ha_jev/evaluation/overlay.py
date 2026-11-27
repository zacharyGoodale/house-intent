"""Overlay loading for unseen_entity cases (spec §11.2). Merging happens in StubHA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ha_jev.ha.stub import OverlayError, StubHA


def load_overlays(names: list[str], directory: str | Path) -> list[dict[str, Any]]:
    out = []
    for name in names:
        path = Path(directory) / f"{name}.json"
        if not path.is_file():
            raise OverlayError(f"overlay not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise OverlayError(f"overlay {name}: invalid JSON: {e}") from e
        out.append(data)
    return out


def validate_overlays(directory: str | Path) -> list[str]:
    """Load and apply every overlay once on its own; raises OverlayError on the first invalid one."""
    names = sorted(p.stem for p in Path(directory).glob("*.json"))
    for name in names:
        StubHA.default(overlays=load_overlays([name], directory))
    return names
