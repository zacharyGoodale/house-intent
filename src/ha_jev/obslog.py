"""Per-request JSONL log (spec §13.3). Keys are never logged; Authorization headers and bearer tokens are redacted."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_ENV = ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "HA_TOKEN", "HERMES_API_KEY")
_AUTH = re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(bearer\s+)?[^\s\"',}]+")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/=]+")
_ACCESS_TOKEN = re.compile(r"(?i)(\"access_token\"\s*:\s*\")[^\"]*")
REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    for name in SECRET_ENV:
        value = os.environ.get(name)
        if value and len(value) >= 6:
            text = text.replace(value, REDACTED)
    text = _AUTH.sub(lambda m: m.group(1) + REDACTED, text)
    text = _BEARER.sub("Bearer " + REDACTED, text)
    text = _ACCESS_TOKEN.sub(lambda m: m.group(1) + REDACTED, text)
    return text


def redact_obj(obj: Any) -> Any:
    return json.loads(redact(json.dumps(obj, ensure_ascii=False, default=str)))


class RequestLog:
    def __init__(self, directory: str | Path, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled

    def write(self, record: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.directory / f"requests-{day}.jsonl"
        line = redact(json.dumps(record, ensure_ascii=False, default=str))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path
