"""Real Luna (spec §14): at most two tests. Opt in with `uv run pytest -m live_llm`."""

import os
from datetime import datetime

import pytest

from ha_jev.config import load_settings
from ha_jev.luna import LunaError, OpenAILuna

pytestmark = [pytest.mark.live_llm,
              pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")]
NOW = datetime.fromisoformat("2037-09-22T18:00:00-04:00")


async def test_general_question():
    luna = OpenAILuna(load_settings().luna)
    reply = await luna.answer("What is the capital of France?", now=NOW, speaker_area="Living Room",
                              previous_response_id=None)
    assert "paris" in reply.text.lower()
    assert reply.response_id and reply.input_tokens > 0 and reply.output_tokens > 0
    assert len(reply.text.split()) <= 45


async def test_tiny_timeout_is_a_luna_error():
    from dataclasses import replace

    settings = replace(load_settings().luna, timeout_s=0.001, max_retries=0)
    with pytest.raises(LunaError):
        await OpenAILuna(settings).answer("hi", now=NOW, speaker_area=None, previous_response_id=None)
