"""Normalize Executor native-measurement envelopes without touching frozen parsers.

Newer Executor CLI versions wrap a native measurement as a JSON string inside
``data.content[0].text`` instead of a ``data.structuredContent`` object. This
module re-envelopes the identical inner payload into the structured shape so
the frozen shared parsers (``caption_io.executor_measurement`` and its
callers) validate the unchanged bytes, including the read-only source
contract. No payload field is added, removed or altered here.
"""

from __future__ import annotations

import json
from typing import Final

from replay.recordings import record

MAX_ENVELOPE_BYTES: Final[int] = 128 * 1024 * 1024


def normalize_envelope(value: object) -> object:
    """Return the structured-shape envelope holding the identical payload."""
    response = record(value)
    if response.get("ok") is not True:
        raise ValueError("Executor measurement did not succeed")
    data = record(response.get("data"))
    error = data.get("isError")
    if error is not None and error is not False:
        raise ValueError("Native measurement returned an error")
    if data.get("structuredContent") is not None:
        return value
    content = data.get("content")
    if not isinstance(content, list) or len(content) != 1:
        raise ValueError("Executor measurement has no single content envelope")
    item = record(content[0])
    if item.get("type") != "text":
        raise ValueError("Executor measurement content is not text")
    text = item.get("text")
    if not isinstance(text, str) or len(text) > MAX_ENVELOPE_BYTES:
        raise ValueError("Executor measurement text exceeds its budget")
    inner = record(json.loads(text))
    return {"ok": True, "data": {"structuredContent": inner}}
