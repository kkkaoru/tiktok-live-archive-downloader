"""Import observed Android replay metadata/media pairs without credentials."""

import json
import re
from collections.abc import Iterable
from urllib.parse import parse_qs, urlsplit

from replay.core import Candidate, ReplayError, Scope


def replay_candidates(lines: Iterable[str]) -> dict[str, Candidate]:
    """Pair a single observed room ID with subsequent HLS playback URLs.

    Only use logs captured while opening the authorized creator's replay cards.
    Multiple-room API calls are deliberately not guessed or associated.
    """
    current: str | None = None
    result: dict[str, Candidate] = {}
    for line in lines:
        if len(line) > 65536:
            continue
        try:
            data: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload: object = data.get("payload") if isinstance(data, dict) else None
        if not isinstance(payload, dict) or payload.get("type") != "url":
            continue
        url: object = payload.get("url")
        if not isinstance(url, str):
            continue
        try:
            parts = urlsplit(url)
            host = parts.hostname
            if host is None:
                continue
            Scope((host,)).check(url)
        except (ReplayError, ValueError):
            continue
        if host.endswith(".tiktokv.com") and parts.path == "/webcast/room/replay/info/":
            ids = parse_qs(parts.query).get("room_ids", [])
            current = ids[0] if len(ids) == 1 and re.fullmatch(r"[0-9]{1,32}", ids[0]) else None
        elif (
            current is not None and host.endswith(".tiktokcdn.com") and parts.path.endswith(".m3u8")
        ):
            result[current] = Candidate(url, {}, "hls", "api-advertised")
    return result
