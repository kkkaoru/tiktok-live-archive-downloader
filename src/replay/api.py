"""Bounded extraction of media URLs advertised by an authorized replay API."""

import json
import zlib
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from replay.core import Candidate, ReplayError, Scope, media_kind, save_candidate

MAX_API_BYTES = 2 * 1024 * 1024
MAX_API_NODES = 50000


def advertised_media(data: object) -> list[Candidate]:
    pending = [data]
    found: dict[str, Candidate] = {}
    visited = 0
    while pending and visited < MAX_API_NODES:
        value = pending.pop()
        visited += 1
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and len(value) <= MAX_API_BYTES:
            try:
                if value.startswith(("{", "[")):
                    pending.append(json.loads(value))
                    continue
                parts = urlsplit(value)
                if parts.scheme and parts.query:
                    pending.extend(pair[1] for pair in parse_qsl(parts.query, max_num_fields=1000))
                host = parts.hostname
                if host is None:
                    continue
                native_link = parts.scheme in ("sslocal", "snssdk1180", "snssdk1233")
                replay_link = any(word in value.lower() for word in ("replay", "record"))
                if not native_link:
                    Scope((host,)).check(value)
                kind = media_kind(value, "")
                if native_link or (kind is None and replay_link):
                    kind = "replay-link" if replay_link else None
                if kind is not None:
                    candidate = Candidate(value, {}, kind, "api-advertised")
                    found[candidate.id] = candidate
            except (ValueError, ReplayError, RecursionError):
                continue
    return list(found.values())


def api_stream(
    *, root: Path, encoding: str, report: Callable[[str], None]
) -> Callable[[bytes], bytes]:
    buffer = bytearray()
    overflow = False

    def stream(chunk: bytes) -> bytes:
        nonlocal overflow
        if chunk:
            if len(buffer) + len(chunk) > MAX_API_BYTES:
                overflow = True
                buffer.clear()
            if not overflow:
                buffer.extend(chunk)
            return chunk
        if overflow:
            report("Replay API body exceeded inspection limit; not stored.")
            return chunk
        try:
            body = bytes(buffer)
            buffer.clear()
            if encoding in ("gzip", "deflate"):
                decoder = zlib.decompressobj(31 if encoding == "gzip" else 15)
                body = decoder.decompress(body, MAX_API_BYTES + 1)
                if len(body) > MAX_API_BYTES or not decoder.eof:
                    report("Replay API decompression limit reached; not stored.")
                    return chunk
            elif encoding not in ("", "identity"):
                report("Replay API encoding unsupported; not stored.")
                return chunk
            data: object = json.loads(body)
        except (ValueError, zlib.error, RecursionError):
            report("Replay API was not readable JSON; not stored.")
            return chunk
        candidates = advertised_media(data)
        report(f"Replay API advertised {len(candidates)} recognizable media URLs or replay links.")
        for candidate in candidates:
            if save_candidate(root, candidate):
                report(f"Replay candidate: {candidate.label}")
        return chunk

    return stream
