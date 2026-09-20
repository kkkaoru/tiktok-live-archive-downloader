"""Validated recording metadata from authorized TikTok notifications."""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TypeGuard
from urllib.parse import parse_qs, urlsplit

from replay.core import ReplayError, Scope

JST = timezone(timedelta(hours=9), name="JST")


@dataclass(frozen=True, kw_only=True)
class Notice:
    replay_id: str
    anchor_id: str
    user_type: str


@dataclass(frozen=True, kw_only=True)
class Recording:
    replay_id: str
    title: str
    available: bool
    duration: float | None = None
    media_url: str | None = field(default=None, repr=False)
    start_time: int | None = None


class UnavailableReplay(ReplayError):
    """An observed recording is not currently downloadable."""


def is_record(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def record(value: object) -> dict[str, object]:
    if not is_record(value):
        raise ReplayError("Malformed TikTok API object.")
    return value


def numeric_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{1,32}", value) is None:
        raise ReplayError("Invalid numeric recording or creator ID.")
    return value


def broadcast_datetime(value: object) -> datetime:
    """Validate API start_time as Unix seconds, independent of the machine timezone."""
    if type(value) is not int or value <= 0:
        raise ReplayError("Missing or invalid broadcast start time; refusing a guessed filename.")
    try:
        return datetime.fromtimestamp(value, tz=JST)
    except (OverflowError, OSError, ValueError) as error:
        raise ReplayError("Broadcast start time is outside the supported date range.") from error


def recording_filename(recording: Recording) -> str:
    replay_id = numeric_id(recording.replay_id)
    start = broadcast_datetime(recording.start_time)
    return f"{start:%Y-%m-%d_%H-%M-%S}_JST_replay-{replay_id}.mp4"


def notice_from_json(value: object) -> Notice | None:
    template = record(value).get("template_notice")
    if not is_record(template):
        return None
    schema = template.get("schema_url")
    if not isinstance(schema, str):
        return None
    try:
        url = urlsplit(schema)
    except ValueError as error:
        raise ReplayError("Malformed notification link.") from error
    if (
        url.scheme not in {"sslocal", "snssdk1233", "snssdk1180"}
        or url.netloc != "webcast_replay_video"
    ):
        return None
    params = parse_qs(url.query)
    if any(len(params.get(key, [])) != 1 for key in ("roomId", "anchor_id", "user_type")):
        raise ReplayError("Incomplete or ambiguous replay notification link.")
    if params["user_type"][0] not in {"0", "1"}:
        raise ReplayError("Unsupported replay notification role.")
    return Notice(
        replay_id=numeric_id(params["roomId"][0]),
        anchor_id=numeric_id(params["anchor_id"][0]),
        user_type=params["user_type"][0],
    )


def recording_from_json(value: object, *, replay_id: str) -> Recording:
    items = record(value).get("replays")
    if not isinstance(items, list):
        raise ReplayError("Replay API omitted its replay list; this is not an empty catalog.")
    if not items:
        return Recording(replay_id=replay_id, title="Unavailable", available=False)
    if len(items) != 1:
        raise ReplayError("Replay API returned an ambiguous result.")
    item = record(items[0])
    if item.get("id") != replay_id:
        raise ReplayError("Replay API returned a different recording ID.")
    title = item.get("title")
    available = item.get("available")
    if not isinstance(title, str) or not isinstance(available, bool):
        raise ReplayError("Malformed replay availability or title.")
    if not available:
        return Recording(replay_id=replay_id, title=title, available=False)
    url = item.get("m3u8_url")
    if not isinstance(url, str) or not url:
        raise ReplayError("Available replay omitted its HLS URL.")
    try:
        parts = urlsplit(url)
        if parts.hostname is None or not parts.hostname.endswith(".tiktokcdn.com"):
            raise ReplayError("Replay advertised an unexpected media host.")
        Scope((parts.hostname,)).check(url)
    except ValueError as error:
        raise ReplayError("Malformed replay media URL.") from error
    duration = record(item.get("hls_video_meta_info")).get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ReplayError("Malformed replay duration.")
    try:
        seconds = float(duration)
    except OverflowError as error:
        raise ReplayError("Invalid replay duration.") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ReplayError("Invalid replay duration.")
    raw_start = item.get("start_time")
    start_time = None if raw_start is None else int(broadcast_datetime(raw_start).timestamp())
    return Recording(
        replay_id=replay_id,
        title=title,
        available=True,
        duration=seconds,
        media_url=url,
        start_time=start_time,
    )
