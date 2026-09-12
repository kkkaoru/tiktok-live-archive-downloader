"""Acquire authorized recordings with fresh API URLs and stable-ID locks."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import httpx

from replay.core import Candidate, ReplayError, Scope, save_candidate
from replay.download import Downloader
from replay.jobs import already_complete, download_once
from replay.recordings import Notice, UnavailableReplay
from replay.tiktok_api import TikTokAPI

DEFAULT_MEDIA_HOSTS = ("v16m.tiktokcdn.com", "sf16-videoarch-live-tos-sign.tiktokcdn.com")


class RecordingRow(TypedDict):
    replay_id: str
    anchor_id: str
    title: str
    available: bool
    duration: float | None
    completed: bool


@dataclass(frozen=True, kw_only=True)
class DownloadOptions:
    store: Path
    output: Path
    hosts: tuple[str, ...] = DEFAULT_MEDIA_HOSTS
    workers: int = 6
    max_bytes: int = 20 * 1024**3


@dataclass
class DownloadSummary:
    saved: int = 0
    skipped: int = 0
    unavailable: int = 0


def recording_rows(api: TikTokAPI, *, anchor_id: str, jobs: Path) -> list[RecordingRow]:
    notices = tuple(api.notices(anchor_id=anchor_id))
    rows: list[RecordingRow] = []
    for notice in notices:
        replay = api.resolve(notice)
        rows.append(
            {
                "replay_id": notice.replay_id,
                "anchor_id": notice.anchor_id,
                "title": replay.title,
                "available": replay.available,
                "duration": replay.duration,
                "completed": already_complete(jobs / f"{notice.replay_id}.json"),
            }
        )
    return rows


def acquire_one(api: TikTokAPI, *, notice: Notice, options: DownloadOptions) -> bool:
    output = options.output / f"replay-{notice.replay_id}.mp4"

    def fetch() -> None:
        # Resolution occurs inside the room lock, after the completed-ID check.
        replay = api.resolve(notice)
        if not replay.available or replay.media_url is None:
            raise UnavailableReplay("Recording is currently unavailable.")
        candidate = Candidate(replay.media_url, {}, "hls", "api-advertised")
        save_candidate(options.store, candidate)
        # Separate client: no API cookies or account headers can reach media hosts.
        with httpx.Client(timeout=httpx.Timeout(30, connect=15), trust_env=False) as client:
            Downloader(
                client=client,
                scope=Scope(options.hosts),
                candidate=candidate,
                workers=options.workers,
                max_bytes=options.max_bytes,
            ).download(output)

    return download_once(
        root=options.store.parent / "jobs",
        replay_id=notice.replay_id,
        output=output,
        download=fetch,
    )


def acquire_all(api: TikTokAPI, *, anchor_id: str, options: DownloadOptions) -> DownloadSummary:
    if not 1 <= options.workers <= 16 or options.max_bytes <= 0:
        raise ReplayError("Invalid worker count or download byte limit.")
    # Finish pagination before any media transfer, so partial lists are not treated as complete.
    notices = tuple(api.notices(anchor_id=anchor_id))
    options.output.mkdir(parents=True, exist_ok=True)
    summary = DownloadSummary()
    for notice in notices:
        try:
            saved = acquire_one(api, notice=notice, options=options)
        except UnavailableReplay:
            summary.unavailable += 1
            continue
        if saved:
            summary.saved += 1
        else:
            summary.skipped += 1
    return summary
