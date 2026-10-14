"""Publish saved on-device transcript summaries to the private web viewer.

``transcript-summary generate`` saves a validated summary beside each transcript in
``<transcript>.summaries/current.json``. This module copies those summaries into the
web viewer's local ``data/<id>.json`` files and ``catalog.json`` and uploads only the
changed objects to the viewer's R2 bucket with the authenticated ``cf`` CLI. It never
generates or edits summary text, never reads a media file and never touches a video
whose catalog entry has no ``transcriptPath`` or whose transcript has no saved summary.
"""

import argparse
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from replay.core import ReplayError, write_private_text
from replay.recordings import record

SUMMARY_VERSION: int = 1
VIDEO_ID: re.Pattern[str] = re.compile(r"[A-Za-z0-9._-]+")
CATALOG_KEY: str = "catalog.json"

Runner = Callable[[Sequence[str]], int]


def _seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ReplayError("Saved summary has an invalid section time.")
    return float(value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError("Saved summary has empty text.")
    return value


def load_saved_summary(transcript: Path) -> dict[str, object] | None:
    """Read the saved summary in the viewer's shape; ``None`` when none is saved."""
    path = transcript.with_name(transcript.name + ".summaries") / "current.json"
    if not path.is_file():
        return None
    saved = record(json.loads(path.read_text(encoding="utf-8")))
    if saved.get("version") != SUMMARY_VERSION:
        raise ReplayError("Saved summary has an unsupported version.")
    rows = saved.get("sections")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Saved summary has no sections.")
    sections: list[object] = []
    for row in rows:
        section = record(row)
        start = _seconds(section.get("startSeconds"))
        end = _seconds(section.get("endSeconds"))
        if end < start:
            raise ReplayError("Saved summary has a reversed section.")
        sections.append({"start": start, "end": end, "text": _text(section.get("text"))})
    return {"overview": _text(saved.get("overview")), "sections": sections}


def _video_id(entry: dict[str, object]) -> str:
    value = entry.get("id")
    if not isinstance(value, str) or VIDEO_ID.fullmatch(value) is None:
        raise ReplayError("Catalog entry has an invalid video id.")
    return value


def merge_summaries(web_dir: Path, *, include_unchanged: bool = False) -> list[str]:
    """Copy saved summaries into the viewer data; return the ids to publish.

    Only changed videos are returned unless ``include_unchanged`` is set, which also
    returns videos whose local data already carries the saved summary (for example after
    the catalog was rebuilt locally but never uploaded).
    """
    catalog_path = web_dir / "data" / CATALOG_KEY
    catalog = record(json.loads(catalog_path.read_text(encoding="utf-8")))
    videos = catalog.get("videos")
    if not isinstance(videos, list):
        raise ReplayError("Viewer catalog has no video list.")
    changed: list[str] = []
    for item in videos:
        entry = record(item)
        recorded = entry.get("transcriptPath")
        summary = load_saved_summary(Path(recorded)) if isinstance(recorded, str) else None
        if summary is None:
            continue
        video_id = _video_id(entry)
        data_path = web_dir / "data" / f"{video_id}.json"
        data = record(json.loads(data_path.read_text(encoding="utf-8")))
        if data.get("summary") == summary and entry.get("hasSummary") is True:
            if include_unchanged:
                changed.append(video_id)
            continue
        data["summary"] = summary
        entry["hasSummary"] = True
        write_private_text(data_path, json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        changed.append(video_id)
    if changed:
        write_private_text(catalog_path, json.dumps(catalog, ensure_ascii=False, indent=1))
    return changed


def _run(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), check=False).returncode


def upload_objects(web_dir: Path, ids: Sequence[str], *, bucket: str, runner: Runner) -> None:
    """Upload each changed data file, then the catalog, with ``cf r2 objects put``."""
    pairs = [(f"data/{video_id}.json", web_dir / "data" / f"{video_id}.json") for video_id in ids]
    pairs.append((CATALOG_KEY, web_dir / "data" / CATALOG_KEY))
    for key, path in pairs:
        argv = ["cf", "r2", "objects", "put", key, "--bucket-name", bucket]
        argv += ["--file", str(path), "--content-type", "application/json"]
        if runner(argv) != 0:
            raise ReplayError(f"Upload failed for {key}; earlier objects were already uploaded.")


def main(argv: Sequence[str] | None = None, runner: Runner = _run) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--all", action="store_true", help="also publish videos whose local data is current"
    )
    args = parser.parse_args(argv)
    changed = merge_summaries(args.web_dir, include_unchanged=args.all)
    print(f"summaries merged: {len(changed)}")
    if args.dry_run or not changed:
        return 0
    upload_objects(args.web_dir, changed, bucket=args.bucket, runner=runner)
    print(f"uploaded: {len(changed) + 1} objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
