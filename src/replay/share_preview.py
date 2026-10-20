"""Link-preview (OGP) images for the web viewer's share links.

One 1200x630 JPEG per catalog video with a local movie: the frame at the video's
strongest excitement peak (``<transcript>.excitement.json``; 30% of the way in without
one), centred over a blurred, darkened copy of itself so vertical and horizontal videos
both fill the card. Cards are cached by movie content key and frame time, and only
changed cards are uploaded to R2 as ``og/<video id>.jpg`` with the authenticated ``cf``
CLI. The Worker serves that one image publicly for a valid share; video, transcript and
summary stay behind the share's bot check.
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError
from replay.excitement import DEFAULT_CACHE, movie_key, sidecar
from replay.recordings import is_record, record

WIDTH: int = 1200
HEIGHT: int = 630
FALLBACK_POSITION: float = 0.3
VIDEO_ID: re.Pattern[str] = re.compile(r"[A-Za-z0-9._-]+")
UPLOADED_STATE: str = "og-uploaded.json"
CARD_FILTER: str = (
    f"[0:v]split[a][b];[a]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
    f"crop={WIDTH}:{HEIGHT},boxblur=24:2,eq=brightness=-0.18[bg];"
    f"[b]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuvj420p"
)

Runner = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class Preview:
    video_id: str
    movie: Path
    seconds: float


def _run(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), check=False, stdout=subprocess.DEVNULL).returncode


def _local(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and Path(value).is_file() else None


def peak_seconds(transcript: Path | None, *, duration: float) -> float:
    """Middle of the strongest bin (combined, else speech), or a fixed share of the video."""
    path = None if transcript is None else sidecar(transcript, "excitement.json")
    estimate = json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None
    # An estimate on another clock (an edit's transcript beside its original) would pick
    # the wrong frame, so its peak is only used when the durations agree.
    clock = estimate.get("duration") if is_record(estimate) else None
    if not is_record(estimate) or not isinstance(clock, int | float) or abs(clock - duration) > 1:
        return round(duration * FALLBACK_POSITION, 3)
    width = estimate.get("binSeconds")
    for name in ("combined", "speech"):
        series = estimate.get(name)
        score = record(series).get("score") if is_record(series) else None
        if isinstance(score, list) and score and isinstance(width, int | float):
            best = max(range(len(score)), key=score.__getitem__)
            return round(min(duration, (best + 0.5) * width), 3)
    return round(duration * FALLBACK_POSITION, 3)


def previews(catalog: Path) -> list[Preview]:
    """Videos whose playback movie is on this machine, with the frame time to use."""
    videos = record(json.loads(catalog.read_text(encoding="utf-8"))).get("videos")
    if not isinstance(videos, list):
        raise ReplayError("Catalog has no video list")
    found: list[Preview] = []
    for entry in videos:
        video = record(entry)
        video_id, duration = video.get("id"), video.get("duration")
        original = video.get("original")
        movie = (
            _local(video.get("localPath"))
            if video.get("movie") is not None
            else (_local(record(original).get("localPath")) if is_record(original) else None)
        )
        if movie is None or not isinstance(duration, int | float) or not isinstance(video_id, str):
            continue
        if VIDEO_ID.fullmatch(video_id) is None:
            raise ReplayError("Catalog entry has an invalid video id")
        seconds = peak_seconds(_local(video.get("transcriptPath")), duration=float(duration))
        found.append(Preview(video_id, movie, seconds))
    return found


def render(preview: Preview, *, cache: Path, runner: Runner) -> Path:
    """The card for this movie content and frame time, rendered once."""
    out = cache / "og" / f"{movie_key(preview.movie)}-{preview.seconds:.3f}.jpg"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.jpg")
    argv = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{preview.seconds:.3f}"]
    argv += ["-i", str(preview.movie), "-filter_complex", CARD_FILTER, "-frames:v", "1"]
    argv += ["-q:v", "3", str(partial)]
    if runner(argv) != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise ReplayError(f"Could not render a preview for {preview.video_id}")
    partial.replace(out)
    return out


def publish(catalog: Path, *, bucket: str, cache: Path, runner: Runner, dry_run: bool) -> list[str]:
    """Render and upload changed cards; returns one line per video."""
    state_path = cache / UPLOADED_STATE
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    uploaded = record(state) if is_record(state) else {}
    lines: list[str] = []
    for preview in previews(catalog):
        card = render(preview, cache=cache, runner=runner)
        digest = hashlib.sha256(card.read_bytes()).hexdigest()
        if uploaded.get(preview.video_id) == digest:
            lines.append(f"unchanged {preview.video_id}")
            continue
        if not dry_run:
            argv = ["cf", "r2", "objects", "put", f"og/{preview.video_id}.jpg"]
            argv += ["--bucket-name", bucket, "--file", str(card), "--content-type", "image/jpeg"]
            if runner(argv) != 0:
                raise ReplayError(f"Upload failed for og/{preview.video_id}.jpg")
            uploaded[preview.video_id] = digest
            state_path.write_text(json.dumps(uploaded, indent=1) + "\n", encoding="utf-8")
        lines.append(f"{'would upload' if dry_run else 'uploaded'} {preview.video_id}")
    return lines


def main(argv: Sequence[str] | None = None, runner: Runner = _run) -> int:
    parser = argparse.ArgumentParser(prog="python -m replay.share_preview", description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    lines = publish(
        args.catalog, bucket=args.bucket, cache=args.cache, runner=runner, dry_run=args.dry_run
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
