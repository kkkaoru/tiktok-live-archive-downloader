"""Render a YouTube edit from its original video, validated cues and per-video config.

Captions are burned in exactly once from ASS. No subtitle track is embedded, so a
player cannot show a second copy over the burned text; the SRT sidecar is kept for
TranscriptViewer. Sources inside an edits directory are refused because they may
already carry burned captions.

Rendering happens in a hidden staging directory beside the delivery. Only a fully
decoded result is moved into the dated delivery path, so a player never opens a
movie that is still being written. An existing delivery is replaced only with
--replace, after the new one has been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import read_cues, render_ass, render_srt
from replay.youtube_delivery import source_stem

CONFIG_KEYS = frozenset(
    {
        "version",
        "video_id",
        "source_video",
        "source_audio",
        "cut_frames",
        "cues",
        "publication",
        "output_root",
        "ffmpeg",
        "ffprobe",
    }
)
EDITED_MARKERS = ("youtube-edits", "-edit-", "-review")
MAX_CUTS = 10_000


@dataclass(frozen=True)
class RenderConfig:
    video_id: str
    source_video: Path
    source_audio: Path | None
    cut_frames: tuple[tuple[int, int], ...]
    cues: Path
    publication: Path
    output_root: Path
    ffmpeg: Path
    ffprobe: Path


def _path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ReplayError(f"{label} must be an absolute path")
    return Path(value)


def _cuts(path: Path) -> tuple[tuple[int, int], ...]:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) > MAX_CUTS:
        raise ReplayError("Cut frames must be a bounded list")
    cuts: list[tuple[int, int]] = []
    previous = -1
    for item in data:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(type(value) is int for value in item)
            or not previous < item[0] <= item[1]
        ):
            raise ReplayError("Cut frames must be ordered, disjoint [first, last] pairs")
        cuts.append((item[0], item[1]))
        previous = item[1]
    return tuple(cuts)


def load_config(path: Path) -> RenderConfig:
    data = record(json.loads(path.read_text(encoding="utf-8")))
    if set(data) != CONFIG_KEYS or data.get("version") != 1:
        raise ReplayError("Render config has unknown, missing or unsupported fields")
    video_id = data["video_id"]
    if not isinstance(video_id, str) or not video_id:
        raise ReplayError("Render config needs a video ID")
    source = _path(data["source_video"], "source_video")
    if any(marker in str(source) for marker in EDITED_MARKERS):
        raise ReplayError("Render source must be the original, not an edited video")
    audio = None if data["source_audio"] is None else _path(data["source_audio"], "source_audio")
    base = path.parent
    cuts_path = base / str(data["cut_frames"])
    cues_path = base / str(data["cues"])
    return RenderConfig(
        video_id=video_id,
        source_video=source,
        source_audio=audio,
        cut_frames=_cuts(cuts_path),
        cues=cues_path,
        publication=_path(data["publication"], "publication"),
        output_root=_path(data["output_root"], "output_root"),
        ffmpeg=_path(data["ffmpeg"], "ffmpeg"),
        ffprobe=_path(data["ffprobe"], "ffprobe"),
    )


def video_filter(cuts: tuple[tuple[int, int], ...], ass: Path) -> str:
    """Drop the configured source frames, retime, then burn the ASS once."""
    escaped = str(ass).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    burn = f"ass=filename='{escaped}'"
    if not cuts:
        return f"[0:v:0]{burn}[v]"
    removed = "+".join(f"between(n,{first},{last})" for first, last in cuts)
    return f"[0:v:0]select='not({removed})',setpts=N/(30*TB),{burn}[v]"


def render(config: RenderConfig, *, replace: bool = False) -> Path:
    """Create <date>_JST_youtube-<id>-review/ with MP4/MOV, ASS and SRT.

    An existing delivery is kept untouched unless ``replace`` is set, and then it
    stays in place until the new render has passed verification.
    """
    publication: object = json.loads(config.publication.read_text(encoding="utf-8"))
    stem = f"{source_stem(config.video_id, publication)}-review"
    final = config.output_root / stem
    if final.exists() and not replace:
        raise FileExistsError(f"Delivery already exists; pass --replace to rebuild {final}")
    # A leftover staging directory from an interrupted render needs inspection.
    output = config.output_root / f".{stem}.rendering"
    output.mkdir(mode=0o700)
    cues = read_cues(config.cues)
    ass = output / f"{stem}.ass"
    srt = output / f"{stem}.srt"
    with ass.open("x", encoding="utf-8") as stream:
        stream.write(render_ass(cues))
    with srt.open("x", encoding="utf-8") as stream:
        stream.write(render_srt(cues))
    wav_audio = config.source_audio is not None and config.source_audio.suffix == ".wav"
    movie = output / f"{stem}.{'mov' if wav_audio else 'mp4'}"
    command = [str(config.ffmpeg), "-nostdin", "-v", "error", "-n", "-i", str(config.source_video)]
    if config.source_audio is not None:
        command += ["-i", str(config.source_audio)]
    command += [
        "-filter_complex",
        video_filter(config.cut_frames, ass),
        "-map",
        "[v]",
        "-map",
        "1:a:0" if config.source_audio is not None else "0:a:0",
        "-sn",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
    ]
    command += ["-c:a", "copy"] if wav_audio else ["-c:a", "aac", "-b:a", "256k"]
    command += ["-movflags", "+faststart", str(movie)]
    subprocess.run(command, check=True)
    verify(config, movie)
    with movie.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    with (output / "render.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "movie": movie.name,
                "sha256": digest.hexdigest(),
                "cues": len(cues),
                "cut_spans": len(config.cut_frames),
                "source_video": str(config.source_video),
                "source_audio": str(config.source_audio) if config.source_audio else None,
                "captions": "burned once from ASS; no embedded subtitle track",
                "full_decode": "passed",
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    _publish(output, final)
    return final


def _publish(staging: Path, final: Path) -> None:
    """Swap a verified render into the delivery path, then drop the old delivery."""
    if not final.exists():
        staging.rename(final)
        return
    retired = final.with_name(f".{final.name}.replaced")
    final.rename(retired)
    staging.rename(final)
    shutil.rmtree(retired)


def verify(config: RenderConfig, movie: Path) -> None:
    """Full decode, and refuse any subtitle stream that could double the captions."""
    probe = subprocess.run(
        [
            str(config.ffprobe),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(movie),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = record(json.loads(probe.stdout)).get("streams")
    kinds = (
        sorted(str(record(item).get("codec_type")) for item in streams)
        if isinstance(streams, list)
        else []
    )
    if kinds != ["audio", "video"]:
        raise ReplayError(f"Expected exactly one video and one audio stream, found {kinds}")
    subprocess.run(
        [
            str(config.ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(movie),
            "-f",
            "null",
            "-",
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="edits/youtube-<id>/render-config.json")
    parser.add_argument(
        "--replace", action="store_true", help="replace an existing delivery after verification"
    )
    args = parser.parse_args(argv)
    print(render(load_config(args.config), replace=args.replace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
