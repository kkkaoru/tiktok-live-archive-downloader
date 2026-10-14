"""Independent completion check for the YouTube Motion-overlay edit route.

The legacy ``replay.edit_completion check`` only recognizes dated deliveries
under ``revisions/**/delivery/*-review.*`` or native-FCP evidence, so a
YouTube edit delivered as ``delivery/<timestamp>_JST_<id>-edit-*/`` with a
Motion-alpha-overlay + ffmpeg composite can never pass it. This module
assesses exactly that route against the same five required stages, using only
measured evidence (file hashes, ffprobe decodes, recorded planner receipts).
Nothing here edits ``edit_completion.py`` or ``edit_config.py``; the config
is loaded read-only through their existing loader.

A stage that cannot be proven is reported as unmet, never as complete.
Caption placement inside the measured safe region is an additional explicit
gate beside the five stages.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from replay import caption_ass_completion
from replay.caption_style_defaults import CAPTION_SCREEN_INSET_PX, HEAD_CAPTION_REFERENCE_HEIGHT
from replay.core import ReplayError
from replay.edit_completion import NO_CAPTION_BAND
from replay.edit_config import REQUIRED_STAGES, load_config
from replay.recordings import record
from replay.stages import file_digest

MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
SURVEY_MINIMUM_FRAMES: Final[int] = 120
# Caption-band zone, derived from the retained 8-frame measurement
# (mask-observations/attempt-sp3g5u9c): chat-pattern boxes in the lower band
# end at x<=488 (p99), so a speech-synchronized centered band starts right of
# x=600 while staying below y=500 on the 1280x720 source canvas.
BAND_MIN_X_CENTER: Final[float] = 600.0
BAND_MIN_Y_TOP: Final[float] = 500.0
REPEAT_SPAN_FAIL: Final[int] = 3
REPEAT_GAP_STEPS: Final[int] = 2
CUT_TOLERANCE_SECONDS: Final[float] = 1.5
MINIMUM_REMOVED_SECONDS: Final[float] = 60.0
IDENTITY_WINDOWS: Final[int] = 8
IDENTITY_WINDOW_SECONDS: Final[float] = 30.0
IDENTITY_MATCH_DB: Final[float] = 0.1
# Placement gate: the safe-region left edge (chat p99 + inset margin). Only
# settled captions (displayed longer than two frames) count; a 5px tolerance
# covers shadow/anti-aliasing spill past the glyph box, staying inside the
# 9px margin. One-frame transient cues are listed, never counted.
SAFE_LEFT_PX: Final[float] = 745.0
SETTLED_MIN_FRAMES: Final[int] = 2
EDGE_TOLERANCE_PX: Final[float] = 5.0


def _manifest_version(candidate: Path) -> str:
    match = re.fullmatch(r"composite-full-(.+)-parts\.json", candidate.name)
    return match.group(1) if match else ""


def _evidence_set_complete(workdir: Path, version: str) -> bool:
    """Every render/wrap/placement record the check consumes must exist."""
    if not version:
        return False
    return all(
        (workdir / name).is_file()
        for name in (
            f"titles/wrap-evidence-{version}.json",
            f"titles/tail-wrap-evidence-{version}.json",
            f"titles/left-edge-{version}-settled.json",
            f"titles/left-edge-tail-{version}.json",
        )
    )


@dataclass(frozen=True)
class Delivery:
    directory: Path
    movie: Path
    transcript: Path
    summaries: Path


def _items(value: object, message: str) -> list[object]:
    if not isinstance(value, list):
        raise ReplayError(message)
    return value


def _integer(value: object, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayError(message)
    return value


def _count(value: object, message: str) -> int:
    """Integral JSON numbers or the decimal strings ffprobe emits."""
    if isinstance(value, bool):
        raise ReplayError(message)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ReplayError(message)


def _number(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(message)
    return float(value)


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ReplayError(f"Missing evidence: {path}")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _resolve_media(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReplayError(f"Delivery media is missing: {path}") from error
    if not resolved.is_file() or resolved.suffix.lower() not in (".mp4", ".mov", ".m4v"):
        raise ReplayError(f"Delivery media must resolve to a media file: {path}")
    return resolved


def find_deliveries(root: Path) -> list[Delivery]:
    """Collectors under delivery/<dated>/ holding a movie, transcript and summaries."""
    found: list[Delivery] = []
    delivery_root = root / "delivery"
    if not delivery_root.is_dir():
        return found
    for directory in sorted(delivery_root.iterdir()):
        if not directory.is_dir() or "_JST_" not in directory.name:
            continue
        movies = [
            path
            for path in sorted(directory.iterdir())
            if path.suffix.lower() in (".mp4", ".mov", ".m4v") and path.is_file()
        ]
        transcripts = [
            path
            for path in sorted(directory.iterdir())
            if path.suffix.lower() == ".srt" and path.is_file()
        ]
        if not movies and not transcripts:
            continue
        # A transcript-only generation is reported, never completed: the
        # movie path below does not exist, so assessment fails honestly.
        movie = movies[0] if movies else transcripts[0].with_suffix(".mp4")
        transcript = transcripts[0] if transcripts else movie.with_suffix(".srt")
        found.append(
            Delivery(
                directory=directory,
                movie=movie,
                transcript=transcript,
                summaries=Path(str(transcript) + ".summaries"),
            )
        )
    return found


def _run(executable: str, args: list[str], *, timeout: int) -> str:
    try:
        completed = subprocess.run(
            [executable, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReplayError(f"Measurement tool failed: {executable} ({error})") from error
    if completed.returncode != 0:
        raise ReplayError(f"Measurement failed: {executable} {' '.join(args[:4])}")
    # Tool diagnostics (e.g. volumedetect levels) go to stderr; ffprobe JSON
    # stays on stdout with -v error, so combining both loses nothing.
    return completed.stdout + completed.stderr


def _run_ffmpeg(args: list[str], *, timeout: int) -> str:
    """Polite ffmpeg: lowest priority and two threads, never stalling renders."""
    return _run("nice", ["-n", "19", "ffmpeg", "-threads", "2", *args], timeout=timeout)


def probe_streams(media: Path, *, ffprobe: str = "ffprobe") -> dict[str, object]:
    """Read-only stream/duration facts; never modifies media or evidence."""
    out = _run(
        ffprobe,
        [
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-show_entries",
            "stream=codec_name,profile,width,height,avg_frame_rate,nb_frames,"
            "pix_fmt,sample_rate,channels,bit_rate",
            "-of",
            "json",
            str(media),
        ],
        timeout=300,
    )
    return record(json.loads(out))


def _video_stream(probe: dict[str, object]) -> dict[str, object]:
    for stream in _items(probe.get("streams"), "Probe has no streams"):
        item = record(stream)
        if item.get("codec_type") == "video" or item.get("width") is not None:
            return item
    raise ReplayError("No video stream in probe")


def _audio_stream(probe: dict[str, object]) -> dict[str, object]:
    for stream in _items(probe.get("streams"), "Probe has no streams"):
        item = record(stream)
        if item.get("codec_type") == "audio" or item.get("sample_rate") is not None:
            return item
    raise ReplayError("No audio stream in probe")


def stream_duration_seconds(probe: dict[str, object]) -> float:
    value = record(probe.get("format", {})).get("duration")
    if not isinstance(value, (str, int, float)) or float(value) <= 0:
        raise ReplayError("Probe has no positive duration")
    return float(value)


def has_faststart(media: Path) -> bool:
    """True when the moov box precedes mdat in the first megabytes."""
    with media.open("rb") as stream:
        header = stream.read(8 * 1024 * 1024)
    offset = 0
    seen_moov = False
    while offset + 8 <= len(header):
        size, kind = struct.unpack(">I4s", header[offset : offset + 8])
        if size < 8:
            break
        if kind == b"moov":
            seen_moov = True
            break
        if kind == b"mdat":
            return False
        offset += size
    return seen_moov


def decode_ok(media: Path, *, stream: str = "v") -> bool:
    """A full decode with error detection; exit zero means every packet decoded."""
    target = "0:v:0" if stream == "v" else "0:a:0"
    try:
        _run_ffmpeg(
            ["-nostdin", "-v", "error", "-i", str(media), "-map", target, "-f", "null", "-"],
            timeout=3600,
        )
    except ReplayError as error:
        raise ReplayError(f"media does not fully decode: {media.name} ({error})") from error
    return True


def volume_levels(media: Path, start: float, duration: float = 10.0) -> tuple[float, float]:
    """Measured mean/max volume in dB over one audio window."""
    out = _run_ffmpeg(
        [
            "-nostdin",
            "-v",
            "info",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.1f}",
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        timeout=600,
    )
    mean = maximum = None
    for line in out.splitlines():
        if "mean_volume:" in line:
            try:
                mean = float(line.split("mean_volume:")[1].split("dB")[0])
            except ValueError as error:
                raise ReplayError("volumedetect mean is unreadable") from error
        if "max_volume:" in line:
            try:
                maximum = float(line.split("max_volume:")[1].split("dB")[0])
            except ValueError as error:
                raise ReplayError("volumedetect maximum is unreadable") from error
    if mean is None or maximum is None:
        raise ReplayError("volumedetect produced no levels")
    return mean, maximum


def _srt_cues(transcript: Path) -> list[tuple[float, float, str]]:
    """Parse SRT cues; timestamps are measured delivery facts, not approval."""
    cues: list[tuple[float, float, str]] = []

    def clock(value: str) -> float:
        match = re.fullmatch(r"(\d+):(\d\d):(\d\d),(\d\d\d)", value.strip())
        if match is None:
            raise ReplayError(f"Invalid SRT clock: {value!r}")
        hours, minutes, seconds, millis = (int(part) for part in match.groups())
        return hours * 3600 + minutes * 60 + seconds + millis / 1000

    if not transcript.is_file():
        raise ReplayError(f"Missing transcript: {transcript.name}")
    blocks = transcript.read_text(encoding="utf-8", errors="strict").split("\n\n")
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        start_text, _, end_text = lines[1].partition("-->")
        cues.append((clock(start_text), clock(end_text), "\n".join(lines[2:])))
    if not cues:
        raise ReplayError("Transcript has no cues")
    return cues


def check_bgm_removal(
    config_source: Path,
    delivery: Delivery,
    workdir: Path,
    *,
    cut_program: Path | None = None,
) -> tuple[str | None, dict[str, object]]:
    """Music removal stands on separation evidence plus delivery-audio facts."""
    info: dict[str, object] = {}
    result_path = workdir / "separation" / "result.json"
    result = _read_json(result_path)
    if result.get("device") != "mps":
        return "separation was not measured on the GPU path", info
    vocals = Path(str(result.get("vocals")))
    if not vocals.is_file() or vocals.stat().st_size == 0:
        return "separated vocals are missing", info
    source_probe = probe_streams(config_source)
    source_seconds = stream_duration_seconds(source_probe)
    vocals_seconds = stream_duration_seconds(probe_streams(vocals))
    info["sourceSeconds"] = source_seconds
    info["separatedSeconds"] = round(vocals_seconds, 3)
    if abs(vocals_seconds - source_seconds) > 2.0:
        return "separated audio does not cover the source", info
    movie = _resolve_media(delivery.movie)
    probe = probe_streams(movie)
    audio = _audio_stream(probe)
    info["deliveryAudio"] = {
        "codec": audio.get("codec_name"),
        "sampleRate": audio.get("sample_rate"),
        "channels": audio.get("channels"),
        "bitRate": audio.get("bit_rate"),
    }
    if audio.get("codec_name") != "aac":
        return "delivery audio is not AAC", info
    audio_profile = audio.get("profile")
    audio_rate = audio.get("sample_rate")
    channels = audio.get("channels")
    if audio_profile != "LC" or str(audio_rate) != "44100" or channels != 2:
        return "delivery audio is not LC AAC 44.1kHz stereo", info
    try:
        decode_ok(movie, stream="a")
    except ReplayError:
        return "delivery audio does not fully decode", info
    if cut_program is None:
        info["audioIdentity"] = "unverified-no-reference"
        return None, info
    duration = stream_duration_seconds(probe)
    matched = []
    for index in range(IDENTITY_WINDOWS):
        start = min(duration * (index + 0.5) / IDENTITY_WINDOWS, max(0.0, duration - 1.0))
        first = volume_levels(movie, start, IDENTITY_WINDOW_SECONDS)
        second = volume_levels(cut_program, start, IDENTITY_WINDOW_SECONDS)
        delta = max(abs(first[0] - second[0]), abs(first[1] - second[1]))
        matched.append(round(delta, 3))
        if delta > IDENTITY_MATCH_DB:
            return f"delivery audio differs from the cut program by {delta:.2f}dB", info
    info["audioIdentityDb"] = matched
    info["audioIdentity"] = "verified"
    return None, info


def _select_cut_plan(
    workdir: Path, delivery_seconds: float, source_seconds: float
) -> tuple[Path, list[tuple[float, float, float]], int]:
    """Cut plan matching the delivery, as source-second kept segments."""
    plans = sorted((workdir / "cut-plan").glob("*/cuts.json"))
    if not plans:
        raise ReplayError("no cut plan with measured quiet")
    for plan_path in plans:
        plan = _read_json(plan_path)
        fps = plan.get("fps")
        cuts = plan.get("cuts")
        if type(fps) is not int or not isinstance(cuts, list) or not cuts:
            continue
        segments: list[tuple[float, float, float]] = []
        output = 0.0
        valid = True
        for cut in cuts:
            fields = record(cut)
            end = fields.get("source_end_frame", 0)
            start = fields.get("source_start_frame", 0)
            if not isinstance(end, int) or not isinstance(start, int) or end <= start:
                valid = False
                break
            segments.append((start / fps, end / fps, output))
            output += (end - start) / fps
        if not valid:
            continue
        if abs(output - delivery_seconds) > CUT_TOLERANCE_SECONDS:
            continue
        if not (plan_path.parent / "peaks.txt").is_file():
            raise ReplayError("cut plan lacks its quiet measurement")
        return plan_path, segments, fps
    raise ReplayError("no cut plan matches the delivery duration")


def _map_to_cut(segments: list[tuple[float, float, float]], time: float) -> float | None:
    """Project a source-clock time onto the cut timeline, or None when removed."""
    for start, end, output in segments:
        if start <= time < end:
            return output + (time - start)
        if time < start:
            return None
    return None


def check_silence_cut(
    config_source: Path, delivery_seconds: float, workdir: Path
) -> tuple[str | None, dict[str, object]]:
    """A cut plan whose kept total matches the delivery, with measured peaks."""
    info: dict[str, object] = {}
    source_seconds = stream_duration_seconds(probe_streams(config_source))
    try:
        plan_path, segments, _ = _select_cut_plan(workdir, delivery_seconds, source_seconds)
    except ReplayError as error:
        return str(error), info
    kept = sum(end - start for start, end, _ in segments)
    removed = source_seconds - kept
    info.update(
        {
            "plan": str(plan_path),
            "keptSeconds": round(kept, 3),
            "removedSeconds": round(removed, 3),
            "cuts": len(segments),
        }
    )
    if removed < MINIMUM_REMOVED_SECONDS:
        return "cut plan removes no substantial silence", info
    return None, info


def analyze_mask_survey(
    frames: list[dict[str, object]], *, width: int, height: int
) -> dict[str, object]:
    """Decide whether a repeatable bottom-center caption band was observed.

    The band zone (right of x=600, below y=500 on the 1280x720 source canvas)
    excludes the measured left chat column (chat-pattern p99 x_end 488) and the
    upper diagnostic panel. A normalized text recurring in at least three
    frames no more than two sample steps apart is a repeatable band.
    """
    if width <= 0 or height <= 0:
        raise ReplayError("Survey canvas must be positive")
    occurrences: dict[str, list[int]] = {}
    band_boxes = 0
    ocr_ok = 0
    for index, frame in enumerate(frames):
        item = record(frame)
        boxes = item.get("boxes")
        if item.get("error") is not None or not isinstance(boxes, list):
            continue
        ocr_ok += 1
        time = item.get("time")
        if not isinstance(time, (int, float)):
            raise ReplayError("Survey frame lacks its time")
        for raw in boxes:
            box = record(raw)
            text = box.get("text")
            try:
                coords = [
                    _number(box[key], "Survey box lacks pixel coordinates")
                    for key in ("x0", "y0", "x1", "y1")
                ]
            except ReplayError as error:
                raise ReplayError("Survey box lacks pixel coordinates") from error
            if not isinstance(text, str) or not text.strip():
                continue
            x0, y0, x1, _ = coords
            center = (x0 + x1) / 2
            if y0 >= BAND_MIN_Y_TOP and center >= BAND_MIN_X_CENTER:
                band_boxes += 1
                normalized = re.sub(r"\s+", "", text)
                occurrences.setdefault(normalized, []).append(index)
    clusters: list[dict[str, object]] = []
    for text, steps in occurrences.items():
        ordered = sorted(steps)
        start = previous = ordered[0]
        span = 1
        for first, second in zip(ordered, ordered[1:], strict=False):
            if second - first <= REPEAT_GAP_STEPS:
                span += 1
                previous = second
            else:
                if span >= 2:
                    clusters.append({"text": text, "span": span, "from": start, "to": previous})
                start = previous = second
                span = 1
        if span >= 2:
            clusters.append({"text": text, "span": span, "from": start, "to": previous})
    repeated = [
        cluster
        for cluster in clusters
        if _integer(cluster["span"], "Cluster span is unreadable") >= REPEAT_SPAN_FAIL
    ]
    return {
        "totalFrames": len(frames),
        "ocrOkFrames": ocr_ok,
        "bandBoxes": band_boxes,
        "clusters": sorted(
            clusters, key=lambda item: -_integer(item["span"], "Cluster span is unreadable")
        )[:10],
        "maxRepeatSpan": max(
            (_integer(item["span"], "Cluster span is unreadable") for item in clusters),
            default=0,
        ),
        "reason": NO_CAPTION_BAND if not repeated else "repeated-caption-band-detected",
    }


def _band_box_detail(raw_path: Path) -> dict[str, object]:
    """Re-derive band-zone boxes from the bound raw survey (never trust counts)."""
    frames = _items(json.loads(raw_path.read_text(encoding="utf-8")), "Survey frames unreadable")
    boxes: list[dict[str, object]] = []
    for frame in frames:
        item = record(frame)
        if item.get("error") is not None:
            continue
        for raw in _items(item.get("boxes", []), "Survey boxes are missing"):
            box = record(raw)
            text = box.get("text")
            coords = [
                _number(box[key], "Survey box lacks pixel coordinates")
                for key in ("x0", "y0", "x1", "y1")
            ]
            if not isinstance(text, str) or not text.strip():
                continue
            x0, y0, x1, _ = coords
            if y0 >= BAND_MIN_Y_TOP and (x0 + x1) / 2 >= BAND_MIN_X_CENTER:
                boxes.append(
                    {
                        "time": item.get("time"),
                        "text": text,
                        "box": [round(value, 1) for value in coords],
                    }
                )
    return {"count": len(boxes), "boxes": boxes[:10]}


def check_source_caption_mask(root: Path) -> tuple[str | None, dict[str, object]]:
    """No source speech captions: a thick local Vision survey must say so."""
    summary_path = root / "evidence" / "source-masks-none" / "summary.json"
    if not summary_path.is_file():
        return "no source-masks-none survey evidence", {}
    summary = _read_json(summary_path)
    reason = summary.get("reason")
    total = summary.get("totalFrames")
    if reason == NO_CAPTION_BAND:
        if not isinstance(total, int) or total < SURVEY_MINIMUM_FRAMES:
            return "caption-band survey is thinner than 120 frames", {"totalFrames": total}
        raw = summary.get("framesFile")
        if not isinstance(raw, dict):
            return "survey lacks its bound raw frames", {}
        raw_path = Path(str(raw.get("path")))
        if file_digest(raw_path) != raw.get("sha256"):
            return "survey raw frames do not match their digest", {}
        detail = _band_box_detail(raw_path)
        if detail["count"] != summary.get("bandBoxes"):
            return "survey summary differs from its raw frames", {}
        return None, {
            "totalFrames": total,
            "bandBoxes": detail["count"],
            "bandBoxDetail": detail["boxes"],
        }
    if summary.get("noSourceSubtitlesDetected") is True and summary.get("observedTextFrames") == 0:
        return None, {"totalFrames": total}
    return f"source captions left visible ({reason})", {}


def check_transcription(
    delivery: Delivery, workdir: Path, delivery_seconds: float, source_seconds: float
) -> tuple[str | None, dict[str, object]]:
    """Filtered word recognition must cover the delivered transcript.

    Recognition labels run on the source clock while the transcript runs on
    the cut clock, so labels are projected through the delivery's cut plan
    before any comparison. Unmapped labels (removed silence) do not count.
    """
    info: dict[str, object] = {}
    candidate = workdir / "asr-candidate-filtered.json"
    data = _read_json(candidate)
    labels = data.get("labels")
    if data.get("clock") != "input-pcm-not-projected" or not isinstance(labels, list) or not labels:
        return "filtered recognition is missing or empty", info
    try:
        plan_path, segments, _ = _select_cut_plan(workdir, delivery_seconds, source_seconds)
    except ReplayError as error:
        return f"transcript clock has no cut plan ({error})", info
    mapped: list[float] = []
    unmapped = 0
    for label in labels:
        fields = record(label)
        start = _number(fields.get("start", 0), "Recognition start is not numeric")
        end = _number(fields.get("end", 0), "Recognition end is not numeric")
        projected = _map_to_cut(segments, (start + end) / 2)
        if projected is None:
            unmapped += 1
        else:
            mapped.append(projected)
    info["labels"] = len(labels)
    info["mappedLabels"] = len(mapped)
    info["unmappedLabels"] = unmapped
    info["labelClock"] = f"source-presentation mapped to cut clock via {plan_path.name}"
    info["hallucinationFilter"] = data.get("hallucinationFilter")
    if not mapped:
        return "no recognized words survive on the cut timeline", info
    transcript = delivery.transcript
    if not transcript.is_file() or "-->" not in transcript.read_text(encoding="utf-8"):
        return "delivery has no transcript track file", info
    cues = _srt_cues(transcript)
    info["srtCues"] = len(cues)
    if cues[0][0] < min(mapped) - 1.0 or cues[-1][1] > max(mapped) + 1.0:
        return "transcript span exceeds the recognized span", info
    if len(cues) < len(mapped) * 0.5:
        return "transcript covers less than half the recognized words", info
    return None, info


def check_dynamic_captions(
    delivery: Delivery,
    workdir: Path,
    delivery_seconds: float,
    *,
    cut_program: Path | None = None,
) -> tuple[str | None, dict[str, object]]:
    """Rendered overlay parts, wrap evidence and the delivered file must agree."""
    info: dict[str, object] = {}
    manifests = sorted(workdir.glob("composite-full-*-parts.json"))
    if not manifests:
        return "no composite parts manifest", info
    manifest: dict[str, object] | None = None
    manifest_version = ""
    candidates: list[tuple[Path, dict[str, object]]] = []
    for candidate in manifests:
        data = _read_json(candidate)
        raw_parts = data.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            continue
        entry_frames: list[int] = []
        for raw in raw_parts:
            fields = record(raw)
            offset = _integer(fields.get("offsetFrame", 0), "Part offset is not integral")
            count = _integer(fields.get("frames", 0), "Part frames are not integral")
            entry_frames.append(offset + count)
        rate = data.get("rate")
        if isinstance(rate, int) and abs(max(entry_frames) / rate - delivery_seconds) <= 2.0:
            candidates.append((candidate, data))
    if not candidates:
        return "no parts manifest matches the delivery duration", info
    # Several generations can match one duration; only a manifest whose full
    # evidence set exists is usable. The choice is recorded, never silent.
    usable = [
        (c, d) for c, d in candidates if _evidence_set_complete(workdir, _manifest_version(c))
    ]
    if not usable:
        return "no parts manifest has its complete evidence set", info
    candidates = sorted(usable, key=lambda item: item[0].name)
    candidate, manifest = candidates[-1]
    manifest_version = _manifest_version(candidate)
    info["manifest"] = candidate.name
    info["rejectedManifests"] = [c.name for c, _ in candidates[:-1]]
    parts = [record(part) for part in _items(manifest["parts"], "Parts are missing")]
    for first, second in zip(parts, parts[1:], strict=False):
        first_end = _integer(first.get("offsetFrame"), "Part offset is not integral") + _integer(
            first.get("frames"), "Part frames are not integral"
        )
        if _integer(second.get("offsetFrame"), "Part offset is not integral") < first_end:
            return "overlay parts overlap", info
    missing = [str(part.get("path")) for part in parts if not Path(str(part.get("path"))).is_file()]
    if missing:
        return (
            f"{len(missing)} rendered overlay parts are missing",
            {**info, "missingHead": missing[:3]},
        )
    info["parts"] = len(parts)
    probed = 0
    for part in (parts[0], parts[len(parts) // 2], parts[-1]):
        probe = probe_streams(Path(str(part["path"])))
        video = _video_stream(probe)
        if video.get("pix_fmt") != "yuva444p12le" or video.get("width") != 1920:
            return "overlay render is not 1920-wide alpha", info
        rendered = _count(video.get("nb_frames"), "Render frame count is unreadable")
        if rendered != _integer(part["frames"], "Part frames are not integral"):
            return "overlay render frame count differs from its manifest", info
        probed += 1
    info["spotProbed"] = probed
    for name in (
        f"titles/wrap-evidence-{manifest_version}.json",
        f"titles/tail-wrap-evidence-{manifest_version}.json",
    ):
        evidence = _read_json(workdir / name)
        if _integer(evidence.get("failures", 0) or 0, "Wrap failures are unreadable") != 0:
            return f"caption wrap has failures ({name})", info
        widest = _number(evidence.get("widestRewrappedLine", 0), "Wrap width is unreadable")
        safe = _number(evidence.get("safeWidthPx", 0), "Safe width is unreadable")
        if widest > safe:
            return f"caption line exceeds its safe width ({name})", info
    movie = _resolve_media(delivery.movie)
    probe = probe_streams(movie)
    video = _video_stream(probe)
    audio = _audio_stream(probe)
    info["deliveryVideo"] = {
        "codec": video.get("codec_name"),
        "size": [video.get("width"), video.get("height")],
        "frames": video.get("nb_frames"),
        "pixFmt": video.get("pix_fmt"),
        "faststart": has_faststart(movie),
    }
    info["deliveryAudio"] = {
        "codec": audio.get("codec_name"),
        "sampleRate": audio.get("sample_rate"),
        "channels": audio.get("channels"),
    }
    if video.get("codec_name") != "h264" or video.get("pix_fmt") != "yuv420p":
        return "delivery video is not QuickTime-compatible H.264", info
    if [video.get("width"), video.get("height")] != [1920, 1080]:
        return "delivery video is not 1920x1080", info
    if not has_faststart(movie):
        return "delivery moov is not ahead of mdat", info
    reported = _count(video.get("nb_frames"), "Delivery frame count is unreadable")
    if abs(reported - round(delivery_seconds * 30)) > 2:
        return "delivery frame count differs from its duration", info
    if audio.get("codec_name") != "aac" or audio.get("profile") != "LC":
        return "delivery audio is not LC AAC", info
    if str(audio.get("sample_rate")) != "44100" or audio.get("channels") != 2:
        return "delivery audio is not 44.1kHz stereo", info
    try:
        program_ok = decode_ok(movie, stream="v") and decode_ok(movie, stream="a")
    except ReplayError:
        return "delivery does not fully decode", info
    if not program_ok:
        return "delivery does not fully decode", info
    failure, burn = check_burn_in(workdir, manifest_version, cut_program, movie)
    info["burnIn"] = burn
    if failure is not None:
        return failure, info
    failure, clip = check_clip_geometry(
        workdir, manifest_version, parts, program=cut_program, delivery_movie=movie
    )
    info["clipGeometry"] = clip
    if failure is not None:
        return failure, info
    return None, info


def _multiline_burn_times(workdir: Path, version: str, *, limit: int = 3) -> list[float]:
    """Global seconds of rewrapped multi-line captions from the title bundle."""
    bundle = _read_json(workdir / "titles" / f"bundle-{version}.json")
    evidence = _read_json(workdir / "titles" / f"wrap-evidence-{version}.json")
    offsets: list[int] = []
    for raw in _items(bundle.get("titleParts"), "Title bundle parts are missing"):
        offsets.append(_integer(record(raw).get("offsetFrame"), "Bundle offset unreadable"))
    times: list[float] = []
    for raw in _items(evidence.get("records", []), "Wrap records are missing"):
        item = record(raw)
        lines = item.get("lines")
        part = item.get("part")
        start = item.get("startFrame")
        if not isinstance(lines, list) or len(lines) < 2:
            continue
        if not isinstance(part, int) or not 0 <= part < len(offsets):
            raise ReplayError("Wrap record part is out of range")
        times.append((offsets[part] + _integer(start, "Wrap start unreadable") + 6) / 30)
        if len(times) >= limit:
            break
    return times


def _diff_band_box(
    program: Path, delivery: Path, time: float, *, band_top: int = 800
) -> tuple[int, int, int, int] | None:
    """Bounding box of strong program/delivery differences in the caption band."""
    width, height = 1920, 1080

    def frame(video: Path) -> bytes:
        completed = subprocess.run(
            [
                "nice",
                "-n",
                "19",
                "ffmpeg",
                "-threads",
                "2",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{time:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:{height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=180,
        )
        if completed.returncode != 0:
            raise ReplayError("Caption proof frame did not decode")
        return completed.stdout

    base, final = frame(program), frame(delivery)
    if len(base) != width * height * 3 or len(final) != width * height * 3:
        raise ReplayError("Caption proof frames did not decode")
    xs: list[int] = []
    ys: list[int] = []
    for index in range(0, len(base), 3):
        y = (index // 3) // width
        if y < band_top:
            continue
        if (
            abs(base[index] - final[index]) > 40
            or abs(base[index + 1] - final[index + 1]) > 40
            or abs(base[index + 2] - final[index + 2]) > 40
        ):
            xs.append((index // 3) % width)
            ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _diff_band_height(program: Path, delivery: Path, time: float, *, band_top: int = 800) -> int:
    """Height of strong program/delivery differences in the caption band."""
    box = _diff_band_box(program, delivery, time, band_top=band_top)
    return 0 if box is None else box[3] - box[1] + 1


def check_burn_in(
    workdir: Path, version: str, program: Path | None, delivery_movie: Path
) -> tuple[str | None, dict[str, object]]:
    """Multi-line captions from this render generation must be measurable."""
    try:
        times = _multiline_burn_times(workdir, version)
    except ReplayError as error:
        return f"burn-in proof has no measurable times ({error})", {}
    if not times:
        return "burn-in proof has no multi-line captions", {}
    if program is None:
        return None, {"burnIn": "unverified-no-reference", "sampleTimes": times}
    heights = [_diff_band_height(program, delivery_movie, time) for time in times]
    info: dict[str, object] = {"sampleTimes": times, "bandHeightsPx": heights}
    if not [height for height in heights if height >= 150]:
        return "no multi-line caption burn-in measured", info
    return None, info


def _alpha_box(
    render: Path, time: float, *, threshold: int = 16
) -> tuple[int, int, int, int] | None:
    """Bounding box of rendered caption alpha at one render-local second."""
    width, height = 1920, 1080
    completed = subprocess.run(
        [
            "nice",
            "-n",
            "19",
            "ffmpeg",
            "-threads",
            "2",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{time:.3f}",
            "-i",
            str(render),
            "-frames:v",
            "1",
            "-vf",
            "alphaextract,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise ReplayError("Caption alpha frame did not decode")
    raw = completed.stdout
    if len(raw) != width * height:
        raise ReplayError("Caption alpha frame has an unexpected canvas")
    xs: list[int] = []
    ys: list[int] = []
    for index, value in enumerate(raw):
        if value >= threshold:
            xs.append(index % width)
            ys.append(index // width)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _clip_samples(
    workdir: Path, version: str, manifest_parts: list[dict[str, object]]
) -> list[dict[str, object]]:
    """One 1/2/3-line cue per bundle, bound to its render file and local time."""
    spans = [
        (
            _integer(part.get("offsetFrame"), "Part offset is not integral"),
            _integer(part.get("frames"), "Part frames are not integral"),
            str(part.get("path")),
        )
        for part in manifest_parts
    ]
    samples: list[dict[str, object]] = []
    for bundle_name in (f"bundle-{version}.json", f"tail-bundle-{version}.json"):
        bundle = _read_json(workdir / "titles" / bundle_name)
        wanted = {1, 2, 3} if bundle_name.startswith("bundle-") else {2}
        for raw_part in _items(bundle.get("titleParts"), "Bundle parts are missing"):
            part = record(raw_part)
            offset = _integer(part.get("offsetFrame"), "Bundle offset unreadable")
            for raw_cue in _items(part.get("placements", []), "Bundle cues are missing"):
                cue = record(raw_cue)
                text = cue.get("text")
                start = _integer(cue.get("startFrame"), "Cue start unreadable")
                end = _integer(cue.get("endFrame"), "Cue end unreadable")
                if not isinstance(text, str) or end - start < 10:
                    continue
                lines = text.count("\n") + 1
                if lines not in wanted:
                    continue
                wanted.discard(lines)
                moment = offset + (start + end) / 2
                match = [
                    (begin, count, path)
                    for begin, count, path in spans
                    if begin <= moment < begin + count
                ]
                if not match:
                    raise ReplayError("Clip sample is outside its overlay parts")
                begin, _, path = match[0]
                samples.append(
                    {
                        "render": path,
                        "time": (moment - begin) / 30,
                        "global": moment / 30,
                        "lines": lines,
                    }
                )
            if not wanted:
                break
    if len([s for s in samples if s["lines"] == 1]) < 1:
        raise ReplayError("Clip proof lacks a one-line caption sample")
    return samples


def check_clip_geometry(
    workdir: Path,
    version: str,
    manifest_parts: list[dict[str, object]],
    *,
    program: Path | None,
    delivery_movie: Path,
) -> tuple[str | None, dict[str, object]]:
    """Rendered alpha bottoms and final pixels must clear the canvas edge."""
    inset = CAPTION_SCREEN_INSET_PX * 1080 / HEAD_CAPTION_REFERENCE_HEIGHT
    try:
        samples = _clip_samples(workdir, version, manifest_parts)
    except ReplayError as error:
        return f"clip proof has no measurable samples ({error})", {}
    info: dict[str, object] = {"samples": len(samples)}
    for sample in samples:
        moment = _number(sample["time"], "Sample time unreadable")
        box = _alpha_box(Path(str(sample["render"])), moment)
        if box is None:
            return "caption has no rendered alpha", info
        sample["alphaBox"] = list(box)
        if box[3] > int(1080 - inset):
            return "overlay alpha bottom exceeds the safe area", info
    if program is None:
        info["finalClip"] = "unverified-no-reference"
        return None, info
    for sample in samples:
        if _integer(sample["lines"], "Sample lines unreadable") < 2:
            continue
        moment = _number(sample["global"], "Sample time unreadable")
        box = _diff_band_box(program, delivery_movie, moment)
        if box is None:
            continue
        sample["finalBox"] = list(box)
        if box[3] > 1072:
            return "final caption touches the canvas edge", info
    return None, info


def check_placement(workdir: Path, manifest_version: str) -> tuple[bool, dict[str, object]]:
    """Caption boxes rendered by the overlay must stay inside the safe region.

    Only settled captions (displayed longer than two frames) count: one-frame
    transient cues are listed, never counted. A 5px tolerance covers
    shadow/anti-aliasing spill past the glyph box, staying inside the 9px
    margin the 745px edge was derived with. Bundle clocks map each sample to
    its displayed cue; unmapped samples stay violations.
    """
    settled = _read_json(workdir / "titles" / f"left-edge-{manifest_version}-settled.json")
    tail = _read_json(workdir / "titles" / f"left-edge-tail-{manifest_version}.json")

    def intervals(bundle_name: str) -> dict[int, list[tuple[int, int]]]:
        bundle = _read_json(workdir / "titles" / bundle_name)
        placements: dict[int, list[tuple[int, int]]] = {}
        for index, raw in enumerate(_items(bundle.get("titleParts"), "Bundle parts missing")):
            cues = []
            for entry in _items(record(raw).get("placements", []), "Bundle cues missing"):
                cue = record(entry)
                cues.append(
                    (
                        _integer(cue.get("startFrame"), "Cue start unreadable"),
                        _integer(cue.get("endFrame"), "Cue end unreadable"),
                    )
                )
            placements[index] = cues
        return placements

    bundles = {
        "settled": intervals(f"bundle-{manifest_version}.json"),
        "tail": intervals(f"tail-bundle-{manifest_version}.json"),
    }
    violations: list[dict[str, object]] = []
    transient: list[dict[str, object]] = []
    tolerated: list[dict[str, object]] = []
    undetected = 0
    for name, data in (("settled", settled), ("tail", tail)):
        summary = record(data.get("summary", {}))
        undetected += _integer(summary.get("undetected", 0) or 0, "Undetected count unreadable")
        for row in _items(data.get("rows", []), "Placement rows are missing"):
            item = record(row)
            part = _integer(item.get("part", -1), "Placement part unreadable")
            start = _integer(item.get("startFrame", -1), "Placement start unreadable")
            box = item.get("box")
            shown = [
                end - begin for begin, end in bundles[name].get(part, []) if begin <= start < end
            ]
            duration = max(shown, default=0)
            entry = {"evidence": name, "part": part, "box": box, "cueFrames": duration}
            if part not in bundles[name]:
                violations.append({**entry, "unmapped": True})
                continue
            if not isinstance(box, list) or len(box) != 4:
                transient.append(entry)
                continue
            left = _number(box[0], "Box edge unreadable")
            if duration <= SETTLED_MIN_FRAMES:
                transient.append(entry)
            elif left >= SAFE_LEFT_PX:
                continue
            elif SAFE_LEFT_PX - left <= EDGE_TOLERANCE_PX:
                tolerated.append({**entry, "intrusionPx": round(SAFE_LEFT_PX - left, 1)})
            else:
                violations.append({**entry, "intrusionPx": round(SAFE_LEFT_PX - left, 1)})
    return not violations, {
        "violations": violations,
        "transient": transient,
        "tolerated": tolerated,
        "undetected": undetected,
    }


def assess_delivery(
    root: Path,
    delivery: Delivery,
    *,
    workdir: Path,
    cut_program: Path | None = None,
    ass_evidence: Path | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Return per-stage failures plus measured facts; empty failures pass stages."""
    config = load_config(root)
    movie = _resolve_media(delivery.movie)
    delivery_seconds = stream_duration_seconds(probe_streams(movie))
    source_seconds = stream_duration_seconds(probe_streams(config.source))
    failures: dict[str, str] = {}
    facts: dict[str, object] = {
        "movieSha256": file_digest(movie),
        "deliverySeconds": round(delivery_seconds, 3),
    }
    failure, info = check_bgm_removal(config.source, delivery, workdir, cut_program=cut_program)
    facts["bgmRemoval"] = info
    if failure is not None:
        failures["bgm_removal"] = failure
    failure, info = check_silence_cut(config.source, delivery_seconds, workdir)
    facts["silenceCut"] = info
    if failure is not None:
        failures["silence_cut"] = failure
    failure, info = check_source_caption_mask(root)
    facts["sourceCaptionMask"] = info
    if failure is not None:
        failures["source_caption_mask"] = failure
    if ass_evidence is None:
        failure, info = check_transcription(delivery, workdir, delivery_seconds, source_seconds)
    else:
        proof = caption_ass_completion.read_document(ass_evidence)
        refs = record(proof.get("artifacts"))
        failure, info = caption_ass_completion.transcription(
            caption_ass_completion.bound_path(refs.get("plan")),
            caption_ass_completion.bound_path(refs.get("cues")),
            caption_ass_completion.bound_path(refs.get("cuts")),
        )
    facts["transcription"] = info
    if failure is not None:
        failures["transcription"] = failure
    if ass_evidence is None:
        failure, info = check_dynamic_captions(
            delivery, workdir, delivery_seconds, cut_program=cut_program
        )
    else:
        failure, info, placement = caption_ass_completion.assess(
            ass_evidence, movie, delivery.transcript
        )
        video = _video_stream(probe_streams(movie))
        plan = caption_ass_completion.read_document(
            caption_ass_completion.bound_path(refs.get("plan"))
        )
        if (
            video.get("width") != 1920
            or video.get("height") != 1080
            or video.get("avg_frame_rate") != "30/1"
            or video.get("codec_name") != "h264"
            or video.get("pix_fmt") != "yuv420p"
            or _count(video.get("nb_frames"), "ASS frame count is unreadable")
            != _integer(plan.get("frames"), "ASS plan frame count is missing")
            or not has_faststart(movie)
        ):
            failure = "ASS delivery format differs from its output contract"
        decode_ok(movie, stream="v")
    facts["dynamicCaptions"] = info
    if failure is not None:
        failures["dynamic_captions"] = failure
    manifest_name = str(info.get("manifest", ""))
    match = re.fullmatch(r"composite-full-(.+)-parts\.json", manifest_name)
    if ass_evidence is None:
        placement_ok, placement = check_placement(workdir, match.group(1) if match else "")
    facts["placement"] = placement
    return failures, facts


def completion_report(
    root: Path,
    *,
    workdir: Path | None = None,
    cut_program: Path | None = None,
    ass_evidence: Path | None = None,
) -> dict[str, object]:
    """Summarize YouTube-route deliveries; complete needs stages plus placement."""
    load_config(root)
    registered = root / "evidence" / "caption-ass-delivery.json"
    if ass_evidence is None and (registered.exists() or registered.is_symlink()):
        ass_evidence = registered
    work = workdir or (root.parent / "work" / root.name)
    deliveries = find_deliveries(root)
    selected_movie: Path | None = None
    if ass_evidence is not None:
        refs = record(caption_ass_completion.read_document(ass_evidence).get("artifacts"))
        selected_movie = caption_ass_completion.bound_path(refs.get("movie"))
    rows: list[dict[str, object]] = []
    for delivery in deliveries:
        # A fingerprint-bound published selection takes precedence over a
        # generation timestamp. Historical generations remain reported.
        current = (
            delivery.movie.resolve() == selected_movie
            if selected_movie is not None
            else delivery == deliveries[-1]
        )
        try:
            failures, facts = assess_delivery(
                root,
                delivery,
                workdir=work,
                cut_program=cut_program,
                ass_evidence=ass_evidence if current else None,
            )
            placement = record(facts.get("placement", {}))
            placement_ok = not _items(placement.get("violations", []), "Placement unreadable")
        except (ReplayError, OSError, ValueError) as error:
            failures, facts, placement_ok = {"delivery": str(error)}, {}, False
        rows.append(
            {
                "delivery": delivery.directory.name,
                "complete": not failures and placement_ok and current,
                "superseded": not current,
                "missingStages": failures,
                "facts": facts,
            }
        )
    return {
        "route": "youtube-libass-phrase" if ass_evidence is not None else "youtube-motion-overlay",
        "recording": root.name,
        "complete": any(row["complete"] for row in rows),
        "requiredStages": list(REQUIRED_STAGES),
        "deliveries": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Assess YouTube-route deliveries")
    check.add_argument("root", type=Path)
    check.add_argument("--workdir", type=Path, default=None)
    check.add_argument("--cut-program", type=Path, default=None)
    check.add_argument("--candidate", type=Path, default=None)
    check.add_argument("--transcript", type=Path, default=None)
    check.add_argument("--ass-evidence", type=Path, default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).absolute()
    workdir = Path(args.workdir).absolute() if args.workdir else None
    cut_program = Path(args.cut_program).absolute() if args.cut_program else None
    try:
        if args.candidate is not None:
            if args.transcript is None or args.ass_evidence is None:
                raise ReplayError("Candidate check requires transcript and ASS evidence")
            movie = args.candidate.absolute()
            delivery = Delivery(
                movie.parent,
                movie,
                args.transcript.absolute(),
                Path(str(args.transcript) + ".summaries"),
            )
            failures, facts = assess_delivery(
                root,
                delivery,
                workdir=workdir or root.parent / "work" / root.name,
                cut_program=cut_program,
                ass_evidence=args.ass_evidence.absolute(),
            )
            report = {
                "route": "youtube-libass-phrase",
                "candidateOnly": True,
                "complete": not failures,
                "requiredStages": list(REQUIRED_STAGES),
                "missingStages": failures,
                "facts": facts,
            }
        else:
            report = completion_report(
                root,
                workdir=workdir,
                cut_program=cut_program,
                ass_evidence=args.ass_evidence.absolute() if args.ass_evidence else None,
            )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["complete"] else 2
    except (ReplayError, OSError, ValueError) as error:
        print(f"YOUTUBE_EDIT_COMPLETION_ERROR {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
