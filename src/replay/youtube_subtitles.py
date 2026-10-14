"""Config-driven, original-audio YouTube captions using native ASR segment clocks."""

import argparse
import importlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from hashlib import file_digest
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import SpeakerCue, render_ass, render_srt
from replay.youtube_delivery import new_edit_directory, source_stem


class UnrenderableSegment(ReplayError):
    """Native predictions cannot be rendered within their measured input clock."""


@dataclass(frozen=True, kw_only=True)
class SubtitleJob:
    video_id: str
    source: Path
    publication: Path
    output_root: Path
    keep_bgm: bool
    audio_policy: str
    subtitle_mode: str
    speakers: int
    model: Path
    weights_sha256: str
    ffmpeg: Path
    ffprobe: Path


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError("Expected nonempty configuration text")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError("Expected finite recognition clock")
    return float(value)


def load_job(path: Path) -> SubtitleJob:
    data = record(json.loads(path.read_text(encoding="utf-8")))
    required = {
        "version",
        "video_id",
        "source",
        "publication",
        "output_root",
        "keep_bgm",
        "audio_policy",
        "subtitle_mode",
        "speakers",
        "model",
        "weights_sha256",
        "ffmpeg",
        "ffprobe",
    }
    if set(data) != required or type(data["version"]) is not int or data["version"] != 1:
        raise ReplayError("Unknown or incomplete subtitle configuration")
    if type(data["keep_bgm"]) is not bool or type(data["speakers"]) is not int:
        raise ReplayError("Invalid per-video BGM or speaker policy")
    # A reduced input must already exist; this runner never substitutes an
    # unverified separator or infers speaker identity from performer count.
    policy = "original" if data["keep_bgm"] else "existing_reduced"
    if (
        data["audio_policy"] != policy
        or data["speakers"] not in (1, 3)
        or data["subtitle_mode"] != "burn_and_embed"
    ):
        raise ReplayError("Conflicting per-video input audio policy")
    source = Path(_text(data["source"])).resolve(strict=True)
    if "youtube-edits" in source.parts or any(
        marker in source.name for marker in ("-edit-", "-review")
    ):
        # An edited video may already carry burned captions; burning again doubles them.
        raise ReplayError("Subtitle source must not be an edited video")
    publication = Path(_text(data["publication"])).resolve(strict=True)
    video_id = _text(data["video_id"])
    source_stem(video_id, json.loads(publication.read_text(encoding="utf-8")))
    checksum = _text(data["weights_sha256"])
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ReplayError("Invalid pinned model checksum")
    model = Path(_text(data["model"])).resolve(strict=True)
    if not source.is_file() or not publication.is_file() or not model.is_dir():
        raise ReplayError("Expected existing source, publication evidence and local model")
    return SubtitleJob(
        video_id=video_id,
        source=source,
        publication=publication,
        output_root=Path(_text(data["output_root"])),
        keep_bgm=data["keep_bgm"],
        audio_policy=policy,
        subtitle_mode="burn_and_embed",
        speakers=data["speakers"],
        model=model,
        weights_sha256=checksum,
        ffmpeg=Path(_text(data["ffmpeg"])),
        ffprobe=Path(_text(data["ffprobe"])),
    )


def segment_cues(value: object, *, offset: float, duration: float) -> tuple[SpeakerCue, ...]:
    """Use actual decoded segment tokens, never assign interpolated word clocks."""
    result = record(value)
    segments = result.get("segments")
    if result.get("language") != "ja" or not isinstance(segments, list) or len(segments) > 50000:
        raise ReplayError("Invalid Japanese native segment transcription")
    if not math.isfinite(offset) or offset < 0 or not math.isfinite(duration) or duration <= 0:
        raise ReplayError("Invalid measured audio extent")
    cues: list[SpeakerCue] = []
    for raw in segments:
        item = record(raw)
        text = item.get("text")
        if not isinstance(text, str):
            raise ReplayError("Invalid native segment text")
        if not text.strip():
            continue
        start, end = _number(item.get("start")), _number(item.get("end"))
        if not 0 <= start < end <= duration + 1e-6 or round(start * 100) >= round(end * 100):
            raise UnrenderableSegment(
                "Unrenderable native segment; recover from audio, never stretch"
            )
        if cues and start < cues[-1].start - offset:
            raise ReplayError("Reversed native segment order")
        cues.append(SpeakerCue(start + offset, end + offset, "unassigned", text.strip()))
    return tuple(cues)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd).stdout


def _duration(job: SubtitleJob) -> float:
    probe = record(
        json.loads(
            _run(
                [
                    str(job.ffprobe),
                    "-v",
                    "error",
                    "-show_format",
                    "-of",
                    "json",
                    str(job.source),
                ]
            )
        )
    )
    duration = float(_text(record(probe.get("format")).get("duration")))
    if not math.isfinite(duration) or not 0 < duration <= 86400:
        raise ReplayError("Source duration exceeds the local job budget")
    return duration


def owned_segment_cues(
    value: object, *, offset: float, duration: float, core: int
) -> tuple[SpeakerCue, ...]:
    """Unowned context-tail predictions cannot create or erase owned captions."""
    data = record(value)
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ReplayError("Missing native segment list")
    retained: list[object] = []
    for raw in segments:
        item = record(raw)
        start = _number(item.get("start"))
        end = _number(item.get("end"))
        if start + offset >= core + 24 or end + offset <= core:
            continue
        retained.append(raw)
    return tuple(
        cue
        for cue in segment_cues(
            {"language": data.get("language"), "segments": retained},
            offset=offset,
            duration=duration,
        )
        if core <= (cue.start + cue.end) / 2 < core + 24
    )


def _job_data(job: SubtitleJob) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value for key, value in asdict(job).items()
    }


def run_job(job: SubtitleJob, *, resume: Path | None = None) -> Path:
    """Independent chunk decoding prevents long-form skipping/repetition cascades."""
    with (job.model / "weights.npz").open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != job.weights_sha256:
            raise ReplayError("Local ASR model checksum changed")
    duration = _duration(job)
    transcribe: object = importlib.import_module("mlx_whisper").transcribe
    if not callable(transcribe):
        raise ReplayError("Installed local recognizer is not callable")
    publication: object = json.loads(job.publication.read_text(encoding="utf-8"))
    if resume is None:
        output = new_edit_directory(job.output_root, video_id=job.video_id, metadata=publication)
        (output / "metadata").mkdir(mode=0o700)
        (output / "metadata/job.json").write_text(
            json.dumps(_job_data(job), indent=2),
            encoding="utf-8",
        )
    else:
        output = resume.resolve(strict=True)
        if (
            output.parent != job.output_root.resolve()
            or not output.name.startswith(source_stem(job.video_id, publication) + "-edit-")
            or json.loads((output / "metadata/job.json").read_text(encoding="utf-8"))
            != _job_data(job)
            or (output / "metadata/result.json").exists()
        ):
            raise ReplayError("Resume job/config mismatch or already completed output")
    metadata = output / "metadata"
    all_cues: list[SpeakerCue] = []
    # Native overlapping utterances stay intact. Midpoint ownership chooses one
    # independent context window; neither text nor timestamps are clipped.
    for index, core in enumerate(range(0, math.ceil(duration), 24)):
        start = max(0.0, core - 3.0)
        end = min(duration, core + 27.0)
        wave = metadata / f"chunk-{index:05}.wav"
        cached = metadata / f"chunk-{index:05}.json"
        if cached.exists():
            stored = record(json.loads(cached.read_text(encoding="utf-8")))
            if stored.get("source_offset") != start or stored.get("core_start") != core:
                raise ReplayError("Cached chunk clock mismatch")
            recognition = stored.get("recognition")
        else:
            if not wave.exists():
                _run(
                    [
                        str(job.ffmpeg),
                        "-nostdin",
                        "-v",
                        "error",
                        "-n",
                        "-ss",
                        str(start),
                        "-i",
                        str(job.source),
                        "-t",
                        str(end - start),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        str(wave),
                    ]
                )
            recognition = transcribe(
                str(wave),
                path_or_hf_repo=str(job.model),
                language="ja",
                verbose=False,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            with cached.open("x", encoding="utf-8") as stream:
                json.dump(
                    {"source_offset": start, "core_start": core, "recognition": recognition},
                    stream,
                    ensure_ascii=False,
                )
        try:
            cues = owned_segment_cues(recognition, offset=start, duration=end - start, core=core)
        except UnrenderableSegment as error:
            # Recover cut utterances from a DIFFERENT measured context; never
            # legitimize old out-of-input clocks by just extending their limit.
            previous_extent = (start, end)
            for before, after in ((6, 30), (12, 48)):
                extended_start = max(0.0, core - before)
                extended_end = min(duration, core + after)
                if previous_extent == (extended_start, extended_end):
                    raise error
                previous_extent = (extended_start, extended_end)
                name = f"recovery-{index:05}-extended{before + after}"
                recovered = metadata / f"{name}.json"
                if recovered.exists():
                    measured = record(json.loads(recovered.read_text(encoding="utf-8")))
                    if measured.get("source_offset") != extended_start:
                        raise ReplayError("Recovery context clock mismatch") from error
                    alternate = measured.get("recognition")
                else:
                    extended_wave = metadata / f"{name}.wav"
                    if not extended_wave.exists():
                        _run(
                            [
                                str(job.ffmpeg),
                                "-nostdin",
                                "-v",
                                "error",
                                "-n",
                                "-ss",
                                str(extended_start),
                                "-i",
                                str(job.source),
                                "-t",
                                str(extended_end - extended_start),
                                "-vn",
                                "-ac",
                                "1",
                                "-ar",
                                "16000",
                                "-c:a",
                                "pcm_s16le",
                                str(extended_wave),
                            ]
                        )
                    alternate = transcribe(
                        str(extended_wave),
                        path_or_hf_repo=str(job.model),
                        language="ja",
                        verbose=False,
                        condition_on_previous_text=False,
                        word_timestamps=False,
                    )
                    with recovered.open("x", encoding="utf-8") as stream:
                        json.dump(
                            {
                                "source_offset": extended_start,
                                "core_start": core,
                                "recognition": alternate,
                                "reason": "new context for unrenderable original native segment",
                            },
                            stream,
                            ensure_ascii=False,
                        )
                try:
                    cues = owned_segment_cues(
                        alternate,
                        offset=extended_start,
                        duration=extended_end - extended_start,
                        core=core,
                    )
                except UnrenderableSegment:
                    if after == 48:
                        raise
                else:
                    break
        all_cues.extend(cues)
        print(f"Recognized {index + 1}/{math.ceil(duration / 24)} source windows", flush=True)
    ordered = tuple(sorted(all_cues, key=lambda cue: (cue.start, cue.end)))
    if not ordered:
        raise ReplayError("No native captions; empty ASR does not prove absent speech")
    return render_subtitles(job, output=output, cues=ordered)


def render_subtitles(job: SubtitleJob, *, output: Path, cues: tuple[SpeakerCue, ...]) -> Path:
    """Shared non-overwriting burn-in/full-decode renderer.

    Captions are burned exactly once from ASS (no speaker names, bordered styles).
    No subtitle track is embedded, so players cannot display a second copy; the SRT
    sidecar remains for TranscriptViewer.
    """
    ordered = cues
    metadata = output / "metadata"
    if not ordered:
        raise ReplayError("Cannot render an empty subtitle selection")
    subtitle = output / f"{output.name}.srt"
    with subtitle.open("x", encoding="utf-8") as stream:
        stream.write(render_srt(ordered))
    burned = output / f"{output.name}.ass"
    with burned.open("x", encoding="utf-8") as stream:
        stream.write(render_ass(ordered))
    movie = output / f"{output.name}.mp4"
    _run(
        [
            str(job.ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(job.source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            f"ass={burned.name}",
            "-sn",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "8000k",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        cwd=output,
    )
    _run(
        [
            str(job.ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(movie),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ]
    )
    with (metadata / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "automatic-subtitles-native-segment-estimates",
                "cues": len(ordered),
                "movie": str(movie),
                "subtitle": str(subtitle),
                "burned_subtitle": str(burned),
                "keep_bgm": job.keep_bgm,
                "audio_policy": job.audio_policy,
                "audio": "selected input, AAC256k; no new separation/muting",
                "video": "H264 burn-in from ASS; no embedded subtitle track",
                "subtitle_mode": job.subtitle_mode,
                "full_decode": "passed",
                "limitations": (
                    "ASR estimates, not exact word onsets or verified speaker separation"
                ),
                "pending_audit": "context seam, repetition, source gap and subtitle display audit",
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="matching incomplete dated job; never a completed output"
    )
    args = parser.parse_args()
    print(run_job(load_job(args.config), resume=args.resume))


if __name__ == "__main__":
    main()
