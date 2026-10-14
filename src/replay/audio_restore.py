"""Create a verified, voice-preserving BGM-reduced delivery revision.

Run as ``python -m replay.audio_restore INPUT.mp4 NEW_REVISION_DIR``. Source
and destination must be under this checkout's ``edits`` tree; the revision
folder must be new. Work WAVs and the machine-readable quality report stay
beside the delivery so the processing is reproducible and auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.audio_expander import (
    ExpanderPlanDict,
    ExpanderPolicy,
    ExpansionVerificationDict,
    plan_from_texts,
    transfer_points,
    verify_from_texts,
)
from replay.bgm_separation import (
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    MODEL_SHA256,
    SeparationConfig,
    SeparationPolicy,
    SpeechQuality,
    _load_backend,
    _read_wav,
    _write_wav,
    adapt_vocal_mix,
    apply_wiener_vocal_mask,
    measure_speech_quality,
    resolve_device,
    separate_vocals,
    signal_level_dbfs,
    verify_music_removal,
)
from replay.core import ReplayError
from replay.delivery import (
    delivery_lock,
    delivery_stem,
    require_new_delivery,
    retire_processed_source,
)

MINIMUM_SPEECH_CORRELATION = 0.995
MUSIC_PROBE_SECONDS = 60
VOICE_OUTPUT_GAIN_DB = 0.5
EXPANDER_POLICY = ExpanderPolicy(maximum_threshold_above_floor_db=60.0)
SPEECH_PADDING_SECONDS = 0.5
MAXIMUM_SRT_CUES = 100_000
_SRT_RANGE = re.compile(
    r"(?P<sh>[0-9]+):(?P<sm>[0-5][0-9]):(?P<ss>[0-5][0-9]),(?P<sms>[0-9]{3})"
    r"\s+-->\s+"
    r"(?P<eh>[0-9]+):(?P<em>[0-5][0-9]):(?P<es>[0-5][0-9]),(?P<ems>[0-9]{3})"
)


@dataclass(frozen=True)
class ReusedSeparation:
    """A published listening-approved delivery whose vocals can be reused."""

    delivery: Path
    realtime_factor: object


@dataclass(frozen=True)
class MusicProbe:
    """A measured instrumental level on one deterministic in-video window."""

    start_seconds: float
    before_dbfs: float
    after_dbfs: float | None = None
    drop_db: float | None = None
    passed: bool | None = None
    reason: str | None = None


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        raise ReplayError(f"Cannot run {arguments[0]}; install FFmpeg and ffprobe") from error
    if completed.returncode:
        detail = completed.stderr.strip()[-500:]
        raise ReplayError(f"{arguments[0]} failed: {detail or completed.returncode}")
    return completed


def _video_frames(path: Path) -> int:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        frames = int(completed.stdout.strip())
    except ValueError as error:
        raise ReplayError(f"Cannot determine video frame count: {path}") from error
    if frames <= 0:
        raise ReplayError(f"Video has no decoded frames: {path}")
    return frames


def _validate_paths(
    source: Path,
    revision: Path,
    repository_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    repository = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    edits = (repository / "edits").resolve()
    source = source.resolve(strict=True)
    revision = revision.resolve()
    try:
        source.relative_to(edits)
        revision.relative_to(edits)
    except ValueError as error:
        raise ReplayError(
            "Input and output must stay inside this checkout's edits directory"
        ) from error
    if source.name != "full-review.mp4" or source.parent.name != "delivery":
        raise ReplayError("Input must be a processed delivery/full-review.mp4")
    revisions = next((parent for parent in source.parents if parent.name == "revisions"), None)
    if revisions is None or revision.parent != revisions:
        raise ReplayError("New revision must be a sibling of the input revision")
    if revision.exists() or revision.is_symlink():
        raise ReplayError(f"Refusing to overwrite existing revision: {revision}")
    stem = require_new_delivery(revisions.parent)
    if not revision.name.startswith(stem + "-"):
        revision = revision.with_name(f"{stem}-{revision.name}")
    if revision.exists() or revision.is_symlink():
        raise ReplayError(f"Refusing to overwrite existing revision: {revision}")
    return repository, source, revision


def _loudness_log(path: Path) -> str:
    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ]
    )
    return completed.stderr or ""


def _expander_filter(threshold_db: float, floor_db: float) -> tuple[str, float]:
    pre_gain_db = max(0.0, -69.0 - threshold_db, -75.0 - floor_db)
    try:
        points = transfer_points(threshold_db + pre_gain_db)
    except ValueError as error:
        raise ReplayError(f"Expander threshold is invalid: {error}") from error
    return (
        f"volume={pre_gain_db:.2f}dB,compand=attacks=0.02:decays=0.25:points={points},volume=-{pre_gain_db:.2f}dB",
        pre_gain_db,
    )


def _apply_downward_expander(
    source: Path,
    destination: Path,
    subtitle_text: str,
    work: Path,
) -> tuple[ExpanderPlanDict, ExpansionVerificationDict, float]:
    before_log = _loudness_log(source)
    (work / "expander-before.log").write_text(before_log, encoding="utf-8")
    try:
        plan = plan_from_texts(before_log, subtitle_text, EXPANDER_POLICY)
    except ValueError as error:
        raise ReplayError(f"Cannot derive an expander plan: {error}") from error
    if not plan["feasible"]:
        return (
            plan,
            {
                "passed": False,
                "suppressionDb": 0.0,
                "speechLossDb": 0.0,
                "reason": plan["reason"],
            },
            0.0,
        )
    expander_filter, pre_gain_db = _expander_filter(plan["thresholdDb"], plan["floorDb"])
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-af",
            expander_filter,
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(MODEL_SAMPLE_RATE),
            str(destination),
        ]
    )
    after_log = _loudness_log(destination)
    (work / "expander-after.log").write_text(after_log, encoding="utf-8")
    try:
        verification = verify_from_texts(before_log, after_log, subtitle_text, EXPANDER_POLICY)
    except ValueError as error:
        raise ReplayError(f"Cannot verify expander output: {error}") from error
    return plan, verification, pre_gain_db


def _speech_mask(subtitle: Path, samples: int) -> NDArray[np.bool_]:
    try:
        text = subtitle.read_text(encoding="utf-8").replace("\r\n", "\n")
    except (OSError, UnicodeError) as error:
        raise ReplayError(f"Cannot use transcript for speech protection: {subtitle}") from error
    mask = np.zeros(samples, dtype=np.bool_)
    cue_count = 0
    for block in re.split(r"\n\s*\n", text.strip()):
        rows = [row.strip() for row in block.splitlines() if row.strip()]
        if len(rows) < 2:
            continue
        match = _SRT_RANGE.fullmatch(rows[1][:128])
        if not rows[0].isdigit() or match is None:
            raise ReplayError(f"Cannot use transcript for speech protection: {subtitle}")
        values = match.groupdict()
        start = (
            int(values["sh"]) * 3600
            + int(values["sm"]) * 60
            + int(values["ss"])
            + int(values["sms"]) / 1000.0
        )
        end = (
            int(values["eh"]) * 3600
            + int(values["em"]) * 60
            + int(values["es"])
            + int(values["ems"]) / 1000.0
        )
        if end <= start:
            raise ReplayError(f"Cannot use transcript for speech protection: {subtitle}")
        cue_count += 1
        if cue_count > MAXIMUM_SRT_CUES:
            raise ReplayError("Transcript cue count exceeds processing limits")
        first = max(0, min(samples, int((start - SPEECH_PADDING_SECONDS) * MODEL_SAMPLE_RATE)))
        last = max(0, min(samples, int((end + SPEECH_PADDING_SECONDS) * MODEL_SAMPLE_RATE)))
        if last > first:
            mask[first:last] = True
    if cue_count == 0:
        raise ReplayError(f"Cannot use transcript for speech protection: {subtitle}")
    if not mask.any():
        raise ReplayError("Transcript contains no speech overlapping the audio")
    return mask


def _probe_starts(samples: int) -> tuple[int, ...]:
    """Tile the full output with non-overlapping 60-second music windows."""
    return tuple(range(0, samples, MUSIC_PROBE_SECONDS * MODEL_SAMPLE_RATE))


def _measure_music(
    audio: NDArray[np.float32],
    vocals: NDArray[np.float32],
) -> tuple[MusicProbe, ...]:
    if audio.shape != vocals.shape:
        raise ReplayError("Vocal estimate does not align with audio")
    window_samples = MUSIC_PROBE_SECONDS * MODEL_SAMPLE_RATE
    probes: list[MusicProbe] = []
    for start in _probe_starts(audio.shape[1]):
        end = min(start + window_samples, audio.shape[1])
        residual = audio[:, start:end] - vocals[:, start:end]
        probes.append(
            MusicProbe(
                start_seconds=start / MODEL_SAMPLE_RATE,
                before_dbfs=signal_level_dbfs(residual),
            )
        )
    return tuple(probes)


def _verify_probes(
    before: tuple[MusicProbe, ...],
    after: tuple[MusicProbe, ...],
    speech: SpeechQuality,
    policy: SeparationPolicy,
) -> tuple[MusicProbe, ...]:
    if len(before) != len(after):
        raise ReplayError("Full-output music windows do not align")
    verified: list[MusicProbe] = []
    for source, cleaned in zip(before, after, strict=True):
        if source.start_seconds != cleaned.start_seconds:
            raise ReplayError("Full-output music windows are out of order")
        after_dbfs = cleaned.before_dbfs
        if source.before_dbfs <= policy.quiet_instrumental_db:
            if after_dbfs <= policy.quiet_instrumental_db:
                verified.append(
                    MusicProbe(
                        start_seconds=source.start_seconds,
                        before_dbfs=source.before_dbfs,
                        after_dbfs=after_dbfs,
                        drop_db=source.before_dbfs - after_dbfs,
                        passed=None,
                        reason="Both source and output are below the quiet threshold",
                    )
                )
            else:
                verified.append(
                    MusicProbe(
                        start_seconds=source.start_seconds,
                        before_dbfs=source.before_dbfs,
                        after_dbfs=after_dbfs,
                        drop_db=source.before_dbfs - after_dbfs,
                        passed=False,
                        reason="Output introduced instrumental content in a quiet source window",
                    )
                )
            continue
        check = verify_music_removal(
            source.before_dbfs,
            after_dbfs,
            0.0,
            speech.level_delta_db_p50,
            policy,
        )
        verified.append(
            MusicProbe(
                start_seconds=source.start_seconds,
                before_dbfs=source.before_dbfs,
                after_dbfs=after_dbfs,
                drop_db=check.instrumental_drop_db,
                passed=check.passed,
                reason=check.reason,
            )
        )
    return tuple(verified)


def _select_expander(
    source_probes: tuple[MusicProbe, ...],
    before: NDArray[np.float32],
    expanded: NDArray[np.float32],
    speech_mask: NDArray[np.bool_],
    verification: ExpansionVerificationDict,
    policy: SeparationPolicy,
) -> tuple[bool, SpeechQuality | None]:
    """Keep an expander only if it preserves speech in this video's quiet windows.

    Quiet source windows are the available speech control: loud music in the
    transcript otherwise looks like a change in voice level. No quiet control
    means the expander has not demonstrated voice safety and is skipped.
    """
    if not verification["passed"]:
        return False, None
    quiet_speech = np.zeros_like(speech_mask)
    window_samples = MUSIC_PROBE_SECONDS * MODEL_SAMPLE_RATE
    for probe in source_probes:
        if probe.before_dbfs <= policy.quiet_instrumental_db:
            start = int(probe.start_seconds * MODEL_SAMPLE_RATE)
            end = min(start + window_samples, speech_mask.size)
            quiet_speech[start:end] = speech_mask[start:end]
    if not quiet_speech.any():
        return False, None
    quality = measure_speech_quality(before, expanded, quiet_speech)
    return (
        quality.level_delta_db_p05 >= -policy.maximum_speech_loss_db
        and quality.correlation >= MINIMUM_SPEECH_CORRELATION,
        quality,
    )


def _timing_report(started: float, audio_seconds: float) -> dict[str, float]:
    elapsed = time.perf_counter() - started
    return {
        "processing_wall_seconds": elapsed,
        "processing_minutes_per_audio_hour": elapsed * 60.0 / audio_seconds,
    }


def _save_report(path: Path, report: dict[str, object]) -> None:
    if path.exists():
        raise ReplayError(f"Refusing to overwrite verification report: {path}")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def restore_delivery(
    source_path: Path,
    revision_path: Path,
    device: str = "auto",
) -> dict[str, object]:
    """Serialize publication across all revision names for the same source."""
    _, source, revision = _validate_paths(source_path, revision_path)
    with delivery_lock(revision.parent.parent):
        before = source.stat()
        report = _restore_delivery(source, revision, device)
        if report["status"] == "verified":
            output = revision / "delivery" / f"{delivery_stem(revision.parent.parent)}.mp4"
            retire_processed_source(source, output, before)
        return report


def _restore_delivery(
    source_path: Path,
    revision_path: Path,
    device: str,
) -> dict[str, object]:
    """Create a new revision only when full speech and music gates pass."""
    repository, source, revision = _validate_paths(source_path, revision_path)
    subtitle = source.with_suffix(".srt")
    if not subtitle.is_file():
        raise ReplayError(f"Matching SRT sidecar is required: {subtitle}")
    try:
        subtitle.resolve(strict=True).relative_to((repository / "edits").resolve())
    except (OSError, ValueError) as error:
        raise ReplayError("Transcript sidecar must remain inside edits") from error
    original_frames = _video_frames(source)
    revision.mkdir()
    work = revision / "work"
    delivery = revision / "delivery"
    metadata = revision / "metadata"
    for directory in (work, delivery, metadata):
        directory.mkdir()

    started = time.perf_counter()
    input_wav = work / "input.wav"
    vocals_wav = work / "mlx-vocals.wav"
    encoded_vocals_wav = work / "encoded-vocals.wav"
    expander_input_wav = work / "expander-input.wav"
    expanded_wav = work / "expanded-audio.wav"
    temporary_video = work / source.name
    output_video = delivery / f"{delivery_stem(revision.parent.parent)}.mp4"
    policy = SeparationPolicy()
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(MODEL_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(input_wav),
        ]
    )
    original, channels = _read_wav(input_wav)
    if channels != 2 or original.shape[1] == 0:
        raise ReplayError("FFmpeg failed to produce non-empty stereo separation input")
    audio_seconds = original.shape[1] / MODEL_SAMPLE_RATE
    mask = _speech_mask(subtitle, original.shape[1])
    config = SeparationConfig(device=device)
    backend = _load_backend(config.model_name, resolve_device(config.device, config.model_name))
    separation = separate_vocals(input_wav, vocals_wav, config, _backend=backend)
    vocals, vocal_channels = _read_wav(vocals_wav)
    if vocal_channels != channels or vocals.shape != original.shape:
        raise ReplayError("Separated vocals do not align with source audio")
    before = _measure_music(original, vocals)
    positive = tuple(probe for probe in before if probe.before_dbfs > policy.quiet_instrumental_db)
    if not positive:
        report: dict[str, object] = {
            "status": "no-instrumental-content",
            "source": str(source.relative_to(repository)),
            "video_frames": original_frames,
            "speech_mask_fraction": float(np.mean(mask)),
            "music_windows_per_video": len(before),
            "music_probes": [asdict(probe) for probe in before],
            **_timing_report(started, audio_seconds),
        }
        _save_report(metadata / "verification.json", report)
        return report

    masked = apply_wiener_vocal_mask(original, vocals)
    adaptive = adapt_vocal_mix(vocals, masked, mask)
    voice_gain = np.float32(10.0 ** (VOICE_OUTPUT_GAIN_DB / 20.0))
    output_audio = adaptive.audio * voice_gain
    speech_loss = max(0.0, -adaptive.speech_level_delta_db)
    if (
        speech_loss > policy.maximum_speech_loss_db
        or adaptive.lower_tail_speech_delta_db < -policy.maximum_speech_loss_db - 0.02
        or adaptive.speech_correlation < MINIMUM_SPEECH_CORRELATION
    ):
        _save_report(
            metadata / "verification.json",
            {
                "status": "rejected-adaptive-speech-quality",
                "source": str(source.relative_to(repository)),
                "speech_level_delta_db_p50": adaptive.speech_level_delta_db,
                "speech_level_delta_db_p05": adaptive.lower_tail_speech_delta_db,
                "speech_correlation": adaptive.speech_correlation,
                "policy": asdict(policy),
                **_timing_report(started, audio_seconds),
            },
        )
        raise ReplayError("Full-video speech quality gate failed; no delivery was published")
    _write_wav(expander_input_wav, output_audio)
    expander_plan, expander_verification, expander_pre_gain_db = _apply_downward_expander(
        expander_input_wav,
        expanded_wav,
        subtitle.read_text(encoding="utf-8"),
        work,
    )
    encoded_input = expander_input_wav
    expander_applied = False
    quiet_speech_quality: SpeechQuality | None = None
    if expander_plan["feasible"]:
        expanded_audio, expanded_channels = _read_wav(expanded_wav)
        if expanded_channels != channels or expanded_audio.shape != original.shape:
            raise ReplayError("Expander output does not align with source audio")
        expander_applied, quiet_speech_quality = _select_expander(
            before, output_audio, expanded_audio, mask, expander_verification, policy
        )
        if expander_applied:
            encoded_input = expanded_wav
    expander_report: dict[str, object] = {
        "expander_plan": expander_plan,
        "expander_verification": expander_verification,
        "expander_passed": expander_verification["passed"],
        "expander_applied": expander_applied,
        "expander_quiet_speech_quality": (
            asdict(quiet_speech_quality) if quiet_speech_quality is not None else None
        ),
        "expander_pre_gain_db": expander_pre_gain_db,
        "expander_policy": asdict(EXPANDER_POLICY),
    }
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(source),
            "-i",
            str(encoded_input),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            "-movflags",
            "+faststart",
            str(temporary_video),
        ]
    )
    encoded_audio_wav = work / "encoded-audio.wav"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(temporary_video),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(MODEL_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(encoded_audio_wav),
        ]
    )
    encoded_audio, encoded_channels = _read_wav(encoded_audio_wav)
    if encoded_channels != channels or encoded_audio.shape != original.shape:
        raise ReplayError("AAC output audio does not align with source audio")
    output_separation = separate_vocals(
        encoded_audio_wav, encoded_vocals_wav, config, _backend=backend
    )
    encoded_vocals, output_vocal_channels = _read_wav(encoded_vocals_wav)
    if output_vocal_channels != channels or encoded_vocals.shape != vocals.shape:
        raise ReplayError("Separated output vocals do not align with the source")
    speech_quality = measure_speech_quality(vocals, encoded_vocals, mask)
    after = _measure_music(encoded_audio, encoded_vocals)
    verified = _verify_probes(before, after, speech_quality, policy)
    failed_music = next((probe for probe in verified if probe.passed is False), None)
    speech_loss = max(0.0, -speech_quality.level_delta_db_p50)
    if (
        speech_loss > policy.maximum_speech_loss_db
        or speech_quality.level_delta_db_p05 < -policy.maximum_speech_loss_db - 0.02
        or speech_quality.correlation < MINIMUM_SPEECH_CORRELATION
    ):
        _save_report(
            metadata / "verification.json",
            {
                "status": "rejected-encoded-speech-quality",
                "source": str(source.relative_to(repository)),
                "speech_mask_fraction": adaptive.protected_fraction,
                "pre_aac_speech_level_delta_db_p50": adaptive.speech_level_delta_db,
                "pre_aac_speech_level_delta_db_p05": adaptive.lower_tail_speech_delta_db,
                "mean_vocal_blend_strength": adaptive.mean_strength,
                "output_voice_gain_db": VOICE_OUTPUT_GAIN_DB,
                "music_gate_passed": failed_music is None,
                "music_windows_per_video": len(verified),
                "music_probes": [asdict(probe) for probe in verified],
                "speech_level_delta_db_p50": speech_quality.level_delta_db_p50,
                "speech_level_delta_db_p05": speech_quality.level_delta_db_p05,
                "speech_correlation": speech_quality.correlation,
                "policy": asdict(policy),
                **expander_report,
                **_timing_report(started, audio_seconds),
            },
        )
        raise ReplayError(
            "Encoded full-video speech quality gate failed; no delivery was published"
        )
    if failed_music is not None:
        _save_report(
            metadata / "verification.json",
            {
                "status": "rejected-music-quality",
                "source": str(source.relative_to(repository)),
                "speech_level_delta_db_p50": speech_quality.level_delta_db_p50,
                "speech_level_delta_db_p05": speech_quality.level_delta_db_p05,
                "speech_correlation": speech_quality.correlation,
                "mean_vocal_blend_strength": adaptive.mean_strength,
                "output_voice_gain_db": VOICE_OUTPUT_GAIN_DB,
                "music_probes": [asdict(probe) for probe in verified],
                "policy": asdict(policy),
                **expander_report,
                **_timing_report(started, audio_seconds),
            },
        )
        raise ReplayError(
            f"Music verification failed at {failed_music.start_seconds:.1f}s: {failed_music.reason}"
        )
    output_subtitle = output_video.with_suffix(".srt")
    if output_subtitle.exists():
        raise ReplayError(f"Refusing to overwrite subtitle: {output_subtitle}")
    output_subtitle.write_bytes(subtitle.read_bytes())
    delivered_frames = _video_frames(temporary_video)
    if delivered_frames != original_frames:
        raise ReplayError("Video frame count changed during audio-only revision")
    timing = _timing_report(started, audio_seconds)
    if timing["processing_minutes_per_audio_hour"] > 30.0:
        _save_report(
            metadata / "verification.json",
            {
                "status": "rejected-runtime",
                "source": str(source.relative_to(repository)),
                "speech_level_delta_db_p50": speech_quality.level_delta_db_p50,
                "speech_level_delta_db_p05": speech_quality.level_delta_db_p05,
                "speech_correlation": speech_quality.correlation,
                "music_probes": [asdict(probe) for probe in verified],
                "policy": asdict(policy),
                **expander_report,
                **timing,
            },
        )
        raise ReplayError("Processing exceeded the 30-minutes-per-audio-hour limit")
    if output_video.exists():
        raise ReplayError(f"Refusing to overwrite video delivery: {output_video}")
    os.link(temporary_video, output_video)
    report = {
        "status": "verified",
        "source": str(source.relative_to(repository)),
        "delivery": str(output_video.relative_to(repository)),
        "audio_seconds": audio_seconds,
        **timing,
        "model": config.model_name,
        "separator_realtime_factor_source": separation.realtime_factor,
        "separator_realtime_factor_output": output_separation.realtime_factor,
        "mean_vocal_blend_strength": adaptive.mean_strength,
        "output_voice_gain_db": VOICE_OUTPUT_GAIN_DB,
        "pre_aac_speech_level_delta_db_p50": adaptive.speech_level_delta_db,
        "pre_aac_speech_level_delta_db_p05": adaptive.lower_tail_speech_delta_db,
        "speech_mask_fraction": adaptive.protected_fraction,
        "speech_level_delta_db_p50": speech_quality.level_delta_db_p50,
        "speech_level_delta_db_p05": speech_quality.level_delta_db_p05,
        "speech_correlation": speech_quality.correlation,
        "music_gate_passed": True,
        "music_windows_per_video": len(verified),
        "music_probes": [asdict(probe) for probe in verified],
        **expander_report,
        "video_frames": delivered_frames,
        "policy": asdict(policy),
    }
    _save_report(metadata / "verification.json", report)
    return report


def apply_listening_approved(
    source_path: Path,
    revision_path: Path,
    device: str = "auto",
) -> dict[str, object]:
    """Use the same per-source lock as the strict restoration workflow."""
    _, source, revision = _validate_paths(source_path, revision_path)
    with delivery_lock(revision.parent.parent):
        before = source.stat()
        report = _apply_listening_approved(source, revision, device)
        output = revision / "delivery" / f"{delivery_stem(revision.parent.parent)}.mp4"
        retire_processed_source(source, output, before)
        return report


def _apply_listening_approved(
    source_path: Path,
    revision_path: Path,
    device: str,
) -> dict[str, object]:
    """Create a separate, explicitly listening-approved revision without numeric music claims.

    Only raw pinned MLX vocals are used. Full decode, duration, frame count, and
    measured end-to-end throughput must pass before the new video is published.
    The original strict BGM/speech gates remain unchanged in restore_delivery.
    """
    repository, source, revision = _validate_paths(source_path, revision_path)
    subtitle = source.with_suffix(".srt")
    if not subtitle.is_file():
        raise ReplayError(f"Matching SRT sidecar is required: {subtitle}")
    try:
        subtitle.resolve(strict=True).relative_to((repository / "edits").resolve())
    except (OSError, ValueError) as error:
        raise ReplayError("Transcript sidecar must remain inside edits") from error
    frames = _video_frames(source)
    config = SeparationConfig(device=device)
    reused = _reusable_separation(repository, source, revision.parent)
    revision.mkdir()
    work = revision / "work"
    delivery = revision / "delivery"
    metadata = revision / "metadata"
    for directory in (work, delivery, metadata):
        directory.mkdir()
    started = time.perf_counter()
    input_wav = work / "input.wav"
    vocals_wav = work / "mlx-vocals.wav"
    temporary_video = work / source.name
    output_video = delivery / f"{delivery_stem(revision.parent.parent)}.mp4"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(MODEL_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(input_wav),
        ]
    )
    original, channels = _read_wav(input_wav)
    if channels != 2:
        raise ReplayError("FFmpeg failed to produce stereo MLX input")
    audio_seconds = original.shape[1] / MODEL_SAMPLE_RATE
    del original
    if reused is None:
        backend = _load_backend(config.model_name, resolve_device(config.device, config.model_name))
        inference = separate_vocals(input_wav, vocals_wav, config, _backend=backend)
        if abs(inference.audio_seconds - audio_seconds) > 1 / MODEL_SAMPLE_RATE:
            raise ReplayError("MLX vocals do not match the input duration")
        realtime_factor: object = inference.realtime_factor
        vocals_input, audio_codec = vocals_wav, ["-c:a", "aac", "-b:a", "320k"]
    else:
        # The same pinned model already separated byte-identical audio; its vocals
        # are reused as published instead of separating them again.
        vocals_input, audio_codec = reused.delivery, ["-c:a", "copy"]
        realtime_factor = reused.realtime_factor
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-n",
            "-i",
            str(source),
            "-i",
            str(vocals_input),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "0",
            "-c:v",
            "copy",
            *audio_codec,
            "-movflags",
            "+faststart",
            str(temporary_video),
        ]
    )
    duration_text = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(temporary_video),
        ]
    ).stdout.strip()
    try:
        duration = float(duration_text)
    except ValueError as error:
        raise ReplayError("Cannot determine revised AAC duration") from error
    if not np.isfinite(duration) or abs(duration - audio_seconds) > 0.1:
        raise ReplayError("Revised AAC duration does not match the source audio")
    if _video_frames(temporary_video) != frames:
        raise ReplayError("Video frame count changed during audio-only revision")
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(temporary_video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    timing = _timing_report(started, audio_seconds)
    report: dict[str, object] = {
        "status": "listening-approved"
        if timing["processing_minutes_per_audio_hour"] < 30.0
        else "rejected-runtime",
        "source": str(source.relative_to(repository)),
        "model": f"{config.model_name}@{MODEL_REVISION}",
        "model_sha256": MODEL_SHA256,
        "audio_seconds": audio_seconds,
        "aac_duration_seconds": duration,
        "video_frames": frames,
        "separator_realtime_factor": realtime_factor,
        "separation_reused_from": (
            str(reused.delivery.relative_to(repository)) if reused is not None else None
        ),
        "numeric_music_and_speech_gates": "not asserted; user approved MLX by listening",
        "full_audio_video_decode": "passed",
        **timing,
    }
    if report["status"] == "rejected-runtime":
        _save_report(metadata / "verification.json", report)
        raise ReplayError("Processing exceeded the 30-minutes-per-audio-hour limit")
    output_subtitle = output_video.with_suffix(".srt")
    output_subtitle.write_bytes(subtitle.read_bytes())
    os.link(temporary_video, output_video)
    report["delivery"] = str(output_video.relative_to(repository))
    _save_report(metadata / "verification.json", report)
    return report


def _audio_fingerprint(path: Path) -> str:
    """MD5 of the first audio stream's packets, independent of the container."""
    result = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-f",
            "md5",
            "-",
        ]
    ).stdout.strip()
    if not result.startswith("MD5="):
        raise ReplayError(f"Cannot fingerprint audio packets: {path}")
    return result


def _reusable_separation(
    repository: Path, source: Path, revisions: Path
) -> ReusedSeparation | None:
    """An earlier listening-approved delivery separated from byte-identical audio.

    The earlier render keeps voice.m4a, the exact audio track it muxed into its
    full-review.mp4, so its packets identify the separated input.
    """
    fingerprint: str | None = None
    for path in sorted(revisions.glob("*/metadata/verification.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        delivery, earlier = report.get("delivery"), report.get("source")
        if (
            report.get("status") != "listening-approved"
            or report.get("full_audio_video_decode") != "passed"
            or report.get("model_sha256") != MODEL_SHA256
            or not isinstance(delivery, str)
            or not isinstance(earlier, str)
        ):
            continue
        published = repository / delivery
        voice = (repository / earlier).parent.parent / "voice.m4a"
        if not published.is_file() or not voice.is_file():
            continue
        fingerprint = fingerprint or _audio_fingerprint(source)
        if _audio_fingerprint(voice) == fingerprint:
            return ReusedSeparation(published, report.get("separator_realtime_factor"))
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="processed delivery/full-review.mp4")
    parser.add_argument("revision", type=Path, help="new sibling revision directory")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--listening-approved",
        action="store_true",
        help="publish a separately labeled listening-approved MLX vocal revision",
    )
    arguments = parser.parse_args(argv)
    try:
        processor = apply_listening_approved if arguments.listening_approved else restore_delivery
        report = processor(arguments.source, arguments.revision, arguments.device)
    except (OSError, ValueError, ReplayError) as error:
        print(f"audio restore stopped safely: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
