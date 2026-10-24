"""Measure stereo PCM quiet and protect source-clock recognition when planning cuts."""

import argparse
import json
import os
import subprocess
import wave
from bisect import bisect_right
from dataclasses import asdict
from fractions import Fraction
from itertools import groupby
from math import ceil, isfinite
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.caption_projection import CutSpan, MeasuredQuietSpan
from replay.caption_repair import ClockLabel
from replay.core import ReplayError
from replay.delivery_media import read_clock
from replay.input_hash_snapshot import InputHashSnapshot
from replay.quiet_cuts import quiet_cut_frames
from replay.recordings import record
from replay.stages import file_digest
from replay.waveform_quiet import measured_quiet_runs

# The quiet peak threshold (the waveform_quiet default) for a stem written at the fixed
# extraction gain (separation_pipeline_prepare.GAIN_DB) and unit output gain.
QUIET_THRESHOLD_DBFS = -60.0
FIXED_EXTRACTION_GAIN_DB = -0.1
MODEL_INPUT_NAME = "model-input.wav"  # separation_pipeline_prepare's and delivery_audio's input


def jitter_gaps(clock: AudioFrameClock) -> dict[Fraction, Fraction]:
    """Gaps between decoded frames that are timestamp jitter: each frame end -> next start.

    A gap counts as jitter when it is shorter than half the frame before it and the clock's
    drift (PTS time against summed decoded samples) since the last real gap stays within one
    frame after it. Encoder rounding (2064 for 2048 samples) or 40/60 ms FLV steps for 42.67 ms
    frames qualify; a dropped frame leaves a gap of about one frame, stays a gap and becomes the
    reference the following drift is measured from.
    """
    origin = clock.pts[0] * clock.time_base
    reference = Fraction(0)
    gaps: dict[Fraction, Fraction] = {}
    for index in range(len(clock.pts) - 1):
        frame = Fraction(clock.samples[index], clock.sample_rate)
        end = clock.pts[index] * clock.time_base + frame
        start = clock.pts[index + 1] * clock.time_base
        drift = start - origin - Fraction(clock.sample_ends[index], clock.sample_rate)
        if start <= end:
            continue
        if start - end < frame / 2 and abs(drift - reference) < frame:
            gaps[end] = start
        else:
            reference = drift
    return gaps


def source_quiet(
    quiet: tuple[MeasuredQuietSpan, ...],
    clock: AudioFrameClock,
    *,
    measurement_rate: int | None = None,
) -> tuple[MeasuredQuietSpan, ...]:
    """Require every native owner to be measured quiet; gaps remain unclassified.

    Split at decoded-frame boundaries before mapping. Count decoded owners and
    individually quiet owners on exact rational intervals. An overlapping loud
    or unknown owner protects the interval; quiet owners never cancel it by an
    averaged waveform. Zero owners is missing support, never measured silence;
    only a timestamp-jitter gap (``jitter_gaps``), which holds no decoded sample at
    all, joins the quiet spans on both of its sides.
    """
    rate = clock.sample_rate if measurement_rate is None else measurement_rate
    if type(rate) is not int or not 8000 <= rate <= 192000:
        raise ReplayError("Invalid quiet measurement sample rate")
    events: list[tuple[Fraction, int, int]] = []
    for pts, samples in zip(clock.pts, clock.samples, strict=True):
        start = pts * clock.time_base
        end = start + Fraction(samples, clock.sample_rate)
        events.extend(((start, 1, 0), (end, -1, 0)))
    previous_quiet_end = 0.0
    for span in quiet:
        if span.start < previous_quiet_end:
            raise ReplayError("Quiet spans must be ordered and disjoint")
        previous_quiet_end = span.end
        # measured_quiet_runs emits exact sample-index boundaries as floats.
        # Recover those integer indices before rational cross-rate conversion.
        left = Fraction(round(span.start * rate), rate) * clock.sample_rate
        right = Fraction(round(span.end * rate), rate) * clock.sample_rate
        if right > clock.sample_ends[-1] + Fraction(clock.sample_rate, rate):
            raise ReplayError("Quiet span exceeds decoded sample support")
        # A duration-preserving resampler may finish with a fractional source
        # sample. That unsupported tail is omitted, never extended or stretched.
        right = min(right, Fraction(clock.sample_ends[-1]))
        index = min(bisect_right(clock.sample_ends, left), len(clock.pts) - 1)
        while left < right:
            prior = clock.sample_ends[index - 1] if index else 0
            stop = min(right, Fraction(clock.sample_ends[index]))
            origin = clock.pts[index] * clock.time_base
            start = origin + (left - prior) / clock.sample_rate
            end = origin + (stop - prior) / clock.sample_rate
            events.extend(((start, 0, 1), (end, 0, -1)))
            left = stop
            index += 1
    # Aggregate all simultaneous boundaries before evaluating the next half-open
    # interval. Merge exact touching support, not nearby float timestamps.
    result: list[tuple[Fraction, Fraction]] = []
    joins = jitter_gaps(clock)
    owners = quiet_owners = 0
    previous = Fraction(0)
    events.sort(key=lambda event: event[0])
    for point, group in groupby(events, key=lambda event: event[0]):
        if joins.get(previous) == point and owners == 0 and result and result[-1][1] == previous:
            # Nothing plays in a jitter gap, so it must not split one quiet run into frame-sized
            # pieces.
            result[-1] = (result[-1][0], point)
        if previous < point and owners > 0 and owners == quiet_owners:
            beginning = previous
            if result and result[-1][1] == previous:
                beginning = result.pop()[0]
            result.append((beginning, point))
        for _, owner_delta, quiet_delta in group:
            owners += owner_delta
            quiet_owners += quiet_delta
        previous = point
    return tuple(MeasuredQuietSpan(float(start), float(end)) for start, end in result)


def _read(path: Path) -> dict[str, object]:
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ReplayError("Planning JSON exceeds byte budget")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _binding(value: object) -> tuple[Path, str]:
    item = record(value)
    path, digest = item.get("path"), item.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ReplayError("Expected a pinned input artifact")
    return Path(path), digest


def prepare_config(template: Path, output: Path) -> None:
    """Pin completed artifacts from a data-only template at its workflow boundary.

    The template expresses roles and transform intent, not fabricated future
    digests. Missing inputs fail; this function neither waits nor starts producers.
    """
    if (
        output.exists()
        or output.is_symlink()
        or output.parent != output.parent.resolve(strict=True)
    ):
        raise ReplayError("Expected a new canonical configuration output")
    template_digest = file_digest(template)
    data = _read(template)
    kind = data.pop("prepareKind", None)
    fields = record(data.pop("artifactPaths", None))
    contracts = {
        "projection": {
            "labels",
            "pcm",
            "audioClock",
            "asrInputs",
            "clockReceipt",
            "pcmReceipt",
            "lineageEvidence",
        },
        "quiet": {
            "labels",
            "stereoPCM",
            "audioClock",
            "videoClock",
            "projectionConfig",
            "lineageEvidence",
        },
        "lineage": {"stereoPCM", "monoPCM", "audioClock"},
    }
    if not isinstance(kind, str) or kind not in contracts or set(fields) != contracts[kind]:
        raise ReplayError("Unsupported preparation roles")
    bindings: dict[str, dict[str, str]] = {}
    inputs = [(template, template_digest)]
    for name, value in fields.items():
        if not isinstance(value, str):
            raise ReplayError("Artifact role must name a canonical path")
        path = Path(value)
        if not path.is_absolute() or path != path.resolve(strict=True):
            raise ReplayError("Artifact role must name a canonical path")
        digest = file_digest(path)
        inputs.append((path, digest))
        bindings[name] = {"path": str(path), "sha256": digest}
    snapshot = InputHashSnapshot.capture(inputs)
    if kind == "lineage":
        data.update(bindings)
    else:
        data["artifacts"] = bindings
    data["templateSHA256"] = template_digest
    snapshot.finish()
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)


def _labels(data: dict[str, object]) -> tuple[ClockLabel, ...]:
    raw = data.get("labels")
    if (
        data.get("clock") != "source-presentation"
        or not isinstance(raw, list)
        or len(raw) > 100_000
    ):
        raise ReplayError("Expected source-presentation recognition")
    result: list[ClockLabel] = []
    for entry in raw:
        item = record(entry)
        text, provenance, start, end = (
            item.get(key) for key in ("text", "provenance", "start", "end")
        )
        if (
            not isinstance(text, str)
            or not isinstance(provenance, str)
            or isinstance(start, bool)
            or not isinstance(start, (float, int))
            or isinstance(end, bool)
            or not isinstance(end, (float, int))
        ):
            raise ReplayError("Invalid source recognition label")
        result.append(ClockLabel(text, float(start), float(end), provenance))
    return tuple(result)


def stem_gain_db(stereo: Path, digest: str) -> float:
    """The static gain separation applied to this PCM relative to the fixed-gain audio the quiet
    threshold was set on, from pinned records only; PCM without those records (made before the
    gains existed) has no offset.

    The quiet input is either the separated stem, or (older transcription flows) the separation
    model input itself. A stem carries the measured extraction gain beyond the fixed −0.1 dB plus
    the writer's output gain (its separation ``result.json`` must pin it); a model input carries
    only the extraction gain (its PCM preparation receipt must pin it)."""
    if stereo.name == MODEL_INPUT_NAME:
        return _extraction_gain(stereo.parent / "pcm-preparation.json", stereo, digest)
    result_path = stereo.parent / "result.json"
    if not result_path.exists():
        return 0.0
    result = _read(result_path)
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not any(
        record(item).get("path") == str(stereo) and record(item).get("sha256") == digest
        for item in outputs
    ):
        raise ReplayError("The separation receipt does not pin this stem")
    model_input = Path(str(result.get("modelInput")))
    receipt = model_input.parent / "pcm-preparation.json"
    return _number(result.get("outputGainDB", 0.0)) + _extraction_gain(receipt, model_input, None)


def _extraction_gain(receipt_path: Path, model_input: Path, digest: str | None) -> float:
    """The measured extraction gain beyond the fixed one, from the PCM preparation receipt's
    pinned ``input-gain.json``; 0 without a receipt or a measured gain. With ``digest`` the
    receipt must pin that model input."""
    if not receipt_path.exists():
        return 0.0
    receipt = _read(receipt_path)
    if digest is not None and record(receipt.get("output", {})).get("sha256") != digest:
        raise ReplayError(f"The PCM preparation receipt does not pin {model_input}")
    pinned = record(receipt.get("inputGain", {}))
    if not pinned:
        return 0.0
    gain_path = Path(str(pinned.get("path")))
    if file_digest(gain_path) != pinned.get("sha256"):
        raise ReplayError("The extraction gain record differs from its pin")
    return _number(_read(gain_path).get("gainDB")) - FIXED_EXTRACTION_GAIN_DB


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ReplayError("A recorded stem gain is not a finite number")
    return float(value)


def quiet_threshold_dbfs(stereo: Path, digest: str) -> float:
    """The quiet peak threshold, moved with the stem's recorded static gain so the same audio
    is quiet whatever level separation wrote it at."""
    return QUIET_THRESHOLD_DBFS + stem_gain_db(stereo, digest)


def run(config: Path, output: Path, *, ffmpeg: str) -> tuple[CutSpan, ...]:
    """Produce candidate cuts, never a non-speech or perceptual-quality certificate."""
    if (
        output.exists()
        or output.is_symlink()
        or output.parent != output.parent.resolve(strict=True)
    ):
        raise ReplayError("Expected a new canonical planning directory")
    config_digest = file_digest(config)
    data = _read(config)
    fps = data.get("fps")
    if (
        data.get("version") != 1
        or data.get("pcmClock") != "continuous-decoded-pcm"
        or type(fps) is not int
        or not 1 <= fps <= 120
    ):
        raise ReplayError("Invalid source-clock planning contract")
    raw = record(data.get("artifacts"))
    names = {
        "labels",
        "stereoPCM",
        "audioClock",
        "videoClock",
        "projectionConfig",
        "lineageEvidence",
    }
    if set(raw) != names:
        raise ReplayError("Incomplete quiet/recognition clock lineage")
    bindings = {name: _binding(value) for name, value in raw.items()}
    projection = _read(bindings["projectionConfig"][0])
    projection_inputs = record(projection.get("artifacts"))
    mono = _binding(projection_inputs.get("pcm"))
    snapshot = InputHashSnapshot.capture([(config, config_digest), mono, *bindings.values()])
    lineage = _read(bindings["lineageEvidence"][0])
    if (
        lineage.get("clock") != "continuous-decoded-pcm"
        or lineage.get("stereoPCM")
        != {"path": str(bindings["stereoPCM"][0]), "sha256": bindings["stereoPCM"][1]}
        or lineage.get("monoPCM") != {"path": str(mono[0]), "sha256": mono[1]}
        or lineage.get("audioClock")
        != {"path": str(bindings["audioClock"][0]), "sha256": bindings["audioClock"][1]}
        or lineage.get("transforms")
        != ["concatenate-decoded-frames", "duration-preserving-resample"]
    ):
        raise ReplayError("Missing same-clock stereo/recognition PCM manifest")
    labels = _read(bindings["labels"][0])
    if (
        labels.get("configSHA256") != bindings["projectionConfig"][1]
        or _binding(projection_inputs.get("audioClock")) != bindings["audioClock"]
        or record(labels.get("parentSHA256")).get(str(mono[0])) != mono[1]
    ):
        raise ReplayError("Quiet and recognition do not bind the same decoded clock")
    clock = AudioFrameClock.from_probe(_read(bindings["audioClock"][0]))
    with wave.open(str(bindings["stereoPCM"][0]), "rb") as stream:
        samples, rate = stream.getnframes(), stream.getframerate()
        if (
            stream.getnchannels() != 2
            or stream.getsampwidth() != 2
            or stream.getcomptype() != "NONE"
            or not 8000 <= rate <= 192000
            or not 0 < samples <= rate * 21600
            or abs(Fraction(samples, rate) - Fraction(clock.sample_ends[-1], clock.sample_rate))
            > Fraction(1, rate)
        ):
            raise ReplayError("Quiet input must preserve stereo decoded-sample duration")
    video = read_clock(bindings["videoClock"][0])
    source_frames = ceil((video.pts[-1] + video.last_duration) * video.time_base * fps)
    output.mkdir(mode=0o700)
    peaks = output / "peaks.txt"
    window = max(1, rate // 50)
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(bindings["stereoPCM"][0]),
        "-af",
        f"asetnsamples=n={window}:p=0,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.Peak_level:file=-",
        "-f",
        "null",
        "-",
    ]
    with peaks.open("xb") as stdout, (output / "ffmpeg.stderr.log").open("xb") as stderr:
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=True,
            timeout=3600,
        )
    threshold = quiet_threshold_dbfs(*bindings["stereoPCM"])
    with peaks.open(encoding="utf-8") as lines:
        quiet = measured_quiet_runs(
            lines, sample_count=samples, sample_rate=rate, window_samples=window,
            threshold_dbfs=threshold,
        )  # fmt: skip
    mapped = source_quiet(quiet, clock, measurement_rate=rate)
    cuts = quiet_cut_frames(mapped, _labels(labels), source_frames=source_frames, fps=fps)
    snapshot.finish()
    result = {
        "version": 1,
        "clock": "source-presentation",
        "fps": fps,
        "sourceFrames": source_frames,
        "quietThresholdDBFS": threshold,
        "cuts": [asdict(cut) for cut in cuts],
        "sourceQuiet": [asdict(span) for span in mapped],
        "parentSHA256": {str(path): digest for path, digest in bindings.values()},
        "configSHA256": config_digest,
        "peakSHA256": file_digest(peaks),
        "plannerSHA256": file_digest(Path(__file__)),
        "command": command,
        "lineageIsCallerDeclared": True,
        "nonSpeechRemovalVerified": False,
        "humanReviewed": False,
        "deliveryComplete": False,
    }
    with (output / "cuts.json").open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return cuts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        prepare_config(args.config, args.output)
    else:
        if not isinstance(args.ffmpeg, str):
            raise ReplayError("Planning requires an explicit FFmpeg executable")
        run(args.config, args.output, ffmpeg=args.ffmpeg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
