"""Run a pinned native-clock PCM selection; never pad audio to the video duration."""

import argparse
import wave
from fractions import Fraction
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.caption_measured_job import _json, _local, _object, _pin, _rational
from replay.caption_native_layout import NativeCaptionLayout
from replay.continuous_cuts import retained_pcm_spans, write_pcm_spans
from replay.foreground_evidence import _bound_bytes
from replay.stages import file_digest
from replay.tracking_observations import _digest

MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_CLOCK_BYTES = 128 * 1024 * 1024
MAX_PCM_BYTES = 12 * 1024**3
MAX_EVIDENCE = 128
MAX_INTERVALS = 100_000
CODE_FILES = (
    "continuous_cut_job.py",
    "continuous_cuts.py",
    "audio_clock.py",
    "caption_measured_job.py",
    "caption_io.py",
    "caption_native_layout.py",
    "tracking_observations.py",
    "foreground_evidence.py",
    "foreground_observations.py",
    "stages.py",
)


def select(config: Path, output: Path, *, sha256: str) -> Path:
    """Keep all supplied provenance and actual sample/duration differences.

    Hashes bind inputs, not their semantic correspondence. Callers must separately
    verify that the waveform, decoded clock and intervals describe the same source,
    independently compare selected bytes and verify final A/V synchronization.
    """
    config, output = _local(str(config)), _local(str(output))
    if output.exists():
        raise FileExistsError(output)
    data = _object(
        _json(_bound_bytes(config, sha256, MAX_CONFIG_BYTES)),
        "version coordinateSpace source clock intervals evidence",
    )
    if (
        type(data["version"]) is not int
        or data["version"] != 1
        or (data["coordinateSpace"] != "native-presentation-seconds")
    ):
        raise ValueError("Expected version1 native presentation intervals")
    bindings = {str(config): sha256}
    source_ref = _object(data["source"], "path sha256")
    source = _local(source_ref["path"])
    source_hash = _digest(source_ref["sha256"])
    if not 0 < source.stat().st_size <= MAX_PCM_BYTES or file_digest(source) != source_hash:
        raise ValueError("Pinned PCM source changed or exceeds its byte budget")
    bindings[str(source)] = source_hash
    clock_path = _pin(data["clock"], bindings)
    clock = AudioFrameClock.from_probe(
        _json(_bound_bytes(clock_path, bindings[str(clock_path)], MAX_CLOCK_BYTES))
    )
    evidence = data["evidence"]
    if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE:
        raise ValueError("Expected bounded PCM provenance")
    for reference in evidence:
        _pin(reference, bindings)
    raw_intervals = data["intervals"]
    if not isinstance(raw_intervals, list) or not 1 <= len(raw_intervals) <= MAX_INTERVALS:
        raise ValueError("Expected bounded nonempty native intervals")
    intervals: list[tuple[Fraction, Fraction]] = []
    for value in raw_intervals:
        interval = _object(value, "start end")
        intervals.append((_rational(interval["start"]), _rational(interval["end"])))
    for name in CODE_FILES:
        path = Path(__file__).resolve().with_name(name)
        bindings[str(path)] = file_digest(path)
    with wave.open(str(source), "rb") as reader:
        rate, samples = reader.getframerate(), reader.getnframes()
        channels, width = reader.getnchannels(), reader.getsampwidth()
    spans = retained_pcm_spans(clock, tuple(intervals), output_rate=rate, output_samples=samples)
    output.mkdir(mode=0o700)
    target = output / "selected.wav"
    count = write_pcm_spans(source, target, spans)
    target.chmod(0o600)
    with wave.open(str(target), "rb") as reader:
        if count != sum(end - start for start, end in spans) or (
            reader.getframerate(),
            reader.getnframes(),
            reader.getnchannels(),
            reader.getsampwidth(),
        ) != (rate, count, channels, width):
            raise ValueError("Selected PCM header differs from the supplied source format")
    for name, digest in bindings.items():
        if file_digest(Path(name)) != digest:
            raise ValueError("PCM selection input changed during execution")
    native_duration = sum((end - start for start, end in intervals), Fraction(0))
    receipt = output / "selection.json"
    NativeCaptionLayout._save(
        receipt,
        {
            "version": 1,
            "inputs": bindings,
            "output": str(target),
            "outputSHA256": file_digest(target),
            "spans": spans,
            "sourceSamples": samples,
            "selectedSamples": count,
            "sampleRate": rate,
            "channels": channels,
            "sampleWidth": width,
            "nativeIntervalDuration": str(native_duration),
            "audioDuration": str(Fraction(count, rate)),
            "durationDifference": str(Fraction(count, rate) - native_duration),
            "paddingSamples": 0,
            "resamplingPerformed": False,
            "fadesApplied": False,
            "independentPCMComparisonPassed": False,
            "sourceCorrespondenceVerified": False,
            "AVSyncVerified": False,
            "deliveryComplete": False,
        },
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    select(args.config, args.output, sha256=args.sha256)


if __name__ == "__main__":
    main()
