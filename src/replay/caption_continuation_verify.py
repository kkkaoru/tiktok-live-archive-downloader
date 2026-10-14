"""Read-only domain checks for existing recognition projection and quiet plans."""

import argparse
import wave
from dataclasses import asdict
from math import ceil
from pathlib import Path

from replay import asr_clock_projection as projection
from replay import audio_clock
from replay import continuous_quiet_plan as quiet
from replay.audio_clock import AudioFrameClock
from replay.core import ReplayError
from replay.delivery_media import read_clock
from replay.input_hash_snapshot import InputHashSnapshot
from replay.quiet_cuts import quiet_cut_frames
from replay.recordings import record
from replay.stages import file_digest
from replay.waveform_quiet import measured_quiet_runs


def verify(kind: str, config: Path, output: Path) -> None:
    """Recompute domain results, without inference, media decoding or writes.

    Producer execution remains responsible for its input contract. This check
    requires that exact current producer and its pinned inputs, then checks the
    resulting clock/text/cuts against the same pure domain functions.
    """
    if kind not in {"projection", "quiet"}:
        raise ReplayError("Unknown continuation verification kind")
    config, output = config.resolve(strict=True), output.resolve(strict=True)
    data, result = projection._read(config), projection._read(output)
    bindings = {
        name: projection._binding(value) for name, value in record(data.get("artifacts")).items()
    }
    required = (
        {
            "labels",
            "pcm",
            "audioClock",
            "asrInputs",
            "clockReceipt",
            "pcmReceipt",
            "lineageEvidence",
        }
        if kind == "projection"
        else {
            "labels",
            "stereoPCM",
            "audioClock",
            "videoClock",
            "projectionConfig",
            "lineageEvidence",
        }
    )
    producer = Path(projection.__file__ if kind == "projection" else quiet.__file__)
    producer_key = "projectionSHA256" if kind == "projection" else "plannerSHA256"
    pins = [(config, file_digest(config)), (output, file_digest(output)), *bindings.values()]
    pins.extend(
        (path, file_digest(path)) for path in (producer, Path(audio_clock.__file__), Path(__file__))
    )
    snapshot = InputHashSnapshot.capture(pins)
    if (
        set(bindings) != required
        or result.get("version") != 1
        or result.get("clock") != "source-presentation"
        or result.get("configSHA256") != pins[0][1]
        or result.get(producer_key) != file_digest(producer)
        or result.get("parentSHA256") != {str(path): sha for path, sha in bindings.values()}
        or result.get("deliveryComplete") is not False
        or result.get("humanReviewed") is not False
    ):
        raise ReplayError("Continuation producer/input identity mismatch")
    clock = AudioFrameClock.from_probe(projection._read(bindings["audioClock"][0]))
    labels = projection._read(bindings["labels"][0])
    if kind == "projection":
        expected = projection.project_labels(projection._labels(labels), clock)
        if (
            result.get("labels") != [asdict(label) for label in expected]
            or result.get("audioClockCodeSHA256") != file_digest(Path(audio_clock.__file__))
            or result.get("sourceStart") != str(clock.pts[0] * clock.time_base)
            or result.get("unresolvedPoints") != sum(label.start == label.end for label in expected)
        ):
            raise ReplayError("Projected recognition differs from original clock/text")
    else:
        _verify_quiet(data, result, bindings, clock, labels, output)
    snapshot.finish()


def _verify_quiet(
    data: dict[str, object],
    result: dict[str, object],
    bindings: dict[str, tuple[Path, str]],
    clock: AudioFrameClock,
    labels: dict[str, object],
    output: Path,
) -> None:
    fps = data.get("fps")
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ReplayError("Invalid continuation frame rate")
    peaks = output.parent / "peaks.txt"
    if peaks.stat().st_size > 128 * 1024 * 1024:
        raise ReplayError("Quiet evidence exceeds byte budget")
    peak_sha = file_digest(peaks)
    if result.get("peakSHA256") != peak_sha:
        raise ReplayError("Quiet measurement identity mismatch")
    with wave.open(str(bindings["stereoPCM"][0]), "rb") as stream:
        samples, rate = stream.getnframes(), stream.getframerate()
    with peaks.open(encoding="utf-8") as lines:
        measured = measured_quiet_runs(
            lines, sample_count=samples, sample_rate=rate, window_samples=max(1, rate // 50)
        )
    mapped = quiet.source_quiet(measured, clock, measurement_rate=rate)
    video = read_clock(bindings["videoClock"][0])
    frames = ceil((video.pts[-1] + video.last_duration) * video.time_base * fps)
    cuts = quiet_cut_frames(mapped, quiet._labels(labels), source_frames=frames, fps=fps)
    if (
        result.get("fps") != fps
        or result.get("sourceFrames") != frames
        or result.get("sourceQuiet") != [asdict(span) for span in mapped]
        or result.get("cuts") != [asdict(cut) for cut in cuts]
        or file_digest(peaks) != peak_sha
    ):
        raise ReplayError("Quiet cuts differ from measured source-clock support")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("projection", "quiet"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    verify(args.kind, args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
