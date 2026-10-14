"""Exact selected-PCM placement diagnostics; never infer synchronization from duration."""

import argparse
import sys
from bisect import bisect_right
from fractions import Fraction
from itertools import accumulate
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.caption_measured_job import ROOT, _json, _local, _rational
from replay.caption_native_layout import NativeCaptionLayout
from replay.continuous_cuts import retained_pcm_spans
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import _digest

MAX_METADATA_BYTES = 128 * 1024**2


def integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("Expected bounded nonnegative integer")
    return value


def spans_from_json(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 1_000_000:
        raise ValueError("Expected bounded nonempty sample spans")
    spans: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Invalid sample span")
        start, end = integer(item[0]), integer(item[1])
        if end <= start or (spans and start < spans[-1][1]):
            raise ValueError("Reversed or overlapping sample spans")
        spans.append((start, end))
    return tuple(spans)


def cut_intervals(value: object, rate: int) -> tuple[tuple[Fraction, Fraction], ...]:
    rows = record(value).get("cuts")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100_000 or rate <= 0:
        raise ValueError("Invalid cuts/rate")
    result: list[tuple[Fraction, Fraction]] = []
    offset = 0
    for item in rows:
        row = record(item)
        start = integer(row.get("source_start_frame"))
        end = integer(row.get("source_end_frame"))
        if end <= start or (result and Fraction(start, rate) < result[-1][1]):
            raise ValueError("Cuts overlap or reverse")
        if "output_start_frame" in row and integer(row["output_start_frame"]) != offset:
            raise ValueError("Noncontiguous output cuts")
        result.append((Fraction(start, rate), Fraction(end, rate)))
        offset += end - start
    return tuple(result)


def placement(
    clock: AudioFrameClock,
    spans: tuple[tuple[int, int], ...],
    *,
    rate: int,
    cuts: tuple[tuple[Fraction, Fraction], ...],
) -> dict[str, object]:
    """Split every selected span at frame/cut boundaries using exact rational clocks.

    A sample outside retained native support is unmapped, not snapped to a cut.
    Fractional rate-projected frame edges retain their exact offset and fail approval.
    """
    frame_ends = tuple(Fraction(end * rate, clock.sample_rate) for end in clock.sample_ends)
    aligned = all(end.denominator == 1 for end in frame_ends)
    ends = tuple(-(-end.numerator // end.denominator) for end in frame_ends)
    if not spans or spans[-1][1] > ends[-1] or not cuts:
        raise ValueError("Selected spans exceed source clock")
    cut_starts = tuple(start for start, _ in cuts)
    offsets = (Fraction(0), *accumulate(end - start for start, end in cuts))
    rows: list[dict[str, object]] = []
    output = 0
    maximum = Fraction(0)
    preserved = aligned
    first_drifts: list[Fraction | None] = [None] * len(cuts)
    last_drifts: list[Fraction | None] = [None] * len(cuts)
    output_ends: list[int | None] = [None] * len(cuts)
    minima: list[Fraction | None] = [None] * len(cuts)
    maxima: list[Fraction | None] = [None] * len(cuts)
    for start, end in spans:
        cursor = start
        while cursor < end:
            frame = bisect_right(frame_ends, cursor)
            prior = frame_ends[frame - 1] if frame else Fraction(0)
            native = clock.pts[frame] * clock.time_base + (cursor - prior) / rate
            stop = min(end, ends[frame])
            cut = bisect_right(cut_starts, native) - 1
            mapped = cut >= 0 and native < cuts[cut][1]
            boundary = (
                cuts[cut][1]
                if mapped
                else next(
                    (first for first in cut_starts if first > native),
                    native + Fraction(stop - cursor, rate),
                )
            )
            # Ceil retains the whole sample whose start lies before a native boundary.
            extent = (boundary - native) * rate
            stop = min(stop, cursor + -(-extent.numerator // extent.denominator))
            count = stop - cursor
            if count <= 0:
                raise ValueError("Nonprogressing clock mapping")
            expected = offsets[cut] + native - cuts[cut][0] if mapped else None
            drift = Fraction(output, rate) - expected if expected is not None else None
            support = mapped and native + Fraction(count, rate) <= cuts[cut][1]
            preserved &= support and drift == 0
            if drift is not None:
                maximum = max(maximum, abs(drift))
                prior_min, prior_max = minima[cut], maxima[cut]
                minima[cut] = drift if prior_min is None else min(prior_min, drift)
                maxima[cut] = drift if prior_max is None else max(prior_max, drift)
                if first_drifts[cut] is None:
                    first_drifts[cut] = drift
                last_drifts[cut] = drift
                output_ends[cut] = output + count
            rows.append(
                {
                    "sourceStartSample": cursor,
                    "sourceEndSample": stop,
                    "outputStartSample": output,
                    "outputEndSample": output + count,
                    "nativeStart": str(native),
                    "nativeEnd": str(native + Fraction(count, rate)),
                    "cutIndex": cut if mapped else None,
                    "expectedOutputStart": str(expected) if expected is not None else None,
                    "actualOutputStart": str(Fraction(output, rate)),
                    "drift": str(drift) if drift is not None else None,
                    "withinNativeSupport": support,
                }
            )
            if len(rows) > 2_000_000:
                raise ValueError("Mapping piece budget exceeded")
            output += count
            cursor = stop
    difference = Fraction(output, rate) - offsets[-1]
    per_cut = [
        {
            "cutIndex": index,
            "expectedOutputStart": str(offsets[index]),
            "expectedOutputEnd": str(offsets[index + 1]),
            "firstSelectedSampleDrift": str(first) if first is not None else None,
            "lastSelectedSampleDrift": str(last) if last is not None else None,
            "minimumDrift": str(minima[index]) if minima[index] is not None else None,
            "maximumDrift": str(maxima[index]) if maxima[index] is not None else None,
            "outputEndMinusExpectedCutEnd": (
                str(Fraction(end, rate) - offsets[index + 1]) if end is not None else None
            ),
        }
        for index, (first, last, end) in enumerate(
            zip(first_drifts, last_drifts, output_ends, strict=True)
        )
    ]
    return {
        "pieces": rows,
        "cuts": per_cut,
        "selectedSamples": output,
        "nativeDuration": str(offsets[-1]),
        "durationDifference": str(difference),
        "maximumAbsoluteDrift": str(maximum),
        "minimumDrift": str(min(value for value in minima if value is not None))
        if any(value is not None for value in minima)
        else None,
        "maximumDrift": str(max(value for value in maxima if value is not None))
        if any(value is not None for value in maxima)
        else None,
        "nativeTimesPreserved": preserved and difference == 0,
        "sourceFrameBoundariesSampleAligned": aligned,
    }


def metadata_reference(value: object, bindings: dict[str, str]) -> dict[str, object]:
    """Read/hash only bounded metadata; waveform identities remain historical claims."""
    ref = record(value)
    path, digest = _local(ref.get("path")), _digest(ref.get("sha256"))
    if str(path) in bindings and bindings[str(path)] != digest:
        raise ValueError("Conflicting metadata bindings")
    data = record(_json(_bound_bytes(path, digest, MAX_METADATA_BYTES)))
    bindings[str(path)] = digest
    return data


def analyze_metadata(config: Path, output: Path, *, sha256: str) -> Path:
    """Diagnose native placement without opening PCM, media, tools or models.

    The selection producer configuration must describe the render's unchanged cuts.
    Byte identity is referenced from the audit, never certified as current here.
    """
    config, output = _local(str(config.absolute())), _local(str(output.absolute()))
    if not output.is_relative_to(ROOT / "edits/work"):
        raise ValueError("Report must remain under checkout edits/work")
    if output.exists():
        raise FileExistsError(output)
    bindings: dict[str, str] = {}
    data = metadata_reference({"path": str(config), "sha256": sha256}, bindings)
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Expected version 1 mapping configuration")
    documents = {
        key: metadata_reference(data.get(key), bindings)
        for key in (
            "renderConfig",
            "cuts",
            "selectionConfig",
            "selection",
            "originalClock",
            "pcmAudit",
        )
    }
    render, selection, producer, audit = (
        documents[key] for key in ("renderConfig", "selection", "selectionConfig", "pcmAudit")
    )
    if any(
        type(doc.get("version")) is not int or doc["version"] != 1 for doc in (selection, producer)
    ):
        raise ValueError("Expected version 1 selection/config")
    if type(audit.get("version")) is not int or audit["version"] != 2:
        raise ValueError("Expected version 2 PCM audit")
    audio = record(render.get("audio"))
    source, selected = _local(audio.get("source")), _local(selection.get("output"))
    clock_path = _local(record(data["originalClock"]).get("path"))
    if (
        _local(audio.get("clock")) != clock_path
        or _local(render.get("cuts")) != _local(record(data["cuts"]).get("path"))
        or _local(record(producer.get("source")).get("path")) != source
        or _local(record(producer.get("clock")).get("path")) != clock_path
        or source == selected
    ):
        raise ValueError("Render/producer source, clock or cuts references differ")
    selection_pins, audit_pins = record(selection.get("inputs")), record(audit.get("inputs"))
    source_digest = _digest(record(producer["source"]).get("sha256"))
    if _digest(record(producer["clock"]).get("sha256")) != bindings[str(clock_path)]:
        raise ValueError("Producer clock binding differs")
    expected = {
        str(source): source_digest,
        str(clock_path): bindings[str(clock_path)],
        str(_local(record(data["selectionConfig"]).get("path"))): bindings[
            str(_local(record(data["selectionConfig"]).get("path")))
        ],
    }
    if any(selection_pins.get(name) != digest for name, digest in expected.items()):
        raise ValueError("Selection does not bind producer/source/clock")
    expected.update(
        {
            str(selected): _digest(selection.get("outputSHA256")),
            str(_local(record(data["selection"]).get("path"))): bindings[
                str(_local(record(data["selection"]).get("path")))
            ],
        }
    )
    if any(audit_pins.get(name) != digest for name, digest in expected.items()):
        raise ValueError("Audit does not bind selection/producer/source/clock/output")
    if (
        audit.get("kind") != "independent-pcm-selection-clock-audit"
        or audit.get("selectedPCMBytesEqual") is not True
        or integer(audit.get("mismatchedBytes")) != 0
    ):
        raise ValueError("Historical PCM comparison did not pass")
    cuts = cut_intervals(documents["cuts"], integer(render.get("rate"), minimum=1))
    raw_intervals = producer.get("intervals")
    if (
        producer.get("coordinateSpace") != "native-presentation-seconds"
        or not isinstance(raw_intervals, list)
        or len(raw_intervals) != len(cuts)
        or tuple(
            (_rational(record(row).get("start")), _rational(record(row).get("end")))
            for row in raw_intervals
        )
        != cuts
    ):
        raise ValueError("Producer native intervals differ from render cuts")
    spans = spans_from_json(selection.get("spans"))
    if spans_from_json(audit.get("spans")) != spans:
        raise ValueError("Audit spans differ")
    rate = integer(selection.get("sampleRate"), minimum=1)
    samples = integer(selection.get("sourceSamples"), minimum=1)
    count = sum(end - start for start, end in spans)
    if integer(selection.get("selectedSamples"), minimum=1) != count:
        raise ValueError("Selected sample count differs")
    for key in ("sampleRate", "sourceSamples", "selectedSamples", "channels", "sampleWidth"):
        if integer(audit.get(key), minimum=1) != integer(selection.get(key), minimum=1):
            raise ValueError("Audit PCM metadata differs")
    if (
        integer(selection.get("sampleWidth")) != 2
        or integer(audit.get("comparedBytes"), minimum=1)
        != count * integer(selection.get("channels"), minimum=1) * 2
    ):
        raise ValueError("Historical comparison byte extent differs")
    clock = AudioFrameClock.from_probe(documents["originalClock"])
    if retained_pcm_spans(clock, cuts, output_rate=rate, output_samples=samples) != spans:
        raise ValueError("Selection differs from native cut projection")
    mapping = placement(clock, spans, rate=rate, cuts=cuts)
    difference = Fraction(samples) - Fraction(clock.sample_ends[-1] * rate, clock.sample_rate)
    mapping["sourceSamplesMinusClockProjection"] = str(difference)
    if difference:
        mapping["nativeTimesPreserved"] = False
    for name in (
        "delivery_audio_mapping.py",
        "audio_clock.py",
        "continuous_cuts.py",
        "caption_measured_job.py",
        "foreground_evidence.py",
        "caption_native_layout.py",
        "tracking_observations.py",
        "stages.py",
        "recordings.py",
    ):
        path = Path(__file__).with_name(name).resolve()
        bindings[str(path)] = file_digest(path)
    if any(file_digest(Path(name)) != digest for name, digest in bindings.items()):
        raise ValueError("Metadata changed during analysis")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    NativeCaptionLayout._save(
        output,
        {
            "version": 1,
            "kind": "delivery-audio-native-mapping",
            "inputs": bindings,
            "measurements": {"mapping": mapping},
            "historicalPCMIdentities": {
                str(source): source_digest,
                str(selected): expected[str(selected)],
            },
            "historicalPCMComparisonReferenced": True,
            "currentPCMBytesRechecked": False,
            "metadataInputsStable": True,
            "aacSampleTimeMapping": None,
            "acceptedQuality": False,
            "accepted": False,
            "limitations": [
                "PCM byte identity refers to the existing audit; current bytes were not rechecked.",
                "Source/protected-speech provenance and quality are not revalidated.",
                "Final AAC clock, resampling and codec extent are a separate unevaluated phase.",
            ],
        },
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    print(analyze_metadata(args.config, args.output, sha256=args.sha256))
    # A completed diagnostic is never quality acceptance.
    return 1


if __name__ == "__main__":
    sys.exit(main())
