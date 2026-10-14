"""Refine a bounded PCM label block from independent, completed native ASR.

Text changes remain explicit candidates. No source-presentation projection, cut
projection, timestamp interpolation or mutation of the original assembly occurs.
"""

import argparse
import json
import os
import re
import wave
from dataclasses import asdict
from itertools import pairwise
from math import ceil, floor, isclose, ulp
from pathlib import Path

from replay.asr_assembly import _labels as native_labels
from replay.asr_assembly import _windows
from replay.asr_clock_projection import _binding, _labels
from replay.asr_label_refine import _alternative as original_alternative
from replay.asr_label_refine import _read, replace_block
from replay.audio_windows import AudioWindow
from replay.batch_asr import RATE
from replay.caption_linguistics import atomic_tokens, attached_tokens
from replay.caption_repair import ClockLabel
from replay.caption_timing import CaptionPolicy, SpokenToken
from replay.input_hash_snapshot import InputHashSnapshot
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_MODEL, DEFAULT_WEIGHTS_SHA256
from replay.recordings import record
from replay.stages import file_digest

ARTIFACTS = frozenset(
    {"labels", "originalInputs", "inputs", "complete", "raw", "clock", "window", "audio"}
)
MAX_BLOCK_LABELS = 512
_NATIVE_PROVENANCE = re.compile(r"(?P<context>.*:)?[0-9]{12}:[0-9]+")


def _context(provenance: str) -> str:
    match = _NATIVE_PROVENANCE.fullmatch(provenance)
    return (match.group("context") or "") if match else provenance


def _sample_boundary(seconds: float, *, upper: bool = False) -> int:
    """Avoid a spurious extra sample from binary float arithmetic only."""
    samples = seconds * RATE
    nearest = round(samples)
    if isclose(samples, nearest, rel_tol=0, abs_tol=8 * ulp(samples)):
        return nearest
    return ceil(samples) if upper else floor(samples)


def _word_units(alternative: tuple[ClockLabel, ...]) -> tuple[SpokenToken, ...]:
    """Keep dependent tails with whole words, including after repeated units."""
    gap = CaptionPolicy().maximum_gap
    units = atomic_tokens(
        [
            SpokenToken(
                label.text,
                label.start,
                label.end,
                _context(label.provenance) or "verified-pcm-window",
            )
            for label in alternative
            if label.start < label.end
        ],
        maximum_gap=gap,
        complete_phrases=True,
    )
    result: list[SpokenToken] = []
    for unit, attached in zip(units, attached_tokens(units, maximum_gap=gap), strict=True):
        if attached:
            previous = result.pop()
            unit = SpokenToken(
                previous.text + unit.text,
                previous.start,
                max(previous.end, unit.end),
                previous.utterance,
            )
        result.append(unit)
    return tuple(result)


def derive_replacement(
    original: tuple[ClockLabel, ...],
    alternative: tuple[ClockLabel, ...],
    *,
    window: AudioWindow,
    target_provenance: str,
) -> dict[str, int]:
    """Expand only to observed wholeword edges and unchanged-neighbor support.

    The sample envelope selects bytes; it never retimes recognition. Cut/scene
    projection remains a separate guarded producer on the original source clock.
    """
    matches = [
        index for index, label in enumerate(original) if label.provenance == target_provenance
    ]
    if not target_provenance or len(matches) != 1:
        raise ValueError("Expected one exact original target provenance")
    first, stop = matches[0], matches[0] + 1
    context = _context(target_provenance)
    # Native word records can themselves be fragments (e.g. a loanword stem
    # and its final kana). Use the same lexical units as caption paging to
    # retain complete words, while their original fragment clocks remain raw.
    units = _word_units(alternative)
    start = _sample_boundary(original[first].start)
    end = _sample_boundary(original[first].end, upper=True)
    while True:
        previous = first, stop, start, end
        if (
            stop - first > MAX_BLOCK_LABELS
            or not window.start_sample <= start < end <= window.end_sample
            or end - start > RATE * 60
        ):
            raise ValueError("Wholeword replacement exceeds bounded verified extraction")
        if any(_context(label.provenance) != context for label in original[first:stop]):
            raise ValueError(
                "Wholeword replacement crosses an explicit source/scene/speaker boundary"
            )
        selected = tuple(
            label
            for label in alternative
            if label.end > start / RATE
            and label.start < end / RATE
            or start / RATE <= label.start == label.end < end / RATE
        )
        if not selected:
            raise ValueError("No recognized wholeword support for target provenance")
        lexical = tuple(
            unit for unit in units if unit.end > start / RATE and unit.start < end / RATE
        )
        support: tuple[ClockLabel | SpokenToken, ...] = (*selected, *lexical)
        if len({_context(label.provenance) for label in selected}) > 1:
            raise ValueError("Wholeword replacement crosses an alternative recognition boundary")
        start = min(start, _sample_boundary(min(label.start for label in support)))
        end = max(end, _sample_boundary(max(label.end for label in support), upper=True))
        while first and original[first - 1].end > start / RATE:
            first -= 1
            start = min(start, _sample_boundary(original[first].start))
            end = max(end, _sample_boundary(original[first].end, upper=True))
        while stop < len(original) and original[stop].start < end / RATE:
            start = min(start, _sample_boundary(original[stop].start))
            end = max(end, _sample_boundary(original[stop].end, upper=True))
            stop += 1
        if previous == (first, stop, start, end):
            replace_block(
                original,
                alternative,
                first=first,
                stop=stop,
                start_sample=start,
                end_sample=end,
            )
            return {"firstLabel": first, "stopLabel": stop, "startSample": start, "endSample": end}


def _dependencies(document: dict[str, object]) -> list[tuple[Path, str]]:
    values = record(document.get("inputs"))
    if not 1 <= len(values) <= 16:
        raise ValueError("Expected bounded original and refined ASR dependencies")
    return [_binding({"path": name, "sha256": digest}) for name, digest in values.items()]


def _interval(document: dict[str, object]) -> tuple[int, int]:
    start, end = document.get("startSample"), document.get("endSample")
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end
        or end - start > RATE * 60
    ):
        raise ValueError("Expected a positive sample-exact window of at most 60 seconds")
    return start, end


def _pcm(window: dict[str, object], audio: Path, original: dict[str, object]) -> AudioWindow:
    start, end = _interval(window)
    source = window.get("source")
    if (
        not isinstance(source, str)
        or not Path(source).is_absolute()
        or Path(source) != Path(source).resolve(strict=True)
        or window.get("version") != 1
        or window.get("sampleRate") != RATE
        or window.get("clock") != "continuous-input-pcm"
        or window.get("sourceClockProjected") is not False
        or record(original.get("inputs")).get(source) != window.get("sourceSHA256")
        or window.get("windowSHA256") != file_digest(audio)
    ):
        raise ValueError("Extraction must bind the original unprojected PCM and exact audio")
    with wave.open(source, "rb") as whole, wave.open(str(audio), "rb") as clip:
        expected = (1, 2, RATE, "NONE")
        if (
            (whole.getnchannels(), whole.getsampwidth(), whole.getframerate(), whole.getcomptype())
            != expected
            or (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.getcomptype())
            != expected
            or whole.getnframes() != original.get("samples")
            or end > whole.getnframes()
            or clip.getnframes() != end - start
        ):
            raise ValueError("Extraction PCM format, sample extent or original length mismatch")
        whole.setpos(start)
        payload = whole.readframes(end - start)
        if len(payload) != (end - start) * 2 or payload != clip.readframes(end - start):
            raise ValueError("Extraction samples differ from the original PCM interval")
    return AudioWindow(start, end - start)


def _alternative(
    paths: dict[str, Path],
    pins: dict[str, str],
    original: dict[str, object],
    assembled: dict[str, object],
) -> tuple[AudioWindow, tuple[ClockLabel, ...]]:
    window = _pcm(_read(paths["window"]), paths["audio"], original)
    inputs, complete, clock = (_read(paths[name]) for name in ("inputs", "complete", "clock"))
    windows = _windows(inputs)
    model = DEFAULT_MODEL.resolve()
    dependencies = record(inputs.get("inputs"))
    if paths["inputs"] == paths["originalInputs"]:
        if (
            clock.get("startSample") != window.start_sample
            or clock.get("endSample") != window.end_sample
            or inputs.get("device") != "cpu"
            or inputs.get("dtypeContract") != "provider-default-fp16-unchanged"
            or dependencies.get(str(model / "config.json")) != DEFAULT_CONFIG_SHA256
            or dependencies.get(str(model / "weights.npz")) != DEFAULT_WEIGHTS_SHA256
        ):
            raise ValueError("Original overlap differs from default-model extraction clock")
        return window, original_alternative(paths, pins, assembled)
    if (
        len(windows) != 1
        or windows[0] != AudioWindow(0, window.sample_count)
        or inputs.get("device") != "cpu"
        or inputs.get("dtypeContract") != "provider-default-fp16-unchanged"
        or dependencies.get(str(paths["audio"])) != pins["audio"]
        or dependencies.get(str(model / "config.json")) != DEFAULT_CONFIG_SHA256
        or dependencies.get(str(model / "weights.npz")) != DEFAULT_WEIGHTS_SHA256
        or complete != {"rawSHA256": [pins["raw"]], "windows": 1, "deliveryComplete": False}
        or clock
        != {
            "startSample": 0,
            "endSample": window.sample_count,
            "sampleRate": RATE,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": pins["raw"],
        }
    ):
        raise ValueError("Expected one completed default-model CPU window with native local clock")
    return window, tuple(
        native_labels(
            _read(paths["raw"]),
            window=window,
            owner_start=window.start_sample / RATE,
            owner_end=window.end_sample / RATE,
        )
    )


def _plan(config: Path) -> dict[str, object]:
    """Recompute the bounded candidate from pinned evidence without writing files."""
    config_digest = file_digest(config)
    data = _read(config)
    selection = "targetProvenance" if "targetProvenance" in data else "replacement"
    if set(data) != {"version", "artifacts", selection} or data["version"] != 1:
        raise ValueError("Unsupported PCM-window refinement configuration")
    artifacts = record(data["artifacts"])
    if set(artifacts) != ARTIFACTS:
        raise ValueError(
            "Expected original assembly, extraction and completed recognition bindings"
        )
    bindings = {name: _binding(value) for name, value in artifacts.items()}
    paths = {name: binding[0] for name, binding in bindings.items()}
    pins = {name: binding[1] for name, binding in bindings.items()}
    original_inputs, inputs = (_read(paths[name]) for name in ("originalInputs", "inputs"))
    _windows(original_inputs)
    dependencies = _dependencies(original_inputs) + _dependencies(inputs)
    word_inputs = (
        {
            str(path): file_digest(path)
            for path in (
                Path(__file__).with_name("caption_linguistics.py"),
                Path(__file__).with_name("caption_timing.py"),
            )
        }
        if selection == "targetProvenance"
        else {}
    )
    snapshot = InputHashSnapshot.capture(
        [
            (config, config_digest),
            *bindings.values(),
            *dependencies,
            *((Path(name), digest) for name, digest in word_inputs.items()),
        ]
    )
    original = _read(paths["labels"])
    labels = _labels(original)
    if (
        record(original.get("parentSHA256")).get(str(paths["originalInputs"]))
        != pins["originalInputs"]
        or original.get("humanReviewed") is not False
        or original.get("speechAccuracyVerified") is not False
        or original.get("deliveryComplete") is not False
    ):
        raise ValueError("Original labels must bind the unprojected, unreviewed PCM assembly")
    window, alternative = _alternative(paths, pins, original_inputs, original)
    target = data.get("targetProvenance")
    if selection == "targetProvenance" and not isinstance(target, str):
        raise ValueError("Expected one exact original target provenance")
    interval: dict[str, object] = (
        dict(derive_replacement(labels, alternative, window=window, target_provenance=target))
        if isinstance(target, str)
        else record(data["replacement"])
    )
    names = ("firstLabel", "stopLabel", "startSample", "endSample")
    if set(interval) != set(names) or any(type(interval[name]) is not int for name in names):
        raise ValueError("Expected bounded integer replacement indices and samples")
    first, stop, start, end = (int(str(interval[name])) for name in names)
    if not window.start_sample <= start < end <= window.end_sample:
        raise ValueError("Replacement interval exceeds the verified extraction")
    if any(label.start < start / RATE or label.end > end / RATE for label in labels[first:stop]):
        raise ValueError("Replacement boundary splits an original recognized label")
    result, selected = replace_block(
        labels, alternative, first=first, stop=stop, start_sample=start, end_sample=end
    )
    snapshot.finish()
    return {
        **original,
        "labels": [asdict(label) for label in result],
        "parentSHA256": {
            **record(original["parentSHA256"]),
            **{str(path): digest for path, digest in bindings.values()},
            **word_inputs,
        },
        "refinementSHA256": file_digest(Path(__file__)),
        "refinement": {
            "configSHA256": config_digest,
            "replacement": interval,
            "targetProvenance": target,
            "wordBoundaryInputs": word_inputs,
            "recognitionWindowKind": "original-overlap"
            if paths["inputs"] == paths["originalInputs"]
            else "bounded-new-window",
            "extractionOffsetSample": window.start_sample,
            "removed": [asdict(label) for label in labels[first:stop]],
            "selected": [asdict(label) for label in selected],
            "clockOffsetApplications": 1,
            "sourceProjectionApplications": 0,
            "candidateOnly": True,
        },
        "onsetReversals": 0,
        "adjacentOverlaps": sum(b.start < a.end for a, b in pairwise(result)),
        "unresolvedPoints": sum(label.start == label.end for label in result),
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
    }


def run(config: Path, output: Path) -> int:
    """Keep every external label and apply the recognized extraction offset once."""
    if (
        output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.parent != output.parent.resolve(strict=True)
    ):
        raise ValueError("Expected an absent output in an existing canonical directory")
    document = _plan(config)
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(_labels(document))


def verify(config: Path, output: Path) -> int:
    """Read-only full candidate equality; not recognition meaning or media review."""
    pin = file_digest(output)
    snapshot = InputHashSnapshot.capture([(output, pin)])
    actual, expected = _read(output), _plan(config)
    if actual != expected:
        raise ValueError("PCM refinement candidate differs from its pinned plan")
    snapshot.finish()
    return len(_labels(actual))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    count = verify(args.config, args.output) if args.verify else run(args.config, args.output)
    print(json.dumps({"labels": count, "candidateOnly": True, "speechAccuracyVerified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
