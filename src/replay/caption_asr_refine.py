"""Build an unreviewed token replacement candidate; never mutate or run a job.

Only identity PCM-to-source clocks with explicit pinned evidence are supported.
The caller owns semantic review, inactive-job verification and manifest promotion.
"""

import argparse
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from replay.caption_io import _array, _fields, mlx_groups
from replay.caption_measured_job import _document, _json, _local, _object, _pin, _span, _token
from replay.caption_projection import CutSpan, project_tokens
from replay.caption_source_support import RetainedSourceSpan
from replay.caption_timing import SpokenToken
from replay.delivery_captions import read_cuts
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_MODEL, DEFAULT_WEIGHTS_SHA256, pcm_audio
from replay.stages import file_digest
from replay.tracking_observations import _digest, _integer, _read, _write

RATE = 16000
MAX_BYTES = 16 * 1024 * 1024
EPSILON = 1e-7


def _batch_inputs(
    raw: dict[str, object], bindings: dict[str, str], wav: Path, samples: int
) -> None:
    """Verify one completed batch window without relabeling it as local_asr."""
    inputs = _object(
        _document(raw["inputs"], bindings),
        "version inputs samples sampleRate windowSamples strideSamples clock device dtypeContract",
    )
    pins = _fields(inputs["inputs"])
    for path, digest in pins.items():
        _local(path)
        _digest(digest)
    model = DEFAULT_MODEL.resolve()
    clock = _object(
        _document(raw["recognitionClock"], bindings),
        "startSample endSample sampleRate humanReviewed emptyRecognitionIsNotSilence rawSHA256",
    )
    complete = _object(
        _document(raw["recognitionComplete"], bindings),
        "rawSHA256 windows deliveryComplete",
    )
    raw_hash = _digest(_fields(raw["recognition"])["sha256"])
    if (
        _integer(inputs["version"]) != 1
        or inputs["clock"] != "input-pcm-not-projected"
        or inputs["device"] != "cpu"
        or inputs["dtypeContract"] != "provider-default-fp16-unchanged"
        or _integer(inputs["samples"]) != samples
        or _integer(inputs["sampleRate"]) != RATE
        or _integer(inputs["windowSamples"]) != 60 * RATE
        or _integer(inputs["strideSamples"]) != 30 * RATE
        or pins.get(str(wav)) != bindings[str(wav)]
        or pins.get(str(model / "config.json")) != DEFAULT_CONFIG_SHA256
        or pins.get(str(model / "weights.npz")) != DEFAULT_WEIGHTS_SHA256
        or _integer(clock["startSample"]) != 0
        or _integer(clock["endSample"]) != samples
        or _integer(clock["sampleRate"]) != RATE
        or clock["humanReviewed"] is not False
        or clock["emptyRecognitionIsNotSilence"] is not True
        or clock["rawSHA256"] != raw_hash
        or complete["rawSHA256"] != [raw_hash]
        or _integer(complete["windows"]) != 1
        or complete["deliveryComplete"] is not False
    ):
        raise ValueError("Completed batch window, model or input clock mismatch")


def _interval(value: object) -> tuple[Fraction, Fraction]:
    raw = _object(value, "startSample endSample")
    start, end = _integer(raw["startSample"]), _integer(raw["endSample"])
    if not 0 <= start < end:
        raise ValueError("Replacement requires an explicit positive source interval")
    return Fraction(start, RATE), Fraction(end, RATE)


def _recognition(raw: dict[str, object], bindings: dict[str, str]) -> tuple[SpokenToken, ...]:
    window = _object(
        _document(raw["window"], bindings),
        "version source sourceSHA256 startSample endSample sampleRate windowSHA256 "
        "clock sourceClockProjected",
    )
    _local(window["source"])
    clock = _fields(_document(raw["clockEvidence"], bindings))
    wav = _pin(raw["audio"], bindings)
    source_hash = _digest(raw["sourceSHA256"])
    start, end = _interval({"startSample": window["startSample"], "endSample": window["endSample"]})
    samples = len(pcm_audio(wav))
    if (
        _integer(window["version"]) != 1
        or window["clock"] != "continuous-input-pcm"
        or window["sourceClockProjected"] is not False
        or clock.get("clock") != "native-source-presentation-already-projected"
        or window["sourceSHA256"] != source_hash
        or window["windowSHA256"] != bindings[str(wav)]
        or _integer(window["sampleRate"]) != RATE
        or Fraction(samples, RATE) != end - start
    ):
        raise ValueError("Recognition input, model or identity source clock mismatch")
    if raw["version"] == 2:
        if (
            clock.get("sourceSHA256") != source_hash
            or _local(clock.get("source")) != _local(window["source"])
            or _integer(clock.get("sampleRate")) != RATE
            or _integer(clock.get("samples")) < _integer(window["endSample"])
            or clock.get("humanReviewed") is not False
        ):
            raise ValueError("Native PCM plan does not support this recognition window")
        _batch_inputs(raw, bindings, wav, samples)
    else:
        inputs = _object(
            _document(raw["inputs"], bindings),
            "version source model source_sha256 config_sha256 weights_sha256 sample_rate "
            "samples clock human_reviewed",
        )
        _local(inputs["model"])
        if (
            _integer(inputs["version"]) != 1
            or inputs["clock"] != "continuous-input-pcm"
            or inputs["human_reviewed"] is not False
            or clock.get("noSecondClockProjection") is not True
            or inputs["source_sha256"] != bindings[str(wav)]
            or _local(inputs["source"]) != wav
            or inputs["config_sha256"] != DEFAULT_CONFIG_SHA256
            or inputs["weights_sha256"] != DEFAULT_WEIGHTS_SHA256
            or _integer(inputs["sample_rate"]) != RATE
            or samples != _integer(inputs["samples"])
        ):
            raise ValueError("Recognition input, model or identity source clock mismatch")
    selected_start, selected_end = _interval(raw["replacement"])
    if not start <= selected_start < selected_end <= end:
        raise ValueError("Replacement interval exceeds recognition window")
    groups = mlx_groups(_document(raw["recognition"], bindings), duration=samples / RATE)
    recognition_hash = _digest(_fields(raw["recognition"])["sha256"])
    tokens = tuple(
        SpokenToken(
            group.text,
            group.start + float(start),
            group.end + float(start),
            f"refine:{recognition_hash}:mlx:"
            + ",".join(str(index) for index in group.source_segments),
        )
        for group in groups
    )
    _ordered(tokens)
    chosen = tuple(token for token in tokens if _selected(token, selected_start, selected_end))
    if not chosen:
        raise ValueError("Replacement has no recognized groups; empty ASR is not silence")
    return chosen


def _ordered(tokens: tuple[SpokenToken, ...]) -> None:
    if any(
        current.start < previous.end - EPSILON
        for previous, current in zip(tokens, tokens[1:], strict=False)
    ):
        raise ValueError("Ambiguous overlapping recognition or token clocks")


def _selected(token: SpokenToken, start: Fraction, end: Fraction) -> bool:
    if token.end <= float(start) + EPSILON or token.start >= float(end) - EPSILON:
        return False
    if token.start < float(start) - EPSILON or token.end > float(end) + EPSILON:
        raise ValueError("Token straddles replacement boundary")
    return True


def _verify_span(
    span: RetainedSourceSpan, *, cuts: tuple[CutSpan, ...], rate: int, offset: Fraction
) -> None:
    cursor = span.source_start
    for cut in cuts:
        left = max(span.source_start, Fraction(cut.source_start_frame, rate))
        right = min(span.source_start + span.duration, Fraction(cut.source_end_frame, rate))
        if left >= right:
            continue
        mapped = left + Fraction(cut.output_start_frame - cut.source_start_frame, rate) - offset
        if left != cursor or mapped != span.output_start + left - span.source_start:
            raise ValueError("Manifest spans disagree with cuts or segment offset")
        cursor = right
    if cursor != span.source_start + span.duration:
        raise ValueError("Manifest span lacks cut support")


def _covered(start: Fraction, end: Fraction, *, spans: tuple[RetainedSourceSpan, ...]) -> bool:
    cursor = start
    for span in spans:
        if span.output_start <= cursor < span.output_start + span.duration:
            cursor = min(end, span.output_start + span.duration)
    return start >= 0 and cursor == end


def _ranges(
    raw: dict[str, object], manifest: dict[str, object], bindings: dict[str, str]
) -> tuple[tuple[CutSpan, ...], tuple[tuple[Fraction, Fraction], ...], Fraction, int]:
    rate = _integer(manifest["outputFrameRate"])
    offset_frame = _integer(raw["segmentOffsetFrame"])
    if not 1 <= rate <= 120 or offset_frame < 0:
        raise ValueError("Invalid frame rate or segment offset")
    offset = Fraction(offset_frame, rate)
    _document(raw["cuts"], bindings)
    cuts_path = _local(_fields(raw["cuts"])["path"])
    cuts = tuple(
        CutSpan(c.source_start, c.source_end, c.output_start) for c in read_cuts(cuts_path)
    )
    if len(cuts) > 3000:
        raise ValueError("Caption projection cut budget exceeded")
    spans = tuple(_span(value) for value in _array(manifest["spans"], 3000))
    if not spans or any(
        current.source_start < previous.source_start + previous.duration
        or current.output_start != previous.output_start + previous.duration
        for previous, current in zip(spans, spans[1:], strict=False)
    ):
        raise ValueError("Manifest spans must be ordered and output-contiguous")
    # Prove every manifest span is an exact restriction of the global cut map.
    for span in spans:
        _verify_span(span, cuts=cuts, rate=rate, offset=offset)
    start, end = _interval(raw["replacement"])
    ranges: list[tuple[Fraction, Fraction]] = []
    for cut in cuts:
        left, right = (
            max(start, Fraction(cut.source_start_frame, rate)),
            min(end, Fraction(cut.source_end_frame, rate)),
        )
        if left >= right:
            continue
        shift = Fraction(cut.output_start_frame - cut.source_start_frame, rate) - offset
        a, b = left + shift, right + shift
        if not _covered(a, b, spans=spans):
            raise ValueError("Replacement affects output outside this job")
        ranges.append((a, b))
    if not ranges:
        raise ValueError("Replacement has no retained output")
    return cuts, tuple(ranges), offset, rate


def refine(config: Path, directory: Path, *, config_sha256: str) -> Path:
    """Write new tokens.json and receipt.json, never a runnable manifest.

    Pins bind supplied identity-clock evidence, not its semantic correctness.
    The expected source hash is checked against the extraction receipt; the full
    source and installed model are not reread or used for inference here.
    """
    directory = _local(str(directory.absolute()))
    if not directory.is_relative_to(Path(__file__).resolve().parents[2] / "edits" / "work"):
        raise ValueError("Repair output must stay under edits/work")
    if directory.exists():
        raise FileExistsError(directory)
    bindings: dict[str, str] = {}
    document = _fields(
        _document({"path": str(config.absolute()), "sha256": config_sha256}, bindings)
    )
    version = _integer(document.get("version"))
    if version not in (1, 2):
        raise ValueError("Unsupported repair version")
    fields = (
        "version window audio inputs recognition clockEvidence sourceSHA256 cuts manifest "
        "measurementDirectory segmentOffsetFrame replacement"
    )
    raw = _object(
        document, fields + (" recognitionClock recognitionComplete" if version == 2 else "")
    )
    measurement = _local(raw["measurementDirectory"])
    if (measurement / "candidate.json").exists():
        raise ValueError("Completed measurement must not be repaired")
    manifest = _object(
        _document(raw["manifest"], bindings),
        "version tokens spans foreground native calibration outputFrameRate "
        "region centerX clearance",
    )
    if _integer(manifest["version"]) != 1:
        raise ValueError("Unsupported manifest version")
    original = _object(_document(manifest["tokens"], bindings), "version clock tokens")
    if _integer(original["version"]) != 1 or original["clock"] != "output":
        raise ValueError("Expected segment-local output tokens")
    rows = _array(original["tokens"], 100000)
    old = tuple(_token(row) for row in rows)
    _ordered(old)
    cuts, ranges, offset, rate = _ranges(raw, manifest, bindings)
    source = _recognition(raw, bindings)
    projection = project_tokens(source, cuts, fps=rate)
    if projection.boundary_conflicts:
        raise ValueError("Recognized group conflicts with cut boundary")
    replacement = tuple(
        SpokenToken(t.text, t.start - float(offset), t.end - float(offset), t.utterance)
        for t in projection.tokens
    )
    if not replacement or not all(any(_selected(t, a, b) for a, b in ranges) for t in replacement):
        raise ValueError("Projected replacement lacks job support")
    removed = [
        index for index, token in enumerate(old) if any(_selected(token, a, b) for a, b in ranges)
    ]
    if not removed:
        raise ValueError("Selection replaces no existing tokens")
    kept = [row for index, row in enumerate(rows) if index not in removed]
    result = sorted([*kept, *(asdict(token) for token in replacement)], key=_onset)
    _ordered(tuple(_token(row) for row in result))
    if any(file_digest(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("Repair input changed during validation")
    if (measurement / "candidate.json").exists():
        raise ValueError("Measurement completed during validation")
    directory.mkdir(mode=0o700)
    target = directory / "tokens.json"
    _write(target, {"version": 1, "clock": "output", "tokens": result})
    _write(
        directory / "receipt.json",
        {
            "version": 1,
            "inputs": bindings,
            "configuration": raw,
            "tokens": {"path": str(target), "sha256": file_digest(target)},
            "removedTokenIndices": removed,
            "selectedSourceTokens": [asdict(t) for t in source],
            "omittedByCuts": [asdict(t) for t in projection.omitted],
            "humanReviewed": False,
            "candidateOnly": True,
            "deliveryComplete": False,
            "recognitionInputKind": "batch-asr-single-window" if version == 2 else "local-asr",
            "sourceClockProjection": "identity-native-presentation-no-second-projection",
        },
    )
    return target


def _onset(value: object) -> float:
    return _token(value).start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    # Strict parse before hashing also rejects duplicate fields and nonfinite values.
    _json(_read(args.config, MAX_BYTES))
    print(refine(args.config, args.directory, config_sha256=file_digest(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
