"""Bounded rate-aware PCM/AAC comparisons; observations never retime a product.

The original PCM is sampled at exact source indices. Each temporary source view
is resampled once; decoded AAC retains its actual demuxed PTS. Neither edit lists
nor skip samples are applied again. Local correspondence cannot approve a whole.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay import aac_presentation_acceptance as aac
from replay.audio_pcm_alignment import AlignmentRejected, match_window
from replay.core import ReplayError
from replay.recordings import record

WINDOW_SECONDS = 2
PRE_ROLL_SECONDS = 1
MAX_LOG_BYTES = 1024**2
MAX_VIEW_BYTES = 8 * 1024**2
PROCESS_SECONDS = 120
FRAME_RE = re.compile(r"\bn:(\d+) pts:(-?\d+) .*?\bchannels:(\d+) .*?\brate:(\d+) nb_samples:(\d+)")
PCM = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Request:
    reference: aac.Reference
    original: aac.ObservationRequest
    composition: aac.Reference
    source_identity: aac.FileIdentity
    rate: int
    decoded_end: int


@dataclass(frozen=True, slots=True)
class Window:
    name: str
    source_start: int
    source_count: int
    target_start: int
    target_count: int
    actual_count: int


def load(reference: aac.Reference) -> Request:
    if reference.path.stat().st_size > aac.MAX_REQUEST_BYTES:
        raise ReplayError("Diagnostic request exceeds its byte budget")
    row = aac._evidence(reference).document()
    if (
        type(row.get("version")) is not int
        or row.get("version") != 1
        or row.get("kind") != "aac-content-diagnostic-request"
    ):
        raise ReplayError("Unsupported content diagnostic request")
    original = aac.load_observation_request(aac._reference(row.get("presentationRequest")))
    composition = aac._reference(row.get("composition"))
    if composition.path.stat().st_size > aac.MAX_REQUEST_BYTES:
        raise ReplayError("Diagnostic composition evidence exceeds its byte budget")
    report = aac._evidence(composition).document()
    assessment = record(report.get("aacAssessment"))
    rate = aac._integer(row.get("comparisonRate"), minimum=8000)
    if (
        rate > 192000
        or row.get("windowSeconds") != WINDOW_SECONDS
        or original.source is None
        or original.source.start != 0
        or report.get("version") != 1
        or report.get("kind") != "video-composition-timing-observation"
        or report.get("newDecode") is not False
        or report.get("wholeSHAFreshlyVerified") is not False
        or report.get("request") != original.reference.json()
        or report.get("candidate") != original.candidate.json()
        or report.get("expectedIdentity") != original.identity.json()
        or assessment.get("trackEditMappingAccepted") is not True
        or assessment.get("packetDecodedPresentationAccepted") is not True
        or assessment.get("decodedStart") != "0"
        or assessment.get("source") != original.source.reference.json()
    ):
        raise ReplayError("Content diagnostic needs bound, continuous original presentation")
    end = Fraction(str(assessment.get("decodedEnd"))) * rate
    if end.denominator != 1 or assessment.get("decodedSamples") != int(end):
        raise ReplayError("Comparison rate differs from the actual decoded AAC clock")
    request = Request(
        reference, original, composition, aac._identity(row.get("sourceIdentity")), rate, int(end)
    )
    windows(request)
    return request


def windows(request: Request) -> tuple[Window, ...]:
    source = request.original.source
    if source is None or source.samples < WINDOW_SECONDS * source.sample_rate:
        raise ReplayError("Original PCM cannot support bounded diagnostic windows")
    count = WINDOW_SECONDS * source.sample_rate
    total = Fraction(source.samples * request.rate, source.sample_rate)
    if total.denominator != 1 or request.decoded_end > total:
        raise ReplayError("Original program has a fractional or unsupported comparison extent")
    quantum = source.sample_rate // math.gcd(source.sample_rate, request.rate)
    starts = (0, ((source.samples - count) // 2) // quantum * quantum, source.samples - count)
    if not starts[0] < starts[1] < starts[2]:
        raise ReplayError("Beginning, middle and tail windows must be distinct")
    result = []
    for name, start in zip(("first", "middle", "tail"), starts, strict=True):
        projected = Fraction(start * request.rate, source.sample_rate)
        if projected.denominator != 1:
            raise ReplayError("Diagnostic source window lacks an integral AAC boundary")
        first = int(projected)
        actual = min(WINDOW_SECONDS * request.rate, request.decoded_end - first)
        if actual < request.rate:
            raise ReplayError("AAC supports less than one second of a diagnostic window")
        result.append(Window(name, start, count, first, WINDOW_SECONDS * request.rate, actual))
    return tuple(result)


def _guard(request: Request) -> None:
    if load(request.reference) != request:
        raise ReplayError("Typed diagnostic request differs from its pinned bytes")
    aac._current_candidate(request.original)
    source = request.original.source
    if source is None:
        raise ReplayError("Original PCM is missing")
    aac._canonical(source.reference.path)
    if aac.FileIdentity.from_stat(source.reference.path.lstat()) != request.source_identity:
        raise ReplayError("Original PCM identity changed")
    receipt, _ = aac._original_provenance(request.original)
    streams = record(receipt.get("streams")).get("streams")
    if not isinstance(streams, list):
        raise ReplayError("Original stream inventory is missing")
    audios = [record(row) for row in streams if record(row).get("codec_type") == "audio"]
    if len(audios) != 1 or (
        audios[0].get("codec_name") != "aac"
        or audios[0].get("sample_rate") != str(request.rate)
        or audios[0].get("time_base") != f"1/{request.rate}"
        or audios[0].get("channels") != 2
    ):
        raise ReplayError("Diagnostic must bind the one actual AAC stereo sample clock")


def _pin(path: Path, *, empty: bool = False) -> aac.Reference:
    aac._canonical(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        data = stream.read(MAX_VIEW_BYTES + 1)
        if (
            not (0 if empty else 1) <= len(data) <= MAX_VIEW_BYTES
            or aac._file_facts(before) != aac._file_facts(os.fstat(stream.fileno()))
            or aac._file_facts(before) != aac._file_facts(path.lstat())
            or before.st_nlink != 1
        ):
            raise ReplayError("Diagnostic artifact changed or exceeds its bounded budget")
    return aac.Reference(path, hashlib.sha256(data).hexdigest())


def _source_view(request: Request, window: Window, output: Path) -> None:
    source = request.original.source
    if source is None:
        raise ReplayError("Original PCM is missing")
    with os.fdopen(os.open(source.reference.path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as descriptor:
        if aac.FileIdentity.from_stat(os.fstat(descriptor.fileno())) != request.source_identity:
            raise ReplayError("Original PCM FD changed")
        with wave.open(descriptor, "rb") as stream:
            if (
                stream.getframerate(),
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getnframes(),
                stream.getcomptype(),
            ) != (source.sample_rate, 2, 2, source.samples, "NONE"):
                raise ReplayError("Original WAV header differs from exact producer facts")
            stream.setpos(window.source_start)
            payload = stream.readframes(window.source_count)
            if len(payload) != window.source_count * 4:
                raise ReplayError("Original PCM view is truncated")
        if aac.FileIdentity.from_stat(os.fstat(descriptor.fileno())) != request.source_identity:
            raise ReplayError("Original PCM changed while reading a window")
    with os.fdopen(
        os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb"
    ) as target:
        with wave.open(target, "wb") as stream:
            stream.setparams((2, 2, source.sample_rate, 0, "NONE", "not compressed"))
            stream.writeframes(payload)
        target.flush()
        os.fsync(target.fileno())


def commands(request: Request, window: Window, directory: Path) -> dict[str, list[str]]:
    prefix = [
        str(request.original.tools[0].path),
        "-nostdin",
        "-hide_banner",
        "-nostats",
        "-v",
        "info",
        "-xerror",
        "-max_error_rate",
        "0",
        "-n",
        "-filter_threads",
        "1",
    ]
    input_options = ["-protocol_whitelist", "file,pipe", "-threads", "1"]
    source = (
        prefix
        + input_options
        + [
            "-i",
            str(directory / "source.wav"),
            "-map",
            "0:a:0",
            "-af",
            f"aresample={request.rate}:async=0,ashowinfo",
            "-c:a",
            "pcm_s16le",
            str(directory / "source-comparison.wav"),
        ]
    )
    seek = max(0, window.target_start // request.rate - PRE_ROLL_SECONDS)
    candidate = (
        prefix
        + ["-copyts"]
        + input_options
        + [
            "-seek_timestamp",
            "1",
            "-ss",
            str(seek),
            "-t",
            str(WINDOW_SECONDS + 2 * PRE_ROLL_SECONDS),
            "-i",
            str(request.original.candidate.path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-af",
            f"atrim=start_pts={window.target_start}:"
            f"end_pts={window.target_start + window.target_count},ashowinfo",
            "-c:a",
            "pcm_s16le",
            str(directory / "candidate.wav"),
        ]
    )
    return {"resample": source, "decode-window": candidate}


def _run(command: list[str], directory: Path, name: str) -> None:
    environment = dict(os.environ)
    environment.pop("FFREPORT", None)
    environment.update(
        TMPDIR=str(directory),
        TMP=str(directory),
        TEMP=str(directory),
        XDG_CACHE_HOME=str(directory),
    )
    started = time.monotonic()
    with (
        (directory / f"{name}.stdout").open("xb") as stdout,
        (directory / f"{name}.stderr").open("xb") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=aac.ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            timeout=PROCESS_SECONDS,
            check=False,
        )
    aac._exclusive(
        directory / f"{name}.process.json",
        aac._json_bytes(
            {
                "command": command,
                "returncode": completed.returncode,
                "seconds": time.monotonic() - started,
            }
        ),
    )
    if completed.returncode != 0:
        raise ReplayError("Diagnostic child failed; retained artifacts cannot be blindly retried")


def trace(payload: bytes, *, rate: int, start: int, count: int) -> None:
    if len(payload) > MAX_LOG_BYTES:
        raise ReplayError("Diagnostic log exceeds its byte budget")
    index, next_point = 0, start
    for line in payload.decode("utf-8").splitlines():
        if "[Parsed_ashowinfo_" not in line:
            continue
        match = FRAME_RE.search(line)
        if match is None:
            raise ReplayError("Actual decoded frame clock is missing")
        serial, pts, channels, actual_rate, samples = map(int, match.groups())
        if (
            serial != index
            or pts != next_point
            or channels != 2
            or actual_rate != rate
            or not 1 <= samples <= 65536
        ):
            raise ReplayError("Actual decoded diagnostic clock has a gap or differs")
        next_point += samples
        index += 1
    if index == 0 or next_point != start + count:
        raise ReplayError("Actual diagnostic frame extent differs from retained full clock")


def pcm(path: Path, *, rate: int, count: int) -> PCM:
    _pin(path)
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getnframes(),
            stream.getcomptype(),
        ) != (rate, 2, 2, count, "NONE"):
            raise ReplayError("Diagnostic WAV extent or format differs")
        data = stream.readframes(count)
    if len(data) != count * 4:
        raise ReplayError("Diagnostic WAV is truncated")
    return np.frombuffer(data, dtype="<i2").reshape(count, 2).astype(np.float64) / 32768


def level(values: PCM) -> dict[str, object]:
    if not values.size or not np.all(np.isfinite(values)):
        raise ReplayError("Signal level requires finite, nonempty PCM")
    return {
        "samples": len(values),
        "rms": float(np.sqrt(np.mean(values**2))),
        "peak": float(np.max(np.abs(values))),
        "exactZeroFraction": float(np.count_nonzero(values == 0) / values.size),
        "speechClassified": False,
    }


def compare(source: PCM, candidate: PCM, *, rate: int) -> dict[str, object]:
    if len(source) != WINDOW_SECONDS * rate or not rate <= len(candidate) <= len(source):
        raise ReplayError("Comparison views lack their exact supported extents")
    middle = source[rate // 2 : rate // 2 + rate]
    correspondence: dict[str, object]
    try:
        matched = match_window(middle, candidate, rate=rate)
        correspondence = {
            "status": "matched",
            "correlation": matched.correlation,
            "runnerUp": matched.runner_up,
            "candidateMinusSourceSamples": matched.offset_samples - rate // 2,
            "appliedToProduct": False,
        }
    except AlignmentRejected as error:
        correspondence = {
            "status": "not-established",
            "reason": str(error),
            "appliedToProduct": False,
        }
    missing = source[len(candidate) :]
    return {
        "sourceLevel": level(source),
        "candidateLevel": level(candidate),
        "nominalOverlapErrorRMS": float(
            np.sqrt(np.mean((source[: len(candidate)] - candidate) ** 2))
        ),
        "correspondence": correspondence,
        "missingSourceSamples": len(missing),
        "missingSourceLevel": None if not len(missing) else level(missing),
        "silenceOrSpeechAssumed": False,
    }


def _window_report(request: Request, window: Window, directory: Path) -> dict[str, object]:
    vectors = commands(request, window, directory)
    artifacts: list[dict[str, str]] = []
    for name, command in vectors.items():
        process = _pin(directory / f"{name}.process.json")
        report = aac._evidence(process).document()
        seconds = report.get("seconds")
        if (
            report.get("command") != command
            or type(report.get("returncode")) is not int
            or report.get("returncode") != 0
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ReplayError("Diagnostic process provenance differs")
        stdout, stderr = (
            _pin(directory / f"{name}.{kind}", empty=True) for kind in ("stdout", "stderr")
        )
        if aac._read(stdout, empty=True):
            raise ReplayError("Diagnostic child stdout must be empty")
        trace(
            aac._read(stderr),
            rate=request.rate,
            start=0 if name == "resample" else window.target_start,
            count=window.target_count if name == "resample" else window.actual_count,
        )
        artifacts.extend((process.json(), stdout.json(), stderr.json()))
    source_ref = _pin(directory / "source.wav")
    converted_ref = _pin(directory / "source-comparison.wav")
    candidate_ref = _pin(directory / "candidate.wav")
    source = pcm(converted_ref.path, rate=request.rate, count=window.target_count)
    candidate = pcm(candidate_ref.path, rate=request.rate, count=window.actual_count)
    return {
        "name": window.name,
        "sourceStartSample": window.source_start,
        "sourceSamples": window.source_count,
        "candidateStartPTS": window.target_start,
        "candidateSamples": window.actual_count,
        "candidateEndPTS": window.target_start + window.actual_count,
        "sourceResampleCount": 1,
        "candidateResampleCount": 0,
        "primingAppliedAgain": False,
        "sourceView": source_ref.json(),
        "comparisonView": converted_ref.json(),
        "candidateView": candidate_ref.json(),
        "processArtifacts": artifacts,
        "comparison": compare(source, candidate, rate=request.rate),
    }


def _report(request: Request, output: Path) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "aac-content-diagnostic-result",
        "request": request.reference.json(),
        "originalRequest": request.original.reference.json(),
        "composition": request.composition.json(),
        "source": None
        if request.original.source is None
        else request.original.source.reference.json(),
        "sourceIdentity": request.source_identity.json(),
        "candidate": request.original.candidate.json(),
        "candidateIdentity": request.original.identity.json(),
        "comparisonRate": request.rate,
        "windows": [
            _window_report(request, window, output.parent / window.name)
            for window in windows(request)
        ],
        "wholeMediaHashFreshlyVerified": False,
        "newFullDecode": False,
        "resampleFilterStateResetPerView": True,
        "temporaryPCM16Quantization": True,
        "sourceExtentAccepted": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }


def collect(request: Request, output: Path) -> None:
    _guard(request)
    aac._output_paths(output)
    paths = tuple(output.parent / window.name for window in windows(request))
    if (
        output.exists()
        or output.is_symlink()
        or any(path.exists() or path.is_symlink() for path in paths)
    ):
        raise FileExistsError(
            "Diagnostic outputs are occupied; verify completed evidence without rerunning"
        )
    for window, directory in zip(windows(request), paths, strict=True):
        directory.mkdir(mode=0o700)
        _source_view(request, window, directory / "source.wav")
        for name, command in commands(request, window, directory).items():
            _guard(request)
            _run(command, directory, name)
            _guard(request)
    report = _report(request, output)
    _guard(request)
    aac._exclusive(output, aac._json_bytes(report))
    _guard(request)


def verify(request: Request, output: Path) -> None:
    _guard(request)
    aac._output_paths(output)
    result = _pin(output)
    if aac._evidence(result).document() != _report(request, output):
        raise ReplayError("Diagnostic result differs from actual clocks, processes or content")
    aac._read(result)
    _guard(request)


def prepare(original_reference: aac.Reference, composition: aac.Reference, output: Path) -> None:
    """Pin current identities without hashing the whole movie or original PCM."""
    original = aac.load_observation_request(original_reference)
    if original.source is None:
        raise ReplayError("Diagnostic preparation requires an original PCM producer")
    receipt, _ = aac._original_provenance(original)
    inventory = record(receipt.get("streams")).get("streams")
    if not isinstance(inventory, list):
        raise ReplayError("Diagnostic preparation lacks stream observations")
    streams = [record(row) for row in inventory if record(row).get("codec_type") == "audio"]
    if len(streams) != 1:
        raise ReplayError("Diagnostic preparation requires exactly one audio stream")
    rate = streams[0].get("sample_rate")
    if not isinstance(rate, str) or not rate.isdecimal():
        raise ReplayError("Diagnostic preparation lacks an actual sample rate")
    aac._canonical(original.source.reference.path)
    identity = aac.FileIdentity.from_stat(original.source.reference.path.lstat())
    document = {
        "version": 1,
        "kind": "aac-content-diagnostic-request",
        "presentationRequest": original.reference.json(),
        "composition": composition.json(),
        "sourceIdentity": identity.json(),
        "comparisonRate": int(rate),
        "windowSeconds": WINDOW_SECONDS,
    }
    aac._output_paths(output)
    aac._exclusive(output, aac._json_bytes(document))
    _guard(load(_pin(output)))


def missing_native_range(
    window: Window, *, source_rate: int, comparison_rate: int
) -> tuple[Fraction, int, int]:
    """Cover the fractional boundary without inventing or discarding edge samples."""
    if (
        not 8000 <= source_rate <= 192000
        or not 8000 <= comparison_rate <= 192000
        or not 0 < window.actual_count < window.target_count
    ):
        raise ReplayError("Original-rate tail needs an actual missing comparison extent")
    boundary = Fraction(window.actual_count * source_rate, comparison_rate)
    if not 0 < boundary < window.source_count:
        raise ReplayError("Missing boundary exceeds retained original-rate support")
    return boundary, math.floor(boundary), math.ceil(boundary)


def native_tail(request: Request, *, parent: aac.Reference) -> dict[str, object]:
    """Read retained original-rate view bytes only; no decoder or resampler runs."""
    _guard(request)
    prior = aac._evidence(parent).document()
    if prior != _report(request, parent.path):
        raise ReplayError("Tail parent differs from its complete measured diagnostic")
    source = request.original.source
    if source is None:
        raise ReplayError("Original PCM is missing")
    window = windows(request)[-1]
    boundary, first, whole = missing_native_range(
        window, source_rate=source.sample_rate, comparison_rate=request.rate
    )
    view = _pin(parent.path.parent / window.name / "source.wav")
    values = pcm(view.path, rate=source.sample_rate, count=window.source_count)
    result = {
        "version": 1,
        "kind": "aac-original-rate-missing-tail-observation",
        "request": request.reference.json(),
        "parent": parent.json(),
        "sourceView": view.json(),
        "sourceRate": source.sample_rate,
        "comparisonRate": request.rate,
        "missingComparisonSamples": window.target_count - window.actual_count,
        "relativeSourceBoundary": str(boundary),
        "absoluteSourceBoundary": str(window.source_start + boundary),
        "coveringStartSampleFloor": window.source_start + first,
        "whollyUnsupportedStartSampleCeil": window.source_start + whole,
        "sourceEndSample": window.source_start + window.source_count,
        "partialEdgeSampleIncluded": first != whole,
        "coveringNativeLevel": level(values[first:]),
        "whollyUnsupportedNativeLevel": level(values[whole:]),
        "newDecode": False,
        "newResample": False,
        "wholeMediaHashFreshlyVerified": False,
        "speechClassified": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }
    if _pin(view.path) != view:
        raise ReplayError("Retained original-rate view changed")
    _guard(request)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "observe", "verify", "tail-observe", "tail-verify")
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--composition", type=Path)
    parser.add_argument("--composition-sha256")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-sha256")
    args = parser.parse_args(argv)
    try:
        reference = aac.Reference(args.request, args.sha256)
        if args.action == "prepare":
            if args.composition is None or args.composition_sha256 is None:
                raise ReplayError("Preparation needs both composition path and SHA256")
            prepare(
                reference, aac.Reference(args.composition, args.composition_sha256), args.output
            )
            return 0
        request = load(reference)
        if args.action.startswith("tail-"):
            if args.parent is None or args.parent_sha256 is None:
                raise ReplayError("Native tail observation needs parent path and SHA256")
            result = native_tail(request, parent=aac.Reference(args.parent, args.parent_sha256))
            aac._output_paths(args.output)
            if args.action == "tail-observe":
                aac._exclusive(args.output, aac._json_bytes(result))
            elif aac._evidence(_pin(args.output)).document() != result:
                raise ReplayError("Original-rate tail result differs from actual view evidence")
            return 0
        if args.action == "observe":
            collect(request, args.output)
        else:
            verify(request, args.output)
    except (ReplayError, OSError, ValueError, subprocess.TimeoutExpired, wave.Error) as error:
        parser.exit(1, f"AAC content diagnostic: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
