"""Plan lossless HEVC part assembly with one independently clocked audio track.

This does not submit Compressor jobs, retire sources, or approve editing quality.
Pinned measurement artifacts are required; pending parts remain resumable and any
measured rejection selects the declared ProRes workflow without launching it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from replay import whole_technical_batch as metadata
from replay.audio_clock import AudioFrameClock
from replay.caption_measured_job import _json as strict_json
from replay.compressed_part_retirement import verify as verify_retired
from replay.core import ReplayError
from replay.delivery_clock_contract import load_clock_contract
from replay.foreground_evidence import _bound_bytes
from replay.native_join import PartFile, concat_list
from replay.recordings import record
from replay.stages import file_digest
from replay.workflow import parse_stages, stage_graph

ROOT = Path(__file__).resolve().parents[2]
MAX_PARTS = 128
MAX_FRAMES = 4_000_000
MAX_TICKS = MAX_FRAMES * 1_000_000_000
MAX_PROBE_BYTES = 256 * 1024 * 1024
VIDEO_FIELDS = (
    "codec_name",
    "codec_tag_string",
    "profile",
    "level",
    "width",
    "height",
    "pix_fmt",
    "sample_aspect_ratio",
    "field_order",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "time_base",
    "extradata_hash",
)
METRICS = ("ssim", "psnr", "vmaf")
PART_PROOFS = ("sourceNative", "nativeStatus", "fullAV", "videoProbe", "firstPacket", "quality")
AUDIO_PROOFS = ("nativeStatus", "fullAV", "decodedProbe", "packetProbe")


@dataclass(frozen=True)
class CurrentOriginal:
    reference: metadata.Reference


@dataclass(frozen=True)
class RetiredOriginal:
    provenance: metadata.Reference
    inventory: tuple[metadata.Reference, ...]


@dataclass(frozen=True)
class Part:
    number: int
    start: int
    frames: int
    master: metadata.Reference
    movie: metadata.Reference | None
    evidence: metadata.Reference | None
    original: CurrentOriginal | RetiredOriginal


@dataclass(frozen=True)
class Plan:
    config: Path
    rate: int
    width: int
    height: int
    parts: tuple[Part, ...]
    audio: dict[str, object] | None
    thresholds: dict[str, float]
    fallback: metadata.Reference
    contract: metadata.Reference
    output: Path
    tools: dict[str, metadata.Reference]
    allow_clock_candidate: bool = False
    python_invocation: Path | None = None

    @property
    def frames(self) -> int:
        return sum(part.frames for part in self.parts)


def _count(value: object, *, maximum: int = MAX_FRAMES, zero: bool = False) -> int:
    if type(value) is not int or not (0 if zero else 1) <= value <= maximum:
        raise ReplayError("Compressed delivery requires a bounded integer")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ReplayError("Quality measurements must be finite numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayError("Quality measurements must be finite numbers")
    return result


def _point(value: object) -> int:
    if type(value) is not int or not -MAX_TICKS <= value <= MAX_TICKS:
        raise ReplayError("Measured PTS must be a bounded integer")
    return value


def _read(ref: metadata.Reference) -> dict[str, object]:
    return metadata._metadata(ref.path, ref.sha256)


def _read_probe(ref: metadata.Reference) -> dict[str, object]:
    return record(strict_json(_bound_bytes(ref.path, ref.sha256, MAX_PROBE_BYTES)))


def _ref(value: object) -> metadata.Reference:
    return metadata._reference(value)


def _python_invocation(value: object, target: metadata.Reference) -> Path | None:
    """Allow only an explicitly bound repo executable alias, never media aliases."""
    if value is None:
        return None
    row = record(value)
    if set(row) != {"path", "target", "linkTarget"} or row["target"] != target.json():
        raise ReplayError("Python invocation requires its pinned canonical executable")
    path = Path(str(row["path"]))
    root = metadata.ROOT.resolve()
    if (
        not path.is_absolute()
        or not path.is_relative_to(root)
        or any(part in {"..", "private", "downloads"} for part in path.parts)
        or path.parent.resolve(strict=True) != path.parent
        or not stat.S_ISLNK(path.lstat().st_mode)
        or not isinstance(row["linkTarget"], str)
        or os.readlink(path) != row["linkTarget"]
        or path.resolve(strict=True) != target.path
        or not os.access(path, os.X_OK)
        or file_digest(target.path) != target.sha256
    ):
        raise ReplayError("Python invocation alias/target changed or escaped this checkout")
    return path


def _fraction(value: object) -> Fraction:
    if not isinstance(value, str) or len(value) > 64:
        raise ReplayError("Measured clock must be bounded rational text")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Measured clock is invalid") from error
    if result <= 0 or result.denominator > 1_000_000_000:
        raise ReplayError("Measured clock exceeds its rational budget")
    return result


def _parts(value: object) -> tuple[Part, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PARTS:
        raise ReplayError("Compressed delivery requires ordered bounded parts")
    result: list[Part] = []
    cursor = 0
    for number, raw in enumerate(value, 1):
        row = record(raw)
        required = {"number", "startFrame", "frames", "master", "movie", "evidence"}
        if not required.issubset(row) or set(row).difference(required | {"retirement"}):
            raise ReplayError("Compressed part fields differ")
        count = _count(row["frames"])
        if (
            row["number"] != number
            or type(row["number"]) is not int
            or (_count(row["startFrame"], zero=True) != cursor)
        ):
            raise ReplayError("Compressed parts must preserve exact ordered offsets")
        movie = _ref(row["movie"]) if row["movie"] is not None else None
        evidence = _ref(row["evidence"]) if row["evidence"] is not None else None
        if (movie is None) != (evidence is None):
            raise ReplayError("Compressed movie and completed evidence must be paired")
        master = _ref(row["master"])
        original: CurrentOriginal | RetiredOriginal = CurrentOriginal(master)
        if row.get("retirement") is not None:
            if movie is None:
                raise ReplayError("Pending compression must retain its current original")
            provenance = _ref(row["retirement"])
            retired = verify_retired(provenance, source=master, movie=movie, part=number)
            if retired.rebuild_required is not True:
                raise ReplayError(
                    "Retired ProRes fallback must declare native rebuild prerequisites"
                )
            original = RetiredOriginal(provenance, retired.inventory)
        result.append(Part(number, cursor, count, master, movie, evidence, original))
        cursor += count
    _count(cursor)
    if len({part.master.path for part in result}) != len(result):
        raise ReplayError("Ordered source parts must be distinct")
    movies = [part.movie.path for part in result if part.movie is not None]
    if len(set(movies)) != len(movies):
        raise ReplayError("Compressed parts must be distinct")
    return tuple(result)


def load(path: Path) -> Plan:
    """Validate trusted generic configuration without media hashing or app calls."""
    path = metadata._path(str(path), work=True)
    raw = metadata._metadata(path)
    required = {
        "version",
        "rate",
        "width",
        "height",
        "parts",
        "audio",
        "thresholds",
        "fallback",
        "clockContract",
        "output",
        "tools",
    }
    if (
        not required.issubset(raw)
        or set(raw).difference(required | {"allowClockCandidate", "pythonInvocation"})
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ReplayError("Unsupported compressed-delivery configuration")
    if type(raw.get("allowClockCandidate", False)) is not bool:
        raise ReplayError("Clock candidate policy must be an explicit boolean")
    thresholds = record(raw["thresholds"])
    if not thresholds or not set(thresholds).issubset(METRICS):
        raise ReplayError("A nonempty declared quality-metric policy is required")
    limits = {name: _number(value) for name, value in thresholds.items()}
    if any(
        value <= 0 or (name == "ssim" and value > 1) or (name == "vmaf" and value > 100)
        for name, value in limits.items()
    ):
        raise ReplayError("Quality thresholds are outside their physical domain")
    tools = {name: _ref(value) for name, value in record(raw["tools"]).items()}
    if set(tools) != {"python", "ffmpeg", "ffprobe"}:
        raise ReplayError("Three pinned executables are required")
    output = metadata._path(raw["output"])
    parts = _parts(raw["parts"])
    if output.suffix != ".mp4" or output in {
        *(part.master.path for part in parts),
        *(part.movie.path for part in parts if part.movie is not None),
    }:
        raise ReplayError("Final MP4 must be distinct from every input")
    plan = Plan(
        path,
        _count(raw["rate"], maximum=240),
        _count(raw["width"], maximum=16384),
        _count(raw["height"], maximum=16384),
        parts,
        record(raw["audio"]) if raw["audio"] is not None else None,
        limits,
        _ref(raw["fallback"]),
        _ref(raw["clockContract"]),
        output,
        tools,
        raw.get("allowClockCandidate") is True,
        _python_invocation(raw.get("pythonInvocation"), tools["python"]),
    )
    expectation = load_clock_contract(plan.contract.path)
    if file_digest(plan.contract.path) != plan.contract.sha256 or (
        (expectation.rate, expectation.width, expectation.height, expectation.expected_frames)
        != (plan.rate, plan.width, plan.height, plan.frames)
        or not output.is_relative_to(expectation.delivery_root)
    ):
        raise ReplayError("Part assembly must retain the pinned edited clock and delivery storage")
    return plan


def _quality(
    raw: dict[str, object],
    *,
    plan: Plan,
    source: dict[str, str],
    candidate: dict[str, str],
    frames: int,
) -> tuple[list[str], list[dict[str, object]]]:
    if raw.get("source") != source or raw.get("candidate") != candidate:
        raise ReplayError("Quality evidence must bind the unchanged source and candidate")
    windows = raw.get("windows")
    if not isinstance(windows, list) or not 1 <= len(windows) <= MAX_PARTS * 3:
        raise ReplayError("Representative quality windows are required")
    failures: list[str] = []
    observations: list[dict[str, object]] = []
    for value in windows:
        window = record(value)
        left, right = _count(window.get("startFrame"), zero=True), _count(window.get("endFrame"))
        if not left < right <= frames:
            raise ReplayError("Quality window escapes the actual frame extent")
        measured = {name: _number(window.get(name)) for name in plan.thresholds}
        if any(
            value < 0 or (name == "ssim" and value > 1) or (name == "vmaf" and value > 100)
            for name, value in measured.items()
        ):
            raise ReplayError("Measured quality is outside its physical domain")
        observations.append({"startFrame": left, "endFrame": right, **measured})
        failures.extend(
            f"{name} {measured[name]} < {plan.thresholds[name]} at frames {left}..{right}"
            for name in plan.thresholds
            if measured[name] < plan.thresholds[name]
        )
    anchors = (0, frames // 2, frames - 1)
    if any(
        not any(
            _point(row["startFrame"]) <= point < _point(row["endFrame"]) for row in observations
        )
        for point in anchors
    ):
        raise ReplayError("Quality windows must represent the beginning, middle and end")
    return failures, observations


def _first_vcl(raw: dict[str, object], movie: metadata.Reference) -> int:
    if raw.get("movie") != movie.json() or raw.get("nalLengthBytes") != 4:
        raise ReplayError("First HEVC packet must bind this movie and four-byte NAL lengths")
    value = raw.get("dataHex")
    if not isinstance(value, str) or not 12 <= len(value) <= 4 * 1024 * 1024:
        raise ReplayError("First HEVC packet bytes must be bounded")
    try:
        payload = bytes.fromhex(value)
    except ValueError as error:
        raise ReplayError("HEVC packet bytes are invalid") from error
    cursor = 0
    first_vcl: int | None = None
    while cursor < len(payload):
        if len(payload) - cursor < 4:
            raise ReplayError("HEVC packet NAL length is truncated")
        size = int.from_bytes(payload[cursor : cursor + 4], "big")
        cursor += 4
        if size < 2 or cursor + size > len(payload):
            raise ReplayError("HEVC packet NAL length is invalid")
        nal = payload[cursor : cursor + size]
        if nal[0] & 0x80 or nal[1] & 7 == 0:
            raise ReplayError("HEVC NAL header is invalid")
        kind = (nal[0] >> 1) & 63
        if kind <= 31 and first_vcl is None:
            first_vcl = kind
        cursor += size
    if first_vcl is not None:
        return first_vcl
    raise ReplayError("HEVC first packet contains no coded picture")


def _part_check(
    part: Part, plan: Plan
) -> tuple[dict[str, object], list[str], list[dict[str, object]]]:
    if part.movie is None or part.evidence is None:
        raise ReplayError("Pending compressed part has no completed evidence")
    raw = _read(part.evidence)
    if raw.get("kind") == "converted-compressed-part-evidence":
        return _converted_part(part, plan, raw)
    if (
        set(raw) != {"version", "kind", "source", "movie", *PART_PROOFS}
        or (
            raw.get("version") != 1
            or type(raw.get("version")) is not int
            or raw.get("kind") != "compressed-part-evidence"
        )
        or (raw.get("source") != part.master.json() or raw.get("movie") != part.movie.json())
    ):
        raise ReplayError("Compressed part provenance differs")
    native = _read(_ref(raw.get("nativeStatus")))
    source_native = _read(_ref(raw.get("sourceNative")))
    full = _read(_ref(raw.get("fullAV")))
    probe = _read_probe(_ref(raw.get("videoProbe")))
    failures: list[str] = []
    if source_native.get("movie") != part.master.json() or (
        source_native.get("completionVerified") is not True
        or source_native.get("shareObservedIdle") is not True
        or source_native.get("fullAVDecodeExitCode") != 0
        or type(source_native.get("fullAVDecodeExitCode")) is not int
    ):
        failures.append("source full production part has no bound native completion/full AV proof")
    if native.get("status") != "Successful" or native.get("output") != str(part.movie.path):
        failures.append("native Compressor success/output not verified")
    if native.get("source") != str(part.master.path):
        failures.append("native Compressor source binding differs")
    if (
        full.get("movie") != part.movie.json()
        or full.get("exitCode") != 0
        or (type(full.get("exitCode")) is not int)
    ):
        failures.append("full AV decode failed or was not bound to this movie")
    streams, frames = probe.get("streams"), probe.get("frames")
    if (
        probe.get("movie") != part.movie.json()
        or not isinstance(streams, list)
        or (len(streams) != 1 or not isinstance(frames, list) or len(frames) > MAX_FRAMES)
    ):
        raise ReplayError("Complete video-only probe observations are required")
    video = record(streams[0])
    if video.get("codec_name") != "hevc" or video.get("codec_tag_string") != "hvc1":
        failures.append("compressed video must be HEVC/hvc1")
    if any(video.get(key) is None for key in VIDEO_FIELDS):
        raise ReplayError("Complete codec configuration observations are required")
    base = _fraction(video["time_base"])
    if base.numerator != 1 or base <= 0 or _fraction(video.get("avg_frame_rate")) != plan.rate:
        failures.append("frame rate or mux time base differs")
    if (video.get("width"), video.get("height")) != (plan.width, plan.height):
        failures.append("video geometry differs")
    extent = Fraction(part.frames, plan.rate)
    if (
        video.get("start_pts") != 0
        or video.get("duration_ts") is None
        or (
            _count(video["duration_ts"], maximum=MAX_TICKS) * base != extent
            or len(frames) != part.frames
        )
    ):
        failures.append(f"video duration/frame count differs from {part.frames}/{plan.rate}")
    for index, value in enumerate(frames):
        frame = record(value)
        if _point(frame.get("pts")) * base != Fraction(index, plan.rate) or (
            _point(frame.get("duration")) * base != Fraction(1, plan.rate)
        ):
            failures.append(f"video PTS/duration differs at frame {index}")
            break
    first = _read(_ref(raw.get("firstPacket")))
    if _first_vcl(first, part.movie) not in (19, 20):
        failures.append("first coded picture has no verified HEVC IDR")
    quality, observations = _quality(
        _read(_ref(raw.get("quality"))),
        plan=plan,
        source=part.master.json(),
        candidate=part.movie.json(),
        frames=part.frames,
    )
    return video, [*failures, *quality], observations


def _converted_part(
    part: Part, plan: Plan, raw: dict[str, object]
) -> tuple[dict[str, object], list[str], list[dict[str, object]]]:
    from replay import compressed_part_observation as observer

    if (
        set(raw)
        != {
            "version",
            "kind",
            "source",
            "movie",
            "sourceNative",
            "converterExecution",
            "observation",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or part.movie is None
        or raw["source"] != part.master.json()
        or raw["movie"] != part.movie.json()
    ):
        raise ReplayError("Converted part provenance differs")
    reference = _ref(raw["observation"])
    saved = _read(reference)
    request = observer.load(_ref(saved.get("request")))
    if (
        request.converter != _ref(raw["converterExecution"])
        or request.native != _ref(raw["sourceNative"])
        or request.source != part.master
        or request.movie != part.movie
        or request.frames != part.frames
        or request.rate != plan.rate
        or (request.video["width"], request.video["height"]) != (plan.width, plan.height)
    ):
        raise ReplayError("Converted observation does not bind the selected delivery part")
    if reference.path != reference.path.parent / "observation.json":
        raise ReplayError("Converted observation receipt namespace differs")
    observed = observer.verify(request, reference.path.parent)
    if _read(reference) != observed:
        raise ReplayError("Converted observation receipt changed")
    failures, windows = _quality(
        {
            "source": part.master.json(),
            "candidate": part.movie.json(),
            "windows": observed["qualityWindows"],
        },
        plan=plan,
        source=part.master.json(),
        candidate=part.movie.json(),
        frames=part.frames,
    )
    return record(observed["videoConfiguration"]), failures, windows


def _audio_check(plan: Plan) -> list[str]:
    if plan.audio is None:
        return []
    raw = plan.audio
    if set(raw) != {"pcm", "movie", "evidence"}:
        raise ReplayError("Exactly one whole PCM and one encoded audio are required")
    pcm, movie = _ref(raw["pcm"]), _ref(raw["movie"])
    proof = _read(_ref(raw["evidence"]))
    if set(proof) != {"source", "movie", "encodingPasses", *AUDIO_PROOFS} or (
        proof.get("source") != pcm.json() or proof.get("movie") != movie.json()
    ):
        raise ReplayError("Whole audio source/output binding differs")
    if proof.get("encodingPasses") != 1 or type(proof.get("encodingPasses")) is not int:
        return ["whole audio must be encoded once from PCM"]
    native = _read(_ref(proof.get("nativeStatus")))
    full = _read(_ref(proof.get("fullAV")))
    decoded = _read_probe(_ref(proof.get("decodedProbe")))
    packets = _read_probe(_ref(proof.get("packetProbe")))
    if decoded.get("movie") != movie.json() or packets.get("movie") != movie.json():
        raise ReplayError("Decoded and packet audio clocks must bind the whole audio")
    clock = AudioFrameClock.from_probe(decoded)
    samples, rate = clock.sample_ends[-1], clock.sample_rate
    expected = Fraction(plan.frames, plan.rate)
    failures: list[str] = []
    if (
        native.get("status") != "Successful"
        or native.get("source") != str(pcm.path)
        or (native.get("output") != str(movie.path))
    ):
        failures.append("whole audio native source/success/output not verified")
    if (
        full.get("movie") != movie.json()
        or full.get("exitCode") != 0
        or (type(full.get("exitCode")) is not int)
    ):
        failures.append("whole audio full decode unverified")
    if any(
        (right - left) * clock.time_base != Fraction(count, rate)
        for (left, right), count in zip(pairwise(clock.pts), clock.samples, strict=False)
    ):
        failures.append("whole audio decoded clock has gaps or overlaps")
    end = clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], rate)
    if clock.pts[0] != 0 or Fraction(samples, rate) != expected or end != expected:
        failures.append(
            f"whole audio clock differs: {samples}/{rate}, end {end}, expected {expected}"
        )
    if not _audio_packets(packets, clock):
        failures.append("AAC packet priming/remainder and decoded extent differ")
    return failures


def _skip(value: dict[str, object], field: str) -> int:
    rows = value.get("side_data_list", [])
    if not isinstance(rows, list) or len(rows) > 16:
        raise ReplayError("AAC side data must be bounded")
    support = [record(row) for row in rows if record(row).get("side_data_type") == "Skip Samples"]
    if len(support) > 1:
        raise ReplayError("AAC skip sample observations are ambiguous")
    return _count(support[0].get(field), maximum=65536, zero=True) if support else 0


def _audio_packets(raw: dict[str, object], clock: AudioFrameClock) -> bool:
    streams, packets = raw.get("streams"), raw.get("packets")
    if (
        not isinstance(streams, list)
        or len(streams) != 1
        or not isinstance(packets, list)
        or (not 1 <= len(packets) <= 1_000_000)
    ):
        raise ReplayError("Complete bounded AAC packet observations are required")
    stream = record(streams[0])
    if (
        stream.get("codec_name") != "aac"
        or stream.get("profile") != "LC"
        or (stream.get("sample_rate") != str(clock.sample_rate) or stream.get("channels") != 2)
    ):
        return False
    base = _fraction(stream.get("time_base"))
    rows = tuple(record(value) for value in packets)
    if any(
        (_point(left.get("pts")) + _count(left.get("duration"), maximum=65536)) * base
        != _point(right.get("pts")) * base
        for left, right in pairwise(rows)
    ):
        return False
    first, last = rows[0], rows[-1]
    start = _point(first.get("pts")) * base + Fraction(
        _skip(first, "skip_samples"), clock.sample_rate
    )
    end = (_point(last.get("pts")) + _count(last.get("duration"), maximum=65536)) * base
    end -= Fraction(_skip(last, "discard_padding"), clock.sample_rate)
    return start == clock.pts[0] * clock.time_base and (
        end == clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
    )


def assess(plan: Plan) -> dict[str, object]:
    """Return pending, compressed-ready or ProRes-fallback with measured reasons."""
    pending = [part.number for part in plan.parts if part.movie is None]
    observations: list[dict[str, object]] = []
    failures: list[str] = []
    signature: dict[str, object] | None = None
    for part in plan.parts:
        if part.movie is None:
            continue
        video, rejected, measured = _part_check(part, plan)
        current = {key: video[key] for key in VIDEO_FIELDS}
        if signature is not None and current != signature:
            rejected.append("mixed HEVC profiles/codec configuration/time bases")
        signature = current if signature is None else signature
        failures.extend(f"part {part.number}: {reason}" for reason in rejected)
        observations.append({"part": part.number, "qualityWindows": measured})
    audio_failures = _audio_check(plan)
    observed_clock_candidate = (
        bool(audio_failures)
        and not failures
        and all(reason.startswith("whole audio clock differs:") for reason in audio_failures)
        and plan.allow_clock_candidate
    )
    failures.extend(audio_failures)
    route = (
        "compressed-clock-candidate"
        if observed_clock_candidate and not pending
        else "prores-fallback"
        if failures
        else ("pending" if pending or plan.audio is None else "compressed-ready")
    )
    return {
        "version": 1,
        "route": route,
        "pendingParts": pending,
        "audioPending": plan.audio is None,
        "reasons": failures,
        "observations": observations,
        "videoConfiguration": signature,
        "fallback": plan.fallback.json(),
        "fallbackRebuildRequired": any(
            isinstance(part.original, RetiredOriginal) for part in plan.parts
        ),
        "frames": plan.frames,
        "rate": plan.rate,
        "assemblyReady": route in {"compressed-ready", "compressed-clock-candidate"},
        "strictAccepted": route == "compressed-ready",
        "deliveryApproved": False,
        "finalVerificationRequired": [
            "complete joined frame PTS",
            "full AV decode",
            "single audio priming/payload clock",
            "joined representative SSIM/PSNR/VMAF",
            "audio/video synchronization",
        ],
    }


def _references(plan: Plan) -> tuple[metadata.Reference, ...]:
    if (
        _python_invocation(
            metadata._metadata(plan.config).get("pythonInvocation"), plan.tools["python"]
        )
        != plan.python_invocation
    ):
        raise ReplayError("Python invocation binding changed after loading")
    references = [
        metadata.Reference(plan.config, file_digest(plan.config)),
        plan.fallback,
        plan.contract,
        *plan.tools.values(),
    ]
    references.extend(
        metadata.Reference(ref.path, ref.sha256)
        for ref in load_clock_contract(plan.contract.path).inputs
    )
    for part in plan.parts:
        if isinstance(part.original, CurrentOriginal):
            references.append(part.master)
        else:
            references.extend((part.original.provenance, *part.original.inventory))
        if part.movie is not None and part.evidence is not None:
            references.extend((part.movie, part.evidence))
            row = _read(part.evidence)
            if row.get("kind") == "converted-compressed-part-evidence":
                from replay import compressed_part_observation as observer

                reference = _ref(row["observation"])
                request = observer.load(_ref(_read(reference).get("request")))
                observed = observer.verify(request, reference.path.parent)
                references.extend(
                    _ref(row.get(name))
                    for name in ("sourceNative", "converterExecution", "observation")
                )
                values = observed["inputs"]
                if not isinstance(values, list):
                    raise ReplayError("Converted observation input closure differs")
                references.extend(_ref(value) for value in values)
                artifacts = observed["artifacts"]
                if not isinstance(artifacts, list):
                    raise ReplayError("Converted raw observation closure differs")
                # Empty successful stdout/stderr remain freshly verified by the
                # observer; ordinary workflow input digests require nonempty files.
                references.extend(
                    ref
                    for ref in (_ref(value) for value in artifacts)
                    if ref.path.stat().st_size > 0
                )
                references.extend(_ref(observed[name]) for name in ("initial", "terminal"))
            else:
                references.extend(_ref(row.get(name)) for name in PART_PROOFS)
    if plan.audio is not None:
        references.extend(_ref(plan.audio[name]) for name in ("pcm", "movie", "evidence"))
        row = _read(_ref(plan.audio["evidence"]))
        references.extend(_ref(row[name]) for name in AUDIO_PROOFS)
    unique: dict[Path, metadata.Reference] = {}
    for reference in references:
        previous = unique.get(reference.path)
        if previous is not None and previous.sha256 != reference.sha256:
            raise ReplayError("Conflicting reference hashes name one file")
        unique[reference.path] = reference
    if any(
        part.master.path in unique
        for part in plan.parts
        if isinstance(part.original, RetiredOriginal)
    ):
        raise ReplayError("Retired original remains a materialized input dependency")
    return tuple(unique[path] for path in sorted(unique))


def admit(plan: Plan) -> dict[str, object]:
    """Check every expected media, proof, tool and clock hash before assembly."""
    references = _references(plan)
    for reference in references:
        metadata._path(str(reference.path))
        if file_digest(reference.path) != reference.sha256:
            raise ReplayError(f"Compressed assembly input hash changed: {reference.path.name}")
    if any(not os.access(ref.path, os.X_OK) for ref in plan.tools.values()):
        raise ReplayError("Pinned assembly tools must be executable")
    # Validate the fallback as a real workflow without invoking its producers.
    stage_graph(parse_stages(_read(plan.fallback)))
    report = assess(plan)
    report["inputsVerified"] = [ref.json() for ref in references]
    return report


def part_acceptance(config: Path, *, number: int) -> dict[str, object]:
    """Pin one actual compressed-part comparison before any ProRes retirement."""
    plan = load(config)
    number = _count(number, maximum=len(plan.parts))
    part = plan.parts[number - 1]
    if (
        not isinstance(part.original, CurrentOriginal)
        or part.movie is None
        or part.evidence is None
    ):
        raise ReplayError(
            "New part acceptance requires a materialized original and compressed evidence"
        )
    selected = Plan(
        plan.config,
        plan.rate,
        plan.width,
        plan.height,
        (part,),
        None,
        plan.thresholds,
        plan.fallback,
        plan.contract,
        plan.output,
        plan.tools,
        python_invocation=plan.python_invocation,
    )
    references = (
        *_references(selected),
        *(metadata.Reference(path, file_digest(path)) for path in code_inputs()),
    )
    for ref in references:
        if file_digest(ref.path) != ref.sha256:
            raise ReplayError("Compressed part acceptance input hash changed")
    video, reasons, windows = _part_check(part, plan)
    return {
        "version": 1,
        "kind": "accepted-compressed-part",
        "part": part.number,
        "source": part.master.json(),
        "movie": part.movie.json(),
        "evidence": part.evidence.json(),
        "accepted": not reasons,
        "reasons": reasons,
        "qualityWindows": windows,
        "videoConfiguration": {name: video[name] for name in VIDEO_FIELDS},
        "inputs": [ref.json() for ref in references],
        "deliveryApproved": False,
    }


def code_inputs() -> tuple[Path, ...]:
    """Declare the shared producer and verifier implementation closure."""
    names = (
        "compressed_part_delivery.py",
        "compressed_audio_verification.py",
        "compressed_part_retirement.py",
        "whole_technical_batch.py",
        "audio_clock.py",
        "delivery_clock_contract.py",
        "delivery_verification.py",
        "native_join.py",
        "workflow.py",
        "stages.py",
        "recordings.py",
        "core.py",
        "input_hash_cache.py",
        "delivery_render.py",
        "edit_completion.py",
        "fcp_caption_srt.py",
        "caption_io.py",
        "caption_font_supplier.py",
        "caption_measured_job.py",
        "foreground_evidence.py",
        "tracking_observations.py",
        "source_caption_binding.py",
        "stage_recovery.py",
    )
    return tuple(Path(__file__).parent / name for name in names)


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare(config: Path, output: Path) -> dict[str, object]:
    """Reserve small reusable graph artifacts; never launch apps or touch media."""
    plan = load(config)
    output = metadata._path(str(output), work=True)
    report = assess(plan)
    if plan.output.exists() or plan.output.is_symlink():
        raise ReplayError("Refusing to overwrite a final output")
    output.mkdir(mode=0o700)
    _write(output / "selection.json", report)
    if report["assemblyReady"] is not True:
        return report
    movies = tuple(part.movie for part in plan.parts if part.movie is not None)
    base = _fraction(record(report["videoConfiguration"])["time_base"])
    script = output / "parts.ffconcat"
    script.write_text(
        concat_list(
            [PartFile(ref.path, part.frames) for ref, part in zip(movies, plan.parts, strict=True)],
            plan.rate,
            base,
        ),
        encoding="utf-8",
    )
    if plan.audio is None:
        raise ReplayError("Assembly requires one completed whole audio")
    audio = _ref(plan.audio["movie"])
    ffmpeg = str(plan.tools["ffmpeg"].path)
    python = str(plan.python_invocation or plan.tools["python"].path)
    inputs = sorted(
        {str(ref.path) for ref in _references(plan)} | {str(path) for path in code_inputs()}
    )
    # Stage artifacts remain regular canonical files. The pinned config retains
    # the lexical invocation and every load/admission rechecks its exact binding.
    readiness = output / "ready.json"
    observation = output / "whole-observation"
    observation_files = [
        str(observation / name)
        for name in (
            "observation.json",
            "combined.stdout",
            "combined.process.json",
            "input-snapshot.json",
        )
    ]
    observe = [
        python,
        "-m",
        "replay.compressed_audio_verification",
        str(config),
        str(observation),
        "--observe",
    ]
    technical_result = output / "whole-technical.json"
    audio_result = output / "audio-mux-audit.json"
    derive = [python, "-m", "replay.compressed_audio_verification", str(config), str(observation)]
    technical_command = [*derive, str(technical_result), "--technical-result"]
    audio_command = [*derive, str(audio_result), "--audio-result"]
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-copyts",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(script),
        "-i",
        str(audio.path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-tag:v",
        "hvc1",
        "-video_track_timescale",
        str(base.denominator),
        "-movflags",
        "+faststart",
        str(plan.output),
    ]
    workflow = {
        "version": 1,
        "stages": [
            {
                "name": "compressed-part-admission",
                "command": [
                    python,
                    "-m",
                    "replay.compressed_part_delivery",
                    str(config),
                    "--admit-output",
                    str(readiness),
                ],
                "inputs": inputs,
                "outputs": [str(readiness)],
                "timeout_seconds": 7200,
                "verify_command": [
                    python,
                    "-m",
                    "replay.compressed_part_delivery",
                    str(config),
                    "--verify-admission",
                    str(readiness),
                ],
            },
            {
                "name": "compressed-lossless-join",
                "command": command,
                "inputs": [*inputs, str(script), str(readiness)],
                "outputs": [str(plan.output), *observation_files],
                "timeout_seconds": 7200,
                "verify_command": observe,
            },
            {
                "name": "compressed-whole-technical",
                "command": technical_command,
                "inputs": [*inputs, str(plan.output), *observation_files],
                "outputs": [str(technical_result)],
                "timeout_seconds": 7200,
                "verify_command": technical_command,
            },
            {
                "name": "compressed-whole-audio",
                "command": audio_command,
                "inputs": [
                    *inputs,
                    str(plan.output),
                    *observation_files,
                    str(technical_result),
                ],
                "outputs": [str(audio_result)],
                "timeout_seconds": 7200,
                "verify_command": audio_command,
            },
        ],
    }
    stage_graph(parse_stages(workflow))
    _write(output / "workflow.json", workflow)
    _write(
        output / "pins.json",
        {
            "config": {"path": str(plan.config), "sha256": file_digest(plan.config)},
            "tools": {name: ref.json() for name, ref in plan.tools.items()},
            "movies": [ref.json() for ref in movies],
            "audio": audio.json(),
            "clockContract": plan.contract.json(),
            "deliveryApproved": False,
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", type=Path)
    action.add_argument("--admit-output", type=Path)
    action.add_argument("--verify-admission", type=Path)
    args = parser.parse_args(argv)
    if args.prepare is not None:
        result = prepare(args.config, args.prepare)
    elif args.admit_output is not None:
        result = admit(load(args.config))
        _write(metadata._path(str(args.admit_output), work=True), result)
    elif args.verify_admission is not None:
        result = admit(load(args.config))
        if metadata._metadata(metadata._path(str(args.verify_admission), work=True)) != result:
            raise ReplayError("Saved admission differs from current source-bound evidence")
    else:
        result = assess(load(args.config))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["assemblyReady"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
