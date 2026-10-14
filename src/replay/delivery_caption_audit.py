"""Read-only Motion alpha audit and frame-derived SRT; never submits render jobs.

Current fingerprints are not historical submission evidence. Missing constraints,
submission bindings or partial coverage fail closed. Alpha intervals retain gaps;
this receipt is not the old whole-video-part/Executor completion schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import selectors
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay import (
    caption_resource_inputs,
    caption_style_defaults,
    compressor_batch,
    delivery_captions,
    motion_dependencies,
    motion_finish,
    motion_stages,
)
from replay import delivery_verification as verification
from replay.caption_measured_job import _pairs
from replay.caption_timing import display_time
from replay.compressor_batch import JOB_ID, Job, parse_snapshot, parse_submission, read_jobs
from replay.core import ReplayError
from replay.delivery_render import Config, load_config, rendered_parts, segments
from replay.fcp_measured import read_measured_captions
from replay.fcp_project import frame_span
from replay.motion_caption import MAX_XML_CHARACTERS
from replay.recordings import record

MAX_JSON = 64 * 1024 * 1024
MAX_FRAMES = 7200
MAX_PIXELS = 4096 * 2160
MAX_PARTS = 20_000
MAX_ROWS = 1_000_000
ERRORS = (ReplayError, ValueError, OSError, KeyError, subprocess.SubprocessError)


@dataclass(frozen=True)
class Cue:
    start: int
    end: int
    text: str
    x: float
    y: float


@dataclass(frozen=True)
class Part:
    name: str
    plan: Path
    source: Path
    offset: int
    frames: int
    cues: tuple[Cue, ...]
    envelopes: tuple[tuple[float, float, float, float], ...] = ()


class Inputs:
    """Memoize current hashes within a run; recheck file identities before saving."""

    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}
        self.missing: list[str] = []

    def bind(self, path: Path, expected: object = None) -> dict[str, object]:
        key = str(verification._canonical(path))
        if key not in self.values:
            self.values[key] = verification._fingerprint(Path(key))
        result = self.values[key]
        if expected is not None and result["sha256"] != expected:
            raise ReplayError(f"Input fingerprint mismatch: {path}")
        return result

    def reference(self, value: object) -> Path:
        row = record(value)
        path = absolute(row.get("path"))
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ReplayError("Reference requires SHA256")
        self.bind(path, digest)
        return path

    def verify(self) -> None:
        for name, row in self.values.items():
            stat = Path(name).stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                row["device"],
                row["inode"],
                row["bytes"],
                row["mtimeNs"],
                row["ctimeNs"],
            ):
                raise ReplayError(f"Input changed during audit: {name}")

    def rows(self) -> list[dict[str, object]]:
        return [
            {key: row[key] for key in ("path", "sha256", "bytes")} for row in self.values.values()
        ]


def absolute(value: object) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ReplayError("Expected absolute input path")
    return verification._canonical(Path(value))


def local(path: Path) -> Path:
    path = verification._canonical(path)
    if not path.is_relative_to(verification.REPOSITORY):
        raise ReplayError("Audit outputs/configuration must stay in checkout")
    return path


def document(path: Path, inputs: Inputs) -> dict[str, object]:
    inputs.bind(path)
    if path.stat().st_size > MAX_JSON:
        raise ReplayError("JSON byte budget exceeded")
    return record(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs))


def rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) > MAX_ROWS:
        raise ReplayError("Expected bounded JSON array")
    return [record(row) for row in value]


def integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReplayError("Expected integer frame/count")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError("Expected finite geometry")
    return float(value)


def cue(row: dict[str, object], frames: int) -> Cue:
    start, end = integer(row.get("startFrame")), integer(row.get("endFrame"))
    text = row.get("text")
    if not 0 <= start < end <= frames or not isinstance(text, str) or not text.strip():
        raise ReplayError("Invalid Motion stage clock/text")
    if len(text) > 32768 or any(ord(c) < 32 and c != "\n" for c in text) or "\n\n" in text:
        raise ReplayError("Invalid SRT text")
    return Cue(start, end, text, number(row.get("x")), number(row.get("y")))


def ordered(cues: tuple[Cue, ...]) -> None:
    end = 0
    for item in cues:
        if item.start < end:
            raise ReplayError("Overlapping/reversed Motion stages")
        end = item.end


def load_parts(config: Config, inputs: Inputs) -> tuple[list[Part], int]:
    inputs.bind(config.path)
    inputs.bind(config.captions / "segments.json")
    inputs.bind(config.cuts)
    total = verification._expected(config)
    parts: list[Part] = []
    next_segment = 0
    previous_offset = -1
    for segment in segments(config):
        if segment["offset"] <= previous_offset or segment["offset"] + segment["frames"] > total:
            raise ReplayError("Segment extent/order differs from cut output")
        previous_offset = segment["offset"]
        next_segment = max(next_segment, segment["offset"] + segment["frames"])
        directory = config.work / "motion" / f"segment-{segment['segment']:04d}"
        manifest = document(directory / "plans" / "manifest.json", inputs)
        if (manifest.get("frames"), manifest.get("rate")) != (segment["frames"], config.rate):
            raise ReplayError("Motion manifest clock mismatch")
        candidate = inputs.reference(manifest.get("candidate"))
        if candidate != config.captions / f"combined-{segment['segment']:04d}.json":
            raise ReplayError("Motion candidate differs from configured captions")
        combined = document(candidate, inputs)
        bindings = record(combined.get("inputs"))
        for name, digest in bindings.items():
            try:
                inputs.bind(absolute(name), digest)
            except (ReplayError, OSError) as error:
                inputs.missing.append(str(error))
        calibration = inputs.reference(manifest.get("renderCalibration"))
        if calibration != config.calibration or combined.get("renderCalibration") != manifest.get(
            "renderCalibration"
        ):
            raise ReplayError("Measured candidate/calibration selection differs from renderConfig")
        envelopes = measured_envelopes(combined, calibration, inputs)
        measured = read_measured_captions(
            candidate,
            sha256=str(inputs.bind(candidate)["sha256"]),
            width=config.width,
            height=config.height,
            rate=config.rate,
            duration=Fraction(segment["frames"], config.rate),
        )
        expected: list[Cue] = []
        for item in measured:
            span = frame_span(display_time(item.cue.start), display_time(item.cue.end), config.rate)
            if span is None:
                raise ReplayError("Measured caption has no display frame")
            expected.append(
                Cue(
                    int(span[0] * config.rate),
                    int(span[1] * config.rate),
                    item.cue.text,
                    item.x - config.width / 2,
                    config.height / 2 - item.y,
                )
            )
        actual: list[Cue] = []
        end = 0
        for index, entry in enumerate(rows(manifest.get("parts"))):
            plan = inputs.reference(entry)
            raw = document(plan, inputs)
            frames = integer(entry.get("frames"), minimum=1)
            offset = integer(entry.get("offsetFrame"))
            if (
                (raw.get("width"), raw.get("height"), raw.get("rate"), raw.get("frames"))
                != (config.width, config.height, config.rate, frames)
                or frames > MAX_FRAMES
                or offset < end
                or offset + frames > segment["frames"]
            ):
                raise ReplayError("Invalid alpha span/canvas")
            states = tuple(cue(row, frames) for row in rows(raw.get("placements")))
            ordered(states)
            if not states or len(states) != entry.get("stages"):
                raise ReplayError("Missing full-caption states")
            actual.extend(Cue(c.start + offset, c.end + offset, c.text, c.x, c.y) for c in states)
            parts.append(
                Part(
                    f"alpha-{segment['segment']:04d}-{index:04d}",
                    plan,
                    directory / f"part-{index:04d}.moti",
                    segment["offset"] + offset,
                    frames,
                    states,
                    tuple(
                        (
                            envelopes[c.text][0] + c.x + config.width / 2,
                            envelopes[c.text][1] + config.height / 2 - c.y,
                            envelopes[c.text][2] + c.x + config.width / 2,
                            envelopes[c.text][3] + config.height / 2 - c.y,
                        )
                        for c in states
                    )
                    if all(c.text in envelopes for c in states)
                    else (),
                )
            )
            end = offset + frames
        if actual != expected:
            raise ReplayError("Motion text/order/frames/placements differ from measured candidate")
    if next_segment != total or not 0 < len(parts) <= MAX_PARTS:
        raise ReplayError("Incomplete segment/cut coverage")
    return parts, total


def measured_envelopes(
    combined: dict[str, object], calibration: Path, inputs: Inputs
) -> dict[str, tuple[float, float, float, float]]:
    """Reconstruct calibrated bounds from retained native measurements, no helper calls."""
    data = document(calibration, inputs)
    font, style = record(data.get("nativeFont")), record(data.get("style"))
    top = -number(data.get("padding")) - number(font.get("renderedAscent"))
    advance = number(style.get("motion_line_advance"))
    metrics: dict[str, tuple[float, float]] = {}
    for name, digest in record(combined.get("inputs")).items():
        if not name.endswith(".response.json"):
            continue
        try:
            path = absolute(name)
            inputs.bind(path, digest)
            response = document(path, inputs)
            if "bounds" not in response or response.get("font") != font:
                continue
            request_path = Path(name.removesuffix(".response.json") + ".request.json")
            request_digest = record(combined.get("inputs")).get(str(request_path))
            if request_digest is None:
                raise ReplayError("Retained metric lacks bound native request")
            inputs.bind(request_path, request_digest)
            request = document(request_path, inputs)
            expected_request = {
                "fontName": style.get("font_name"),
                "assFontSize": style.get("ass_font_size"),
                "border": style.get("border"),
                "rim": style.get("rim"),
                "texts": [row.get("text") for row in rows(response["bounds"])],
            }
            if request != expected_request:
                raise ReplayError("Retained native request differs from calibrated style/text")
            for row in rows(response["bounds"]):
                text = row.get("text")
                size = number(row.get("width")), number(row.get("height"))
                if not isinstance(text, str) or min(size) <= 0:
                    raise ReplayError("Invalid retained native metric")
                if text in metrics and metrics[text] != size:
                    raise ReplayError("Conflicting retained native metrics")
                metrics[text] = size
        except (ReplayError, OSError) as error:
            inputs.missing.append(str(error))
    result: dict[str, tuple[float, float, float, float]] = {}
    for candidate in rows(combined.get("candidates")):
        text = record(candidate.get("cue")).get("text")
        if not isinstance(text, str) or not all(line in metrics for line in text.split("\n")):
            continue
        sizes = [metrics[line] for line in text.split("\n")]
        width = max(size[0] for size in sizes)
        height = max(size[1] for size in sizes) + (len(sizes) - 1) * advance
        result[text] = (-width / 2, top, width / 2, top + height)
    return result


def timestamp(frame: int, rate: int) -> str:
    # Floor BOTH boundaries: at every integer video frame, SRT and Motion agree.
    # Nearest-ms end rounding can leave the previous cue visible on a cut frame.
    ms = frame * 1000 // rate
    hours, tail = divmod(ms, 3600000)
    minutes, tail = divmod(tail, 60000)
    seconds, millis = divmod(tail, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def canonical_srt(parts: list[Part], rate: int) -> str:
    if not 1 <= rate <= 120:
        raise ReplayError("Unsupported SRT frame rate")
    cues = tuple(
        Cue(p.offset + c.start, p.offset + c.end, c.text, c.x, c.y) for p in parts for c in p.cues
    )
    ordered(cues)
    return "".join(
        f"{i}\n{timestamp(c.start, rate)} --> {timestamp(c.end, rate)}\n{c.text}\n\n"
        for i, c in enumerate(cues, 1)
    )


def save(path: Path, value: object) -> None:
    with local(path).open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def raster_bounds(alpha: NDArray[np.uint16]) -> list[int] | None:
    if alpha.ndim != 2 or alpha.dtype != np.uint16 or not 0 < alpha.size <= MAX_PIXELS:
        raise ReplayError("Expected bounded alpha16 raster")
    ys = np.flatnonzero(np.any(alpha != 0, axis=1))
    if not ys.size:
        return None
    xs = np.flatnonzero(np.any(alpha != 0, axis=0))
    return [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]


def append_run(runs: list[dict[str, object]], frame: int, bounds: list[int] | None) -> None:
    if runs and runs[-1]["bounds"] == bounds:
        runs[-1]["endFrame"] = frame + 1
    else:
        runs.append({"startFrame": frame, "endFrame": frame + 1, "bounds": bounds})


def decode_alpha(
    movie: Path, *, tool: Path, config: Config, part: Part, directory: Path, timeout: int
) -> list[dict[str, object]]:
    """One decoder per alpha, bounded pipe/frame memory, no raw intermediate file."""
    if not 0 < config.width * config.height <= MAX_PIXELS:
        raise ReplayError("Canvas exceeds raster budget")
    command = [
        str(tool),
        "-v",
        "error",
        "-nostdin",
        "-xerror",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(movie),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        "alphaextract,format=gray16le",
        "-fps_mode",
        "passthrough",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    env = dict(os.environ, TMPDIR=str(directory), TMP=str(directory), TEMP=str(directory))
    env.pop("FFREPORT", None)
    runs: list[dict[str, object]] = []
    pending = bytearray()
    size, count = config.width * config.height * 2, 0
    diagnostic: dict[str, object] = {"command": command, "returncode": None}
    errors = directory / "alpha.stderr"
    with (
        errors.open("xb") as stderr,
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
            env=env,
        ) as process,
    ):
        try:
            if process.stdout is None or process.stderr is None:
                raise ReplayError("Missing decoder pipes")
            deadline, error_bytes = time.monotonic() + timeout, 0
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stderr, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() > deadline:
                        raise ReplayError("Alpha decode deadline exceeded")
                    for key, _ in selector.select(0.1):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        if key.fd == process.stderr.fileno():
                            error_bytes += len(data)
                            if error_bytes > MAX_JSON:
                                raise ReplayError("Decoder stderr budget exceeded")
                            stderr.write(data)
                            continue
                        pending.extend(data)
                        while len(pending) >= size:
                            if count >= part.frames:
                                raise ReplayError("Extra decoded alpha frames")
                            image = np.frombuffer(bytes(pending[:size]), dtype="<u2").reshape(
                                config.height, config.width
                            )
                            append_run(runs, count, raster_bounds(image))
                            del pending[:size]
                            count += 1
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
            if process.returncode != 0 or error_bytes or pending or count != part.frames:
                raise ReplayError("Alpha decode failed/truncated; inspect retained stderr")
        finally:
            if process.poll() is None:
                process.kill()
            diagnostic["returncode"] = process.wait()
            diagnostic["decodedFrames"] = count
            save(directory / "alpha.process.json", diagnostic)
    return runs


def measure_alpha(
    movie: Path,
    *,
    config: Config,
    part: Part,
    directory: Path,
    ffmpeg: Path,
    ffprobe: Path,
    timeout: int,
) -> dict[str, object]:
    probe = verification._run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=width,height,pix_fmt,time_base:frame=pts,duration,pkt_duration",
            "-of",
            "json",
            str(movie),
        ],
        directory=directory,
        name="clock",
        timeout=timeout,
    )
    data = verification._json(probe)
    streams = rows(data.get("streams"))
    if len(streams) != 1 or (streams[0].get("width"), streams[0].get("height")) != (
        config.width,
        config.height,
    ):
        raise ReplayError("Alpha canvas mismatch")
    pixel_format = streams[0].get("pix_fmt")
    if not isinstance(pixel_format, str) or not pixel_format.startswith(
        ("yuva", "gbrap", "rgba", "bgra", "argb", "abgr")
    ):
        raise ReplayError("Movie has no actual alpha channel")
    clock = verification._video(
        rows(data.get("frames")),
        scale=verification._scale(streams[0].get("time_base")),
        rate=config.rate,
        frames=part.frames,
    )
    runs = decode_alpha(
        movie, tool=ffmpeg, config=config, part=part, directory=directory, timeout=timeout
    )
    return {
        "clock": clock,
        "rasterRuns": runs,
        "decodedFrames": part.frames,
        "alphaFormat": "gray16le/nonzero",
        "fullDecodeVerified": True,
    }


def box(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ReplayError("Expected four measured rectangle coordinates")
    a, b, c, d = (number(item) for item in value)
    if not a < c or not b < d:
        raise ReplayError("Empty/inverted constraint rectangle")
    return a, b, c, d


def constraints_for(path: Path | None, part: Part, inputs: Inputs) -> list[dict[str, object]]:
    """Optional independent all-state constraints; no geometry synthesized from pixels."""
    if path is None:
        return []
    data = document(path, inputs)
    if data.get("version") != 1 or data.get("kind") != "delivery-caption-constraints":
        raise ReplayError("Unsupported constraints contract")
    entries = record(data.get("parts"))
    if part.name not in entries:
        return []
    entry = record(entries[part.name])
    if inputs.reference(entry.get("plan")) != part.plan:
        raise ReplayError("Constraints bind another Motion plan")
    evidence = rows(entry.get("inputs"))
    if not evidence:
        raise ReplayError("Constraints require measured provenance")
    for reference in evidence:
        inputs.reference(reference)
    return rows(entry.get("states"))


def evaluate(
    part: Part, measurement: dict[str, object], constraints: list[dict[str, object]]
) -> dict[str, object]:
    """Check every frame, including internal gaps; partial states never pass."""
    runs = rows(measurement.get("rasterRuns"))
    cursor, gap_frames, visible, checked = 0, 0, 0, 0
    gaps_ok = visible_ok = geometry_ok = envelope_ok = True
    envelope_frames = 0
    bounds_by_frame: list[object] = []
    for run in runs:
        start, end = integer(run.get("startFrame")), integer(run.get("endFrame"))
        if start != cursor or not start < end <= part.frames:
            raise ReplayError("Raster cache does not cover every alpha frame")
        bounds_by_frame.extend([run.get("bounds")] * (end - start))
        cursor = end
    if cursor != part.frames:
        raise ReplayError("Incomplete raster coverage")
    state_index = 0
    for frame, bounds in enumerate(bounds_by_frame):
        while state_index < len(part.cues) and part.cues[state_index].end <= frame:
            state_index += 1
        active = state_index < len(part.cues) and part.cues[state_index].start <= frame
        if not active:
            gap_frames += 1
            gaps_ok &= bounds is None
            continue
        visible += 1
        visible_ok &= bounds is not None
        if bounds is not None and len(part.envelopes) == len(part.cues):
            actual_bounds = box(bounds)
            planned = part.envelopes[state_index]
            envelope_ok &= (
                actual_bounds[0] >= planned[0]
                and actual_bounds[1] >= planned[1]
                and actual_bounds[2] <= planned[2]
                and actual_bounds[3] <= planned[3]
            )
            envelope_frames += 1
        if bounds is not None and state_index < len(constraints):
            state = constraints[state_index]
            c = part.cues[state_index]
            if (state.get("startFrame"), state.get("endFrame"), state.get("text")) != (
                c.start,
                c.end,
                c.text,
            ):
                raise ReplayError("Constraint full-state identity mismatch")
            actual, envelope, region = (
                box(bounds),
                box(state.get("envelope")),
                box(state.get("region")),
            )
            center = number(state.get("centerX"))
            clearance = number(state.get("clearance"))
            # Bounds use top-left pixels; exact measured tolerance is an explicit input.
            tolerance = number(state.get("centerTolerancePixels"))
            if clearance < 0 or tolerance < 0:
                raise ReplayError("Negative geometric constraint")
            top = min(number(state.get(key)) for key in ("foregroundTop", "faceTop", "headTop"))
            contained = all(
                actual[0] >= b[0] and actual[1] >= b[1] and actual[2] <= b[2] and actual[3] <= b[3]
                for b in (envelope, region)
            )
            geometry_ok &= (
                contained
                and actual[3] + clearance <= top
                and abs((actual[0] + actual[2]) / 2 - center) <= tolerance
            )
            checked += 1
    complete = len(constraints) == len(part.cues) and checked == visible and visible > 0
    return {
        "fullCaptionStates": len(part.cues),
        "sampleCount": visible,
        "measuredFrames": cursor,
        "gapFrames": gap_frames,
        "gapsVerified": gaps_ok,
        "visibleStatesVerified": visible_ok,
        "constraintFrames": checked,
        "fullStateConstraintCoverage": complete,
        "measuredEnvelopeFrames": envelope_frames,
        "measuredEnvelopeVerified": envelope_ok and envelope_frames == visible and visible > 0,
        "boundsAndGapsVerified": complete
        and geometry_ok
        and visible_ok
        and gaps_ok
        and envelope_ok
        and envelope_frames == visible,
    }


def completed(job: Job, state_directory: Path, inputs: Inputs) -> tuple[Path, str]:
    state = document(state_directory / f"{job.name}.json", inputs)
    job_id = state.get("jobID")
    if (
        state.get("status") != "Successful"
        or not isinstance(job_id, str)
        or JOB_ID.fullmatch(job_id) is None
    ):
        raise ReplayError("No actual Successful Compressor job receipt")
    output = absolute(state.get("output"))
    requested = absolute(state.get("requested_output", str(job.output)))
    if requested.parent != job.output.parent or not requested.stem.startswith(job.output.stem):
        raise ReplayError("Unreconciled retry path")
    if output.parent != requested.parent or output.stem != requested.stem:
        raise ReplayError("Completed output differs from requested job")
    fingerprint = inputs.bind(output)
    if state.get("bytes") != fingerprint["bytes"] or integer(state.get("bytes"), minimum=1) <= 0:
        raise ReplayError("Completed alpha size changed")
    return output, job_id


def state_binding(path: Path, *, job: Job, movie: Path, job_id: str, inputs: Inputs) -> bool:
    """Read the producer's same-state pins without upgrading legacy receipts.

    These prove source/preset submission identity, not historical tool/code or
    Motion dependency provenance. The latter still needs independent evidence.
    """
    state = document(path, inputs)
    requested = absolute(state.get("requested_output", str(job.output)))
    if requested != job.output and requested != job.output.with_name(
        f"{job.output.stem}.retry-1{job.output.suffix}"
    ):
        raise ReplayError("Unreconciled submission retry output")
    submitted = Job(job.name, job.source, job.preset, requested, job.dependencies)
    inputs.bind(Path(compressor_batch.__file__))
    if not compressor_batch.verify_submission(submitted, state, rehash=True):
        return False
    if (
        state.get("status") != "Successful"
        or state.get("jobID") != job_id
        or state.get("terminalInputsVerified") is not True
        or state.get("output") != str(movie)
        or movie.parent != requested.parent
        or movie.stem != requested.stem
        or state.get("bytes") != inputs.bind(movie)["bytes"]
    ):
        raise ReplayError("Submission terminal/output identity mismatch")
    submit = state.get("submitResponse")
    monitor = inputs.reference(state.get("monitorResponse"))
    inputs.bind(monitor, record(state.get("monitorResponse")).get("sha256"))
    if monitor.stat().st_size > compressor_batch.MAX_SNAPSHOT_BYTES:
        raise ReplayError("Terminal snapshot byte budget exceeded")
    if (
        not isinstance(submit, str)
        or parse_submission(submit) != job_id
        or parse_snapshot(monitor.read_text(encoding="utf-8")).get(job_id) != "Successful"
    ):
        raise ReplayError("Raw direct CLI responses contradict submission state")
    for pin in rows(state.get("inputs")):
        inputs.reference(pin)
    return True


def tool_code_binding(path: Path, *, config: Config, part: Part, inputs: Inputs) -> bool:
    """Submission-time code/CLI plus exact preparation inputs, not app-bundle attestation."""
    state = document(path, inputs)
    if not compressor_batch.verify_runtime(state, rehash=True):
        return False
    if state.get("terminalRuntimeVerified") is not True:
        raise ReplayError("Missing terminal runtime consistency")
    for pin in rows(state.get("runtimeInputs")):
        inputs.reference(pin)
    dependencies = {inputs.reference(pin) for pin in rows(state.get("dependencies"))}
    required = motion_dependencies.preparation_inputs(
        config=config.path, template=config.template, plan=part.plan, calibration=config.calibration
    )
    if not set(required) <= dependencies:
        return False
    raw = document(config.path, inputs)
    motion = record(raw.get("motion", {}))
    assets = config.resolved_assets or ()
    external = {
        path
        for path in (*dependencies, *assets)
        if not path.is_relative_to(verification.REPOSITORY)
        and path != Path(sys.executable).resolve(strict=True)
    }
    if "resourceEvidence" in motion or external:
        resources = caption_resource_inputs.validate(
            config=config.path, plan=part.plan, source=part.source, state=state
        )
        if not external <= resources.pins.keys():
            raise ReplayError("Undeclared external caption dependency")
        for resource, digest in resources.pins.items():
            inputs.bind(resource, digest)
    return True


def historical_binding(
    path: Path | None, *, job: Job, part: Part, movie: Path, job_id: str, inputs: Inputs
) -> bool:
    """Consume retained direct-CLI evidence, never manufacture an Executor response."""
    if path is None:
        return False
    raw = document(path, inputs)
    if raw.get("version") != 1 or raw.get("kind") != "compressor-direct-cli-history":
        raise ReplayError("Unsupported historical receipt contract")
    entries = record(raw.get("jobs"))
    if job.name not in entries:
        return False
    entry = record(entries[job.name])
    if entry.get("fingerprintMoment") != "before-submit" or entry.get("jobID") != job_id:
        raise ReplayError("History lacks actual submission-time identity")
    expected = {"source": job.source, "preset": job.preset, "output": movie, "plan": part.plan}
    for key, value in expected.items():
        if inputs.reference(entry.get(key)) != value:
            raise ReplayError("Historical input/output binding mismatch")
    submit = inputs.reference(entry.get("submitResponse"))
    monitor = inputs.reference(entry.get("monitorResponse"))
    if (
        parse_submission(submit.read_text(encoding="utf-8")) != job_id
        or parse_snapshot(monitor.read_text(encoding="utf-8")).get(job_id) != "Successful"
    ):
        raise ReplayError("Actual direct-CLI responses contradict state")
    command = entry.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ReplayError("Missing actual submit command")
    for flag, value in (
        ("-jobpath", job.source),
        ("-settingpath", job.preset),
        ("-locationpath", movie),
    ):
        if (
            command.count(flag) != 1
            or command.index(flag) + 1 >= len(command)
            or command[command.index(flag) + 1] != str(value)
        ):
            raise ReplayError("Historical submit command mismatch")
    return True


def verify_motion_source(config: Config, part: Part, inputs: Inputs) -> None:
    """Rebuild the selected-style XML in memory and compare the actual job source.

    No renderer, helper process or installed template is modified. Unsupported
    alternate styles fail explicitly instead of inheriting style-20 approval.
    """
    counts = {
        "--top": 3,
        "--bottom": 3,
        "--font-name": 1,
        "--font-size": 1,
        "--stroke-scale": 1,
        "--shadow-scale": 1,
    }
    values: dict[str, tuple[str, ...]] = {}
    index = 0
    while index < len(config.finish):
        flag = config.finish[index]
        if flag not in counts or flag in values:
            raise ReplayError("Source audit requires explicit selected-style finish arguments")
        end = index + 1 + counts[flag]
        if end > len(config.finish):
            raise ReplayError("Incomplete selected-style finish arguments")
        values[flag] = config.finish[index + 1 : end]
        index = end
    if set(values) != set(counts):
        raise ReplayError("Missing selected-style finish arguments")
    top, bottom = values["--top"], values["--bottom"]
    style = motion_finish.selected_finish_style(
        top=(float(top[0]), float(top[1]), float(top[2])),
        bottom=(float(bottom[0]), float(bottom[1]), float(bottom[2])),
        font_name=values["--font-name"][0],
        font_size=float(values["--font-size"][0]),
        stroke_scale=float(values["--stroke-scale"][0]),
        shadow_scale=float(values["--shadow-scale"][0]),
    )
    for path in (
        config.template,
        part.source,
        Path(motion_stages.__file__),
        Path(motion_finish.__file__),
    ):
        inputs.bind(path)
    if max(config.template.stat().st_size, part.source.stat().st_size) > MAX_XML_CHARACTERS:
        raise ReplayError("Motion XML byte budget exceeded")
    expected = motion_finish.finish_sequence(
        motion_stages.prepare(
            config.template.read_text(encoding="utf-8"), document(part.plan, inputs)
        ),
        style=style,
    )
    if part.source.read_bytes() != expected.encode("utf-8"):
        raise ReplayError("Actual Motion source differs from plan/template/selected style")


def cached_measurement(
    movie: Path,
    *,
    config: Config,
    part: Part,
    cache: Path,
    inputs: Inputs,
    ffmpeg: Path,
    ffprobe: Path,
    timeout: int,
) -> dict[str, object]:
    bindings = [
        inputs.bind(path)
        for path in (movie, part.plan, ffmpeg, ffprobe, Path(__file__), Path(verification.__file__))
    ]
    identity = {
        "version": 1,
        "numpyVersion": np.__version__,
        "inputs": bindings,
        "width": config.width,
        "height": config.height,
        "rate": config.rate,
        "frames": part.frames,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = local(cache / key)
    receipt = directory / "measurement.json"
    if receipt.exists():
        data = document(receipt, inputs)
        if data.get("identity") != identity:
            raise ReplayError("Raster cache identity mismatch")
        return record(data.get("measurement"))
    directory.mkdir(mode=0o700, parents=True)
    measurement = measure_alpha(
        movie,
        config=config,
        part=part,
        directory=directory,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        timeout=timeout,
    )
    inputs.verify()
    save(receipt, {"identity": identity, "measurement": measurement})
    inputs.bind(receipt)
    return measurement


def transcript_check(
    path: Path | None, *, config: Config, parts: list[Part], inputs: Inputs
) -> dict[str, object]:
    """Compare exact ordered added characters/onset frames to retained source tokens.

    Source-token evidence must already be projected from the decoded ASR audio
    clock to source presentation seconds. Split-cut tokens require separate
    reconciled evidence; this reader does not invent their text allocation.
    """
    if path is None:
        return {"verified": False, "reason": "missing source-clock recognition evidence"}
    evidence = document(path, inputs)
    if evidence.get("version") != 1 or evidence.get("kind") not in {
        "caption-source-tokens",
        "caption-source-labels",
    }:
        raise ReplayError("Unsupported source-token evidence")
    if inputs.reference(evidence.get("cuts")) != config.cuts:
        raise ReplayError("Transcript evidence binds different cuts")
    labels_contract = evidence.get("kind") == "caption-source-labels"
    source_path = inputs.reference(evidence.get("labels" if labels_contract else "tokens"))
    provenance = rows(evidence.get("inputs"))
    if not provenance:
        raise ReplayError("Recognition clock provenance is missing")
    for ref in provenance:
        inputs.reference(ref)
    source = document(source_path, inputs)
    if labels_contract:
        labels, instant = delivery_captions.read_labels(source_path)
        if instant:
            raise ReplayError("Unresolved zero-length recognition labels")
        tokens: list[dict[str, object]] = [
            {"text": label.text, "start": label.start, "end": label.end} for label in labels
        ]
    else:
        if source.get("clock") != "source-presentation":
            raise ReplayError("Recognition tokens must use source presentation clock")
        tokens = rows(source.get("tokens"))
    inputs.bind(Path(delivery_captions.__file__))
    document(config.cuts, inputs)
    spans = [
        (
            Fraction(cut.source_start, config.rate),
            Fraction(cut.source_end, config.rate),
            cut.output_start,
        )
        for cut in delivery_captions.read_cuts(config.cuts)
    ]
    expected: list[tuple[str, int]] = []
    previous = Fraction(0)
    cut_index = 0
    for token in tokens:
        start, end = (
            display_time(number(token.get("start"))),
            display_time(number(token.get("end"))),
        )
        text = token.get("text")
        if start < previous or end <= start or not isinstance(text, str) or not text:
            raise ReplayError("Invalid ordered source token")
        previous = start
        while cut_index < len(spans) and spans[cut_index][1] <= start:
            cut_index += 1
        if cut_index == len(spans):
            continue
        first, last, offset = spans[cut_index]
        if end <= first:
            continue
        if start < first or end > last:
            return {
                "verified": False,
                "reason": "source token crosses cut; allocation is unverified",
            }
        frame = offset + math.ceil((start - first) * config.rate)
        expected.extend((character, frame) for character in text)
    actual: list[tuple[str, int]] = []
    prior_text, prior_end = "", -1
    for part in parts:
        for c in part.cues:
            text = c.text.replace("\n", "")
            display_start = part.offset + c.start
            added = text
            if (
                display_start == prior_end
                and len(text) > len(prior_text)
                and text.startswith(prior_text)
            ):
                added = text[len(prior_text) :]
            actual.extend((character, display_start) for character in added)
            prior_text, prior_end = text, part.offset + c.end
    return {
        "verified": bool(expected) and expected == actual,
        "textConserved": [c for c, _ in expected] == [c for c, _ in actual],
        "orderedOnsetFramesEqual": expected == actual,
        "sourceCharactersRetained": len(expected),
        "displayCharactersAdded": len(actual),
        "policy": "exact text and integer onset; no timing tolerance or inferred ASR times",
    }


def composition_check(
    path: Path | None,
    movie: Path | None,
    *,
    config: Config,
    parts: list[Part],
    results: list[dict[str, object]],
    inputs: Inputs,
) -> dict[str, object]:
    """Bind actual alpha selections to retained cut/native execution and final movie."""
    if path is None or movie is None:
        return {"verified": False, "reason": "missing final movie/cut execution receipt"}
    evidence = document(path, inputs)
    checks = record(evidence.get("checks"))
    if (
        evidence.get("kind") != "delivery-cut-execution"
        or evidence.get("version") != 1
        or evidence.get("accepted") is not True
        or evidence.get("selectionPlanMatchesProtectedCuts") is not True
        or evidence.get("recordedExecutionMatchesPlan") is not True
        or not checks
        or not all(value is True for value in checks.values())
        or evidence.get("inputsAfter") != evidence.get("inputs")
    ):
        raise ReplayError("Unverified cut execution cannot prove final composition")
    pinned = {inputs.reference(ref) for ref in rows(evidence.get("inputs"))}
    required = {config.path, config.cuts, config.source, movie}
    if not required <= pinned or len(results) != len(parts) or any("error" in r for r in results):
        raise ReplayError("Final composition omits required input/alpha bindings")
    alphas = [inputs.reference(result.get("alpha")) for result in results]
    if not set(alphas) <= pinned:
        raise ReplayError("Cut execution omits actual alpha hashes")
    native = rows(evidence.get("actualNativeReceipts"))
    requests = rendered_parts(config)
    if not requests or len(native) != len(requests):
        raise ReplayError("Final composition lacks actual native receipt coverage")
    cursor = 0
    for (request, frames), receipt in zip(requests, native, strict=True):
        response = request.with_name("render.response.json")
        actual_movie = absolute(receipt.get("movie"))
        if (
            not {request, response, actual_movie} <= pinned
            or receipt.get("actualNativeReceipt") != str(response)
            or receipt.get("receiptSHA256") != inputs.bind(response)["sha256"]
            or receipt.get("movieSHA256Recorded") != inputs.bind(actual_movie)["sha256"]
            or receipt.get("movieSHA256AtAudit") != inputs.bind(actual_movie)["sha256"]
        ):
            raise ReplayError("Actual native render receipt binding mismatch")
        expected = [
            {
                "sourcePath": str(alpha),
                "selection": {
                    "startSeconds": float(Fraction(max(cursor - part.offset, 0), config.rate)),
                    "durationSeconds": float(
                        Fraction(
                            min(cursor + frames, part.offset + part.frames)
                            - max(cursor, part.offset),
                            config.rate,
                        )
                    ),
                    "rate": 1,
                },
                "offsetSeconds": float(Fraction(max(part.offset - cursor, 0), config.rate)),
                "opacity": 1,
            }
            for part, alpha in zip(parts, alphas, strict=True)
            if part.offset < cursor + frames and cursor < part.offset + part.frames
        ]
        recipe = record(document(request, inputs).get("recipe"))
        if recipe.get("additionalVideo", []) != expected:
            raise ReplayError("Actual native alpha selections/placement differ from audited plans")
        cursor += frames
    if cursor != verification._expected(config) or evidence.get("selectedFrames") != cursor:
        raise ReplayError("Final composition clock differs from protected cuts")
    inputs.bind(movie)
    return {"verified": True, "cutExecution": inputs.bind(path), "finalMovie": inputs.bind(movie)}


def probe_alpha(
    config_path: Path,
    plan_path: Path,
    movie: Path,
    output: Path,
    *,
    name: str,
    constraints: Path,
    cache: Path,
    ffmpeg: Path,
    ffprobe: Path,
    timeout: int = 600,
) -> dict[str, object]:
    """Audit one calibration alpha, without a delivery inventory or approval.

    Constraint envelopes must be independently measured and pinned, not derived
    from this raster. No submission/style, ASR or final-composition claim is made.
    """
    if not 1 <= timeout <= 7200 or not name:
        raise ReplayError("Invalid probe deadline/name")
    config = load_config(local(config_path))
    inputs = Inputs()
    inputs.bind(config.path)
    plan = document(absolute(str(plan_path)), inputs)
    frames = integer(plan.get("frames"), minimum=1)
    if frames > MAX_FRAMES or (plan.get("width"), plan.get("height"), plan.get("rate")) != (
        config.width,
        config.height,
        config.rate,
    ):
        raise ReplayError("Probe plan canvas/clock mismatch")
    cues = tuple(cue(row, frames) for row in rows(plan.get("placements")))
    ordered(cues)
    if not cues:
        raise ReplayError("Probe requires full-caption states")
    part = Part(name, plan_path, plan_path, 0, frames, cues)
    states = constraints_for(constraints, part, inputs)
    part = Part(
        name,
        plan_path,
        plan_path,
        0,
        frames,
        cues,
        tuple(box(state.get("envelope")) for state in states),
    )
    measurement = cached_measurement(
        absolute(str(movie)),
        config=config,
        part=part,
        cache=local(cache),
        inputs=inputs,
        ffmpeg=ffmpeg.resolve(strict=True),
        ffprobe=ffprobe.resolve(strict=True),
        timeout=timeout,
    )
    checks = evaluate(part, measurement, states)
    result: dict[str, object] = {
        "version": 1,
        "kind": "caption-alpha-probe",
        "inputs": inputs.rows(),
        "checks": checks,
        "measurement": measurement,
        "alphaOnlyAccepted": checks["boundsAndGapsVerified"] is True,
        "wholeDeliveryAccepted": False,
        "accepted": False,
        "limitations": [
            "Single alpha raster/clock/bounds only; no submission, style, ASR or final movie proof."
        ],
    }
    inputs.verify()
    save(local(output), result)
    return result


def audit(
    config_path: Path,
    candidate_srt: Path,
    output: Path,
    *,
    phase: str,
    cache: Path,
    ffmpeg: Path,
    ffprobe: Path,
    start: int = 0,
    limit: int = MAX_PARTS,
    constraints: Path | None = None,
    history: Path | None = None,
    transcript: Path | None = None,
    final_movie: Path | None = None,
    cut_execution: Path | None = None,
    timeout: int = 600,
) -> dict[str, object]:
    if (
        phase not in {"canonical-srt", "audit"}
        or start < 0
        or not 1 <= limit <= MAX_PARTS
        or not 1 <= timeout <= 7200
    ):
        raise ReplayError("Invalid phase/batch/deadline")
    config = load_config(local(config_path))
    if phase == "audit":
        ffmpeg, ffprobe = ffmpeg.resolve(strict=True), ffprobe.resolve(strict=True)
    inputs = Inputs()
    parts, total = load_parts(config, inputs)
    inputs.bind(candidate_srt)
    inputs.bind(Path(__file__))
    inputs.bind(Path(caption_style_defaults.__file__))
    text = canonical_srt(parts, config.rate)
    projection = transcript_check(transcript, config=config, parts=parts, inputs=inputs)
    output = local(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    canonical = output / "canonical.srt"
    with canonical.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)
    canonical_binding = inputs.bind(canonical)
    current_matches = candidate_srt.read_bytes() == text.encode("utf-8")
    results: list[dict[str, object]] = []
    missing: list[str] = list(inputs.missing)
    if phase == "audit":
        jobs_path = config.work / "alpha-jobs.json"
        inputs.bind(jobs_path)
        jobs = {job.name: job for job in read_jobs(jobs_path)}
        if set(jobs) != {part.name for part in parts}:
            raise ReplayError("Alpha job list does not cover exactly the Motion plans")
        for part in parts[start : start + limit]:
            try:
                job = jobs[part.name]
                if (
                    job.source != part.source
                    or job.preset != config.alpha_preset
                    or job.output.parent != config.storage / "alpha"
                ):
                    raise ReplayError("Alpha source/preset/output scope differs from renderConfig")
                verify_motion_source(config, part, inputs)
                inputs.bind(job.source)
                inputs.bind(job.preset)
                movie, job_id = completed(job, config.work / "alpha-jobs-state", inputs)
                same_state = state_binding(
                    config.work / "alpha-jobs-state" / f"{job.name}.json",
                    job=job,
                    movie=movie,
                    job_id=job_id,
                    inputs=inputs,
                )
                historical = same_state or historical_binding(
                    history, job=job, part=part, movie=movie, job_id=job_id, inputs=inputs
                )
                tool_code = same_state and tool_code_binding(
                    config.work / "alpha-jobs-state" / f"{job.name}.json",
                    config=config,
                    part=part,
                    inputs=inputs,
                )
                measurement = cached_measurement(
                    movie,
                    config=config,
                    part=part,
                    cache=cache,
                    inputs=inputs,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    timeout=timeout,
                )
                checks = evaluate(part, measurement, constraints_for(constraints, part, inputs))
                results.append(
                    {
                        "name": part.name,
                        "offsetFrame": part.offset,
                        "frames": part.frames,
                        "plan": inputs.bind(part.plan),
                        "source": inputs.bind(job.source),
                        "preset": inputs.bind(job.preset),
                        "alpha": inputs.bind(movie),
                        "jobID": job_id,
                        "producerState": inputs.bind(
                            config.work / "alpha-jobs-state" / f"{job.name}.json"
                        ),
                        "backend": "direct-compressor-cli",
                        "historicalSubmissionBound": historical,
                        "sameStateSubmissionBound": same_state,
                        "historicalToolCodeBound": tool_code,
                        "motionSourcePlanStyleVerified": True,
                        "checks": checks,
                        "measurement": measurement,
                    }
                )
                if not historical:
                    missing.append(
                        f"{part.name}: submission-time source/preset/plan/output binding "
                        "and raw CLI receipts"
                    )
                if not checks["fullStateConstraintCoverage"]:
                    missing.append(
                        f"{part.name}: all-state measured envelope/foreground/face/head/center "
                        "constraints"
                    )
            except ERRORS as error:
                results.append({"name": part.name, "error": str(error)})
                missing.append(f"{part.name}: {error}")
    complete = len(results) == len(parts) and all("error" not in item for item in results)
    verified = complete and all(
        record(item["checks"]).get("boundsAndGapsVerified") is True for item in results
    )
    history_ok = complete and all(item.get("historicalSubmissionBound") is True for item in results)
    composition = composition_check(
        cut_execution, final_movie, config=config, parts=parts, results=results, inputs=inputs
    )
    limitations = [
        "Timed full-string pop-ins; no fades, not verified smooth animation.",
        "SRT floors boundaries to milliseconds; exact display contract is integer video frames, "
        "not subframe equality.",
        "Current candidate SRT mismatch is reported exactly, never accepted via timing tolerance.",
        "Measured candidate correspondence is not independent recognition/cut-projection "
        "or rendered-glyph recognition approval.",
        "Final composition proof requires bound cut execution; no FCP, colour or global approval.",
        "Tool/code binding covers exact CLI, Python and preparation inputs at submission; "
        "not historical prepare execution, font resolution, plugins or a hermetic app bundle.",
    ]
    if not complete:
        missing.append("complete actual alpha coverage")
    if projection["verified"] is not True:
        missing.append(
            "independent recognition text conservation and source-to-cut display "
            "projection evidence"
        )
    if composition["verified"] is not True:
        missing.append("actual final movie and cutExecution/native render composition bindings")
    tool_code_ok = complete and all(item.get("historicalToolCodeBound") is True for item in results)
    if not tool_code_ok:
        missing.append("producer submission-time executable/code and Motion dependency bindings")
    checks_out = {
        "measuredPlanCorrespondence": True,
        "orderedFrameTiming": True,
        "canonicalSRTFrameEquivalent": True,
        "canonicalSRTActualDisplayVerified": verified and history_ok,
        "candidateSRTExactMatch": current_matches,
        "fullAlphaCoverage": complete,
        "boundsAndGapsVerified": verified,
        "historicalSubmissionBound": history_ok,
        "recognitionCutProjectionVerified": projection["verified"],
        "historicalToolCodeBound": tool_code_ok,
        "finalCompositionVerified": composition["verified"],
    }
    result: dict[str, object] = {
        "version": 1,
        "kind": "delivery-caption-audit",
        "measurementTools": (
            {"ffmpeg": inputs.bind(ffmpeg), "ffprobe": inputs.bind(ffprobe)}
            if phase == "audit"
            else {}
        ),
        "inputs": inputs.rows(),
        "checks": checks_out,
        "measurements": {
            "phase": phase,
            "expectedAlphaParts": len(parts),
            "auditedAlphaParts": len(results),
            "wholeVideoFrames": total,
            "rate": config.rate,
            "canonicalSRT": canonical_binding,
            "parts": results,
            "transcriptProjection": projection,
            "finalComposition": composition,
            "displayCues": [
                {"alpha": p.name, "offsetFrame": p.offset, "states": [asdict(c) for c in p.cues]}
                for p in parts
            ],
            "missingBindings": missing,
        },
        "alphaOnlyAccepted": verified and history_ok and not inputs.missing,
        "wholeDeliveryAccepted": False,
        "accepted": not missing and all(checks_out.values()),
        "limitations": limitations,
    }
    inputs.verify()
    save(output / "audit.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("canonical-srt", "audit", "probe-alpha"))
    parser.add_argument("render_config", type=Path)
    parser.add_argument("candidate_srt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--plan", type=Path, help="Required for probe-alpha; candidate_srt is the alpha movie"
    )
    parser.add_argument("--name", default="calibration", help="Part key in probe constraints")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/homebrew/bin/ffprobe"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=MAX_PARTS)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--final-movie", type=Path)
    parser.add_argument("--cut-execution", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.phase == "probe-alpha":
            if args.plan is None or args.constraints is None:
                raise ReplayError("probe-alpha requires --plan and --constraints")
            result = probe_alpha(
                args.render_config,
                args.plan.absolute(),
                args.candidate_srt.absolute(),
                args.output,
                name=args.name,
                constraints=args.constraints,
                cache=local(args.cache),
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                timeout=args.timeout,
            )
            print(
                json.dumps(
                    {"receipt": str(args.output), "alphaOnlyAccepted": result["alphaOnlyAccepted"]}
                )
            )
            return 0 if result["alphaOnlyAccepted"] is True else 1
        result = audit(
            args.render_config,
            args.candidate_srt,
            args.output,
            phase=args.phase,
            cache=local(args.cache),
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            start=args.start,
            limit=args.limit,
            constraints=args.constraints,
            history=args.history,
            transcript=args.transcript,
            final_movie=args.final_movie,
            cut_execution=args.cut_execution,
            timeout=args.timeout,
        )
    except ERRORS as error:
        print(f"CAPTION_AUDIT_ERROR {error}", file=sys.stderr)
        return 2
    print(json.dumps({"receipt": str(args.output / "audit.json"), "accepted": result["accepted"]}))
    return 0 if args.phase == "canonical-srt" or result["accepted"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
