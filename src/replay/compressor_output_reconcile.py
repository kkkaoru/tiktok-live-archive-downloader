"""Independently verify existing legacy Compressor output without resubmission.

This readonly disposition cannot reconstruct official terminal observations or
historical pre-submit inputs. Neither success nor reuse changes those facts.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.compressor_batch import JOB_ID, Job, actual_output, read_jobs, verify_submission
from replay.core import ReplayError
from replay.fcp_part_assembly import FrameProbeTool
from replay.input_hash_cache import InputHashCache
from replay.motion_caption import _one, _parse, _project_duration
from replay.native_export_receipt import save_new
from replay.recordings import record
from replay.stages import file_digest
from replay.whole_technical_batch import reader_code_paths

ROOT = Path(__file__).resolve().parents[2]
MEDIA_ROOT = Path("/Volumes/m.2-ssd/movie")
PRESET_ROOT = Path("/Applications/Compressor Creator Studio.app")
TOOL_ROOT = Path("/opt/homebrew/Cellar/ffmpeg")
MAX_BYTES = 8 * 1024**2
MAX_FRAMES = 100_000
OWNER_MAX_AGE = 120
ALPHA_FORMATS = frozenset({"yuva444p12le", "yuva444p10le", "gbrap12le", "argb", "rgba"})
CODE_SEEDS = (
    "compressor_batch.py",
    "compressor_helper_guard.py",
    "core.py",
    "fcp_part_assembly.py",
    "input_hash_cache.py",
    "motion_caption.py",
    "native_export_receipt.py",
    "recordings.py",
    "stages.py",
    "whole_technical_batch.py",
)


def _reader_paths() -> tuple[Path, ...]:
    """Reuse the shared closed reader superset, avoiding a whole-project freeze."""
    paths = {_path(str(path)) for path in reader_code_paths()}
    seeds = {ROOT / "src/replay" / name for name in CODE_SEEDS}
    if not seeds.issubset(paths):
        raise ReplayError("Shared reader closure omits a required reconciliation code seed")
    paths.add(_path(str(ROOT / "src/replay/compressor_output_reconcile.py")))
    if not 1 <= len(paths) <= 512:
        raise ReplayError("Current reader code inventory exceeds its budget")
    return tuple(sorted(paths))


def _path(value: object, *, work: bool = False) -> Path:
    if not isinstance(value, str) or not value or any(c in value for c in "\0\r\n"):
        raise ReplayError("Reconciliation requires canonical absolute paths")
    path = Path(value)
    roots = (
        (ROOT / "edits/work",)
        if work
        else (ROOT / "edits", ROOT / "src/replay", MEDIA_ROOT, PRESET_ROOT, TOOL_ROOT)
    )
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or path.resolve() != path
        or not any(path.is_relative_to(root) for root in roots)
    ):
        raise ReplayError("Reconciliation path escapes authorized roots or uses a link")
    return path


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


def _ref(value: object) -> Reference:
    row = record(value)
    digest = row.get("sha256")
    if (
        set(row) != {"path", "sha256"}
        or not isinstance(digest, str)
        or (re.fullmatch(r"[a-f0-9]{64}", digest) is None)
    ):
        raise ReplayError("Reconciliation references require exact SHA256 pins")
    return Reference(_path(row["path"]), digest)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ReplayError("Reconciliation JSON keys must be unique")
    return dict(pairs)


def _bytes(ref: Reference) -> bytes:
    descriptor = os.open(ref.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
            raise ReplayError("Reconciliation metadata must be bounded regular bytes")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or _identity(before) != _identity(ref.path.lstat()):
        raise ReplayError("Reconciliation metadata changed while reading")
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ReplayError("Reconciliation metadata SHA256 changed")
    return data


def _json(ref: Reference) -> dict[str, object]:
    return record(json.loads(_bytes(ref), object_pairs_hook=_unique))


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _positive(value: object, limit: int) -> int:
    if type(value) is not int or not 1 <= value <= limit:
        raise ReplayError("Reconciliation needs bounded positive integer geometry/clock")
    return value


def _fraction(value: object) -> Fraction:
    if not isinstance(value, str) or len(value) > 64:
        raise ReplayError("Reconciliation needs explicit bounded rational clock values")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid reconciliation clock") from error
    if result <= 0:
        raise ReplayError("Reconciliation clock must be positive")
    return result


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if len(result.stdout) + len(result.stderr) > MAX_BYTES:
        raise ReplayError("Reconciliation tool response exceeds its byte budget")
    return result


def _writers(path: Path) -> dict[str, object]:
    result = _run(["/usr/sbin/lsof", "-F", "pfa", "--", str(path)], timeout=30)
    if result.returncode == 1 and not result.stdout and not result.stderr:
        return {"writerObserved": False, "stdout": "", "returncode": 1}
    accesses = [line[1:] for line in result.stdout.splitlines() if line.startswith("a")]
    if result.returncode != 0 or result.stderr or not accesses or any(a != "r" for a in accesses):
        raise ReplayError("Existing output has a writer or uncertain open-file ownership")
    return {"writerObserved": False, "stdout": result.stdout, "returncode": 0}


def _owner(ref: Reference) -> tuple[Reference, ...]:
    row = _json(ref)
    observed = row.get("observedAtEpochSeconds")
    records = row.get("terminalRecords")
    if (
        set(row)
        != {
            "version",
            "knownOwnersTerminal",
            "activeOwners",
            "observedAtEpochSeconds",
            "terminalRecords",
        }
        or type(row.get("version")) is not int
        or row["version"] != 1
        or row.get("knownOwnersTerminal") is not True
        or row.get("activeOwners") != []
        or not isinstance(observed, (int, float))
        or isinstance(observed, bool)
        or not 0 <= time.time() - observed <= OWNER_MAX_AGE
        or not isinstance(records, list)
        or not 1 <= len(records) <= 16
    ):
        raise ReplayError("Reconciliation requires fresh explicit terminal owner evidence")
    refs = tuple(_ref(item) for item in records)
    if len({item.path for item in refs}) != len(refs):
        raise ReplayError("Owner terminal records must be unique")
    for item in refs:
        text = _bytes(item).decode("utf-8").strip()
        if not re.fullmatch(r"[0-9]{1,3}", text) or int(text) > 255:
            raise ReplayError("Owner record is not a retained terminal exit status")
    return refs


@dataclass(frozen=True)
class Expectation:
    frames: int
    width: int
    height: int
    rate: Fraction
    ranges: tuple[dict[str, str], ...]

    def json(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "rate": str(self.rate),
            "duration": str(self.frames / self.rate),
        }


def _expected(source: Reference, manifest: Reference, plan: Reference) -> Expectation:
    catalog, part = _json(manifest), _json(plan)
    rows = catalog.get("parts")
    if catalog.get("version") != 1 or not isinstance(rows, list) or not 1 <= len(rows) <= 1024:
        raise ReplayError("Expected a bounded validated Motion part manifest")
    matches = [record(row) for row in rows if record(row).get("path") == str(plan.path)]
    frames = _positive(part.get("frames"), MAX_FRAMES)
    width, height = _positive(part.get("width"), 16384), _positive(part.get("height"), 16384)
    rate = Fraction(_positive(part.get("rate"), 240))
    if (
        len(matches) != 1
        or matches[0].get("sha256") != plan.sha256
        or (matches[0].get("frames") != frames or catalog.get("rate") != rate)
    ):
        raise ReplayError("Part plan differs from its pinned Motion manifest")
    root = _parse(_bytes(source).decode("utf-8"))
    settings = _one(root, "scene/sceneSettings")
    if (
        _project_duration(root) != frames / rate
        or _one(settings, "duration").text != str(frames)
        or _one(settings, "frameRate").text != str(rate)
        or _one(settings, "NTSC").text != "0"
        or _one(settings, "width").text != str(width)
        or _one(settings, "height").text != str(height)
    ):
        raise ReplayError("Motion duration/geometry does not agree with the validated part plan")
    ranges: list[dict[str, str]] = []
    for name in ("timeRange", "playRange"):
        # Both observed project serializations; object clipping ranges are not
        # project playback ranges and must not be compared to the project clock.
        for location in (name, f"scene/{name}"):
            for node in root.findall(location):
                diagnostic = {
                    **node.attrib,
                    "element": str(node.tag),
                    "location": location,
                    "clockComparison": "uninterpreted",
                }
                fields = (node.get("duration") or "").split()
                if len(fields) == 4 and all(re.fullmatch(r"[0-9]{1,18}", v) for v in fields):
                    ticks, scale, flags, epoch = (int(value) for value in fields)
                    if scale > 0 and flags == 1 and epoch == 0:
                        duration = Fraction(ticks, scale)
                        diagnostic["observedDuration"] = str(duration)
                        diagnostic["clockComparison"] = (
                            "matches-project-duration"
                            if duration == frames / rate
                            else "conflicts-with-project-duration"
                        )
                ranges.append(diagnostic)
    return Expectation(frames, width, height, rate, tuple(ranges))


def _probe(
    path: Path, tool: FrameProbeTool, expected: Expectation, encoding: dict[str, object]
) -> dict[str, object]:
    command = [
        str(tool.path),
        "-v",
        "error",
        "-threads",
        "0",
        "-protocol_whitelist",
        "file",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
        "-use_absolute_path",
        "0",
        "-count_frames",
        "-show_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,profile,pix_fmt,width,height,start_pts,avg_frame_rate,duration_ts,"
        "time_base,nb_read_frames:frame=best_effort_timestamp",
        "-of",
        "json",
        str(path),
    ]
    result = _run(command, timeout=1800)
    if result.returncode != 0 or result.stderr:
        raise ReplayError("Existing Compressor output frame probe failed")
    raw = record(json.loads(result.stdout, object_pairs_hook=_unique))
    streams, frames = raw.get("streams"), raw.get("frames")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(frames, list):
        raise ReplayError("Output probe lacks a complete single-video frame domain")
    stream = record(streams[0])
    formats = encoding.get("pixelFormats")
    if not isinstance(formats, list):
        raise ReplayError("Output encoding lacks explicit alpha formats")
    if (
        stream.get("codec_name") != encoding["codec"]
        or stream.get("profile") != encoding["profile"]
        or stream.get("pix_fmt") not in formats
        or stream.get("width") != expected.width
        or stream.get("height") != expected.height
        or stream.get("start_pts") != 0
        or stream.get("nb_read_frames") != str(expected.frames)
        or len(frames) != expected.frames
        or _fraction(stream.get("avg_frame_rate")) != expected.rate
    ):
        raise ReplayError("Output encoding/alpha/geometry/frame count differs from its plan")
    base = _fraction(stream.get("time_base"))
    duration = stream.get("duration_ts")
    if type(duration) is not int or duration * base != expected.frames / expected.rate:
        raise ReplayError("Output physical duration differs from the expected frame clock")
    for index, frame in enumerate(frames):
        pts = record(frame).get("best_effort_timestamp")
        if type(pts) is not int or pts * base != index / expected.rate:
            raise ReplayError("Output full PTS clock is incomplete or changed")
    return {
        "command": command,
        "stream": stream,
        "allPTSVerified": True,
        "framesVerified": len(frames),
        "returncode": result.returncode,
    }


def reconcile(request: Path, receipt: Path, *, resume: bool = False) -> Path:
    """Verify current bytes independently; reverify matching completed receipts readonly."""
    request, receipt = _path(str(request), work=True), _path(str(receipt), work=True)
    if request == receipt or not receipt.parent.is_dir():
        raise ReplayError("Independent disposition needs a distinct existing work directory")
    request_ref = Reference(request, file_digest(request))
    row = _json(request_ref)
    keys = {
        "version",
        "jobName",
        "jobID",
        "manifest",
        "legacyReceipt",
        "planManifest",
        "plan",
        "inputs",
        "output",
        "ffprobe",
        "ffmpeg",
        "ownerEvidence",
        "expectedEncoding",
    }
    if set(row) != keys or type(row.get("version")) is not int or row["version"] != 1:
        raise ReplayError("Unsupported independent Compressor output request")
    job_id = row.get("jobID")
    if not isinstance(job_id, str) or JOB_ID.fullmatch(job_id) is None:
        raise ReplayError("Independent disposition requires the original exact job ID")
    refs = {
        key: _ref(row[key])
        for key in (
            "manifest",
            "legacyReceipt",
            "planManifest",
            "plan",
            "output",
            "ffprobe",
            "ffmpeg",
            "ownerEvidence",
        )
    }
    _json(refs["manifest"])
    jobs = [job for job in read_jobs(refs["manifest"].path) if job.name == row.get("jobName")]
    if len(jobs) != 1:
        raise ReplayError("Requested legacy job is not an exact current manifest member")
    job: Job = jobs[0]
    old = _json(refs["legacyReceipt"])
    if (
        old.get("jobID") != job_id
        or old.get("status") != "submitted"
        or (verify_submission(job, old, rehash=True))
    ):
        raise ReplayError("This independent path requires an unchanged unproven legacy submission")
    rows = row.get("inputs")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 130:
        raise ReplayError("Current declared inputs must be explicitly pinned")
    inputs = tuple(_ref(item) for item in rows)
    declared = tuple(_path(str(path)) for path in (job.source, job.preset, *job.dependencies))
    if len({item.path for item in inputs}) != len(inputs) or {item.path for item in inputs} != set(
        declared
    ):
        raise ReplayError("Current input inventory differs from the original job's declared roles")
    source = next(item for item in inputs if item.path == job.source)
    expected = _expected(source, refs["planManifest"], refs["plan"])
    encoding = record(row["expectedEncoding"])
    formats = encoding.get("pixelFormats")
    if (
        set(encoding) != {"codec", "profile", "pixelFormats"}
        or encoding.get("codec") != "prores"
        or encoding.get("profile") not in {"4444", "4444 XQ"}
        or not isinstance(formats, list)
        or not formats
        or any(not isinstance(v, str) or v not in ALPHA_FORMATS for v in formats)
    ):
        raise ReplayError("Existing output reconciliation requires explicit ProRes alpha encoding")
    output = refs["output"].path
    if (
        output in declared
        or _path(str(actual_output(job.output))) != output
        or output.lstat().st_nlink != 1
    ):
        raise ReplayError("Existing output mapping must be exact and unlinked")
    terminal_refs = _owner(refs["ownerEvidence"])
    code_paths = _reader_paths()
    code = tuple(Reference(_path(str(path)), file_digest(path)) for path in code_paths)
    all_refs = (request_ref, *refs.values(), *inputs, *terminal_refs, *code)
    pins: dict[Path, str] = {}
    identities: dict[Path, tuple[int, ...]] = {}
    for item in all_refs:
        if item.path in pins and pins[item.path] != item.sha256:
            raise ReplayError("Conflicting independent verification input pins")
        if file_digest(item.path) != item.sha256:
            raise ReplayError("Independent verification input SHA256 changed")
        pins[item.path] = item.sha256
        identities[item.path] = _identity(item.path.lstat())
    prior: dict[str, object] | None = None
    if receipt.exists() or receipt.is_symlink():
        if not resume:
            raise ReplayError("Occupied disposition must not be overwritten or blindly retried")
        prior = _json(Reference(receipt, file_digest(receipt)))
        if (
            prior.get("request") != request_ref.json()
            or type(prior.get("version")) is not int
            or prior.get("version") != 1
            or prior.get("kind") != "independent-compressor-output-disposition"
            or prior.get("originalJobID") != job_id
            or prior.get("output") != refs["output"].json()
            or prior.get("expected") != expected.json()
            or prior.get("inputs") != {str(path): digest for path, digest in pins.items()}
            or prior.get("currentOutputVerified") is not True
            or prior.get("officialTerminalVerified") is not False
            or prior.get("historicalInputsVerified") is not False
            or prior.get("terminalVerified") is not False
            or prior.get("candidateOnly") is not True
            or prior.get("qualityApproved") is not False
            or prior.get("deliveryApproved") is not False
            or record(prior.get("probe")).get("allPTSVerified") is not True
            or record(prior.get("fullAV")).get("returncode") != 0
        ):
            raise ReplayError(
                "Existing disposition is uncertain or does not match current evidence"
            )
    cache = InputHashCache(tuple(pins), allow_uncached_overflow=True)
    cache.prime()
    if any(cache.digest(path, file_digest) != digest for path, digest in pins.items()):
        raise ReplayError("Fresh initial verification bytes differ from declared pins")
    before = _writers(output)
    probe_tool = FrameProbeTool(refs["ffprobe"].path, refs["ffprobe"].sha256)
    decode_tool = FrameProbeTool(refs["ffmpeg"].path, refs["ffmpeg"].sha256)
    probe_tool.verify()
    decode_tool.verify()
    probe = _probe(output, probe_tool, expected, encoding)
    command = [
        str(decode_tool.path),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "0",
        "-protocol_whitelist",
        "file",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
        "-use_absolute_path",
        "0",
        "-i",
        str(output),
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-f",
        "null",
        "-",
    ]
    decoded = _run(command, timeout=1800)
    if decoded.returncode != 0 or decoded.stderr:
        raise ReplayError("Existing Compressor output full AV decode failed")
    after = _writers(output)
    for path, digest in pins.items():
        if _identity(path.lstat()) != identities[path] or file_digest(path) != digest:
            raise ReplayError("Input/output changed during independent verification")
    if output.lstat().st_nlink != 1 or _reader_paths() != code_paths:
        raise ReplayError("Reader inventory or output links changed during verification")
    if tuple(_path(str(path)) for path in (job.source, job.preset, *job.dependencies)) != declared:
        raise ReplayError("Declared input target mapping changed during verification")
    if cache.verify() != {str(path): digest for path, digest in pins.items()}:
        raise ReplayError("Fresh terminal bytes differ from independent verification pins")
    if prior is None:
        save_new(
            receipt,
            {
                "version": 1,
                "kind": "independent-compressor-output-disposition",
                "request": request_ref.json(),
                "originalJobID": job_id,
                "jobName": job.name,
                "legacyReceipt": refs["legacyReceipt"].json(),
                "manifest": refs["manifest"].json(),
                "output": refs["output"].json(),
                "inputs": {str(p): d for p, d in pins.items()},
                "readerClosureContract": "closed-reader-superset-plus-reconcile-leaf",
                "expected": expected.json(),
                "motionRanges": list(expected.ranges),
                "expectationBasis": "pinned-part-plan-and-Motion-duration-agreement",
                "motionRangesUsedForExpectedClock": False,
                "probe": probe,
                "fullAV": {"command": command, "returncode": decoded.returncode},
                "openFilesBefore": before,
                "openFilesAfter": after,
                "digestMetrics": {
                    "scope": "explicit-fresh-cache-only",
                    "hashReads": cache.hash_reads,
                    "bytesRead": cache.bytes_read,
                    "cacheHits": cache.cache_hits,
                },
                "officialTerminalVerified": False,
                "terminalVerified": False,
                "historicalInputsVerified": False,
                "currentDeclaredInputsVerified": True,
                "currentOutputVerified": True,
                "candidateOnly": True,
                "qualityApproved": False,
                "deliveryApproved": False,
            },
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    reconcile(args.request, args.receipt, resume=args.resume)


if __name__ == "__main__":
    main()
