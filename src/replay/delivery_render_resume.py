"""Prepare isolated render retries from pinned historical and assembly evidence.

Preparation never renders or changes old artifacts. Run offline, with all old
workflow owners stopped. The manifest explicitly pins config, ordered manifests,
workflows (each with every canonical/preserved receipt), assembly index/receipt,
and completed movies keyed by global render index. Completed movies are decoded
independently, not adopted from native response fields. Use the generated config
with pipeline --start render; call verify before joining after any interruption.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from replay import delivery_render as render
from replay import native_invocation, source_caption_binding
from replay.core import ReplayError
from replay.native_edit_parts import _selection, partition
from replay.recordings import record
from replay.stages import Stage, StageError, file_digest
from replay.workflow import parse_stages

MAX_PARTS = 10000
MAX_MOVIE_FRAMES = 36000
MovieValidator = Callable[[Path, int, int, Path], None]


@dataclass
class Pins:
    values: dict[str, str] = field(default_factory=dict)
    identities: dict[str, list[int]] = field(default_factory=dict)

    def bind(self, path: Path, expected: object) -> Path:
        checked_path(str(path))
        before = identity(path)
        actual = file_digest(path)
        if (
            expected != actual
            or self.values.get(str(path), actual) != actual
            or identity(path) != before
        ):
            raise ReplayError(f"Changed input pin: {path}")
        self.values[str(path)] = actual
        self.identities[str(path)] = before
        return path

    def ref(self, value: object) -> Path:
        row = record(value)
        if set(row) != {"path", "sha256"}:
            raise ReplayError("Expected exact path/sha256 reference")
        return self.bind(checked_path(row["path"]), row["sha256"])

    def check(self) -> None:
        for path, sha in tuple(self.values.items()):
            self.bind(Path(path), sha)


def checked_path(value: object, *, local: bool = False) -> Path:
    path = render._path(value)
    if path != path.resolve() or (local and not path.is_relative_to(Path.cwd().resolve())):
        raise ReplayError("Paths must be unlinked and evidence checkout-local")
    return path


def document(path: Path) -> dict[str, object]:
    return record(render._json(path))


def array(value: object) -> list[object]:
    rows = render._list(value)
    if not 1 <= len(rows) <= MAX_PARTS:
        raise ReplayError("Empty or excessive inventory")
    return rows


def preserved(path: Path, sha: object, pins: Pins) -> Path:
    """Never repair canonical inputs or scan past a changed canonical file."""
    checked_path(str(path))
    if path.exists():
        return pins.bind(path, sha)
    candidates = sorted(path.parent.parent.glob(f"{path.parent.name}.failed-*/{path.name}"))
    matching = [p for p in candidates if file_digest(checked_path(str(p))) == sha]
    if not matching:
        raise ReplayError("No preserved request matches its manifest hash")
    return pins.bind(matching[0], sha)


def assembly(index: Path, receipt: Path, config: render.Config, pins: Pins) -> render.Chunk:
    facts = document(receipt)
    if (
        facts.get("version") != 1
        or facts.get("indexSHA256") != pins.values[str(index)]
        or facts.get("exactPresentationClockVerified") is not True
        or facts.get("compressedPacketsIdentical") is not True
    ):
        raise ReplayError("Assembly receipt lacks exact clock/payload facts")
    inputs = record(facts.get("inputs"))
    if inputs.get(str(config.chunks)) != file_digest(config.chunks):
        raise ReplayError("Assembly does not bind the original chunk index")
    for path, sha in inputs.items():
        pins.bind(checked_path(path), sha)
    probes = record(facts.get("probeEvidence"))
    if not probes:
        raise ReplayError("Assembly probe evidence is missing")
    for path, sha in probes.items():
        pins.bind(checked_path(path), sha)
    old, new = render.read_chunks(config.chunks), render.read_chunks(index)
    rows = array(document(index).get("ranges"))
    if len(new) != 1 or new[0].start != old[0].start or new[0].end != old[-1].end:
        raise ReplayError("Assembly must preserve the exact complete original source range")
    row = record(rows[0])
    original_frames = sum(
        render._integer(record(r).get("frames"))
        for r in array(document(config.chunks).get("ranges"))
    )
    if facts.get("frames") != row.get("frames") or row.get("frames") != original_frames:
        raise ReplayError("Assembly frame accounting differs")
    media = pins.ref(facts.get("output"))
    if media != new[0].media or row.get("sha256") != pins.values[str(media)]:
        raise ReplayError("Assembly output/index media binding differs")
    if any(str(c.media) not in inputs for c in old):
        raise ReplayError("Assembly does not bind every old candidate")
    return new[0]


def workflow_inventory(
    rows: object, pins: Pins
) -> dict[str, tuple[Stage, dict[str, object] | None]]:
    result: dict[str, tuple[Stage, dict[str, object] | None]] = {}
    for value in array(rows):
        row = record(value)
        if set(row) != {"plan", "receipts"}:
            raise ReplayError("Unknown workflow inventory fields")
        path = pins.ref(row["plan"])
        state = path.with_name(f"{path.stem}-state")
        receipts = [pins.ref(ref) for ref in render._list(row["receipts"])]
        discovered = set(state.glob("*.json")) | set(
            state.parent.glob(f"{state.name}.failed-*/*.json")
        )
        if set(receipts) != discovered or len(set(receipts)) != len(receipts):
            raise ReplayError("Workflow receipt inventory changed or is incomplete")
        stages = parse_stages(document(path))
        if any(p.stem not in {s.name for s in stages} for p in receipts):
            raise ReplayError("Unknown stage receipt")
        for stage in stages:
            saved = [document(p) for p in receipts if p.stem == stage.name]
            if any(r.get("status") not in {"running", "complete"} for r in saved):
                raise ReplayError("Unknown stage status")
            complete = [r for r in saved if r.get("status") == "complete"]
            if any(item != complete[0] for item in complete[1:]) or stage.name in result:
                raise ReplayError("Conflicting render stage/completion")
            result[stage.name] = (stage, complete[0] if complete else None)
    return result


def fingerprint(stage: Stage, inputs: dict[str, str]) -> str:
    definition = {
        "command": stage.command,
        "verifyCommand": stage.verify_command,
        "inputs": inputs,
        "outputs": [str(p) for p in stage.outputs],
        "timeoutSeconds": stage.timeout_seconds,
        "workingDirectory": str(Path.cwd().resolve()),
    }
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def validate_movie(movie: Path, frames: int, rate: int, evidence: Path) -> None:
    """Read-only complete video decode and exact zero-based CFR support check."""
    evidence.mkdir(mode=0o700)
    env = {**os.environ, **dict.fromkeys(("TMPDIR", "TMP", "TEMP"), str(evidence))}
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=codec_type,time_base:frame=media_type,pts,duration",
            "-of",
            "json",
            str(movie),
        ],
        capture_output=True,
        timeout=render.RENDER_TIMEOUT_SECONDS,
        env=env,
        check=True,
    )
    (evidence / "probe.json").write_bytes(probe.stdout)
    (evidence / "probe.stderr").write_bytes(probe.stderr)
    if probe.stderr or len(probe.stdout) > render.MAX_JSON_BYTES:
        raise ReplayError("Movie probe has errors or exceeds budget")
    data = record(json.loads(probe.stdout))
    streams = array(data.get("streams"))
    if len(streams) != 1 or record(streams[0]).get("codec_type") != "video":
        raise ReplayError("Expected one video-only native render")
    base = record(streams[0]).get("time_base")
    if not isinstance(base, str):
        raise ReplayError("Missing movie clock")
    scale = Fraction(base)
    measured = render._list(data.get("frames"))
    if len(measured) > MAX_MOVIE_FRAMES:
        raise ReplayError("Movie frame budget exceeded")
    if len(measured) != frames:
        raise ReplayError("Decoded movie frame count differs")
    for index, raw in enumerate(measured):
        frame = record(raw)
        pts, duration = frame.get("pts"), frame.get("duration")
        if (
            type(pts) is not int
            or type(duration) is not int
            or pts * scale != Fraction(index, rate)
            or duration * scale != Fraction(1, rate)
        ):
            raise ReplayError("Decoded movie clock differs")
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(movie),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=render.RENDER_TIMEOUT_SECONDS,
        env=env,
        check=True,
    )
    (evidence / "decode.stderr").write_bytes(result.stderr)
    if result.stderr:
        raise ReplayError("Movie full decode reported errors")
    render._write_new(
        evidence / "receipt.json",
        {
            "movie": {"path": str(movie), "sha256": file_digest(movie)},
            "frames": frames,
            "rate": rate,
            "probeSHA256": file_digest(evidence / "probe.json"),
            "decodeExitCode": result.returncode,
            "exactCFRClockVerified": True,
            "fullDecodePassed": True,
            "qualityApproved": False,
        },
    )


def source_start(clip: dict[str, object], chunk: render.Chunk, rate: int) -> Fraction:
    """Recover the exact planned frame-grid start, never quantize a VFR origin."""
    selection = record(clip.get("selection"))
    raw = selection.get("startSeconds")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ReplayError("Invalid original selection start")
    frame = round((Fraction(raw) + chunk.start) * rate)
    start = Fraction(frame, rate)
    if float(start - chunk.start) != raw:
        raise ReplayError("Original local start does not exactly encode its source-grid mapping")
    return start


def rebase(
    request: dict[str, object], config: render.Config, target: render.Chunk
) -> dict[str, object]:
    changed = copy.deepcopy(request)
    recipe = record(changed.get("recipe"))
    [clip_raw] = array(recipe.get("clips"))
    clip = record(clip_raw)
    source = checked_path(clip.get("sourcePath"))
    matches = [c for c in render.read_chunks(config.chunks) if c.media == source]
    if len(matches) != 1:
        raise ReplayError("Unknown or ambiguous original source chunk")
    selection = _selection(clip.get("selection"))
    start = source_start(clip, matches[0], config.rate) - target.start
    if start < 0 or start + selection.duration > target.end - target.start:
        raise ReplayError("Rebased selection lacks full assembled source support")
    clip["sourcePath"] = str(target.media)
    clip["selection"] = {**record(clip["selection"]), "startSeconds": float(start)}
    recipe["clips"] = [clip]
    changed["recipe"] = recipe
    return changed


def original_parts(
    manifests: object, config: render.Config, pins: Pins
) -> list[tuple[Path, Path, int]]:
    paths = [pins.ref(ref) for ref in array(manifests)]
    discovered = sorted(config.composition.glob("parts-[0-9][0-9][0-9][0-9]/manifest.json"))
    if paths != discovered:
        raise ReplayError("Original ordered manifest inventory differs")
    parts: list[tuple[Path, Path, int]] = []
    for path in paths:
        raw = document(path)
        source = pins.ref(raw.get("sourceRequest"))
        expected = partition(document(source))
        rows = array(raw.get("parts"))
        if len(rows) != len(expected):
            raise ReplayError("Partition count differs from source request")
        for value, (offset, frames, request) in zip(rows, expected, strict=True):
            row = record(value)
            canonical = checked_path(row.get("path"))
            available = preserved(canonical, row.get("sha256"), pins)
            actual = document(available)
            if (
                row.get("offsetFrame") != offset
                or row.get("frames") != frames
                or {**actual, "outputDirectory": request["outputDirectory"]} != request
            ):
                raise ReplayError("Partition changed source request or frame ordering")
            parts.append((canonical, available, frames))
    remaining = [
        (Fraction(c.start, config.rate), Fraction(c.end, config.rate))
        for c in render.read_cuts(config.cuts)
    ]
    cursor = 0
    start = remaining[0][0]
    chunks = {str(c.media): c for c in render.read_chunks(config.chunks)}
    for _, available, frames in parts:
        clip = record(array(record(document(available)["recipe"])["clips"])[0])
        chunk = chunks.get(str(clip.get("sourcePath")))
        selected = _selection(clip.get("selection"))
        if (
            chunk is None
            or cursor >= len(remaining)
            or source_start(clip, chunk, config.rate) != start
            or selected.duration != Fraction(frames, config.rate)
            or start + selected.duration > remaining[cursor][1]
        ):
            raise ReplayError("Original sequence source clocks differ from cuts")
        start += selected.duration
        if start == remaining[cursor][1]:
            cursor += 1
            if cursor < len(remaining):
                start = remaining[cursor][0]
    if cursor != len(remaining):
        raise ReplayError("Original sequence frame total differs from cuts")
    return parts


def prepare(
    manifest: Path, output: Path, *, sha256: str, validator: MovieValidator = validate_movie
) -> Path:
    """Create requests/workflows only; retain all old media, state and input bytes."""
    output = checked_path(str(output), local=True)
    if output.exists():
        raise FileExistsError(output)
    pins = Pins()
    raw = document(pins.bind(checked_path(str(manifest), local=True), sha256))
    if (
        set(raw)
        != {
            "version",
            "config",
            "native",
            "cuts",
            "index",
            "assembly",
            "manifests",
            "workflows",
            "completed",
        }
        or raw["version"] != 1
    ):
        raise ReplayError("Unknown resume manifest fields/version")
    pins.bind(Path(__file__).resolve(), file_digest(Path(__file__).resolve()))
    config_path = pins.ref(raw["config"])
    config = render.load_config(config_path)
    if pins.ref(raw["native"]) != config.native or pins.ref(raw["cuts"]) != config.cuts:
        raise ReplayError("Native/cut pins differ from original configuration")
    index = pins.ref(raw["index"])
    target = assembly(index, pins.ref(raw["assembly"]), config, pins)
    parts = original_parts(raw["manifests"], config, pins)
    stages = workflow_inventory(raw["workflows"], pins)
    if set(stages) != {f"render-{i:06d}" for i in range(len(parts))}:
        raise ReplayError("Workflow stages differ from complete ordered part inventory")
    complete = record(raw["completed"])
    expected_complete = {
        str(i) for i in range(len(parts)) if stages[f"render-{i:06d}"][1] is not None
    }
    if set(complete) != expected_complete:
        raise ReplayError("Every completed movie must be pinned, no unknown completions")
    if len(complete) == len(parts):
        raise ReplayError("No unfinished render parts; use verified originals, not a retry plan")
    storage = (
        checked_path(str(config.storage))
        / f"resume-{hashlib.sha256(str(output).encode()).hexdigest()}"
    )
    targets = [
        storage,
        *(
            config.work / name
            for name in (
                "join-plan.json",
                "join-workflow.json",
                "encode-job.json",
                "final-mux-workflow.json",
                "final-job.json",
            )
        ),
        *(
            config.storage / name
            for name in (
                "join-inputs",
                "joined-video.mp4",
                "encoded-video.mov",
                "av-master.mov",
                "final",
            )
        ),
    ]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise ReplayError("Retry or downstream output namespace is occupied")
    output.mkdir(mode=0o700)
    entries: list[dict[str, object]] = []
    retries: list[dict[str, object]] = []
    for number, (canonical, available, frames) in enumerate(parts):
        stage, saved = stages[f"render-{number:06d}"]
        response = canonical.with_name("render.response.json")
        expected_command = (
            "/bin/bash",
            "-c",
            'exec "$1" edit-media "$2" > "$3"',
            "native-render",
            str(config.native),
            str(canonical),
            str(response),
        )
        richer = stage.command == native_invocation.command(
            config.native, canonical, config.storage
        )
        if richer:
            if (
                stage.outputs != native_invocation.paths(canonical)
                or stage.verify_command
                != native_invocation.command(config.native, canonical, config.storage, verify=True)
                or stage.adopt_existing
                or not set(
                    (
                        *native_invocation.inputs(config.native, available),
                        Path(native_invocation.__file__).resolve(),
                    )
                ).issubset(stage.inputs)
            ):
                raise ReplayError("Richer native invocation contract changed")
            if saved is None and any(
                path.exists() or path.is_symlink()
                for path in (
                    *native_invocation.paths(canonical),
                    canonical.with_name("render.stderr"),
                    canonical.with_name("render.attempt"),
                )
            ):
                raise ReplayError(
                    "Unknown native outcome requires reconciliation; no duplicate render"
                )
        elif (
            stage.command != expected_command
            or stage.inputs != (canonical, config.native)
            or stage.outputs != (response,)
            or stage.verify_command
            or stage.adopt_existing
        ):
            raise ReplayError("Original native stage contract changed")
        request = document(available)
        if saved is not None:
            if canonical != available:
                raise ReplayError("Completed canonical request is missing")
            digest = fingerprint(
                stage,
                {
                    str(path): file_digest(pins.bind(path, file_digest(path)))
                    for path in stage.inputs
                },
            )
            if saved != {
                "status": "complete",
                "fingerprint": digest,
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            }:
                raise ReplayError("Completed stage fingerprint/response changed")
            for artifact in stage.outputs:
                pins.bind(artifact, file_digest(artifact))
            if richer:
                native_invocation.verify(config.native, canonical, config.storage)
            movie = pins.ref(complete[str(number)])
            result = document(response)
            if result.get("outputPath") != str(movie) or result.get("frameCount") != frames:
                raise ReplayError("Completed movie differs from response")
            evidence = output / f"validation-{number:06d}"
            validator(movie, frames, config.rate, evidence)
            for proof in evidence.glob("*.json"):
                pins.bind(proof, file_digest(proof))
            path = canonical
        else:
            if response.exists():
                raise ReplayError("Unregistered response requires outcome reconciliation")
            request = rebase(request, config, target)
            request["outputDirectory"] = str(storage / f"part-{number:06d}")
            path = render._write_new(output / f"retry-{number:06d}" / "request.json", request)
            retries.append(
                {
                    "name": stage.name,
                    "command": list(native_invocation.command(config.native, path, config.storage)),
                    "verify_command": list(
                        native_invocation.command(config.native, path, config.storage, verify=True)
                    ),
                    "inputs": [
                        *map(str, native_invocation.inputs(config.native, path)),
                        str(Path(native_invocation.__file__).resolve()),
                        str(output / "resume.json"),
                    ],
                    "outputs": list(map(str, native_invocation.paths(path))),
                    "timeout_seconds": render.RENDER_TIMEOUT_SECONDS,
                }
            )
        for layer in render._list(record(request.get("recipe")).get("additionalVideo", [])):
            media = checked_path(record(layer).get("sourcePath"))
            pins.bind(media, file_digest(media))
        entries.append(
            {
                "path": str(path),
                "sha256": file_digest(path),
                "frames": frames,
                "offsetFrame": sum(render._integer(p["frames"]) for p in entries),
                "reused": saved is not None,
                "invocationContract": "native-invocation-v1"
                if richer or saved is None
                else "legacy-direct-native",
            }
        )
    pins.check()
    config_raw = {
        **document(config_path),
        "sourceChunks": str(index),
        "compositionDirectory": str(output),
    }
    new_config = render._write_new(output / "config.json", config_raw)
    cleanup = source_caption_binding.require_cleanup(new_config)
    bindings: dict[str, Path] = {}
    for retry_stage in retries:
        name = str(retry_stage["name"])
        number = int(name.removeprefix("render-"))
        request_path = Path(str(entries[number]["path"]))
        binding = source_caption_binding.create_binding(
            cleanup,
            consumer=request_path,
            kind="native-request",
            output=request_path.with_name("source-caption-binding.json"),
        )
        retry_stage["inputs"] = list(
            dict.fromkeys(
                [
                    *[str(value) for value in render._list(retry_stage["inputs"])],
                    str(binding),
                    str(cleanup.clean),
                ]
            )
        )
        entries[number]["sourceCaptionBinding"] = str(binding)
        pins.bind(binding, file_digest(binding))
        bindings[name] = binding
    for entry in entries:
        pins.bind(Path(str(entry["path"])), entry["sha256"])
    pins.bind(new_config, file_digest(new_config))
    mixed = render._write_new(
        output / "parts-0000" / "manifest.json", {"version": 1, "parts": entries}
    )
    pins.bind(mixed, file_digest(mixed))
    render._write_new(
        output / "resume.json",
        {
            "version": 2
            if any(p["invocationContract"] == "native-invocation-v1" for p in entries)
            else 1,
            "authorizedOutputDirectory": str(config.storage),
            "inputs": pins.values,
            "native": str(config.native),
            "config": str(new_config),
            "parts": entries,
            "identities": pins.identities,
        },
    )
    render._workflows(output, "render", retries, 3, bindings=bindings)
    return new_config


def identity(path: Path) -> list[int]:
    """Identity of already hashed bytes; ctime catches writes restoring mtime."""
    checked_path(str(path))
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def verify(path: Path, *, quick: bool = False) -> dict[str, object]:
    """Recheck the sealed immutable input/movie bindings before execution or join."""
    raw = document(checked_path(str(path), local=True))
    if raw.get("version") not in (1, 2):
        raise ReplayError("Unsupported resume seal")
    if raw.get("version") == 2:
        checked_path(raw.get("authorizedOutputDirectory"))
        if any(
            record(row).get("invocationContract")
            not in {"native-invocation-v1", "legacy-direct-native"}
            for row in array(raw.get("parts"))
        ):
            raise ReplayError("Unsupported richer resume invocation contract")
    pins = Pins()
    for name, sha in record(raw.get("inputs")).items():
        source = checked_path(name)
        if quick:
            if record(raw.get("identities")).get(name) != identity(source):
                raise ReplayError("Previously hashed input identity changed; revalidate offline")
        else:
            pins.bind(source, sha)
    return raw


def run(path: Path, number: int) -> None:
    raw = verify(path, quick=True)
    rows = array(raw.get("parts"))
    if not 0 <= number < len(rows):
        raise ReplayError("Unknown render index")
    row = record(rows[number])
    if row.get("reused") is not False:
        raise ReplayError("Completed parts must never be rerendered")
    request = checked_path(row.get("path"))
    if (
        raw.get("version") == 1
        and row.get("invocationContract", "legacy-direct-native") != "legacy-direct-native"
    ):
        raise ReplayError("Richer invocation cannot use a legacy resume seal")
    cleanup = source_caption_binding.require_cleanup(checked_path(raw.get("config")))
    binding = checked_path(row.get("sourceCaptionBinding"))
    source_caption_binding.validate_binding(
        binding,
        consumer_inputs=(binding, cleanup.clean, request),
        consumer_command=native_invocation.command(
            checked_path(raw.get("native")),
            request,
            checked_path(raw.get("authorizedOutputDirectory")),
        ),
    )
    if raw.get("version") != 2 or row.get("invocationContract") != "native-invocation-v1":
        raise ReplayError("Legacy resume execution requires a new verified preparation")
    native_invocation.run(
        checked_path(raw.get("native")),
        request,
        checked_path(raw.get("authorizedOutputDirectory")),
    )
    verify(path, quick=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("manifest", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--sha256", required=True)
    check = commands.add_parser("verify")
    check.add_argument("seal", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("seal", type=Path)
    execute.add_argument("number", type=int)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            print(prepare(args.manifest, args.output, sha256=args.sha256))
        elif args.action == "verify":
            verify(args.seal)
        else:
            run(args.seal, args.number)
    except (ReplayError, StageError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"RENDER_RESUME_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
