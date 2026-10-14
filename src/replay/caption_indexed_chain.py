"""Derive ordinary caption workflows from actual completed native indices."""

import argparse
import hashlib
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from replay import caption_native_inputs, caption_reflow_verify
from replay.cleanup_lane_successor import _unlocked
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.stages import _fingerprint, file_digest
from replay.workflow import load_stages, parse_stages

KIND = "caption-indexed-chain"
FIELDS = {
    "version",
    "kind",
    "recognitionDirectory",
    "source",
    "pagingCommandTemplate",
    "selectionGroups",
    "cacheRoot",
    "cacheSegments",
    "inputPins",
}


def _path(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("Expected canonical path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("Expected canonical path")
    return path


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Expected nonnegative integer")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("Expected SHA256")
    return value


def _numbers(value: object, *, empty: bool = False) -> list[int]:
    if not isinstance(value, list) or not (0 if empty else 1) <= len(value) <= 512:
        raise ValueError("Expected bounded segment list")
    return [_integer(item) for item in value]


def _read(path: Path, pins: dict[str, str]) -> dict[str, object]:
    data = bounded_read(path)
    _bind(path, hashlib.sha256(data).hexdigest(), pins)
    return decode(data)


def _bind(path: Path, digest: str, pins: dict[str, str]) -> None:
    if pins.get(str(path), digest) != digest or file_digest(path) != digest:
        raise ValueError("Indexed-chain input changed")
    pins[str(path)] = digest


@dataclass(frozen=True)
class Plan:
    workflow: dict[str, object]
    receipt: dict[str, object]
    pins: dict[str, str]
    spans: tuple[tuple[int, int], ...]
    source: Path
    rate: int


def _derive(request: Path, output: Path) -> Plan:
    request, output = _path(str(request)), _path(str(output))
    pins: dict[str, str] = {}
    data = _read(request, pins)
    if set(data) != FIELDS or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Expected exact version1 indexed-chain request")
    if data["kind"] != KIND:
        raise ValueError("Unknown indexed-chain kind")
    directory, cache = _path(data["recognitionDirectory"]), _path(data["cacheRoot"])
    source_ref = record(data["source"])
    if set(source_ref) != {"path", "sha256"}:
        raise ValueError("Expected source reference")
    source = _path(source_ref["path"])
    _bind(source, _digest(source_ref["sha256"]), pins)
    fixed = record(data["inputPins"])
    if not 1 <= len(fixed) <= 220:
        raise ValueError("Expected bounded fixed inputs")
    for name, digest in fixed.items():
        _bind(_path(name), _digest(digest), pins)
    _bind(Path(__file__).resolve(), file_digest(Path(__file__)), pins)
    index = _read(directory / "segments.json", pins)
    _read(directory / "receipt.json", pins)
    caption_native_inputs.verify(directory)
    rate = _integer(index.get("rate"))
    rows = index.get("segments")
    if (
        type(index.get("version")) is not int
        or index.get("version") != 1
        or rate != 30
        or not isinstance(rows, list)
    ):
        raise ValueError("Expected native frame index")
    if not 1 <= len(rows) <= 512:
        raise ValueError("Expected bounded native index")
    previous_offset = -1
    members: list[dict[str, object]] = []
    for number, value in enumerate(rows):
        row = record(value)
        frames = _integer(row.get("frames"))
        offset = _integer(row.get("offsetFrame"))
        if _integer(row.get("segment")) != number or offset < previous_offset:
            raise ValueError("Native index order is invalid")
        if not 0 < frames <= 120 * rate or offset + frames > 10_000_000:
            raise ValueError("Native segment has invalid duration")
        previous_offset = offset
        members.append(row)
    groups_raw = data["selectionGroups"]
    if not isinstance(groups_raw, list) or not 1 <= len(groups_raw) <= 512:
        raise ValueError("Expected bounded selection groups")
    groups = [_numbers(value) for value in groups_raw]
    selected: set[int] = set()
    for group in groups:
        if group != list(range(group[0], group[-1] + 1)):
            raise ValueError("Selection group is discontinuous")
        if group[-1] >= len(members) or selected.intersection(group):
            raise ValueError("Selection groups overlap or exceed index")
        selected.update(group)
    requested_groups = groups
    groups = []
    for requested in requested_groups:
        run = [requested[0]]
        for number in requested[1:]:
            previous = members[run[-1]]
            previous_end = _integer(previous["offsetFrame"]) + _integer(previous["frames"])
            if _integer(members[number]["offsetFrame"]) != previous_end:
                groups.append(run)
                run = []
            run.append(number)
        groups.append(run)
    caches = _numbers(data["cacheSegments"], empty=True)
    if len(set(caches)) != len(caches) or any(n >= len(members) for n in caches):
        raise ValueError("Invalid cache selection")
    cache_files: dict[int, Path] = {}
    for number in caches:
        path = cache / f"part-{number:04d}.json"
        if not path.exists():
            path = cache / f"failure-{number:04d}.json"
        document = _read(path, pins)
        if not any(name.endswith(".request.json") for name in record(document.get("inputs"))):
            raise ValueError("Cache has no pinned glyph requests")
        cache_files[number] = path
    template_raw = data["pagingCommandTemplate"]
    if not isinstance(template_raw, list) or not template_raw or len(template_raw) > 200:
        raise ValueError("Expected bounded paging command")
    template: list[str] = []
    for item in template_raw:
        if not isinstance(item, str) or not item or "\0" in item:
            raise ValueError("Invalid paging argument")
        template.append(item)
    if template.count("{recognition}") != 1 or template.count("{output}") != 1:
        raise ValueError("Expected exact paging placeholders")
    if "replay.caption_native_paging" not in template or "--source-media" not in template:
        raise ValueError("Expected normal native paging command")
    forbidden = {
        "--segment",
        "--measurement-cache",
        "--measurement-cache-segment",
        "--audit-support-only",
        "--resume-producer-evidence",
        "--token-sources",
    }
    if forbidden.intersection(template):
        raise ValueError("Selections belong to indexed-chain request")
    source_position = template.index("--source-media") + 1
    if source_position >= len(template) or template[source_position] != str(source):
        raise ValueError("Paging command source differs")
    module_position = template.index("replay.caption_native_paging")
    if template[module_position + 1 : module_position + 3] != ["{recognition}", "{output}"]:
        raise ValueError("Paging positional arguments differ")
    if module_position + 3 >= len(template) or template[module_position + 3] not in fixed:
        raise ValueError("Native executable lacks fixed pin")
    for i, argument in enumerate(template):
        if argument in {"--evidence", "--long-executable"} and (
            i + 1 >= len(template) or template[i + 1] not in fixed
        ):
            raise ValueError("Native evidence lacks fixed pin")
    stages: list[dict[str, object]] = []
    spans: list[tuple[int, int]] = []
    for span, group in enumerate(groups):
        for number in group:
            path = caption_native_inputs.segment_binding(directory, members[number])
            document = _read(path, pins)
            for name, digest in record(document.get("inputs")).items():
                _bind(_path(name), _digest(digest), pins)
        start = _integer(members[group[0]]["offsetFrame"])
        end = _integer(members[group[-1]]["offsetFrame"]) + _integer(members[group[-1]]["frames"])
        spans.append((start, end - start))
        target = output / f"span-{span:03d}"
        command = [
            str(directory)
            if item == "{recognition}"
            else str(target)
            if item == "{output}"
            else item
            for item in template
        ]
        command.extend(arg for n in group for arg in ("--segment", str(n)))
        used_cache = [n for n in group if n in cache_files]
        if used_cache:
            command.extend(("--measurement-cache", str(cache)))
            command.extend(
                arg for n in used_cache for arg in ("--measurement-cache-segment", str(n))
            )
        inputs = [
            str(request),
            str(directory / "segments.json"),
            str(directory / "receipt.json"),
            str(output / "prepared.json"),
            str(source),
            str(Path(__file__).resolve()),
            *fixed,
            *(str(cache_files[n]) for n in used_cache),
        ]
        stage_inputs = list(dict.fromkeys(inputs))
        stage = {
            "name": f"native-paging-span-{span:03d}",
            "inputs": stage_inputs,
            "outputs": [
                str(target / "reflow.json"),
                *(str(target / f"part-{n:04d}.json") for n in group),
            ],
            "command": command,
            "verify_command": [
                sys.executable,
                "-B",
                "-m",
                "replay.caption_indexed_chain",
                "verify-span",
                str(request),
                str(output),
                "--span",
                str(span),
            ],
            "timeout_seconds": 3600,
        }
        if len(stage_inputs) > 256 or len(command) > 256:
            raise ValueError("Child stage exceeds normal workflow limits")
        stages.append(stage)
    workflow: dict[str, object] = {"version": 1, "stages": stages}
    parse_stages(workflow)
    snapshot = InputHashSnapshot.capture((Path(name), digest) for name, digest in pins.items())
    snapshot.finish()
    receipt: dict[str, object] = {
        "version": 1,
        "kind": KIND,
        "candidateOnly": True,
        "fullSRTVerified": False,
        "excludedSegments": [n for n in range(len(members)) if n not in selected],
        "requestedGroups": requested_groups,
        "derivedGroups": groups,
        "pins": pins,
        "spans": [[start, frames] for start, frames in spans],
        "workflow": workflow,
    }
    return Plan(workflow, receipt, pins, tuple(spans), source, rate)


def prepare(request: Path, output: Path) -> Path:
    """Publish a new ordinary child workflow from actual verified index bytes."""
    if output.exists() or output.is_symlink():
        raise ValueError("Occupied indexed-chain output; preserve it")
    plan = _derive(request, output)
    snapshot = InputHashSnapshot.capture((Path(name), digest) for name, digest in plan.pins.items())
    output.mkdir(mode=0o700)
    publish(output / "prepared.json", encode(plan.receipt))
    publish(output / "child-workflow.json", encode(plan.workflow))
    snapshot.finish()
    return output / "child-workflow.json"


def verify_plan(request: Path, output: Path) -> Plan:
    """Reconstruct both prepared artifacts without rewriting any evidence."""
    plan = _derive(request, output)
    snapshot = InputHashSnapshot.capture((Path(name), digest) for name, digest in plan.pins.items())
    if decode(bounded_read(output / "prepared.json")) != plan.receipt:
        raise ValueError("Prepared indexed-chain receipt changed")
    if decode(bounded_read(output / "child-workflow.json")) != plan.workflow:
        raise ValueError("Prepared child workflow changed")
    snapshot.finish()
    return plan


def verify_span(request: Path, output: Path, span: int) -> None:
    """Require the normal reflow verifier on the exact current indexed clock."""
    plan = verify_plan(request, output)
    if type(span) is not int or not 0 <= span < len(plan.spans):
        raise ValueError("Unknown indexed span")
    start, frames = plan.spans[span]
    snapshot = InputHashSnapshot.capture((Path(name), digest) for name, digest in plan.pins.items())
    caption_reflow_verify.verify(
        output / f"span-{span:03d}" / "reflow.json",
        source=plan.source,
        frames=frames,
        rate=plan.rate,
        start_frame=start,
    )
    snapshot.finish()


def verify_child(request: Path, output: Path, state: Path) -> None:
    """Check genuine unlocked child completion records and every span domain."""
    state = _path(str(state))
    plan = verify_plan(request, output)
    stages = load_stages(output / "child-workflow.json")
    snapshot = InputHashSnapshot.capture((Path(name), digest) for name, digest in plan.pins.items())
    with ExitStack() as leases:
        leases.enter_context(_unlocked(state / ".workflow.lock"))
        for span, stage in enumerate(stages):
            leases.enter_context(_unlocked(state / f"{stage.name}.lock"))
            expected = {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            }
            if decode(bounded_read(state / f"{stage.name}.json")) != expected:
                raise ValueError("Child stage lacks genuine current completion")
            verify_span(request, output, span)
        snapshot.finish()


class Arguments(argparse.Namespace):
    operation: str
    request: Path
    output: Path
    span: int | None
    state: Path | None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("prepare", "verify-plan", "verify-span", "verify-child")
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--span", type=int)
    parser.add_argument("--state", type=Path)
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    if args.operation == "prepare":
        prepare(args.request, args.output)
    elif args.operation == "verify-plan":
        verify_plan(args.request, args.output)
    elif args.operation == "verify-span" and args.span is not None:
        verify_span(args.request, args.output, args.span)
    elif args.operation == "verify-child" and args.state is not None:
        verify_child(args.request, args.output, args.state)
    else:
        parser.error("Verification requires --span or --state")


if __name__ == "__main__":
    main()
