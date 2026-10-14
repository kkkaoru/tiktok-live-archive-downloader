"""Prepare exact caption preview XML from genuinely completed clean lineage."""

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import code_dependency_inventory as inventory
from replay import completion_barrier as barrier
from replay import fcp_transformed_source as transformed
from replay import source_caption_binding as binding
from replay import workflow
from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_output_window import MAX_XML_BYTES, window
from replay.fcp_part_assembly import _reference, _save
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
FIELDS = frozenset(
    {
        "version",
        "kind",
        "sourceConfig",
        "sourceBinding",
        "sourceXML",
        "completion",
        "windows",
        "receiptOutput",
    }
)


@dataclass(frozen=True)
class Window:
    start: Fraction
    end: Fraction
    output: Path
    binding: Path


@dataclass(frozen=True)
class Plan:
    request: Path
    config: Path
    binding: Path
    source: Path
    completion: Path
    windows: tuple[Window, ...]
    receipt: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            *tuple(path for item in self.windows for path in (item.output, item.binding)),
            self.receipt,
        )


def _local(value: object) -> Path:
    path = _path(value)
    if not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Post-clean metadata must remain inside checkout edits/work")
    return path


def load(request: Path, *, sha256: str) -> Plan:
    _local(str(request))
    _pin({"path": str(request), "sha256": sha256})
    data = _json(request)
    rows = data.get("windows")
    if (
        set(data) != FIELDS
        or data.get("version") != 1
        or data.get("kind") != "post-clean-xml-plan"
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= 8
    ):
        raise ReplayError("Invalid bounded post-clean plan schema")
    windows: list[Window] = []
    for raw in rows:
        row = record(raw)
        if (
            set(row) != {"start", "end", "output", "bindingOutput"}
            or not isinstance(row.get("start"), str)
            or not isinstance(row.get("end"), str)
        ):
            raise ReplayError("Expected exact rational preview window")
        try:
            start, end = Fraction(str(row["start"])), Fraction(str(row["end"]))
        except (ValueError, ZeroDivisionError) as error:
            raise ReplayError("Invalid rational preview clock") from error
        if not 0 <= start < end:
            raise ReplayError("Invalid preview window order")
        windows.append(
            Window(start, end, _local(row.get("output")), _local(row.get("bindingOutput")))
        )
    plan = Plan(
        request,
        _local(data.get("sourceConfig")),
        _local(data.get("sourceBinding")),
        _local(data.get("sourceXML")),
        _local(data.get("completion")),
        tuple(windows),
        _local(data.get("receiptOutput")),
    )
    inputs = {
        request,
        plan.config,
        plan.binding,
        plan.source,
        plan.completion,
        plan.source.with_suffix(".source-lineage.json"),
    }
    if len(set(plan.outputs)) != len(plan.outputs) or inputs.intersection(plan.outputs):
        raise ReplayError("Post-clean inputs and outputs must be distinct")
    return plan


def _inputs(plan: Plan) -> tuple[Path, ...]:
    config = _json(plan.config)
    top = tuple(
        _path(record(value).get("path"))
        for value in config.values()
        if isinstance(value, dict) and "path" in value
    )
    manifest = _json(_pin(config.get("assemblyManifest")))
    receipt = _json(_pin(config.get("assemblyReceipt")))
    rows = manifest.get("chunks")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ReplayError("Expected bounded genuine cleanup chunks")
    paths = [
        plan.request,
        plan.config,
        plan.binding,
        plan.source,
        plan.completion,
        plan.source.with_suffix(".source-lineage.json"),
        *top,
        _path(record(receipt.get("output")).get("path")),
        _path(record(manifest.get("index")).get("path")),
    ]
    for raw in rows:
        row = record(raw)
        candidate = _path(record(row.get("candidate")).get("path"))
        producer = _path(record(row.get("producer")).get("path"))
        paths.extend(
            (
                candidate,
                producer,
                _path(_json(producer).get("outputDirectory")) / "candidate.mp4",
            )
        )
    paths.extend(_path(path) for path in record(receipt.get("probeEvidence")))
    paths.extend(item.path for item in inventory.capture(("post_clean_xml_plan",)))
    return tuple(dict.fromkeys(paths))


def _source(plan: Plan) -> binding.Cleanup:
    config = _json(plan.config)
    original, clean, duration = binding.clean_source(config)
    completion = _json(plan.completion)
    if completion.get("status") != "complete" or record(completion.get("outputs")).get(
        str(plan.source)
    ) != file_digest(plan.source):
        raise ReplayError("Clean XML producer checkpoint does not prove this actual source")
    binding.validate_binding(
        plan.binding,
        consumer_inputs=(plan.config, plan.binding, clean),
        consumer_command=("python", "-m", "replay.fcp_transformed_source", str(plan.config)),
    )
    transformed.verify(plan.config, plan.source)
    return binding.Cleanup(plan.config, original, clean, duration)


def _content(plan: Plan, item: Window) -> bytes:
    if not 0 < plan.source.stat().st_size <= MAX_XML_BYTES:
        raise ReplayError("Clean XML exceeds bounded preview input")
    root = window(ET.parse(plan.source).getroot(), item.start, item.end)
    content: object = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if not isinstance(content, bytes):
        raise ReplayError("Expected encoded preview XML")
    return content


def _validate_window(item: Window, cleanup: binding.Cleanup) -> None:
    binding.validate_binding(
        item.binding,
        consumer_inputs=(item.binding, item.output, cleanup.clean),
        consumer_command=("native-preview", str(item.output)),
    )


def execute(request: Path, *, sha256: str, verify: bool = False) -> Path:
    """Generate or independently recheck exact windows; never dispatch native apps."""
    plan = load(request, sha256=sha256)
    if not verify and any(path.exists() for path in plan.outputs):
        raise ReplayError("Preview outputs already exist; verify retained artifacts")
    cache = InputHashCache(_inputs(plan), allow_uncached_overflow=True)
    with input_hash_context(cache):
        cache.prime()
        if file_digest(request) != sha256:
            raise ReplayError("Post-clean request changed during validation")
        cleanup = _source(plan)
        for item in plan.windows:
            content = _content(plan, item)
            if verify:
                if (
                    item.output.stat().st_size != len(content)
                    or item.output.read_bytes() != content
                ):
                    raise ReplayError("Preview XML differs from exact clean source window")
            else:
                with item.output.open("xb") as handle:
                    handle.write(content)
                binding.create_binding(
                    cleanup, consumer=item.output, kind="fcpxml", output=item.binding
                )
            _validate_window(item, cleanup)
        inputs = cache.verify()
        evidence: dict[str, object] = {
            "version": 1,
            "kind": "verified-clean-caption-preview-plan",
            "request": _reference(request),
            "inputs": inputs,
            "sourceXML": _reference(plan.source),
            "sourceLineage": _reference(plan.source.with_suffix(".source-lineage.json")),
            "windows": [
                {
                    "start": str(item.start),
                    "end": str(item.end),
                    "xml": _reference(item.output),
                    "sourceCaptionBinding": _reference(item.binding),
                }
                for item in plan.windows
            ],
            "exactWindowXML": True,
            "candidateOnly": True,
            "nativeRenderVerified": False,
            "qualityApproved": False,
        }
        if verify:
            if _json(plan.receipt) != evidence:
                raise ReplayError("Preview plan evidence changed")
        else:
            _save(plan.receipt, evidence)
    return plan.receipt


def prepare(request: Path, *, sha256: str, barrier_request: Path, output: Path) -> Path:
    """Delay genuine future input checks until a normal producer barrier completes."""
    plan = load(request, sha256=sha256)
    _local(str(output))
    if not output.is_dir() or tuple(output.iterdir()):
        raise ReplayError("Require existing empty post-clean metadata root")
    release = barrier.load(barrier_request, file_digest(barrier_request))
    code = tuple(item.path for item in inventory.capture(("post_clean_xml_plan",)))
    python = str(ROOT / ".venv/bin/python")
    arguments = [str(request), "--sha256", sha256]
    child = output / "child-workflow.json"
    _save(
        child,
        {
            "version": 1,
            "stages": [
                {
                    "name": "prepare-preview-xml",
                    "inputs": [
                        str(path)
                        for path in (
                            request,
                            release.output,
                            plan.config,
                            plan.binding,
                            plan.source,
                            plan.source.with_suffix(".source-lineage.json"),
                            plan.completion,
                            *code,
                        )
                    ],
                    "outputs": [str(path) for path in plan.outputs],
                    "command": [python, "-m", "replay.post_clean_xml_plan", "run", *arguments],
                    "verify_command": [
                        python,
                        "-m",
                        "replay.post_clean_xml_plan",
                        "verify",
                        *arguments,
                    ],
                    "timeout_seconds": 7200,
                }
            ],
        },
    )
    child_state = output / "child-state"
    child_command = [python, "-m", "replay.workflow", str(child), "--state", str(child_state)]
    barrier_arguments = [str(barrier_request), "--sha256", file_digest(barrier_request)]
    barrier_inputs = (
        barrier_request,
        release.workflow.path,
        release.capture.path,
        *(item.pin.path for item in release.code),
        *(item.pin.path for item in release.nested),
        *code,
    )
    _save(
        output / "workflow.json",
        {
            "version": 1,
            "stages": [
                {
                    "name": "wait-clean-producer",
                    "inputs": [str(path) for path in dict.fromkeys(barrier_inputs)],
                    "outputs": [str(release.output)],
                    "command": [
                        python,
                        "-m",
                        "replay.completion_barrier",
                        "wait",
                        *barrier_arguments,
                    ],
                    "verify_command": [
                        python,
                        "-m",
                        "replay.completion_barrier",
                        "verify",
                        *barrier_arguments,
                    ],
                    "timeout_seconds": 30000,
                },
                {
                    "name": "run-preview-consumer",
                    "inputs": [str(path) for path in (child, request, release.output, *code)],
                    "outputs": [
                        str(path)
                        for path in (*plan.outputs, child_state / "prepare-preview-xml.json")
                    ],
                    "command": child_command,
                    "verify_command": child_command,
                    "timeout_seconds": 15000,
                },
            ],
        },
    )
    parsed = workflow._load_workflow(output / "workflow.json")
    workflow._check_external_inputs(parsed.stages, parsed.paths)
    return output / "workflow.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify", "prepare"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--barrier-request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.barrier_request is None or args.output is None:
            parser.error("prepare requires --barrier-request and --output")
        print(
            prepare(
                args.request,
                sha256=args.sha256,
                barrier_request=args.barrier_request,
                output=args.output,
            )
        )
    else:
        print(execute(args.request, sha256=args.sha256, verify=args.mode == "verify"))


if __name__ == "__main__":
    main()
