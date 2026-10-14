import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import (
    delivery_render,
    mechanical_join,
    mechanical_lane,
    native_edit_parts,
    native_invocation,
    native_join,
    stages,
    workflow,
)
from replay import mechanical_plan_provenance as provenance
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage
from replay.workflow import parse_stages


@dataclass(frozen=True)
class Facts:
    root: Path
    helper: Path
    tool: Path
    media: Path
    calls: list[tuple[str, ...]]


@pytest.fixture
def facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Facts:
    helper, tool, media = tmp_path / "helper.py", tmp_path / "ffprobe", tmp_path / "clean.mov"
    helper.write_bytes(b"# synthetic helper\n")
    tool.write_bytes(b"synthetic executable; never executed")
    tool.chmod(0o700)
    media.write_bytes(b"synthetic already-verified media")
    calls: list[tuple[str, ...]] = []

    def run(
        argv: Sequence[str], *, check: bool, capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, "3.000100\n", "")

    monkeypatch.setattr(provenance, "code_paths", lambda: (helper,))
    monkeypatch.setattr(provenance, "probe_tool", lambda: tool)
    monkeypatch.setattr(subprocess, "run", run)
    return Facts(tmp_path, helper, tool, media, calls)


def bound(facts: Facts) -> tuple[provenance.Generation, Stage]:
    generation = provenance.Generation()
    assert generation.probes.limit(facts.media) == 3
    request = facts.root / "request.json"
    clip: dict[str, object] = {"sourcePath": str(facts.media)}
    generation.record_clip(clip)
    request.write_text(json.dumps({"recipe": {"clips": [clip]}}))
    row: dict[str, object] = {
        "name": "partition",
        "command": [
            sys.executable,
            "-m",
            "replay.native_edit_parts",
            str(request),
            str(facts.root / "parts"),
            "--sha256",
            provenance.pin(request).sha256,
        ],
        "inputs": [str(request), str(facts.media)],
        "outputs": [str(facts.root / "parts/manifest.json")],
        "timeout_seconds": 60,
    }
    generation.bind([row])
    [stage] = parse_stages({"version": 1, "stages": [row]})
    return generation, stage


def altered(stage: Stage, document: dict[str, object]) -> Stage:
    request = stage.inputs[0]
    previous = provenance.receipt_path(request, stage.inputs)
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()
    path = request.with_name(f"request.mechanical-plan-{hashlib.sha256(raw).hexdigest()}.json")
    path.write_bytes(raw)
    original, _ = provenance._consumer(Path(stage.verify_command[3]), stage.verify_command[5])
    changed = replace(
        original, inputs=tuple(path if item == previous else item for item in original.inputs)
    )
    row = provenance._stage_document(changed)
    provenance.bind_verifier(row)
    return parse_stages({"version": 1, "stages": [row]})[0]


def test_generation_reuses_one_actual_probe_and_binds_all_original_inputs(facts: Facts) -> None:
    generation, stage = bound(facts)
    assert generation.probes.limit(facts.media) == 3
    assert facts.calls == [(str(facts.tool), *provenance.PROBE_ARGUMENTS, str(facts.media))]
    receipt = provenance.receipt_path(stage.inputs[0], stage.inputs)
    document = provenance.read(receipt)
    assert stage.inputs[:2] == (facts.root / "request.json", facts.media)
    assert set(stage.inputs[2:]) == {
        receipt,
        facts.helper,
        facts.tool,
        Path(sys.executable).resolve(),
        Path(stage.verify_command[3]),
    }
    assert document["sampleContentVerified"] is False
    assert document["frameContentVerified"] is False
    assert document["probes"] == [generation.probes.observations[facts.media].json()]
    provenance.validate_stages((stage,))


@pytest.mark.parametrize("changed", ["source", "tool", "helper", "request"])
def test_changed_generation_input_refuses_consumer(facts: Facts, changed: str) -> None:
    _, stage = bound(facts)
    paths = {
        "source": facts.media,
        "tool": facts.tool,
        "helper": facts.helper,
        "request": stage.inputs[0],
    }
    paths[changed].write_bytes(b"changed immutable input")
    with pytest.raises(ReplayError):
        provenance.validate_stages((stage,))


@pytest.mark.parametrize("changed", ["source", "tool"])
def test_identity_change_with_same_bytes_size_and_mtime_invalidates_cache(
    facts: Facts, changed: str
) -> None:
    generation, _ = bound(facts)
    path = facts.media if changed == "source" else facts.tool
    before = path.stat()
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(before.st_mode)
    replacement.replace(path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ReplayError, match="changed"):
        generation.probes.limit(facts.media)
    assert len(facts.calls) == 1


def test_changed_source_during_probe_leaves_no_cached_observation(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = provenance.Probes()

    def run(
        argv: Sequence[str], *, check: bool, capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        facts.media.write_bytes(b"changed during observation")
        return subprocess.CompletedProcess(argv, 0, "3", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ReplayError, match="during probe"):
        probes.limit(facts.media)
    assert probes.observations == {}


def test_changed_helper_at_generation_exit_refuses(facts: Facts) -> None:
    generation, _ = bound(facts)
    facts.helper.write_bytes(b"different helper")
    with pytest.raises(ReplayError, match="helper or Python changed"):
        generation.check()


def test_changed_source_during_verification_refuses_at_exit(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, stage = bound(facts)
    original = provenance.parse_result

    def parse(stdout: str, stderr: str, returncode: int) -> Fraction:
        result = original(stdout, stderr, returncode)
        facts.media.write_bytes(b"changed after initial identity acceptance")
        return result

    monkeypatch.setattr(provenance, "parse_result", parse)
    with pytest.raises(ReplayError, match="verifier helper or Python changed"):
        provenance.validate_stages((stage,))


def test_request_changed_after_serialization_refuses(facts: Facts) -> None:
    generation = provenance.Generation()
    generation.probes.limit(facts.media)
    generation.record_clip({"sourcePath": str(facts.media), "selection": {"durationSeconds": 3}})
    request = facts.root / "request.json"
    request.write_text(
        json.dumps(
            {
                "recipe": {
                    "clips": [{"sourcePath": str(facts.media), "selection": {"durationSeconds": 2}}]
                }
            }
        )
    )
    with pytest.raises(ReplayError, match="actual serialized selections"):
        generation.bind([{"inputs": [str(request)]}])
    assert not list(facts.root.glob("*.mechanical-plan-*.json"))


def test_generation_requires_every_serialized_clip_to_be_bound(facts: Facts) -> None:
    generation = provenance.Generation()
    generation.record_clip({"sourcePath": str(facts.media)})
    with pytest.raises(ReplayError, match="unbound serialized selections"):
        generation.finish()


def test_probe_count_budget_refuses_new_observation(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = provenance.Probes()
    monkeypatch.setattr(provenance, "MAX_PROBES", 0)
    with pytest.raises(ReplayError, match="count exceeds"):
        probes.limit(facts.media)
    assert facts.calls == []


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode"),
    [
        ("3", "", 1),
        ("3", "error", 0),
        ("3", "", True),
        ("", "", 0),
        ("N/A", "", 0),
        ("1/0", "", 0),
        ("3\n4", "", 0),
        ("0", "", 0),
        ("-1", "", 0),
        ("1" * (provenance.MAX_PROBE_BYTES + 1), "", 0),
    ],
)
def test_failed_or_ambiguous_probe_result_refuses(
    stdout: str, stderr: str, returncode: int
) -> None:
    with pytest.raises(ReplayError):
        provenance.parse_result(stdout, stderr, returncode)


def test_probe_nonzero_does_not_make_receipt(facts: Facts, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "3", "")
    )
    generation = provenance.Generation()
    with pytest.raises(ReplayError, match="probe failed"):
        generation.probes.limit(facts.media)
    assert not list(facts.root.glob("*.mechanical-plan-*.json"))


def test_content_addressed_receipt_tamper_refuses(facts: Facts) -> None:
    _, stage = bound(facts)
    receipt = provenance.receipt_path(stage.inputs[0], stage.inputs)
    receipt.write_text('{"version":1}')
    with pytest.raises(ReplayError, match="receipt changed"):
        provenance.validate_stages((stage,))


@pytest.mark.parametrize(
    "key",
    [
        "version",
        "request",
        "code",
        "python",
        "sampleContentVerified",
        "frameContentVerified",
        "extra",
    ],
)
def test_malformed_or_mismatched_receipt_fields_refuse(facts: Facts, key: str) -> None:
    _, stage = bound(facts)
    document = provenance.read(provenance.receipt_path(stage.inputs[0], stage.inputs))
    document[key] = True
    changed = altered(stage, document)
    with pytest.raises(ReplayError, match="request/helper provenance differs"):
        provenance.validate_stages((changed,))


@pytest.mark.parametrize(
    "key", ["source", "identity", "argv", "stdout", "stderr", "returncode", "limitSeconds", "extra"]
)
def test_malformed_duration_observation_refuses(facts: Facts, key: str) -> None:
    _, stage = bound(facts)
    document = provenance.read(provenance.receipt_path(stage.inputs[0], stage.inputs))
    probes = document["probes"]
    assert isinstance(probes, list)
    observed = record(probes[0])
    observed[key] = False
    changed = altered(stage, document)
    with pytest.raises(ReplayError):
        provenance.validate_stages((changed,))


@pytest.mark.parametrize("probes", [None, []])
def test_incomplete_observation_list_refuses(facts: Facts, probes: object) -> None:
    _, stage = bound(facts)
    document = provenance.read(provenance.receipt_path(stage.inputs[0], stage.inputs))
    document["probes"] = probes
    with pytest.raises(ReplayError, match="observations are incomplete"):
        provenance.validate_stages((altered(stage, document),))


def test_selected_probe_change_refuses_old_tool(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, stage = bound(facts)
    other = facts.root / "other-tool"
    other.write_bytes(facts.tool.read_bytes())
    other.chmod(0o700)
    monkeypatch.setattr(provenance, "probe_tool", lambda: other)
    with pytest.raises(ReplayError, match="selected probe executable changed"):
        provenance.validate_stages((stage,))


def test_nonexecutable_tool_refuses(facts: Facts) -> None:
    _, stage = bound(facts)
    facts.tool.chmod(0o600)
    with pytest.raises(ReplayError, match="probe executable provenance differs"):
        provenance.validate_stages((stage,))


def test_missing_declared_helper_refuses(facts: Facts) -> None:
    _, stage = bound(facts)
    original, _ = provenance._consumer(Path(stage.verify_command[3]), stage.verify_command[5])
    incomplete = replace(
        original, inputs=tuple(path for path in original.inputs if path != facts.helper)
    )
    row = provenance._stage_document(incomplete)
    provenance.bind_verifier(row)
    incomplete = parse_stages({"version": 1, "stages": [row]})[0]
    with pytest.raises(ReplayError, match="omits required provenance"):
        provenance.validate_stages((incomplete,))


def test_original_request_must_be_used_by_command(facts: Facts) -> None:
    _, stage = bound(facts)
    changed = replace(stage, command=(*stage.command[:3], "different-request", *stage.command[4:]))
    with pytest.raises(ReplayError, match="sealed consumer contract"):
        provenance.validate_stages((changed,))


def test_unbound_legacy_consumer_refuses_before_effect(facts: Facts) -> None:
    request = facts.root / "legacy.json"
    request.write_text('{"recipe":{"clips":[]}}')
    stage = Stage(
        name="legacy",
        command=("native", str(request)),
        inputs=(request,),
        outputs=(facts.root / "output",),
        timeout_seconds=1,
    )
    with pytest.raises(ReplayError, match="legacy mechanical"):
        provenance.validate_stages((stage,))


def test_duplicate_receipt_candidates_are_ambiguous(facts: Facts) -> None:
    _, stage = bound(facts)
    other = facts.root / ("request.mechanical-plan-" + "0" * 64 + ".json")
    other.write_text("{}")
    with pytest.raises(ReplayError, match="ambiguous"):
        provenance.receipt_path(stage.inputs[0])


def test_duplicate_json_fields_refuse(facts: Facts) -> None:
    path = facts.root / "ambiguous.json"
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ReplayError, match="Ambiguous duplicate"):
        provenance.read(path)


@pytest.mark.parametrize("inputs", [None, [], [1], "request"])
def test_invalid_consumer_inputs_refuse(facts: Facts, inputs: object) -> None:
    with pytest.raises(ReplayError):
        provenance.Generation().bind([{"inputs": inputs}])
    with pytest.raises(ReplayError):
        provenance.bind_render([{"inputs": inputs}])


def test_request_without_actual_observation_refuses(facts: Facts) -> None:
    request = facts.root / "request.json"
    request.write_text(json.dumps({"recipe": {"clips": [{"sourcePath": str(facts.media)}]}}))
    generation = provenance.Generation()
    generation.record_clip({"sourcePath": str(facts.media)})
    with pytest.raises(ReplayError, match="lacks its actual duration observation"):
        generation.bind([{"inputs": [str(request)]}])


@pytest.mark.parametrize("clips", [None, [], [{"sourcePath": 3}]])
def test_malformed_request_clips_refuse(facts: Facts, clips: object) -> None:
    generation = provenance.Generation()
    generation.probes.limit(facts.media)
    request = facts.root / "request.json"
    request.write_text(json.dumps({"recipe": {"clips": clips}}))
    with pytest.raises(ReplayError):
        generation.bind([{"inputs": [str(request)]}])


def test_receipt_and_input_budgets_refuse(facts: Facts, monkeypatch: pytest.MonkeyPatch) -> None:
    _, stage = bound(facts)
    monkeypatch.setattr(provenance, "MAX_INPUTS", 1)
    with pytest.raises(ReplayError, match="stage input budget"):
        provenance._extend({"inputs": [str(stage.inputs[0])]}, (facts.helper,))
    monkeypatch.setattr(provenance, "MAX_BYTES", 1)
    with pytest.raises(ReplayError, match="metadata exceeds"):
        provenance.bounded_bytes(facts.helper)


def test_render_consumer_inherits_original_request_and_partition_provenance(facts: Facts) -> None:
    _, stage = bound(facts)
    request = facts.root / "parts/part-0000/request.json"
    request.parent.mkdir(parents=True)
    request.write_bytes(stage.inputs[0].read_bytes())
    manifest = request.parent.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sourceRequest": {
                    "path": str(stage.inputs[0]),
                    "sha256": provenance.pin(stage.inputs[0]).sha256,
                },
                "parts": [{"path": str(request), "sha256": provenance.pin(request).sha256}],
            }
        )
    )
    row: dict[str, object] = {
        "name": "render",
        "command": list(native_invocation.command(facts.tool, request, facts.root)),
        "verify_command": list(
            native_invocation.command(facts.tool, request, facts.root, verify=True)
        ),
        "inputs": [str(request), str(facts.tool), str(facts.media)],
        "outputs": list(map(str, native_invocation.paths(request))),
        "timeout_seconds": 60,
    }
    provenance.bind_render([row])
    [render] = parse_stages({"version": 1, "stages": [row]})
    assert {stage.inputs[0], manifest, provenance.receipt_path(stage.inputs[0])}.issubset(
        render.inputs
    )
    provenance.validate_stages((render,))
    request.write_text("changed part request")
    with pytest.raises(ReplayError, match="does not match"):
        provenance.validate_stages((render,))


def test_render_context_restores_original_after_bound_workflow_error(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, stage = bound(facts)
    request = facts.root / "parts/part-0000/request.json"
    request.parent.mkdir(parents=True)
    request.write_bytes(stage.inputs[0].read_bytes())
    (request.parent.parent / "manifest.json").write_text(
        json.dumps(
            {
                "sourceRequest": {
                    "path": str(stage.inputs[0]),
                    "sha256": provenance.pin(stage.inputs[0]).sha256,
                },
                "parts": [{"path": str(request), "sha256": provenance.pin(request).sha256}],
            }
        )
    )
    captured: list[dict[str, object]] = []

    def workflows(
        directory: Path,
        name: str,
        stages: list[dict[str, object]],
        workers: int,
        *,
        bindings: dict[str, Path] | None = None,
    ) -> list[Path]:
        captured.extend(stages)
        raise RuntimeError("Synthetic workflow write failure")

    monkeypatch.setattr(delivery_render, "_workflows", workflows)
    row: dict[str, object] = {
        "name": "render",
        "command": list(native_invocation.command(facts.tool, request, facts.root)),
        "verify_command": list(
            native_invocation.command(facts.tool, request, facts.root, verify=True)
        ),
        "inputs": [str(request), str(facts.tool), str(facts.media)],
        "outputs": list(map(str, native_invocation.paths(request))),
        "timeout_seconds": 60,
    }
    with (
        pytest.raises(RuntimeError, match="Synthetic workflow write failure"),
        mechanical_lane.render_fixes(),
    ):
        delivery_render._workflows(facts.root, "render", [row], 1)
    assert delivery_render._workflows is workflows
    inputs = captured[0]["inputs"]
    assert isinstance(inputs, list)
    assert str(provenance.receipt_path(stage.inputs[0])) in inputs


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"sourceRequest": {"path": "missing"}},
        {"sourceRequest": {"path": "original", "sha256": "wrong"}},
    ],
)
def test_invalid_original_partition_provenance_refuses(
    facts: Facts, document: dict[str, object]
) -> None:
    request = facts.root / "parts/part-0000/request.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}")
    (request.parent.parent / "manifest.json").write_text(json.dumps(document))
    with pytest.raises((ReplayError, FileNotFoundError)):
        provenance.bind_render([{"inputs": [str(request)]}])


def test_identity_rejects_links_empty_files_and_relative_paths(facts: Facts) -> None:
    link = facts.root / "linked.mov"
    link.symlink_to(facts.media)
    empty = facts.root / "empty"
    empty.touch()
    with pytest.raises(ReplayError):
        provenance.identity(link)
    with pytest.raises(ReplayError):
        provenance.identity(empty)
    with pytest.raises(ReplayError):
        provenance.identity(Path("relative.mov"))
    with pytest.raises(ReplayError):
        provenance.identity(facts.root)


def test_helper_closure_follows_lazy_and_relative_imports_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "entry.py").write_text(
        "from replay import first\nfrom .second import thing\nimport replay.third\n"
        "def lazy():\n    from replay.fourth import item\n"
    )
    (tmp_path / "first.py").write_text("import json\n")
    (tmp_path / "second.py").write_text("from . import first\n")
    (tmp_path / "third.py").write_text("from replay.first import item\n")
    (tmp_path / "fourth.py").write_text("import os\n")
    monkeypatch.setattr(provenance, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(provenance, "SEEDS", ("entry",))
    assert provenance.code_paths() == (
        tmp_path / "entry.py",
        tmp_path / "first.py",
        tmp_path / "fourth.py",
        tmp_path / "second.py",
        tmp_path / "third.py",
    )


@pytest.mark.parametrize("source", ["from .. import escape\n", "from replay.sub.module import x\n"])
def test_helper_closure_rejects_escaping_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    (tmp_path / "entry.py").write_text(source)
    monkeypatch.setattr(provenance, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(provenance, "SEEDS", ("entry",))
    with pytest.raises(ReplayError):
        provenance.code_paths()


def test_actual_helper_closure_binds_current_lane_and_pipeline() -> None:
    paths = provenance.code_paths()
    assert {
        provenance.SOURCE_ROOT / "mechanical_lane.py",
        provenance.SOURCE_ROOT / "delivery_pipeline.py",
        provenance.SOURCE_ROOT / "mechanical_plan_provenance.py",
    }.issubset(paths)
    assert len(paths) < 192


def test_probe_tool_requires_available_executable(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(ReplayError, match="unavailable"):
        provenance.probe_tool()
    monkeypatch.setattr(shutil, "which", lambda _: str(facts.helper))
    with pytest.raises(ReplayError, match="not executable"):
        provenance.probe_tool()


def partitioned(facts: Facts) -> Stage:
    request = facts.root / "request.json"
    clip: dict[str, object] = {
        "sourcePath": str(facts.media),
        "selection": {"startSeconds": 0, "durationSeconds": 1, "rate": 1},
    }
    request.write_text(
        json.dumps(
            {
                "outputDirectory": str(facts.root / "movies"),
                "outputName": "candidate.mp4",
                "recipe": {
                    "video": {
                        "width": 64,
                        "height": 96,
                        "frameRate": 30,
                        "encoding": {"codec": "hevc"},
                    },
                    "clips": [clip],
                },
            }
        )
    )
    generation = provenance.Generation()
    generation.probes.limit(facts.media)
    generation.record_clip(clip)
    output = facts.root / "parts"
    row: dict[str, object] = {
        "name": "partition",
        "command": [
            sys.executable,
            "-m",
            "replay.native_edit_parts",
            str(request),
            str(output),
            "--sha256",
            provenance.pin(request).sha256,
        ],
        "inputs": [str(request), str(facts.media)],
        "outputs": [str(output / "manifest.json")],
        "timeout_seconds": 60,
    }
    generation.bind([row])
    generation.finish()
    return parse_stages({"version": 1, "stages": [row]})[0]


def original_partition(stage: Stage) -> None:
    native_edit_parts.export(stage.inputs[0], stage.outputs[0].parent, sha256=stage.command[-1])


def test_partition_readonly_verifier_uses_original_algorithm_without_export(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = partitioned(facts)
    original_partition(stage)
    before = {path: provenance.pin(path) for path in stage.outputs[0].parent.rglob("*.json")}
    monkeypatch.setattr(
        native_edit_parts,
        "export",
        lambda *_args, **_kwargs: pytest.fail("Read-only domain must never export"),
    )
    assert provenance.main(stage.verify_command[3:]) == 0
    assert before == {path: provenance.pin(path) for path in before}


@pytest.mark.parametrize("changed", ["manifest", "parts", "part", "join", "join-binding"])
def test_partition_domain_rejects_changed_generated_metadata(facts: Facts, changed: str) -> None:
    stage = partitioned(facts)
    original_partition(stage)
    manifest = stage.outputs[0]
    data = provenance.read(manifest)
    if changed == "manifest":
        data["renderVerified"] = True
    elif changed == "parts":
        data["parts"] = []
    elif changed == "part":
        (manifest.parent / "part-0000/request.json").write_text("{}")
    elif changed == "join":
        (manifest.parent / "join-plan.json").write_text('[{"durationFrames":29}]')
    else:
        data["joinPlan"] = {"path": str(manifest.parent / "join-plan.json"), "sha256": "0" * 64}
    manifest.write_text(json.dumps(data))
    assert provenance.main(stage.verify_command[3:]) == 1


def rendered(facts: Facts) -> Stage:
    partition = partitioned(facts)
    original_partition(partition)
    request = partition.outputs[0].parent / "part-0000/request.json"
    row: dict[str, object] = {
        "name": "render",
        "command": list(native_invocation.command(facts.tool, request, facts.root)),
        "verify_command": list(
            native_invocation.command(facts.tool, request, facts.root, verify=True)
        ),
        "inputs": [str(request), str(facts.tool), str(facts.media)],
        "outputs": list(map(str, native_invocation.paths(request))),
        "timeout_seconds": 60,
    }
    provenance.bind_render([row])
    return parse_stages({"version": 1, "stages": [row]})[0]


@pytest.mark.parametrize("failure", ["none", "nonzero", "source-change"])
def test_render_requires_exact_original_readonly_verifier_and_end_identity(
    facts: Facts, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    stage = rendered(facts)
    original, _ = provenance._consumer(Path(stage.verify_command[3]), stage.verify_command[5])
    calls: list[tuple[str, ...]] = []

    def run(
        argv: Sequence[str], *, stdin: int, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        assert stdin == subprocess.DEVNULL and check is False and timeout == 60
        calls.append(tuple(argv))
        if failure == "source-change":
            before = facts.media.stat()
            facts.media.write_bytes(facts.media.read_bytes())
            os.utime(facts.media, ns=(before.st_atime_ns, before.st_mtime_ns))
        return subprocess.CompletedProcess(argv, 1 if failure == "nonzero" else 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert provenance.main(stage.verify_command[3:]) == (0 if failure == "none" else 1)
    assert calls == [original.verify_command]


@pytest.mark.parametrize("changed", ["producer", "verifier", "python", "module", "outputs"])
def test_consumer_contract_rejects_arbitrary_or_altered_domain_arguments(
    facts: Facts, changed: str
) -> None:
    stage = rendered(facts)
    original, _ = provenance._consumer(Path(stage.verify_command[3]), stage.verify_command[5])
    row = provenance._stage_document(original)
    if changed == "producer":
        row["command"] = [*original.command[:3], "verify", *original.command[4:]]
    elif changed == "verifier":
        row["verify_command"] = list(original.command)
    elif changed == "python":
        row["command"] = [str(facts.tool), *original.command[1:]]
    elif changed == "module":
        row["command"] = [
            original.command[0],
            "-m",
            "replay.delivery_render",
            *original.command[3:],
        ]
    else:
        row["outputs"] = [str(facts.root / "other-output.json")]
    with pytest.raises(ReplayError):
        provenance.bind_verifier(row)


@pytest.mark.parametrize("changed", ["receipt", "swapped", "helper", "probe", "source"])
def test_raw_workflow_cannot_accept_invalid_provenance_or_resume_changed_source(
    facts: Facts, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    stage = partitioned(facts)
    if changed in {"receipt", "swapped"}:
        receipt = provenance.receipt_path(stage.inputs[0], stage.inputs)
        document = provenance.read(receipt)
        if changed == "receipt":
            document["sampleContentVerified"] = True
        else:
            document["request"] = {"path": str(facts.root / "other-request.json")}
        stage = altered(stage, document)
    elif changed in {"helper", "probe"}:
        original, _ = provenance._consumer(Path(stage.verify_command[3]), stage.verify_command[5])
        omitted = facts.helper if changed == "helper" else facts.tool
        row = provenance._stage_document(
            replace(original, inputs=tuple(path for path in original.inputs if path != omitted))
        )
        provenance.bind_verifier(row)
        stage = parse_stages({"version": 1, "stages": [row]})[0]
    plan = facts.root / "workflow.json"
    plan.write_text(json.dumps({"version": 1, "stages": [provenance._stage_document(stage)]}))
    state = facts.root / "state"
    commands: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        commands.append(command)
        log.write_text("synthetic in-process CLI boundary\n")
        if command[2] == provenance.VERIFY_MODULE:
            if provenance.main(command[3:]) != 0:
                raise stages.StageError("Required read-only verifier refused")
        else:
            assert command == stage.command
            original_partition(stage)

    def run_stage(current: Stage, *, state_directory: Path) -> str:
        return stages.run_stage(current, state_directory=state_directory, execute=execute)

    monkeypatch.setattr(workflow, "run_stage", run_stage)
    if changed == "source":
        workflow.run_plan(plan, state_directory=state)
        assert json.loads((state / "partition.json").read_text())["status"] == "complete"
        before = facts.media.stat()
        facts.media.write_bytes(facts.media.read_bytes())
        os.utime(facts.media, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(stages.StageError, match="read-only verifier refused"):
        workflow.run_plan(plan, state_directory=state)
    assert commands[-1] == stage.verify_command
    assert commands.count(stage.command) == 1
    if changed != "source":
        assert json.loads((state / "partition.json").read_text())["status"] == "running"


@pytest.mark.parametrize("changed", ["version", "field", "digest", "path"])
def test_cli_refuses_unbound_changed_consumer_contract(facts: Facts, changed: str) -> None:
    stage = partitioned(facts)
    path, digest = Path(stage.verify_command[3]), stage.verify_command[5]
    data = provenance.read(path)
    if changed in {"version", "field"}:
        data["version" if changed == "version" else "extra"] = True
        raw = (json.dumps(data, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = path.with_name(f"request.mechanical-consumer-{digest}.json")
        path.write_bytes(raw)
    elif changed == "path":
        other = path.with_name("other-contract.json")
        other.write_bytes(path.read_bytes())
        path = other
    else:
        digest = "0" * 64
    assert provenance.main([str(path), "--sha256", digest]) == 1


def joined(facts: Facts, monkeypatch: pytest.MonkeyPatch, *, manifest: bool) -> Stage:
    ffmpeg = facts.root / "ffmpeg"
    ffmpeg.write_bytes(b"synthetic converter; never executed")
    ffmpeg.chmod(0o700)
    monkeypatch.setattr(
        provenance, "executable", lambda name: facts.tool if name == "ffprobe" else ffmpeg
    )
    second = facts.root / "second.mp4"
    second.write_bytes(b"second synthetic native output")
    sources = (facts.media, second)
    plan = facts.root / ("join-manifest.json" if manifest else "join-plan.json")
    if manifest:
        mechanical_join.write_manifest(
            plan, [mechanical_join.pin_part(path, 30) for path in sources]
        )
    else:
        plan.write_text(
            json.dumps([{"sourcePath": str(path), "durationFrames": 30} for path in sources])
        )
    output = facts.root / "joined.mp4"
    output.write_bytes(b"synthetic joined output; no media decoder")
    row: dict[str, object] = {
        "name": "join",
        "command": [
            sys.executable,
            "-m",
            "replay.mechanical_join" if manifest else "replay.delivery_render",
            "join-manifest" if manifest else "join-files",
            str(plan),
            "--output",
            str(output),
            "--rate",
            "30",
        ],
        "inputs": [str(plan), *([] if manifest else map(str, sources))],
        "outputs": [str(output)],
        "timeout_seconds": 60,
    }
    provenance.bind_verifier(row)
    return parse_stages({"version": 1, "stages": [row]})[0]


@pytest.mark.parametrize("manifest", [False, True])
def test_join_new_contract_requires_original_packet_domain_and_exact_tool_pins(
    facts: Facts, monkeypatch: pytest.MonkeyPatch, manifest: bool
) -> None:
    stage = joined(facts, monkeypatch, manifest=manifest)
    tools = provenance.validate_join_stage(stage)
    assert tuple(item.path for item in tools) == (
        Path(sys.executable).resolve(),
        facts.tool,
        facts.root / "ffmpeg",
    )
    calls: list[Path] = []

    def probe(path: Path) -> native_join.VideoStream:
        calls.append(path)
        return native_join.VideoStream(
            "hevc", Fraction(1, 15360), 60 if path == stage.outputs[0] else 30
        )

    monkeypatch.setattr(native_join, "probe_video", probe)
    monkeypatch.setattr(
        native_join,
        "join",
        lambda *_args, **_kwargs: pytest.fail("Read-only verifier must never join media"),
    )
    assert provenance.main(stage.verify_command[3:]) == 0
    assert calls == [facts.media, facts.root / "second.mp4", stage.outputs[0]]


@pytest.mark.parametrize("manifest", [False, True])
def test_join_domain_refuses_missing_frame_even_when_output_exists(
    facts: Facts, monkeypatch: pytest.MonkeyPatch, manifest: bool
) -> None:
    stage = joined(facts, monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        native_join,
        "probe_video",
        lambda path: native_join.VideoStream(
            "hevc", Fraction(1, 15360), 59 if path == stage.outputs[0] else 30
        ),
    )
    assert provenance.main(stage.verify_command[3:]) == 1


@pytest.mark.parametrize("changed", ["source", "tool", "helper", "plan", "stage", "unbound"])
def test_join_validator_refuses_changed_identity_pins_or_consumer(
    facts: Facts, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    stage = joined(facts, monkeypatch, manifest=True)
    if changed == "stage":
        stage = replace(stage, inputs=tuple(path for path in stage.inputs if path != facts.tool))
    elif changed == "unbound":
        stage = replace(stage, verify_command=())
    else:
        path = {
            "source": facts.media,
            "tool": facts.tool,
            "helper": facts.helper,
            "plan": stage.inputs[0],
        }[changed]
        before = path.stat()
        path.write_bytes(path.read_bytes())
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ReplayError):
        provenance.validate_join_stage(stage)


def test_manifest_verifier_retains_original_per_part_digest_authority(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = joined(facts, monkeypatch, manifest=True)
    original, _ = provenance._consumer(Path(stage.verify_command[3]), stage.verify_command[5])
    data = provenance.read(stage.inputs[0])
    parts = data["parts"]
    assert isinstance(parts, list)
    record(parts[0])["sha256"] = "0" * 64
    stage.inputs[0].write_text(json.dumps(data))
    row = provenance._stage_document(original)
    provenance.bind_verifier(row)
    current = parse_stages({"version": 1, "stages": [row]})[0]
    monkeypatch.setattr(
        native_join,
        "probe_video",
        lambda _: pytest.fail("Changed part digest must stop before packet probe"),
    )
    assert provenance.main(current.verify_command[3:]) == 1


@pytest.mark.parametrize(
    "plan",
    [
        [],
        [{"sourcePath": "relative", "durationFrames": True}],
        [{"sourcePath": "/part", "durationFrames": 0}],
        [
            {"sourcePath": "/part", "durationFrames": 1},
            {"sourcePath": "/part", "durationFrames": 1},
        ],
    ],
)
def test_classic_join_rejects_empty_ambiguous_or_untyped_plan(
    facts: Facts, monkeypatch: pytest.MonkeyPatch, plan: object
) -> None:
    stage = joined(facts, monkeypatch, manifest=False)
    stage.inputs[0].write_text(json.dumps(plan))
    with pytest.raises(ReplayError):
        provenance._join_parts(stage)


def test_join_generation_restores_fixed_helpers_on_failure_without_rewriting_old_plan(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write, original_stage = delivery_render._write_new, mechanical_join.join_stage
    path = facts.root / "old.json"
    path.write_text("old immutable plan")
    with pytest.raises(FileExistsError), provenance.join_fixes():
        delivery_render._write_new(path, {"new": "value"})
    assert path.read_text() == "old immutable plan"
    assert delivery_render._write_new is original_write
    assert mechanical_join.join_stage is original_stage


@pytest.mark.parametrize(
    "document", [{}, {"version": True, "stages": []}, {"version": 1, "stages": []}]
)
def test_join_writer_rejects_invalid_workflow_before_original_write(
    facts: Facts, document: dict[str, object]
) -> None:
    path = facts.root / "join-workflow.json"
    with pytest.raises(ReplayError), provenance.join_fixes():
        delivery_render._write_new(path, document)
    assert not path.exists()


@pytest.mark.parametrize("parts", [248, 249])
def test_join_selection_accounts_for_final_declared_inputs_at_exact_cap(
    facts: Facts, monkeypatch: pytest.MonkeyPatch, parts: int
) -> None:
    from test_delivery_pipeline import configured

    config = delivery_render.load_config(configured(facts.root))
    ffmpeg = facts.root / "ffmpeg"
    ffmpeg.write_bytes(b"synthetic converter")
    ffmpeg.chmod(0o700)
    monkeypatch.setattr(
        provenance, "executable", lambda name: facts.tool if name == "ffprobe" else ffmpeg
    )
    rendered: list[tuple[Path, int]] = []
    for index in range(parts):
        request = facts.root / f"request-{index}/request.json"
        request.parent.mkdir()
        request.with_name("render.response.json").write_text(
            json.dumps({"frameCount": 30, "outputPath": str(facts.root / f"native-{index}.mp4")})
        )
        rendered.append((request, 30))
    # plan/cuts/chunks + helper + Python + two tools + consumer contract = eight.
    assert provenance.join_needs_manifest(rendered, config) is (parts + 8 > 256)
    assert mechanical_join.needs_manifest(parts, 0) is False


def test_join_selection_deduplicates_existing_known_helper_inputs(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_delivery_pipeline import configured

    config = delivery_render.load_config(configured(facts.root))
    request = facts.root / "request.json"
    request.with_name("render.response.json").write_text(
        json.dumps({"frameCount": 30, "outputPath": str(facts.media)})
    )
    ffmpeg = facts.root / "ffmpeg"
    ffmpeg.write_bytes(b"synthetic converter")
    ffmpeg.chmod(0o700)
    monkeypatch.setattr(
        provenance, "executable", lambda name: facts.tool if name == "ffprobe" else ffmpeg
    )
    monkeypatch.setattr(mechanical_join, "WORKFLOW_ARRAY_LIMIT", 9)
    config = replace(config, cut_evidence=(facts.helper,))
    assert provenance.join_needs_manifest([(request, 30)], config) is False


def test_real_workflow_cli_runs_guarded_original_partition_and_readonly_resume(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    facts.tool.write_text("#!/bin/sh\nprintf '3.000100\\n'\n")
    facts.tool.chmod(0o700)
    monkeypatch.setenv("PATH", f"{facts.root}:{os.environ['PATH']}")
    stage = partitioned(facts)
    plan = facts.root / "workflow.json"
    plan.write_text(json.dumps({"version": 1, "stages": [provenance._stage_document(stage)]}))
    command = [
        sys.executable,
        "-m",
        "replay.workflow",
        str(plan),
        "--state",
        str(facts.root / "state"),
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stderr
    assert "partition: completed" in first.stdout
    before = {path: provenance.pin(path) for path in stage.outputs[0].parent.rglob("*.json")}
    resumed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stderr
    assert "partition: reused" in resumed.stdout
    assert {path: provenance.pin(path) for path in before} == before
    facts.media.write_bytes(facts.media.read_bytes())
    changed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=90)
    assert changed.returncode == 1
    assert "partition: reused" not in changed.stdout
    assert {path: provenance.pin(path) for path in before} == before


def test_numeric_type_substitution_cannot_relabel_source_identity(facts: Facts) -> None:
    _, stage = bound(facts)
    receipt = provenance.read(provenance.receipt_path(stage.inputs[0], stage.inputs))
    probes = receipt["probes"]
    assert isinstance(probes, list)
    observed = record(probes[0])
    source_identity = record(observed["identity"])
    source_identity["bytes"] = float(facts.media.stat().st_size)
    with pytest.raises(ReplayError, match="source/probe observation differs"):
        provenance.validate_stages((altered(stage, receipt),))


def test_partition_numeric_type_change_is_not_original_output(facts: Facts) -> None:
    stage = partitioned(facts)
    original_partition(stage)
    manifest = provenance.read(stage.outputs[0])
    parts = manifest["parts"]
    assert isinstance(parts, list)
    record(parts[0])["frames"] = 30.0
    stage.outputs[0].write_text(json.dumps(manifest))
    assert provenance.main(stage.verify_command[3:]) == 1
