import fcntl
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from replay import stage_recovery, stages, workflow
from replay import whole_technical_batch as batch
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache
from replay.stages import Stage, StageError


@dataclass(frozen=True)
class Example:
    config: Path
    manifest: Path
    technical: Path
    source: Path
    source_state: Path
    state: Path
    master: Path
    first: Path
    second: Path
    third: Path
    root: Path

    @property
    def digest(self) -> str:
        return stages.file_digest(self.manifest)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _stage(name: str, source: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(exist_ok=True)
    return {
        "name": name,
        "command": ["/bin/cp", str(source), str(output)],
        "verify_command": ["/usr/bin/cmp", str(source), str(output)],
        "inputs": [str(source)],
        "outputs": [str(output)],
        "timeout_seconds": 10,
    }


def _reuse_example(example: Example, decoder: Path, media: Path) -> Example:
    original = example.root / "original.json"
    document = json.loads(original.read_text())
    join = document["stages"][2]
    join["command"] = ["/bin/cp", str(media), str(example.master)]
    join["inputs"] = sorted(
        {str(media), str(decoder), *join["inputs"], *map(str, batch.reader_code_paths())}
    )
    join["verify_command"] = [
        str(decoder),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-protocol_whitelist",
        "file",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
        "-use_absolute_path",
        "0",
        "-i",
        str(example.master),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    _json(original, document)
    # Synthetic prior fixture is replaced before producing the new original evidence.
    example.master.unlink()
    (example.source_state / "join.json").unlink()
    stages.run_stage(
        workflow.load_stages(original)[2],
        state_directory=example.source_state,
        execute=stages.execute_command,
    )
    config = json.loads(example.config.read_text())
    config["workflow"] = _reference(original)
    config["admission"]["fullAVReuse"] = {
        "version": 1,
        "kind": "completed-canonical-ffmpeg-full-av",
        "stages": ["join"],
    }
    paths = {Path(ref["path"]) for ref in config["admission"]["inputs"]} | {decoder, media}
    config["admission"]["inputs"] = [_reference(path) for path in sorted(paths)]
    _json(example.config, config)
    manifest, technical = (
        example.root / "reuse-manifest.json",
        example.root / "reuse-technical.json",
    )
    batch.prepare(example.config, manifest=manifest, workflow=technical)
    return replace(example, manifest=manifest, technical=technical)


@pytest.fixture
def reused_full_av(example: Example, monkeypatch: pytest.MonkeyPatch) -> tuple[Example, list[str]]:
    decoder = example.root / "pinned-decoder"
    decoder.write_bytes(b"synthetic pinned decoder")
    calls: list[str] = []
    ordinary = stages.execute_command

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command[0] == str(decoder):
            calls.append("full-av")
            log.write_text("synthetic full AV passed", encoding="utf-8")
        else:
            if command[0] == "/usr/bin/cmp":
                calls.append("other-domain")
            ordinary(command, log, timeout)

    monkeypatch.setattr(stages, "execute_command", execute)
    monkeypatch.setattr(batch, "execute_command", execute)
    return _reuse_example(example, decoder, example.source), calls


@pytest.fixture
def example(tmp_path: Path) -> Example:
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic clock and decoded voice\n")
    clean = tmp_path / "clean.bin"
    published = tmp_path / "published.bin"
    master = tmp_path / "master.bin"
    first, second, third = (
        tmp_path / "technical-results/frames.bin",
        tmp_path / "technical-results/metadata.bin",
        tmp_path / "technical-results/pcm.bin",
    )
    original = tmp_path / "original.json"
    _json(
        original,
        {
            "version": 1,
            "stages": [
                _stage("repair", source, clean),
                _stage("retire", clean, published),
                _stage("join", published, master),
                _stage("frames", master, first),
                _stage("metadata", first, second),
                _stage("pcm", master, third),
            ],
        },
    )
    source_state = tmp_path / "original-state"
    workflow.run_plan(original, state_directory=source_state, targets=("join",))
    common = {
        original,
        source,
        clean,
        published,
        master,
        source_state / "repair.json",
        source_state / "retire.json",
        source_state / "join.json",
        Path("/bin/cp").resolve(),
        Path("/usr/bin/cmp").resolve(),
        *batch.reader_code_paths(),
    }
    config = tmp_path / "config.json"
    state = tmp_path / "technical-state"
    _json(
        config,
        {
            "version": 1,
            "workflow": _reference(original),
            "sourceState": str(source_state),
            "targets": ["frames", "metadata", "pcm"],
            "state": str(state),
            "cwd": str(batch.ROOT),
            "admission": {
                "roles": {
                    "wholeJoin": ["join"],
                    "fullAV": ["join"],
                    "clock": ["join"],
                    "cleanup": ["repair"],
                    "retirement": ["retire"],
                },
                "inputs": [_reference(path) for path in sorted(common)],
            },
        },
    )
    manifest, technical = tmp_path / "manifest.json", tmp_path / "technical.json"
    batch.prepare(config, manifest=manifest, workflow=technical)
    return Example(
        config,
        manifest,
        technical,
        source,
        source_state,
        state,
        master,
        first,
        second,
        third,
        tmp_path,
    )


def test_prepare_preserves_exact_original_stages_without_media_digest(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    ordinary = stages.file_digest

    def digest(path: Path) -> str:
        seen.append(path)
        return ordinary(path)

    monkeypatch.setattr(batch, "file_digest", digest)
    batch.prepare(
        example.config,
        manifest=example.root / "next-manifest.json",
        workflow=example.root / "next-technical.json",
    )
    assert seen == [example.config]
    assert workflow.load_stages(example.technical)[0].command == (
        "/bin/cp",
        str(example.master),
        str(example.first),
    )
    assert not example.state.exists()
    assert not example.first.exists()


def test_run_and_independent_verify_use_fresh_cache_and_preserve_bytes(example: Example) -> None:
    receipt = example.root / "result.json"
    run = batch.run(
        example.manifest, example.digest, receipt=receipt, logs=example.root / "run-logs"
    )
    before = receipt.read_bytes()
    checked = batch.verify(
        example.manifest, example.digest, receipt=receipt, logs=example.root / "verify-logs"
    )
    count = len(json.loads(example.config.read_text())["admission"]["inputs"]) + 3
    assert run.hash_reads == count * 2
    assert checked.hash_reads == count * 2
    assert run.cache_hits > 0
    assert checked.cache_hits > 0
    assert receipt.read_bytes() == before
    assert example.first.read_bytes() == b"synthetic clock and decoded voice\n"
    assert example.second.read_bytes() == b"synthetic clock and decoded voice\n"
    assert example.third.read_bytes() == b"synthetic clock and decoded voice\n"
    data = json.loads(before)
    assert data["qualityApproved"] is False
    assert data["syncApproved"] is False
    assert data["deliveryApproved"] is False
    assert data["runCache"]["childIO"] == {"measured": False, "bytesRead": None}
    assert data["runCache"]["outerIO"] == {"measured": False, "bytesRead": None}


def test_actual_workflow_recovery_keeps_complete_prefix_and_runs_missing_only(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    interrupt = True
    original_execute = stages.execute_command

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command[0] == "/bin/cp":
            calls.append(Path(command[-1]).stem)
            if command[-1] == str(example.second) and interrupt:
                example.second.write_bytes(b"uncertain partial output")
                raise RuntimeError("synthetic interruption")
        original_execute(command, log, timeout)

    def run_stage(
        stage: Stage, *, state_directory: Path
    ) -> Literal["completed", "reused", "adopted", "busy"]:
        return stages.run_stage(stage, state_directory=state_directory, execute=execute)

    monkeypatch.setattr(workflow, "run_stage", run_stage)
    receipt = example.root / "result.json"
    with pytest.raises(RuntimeError, match="interruption"):
        batch.run(
            example.manifest, example.digest, receipt=receipt, logs=example.root / "first-logs"
        )
    prefix = (example.state / "frames.json").read_bytes()
    frame_bytes = example.first.read_bytes()
    assert not receipt.exists()
    assert calls == ["frames", "metadata"]
    with pytest.raises(StageError, match="command failed"):
        batch.run(
            example.manifest, example.digest, receipt=receipt, logs=example.root / "refused-logs"
        )
    assert calls == ["frames", "metadata"]
    with stages.execution_lock(example.state, ".workflow") as acquired:
        assert acquired
        with stages.execution_lock(example.state, "metadata") as stage_free:
            assert stage_free
            preserved = stage_recovery.recover(example.technical, example.state)
    assert len(preserved) == 2
    assert example.first.read_bytes() == frame_bytes
    assert (example.state / "frames.json").read_bytes() == prefix
    interrupt = False
    calls.clear()
    batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "resumed-logs")
    assert calls == ["metadata", "pcm"]
    assert example.first.read_bytes() == frame_bytes
    assert (example.state / "frames.json").read_bytes() == prefix
    assert example.second.read_bytes() == b"synthetic clock and decoded voice\n"
    calls.clear()
    batch.run(
        example.manifest,
        example.digest,
        receipt=example.root / "reused.json",
        logs=example.root / "reused-logs",
    )
    assert calls == []


def test_readonly_verify_never_calls_producer_or_recovery(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = example.root / "result.json"
    batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "run")

    def prohibited(*args: object, **kwargs: object) -> None:
        pytest.fail("readonly verification attempted a producer or recovery")

    monkeypatch.setattr(batch, "run_plan", prohibited)
    monkeypatch.setattr(stages, "run_stage", prohibited)
    monkeypatch.setattr(stage_recovery, "recover", prohibited)
    batch.verify(example.manifest, example.digest, receipt=receipt, logs=example.root / "verify")


@pytest.mark.parametrize("target", ["source", "master", "code", "tool"])
def test_changed_input_pin_refuses_before_producers(
    example: Example, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(example.config.read_text())
    path = {
        "source": example.source,
        "master": example.master,
        "code": batch.ROOT / "src/replay/stages.py",
        "tool": Path("/bin/cp").resolve(),
    }[target]
    altered = [row for row in config["admission"]["inputs"] if row["path"] == str(path)][0]
    altered["sha256"] = "0" * 64
    _json(example.config, config)
    manifest = example.root / "new-manifest.json"
    batch.prepare(example.config, manifest=manifest, workflow=example.root / "new-workflow.json")

    def prohibited(*args: object, **kwargs: object) -> None:
        pytest.fail("changed pin launched production")

    monkeypatch.setattr(batch, "run_plan", prohibited)
    with pytest.raises(ReplayError, match="fingerprint changed"):
        batch.run(
            manifest,
            stages.file_digest(manifest),
            receipt=example.root / "result.json",
            logs=example.root / "logs",
        )
    assert not example.first.exists()


@pytest.mark.parametrize("kind", ["missing", "mutated", "checkpoint"])
def test_readonly_missing_or_changed_completed_artifact_refuses(
    example: Example, kind: str
) -> None:
    receipt = example.root / "result.json"
    batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "run")
    if kind == "missing":
        example.first.unlink()
    elif kind == "mutated":
        example.first.write_bytes(b"wrong clock")
    else:
        (example.state / "frames.json").write_text("{}", encoding="utf-8")
    with pytest.raises((ReplayError, StageError, FileNotFoundError)):
        batch.verify(
            example.manifest, example.digest, receipt=receipt, logs=example.root / "verify"
        )
    assert example.second.read_bytes() == b"synthetic clock and decoded voice\n"


def test_generated_outputs_are_never_in_parent_cache(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary_reads: list[Path] = []
    ordinary = stages._uncached_file_digest

    def digest(path: Path) -> str:
        ordinary_reads.append(path)
        return ordinary(path)

    monkeypatch.setattr(stages, "_uncached_file_digest", digest)
    batch.run(
        example.manifest,
        example.digest,
        receipt=example.root / "result.json",
        logs=example.root / "logs",
    )
    assert example.first in ordinary_reads
    assert example.second in ordinary_reads
    assert example.third in ordinary_reads
    assert example.master not in ordinary_reads


def test_cache_overflow_remains_fresh_crypto(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    def small(paths: Iterable[Path], *, allow_uncached_overflow: bool) -> InputHashCache:
        return InputHashCache(paths, max_files=1, allow_uncached_overflow=allow_uncached_overflow)

    monkeypatch.setattr(batch, "InputHashCache", small)
    result = batch.run(
        example.manifest,
        example.digest,
        receipt=example.root / "result.json",
        logs=example.root / "logs",
    )
    assert (
        result.hash_reads
        > (len(json.loads(example.config.read_text())["admission"]["inputs"]) + 3) * 2
    )


@pytest.mark.parametrize("operation", ["run", "verify"])
def test_changed_cwd_rejects_without_effect(
    example: Example, operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = example.root / "result.json"
    if operation == "verify":
        batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "run")
    monkeypatch.chdir(example.root)
    with pytest.raises(ReplayError, match="working directory changed"):
        {"run": batch.run, "verify": batch.verify}[operation](
            example.manifest, example.digest, receipt=receipt, logs=example.root / "changed"
        )
    assert not (example.root / "changed").exists()


@pytest.mark.parametrize("owner", ["source", "stage", "workflow", "batch"])
def test_foreign_lock_refuses_without_commands(example: Example, owner: str) -> None:
    directory, name = {
        "source": (example.source_state, ".workflow"),
        "stage": (example.source_state, "join"),
        "workflow": (example.state, ".workflow"),
        "batch": (example.state.parent / f".{example.state.name}-batch", "owner"),
    }[owner]
    directory.mkdir(exist_ok=True)
    with (directory / f"{name}.lock").open("wb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReplayError, match="Another process owns"):
            batch.run(
                example.manifest,
                example.digest,
                receipt=example.root / "result.json",
                logs=example.root / "logs",
            )
    assert not example.first.exists()
    assert not (example.root / "logs").exists()


def test_input_mutation_between_targets_is_not_completed(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = workflow.run_plan

    def mutate(plan: Path, *, state_directory: Path, targets: tuple[str, ...]) -> None:
        original(plan, state_directory=state_directory, targets=targets)
        example.source.write_bytes(b"changed after first complete target")

    monkeypatch.setattr(batch, "run_plan", mutate)
    with pytest.raises(ReplayError, match="Input changed"):
        batch.run(
            example.manifest,
            example.digest,
            receipt=example.root / "result.json",
            logs=example.root / "logs",
        )
    assert example.first.read_bytes() == b"synthetic clock and decoded voice\n"
    assert not example.second.exists()
    assert not (example.root / "result.json").exists()


def test_domain_failure_keeps_candidate_and_no_terminal(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(command: tuple[str, ...], log: Path, timeout: int) -> None:
        raise StageError("synthetic domain failed")

    monkeypatch.setattr(batch, "execute_command", failed)
    with pytest.raises(StageError, match="domain failed"):
        batch.run(
            example.manifest,
            example.digest,
            receipt=example.root / "result.json",
            logs=example.root / "logs",
        )
    assert not example.first.exists()
    assert example.master.read_bytes() == b"synthetic clock and decoded voice\n"
    assert not (example.root / "result.json").exists()


def test_timeout_propagates_and_does_not_approve(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(command: tuple[str, ...], log: Path, deadline: int) -> None:
        raise subprocess.TimeoutExpired(command, deadline)

    monkeypatch.setattr(batch, "execute_command", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        batch.run(
            example.manifest,
            example.digest,
            receipt=example.root / "result.json",
            logs=example.root / "logs",
        )
    assert not (example.root / "result.json").exists()


def test_preparation_does_not_overwrite_existing_output(example: Example) -> None:
    before = example.manifest.read_bytes()
    with pytest.raises(ReplayError, match="distinct and absent"):
        batch.prepare(
            example.config, manifest=example.manifest, workflow=example.root / "unused.json"
        )
    assert example.manifest.read_bytes() == before
    assert not (example.root / "unused.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "missing-role",
        "nonancestor-role",
        "missing-input",
        "generated-input",
        "duplicate-input",
        "targets",
        "version",
        "unknown",
    ],
)
def test_invalid_admission_contract_rejects_before_output(example: Example, change: str) -> None:
    data = json.loads(example.config.read_text())
    if change == "missing-role":
        del data["admission"]["roles"]["fullAV"]
    elif change == "nonancestor-role":
        data["admission"]["roles"]["clock"] = ["frames"]
    elif change == "missing-input":
        data["admission"]["inputs"].pop()
    elif change == "generated-input":
        example.first.write_bytes(b"existing unregistered output")
        data["admission"]["inputs"].append(_reference(example.first))
    elif change == "duplicate-input":
        data["admission"]["inputs"].append(data["admission"]["inputs"][0])
    elif change == "targets":
        data["targets"] = ["frames", "pcm"]
    elif change == "version":
        data["version"] = True
    else:
        data["unexpected"] = "field"
    _json(example.config, data)
    with pytest.raises((ReplayError, StageError)):
        batch.prepare(
            example.config,
            manifest=example.root / "bad.json",
            workflow=example.root / "bad-graph.json",
        )
    assert not (example.root / "bad.json").exists()
    assert not (example.root / "bad-graph.json").exists()


def test_tampered_technical_graph_rejected(example: Example) -> None:
    data = json.loads(example.technical.read_text())
    data["stages"][0]["timeout_seconds"] = 11
    _json(example.technical, data)
    manifest = json.loads(example.manifest.read_text())
    manifest["technicalWorkflow"] = _reference(example.technical)
    _json(example.manifest, manifest)
    with pytest.raises(ReplayError, match="original commands"):
        batch.load(example.manifest, example.digest)


@pytest.mark.parametrize("path", ["private/secret", "downloads/media", "../escape"])
def test_forbidden_paths_rejected_before_reader(example: Example, path: str) -> None:
    with pytest.raises(ReplayError, match="scope"):
        batch._path(str(batch.ROOT / path))


def test_symlink_input_and_control_escape_rejected(example: Example) -> None:
    alias = example.root / "linked-input"
    alias.symlink_to(example.source)
    with pytest.raises(ReplayError, match="symlinks"):
        batch._path(str(alias))
    with pytest.raises(ReplayError, match="scope"):
        batch._path("/Volumes/not-a-control-output", work=True)


def test_corrupt_receipt_counter_schema_is_not_accepted(example: Example) -> None:
    receipt = example.root / "result.json"
    batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "run")
    data = json.loads(receipt.read_text())
    data["runCache"]["hashReads"] = True
    _json(receipt, data)
    with pytest.raises(ReplayError, match="nonnegative integers"):
        batch.verify(
            example.manifest, example.digest, receipt=receipt, logs=example.root / "verify"
        )


def test_cli_prepare_run_verify(example: Example, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = example.root / "cli-manifest.json"
    batch.main(
        [
            "prepare",
            str(example.config),
            "--manifest",
            str(manifest),
            "--workflow",
            str(example.root / "cli-workflow.json"),
        ]
    )
    digest = stages.file_digest(manifest)
    batch.main(
        [
            "run",
            str(manifest),
            "--sha256",
            digest,
            "--receipt",
            str(example.root / "cli-receipt.json"),
            "--logs",
            str(example.root / "cli-run"),
        ]
    )
    batch.main(
        [
            "verify",
            str(manifest),
            "--sha256",
            digest,
            "--receipt",
            str(example.root / "cli-receipt.json"),
            "--logs",
            str(example.root / "cli-verify"),
        ]
    )
    assert "parent-cache-only" in capsys.readouterr().out


def test_adopts_only_pinned_verified_original_complete_checkpoints(example: Example) -> None:
    original = Path(json.loads(example.config.read_text())["workflow"]["path"])
    workflow.run_plan(original, state_directory=example.source_state, targets=("pcm", "metadata"))
    config = json.loads(example.config.read_text())
    config["admission"]["inputs"].extend(
        [
            _reference(example.source_state / f"{name}.json")
            for name in ("frames", "metadata", "pcm")
        ]
    )
    _json(example.config, config)
    manifest = example.root / "adopt-manifest.json"
    batch.prepare(example.config, manifest=manifest, workflow=example.root / "adopt-graph.json")
    before = example.first.read_bytes(), example.second.read_bytes(), example.third.read_bytes()
    batch.run(
        manifest,
        stages.file_digest(manifest),
        receipt=example.root / "adopt-receipt.json",
        logs=example.root / "adopt-logs",
    )
    assert before == (
        example.first.read_bytes(),
        example.second.read_bytes(),
        example.third.read_bytes(),
    )
    assert (example.state / "frames.json").read_bytes() == (
        example.source_state / "frames.json"
    ).read_bytes()


def test_readonly_domain_output_mutation_is_rejected(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = example.root / "result.json"
    batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "run")
    original = stages.execute_command

    def mutate(command: tuple[str, ...], log: Path, timeout: int) -> None:
        original(command, log, timeout)
        if command[-1] == str(example.third):
            example.third.write_bytes(b"corrupted readonly verifier effect")

    monkeypatch.setattr(batch, "execute_command", mutate)
    with pytest.raises(ReplayError, match="Input changed"):
        batch.verify(
            example.manifest, example.digest, receipt=receipt, logs=example.root / "verify"
        )


@pytest.mark.parametrize(
    "kind", ["duplicate", "empty", "oversize", "directory", "nonobject", "nonfinite"]
)
def test_bounded_strict_metadata(kind: str, tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    if kind == "directory":
        path.mkdir()
    else:
        payload = {
            "duplicate": b'{"version":1,"version":1}',
            "empty": b"",
            "oversize": b" " * (batch.MAX_METADATA_BYTES + 1),
            "nonobject": b"[]",
            "nonfinite": b'{"number":NaN}',
        }[kind]
        path.write_bytes(payload)
    with pytest.raises(ReplayError, match="unique|bounded regular|object|Nonfinite"):
        batch._metadata(path)


def test_declared_cleanup_binding_cannot_be_bypassed_by_complete_ancestor(example: Example) -> None:
    config = json.loads(example.config.read_text())
    original = Path(config["workflow"]["path"])
    binding = example.root / "invalid-cleanup-binding.json"
    _json(binding, {})
    data = json.loads(original.read_text())
    data["sourceCaptionProduction"] = {"bindings": {"repair": str(binding)}}
    data["stages"][0]["inputs"].append(str(binding))
    _json(original, data)
    alternate_state = example.root / "untrusted-original-state"
    ancestors = workflow.load_stages(original)[:3]
    # Deliberately construct complete ordinary receipts without the workflow's
    # cleanup hook: the batch must reject this incomplete production evidence.
    results = [
        stages.run_stage(replace(stage, adopt_existing=True), state_directory=alternate_state)
        for stage in ancestors
    ]
    assert results == ["adopted", "adopted", "adopted"]
    config["workflow"] = _reference(original)
    config["sourceState"] = str(alternate_state)
    inputs = config["admission"]["inputs"]
    config["admission"]["inputs"] = [
        _reference(alternate_state / Path(ref["path"]).name)
        if Path(ref["path"]).parent == example.source_state
        else _reference(original)
        if Path(ref["path"]) == original
        else ref
        for ref in inputs
    ] + [_reference(binding)]
    _json(example.config, config)
    manifest = example.root / "bound-manifest.json"
    batch.prepare(example.config, manifest=manifest, workflow=example.root / "bound-workflow.json")
    with pytest.raises(ReplayError, match="binding version"):
        batch.run(
            manifest,
            stages.file_digest(manifest),
            receipt=example.root / "bound-result.json",
            logs=example.root / "bound-logs",
        )
    assert not example.first.exists()


@pytest.mark.parametrize("kind", ["logs-output", "logs-state", "receipt-input", "prepare-state"])
def test_controls_cannot_overlap_immutable_or_generated_paths(example: Example, kind: str) -> None:
    with pytest.raises(ReplayError, match="separate"):
        if kind == "prepare-state":
            batch.prepare(
                example.config,
                manifest=example.state / "manifest.json",
                workflow=example.root / "other-graph.json",
            )
        else:
            receipt = example.source if kind == "receipt-input" else example.root / "result.json"
            logs = (
                example.first
                if kind == "logs-output"
                else example.state / "logs"
                if kind == "logs-state"
                else example.root / "logs"
            )
            batch.run(example.manifest, example.digest, receipt=receipt, logs=logs)
    assert not example.first.exists()


def test_foreign_target_lock_rejects_before_the_first_target(example: Example) -> None:
    example.state.mkdir()
    with (example.state / "metadata.lock").open("wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReplayError, match="owns technical workflow"):
            batch.run(
                example.manifest,
                example.digest,
                receipt=example.root / "result.json",
                logs=example.root / "logs",
            )
    assert not example.first.exists()
    assert not (example.root / "logs").exists()


@pytest.mark.parametrize("field", ["scope", "childIO", "fields"])
def test_cache_counters_cannot_claim_unmeasured_io(field: str) -> None:
    counters = batch.Counters(1, 2, 3).json()
    if field == "fields":
        counters["unknown"] = 0
    elif field == "childIO":
        counters[field] = {"measured": True, "bytesRead": 0}
    else:
        counters[field] = "all-IO"
    with pytest.raises(ReplayError, match="fields|claim child"):
        batch._metrics(counters)


def test_reader_code_closure_is_required_and_cannot_escape(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(example.config.read_text())
    config["admission"]["inputs"] = [
        ref
        for ref in config["admission"]["inputs"]
        if not ref["path"].endswith("fcp_transformed_source.py")
    ]
    _json(example.config, config)
    with pytest.raises(ReplayError, match="complete closure"):
        batch.prepare(
            example.config,
            manifest=example.root / "bad-manifest.json",
            workflow=example.root / "bad-graph.json",
        )
    monkeypatch.setattr(batch, "CODE_NAMES", ("../escape.py",))
    with pytest.raises(ReplayError, match="module scope"):
        batch.reader_code_paths()


def test_completed_full_av_is_reused_without_repeating_decode(
    reused_full_av: tuple[Example, list[str]],
) -> None:
    example, calls = reused_full_av
    assert calls == ["full-av"]
    receipt = example.root / "result.json"
    batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "run")
    batch.verify(example.manifest, example.digest, receipt=receipt, logs=example.root / "verify")
    assert calls.count("full-av") == 1
    assert calls.count("other-domain") == 7
    evidence = json.loads(receipt.read_text())["reusedFullAV"]
    assert len(evidence) == 1
    assert evidence[0]["newDecodePerformed"] is False
    assert evidence[0]["stage"] == "join"
    assert evidence[0]["checkpoint"] == _reference(example.source_state / "join.json")
    assert evidence[0]["media"] == _reference(example.master)


@pytest.mark.parametrize("changed", ["media", "config", "tool", "code", "missing", "running"])
def test_full_av_reuse_refuses_changed_or_incomplete_original(
    reused_full_av: tuple[Example, list[str]],
    changed: str,
) -> None:
    example, calls = reused_full_av
    if changed == "media":
        example.master.write_bytes(b"different media")
    elif changed == "config":
        data = json.loads(example.config.read_text())
        data["admission"]["fullAVReuse"]["kind"] = "skip"
        _json(example.config, data)
    elif changed == "tool":
        (example.root / "pinned-decoder").write_bytes(b"changed decoder")
    elif changed == "code":
        # Change a recorded code fingerprint without editing the shared repository.
        receipt = json.loads((example.source_state / "join.json").read_text())
        receipt["fingerprint"] = "0" * 64
        _json(example.source_state / "join.json", receipt)
    elif changed == "missing":
        (example.source_state / "join.json").unlink()
    else:
        _json(example.source_state / "join.json", {"status": "running"})
    with pytest.raises((ReplayError, FileNotFoundError)):
        batch.run(
            example.manifest,
            example.digest,
            receipt=example.root / "bad.json",
            logs=example.root / "run",
        )
    assert calls == ["full-av"]
    assert not example.first.exists()


@pytest.mark.parametrize("change", ["kind", "duplicate", "other-role", "version"])
def test_typed_full_av_declaration_rejects_unknown_contract(
    reused_full_av: tuple[Example, list[str]],
    change: str,
) -> None:
    example, _ = reused_full_av
    data = json.loads(example.config.read_text())
    declaration = data["admission"]["fullAVReuse"]
    if change == "kind":
        declaration["kind"] = "skip-ancestor"
    elif change == "duplicate":
        declaration["stages"] = ["join", "join"]
    elif change == "other-role":
        declaration["stages"] = ["repair"]
    else:
        declaration["version"] = True
    _json(example.config, data)
    with pytest.raises(ReplayError, match="typed full AV|whole-join"):
        batch.prepare(
            example.config,
            manifest=example.root / "bad.json",
            workflow=example.root / "bad-workflow.json",
        )


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "drefs",
        "format",
        "wrong-output",
        "no-code",
        "no-tool",
        "prefix",
        "empty",
        "two-outputs",
        "threads-bad",
        "threads-large",
    ],
)
def test_full_av_contract_has_no_hidden_inputs_or_unbound_executor(
    reused_full_av: tuple[Example, list[str]],
    change: str,
) -> None:
    example, _ = reused_full_av
    stage = workflow.load_stages(example.root / "original.json")[2]
    command = list(stage.verify_command)
    if change == "extra":
        command.extend(["-i", str(example.source)])
    elif change == "drefs":
        command[command.index("-enable_drefs") + 1] = "1"
    elif change == "format":
        command[command.index("mov")] = "concat"
    elif change == "wrong-output":
        command[command.index("-i") + 1] = str(example.source)
    elif change == "no-code":
        stage = replace(
            stage, inputs=tuple(path for path in stage.inputs if path.name != "stages.py")
        )
    elif change == "no-tool":
        stage = replace(
            stage, inputs=tuple(path for path in stage.inputs if path.name != "pinned-decoder")
        )
    elif change == "prefix":
        command[1] = "-y"
    elif change == "empty":
        command = []
    elif change == "two-outputs":
        stage = replace(stage, outputs=(example.master, example.root / "extra-output.mov"))
    else:
        command[5:5] = ["-threads", "-1" if change == "threads-bad" else "257"]
    with pytest.raises(ReplayError, match="Full AV|Original full AV"):
        batch._full_av_contract(replace(stage, verify_command=tuple(command)))


@pytest.mark.parametrize("threads", ["0", "2", "256"])
def test_full_av_explicit_bounded_thread_contract(
    reused_full_av: tuple[Example, list[str]],
    threads: str,
) -> None:
    example, _ = reused_full_av
    stage = workflow.load_stages(example.root / "original.json")[2]
    command = list(stage.verify_command)
    command[5:5] = ["-threads", threads]
    result = batch._full_av_contract(replace(stage, verify_command=tuple(command)))
    assert result.stage.name == "join"


@pytest.mark.parametrize("rate", [24, 30])
def test_real_local_mov_full_av_once_then_exact_evidence_reuse(
    example: Example,
    rate: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = shutil.which("ffmpeg")
    assert executable is not None
    ffmpeg = Path(executable).resolve()
    media = example.root / "synthetic.mov"
    subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=size=32x48:rate={rate}:duration=0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.25",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(media),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    calls: list[str] = []
    ordinary = stages.execute_command

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command[0] == str(ffmpeg):
            calls.append("decode")
        ordinary(command, log, timeout)

    monkeypatch.setattr(stages, "execute_command", execute)
    monkeypatch.setattr(batch, "execute_command", execute)
    updated = _reuse_example(example, ffmpeg, media)
    receipt = example.root / "local-result.json"
    batch.run(updated.manifest, updated.digest, receipt=receipt, logs=example.root / "local-run")
    batch.verify(
        updated.manifest, updated.digest, receipt=receipt, logs=example.root / "local-verify"
    )
    assert calls == ["decode"]
    assert example.master.read_bytes() == media.read_bytes()


def test_full_av_reuse_preserves_stopped_stage_recovery_and_readonly_verify(
    reused_full_av: tuple[Example, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    example, decodes = reused_full_av
    ordinary = stages.execute_command
    interrupted = True
    produced: list[str] = []

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command[0] == "/bin/cp":
            produced.append(Path(command[-1]).stem)
            if interrupted and command[-1] == str(example.second):
                example.second.write_bytes(b"partial metadata")
                raise RuntimeError("stopped metadata")
        ordinary(command, log, timeout)

    def run_stage(stage: Stage, *, state_directory: Path) -> str:
        return stages.run_stage(stage, state_directory=state_directory, execute=execute)

    monkeypatch.setattr(workflow, "run_stage", run_stage)
    receipt = example.root / "resumed-result.json"
    with pytest.raises(RuntimeError, match="stopped metadata"):
        batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "attempt1")
    prefix = (example.state / "frames.json").read_bytes()
    with pytest.raises(StageError):
        batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "refused")
    assert produced == ["frames", "metadata"]
    with stages.execution_lock(example.state, ".workflow") as workflow_free:
        assert workflow_free
        with stages.execution_lock(example.state, "metadata") as stage_free:
            assert stage_free
            stage_recovery.recover(example.technical, example.state)
    interrupted = False
    produced.clear()
    batch.run(example.manifest, example.digest, receipt=receipt, logs=example.root / "attempt2")
    assert produced == ["metadata", "pcm"]
    assert (example.state / "frames.json").read_bytes() == prefix

    def prohibited(*args: object, **kwargs: object) -> None:
        pytest.fail("Read-only verification dispatched a producer or recovery")

    monkeypatch.setattr(batch, "run_plan", prohibited)
    monkeypatch.setattr(stage_recovery, "recover", prohibited)
    batch.verify(example.manifest, example.digest, receipt=receipt, logs=example.root / "verify")
    assert decodes.count("full-av") == 1
