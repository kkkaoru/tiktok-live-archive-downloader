"""Synthetic session boundaries only; no native/model/media execution."""

import json
import os
import subprocess
from fractions import Fraction
from io import BufferedWriter
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay import subtitle_session_observations as session
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import execution_lock, file_digest


def fake_native(
    command: list[str],
    *,
    stdin: int,
    stdout: BufferedWriter,
    stderr: BufferedWriter,
    timeout: int,
    check: bool,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    request = record(session._read(Path(command[-1])))
    directory = Path(str(request["checkpointDirectory"]))
    samples = session._times(request["samples"])
    blocks = []
    for start in range(0, len(samples), 8):
        points = samples[start : start + 8]
        raw = directory / f"block-{start // 8}.json"
        session._write(
            raw,
            [
                {
                    "requestedTimeSeconds": float(point),
                    "actualTimeSeconds": float(point),
                    "width": 64,
                    "height": 64,
                    "lines": [],
                }
                for point in points
            ],
        )
        block = {
            "firstSample": start,
            "requested": session._clocks(points),
            "actual": session._clocks(points),
            "measurementsPath": str(raw),
        }
        session._write(directory / f"block-{start // 8}-clocks.json", block)
        blocks.append(block)
    report = {
        "protocolVersion": session.PROTOCOL,
        "sourceModified": False,
        "qualityApproved": False,
        "sourceHashVerified": False,
        "sourceIdentity": session._identity(Path(str(request["path"]))),
        "frames": len(samples),
        "blocks": blocks,
        "timing": {"elapsedSeconds": 0.1},
    }
    session._write(directory / "complete.json", report)
    stdout.write(json.dumps(report).encode())
    return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Mock]:
    source, native, plan = tmp_path / "source", tmp_path / "native", tmp_path / "plan.json"
    source.write_bytes(b"synthetic source")
    native.write_bytes(b"synthetic executable")
    session._write(
        plan,
        {
            "version": 1,
            "protocol": session.PROTOCOL,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": file_digest(native),
            "samples": [{"value": index, "timescale": 30} for index in range(65)],
            "width": 64,
            "height": 64,
        },
    )
    run = Mock(side_effect=fake_native)
    monkeypatch.setattr(subprocess, "run", run)
    return plan, tmp_path / "output", run


def test_collect_verify_resume_and_cli(
    setup: tuple[Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, output, run = setup
    assert session.collect(plan, output) == 65
    assert session.collect(plan, output, verify=True) == 65
    assert session.collect(plan, output) == 65
    assert run.call_count == 2
    producer = record(session._read(output / "inputs.json"))
    assert producer["measurementTool"] == "video_text_recognize_session"
    assert producer["backend"] == "native-session-cli"
    monkeypatch.setattr("sys.argv", ["session", "verify", str(plan), str(output)])
    session.main()
    assert (output / "inputs.json").stat().st_mode & 0o777 == 0o600


def test_retained_producer_is_read_only_and_preserves_data_pins(tmp_path: Path) -> None:
    path = str(tmp_path / "caption_timing.py")
    data_path = str(tmp_path / "data.json")
    recorded: dict[str, object] = {
        "version": 1,
        "inputs": {
            path: "a444cc7b353794f69a38af73e0c4dd98630bc6a9922f625c545c3752aa0ae23e",
            data_path: "a" * 64,
        },
    }
    expected: dict[str, object] = {
        "version": 1,
        "inputs": {path: "c" * 64, data_path: "a" * 64},
    }
    result = session._retained_producer_pins(recorded, expected, verify=True)
    assert result[path] == "a444cc7b353794f69a38af73e0c4dd98630bc6a9922f625c545c3752aa0ae23e"
    with pytest.raises(ValueError, match="producer changed"):
        session._retained_producer_pins(recorded, expected, verify=False)
    with pytest.raises(ValueError, match="producer changed"):
        session._retained_producer_pins(recorded, {**expected, "version": 2}, verify=True)
    with pytest.raises(ValueError, match="producer changed"):
        session._retained_producer_pins(
            recorded, {**expected, "inputs": {path: "c" * 64, data_path: "b" * 64}}, verify=True
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_explicit_workers_are_bound_to_request_and_readonly_resume(
    setup: tuple[Path, Path, Mock], workers: int
) -> None:
    plan, output, run = setup
    data = record(session._read(plan))
    data["workers"] = workers
    plan.write_text(json.dumps(data), encoding="utf-8")
    assert session.collect(plan, output) == 65
    request = record(session._read(Path(run.call_args_list[0].args[0][-1])))
    assert request["workers"] == workers
    assert session.collect(plan, output, verify=True) == 65
    assert run.call_count == 2
    data["workers"] = 2 if workers == 1 else 1
    plan.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        session.collect(plan, output, verify=True)
    assert run.call_count == 2


@pytest.mark.parametrize("workers", [0, 3, True, None, "2", 2.0])
def test_invalid_worker_authority_fails_before_native(
    setup: tuple[Path, Path, Mock], workers: object
) -> None:
    plan, output, run = setup
    data = record(session._read(plan))
    data["workers"] = workers
    plan.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        session.collect(plan, output)
    run.assert_not_called()
    assert not output.exists()


def test_explicit_external_root_is_readonly_and_identity_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    external = tmp_path / "external"
    checkout.mkdir()
    external.mkdir()
    source = external / "source.mov"
    native = checkout / "native"
    source.write_bytes(b"synthetic media")
    native.write_bytes(b"synthetic provider")
    monkeypatch.chdir(checkout)
    plan_path = checkout / "plan.json"
    data = {
        "version": 1,
        "protocol": session.PROTOCOL,
        "source": str(source),
        "sourceSHA256": file_digest(source),
        "nativeExecutable": str(native),
        "nativeSHA256": file_digest(native),
        "samples": [{"value": 0, "timescale": 25}],
        "width": 64,
        "height": 96,
        "allowedSourceRoot": str(external),
    }
    session._write(plan_path, data)
    plan = session.load_plan(plan_path)
    assert plan.source_root is not None
    session._check({str(source): file_digest(source)}, plan)
    with pytest.raises(ValueError):
        session._path(str(external / "output"))
    data.pop("allowedSourceRoot")
    plan_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        session.load_plan(plan_path)
    external.rename(tmp_path / "retained")
    external.mkdir()
    with pytest.raises(ValueError, match="root changed"):
        session._identity(source, plan.source_root)


@pytest.mark.parametrize("kind", ["missing", "directory", "hardlink", "symlink", "fifo"])
def test_source_root_rejects_unusable_or_linked_source(tmp_path: Path, kind: str) -> None:
    root = session._source_root(str(tmp_path))
    source = tmp_path / "source"
    if kind == "directory":
        source.mkdir()
    elif kind == "hardlink":
        original = tmp_path / "original"
        original.write_bytes(b"retained")
        os.link(original, source)
    elif kind == "symlink":
        source.symlink_to(tmp_path / "elsewhere")
    elif kind == "fifo":
        os.mkfifo(source)
    with pytest.raises((ValueError, FileNotFoundError)):
        session._identity(source, root)


def test_root_must_exist_and_cannot_escape_via_link(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        session._source_root(str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        session._source_root("/")
    with pytest.raises(ValueError):
        session._source_root(None)
    child = tmp_path / "child"
    child.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(child, target_is_directory=True)
    with pytest.raises(ValueError):
        session._source_root(str(alias))
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    root = session._source_root(str(child))
    with pytest.raises(ValueError):
        session._source_path(str(outside), root)
    with pytest.raises(ValueError):
        session._source_path(str(child), root)


def test_declared_root_provenance_survives_readonly_verification(
    setup: tuple[Path, Path, Mock],
) -> None:
    plan, output, run = setup
    data = record(session._read(plan))
    data["allowedSourceRoot"] = str(plan.parent)
    plan.write_text(json.dumps(data), encoding="utf-8")
    assert session.collect(plan, output) == 65
    provenance = record(record(session._read(output / "inputs.json"))["allowedSourceRoot"])
    assert provenance["path"] == str(plan.parent)
    assert provenance["device"] == plan.parent.stat().st_dev
    assert provenance["inode"] == plan.parent.stat().st_ino
    assert provenance["readOnly"] is True
    assert session.collect(plan, output, verify=True) == 65
    assert run.call_count == 2


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        [{"value": True, "timescale": 1}],
        [{"value": -1, "timescale": 1}],
        [{"value": 1, "timescale": 0}],
        [{"value": 86401, "timescale": 1}],
        [{"value": 0, "timescale": 1}, {"value": 0, "timescale": 2}],
    ],
)
def test_invalid_clocks(value: object) -> None:
    with pytest.raises(ValueError):
        session._times(value)


@pytest.mark.parametrize("value", [None, "relative", "/tmp/no", str(Path.cwd() / "private/no")])
def test_scope(value: object) -> None:
    with pytest.raises(ValueError):
        session._path(value)


@pytest.mark.parametrize("value", [None, "F" * 64, "a"])
def test_bad_hash(value: object) -> None:
    with pytest.raises(ValueError):
        session._sha(value)


def test_link_and_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.touch()
    with pytest.raises(ValueError):
        session._identity(empty)
    link = tmp_path / "link"
    link.symlink_to(empty)
    with pytest.raises(ValueError):
        session._path(str(link))


def test_duplicate_and_large_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        session._read(path)
    monkeypatch.setattr(session, "MAX_JSON", 1)
    with pytest.raises(ValueError, match="bound"):
        session._read(path)


@pytest.mark.parametrize("field,value", [("version", True), ("protocol", "legacy"), ("width", 0)])
def test_bad_plan(setup: tuple[Path, Path, Mock], field: str, value: object) -> None:
    plan, _, _ = setup
    document = record(session._read(plan))
    document[field] = value
    plan.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        session.load_plan(plan)


def test_source_mutation(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, _ = setup
    session.load_plan(plan).source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cryptographic"):
        session.collect(plan, output)


def test_partial_failure_retained_and_resumed(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, run = setup
    run.side_effect = [subprocess.CompletedProcess([], 1)]
    with pytest.raises(ValueError, match="failed"):
        session.collect(plan, output)
    assert (output / "attempt-000000/boundary.json").exists()
    run.side_effect = fake_native
    assert session.collect(plan, output) == 65
    assert (output / "attempt-000000/stderr.log").exists()


def test_timeout_retains_boundary(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, run = setup
    run.side_effect = subprocess.TimeoutExpired("synthetic", 90)
    with pytest.raises(ValueError, match="failed"):
        session.collect(plan, output)
    with pytest.raises(ValueError, match="Incomplete"):
        session.collect(plan, output, verify=True)


@pytest.mark.parametrize(
    "target",
    [
        "complete.json",
        "attempt-000000/boundary.json",
        "attempt-000000/native/block-0.json",
    ],
)
def test_completed_tamper(setup: tuple[Path, Path, Mock], target: str) -> None:
    plan, output, _ = setup
    session.collect(plan, output)
    (output / target).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        session.collect(plan, output, verify=True)


def test_missing_producer_inputs_rejected(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, _ = setup
    session.collect(plan, output)
    (output / "inputs.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="Malformed TikTok API object"):
        session.collect(plan, output, verify=True)


def test_missing_completion_never_repaired_by_verify(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, run = setup
    session.collect(plan, output)
    (output / "complete.json").unlink()
    with pytest.raises(ValueError, match="Missing session completion"):
        session.collect(plan, output, verify=True)
    assert run.call_count == 2


def test_canvas_and_rational_mismatch(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, _ = setup
    session.collect(plan, output)
    with pytest.raises(ValueError, match="canvas"):
        session.validate_blocks(
            output / "attempt-000000/native",
            session.load_plan(plan).samples[:64],
            width=65,
            height=64,
        )
    with pytest.raises(ValueError, match="clocks"):
        session.validate_blocks(
            output / "attempt-000000/native", (Fraction(1),), width=64, height=64
        )


def test_native_report_failure(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, _ = setup
    session.collect(plan, output)
    with pytest.raises(ValueError, match="completion"):
        session._report(output / "attempt-000000/native", session.load_plan(plan).samples[:64], {})


def failed_prefix_native(
    command: list[str],
    *,
    stdin: int,
    stdout: BufferedWriter,
    stderr: BufferedWriter,
    timeout: int,
    check: bool,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    fake_native(
        command, stdin=stdin, stdout=stdout, stderr=stderr, timeout=timeout, check=check, env=env
    )
    native = Path(command[-1]).parent / "native"
    (native / "block-1-clocks.json").unlink()
    (native / "complete.json").unlink()
    return subprocess.CompletedProcess(command, 1)


def test_eight_frame_failure_prefix_reuses_exact_original_clock(
    setup: tuple[Path, Path, Mock],
) -> None:
    plan, output, run = setup
    run.side_effect = failed_prefix_native
    with pytest.raises(ValueError, match="failed"):
        session.collect(plan, output)
    boundary = record(session._read(output / "attempt-000000/boundary.json"))
    assert boundary["count"] == 8
    run.side_effect = fake_native
    assert session.collect(plan, output) == 65
    request = record(session._read(output / "attempt-000001/request.json"))
    assert session._times(request["samples"])[0] == Fraction(4, 15)
    assert run.call_count == 2


def test_unknown_unsealed_attempt_blocks_relaunch(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, run = setup
    output.mkdir()
    (output / "attempt-000000").mkdir()
    with pytest.raises(FileNotFoundError):
        session.collect(plan, output)
    assert run.call_count == 0


def test_active_lock_blocks_second_owner(setup: tuple[Path, Path, Mock]) -> None:
    plan, output, run = setup
    output.mkdir()
    with execution_lock(output, "collector"), pytest.raises(ValueError, match="owned"):
        session.collect(plan, output)
    assert run.call_count == 0
