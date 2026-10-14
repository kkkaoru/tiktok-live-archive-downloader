import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, file_digest, run_stage
from replay.tracking_observations import load_job, main, observe, tracking_stage, verify


@pytest.fixture
def job(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "synthetic-source.bin"
    source.write_bytes(b"synthetic source only")
    native = tmp_path / "native fake"
    native.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        "assert sys.argv[1:3] == ['measure-media', 'video_face_track']\n"
        "request = json.loads(Path(sys.argv[3]).read_text())\n"
        "assert request['plan']['sourceTimescale'] == 30000\n"
        "print(json.dumps([{'presentationValue': 1014, 'presentationTimescale': 30000, "
        "'width': 16, 'height': 16, 'method': 'missing', "
        "'detectionAttempted': True, 'faces': []}]))\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(source),
                "sourceSHA256": file_digest(source),
                "nativeExecutable": str(native),
                "nativeSHA256": file_digest(native),
                "clock": {"pts": [1014], "timeBase": "1/30000", "lastDuration": 1001},
                "width": 16,
                "height": 16,
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "observations"


def test_collects_and_reuses_only_bound_verified_results(job: tuple[Path, Path]) -> None:
    config, output = job
    stage = tracking_stage(config, output, name="window")
    state = output.parent / "state"
    assert run_stage(stage, state_directory=state) == "completed"
    assert run_stage(stage, state_directory=state) == "reused"
    verify(config, output)
    assert (output / "response.json").stat().st_mode & 0o777 == 0o600
    assert (output / "provenance.json").stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    report = record(json.loads((output / "provenance.json").read_text()))
    assert report["frames"] == 1
    assert report["captionPlacementApproved"] is False
    (output / "response.json").unlink()
    with pytest.raises(FileNotFoundError):
        run_stage(stage, state_directory=state)
    assert not (output / "response.json").exists()


def test_preserves_existing_evidence_without_overwriting(job: tuple[Path, Path]) -> None:
    config, output = job
    observe(config, output)
    before = file_digest(output / "response.json")
    with pytest.raises(FileExistsError):
        observe(config, output)
    assert file_digest(output / "response.json") == before


def test_refuses_unregistered_output_adoption(job: tuple[Path, Path]) -> None:
    config, output = job
    observe(config, output)
    with pytest.raises(StageError, match="adoption"):
        run_stage(
            tracking_stage(config, output, name="window"), state_directory=output.parent / "state"
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"unexpected": 1},
        {"width": True},
        {"height": 20000},
        {"redetectIntervalSeconds": False},
        {"redetectIntervalSeconds": float("nan")},
        {"redetectIntervalSeconds": 0},
        {"redetectIntervalSeconds": "0.5"},
        {"source": "relative"},
        {"source": "\0"},
        {"sourceSHA256": "bad"},
        {"nativeSHA256": 123},
        {"clock": None},
        {"clock": {"pts": [True], "timeBase": "1/30000", "lastDuration": 1}},
        {"clock": {"pts": [1], "timeBase": "1e9999999", "lastDuration": 1}},
        {"clock": {"pts": [1], "timeBase": "1/0", "lastDuration": 1}},
        {"clock": {"pts": None, "timeBase": "1/30000", "lastDuration": 1}},
    ],
)
def test_invalid_configs_stop_before_output(
    job: tuple[Path, Path], changes: dict[str, object]
) -> None:
    config, output = job
    raw = record(json.loads(config.read_text()))
    config.write_text(json.dumps(raw | changes), encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        observe(config, output)
    assert not output.exists()


@pytest.mark.parametrize("contents", [b"", b" " * 1_048_577])
def test_bounds_configuration_reads(tmp_path: Path, contents: bytes) -> None:
    config = tmp_path / "config.json"
    config.write_bytes(contents)
    with pytest.raises(ValueError, match="byte budget"):
        load_job(config)


def test_refuses_nonregular_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        load_job(tmp_path)


def test_refuses_a_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    pipe = tmp_path / "config.fifo"
    os.mkfifo(pipe)
    with pytest.raises(ValueError, match="regular file"):
        load_job(pipe)


def test_refuses_linked_config(job: tuple[Path, Path]) -> None:
    config, output = job
    link = output.parent / "config-link"
    link.symlink_to(config)
    with pytest.raises(OSError):
        load_job(link)


def test_pins_source_and_executable_before_work(job: tuple[Path, Path]) -> None:
    config, output = job
    parsed = load_job(config)
    parsed.source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="changed"):
        observe(config, output)
    assert not output.exists()


def test_requires_executable_permission(job: tuple[Path, Path]) -> None:
    config, output = job
    load_job(config).native.chmod(0o600)
    with pytest.raises(ValueError, match="not executable"):
        observe(config, output)


def test_rejects_linked_output_storage(job: tuple[Path, Path]) -> None:
    config, output = job
    elsewhere = output.parent / "other"
    elsewhere.mkdir()
    output.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        observe(config, output)
    assert tuple(elsewhere.iterdir()) == ()


@pytest.mark.parametrize(
    "artifact,contents",
    [
        ("request.json", "{}"),
        ("provenance.json", "{}"),
        ("stderr.log", "warning"),
        ("response.json", "[]"),
    ],
)
def test_verification_rejects_changed_artifacts(
    job: tuple[Path, Path], artifact: str, contents: str
) -> None:
    config, output = job
    observe(config, output)
    (output / artifact).write_text(contents, encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        verify(config, output)


@pytest.mark.parametrize(
    "field,value", [("nativeExitCode", False), ("frames", True), ("captionPlacementApproved", 0)]
)
def test_receipt_scalar_types_are_not_coerced(
    job: tuple[Path, Path], field: str, value: object
) -> None:
    config, output = job
    observe(config, output)
    path = output / "provenance.json"
    raw = record(json.loads(path.read_text()))
    raw[field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance changed"):
        verify(config, output)


def test_verification_rejects_linked_empty_diagnostics(job: tuple[Path, Path]) -> None:
    config, output = job
    observe(config, output)
    empty = output.parent / "empty"
    empty.touch()
    (output / "stderr.log").unlink()
    (output / "stderr.log").symlink_to(empty)
    with pytest.raises(ValueError, match="regular evidence"):
        verify(config, output)


@pytest.mark.parametrize(
    "extra",
    [
        "raise SystemExit(7)\n",
        "print('warning', file=sys.stderr)\n",
        "Path(request['path']).write_bytes(b'changed')\n",
    ],
)
def test_failed_process_diagnostics_and_input_mutation_keep_partial_evidence(
    job: tuple[Path, Path], extra: str
) -> None:
    config, output = job
    raw = record(json.loads(config.read_text()))
    native = load_job(config).native
    native.write_text(native.read_text() + extra, encoding="utf-8")
    raw["nativeSHA256"] = file_digest(native)
    config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        observe(config, output)
    assert (output / "response.json").exists()
    assert not (output / "provenance.json").exists()


def test_timeout_retains_attempt_without_receipt(
    job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = job

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("synthetic", 1)

    monkeypatch.setattr("replay.tracking_observations.subprocess.run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        observe(config, output)
    assert (output / "request.json").exists()
    assert not (output / "provenance.json").exists()


@pytest.mark.parametrize("action", ["observe", "verify"])
def test_code_mutation_during_collection_or_verification_is_rejected(
    job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    config, output = job
    if action == "verify":
        observe(config, output)
    calls = 0

    def changing_digest(path: Path) -> str:
        nonlocal calls
        if path.name == "caption_io.py":
            calls += 1
            if calls > 1:
                return "0" * 64
        return file_digest(path)

    monkeypatch.setattr("replay.tracking_observations.file_digest", changing_digest)
    with pytest.raises(ValueError, match="changed"):
        if action == "observe":
            observe(config, output)
        else:
            verify(config, output)


@pytest.mark.parametrize("action", ["collect", "observe", "verify"])
def test_cli_routes_to_shared_execution(
    job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    config, output = job
    if action == "verify":
        observe(config, output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tracking",
            action,
            str(config),
            str(output),
            "--state-directory",
            str(output.parent / "state"),
        ],
    )
    main()
    assert (output / "provenance.json").is_file()


def test_collect_requires_explicit_state(
    job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = job
    monkeypatch.setattr(sys, "argv", ["tracking", "collect", str(config), str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()
