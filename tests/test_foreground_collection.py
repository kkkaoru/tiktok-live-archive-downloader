import json
import subprocess
import sys
from pathlib import Path

import pytest

from replay import foreground_collection as collection
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, file_digest, run_stage


@pytest.fixture
def job(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "synthetic-source.bin"
    source.write_bytes(b"synthetic source only")
    native = tmp_path / "native fake"
    native.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        "assert sys.argv[1:3] == ['measure-media', 'video_foreground_measure']\n"
        "request = json.loads(Path(sys.argv[3]).read_text())\n"
        "assert 1 <= len(request['samples']) <= 8\n"
        "print(json.dumps([{'requested': s, 'actual': s, 'width': 16, 'height': 16, "
        "'observationPresent': False, 'instanceLabels': None, 'unionMask': None, "
        "'maskThreshold8Bit': 1, 'captionPlacementApproved': False} "
        "for s in request['samples']]))\n",
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
                "clock": {
                    "pts": [0, 1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008],
                    "timeBase": "1/30000",
                    "lastDuration": 1001,
                },
                "width": 16,
                "height": 16,
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "observations"


def test_batches_exact_clocks_and_reuses_only_verified_receipts(job: tuple[Path, Path]) -> None:
    config, output = job
    stage = collection.foreground_stage(config, output, name="foreground")
    assert run_stage(stage, state_directory=output.parent / "state") == "completed"
    assert run_stage(stage, state_directory=output.parent / "state") == "reused"
    collection.verify(config, output)
    first = record(json.loads((output / "batch-0000/request.json").read_text()))
    last = record(json.loads((output / "batch-0001/request.json").read_text()))
    assert first["samples"] == [
        {"value": 0, "timescale": 30000},
        {"value": 1001, "timescale": 30000},
        {"value": 2002, "timescale": 30000},
        {"value": 3003, "timescale": 30000},
        {"value": 4004, "timescale": 30000},
        {"value": 5005, "timescale": 30000},
        {"value": 6006, "timescale": 30000},
        {"value": 7007, "timescale": 30000},
    ]
    assert last["samples"] == [{"value": 8008, "timescale": 30000}]
    receipt = record(json.loads((output / "provenance.json").read_text()))
    assert receipt["frames"] == 9
    assert receipt["nativeExitCode"] == 0
    assert receipt["captionPlacementApproved"] is False
    assert (output / "batch-0000/response.json").stat().st_mode & 0o777 == 0o600
    assert (output / "inputs.json").stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    assert stage.timeout_seconds == 1200
    (output / "batch-0001/response.json").unlink()
    with pytest.raises(FileNotFoundError):
        collection.verify(config, output)
    with pytest.raises(StageError, match="Stage command failed"):
        run_stage(stage, state_directory=output.parent / "state")
    assert not (output / "batch-0001/response.json").exists()


def test_preserves_existing_output_and_refuses_adoption(job: tuple[Path, Path]) -> None:
    config, output = job
    collection.observe(config, output)
    before = file_digest(output / "provenance.json")
    with pytest.raises(FileExistsError):
        collection.observe(config, output)
    with pytest.raises(StageError, match="adoption"):
        run_stage(
            collection.foreground_stage(config, output, name="f"),
            state_directory=output.parent / "state",
        )
    assert file_digest(output / "provenance.json") == before


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"version": 2},
        {"extra": 1},
        {"redetectIntervalSeconds": 0.5},
        {"width": True},
        {"height": 20000},
        {"source": "relative"},
        {"sourceSHA256": "bad"},
        {"clock": None},
        {"clock": {"pts": [], "timeBase": "1/30000", "lastDuration": 1}},
        {"clock": {"pts": [0] * 129, "timeBase": "1/30000", "lastDuration": 1}},
        {"clock": {"pts": [True], "timeBase": "1/30000", "lastDuration": 1}},
        {"clock": {"pts": [1, 0], "timeBase": "1/30000", "lastDuration": 1}},
        {"clock": {"pts": [0], "timeBase": "1e9999", "lastDuration": 1}},
        {"clock": {"pts": [0], "timeBase": "1/0", "lastDuration": 1}},
        {"clock": {"pts": [0], "timeBase": "1/1", "lastDuration": 121}},
        {"clock": {"pts": [0], "timeBase": "1/1", "lastDuration": 0}},
    ],
)
def test_invalid_config_does_not_create_output(
    job: tuple[Path, Path], change: dict[str, object]
) -> None:
    config, output = job
    config.write_text(json.dumps(record(json.loads(config.read_text())) | change), encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        collection.observe(config, output)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["source", "binary", "permission", "alias", "output-link"])
def test_changed_or_unsafe_inputs_stop_before_work(job: tuple[Path, Path], mode: str) -> None:
    config, output = job
    parsed = collection.load_job(config)
    if mode == "source":
        parsed.source.write_bytes(b"changed")
    elif mode == "binary":
        parsed.native.write_bytes(b"changed")
    elif mode == "permission":
        parsed.native.chmod(0o600)
    elif mode == "alias":
        raw = record(json.loads(config.read_text()))
        config.write_text(
            json.dumps(raw | {"source": str(parsed.native), "sourceSHA256": parsed.native_hash}),
            encoding="utf-8",
        )
    else:
        target = output.parent / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        collection.observe(config, output)
    assert not (output / "inputs.json").exists()


@pytest.mark.parametrize(
    "artifact,contents",
    [
        ("inputs.json", "{}"),
        ("provenance.json", "{}"),
        ("batch-0000/request.json", "{}"),
        ("batch-0000/response.json", "[]"),
        ("batch-0000/stderr.log", "warning"),
    ],
)
def test_changed_artifacts_are_not_reused(
    job: tuple[Path, Path], artifact: str, contents: str
) -> None:
    config, output = job
    collection.observe(config, output)
    (output / artifact).write_text(contents, encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        collection.verify(config, output)


@pytest.mark.parametrize(
    "field,value", [("frames", True), ("nativeExitCode", False), ("captionPlacementApproved", 0)]
)
def test_receipt_types_remain_exact(job: tuple[Path, Path], field: str, value: object) -> None:
    config, output = job
    collection.observe(config, output)
    path = output / "provenance.json"
    raw = record(json.loads(path.read_text()))
    path.write_text(json.dumps(raw | {field: value}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        collection.verify(config, output)


def test_request_boolean_is_not_equal_to_integer_clock(job: tuple[Path, Path]) -> None:
    config, output = job
    collection.observe(config, output)
    path = output / "batch-0000/request.json"
    path.write_text(path.read_text().replace('"value": 0', '"value": false'), encoding="utf-8")
    with pytest.raises(ValueError, match="exact source clock"):
        collection.verify(config, output)


@pytest.mark.parametrize(
    "extra",
    [
        "raise SystemExit(7)\n",
        "print('warning', file=sys.stderr)\n",
        "Path(request['path']).write_bytes(b'changed')\n",
        "Path(sys.argv[3]).write_text(Path(sys.argv[3]).read_text() + ' ')\n",
        "if len(request['samples']) == 1: raise SystemExit(7)\n",
    ],
)
def test_failures_preserve_partial_batches_without_receipt(
    job: tuple[Path, Path], extra: str
) -> None:
    config, output = job
    parsed = collection.load_job(config)
    parsed.native.write_text(parsed.native.read_text() + extra, encoding="utf-8")
    raw = record(json.loads(config.read_text()))
    config.write_text(
        json.dumps(raw | {"nativeSHA256": file_digest(parsed.native)}), encoding="utf-8"
    )
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        collection.observe(config, output)
    assert (output / "inputs.json").exists()
    assert (output / "batch-0000/response.json").exists()
    assert not (output / "provenance.json").exists()


def test_timeout_preserves_attempt(job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    config, output = job

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("synthetic", 60)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        collection.observe(config, output)
    assert (output / "batch-0000/request.json").exists()
    assert not (output / "provenance.json").exists()


@pytest.mark.parametrize("action", ["observe", "verify"])
def test_code_mutation_is_rejected(
    job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    config, output = job
    if action == "verify":
        collection.observe(config, output)
    calls = 0

    def changed(path: Path) -> str:
        nonlocal calls
        if path.name == "foreground_observations.py":
            calls += 1
            if calls > 1:
                return "0" * 64
        return file_digest(path)

    monkeypatch.setattr(collection, "file_digest", changed)
    with pytest.raises(ValueError, match="changed"):
        if action == "observe":
            collection.observe(config, output)
        else:
            collection.verify(config, output)


@pytest.mark.parametrize("action", ["collect", "observe", "verify"])
def test_cli_uses_shared_runner(
    job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    config, output = job
    if action == "verify":
        collection.observe(config, output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "foreground",
            action,
            str(config),
            str(output),
            "--state-directory",
            str(output.parent / "state"),
        ],
    )
    collection.main()
    assert (output / "provenance.json").exists()


def test_collect_requires_state(job: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    config, output = job
    monkeypatch.setattr(sys, "argv", ["foreground", "collect", str(config), str(output)])
    with pytest.raises(SystemExit) as error:
        collection.main()
    assert error.value.code == 2
    assert not output.exists()
