"""Synthetic offline evidence collection; no real Executor or private media."""

import json
import subprocess
from io import BufferedWriter
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay import subtitle_observations as observations


def fake_ocr(
    command: list[str],
    *,
    stdin: int,
    stdout: BufferedWriter,
    stderr: BufferedWriter,
    timeout: int,
    check: bool,
) -> subprocess.CompletedProcess[bytes]:
    native = command[1] == "measure-media"
    request = json.loads(Path(command[-1]).read_text(encoding="utf-8") if native else command[-1])
    frames = [
        {
            "requestedTimeSeconds": sample["timeSeconds"],
            "actualTimeSeconds": sample["timeSeconds"],
            "width": 64,
            "height": 64,
            **({"faces": [], "bodies": []} if "video_head_detect" in command else {"lines": []}),
        }
        for sample in request["samples"]
    ]
    stdout.write(
        json.dumps(
            frames
            if native
            else {
                "ok": True,
                "data": {
                    "structuredContent": {
                        "sourceModified": False,
                        "measurement": frames,
                    }
                },
            }
        ).encode()
    )
    return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Mock]:
    source, plan, output = tmp_path / "source.mp4", tmp_path / "times.json", tmp_path / "evidence"
    source.write_bytes(b"synthetic source")
    plan.write_text("[0, 0.25, 1, 2, 3, 4, 5, 6, 7]", encoding="utf-8")
    run = Mock(side_effect=fake_ocr)
    monkeypatch.setattr(subprocess, "run", run)
    return source, plan, output, run


@pytest.mark.parametrize(
    "value", [[], {}, [True], ["1"], [-1], [86401], [float("nan")], [2, 1], [1, 1]]
)
def test_invalid_times(value: object) -> None:
    with pytest.raises(ValueError):
        observations.observation_times(value)


def test_collect_resume_and_cli(
    collection: tuple[Path, Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, plan, output, run = collection
    assert observations.collect_observations(source, plan, output) == 9
    assert run.call_count == 2
    assert (output / "0000.result.json").stat().st_mode & 0o777 == 0o600
    assert observations.collect_observations(source, plan, output) == 9
    assert run.call_count == 2
    monkeypatch.setattr("sys.argv", ["ocr", str(source), str(plan), str(output)])
    observations.main()
    assert run.call_count == 2
    assert (
        json.loads((output / "complete.json").read_text(encoding="utf-8"))["samplingExhaustive"]
        is False
    )


@pytest.mark.parametrize("native", [False, True])
def test_head_collection_binds_its_mode_and_reuses_batches(
    collection: tuple[Path, Path, Path, Mock],
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
) -> None:
    source, plan, output, run = collection
    executable = source.with_name("native") if native else None
    if executable is not None:
        executable.write_text("synthetic executable", encoding="utf-8")
        executable.chmod(0o700)
    assert (
        observations.collect_observations(
            source, plan, output, native_executable=executable, head_tracking=True
        )
        == 9
    )
    assert run.call_count == 2
    assert "video_head_detect" in run.call_args.args[0]
    assert (
        observations.collect_observations(
            source, plan, output, native_executable=executable, head_tracking=True
        )
        == 9
    )
    assert run.call_count == 2
    inputs = json.loads((output / "inputs.json").read_text(encoding="utf-8"))
    assert inputs["measurementTool"] == "video_head_detect"
    assert len(inputs["headParserSHA256"]) == 64
    with pytest.raises(ValueError, match="inputs changed"):
        observations.collect_observations(source, plan, output, native_executable=executable)
    assert run.call_count == 2
    arguments = ["observe", str(source), str(plan), str(output), "--head-tracking"]
    if executable is not None:
        arguments.extend(["--native-executable", str(executable)])
    monkeypatch.setattr("sys.argv", arguments)
    observations.main()
    assert run.call_count == 2


def test_changed_source_refuses_reuse(collection: tuple[Path, Path, Path, Mock]) -> None:
    source, plan, output, run = collection
    observations.collect_observations(source, plan, output)
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="inputs changed"):
        observations.collect_observations(source, plan, output)
    assert run.call_count == 2


def test_missing_completed_result_never_recollects(
    collection: tuple[Path, Path, Path, Mock],
) -> None:
    source, plan, output, run = collection
    observations.collect_observations(source, plan, output)
    (output / "0000.result.json").unlink()
    with pytest.raises(ValueError, match="refusing recollection"):
        observations.collect_observations(source, plan, output)
    assert run.call_count == 2


def test_changed_completed_receipt_refused(collection: tuple[Path, Path, Path, Mock]) -> None:
    source, plan, output, run = collection
    observations.collect_observations(source, plan, output)
    (output / "complete.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed OCR evidence changed"):
        observations.collect_observations(source, plan, output)
    assert run.call_count == 2


@pytest.mark.parametrize("leaf", ["inputs.json", "complete.json", "0000.result.json"])
def test_linked_receipts_refused(collection: tuple[Path, Path, Path, Mock], leaf: str) -> None:
    source, plan, output, run = collection
    observations.collect_observations(source, plan, output)
    (output / leaf).unlink()
    (output / leaf).symlink_to(plan)
    with pytest.raises(ValueError):
        observations.collect_observations(source, plan, output)
    assert run.call_count == 2


def test_linked_output_refused(collection: tuple[Path, Path, Path, Mock]) -> None:
    source, plan, output, run = collection
    output.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked OCR evidence"):
        observations.collect_observations(source, plan, output)
    run.assert_not_called()


def test_large_plan_refused(collection: tuple[Path, Path, Path, Mock]) -> None:
    source, plan, output, run = collection
    plan.write_bytes(b" " * 1048577)
    with pytest.raises(ValueError, match="byte budget"):
        observations.collect_observations(source, plan, output)
    run.assert_not_called()


def test_command_failure_preserved_and_can_resume(
    collection: tuple[Path, Path, Path, Mock],
) -> None:
    source, plan, output, run = collection
    run.side_effect = None
    run.return_value = subprocess.CompletedProcess([], 1)
    with pytest.raises(ValueError, match="attempt evidence retained"):
        observations.collect_observations(source, plan, output)
    assert len(list(output.glob("attempt-*"))) == 1
    assert not (output / "0000.result.json").exists()
    run.side_effect = fake_ocr
    assert observations.collect_observations(source, plan, output) == 9
    assert len(list(output.glob("attempt-*"))) == 3


def test_empty_response_not_published(collection: tuple[Path, Path, Path, Mock]) -> None:
    source, plan, output, run = collection
    run.side_effect = None
    run.return_value = subprocess.CompletedProcess([], 0)
    with pytest.raises(ValueError, match="Invalid OCR result file"):
        observations.collect_observations(source, plan, output)
    assert not (output / "0000.result.json").exists()


def test_native_cli_preserves_raw_measurements_and_resumes(
    collection: tuple[Path, Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, plan, output, run = collection
    native = source.parent / "apple-pro-apps"
    native.write_bytes(b"synthetic executable")
    native.chmod(0o700)
    assert observations.collect_observations(source, plan, output, native_executable=native) == 9
    assert run.call_count == 2
    assert run.call_args.args[0][:3] == [str(native), "measure-media", "video_text_recognize"]
    assert Path(run.call_args.args[0][-1]).is_absolute()
    assert isinstance(json.loads((output / "0000.result.json").read_text(encoding="utf-8")), list)
    assert (
        json.loads((output / "inputs.json").read_text(encoding="utf-8"))["backend"] == "native-cli"
    )
    monkeypatch.setattr(
        "sys.argv", ["ocr", str(source), str(plan), str(output), "--native-executable", str(native)]
    )
    observations.main()
    assert run.call_count == 2
    native.write_bytes(b"changed executable")
    with pytest.raises(ValueError, match="inputs changed"):
        observations.collect_observations(source, plan, output, native_executable=native)
    assert run.call_count == 2


def test_native_failure_never_falls_back(collection: tuple[Path, Path, Path, Mock]) -> None:
    source, plan, output, run = collection
    native = source.parent / "apple-pro-apps"
    native.write_bytes(b"synthetic executable")
    native.chmod(0o700)
    run.side_effect = None
    run.return_value = subprocess.CompletedProcess([], 1)
    with pytest.raises(ValueError, match="attempt evidence retained"):
        observations.collect_observations(source, plan, output, native_executable=native)
    assert run.call_count == 1
    assert not (output / "complete.json").exists()
    assert not (output / "0000.result.json").exists()


@pytest.mark.parametrize("relative", [True, False])
def test_invalid_explicit_native_executable(
    collection: tuple[Path, Path, Path, Mock], relative: bool
) -> None:
    source, plan, output, run = collection
    native = Path("relative-cli") if relative else source
    with pytest.raises(ValueError, match="absolute and executable"):
        observations.collect_observations(source, plan, output, native_executable=native)
    run.assert_not_called()


def test_result_clock_tampering_refused(collection: tuple[Path, Path, Path, Mock]) -> None:
    source, plan, output, run = collection
    observations.collect_observations(source, plan, output)
    result = output / "0000.result.json"
    value = json.loads(result.read_text(encoding="utf-8"))
    value["data"]["structuredContent"]["measurement"][0]["requestedTimeSeconds"] = 100
    result.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="clocks differ"):
        observations.collect_observations(source, plan, output)
    assert run.call_count == 2
