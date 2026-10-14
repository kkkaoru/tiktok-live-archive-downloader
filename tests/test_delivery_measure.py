import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import delivery_captions as captions
from replay import delivery_measure as measure


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def inputs(tmp_path: Path) -> measure.Inputs:
    pts = [index * 3000 for index in range(128)]
    clock = {"pts": pts, "timeBase": "1/90000", "lastDuration": 3000}
    write(tmp_path / "fg" / "configs" / "0000.json", {"clock": clock})
    write(tmp_path / "fg" / "evidence-0000" / "provenance.json", {})
    tokens = write(
        tmp_path / "tokens.json",
        {"tokens": [{"text": "あ", "start": 0.5, "end": 1.0, "utterance": "u"}]},
    )
    cuts = write(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 90, "output_start_frame": 0}]},
    )
    native, calibration = tmp_path / "native", tmp_path / "calibration.json"
    native.write_bytes(b"native")
    calibration.write_text("{}", encoding="utf-8")
    settings = captions.Settings(30, native, calibration, (0, 0, 1080, 1920), 540, 24, 300, 900)
    return measure.Inputs(tokens, cuts, (tmp_path / "fg",), None, settings)


class Fake:
    """Pretends to run measurement commands; succeeds after `failures` attempts."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str], log: Path) -> subprocess.Popen[bytes]:
        self.commands.append(list(command))
        output = Path(command[command.index("--manifest-sha256") - 1])
        output.mkdir(parents=True)
        if self.failures > 0:
            self.failures -= 1
            return subprocess.Popen(["/usr/bin/false"])
        (output / "candidate.json").write_text("{}", encoding="utf-8")
        return subprocess.Popen(["/usr/bin/true"])


def test_supervise_retries_a_failed_job_once(tmp_path: Path) -> None:
    fake = Fake(failures=1)
    result = measure.supervise(
        tmp_path / "plan", inputs(tmp_path), workers=2, spawn=fake, sleep=lambda _s: None
    )
    assert result == {"done": 1, "total": 1, "failed": []}
    assert len(fake.commands) == 2
    assert fake.commands[0][2] == "replay.caption_clamp"
    assert (tmp_path / "plan" / "measurement-0000.failed-s1").is_dir()
    changes = json.loads((tmp_path / "plan" / "token-changes.json").read_text(encoding="utf-8"))
    assert changes == {"rate": 30, "merges": [], "clipped": []}


def test_supervise_gives_up_after_the_attempt_budget(tmp_path: Path) -> None:
    fake = Fake(failures=2)
    result = measure.supervise(
        tmp_path / "plan", inputs(tmp_path), workers=1, spawn=fake, sleep=lambda _s: None
    )
    assert result == {"done": 0, "total": 1, "failed": [0]}


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    native = tmp_path / "native"
    native.write_bytes(b"x")
    args = [str(tmp_path / "missing.json"), str(tmp_path / "cuts.json"), str(tmp_path / "plan")]
    args += ["--native", str(native), "--calibration", str(native)]
    assert measure.main(args) == 1
    assert "DELIVERY_MEASURE_ERROR" in capsys.readouterr().err


def test_default_spawn_writes_log(tmp_path: Path) -> None:
    process = measure._spawn(["/bin/echo", "measured"], tmp_path / "job.log")
    assert process.wait() == 0
    assert (tmp_path / "job.log").read_text(encoding="utf-8") == "measured\n"
