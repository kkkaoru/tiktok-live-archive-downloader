import subprocess
from collections.abc import Sequence
from pathlib import Path

from replay import delivery_foreground as foreground
from replay import digest_store_run


def planned(tmp_path: Path, count: int) -> Path:
    directory = tmp_path / "plan"
    (directory / "configs").mkdir(parents=True)
    for index in range(count):
        (directory / "configs" / f"{index:04d}.json").write_text("{}", encoding="utf-8")
    return directory


class Fake:
    """Writes a receipt unless the window should fail `failures[index]` more times."""

    def __init__(self, failures: dict[int, int]) -> None:
        self.failures = failures
        self.started: list[int] = []

    def __call__(self, command: Sequence[str], log: Path) -> subprocess.Popen[bytes]:
        output = Path(command[-1])
        index = int(output.name.removeprefix("evidence-"))
        self.started.append(index)
        output.mkdir()
        if self.failures.get(index, 0) > 0:
            self.failures[index] -= 1
            return subprocess.Popen(["/usr/bin/false"])
        (output / "provenance.json").write_text("{}", encoding="utf-8")
        return subprocess.Popen(["/usr/bin/true"])


def test_supervise_observes_every_window_and_retries_once(tmp_path: Path) -> None:
    directory = planned(tmp_path, 3)
    fake = Fake({1: 1})
    result = foreground.supervise(directory, workers=2, spawn=fake, sleep=lambda _s: None)
    assert result == {"done": 3, "total": 3, "failed": []}
    assert sorted(fake.started) == [0, 1, 1, 2]
    assert (directory / "evidence-0001.failed-s1").is_dir()


def test_supervise_reports_exhausted_windows(tmp_path: Path) -> None:
    directory = planned(tmp_path, 1)
    result = foreground.supervise(directory, workers=1, spawn=Fake({0: 2}), sleep=lambda _s: None)
    assert result == {"done": 0, "total": 1, "failed": [0]}


def test_supervise_skips_observed_and_busy_windows(tmp_path: Path) -> None:
    directory = planned(tmp_path, 2)
    (directory / "evidence-0000").mkdir()
    (directory / "evidence-0000" / "provenance.json").write_text("{}", encoding="utf-8")
    (directory / "evidence-0001").mkdir()
    fake = Fake({})
    result = foreground.supervise(directory, workers=1, spawn=fake, sleep=lambda _s: None)
    assert fake.started == []
    assert result == {"done": 1, "total": 2, "failed": [], "busyElsewhere": [1]}


def test_command_uses_the_shared_digest_store(tmp_path: Path) -> None:
    command = foreground.command(tmp_path, 7)
    assert command[2:6] == [
        "replay.digest_store_run",
        str(digest_store_run.default_store()),
        "replay.foreground_collection",
        "observe",
    ]
    assert command[-1] == str(tmp_path / "evidence-0007")


def test_cli_exit_code(tmp_path: Path) -> None:
    directory = planned(tmp_path, 0)
    assert foreground.main([str(directory)]) == 0


def test_default_spawn_writes_log(tmp_path: Path) -> None:
    process = foreground._spawn(["/bin/echo", "observed"], tmp_path / "w.log")
    assert process.wait() == 0
    assert (tmp_path / "w.log").read_text(encoding="utf-8") == "observed\n"
