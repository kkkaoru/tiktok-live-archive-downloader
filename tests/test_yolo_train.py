"""Training refuses to start unless the checkpoint, settings and directories are safe."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest

from replay import yolo_train as train
from replay.core import ReplayError


def settings(root: Path, **changes: object) -> Path:
    config = root / "config"
    (config / "Ultralytics").mkdir(parents=True)
    work = root / "work"
    values: dict[str, object] = {
        "datasets_dir": str(work / "datasets"),
        "runs_dir": str(work / "runs"),
        "sync": False,
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "raytune": False,
        "wandb": False,
        "tensorboard": False,
    }
    values.update(changes)
    (config / "Ultralytics" / "settings.json").write_text(json.dumps(values), encoding="utf-8")
    return config


def test_the_digest_of_a_file_is_its_sha256(tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    assert train.sha256(weights) == hashlib.sha256(b"weights").hexdigest()


def test_a_matching_checkpoint_is_accepted(tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    train.check_checkpoint(weights, hashlib.sha256(b"weights").hexdigest())


def test_a_changed_or_missing_or_linked_checkpoint_is_refused(tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    with pytest.raises(ReplayError, match="differs from its recorded sha256"):
        train.check_checkpoint(weights, "0" * 64)
    with pytest.raises(ReplayError, match="missing"):
        train.check_checkpoint(tmp_path / "absent.pt", "0" * 64)
    link = tmp_path / "link.pt"
    link.symlink_to(weights)
    with pytest.raises(ReplayError, match="missing or differs"):
        train.check_checkpoint(link, hashlib.sha256(b"weights").hexdigest())


def test_pinned_settings_inside_the_work_directory_are_accepted(tmp_path: Path) -> None:
    config = settings(tmp_path)
    assert train.check_settings(config, tmp_path / "work")["sync"] is False


def test_settings_that_are_not_a_json_object_are_refused(tmp_path: Path) -> None:
    config = tmp_path / "config"
    (config / "Ultralytics").mkdir(parents=True)
    (config / "Ultralytics" / "settings.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ReplayError, match="must be a JSON object"):
        train.check_settings(config, tmp_path / "work")


def test_missing_settings_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="settings are missing"):
        train.check_settings(tmp_path / "config", tmp_path / "work")


def test_an_enabled_external_service_is_refused(tmp_path: Path) -> None:
    config = settings(tmp_path, sync=True)
    with pytest.raises(ReplayError, match="external Ultralytics service must be off"):
        train.check_settings(config, tmp_path / "work")


def test_a_directory_outside_the_work_directory_is_refused(tmp_path: Path) -> None:
    config = settings(tmp_path, datasets_dir="/somewhere/else")
    with pytest.raises(ReplayError, match="datasets_dir must stay under"):
        train.check_settings(config, tmp_path / "work")


def test_a_setting_that_is_not_a_string_path_is_refused(tmp_path: Path) -> None:
    config = settings(tmp_path, runs_dir=5)
    with pytest.raises(ReplayError, match="runs_dir must stay under"):
        train.check_settings(config, tmp_path / "work")


def test_dataset_and_run_directories_must_be_inside_work_and_new(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    data = work / "data.yaml"
    data.write_text("names: {}\n", encoding="utf-8")
    train.check_paths(data, work / "runs", work)
    with pytest.raises(ReplayError, match="must stay under the work directory"):
        train.check_paths(tmp_path / "data.yaml", work / "runs", work)
    with pytest.raises(ReplayError, match="must stay under the work directory"):
        train.check_paths(data, tmp_path / "runs", work)
    with pytest.raises(ReplayError, match="description is missing"):
        train.check_paths(work / "absent.yaml", work / "runs", work)
    (work / "runs").mkdir()
    with pytest.raises(ReplayError, match="already exists"):
        train.check_paths(data, work / "runs", work)


def test_main_refuses_a_bad_checkpoint_before_importing_ultralytics(tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    work = tmp_path / "work"
    work.mkdir()
    config = settings(tmp_path)
    with pytest.raises(ReplayError, match="differs from its recorded sha256"):
        train.main(
            [str(work / "data.yaml"), str(work / "runs"), "--weights", str(weights),
             "--weights-sha256", "0" * 64, "--config-directory", str(config),
             "--work", str(work), "--name", "x", "--epochs", "1"]
        )  # fmt: skip


def arguments(*extra: str) -> argparse.Namespace:
    return train.parse(
        ["d.yaml", "runs", "--weights", "w.pt", "--weights-sha256", "0" * 64,
         "--config-directory", "c", "--work", "w", "--name", "x", "--epochs", "5", *extra]
    )  # fmt: skip


def test_the_default_training_arguments_are_deterministic_uncached_and_validated() -> None:
    options = train.train_arguments(arguments())
    assert options["deterministic"] is True and options["cache"] is False
    assert options["fraction"] == 1.0 and options["val"] is True and options["patience"] == 100
    assert options["device"] == "mps" and options["amp"] is False and options["exist_ok"] is False
    assert options["project"] == str(Path("runs").resolve())


def test_speed_options_change_only_what_they_name() -> None:
    options = train.train_arguments(
        arguments("--cache", "ram", "--no-deterministic", "--fraction", "0.25", "--patience", "5",
                  "--no-val", "--batch", "32", "--workers", "4")
    )  # fmt: skip
    assert options["cache"] == "ram" and options["deterministic"] is False
    assert options["fraction"] == 0.25 and options["patience"] == 5 and options["val"] is False
    assert options["batch"] == 32 and options["workers"] == 4


@pytest.mark.parametrize(
    "extra", [("--fraction", "0"), ("--fraction", "1.5"), ("--patience", "0"), ("--workers", "-1")]
)
def test_invalid_speed_options_are_refused(extra: tuple[str, ...]) -> None:
    with pytest.raises(ReplayError, match="Invalid fraction, patience, batch or workers"):
        train.train_arguments(arguments(*extra))


def test_main_refuses_unsafe_epochs_after_the_other_checks(tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    work = tmp_path / "work"
    work.mkdir()
    (work / "data.yaml").write_text("names: {}\n", encoding="utf-8")
    config = settings(tmp_path)
    with pytest.raises(ReplayError, match="Invalid epochs or image size"):
        train.main(
            [str(work / "data.yaml"), str(work / "runs"), "--weights", str(weights),
             "--weights-sha256", hashlib.sha256(b"weights").hexdigest(),
             "--config-directory", str(config), "--work", str(work), "--name", "x",
             "--epochs", "0"]
        )  # fmt: skip


def test_a_resumed_run_needs_the_interrupted_run_directory(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    data = work / "data.yaml"
    data.write_text("names: {}\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="no interrupted run to resume"):
        train.check_paths(data, work / "runs", work, resume=True)
    (work / "runs").mkdir()
    train.check_paths(data, work / "runs", work, resume=True)


def test_the_resume_checkpoint_is_the_runs_last_weights(tmp_path: Path) -> None:
    weights = tmp_path / "runs" / "train" / "weights"
    weights.mkdir(parents=True)
    with pytest.raises(ReplayError, match="no last checkpoint"):
        train.resume_checkpoint(tmp_path / "runs", "train")
    (weights / "last.pt").write_bytes(b"last")
    assert train.resume_checkpoint(tmp_path / "runs", "train") == weights / "last.pt"


def test_a_linked_resume_checkpoint_is_refused(tmp_path: Path) -> None:
    weights = tmp_path / "runs" / "train" / "weights"
    weights.mkdir(parents=True)
    (tmp_path / "elsewhere.pt").write_bytes(b"last")
    (weights / "last.pt").symlink_to(tmp_path / "elsewhere.pt")
    with pytest.raises(ReplayError, match="no last checkpoint"):
        train.resume_checkpoint(tmp_path / "runs", "train")


class FakeYolo:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def train(self, **options: object) -> None:
        FakeYolo.calls.append((Path(self.weights).name, options))


def test_main_resumes_from_the_last_checkpoint_with_only_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    work = tmp_path / "work"
    (work / "runs" / "x" / "weights").mkdir(parents=True)
    (work / "runs" / "x" / "weights" / "last.pt").write_bytes(b"last")
    (work / "data.yaml").write_text("names: {}\n", encoding="utf-8")
    config = settings(tmp_path, datasets_dir=str(work / "d"), runs_dir=str(work / "r"))
    monkeypatch.setattr("ultralytics.models.yolo.model.YOLO", FakeYolo)
    FakeYolo.calls = []
    code = train.main(
        [str(work / "data.yaml"), str(work / "runs"), "--weights", str(weights),
         "--weights-sha256", hashlib.sha256(b"weights").hexdigest(),
         "--config-directory", str(config), "--work", str(work), "--name", "x",
         "--epochs", "3", "--resume"]
    )  # fmt: skip
    assert code == 0
    assert FakeYolo.calls == [("last.pt", {"resume": True, "device": "mps"})]


def test_main_starts_a_new_run_from_the_pinned_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    work = tmp_path / "work"
    work.mkdir()
    (work / "data.yaml").write_text("names: {}\n", encoding="utf-8")
    config = settings(tmp_path, datasets_dir=str(work / "d"), runs_dir=str(work / "r"))
    monkeypatch.setattr("ultralytics.models.yolo.model.YOLO", FakeYolo)
    FakeYolo.calls = []
    train.main(
        [str(work / "data.yaml"), str(work / "runs"), "--weights", str(weights),
         "--weights-sha256", hashlib.sha256(b"weights").hexdigest(),
         "--config-directory", str(config), "--work", str(work), "--name", "x",
         "--epochs", "3"]
    )  # fmt: skip
    assert FakeYolo.calls[0][0] == "w.pt"
    assert FakeYolo.calls[0][1]["epochs"] == 3
