"""Train the avatar-face / source-caption detector from a pinned checkpoint, offline, on MPS.

The wrapper refuses to start unless the checkpoint matches its recorded sha256, the settings file
pins every Ultralytics directory inside the checkout with all external services off, and the
dataset and run directories stay under `edits/work`. Ultralytics is imported only after those
checks, so a bad configuration cannot create files outside the repository.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from replay.core import ReplayError
from replay.mps_graph_cache import released_mps_graphs

REQUIRED_OFF = ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "wandb", "tensorboard")
DEVICES = ("mps", "cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def check_checkpoint(weights: Path, expected: str) -> None:
    """The checkpoint must be a regular file whose digest matches the recorded one."""
    if weights.is_symlink() or not weights.is_file() or sha256(weights) != expected:
        raise ReplayError("Checkpoint is missing or differs from its recorded sha256")


def check_settings(config_directory: Path, work: Path) -> dict[str, object]:
    """Ultralytics settings must keep every directory under `work` and every service off."""
    path = config_directory / "Ultralytics" / "settings.json"
    if not path.is_file():
        raise ReplayError("Pinned Ultralytics settings are missing")
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ReplayError("Ultralytics settings must be a JSON object")
    settings: dict[str, object] = loaded
    if any(settings.get(name) is not False for name in REQUIRED_OFF):
        raise ReplayError("Every external Ultralytics service must be off")
    for name in ("datasets_dir", "runs_dir"):
        value = settings.get(name)
        if not isinstance(value, str) or not inside(Path(value), work):
            raise ReplayError(f"Ultralytics {name} must stay under the work directory")
    return settings


def check_paths(data: Path, project: Path, work: Path, *, resume: bool = False) -> None:
    """A new run needs a fresh run directory; a resumed run needs the interrupted one."""
    for path in (data, project):
        if not inside(path, work):
            raise ReplayError("Dataset and run directories must stay under the work directory")
    if not data.is_file():
        raise ReplayError("Dataset description is missing")
    if project.exists() and not resume:
        raise ReplayError("Run directory already exists")
    if resume and not project.is_dir():
        raise ReplayError("There is no interrupted run to resume")


def resume_checkpoint(project: Path, name: str) -> Path:
    """The interrupted run's `last.pt`, which must be a regular file inside the run directory."""
    last = project / name / "weights" / "last.pt"
    if last.is_symlink() or not last.is_file() or not inside(last, project):
        raise ReplayError("The interrupted run has no last checkpoint")
    return last


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("project", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=DEVICES, default="mps")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", choices=("none", "ram"), default="none")
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the interrupted run PROJECT/NAME from its last.pt (same settings)",
    )
    return parser.parse_args(argv)


def train_arguments(args: argparse.Namespace) -> dict[str, object]:
    """The keyword arguments handed to Ultralytics, validated; speed options are explicit."""
    if not 1 <= args.epochs <= 1000 or not 32 <= args.imgsz <= 2048:
        raise ReplayError("Invalid epochs or image size")
    if not 0 < args.fraction <= 1 or args.patience < 1 or args.batch < 1 or args.workers < 0:
        raise ReplayError("Invalid fraction, patience, batch or workers")
    # Absolute paths: Ultralytics joins a relative project onto its own runs directory.
    return {
        "data": str(args.data.resolve()), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device, "workers": args.workers, "seed": args.seed,
        "deterministic": not args.no_deterministic,
        "cache": "ram" if args.cache == "ram" else False,
        "fraction": args.fraction, "patience": args.patience, "val": not args.no_val,
        "project": str(args.project.resolve()), "name": args.name, "exist_ok": False,
        "pretrained": True,
        "plots": False, "amp": False, "verbose": True,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    check_checkpoint(args.weights, args.weights_sha256)
    check_settings(args.config_directory, args.work)
    check_paths(args.data, args.project, args.work, resume=args.resume)
    options = train_arguments(args)
    last = resume_checkpoint(args.project, args.name) if args.resume else None
    os.environ["YOLO_CONFIG_DIR"] = str(args.config_directory)
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    # Imported only after the safety checks so a bad configuration cannot write outside the repo.
    from ultralytics.models.yolo.model import YOLO

    with released_mps_graphs():
        if last is None:
            YOLO(str(args.weights)).train(**options)
        else:
            # Ultralytics restores every training argument (data, epochs, run directory) from the
            # checkpoint; only the device is passed so the run continues where it stopped.
            YOLO(str(last)).train(resume=True, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
