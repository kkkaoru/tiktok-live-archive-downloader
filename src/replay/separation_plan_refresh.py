"""Re-bind a separation PCM preparation and pipeline plan to the current code closure.

Every plan pins the sha256 of each module it imports, so an unrelated edit of a pinned module
turns a finished or half-run plan into one that no longer verifies, and a worker stops with
`pins changed`. This tool starts a new namespace from an existing config and plan template: it
refreshes only the pinned digests of files that are already in the pinned set, points the
outputs at the new directory, and runs the ordinary `extract` and `prepare-plan` producers there
so every digest is checked by them again. The old namespace is only read. A closure whose
membership changed (a module added or removed) is refused by `prepare-plan`; it is never patched.
The producers run in the separation provider environment (`edits/.venv-mel-roformer`), because
`prepare-plan` checks that the model provider is installed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from replay.core import ReplayError

ROOT = Path(__file__).resolve().parents[2]
MODEL_HOME = ROOT / "edits" / "models" / "mel-roformer"
PROVIDER_PYTHON = ROOT / "edits" / ".venv-mel-roformer" / "bin" / "python"
PRODUCER = "replay.separation_pipeline_prepare"
CONFIG_NAME = "PCM-PREPARATION-CONFIG-v1.json"
TEMPLATE_NAME = "PIPELINE-PLAN-TEMPLATE-v1.json"
PLAN_NAME = "PIPELINE-PLAN-v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refreshed(rows: object) -> list[dict[str, str]]:
    """The pinned rows with each file's current digest; every file must exist under the repo."""
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Expected a non-empty pinned code list")
    result: list[dict[str, str]] = []
    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_absolute() or not path.is_relative_to(ROOT) or not path.is_file():
            raise ReplayError("A pinned code file is missing or outside the checkout")
        result.append({"path": str(path), "sha256": sha256(path)})
    return result


def changed(rows: object) -> list[str]:
    """Names of pinned files whose content no longer matches the recorded digest."""
    if not isinstance(rows, list):
        raise ReplayError("Expected a pinned code list")
    return [
        Path(str(row["path"])).name
        for row in rows
        if sha256(Path(str(row["path"]))) != row["sha256"]
    ]


def write_new(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(data, stream, indent=1)


def environment(directory: Path) -> dict[str, str]:
    return {
        **os.environ,
        "TMPDIR": str(directory / "tmp"),
        "TMP": str(directory / "tmp"),
        "TEMP": str(directory / "tmp"),
        "XDG_CACHE_HOME": str(directory / "cache"),
        "HF_HOME": str(MODEL_HOME),
        "HF_HUB_CACHE": str(MODEL_HOME / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }


def produce(directory: Path, *arguments: str) -> None:
    subprocess.run(
        [str(PROVIDER_PYTHON), "-B", "-m", PRODUCER, *arguments],
        cwd=ROOT,
        env=environment(directory),
        stdin=subprocess.DEVNULL,
        check=True,
        timeout=3600,
    )


def refresh(old_config: Path, old_template: Path, new: Path, *, run: bool) -> Path:
    """Create `new` with refreshed config and template; optionally run extract and prepare-plan."""
    if new.exists() or new.is_symlink() or not new.parent.is_dir():
        raise ReplayError("The new namespace must not exist and needs an existing parent")
    if not new.is_absolute() or not new.is_relative_to(ROOT / "edits" / "work"):
        raise ReplayError("The new namespace must be an absolute path under edits/work")
    config = json.loads(old_config.read_text(encoding="utf-8"))
    template = json.loads(old_template.read_text(encoding="utf-8"))
    config["codeInputs"] = refreshed(config["codeInputs"])
    config["directory"] = str(new)
    template["codeInputs"] = refreshed(template["codeInputs"])
    template["outputDirectory"] = str(new / "pipeline")
    template["source"] = {"path": str(new / "model-input.wav"), "sha256": None}
    new.mkdir(mode=0o700)
    (new / "tmp").mkdir(mode=0o700)
    (new / "cache").mkdir(mode=0o700)
    write_new(new / CONFIG_NAME, config)
    write_new(new / TEMPLATE_NAME, template)
    if run:
        produce(new, "extract", str(new / CONFIG_NAME))
        produce(
            new,
            "prepare-plan",
            str(new / CONFIG_NAME),
            str(new / TEMPLATE_NAME),
            str(new / PLAN_NAME),
        )
    return new / PLAN_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="existing PCM-PREPARATION-CONFIG json")
    parser.add_argument("template", type=Path, help="existing PIPELINE-PLAN-TEMPLATE json")
    parser.add_argument("new", type=Path, help="new namespace directory")
    parser.add_argument("--no-run", action="store_true", help="write the two files only")
    args = parser.parse_args(argv)
    print(refresh(args.config, args.template, args.new, run=not args.no_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
