"""Bind a partition's Save request to fresh native observations without app actions."""

import argparse
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_caption_srt import caption_cues
from replay.recordings import record
from replay.stages import file_digest

MAX_OBSERVATION_AGE = 30


def _name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode()) > 100
        or any(char in value for char in ("/", "\\", "\n", "\r", "\0"))
    ):
        raise ReplayError("Expected a bounded output name prefix")
    return value


def _observation(value: object, *, historical: bool) -> tuple[Path, dict[str, object]]:
    path = _pin(value)
    age = time.time() - path.stat().st_mtime
    if not historical and not 0 <= age <= MAX_OBSERVATION_AGE:
        raise ReplayError("Native observation is stale or future dated")
    raw = _json(path)
    if "data" in raw:
        if raw.get("ok") is not True or raw.get("error") is not None:
            raise ReplayError("Native observation did not complete successfully")
        data = record(raw["data"])
        if data.get("isError") is not False:
            raise ReplayError("Native observation MCP result failed")
        return path, record(data.get("structuredContent"))
    if "result" in raw:
        if raw.get("ok") is not True or raw.get("error") is not None:
            raise ReplayError("Native observation did not complete successfully")
        return path, record(raw["result"])
    return path, record(raw.get("response", raw))


def _transition(
    value: object,
    *,
    earlier: tuple[tuple[Path, dict[str, object]], ...],
    snapshot: tuple[Path, dict[str, object]],
    names: tuple[str, ...],
) -> Path:
    path = _pin(value)
    raw = _json(path)
    if "result" in raw:
        if raw.get("ok") is not True or raw.get("error") is not None:
            raise ReplayError("Share transition failed")
        raw = record(raw["result"])
    if "data" in raw:
        if raw.get("ok") is not True:
            raise ReplayError("Share transition failed")
        raw = record(raw["data"])
    meta = record(raw.get("_meta"))
    target = record(meta.get("target_receipt"))
    pid = target.get("pid")
    process_start = target.get("process_start_identity")
    if (
        meta.get("dispatch_state") != "dispatched"
        or type(pid) is not int
        or pid <= 0
        or type(process_start) is not int
        or process_start <= 0
        or snapshot[1].get("pid") != pid
        or path.stat().st_mtime > snapshot[0].stat().st_mtime
        or any(
            payload.get("pid") != pid or source.stat().st_mtime > path.stat().st_mtime
            for source, payload in earlier
        )
    ):
        raise ReplayError("Share transition process or observation order differs")
    elements = snapshot[1].get("elements")
    if not isinstance(elements, list):
        raise ReplayError("Save transition has no elements")
    fields = [
        record(item) for item in elements if record(item).get("identifier") == "saveAsNameTextField"
    ]
    if len(fields) != 1 or fields[0].get("value") not in names:
        raise ReplayError("Fresh Save filename does not bind the partition project")
    return path


def _idle(payload: dict[str, object]) -> None:
    elements = payload.get("elements")
    if not isinstance(elements, list) or not 1 <= len(elements) <= 500:
        raise ReplayError("Expected bounded native background elements")
    rows = [record(item) for item in elements]
    labels = [row.get("value") for row in rows if row.get("role") == "AXStaticText"]
    statuses = [row.get("value") for row in rows if row.get("identifier") == "_NS:166"]
    if (
        not any(label in ("共有", "Sharing") for label in labels)
        or not statuses
        or any(status not in ("待機中", "Idle") for status in statuses)
    ):
        raise ReplayError("Native background jobs are active or their state is unknown")


def prepare(config_path: Path, observations_path: Path, output: Path) -> None:
    """Write an offline native planner request; never dispatch or infer completion."""
    config = _json(config_path)
    observations = _json(observations_path)
    if config.get("version") != 1 or observations.get("version") != 1:
        raise ReplayError("Expected version-one configuration and observations")
    historical = observations.get("historicalInspection", False)
    if not isinstance(historical, bool):
        raise ReplayError("Historical inspection must be boolean")
    manifest_path = _pin(config.get("partition"))
    manifest = _json(manifest_path)
    parts = manifest.get("parts")
    number = config.get("part")
    if (
        manifest.get("version") != 1
        or not isinstance(parts, list)
        or not 1 <= len(parts) <= 1024
        or type(number) is not int
        or not 1 <= number <= len(parts)
    ):
        raise ReplayError("Expected an existing bounded partition number")
    part = _pin(parts[number - 1])
    caption_cues(part)
    project = ET.parse(part).getroot().find("./library/event/project")
    if project is None or not project.get("name"):
        raise ReplayError("Partition project identity is missing")
    transition = observations.get("shareTransition")
    project_path, identity = _observation(
        observations.get("project"), historical=historical or transition is not None
    )
    native_project = record(record(identity.get("finalCut")).get("project"))
    if native_project.get("name") != project.get("name"):
        raise ReplayError("Native project differs from the selected partition")
    background_path, background = _observation(
        observations.get("background"), historical=historical or transition is not None
    )
    _idle(background)
    snapshot, panel = _observation(observations.get("savePanel"), historical=historical)
    if panel.get("truncated") is True or not isinstance(panel.get("elements"), list):
        raise ReplayError("Save panel must contain native ui_inspect elements")
    directory = _path(config.get("directory"))
    if not directory.is_dir() or directory != _path(observations.get("observedDirectory")):
        raise ReplayError("Save directory must be independently observed and already selected")
    name = f"{_name(config.get('masterPrefix'))}-{number:04d}-master.mov"
    transition_paths: tuple[Path, ...] = ()
    if transition is not None:
        transition_paths = (
            _transition(
                transition,
                earlier=((project_path, identity), (background_path, background)),
                snapshot=(snapshot, panel),
                names=(f"{project.get('name')}.mov", name),
            ),
        )
    target = directory / name
    if target.exists() or target.is_symlink():
        raise ReplayError("Native master destination already exists; do not resubmit")
    title = _name(observations.get("windowTitle"))
    output = _path(str(output))
    receipt_path = output.with_suffix(".binding.json")
    decoded_path = output.with_suffix(".snapshot.json")
    if any(path.exists() or path.is_symlink() for path in (output, receipt_path, decoded_path)):
        raise ReplayError("Native Save planning evidence must not be overwritten")
    with decoded_path.open("x", encoding="utf-8") as handle:
        json.dump(panel, handle, ensure_ascii=False, indent=2)
    request: dict[str, object] = {
        "context": {
            "directory": str(directory),
            "fileName": name,
            "observedDirectory": str(directory),
            "window": {"title": title},
            "saveDispatched": False,
        },
        "snapshotPath": str(decoded_path),
        "snapshotSHA256": file_digest(decoded_path),
        "historicalInspection": historical,
    }
    receipt = {
        "version": 1,
        "part": number,
        "project": project.get("name"),
        "inputs": {
            str(path): file_digest(path)
            for path in (
                config_path,
                observations_path,
                manifest_path,
                part,
                project_path,
                background_path,
                snapshot,
                *transition_paths,
            )
        },
        "outputPath": str(target),
        "state": "awaiting-native-save-plan",
        "dispatched": False,
        "completionVerified": False,
        "requiredNext": (
            "Run native fcp-save-plan; dispatch at most one action, then observe again"
        ),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(request, handle, ensure_ascii=False, indent=2)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)


def capture_paths(
    output: Path,
    *,
    project: Path,
    background: Path,
    save_panel: Path,
    observed_directory: Path,
    window_title: str,
    share_transition: Path | None = None,
) -> None:
    """Bind existing raw observations; no recapture, timestamp change, or GUI access."""
    sources = {"project": project, "background": background, "savePanel": save_panel}
    if share_transition is not None:
        sources["shareTransition"] = share_transition
    document: dict[str, object] = {
        "version": 1,
        "observedDirectory": str(_path(str(observed_directory))),
        "windowTitle": _name(window_title),
        **{
            name: {"path": str(_path(str(path))), "sha256": file_digest(path)}
            for name, path in sources.items()
        },
    }
    with _path(str(output)).open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--capture-paths",
        nargs=5,
        metavar=("PROJECT", "BACKGROUND", "SAVE_PANEL", "DIRECTORY", "WINDOW_TITLE"),
    )
    parser.add_argument("--share-transition", type=Path)
    args = parser.parse_args()
    if args.capture_paths is not None:
        project, background, panel, directory, title = args.capture_paths
        capture_paths(
            args.observations,
            project=Path(project),
            background=Path(background),
            save_panel=Path(panel),
            observed_directory=Path(directory),
            window_title=title,
            share_transition=args.share_transition,
        )
    elif args.share_transition is not None:
        parser.error("--share-transition requires --capture-paths")
    prepare(args.config, args.observations, args.output)


if __name__ == "__main__":
    main()
