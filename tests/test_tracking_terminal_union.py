import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from replay import tracking_observations as storage
from replay import tracking_terminal_selection as terminal
from replay import tracking_terminal_union as union
from replay.stages import Stage, _fingerprint, execution_lock, file_digest


def write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, default=str), encoding="utf-8")


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def checkpoint(stage: Stage, state: Path) -> None:
    write(
        state / f"{stage.name}.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(path): file_digest(path) for path in stage.outputs},
        },
    )


@pytest.fixture
def case(tmp_path: Path) -> tuple[Path, Path]:
    source, native = tmp_path / "source", tmp_path / "native"
    source.write_bytes(b"synthetic")
    native.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        "if sys.argv[1]=='inspect-media':\n"
        " print(json.dumps(dict(durationSeconds=.09,width=16,height=16,firstFrameDecoded=True)))\n"
        "else:\n"
        " p=json.loads(Path(sys.argv[3]).read_text())['plan']\n"
        " print(json.dumps([dict(presentationValue=round(p['startSeconds']*100),"
        "presentationTimescale=100,width=16,height=16,method='missing',"
        "detectionAttempted=True,faces=[])]))\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    cfg0, cfg1 = tmp_path / "zero.json", tmp_path / "one.json"
    base = {
        "version": 1,
        "source": str(source),
        "sourceSHA256": file_digest(source),
        "nativeExecutable": str(native),
        "nativeSHA256": file_digest(native),
        "width": 16,
        "height": 16,
    }
    write(cfg0, {**base, "clock": {"pts": [0], "timeBase": "1/100", "lastDuration": 6}})
    write(cfg1, {**base, "clock": {"pts": [6], "timeBase": "1/100", "lastDuration": 4}})
    old0, old1, new = tmp_path / "old0", tmp_path / "old1", tmp_path / "new"
    state, newstate = tmp_path / "state", tmp_path / "newstate"
    state.mkdir()
    newstate.mkdir()
    storage.observe(cfg0, old0)
    terminal.observe(cfg1, new)
    first = storage.tracking_stage(cfg0, old0, name="first")
    last = storage.tracking_stage(cfg1, old1, name="last")
    newstage = Stage(
        name="terminal",
        timeout_seconds=1200,
        command=(
            sys.executable,
            "-m",
            "replay.tracking_terminal_selection",
            "observe",
            str(cfg1),
            str(new),
        ),
        inputs=(cfg1, source, native),
        outputs=tuple(
            new / n for n in ("inspection.json", "request.json", "response.json", "receipt.json")
        ),
    )
    checkpoint(first, state)
    checkpoint(newstage, newstate)
    workflow, newworkflow = tmp_path / "workflow.json", tmp_path / "newworkflow.json"
    write(workflow, {"version": 1, "stages": [asdict(first), asdict(last)]})
    write(newworkflow, {"version": 1, "stages": [asdict(newstage)]})
    index, clock = tmp_path / "index.json", tmp_path / "clock.json"
    write(
        index,
        {
            "windows": [
                {
                    "index": 0,
                    "name": "first",
                    "config": str(cfg0),
                    "evidence": str(old0),
                    "originalState": str(state),
                    "originalCheckpoint": str(state / "first.json"),
                },
                {
                    "index": 1,
                    "name": "last",
                    "config": str(cfg1),
                    "evidence": str(old1),
                    "originalState": str(state),
                    "originalCheckpoint": str(state / "last.json"),
                },
            ]
        },
    )
    write(clock, {"pts": [0, 6], "timeBase": "1/100", "lastDuration": 4})
    parent, config = tmp_path / "parent.json", tmp_path / "config.json"
    write(
        parent,
        {
            "version": 1,
            "workflow": pin(workflow),
            "index": pin(index),
            "clock": pin(clock),
            "seedOverlap": "reject",
        },
    )
    write(
        config,
        {
            "version": 1,
            "original": pin(parent),
            "terminalWorkflow": pin(newworkflow),
            "terminalState": str(newstate),
        },
    )
    return config, tmp_path / "union.json"


def test_full_union_without_faking_failed_checkpoint(case: tuple[Path, Path]) -> None:
    config, output = case
    catalog = union.finalize(config, output)
    union.finalize(config, output, verify=True)
    data = json.loads(output.read_text())
    assert data["frames"] == 2
    assert data["clock"] == {"pts": [0, 6], "timeBase": "1/100", "lastDuration": 4}
    assert data["legacyFailedWindowAdopted"] is False
    assert data["methods"] == {"missing": 2}
    assert len(catalog.windows) == 2
    assert not (config.parent / "state/last.json").exists()


@pytest.mark.parametrize("field,value", [("version", True), ("version", 2), ("extra", 1)])
def test_schema(case: tuple[Path, Path], field: str, value: object) -> None:
    config, output = case
    data = json.loads(config.read_text())
    data[field] = value
    write(config, data)
    with pytest.raises(ValueError, match="configuration"):
        union.finalize(config, output)


def test_active_actor(case: tuple[Path, Path]) -> None:
    with (
        execution_lock(case[0].parent / "newstate", ".workflow"),
        pytest.raises(ValueError, match="owned"),
    ):
        union.finalize(*case)


def test_changed_union(case: tuple[Path, Path]) -> None:
    union.finalize(*case)
    write(case[1], {})
    with pytest.raises(ValueError, match="receipt changed"):
        union.finalize(*case, verify=True)


def test_missing_completed_window(case: tuple[Path, Path]) -> None:
    (case[0].parent / "state/first.json").unlink()
    with pytest.raises(FileNotFoundError):
        union.finalize(*case)


def test_cli(case: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["union", "finalize", str(case[0]), str(case[1])])
    union.main()


def test_full_clock_mismatch(case: tuple[Path, Path]) -> None:
    config, output = case
    clock = config.parent / "clock.json"
    write(clock, {"pts": [0, 7], "timeBase": "1/100", "lastDuration": 3})
    parent = config.parent / "parent.json"
    value = json.loads(parent.read_text())
    value["clock"] = pin(clock)
    write(parent, value)
    value = json.loads(config.read_text())
    value["original"] = pin(parent)
    write(config, value)
    with pytest.raises(ValueError, match="unchanged full original clock"):
        union.finalize(config, output)


def test_terminal_clock_cannot_be_trimmed(case: tuple[Path, Path]) -> None:
    config, output = case
    clock = config.parent / "clock.json"
    write(clock, {"pts": [0, 6], "timeBase": "1/100", "lastDuration": 3})
    parent = config.parent / "parent.json"
    value = json.loads(parent.read_text())
    value["clock"] = pin(clock)
    write(parent, value)
    value = json.loads(config.read_text())
    value["original"] = pin(parent)
    write(config, value)
    with pytest.raises(ValueError, match="unchanged full original clock"):
        union.finalize(config, output)
