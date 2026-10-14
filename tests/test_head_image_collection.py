import copy
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from test_head_image_support import frame as frame

from replay import head_image_collection as collection
from replay.core import ReplayError
from replay.head_image_plan import ImageFrameCell
from replay.stages import file_digest
from replay.workflow import load_stages


@pytest.fixture
def config(tmp_path: Path) -> Path:
    files: dict[str, Path] = {}
    for name in ("source", "nativeExecutable", "sandbox"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = path
    files["nativeExecutable"].chmod(0o700)
    config = tmp_path / "collection.json"
    cells = [
        ImageFrameCell(
            Fraction(index, 30), Fraction(index + 1, 30), (20, 110, 20, 20) if index != 8 else None
        ).encoded()
        for index in range(10)
    ]
    config.write_text(
        json.dumps(
            {
                "version": 1,
                **{key: str(path) for key, path in files.items()},
                "sourceSHA256": file_digest(files["source"]),
                "nativeSHA256": file_digest(files["nativeExecutable"]),
                "sandboxSHA256": file_digest(files["sandbox"]),
                "width": 100,
                "height": 200,
                "cells": cells,
                "inputs": {str(path): file_digest(path) for path in files.values()},
            }
        ),
        encoding="utf-8",
    )
    return config


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch, frame: dict[str, object]) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def run(args: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        request = json.loads(Path(args[-1]).read_text())
        rows = []
        for clock, face in zip(request["samples"], request["headFaces"], strict=True):
            assert face["presentation"] == clock
            row = copy.deepcopy(frame)
            row["requested"] = row["actual"] = clock
            rows.append(row)
        Path(args[-1]).with_name("response.json").write_text(json.dumps(rows), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_batched_exact_cells_keep_unknown_and_receipts(
    config: Path, tmp_path: Path, native: list[tuple[str, ...]]
) -> None:
    output = tmp_path / "measured"
    collection.observe(config, output)
    arc = collection.read_arc(config, output)
    assert len(native) == 2
    assert native[0][:3] == ("/usr/bin/sandbox-exec", "-f", str(tmp_path / "sandbox"))
    assert len(arc.cells) == 10 and arc.cells[8].visible_top is None
    assert arc.cells[8].start == Fraction(8, 30) and arc.cells[8].end == Fraction(9, 30)
    assert all(cell.visible_top == 100 for index, cell in enumerate(arc.cells) if index != 8)
    receipt = json.loads((output / "provenance.json").read_text())
    assert receipt["measuredFrames"] == 9 and receipt["supportedFrames"] == 9
    assert receipt["semanticHeadVerified"] is False and receipt["captionPlacementApproved"] is False
    with pytest.raises(FileExistsError):
        collection.observe(config, output)


def test_source_mutation_never_reuses_measurements(
    config: Path, tmp_path: Path, native: list[tuple[str, ...]]
) -> None:
    output = tmp_path / "measured"
    collection.observe(config, output)
    (tmp_path / "source").write_text("changed", encoding="utf-8")
    with pytest.raises(ReplayError, match="input changed"):
        collection.read_arc(config, output)


@pytest.mark.parametrize("target", ["request", "response", "receipt", "inputs", "diagnostic"])
def test_domain_verifier_reads_referenced_evidence(
    config: Path, tmp_path: Path, native: list[tuple[str, ...]], target: str
) -> None:
    output = tmp_path / "measured"
    collection.observe(config, output)
    paths = {
        "request": output / "batch-0000/request.json",
        "response": output / "batch-0000/response.json",
        "receipt": output / "provenance.json",
        "inputs": output / "inputs.json",
        "diagnostic": output / "batch-0000/stderr.log",
    }
    paths[target].write_text("{}", encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        collection.read_arc(config, output)


@pytest.mark.parametrize(
    "patch",
    [
        {"version": 2},
        {"extra": True},
        {"width": 0},
        {"height": 16385},
        {"inputs": {}},
        {"nativeSHA256": "a" * 64},
        {"cells": []},
    ],
)
def test_invalid_configs_rejected(config: Path, patch: dict[str, object]) -> None:
    raw = json.loads(config.read_text())
    config.write_text(json.dumps(raw | patch), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        collection.load_job(config)


def test_native_execute_bit_required(config: Path, tmp_path: Path) -> None:
    (tmp_path / "nativeExecutable").chmod(0o600)
    with pytest.raises(ReplayError, match="not executable"):
        collection.observe(config, tmp_path / "measured")


def test_native_request_rejects_missing_or_unbounded_faces(config: Path) -> None:
    job = collection.load_job(config)
    for cells in ((), (job.cells[8],), (job.cells[0],) * 9):
        with pytest.raises(ReplayError, match="observed same-frame"):
            job.request(cells)


def test_collection_cli_both_paths(
    config: Path, tmp_path: Path, native: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "measured"
    for action in ("observe", "verify"):
        monkeypatch.setattr(sys, "argv", ["collection", action, str(config), str(output)])
        collection.main()
    assert len(native) == 2


def test_native_failure_preserves_partial_evidence(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(args: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", fail)
    output = tmp_path / "measured"
    with pytest.raises(subprocess.CalledProcessError):
        collection.observe(config, output)
    assert (output / "batch-0000/request.json").is_file()
    assert not (output / "provenance.json").exists()


def test_workflow_declares_current_collection_inputs_and_domain_verifier(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "workflow.json"
    evidence = tmp_path / "measured"
    monkeypatch.setattr(
        sys,
        "argv",
        ["collection", "workflow", str(config), str(output), "--evidence", str(evidence)],
    )
    collection.main()
    stages = load_stages(output)
    assert len(stages) == 1
    assert str(config) in stages[0].command
    assert str(tmp_path / "sandbox") in {str(path) for path in stages[0].inputs}
    assert stages[0].verify_command[3] == "verify"
    monkeypatch.setattr(
        sys, "argv", ["collection", "workflow", str(config), str(tmp_path / "bad.json")]
    )
    with pytest.raises(SystemExit):
        collection.main()


def test_binding_preserves_original_clock_geometry_and_unapproved_support(
    config: Path, tmp_path: Path, native: list[tuple[str, ...]]
) -> None:
    output = tmp_path / "measured"
    collection.observe(config, output)
    provenance = output / "provenance.json"
    binding = {
        "config": {"path": str(config), "sha256": file_digest(config)},
        "provenance": {"path": str(provenance), "sha256": file_digest(provenance)},
    }
    arc = collection.read_image_binding(
        binding, source_hash=file_digest(tmp_path / "source"), width=100, height=200
    )
    assert arc.cells[0].start == Fraction(0) and arc.cells[0].end == Fraction(1, 30)
    for source_hash, width, height in (
        ("a" * 64, 100, 200),
        (file_digest(tmp_path / "source"), 101, 200),
    ):
        with pytest.raises(ReplayError, match="source or coordinate"):
            collection.read_image_binding(
                binding, source_hash=source_hash, width=width, height=height
            )
    with pytest.raises(ReplayError, match="pinned config"):
        collection.read_image_binding({}, source_hash="a" * 64, width=100, height=200)
