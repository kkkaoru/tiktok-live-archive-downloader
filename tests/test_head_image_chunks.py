import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_head_image_collection import config as config
from test_head_image_collection import frame as frame
from test_head_image_collection import native as native

from replay import head_image_chunks as chunks
from replay import head_image_collection as collection
from replay.core import ReplayError
from replay.head_image_plan import ImageFrameCell, image_cells
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import _uncached_file_digest, execution_lock, file_digest


@pytest.fixture
def manifest(config: Path, tmp_path: Path) -> Path:
    raw = record(json.loads(config.read_text(encoding="utf-8")))
    model = tmp_path / "model.bin"
    model.write_bytes(b"pinned model")
    raw["inputs"] = record(raw["inputs"]) | {str(model): file_digest(model)}
    config.write_text(json.dumps(raw), encoding="utf-8")
    cells = image_cells(raw["cells"], width=100, height=200)
    second = tmp_path / "second.json"
    second.write_text(
        json.dumps(
            raw
            | {
                "cells": [
                    ImageFrameCell(
                        cell.start + Fraction(1, 3), cell.end + Fraction(1, 3), cell.face
                    ).encoded()
                    for cell in cells
                ]
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "ranges": [{"start": "0", "end": "2/3"}],
                "chunks": [
                    {
                        "config": {"path": str(config), "sha256": file_digest(config)},
                        "boundary": "first",
                        "reuse": None,
                    },
                    {
                        "config": {"path": str(second), "sha256": file_digest(second)},
                        "boundary": "budget",
                        "reuse": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_limit_resume_crypto_boundaries_and_unknowns(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = file_digest
    source_hash_calls: list[Path] = []

    def digest(path: Path) -> str:
        if path == tmp_path / "source":
            source_hash_calls.append(path)
        return original(path)

    monkeypatch.setattr(chunks, "file_digest", digest)
    output = tmp_path / "collected"
    first = chunks.collect(manifest, output, limit=1)
    assert (first["complete"], first["frames"], len(native), len(source_hash_calls)) == (
        False,
        10,
        2,
        2,
    )
    assert not (output / "summary.json").exists()
    prior = (output / "chunk-0000/provenance.json").read_bytes()
    second = chunks.collect(manifest, output, limit=1)
    assert (second["complete"], second["frames"], len(native), len(source_hash_calls)) == (
        True,
        20,
        4,
        4,
    )
    assert (output / "chunk-0000/provenance.json").read_bytes() == prior
    arc = chunks.verify(manifest, output)
    assert len(source_hash_calls) == 6
    assert len(arc.cells) == 20 and arc.cells[8].visible_top is None
    assert (arc.cells[8].start, arc.cells[8].end) == (Fraction(4, 15), Fraction(3, 10))
    assert arc.cells[18].visible_top is None
    assert not arc.covers(Fraction(0), Fraction(2, 3))
    assert json.loads((output / "summary.json").read_text())["captionPlacementApproved"] is False


def test_completed_resume_rehashes_even_after_same_bytes_replacement(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]]
) -> None:
    output = tmp_path / "collected"
    chunks.collect(manifest, output)
    summary = (output / "summary.json").read_bytes()
    source = tmp_path / "source"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(source.read_bytes())
    replacement.replace(source)
    assert chunks.collect(manifest, output)["complete"] is True
    assert len(native) == 4 and (output / "summary.json").read_bytes() == summary
    assert len(chunks.verify(manifest, output).cells) == 20


def test_outer_digest_cache_cannot_replace_new_run_crypto_boundaries(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _uncached_file_digest
    reads: list[Path] = []
    source = tmp_path / "source"

    def digest(path: Path) -> str:
        if path == source:
            reads.append(path)
        return original(path)

    monkeypatch.setattr(chunks, "file_digest", digest)
    cache = InputHashCache((source,))
    cache.prime()
    with input_hash_context(cache):
        assert chunks.collect(manifest, tmp_path / "collected")["complete"] is True
    assert reads == [source, source] and cache.hash_reads == 1 and len(native) == 4


def test_manifest_pins_all_direct_helpers(manifest: Path) -> None:
    inputs = chunks.load_manifest(manifest).inputs
    assert {
        "head_image_chunks.py",
        "artifact_json.py",
        "foreground_evidence.py",
        "input_hash_cache.py",
        "vertical_captions.py",
    } <= {Path(name).name for name in inputs}


@pytest.mark.parametrize("boundary", ["source-cut", "zoom"])
def test_current_cells_preserve_explicit_cut_and_zoom_labels(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]], boundary: str
) -> None:
    raw = json.loads(manifest.read_text())
    raw["chunks"][1]["boundary"] = boundary
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    result = chunks.collect(manifest, tmp_path / "collected")
    rows = result["chunks"]
    assert isinstance(rows, list)
    assert record(rows[1])["boundary"] == boundary
    assert len(native) == 4 and chunks.load_manifest(manifest).frames == 20


def test_explicit_source_gap_is_not_interpolated(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]]
) -> None:
    second = tmp_path / "second.json"
    raw = json.loads(second.read_text())
    cells = image_cells(raw["cells"], width=100, height=200)
    raw["cells"] = [
        ImageFrameCell(cell.start + 1, cell.end + 1, cell.face).encoded() for cell in cells
    ]
    second.write_text(json.dumps(raw), encoding="utf-8")
    data = json.loads(manifest.read_text())
    data["chunks"][1]["config"]["sha256"] = file_digest(second)
    data["chunks"][1]["boundary"] = "source-gap"
    data["ranges"] = [{"start": "0", "end": "1/3"}, {"start": "4/3", "end": "5/3"}]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    chunks.collect(manifest, tmp_path / "collected")
    arc = chunks.verify(manifest, tmp_path / "collected")
    assert arc.at(Fraction(1)) is None
    assert len(native) == 4 and len(arc.cells) == 20


@pytest.mark.parametrize(
    "patch", [{"version": 2}, {"chunks": []}, {"chunks": None}, {"extra": True}]
)
def test_invalid_manifest_schema(manifest: Path, patch: dict[str, object]) -> None:
    manifest.write_text(json.dumps(json.loads(manifest.read_text()) | patch), encoding="utf-8")
    with pytest.raises(ReplayError, match="manifest"):
        chunks.load_manifest(manifest)


@pytest.mark.parametrize("boundary", [None, [], "unknown", "first", "source-gap"])
def test_wrong_boundaries_cannot_hide_holes_or_reset_history(
    manifest: Path, boundary: object
) -> None:
    raw = json.loads(manifest.read_text())
    raw["chunks"][1]["boundary"] = boundary
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        chunks.load_manifest(manifest)


@pytest.mark.parametrize(
    "ranges",
    [
        [],
        [{"start": "0", "end": "1"}],
        [{"start": "1", "end": "0"}],
        [{"start": "0", "end": "1"}, {"start": "1/2", "end": "2"}],
        [{"start": "0", "end": "2/3", "ignored": True}],
    ],
)
def test_declared_coverage_rejects_holes_outside_and_overlap(
    manifest: Path, ranges: object
) -> None:
    raw = json.loads(manifest.read_text())
    raw["ranges"] = ranges
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        chunks.load_manifest(manifest)


@pytest.mark.parametrize("change", ["duplicate", "hole", "overlap", "oversize"])
def test_invalid_actual_cells_are_not_repaired(manifest: Path, tmp_path: Path, change: str) -> None:
    second = tmp_path / "second.json"
    raw = json.loads(second.read_text())
    if change == "duplicate":
        raw["cells"] = json.loads((tmp_path / "collection.json").read_text())["cells"]
    elif change == "hole":
        del raw["cells"][2]
    elif change == "overlap":
        raw["cells"][1]["start"] = raw["cells"][0]["start"]
        raw["cells"][1]["face"] = None
    else:
        raw["cells"] *= 13
    second.write_text(json.dumps(raw), encoding="utf-8")
    data = json.loads(manifest.read_text())
    data["chunks"][1]["config"]["sha256"] = file_digest(second)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        chunks.load_manifest(manifest)


@pytest.mark.parametrize("change", ["sourceSHA256", "nativeSHA256", "sandboxSHA256", "width"])
def test_chunk_identity_does_not_mix_sources_geometry_or_backends(
    manifest: Path, tmp_path: Path, change: str
) -> None:
    second = tmp_path / "second.json"
    raw = json.loads(second.read_text())
    if change == "width":
        raw[change] = 101
    else:
        raw[change] = "a" * 64
        key = {
            "sourceSHA256": "source",
            "nativeSHA256": "nativeExecutable",
            "sandboxSHA256": "sandbox",
        }[change]
        raw["inputs"][raw[key]] = "a" * 64
    second.write_text(json.dumps(raw), encoding="utf-8")
    data = json.loads(manifest.read_text())
    data["chunks"][1]["config"]["sha256"] = file_digest(second)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="share exact"):
        chunks.load_manifest(manifest)


@pytest.mark.parametrize("target", ["source", "model.bin", "nativeExecutable", "sandbox"])
def test_new_run_checks_all_crypto_dependencies(
    manifest: Path, tmp_path: Path, target: str
) -> None:
    (tmp_path / target).write_bytes(b"changed")
    with pytest.raises(ReplayError, match="content changed"):
        chunks.collect(manifest, tmp_path / "collected")


def test_same_run_stat_cache_rejects_changed_source_even_if_content_restored(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = chunks._collect

    def changed(job: collection.ImageJob, evidence: Path) -> None:
        original(job, evidence)
        source = tmp_path / "source"
        before = source.stat()
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 100))

    monkeypatch.setattr(chunks, "_collect", changed)
    with pytest.raises(ReplayError, match="Run-local"):
        chunks.collect(manifest, tmp_path / "collected")
    assert len(native) == 2 and not (tmp_path / "collected/summary.json").exists()
    assert not (tmp_path / "collected/chunk-0000/provenance.json").exists()


def test_additional_helper_identity_is_checked_before_provenance_publication(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    collect = chunks._collect
    stamp = chunks._stamp
    changed: list[bool] = []

    def raw(job: collection.ImageJob, evidence: Path) -> None:
        collect(job, evidence)
        changed.append(True)

    def observed(path: Path) -> chunks.FileStamp:
        result = stamp(path)
        if changed and path.name == "artifact_json.py":
            return replace(result, changed_ns=result.changed_ns + 1)
        return result

    monkeypatch.setattr(chunks, "_collect", raw)
    monkeypatch.setattr(chunks, "_stamp", observed)
    with pytest.raises(ReplayError, match="Run-local"):
        chunks.collect(manifest, tmp_path / "collected")
    assert len(native) == 2 and not (tmp_path / "collected/chunk-0000/provenance.json").exists()


@pytest.mark.parametrize(
    "target", ["response", "provenance", "inputs", "request", "diagnostic", "summary"]
)
def test_resume_domain_verifies_every_completed_raw_reference(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]], target: str
) -> None:
    output = tmp_path / "collected"
    chunks.collect(manifest, output)
    paths = {
        "response": "chunk-0000/batch-0000/response.json",
        "provenance": "chunk-0000/provenance.json",
        "inputs": "chunk-0000/inputs.json",
        "request": "chunk-0000/batch-0000/request.json",
        "diagnostic": "chunk-0000/batch-0000/stderr.log",
        "summary": "summary.json",
    }
    (output / paths[target]).write_text("{}", encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        chunks.verify(manifest, output)
    assert len(native) == 4


@pytest.mark.parametrize(
    "patch",
    [
        {"version": 2},
        {"version": True},
        {"extra": True},
        {"sourceStat": {}},
        {"sourceStat": {"device": 1, "inode": 2, "size": 3, "modifiedNS": 4, "changedNS": -1}},
        {"sourceStat": {"device": 1, "inode": 2, "size": True, "modifiedNS": 4, "changedNS": 5}},
    ],
)
def test_summary_schema_and_historical_identity_are_not_ignored(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]], patch: dict[str, object]
) -> None:
    output = tmp_path / "collected"
    chunks.collect(manifest, output)
    summary = output / "summary.json"
    summary.write_text(json.dumps(json.loads(summary.read_text()) | patch), encoding="utf-8")
    with pytest.raises(ReplayError, match="schema|historical"):
        chunks.verify(manifest, output)
    assert len(native) == 4


def test_noncurrent_native_clock_is_rejected(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: dict[str, object]
) -> None:
    def run(args: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        request = json.loads(Path(args[-1]).read_text())
        rows = [
            copy.deepcopy(frame) | {"requested": clock, "actual": {"value": 99, "timescale": 30}}
            for clock in request["samples"]
        ]
        Path(args[-1]).with_name("response.json").write_text(json.dumps(rows), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((ReplayError, ValueError)):
        chunks.collect(manifest, tmp_path / "collected")
    assert not (tmp_path / "collected/chunk-0000/provenance.json").exists()


def test_immutable_external_chunk_can_be_reused_without_native_duplication(
    manifest: Path, config: Path, tmp_path: Path, native: list[tuple[str, ...]]
) -> None:
    old = tmp_path / "existing"
    collection.observe(config, old)
    prior = (old / "provenance.json").read_bytes()
    raw = json.loads(manifest.read_text())
    raw["chunks"][0]["reuse"] = {
        "path": str(old / "provenance.json"),
        "sha256": file_digest(old / "provenance.json"),
    }
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    result = chunks.collect(manifest, tmp_path / "collected")
    assert result["complete"] is True and len(native) == 4
    assert not (tmp_path / "collected/chunk-0000").exists()
    assert (old / "provenance.json").read_bytes() == prior
    assert len(chunks.verify(manifest, tmp_path / "collected").cells) == 20


def test_incomplete_native_attempt_is_retained_and_not_retried(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(args: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", failed)
    output = tmp_path / "collected"
    with pytest.raises(subprocess.CalledProcessError):
        chunks.collect(manifest, output)
    with pytest.raises(ReplayError, match="Incomplete chunk retained"):
        chunks.collect(manifest, output)
    assert (output / "chunk-0000/batch-0000/request.json").is_file()
    assert not (output / "summary.json").exists()


def test_concurrent_owner_is_not_interrupted(manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "collected"
    with execution_lock(output, "current-image-chunks") as owned:
        assert owned is True
        with pytest.raises(ReplayError, match="already owned"):
            chunks.collect(manifest, output)


@pytest.mark.parametrize("limit", [0, -1, True, 2049])
def test_invalid_limits_rejected(manifest: Path, tmp_path: Path, limit: int) -> None:
    with pytest.raises(ReplayError, match="limit"):
        chunks.collect(manifest, tmp_path / "collected", limit=limit)


def test_cli_check_and_collect_verify(
    manifest: Path,
    tmp_path: Path,
    native: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["chunks", "check", str(manifest)])
    chunks.main()
    assert json.loads(capsys.readouterr().out)["frames"] == 20
    output = tmp_path / "collected"
    monkeypatch.setattr(sys, "argv", ["chunks", "collect", str(manifest), str(output)])
    chunks.main()
    assert json.loads(capsys.readouterr().out) == {
        "complete": True,
        "frames": 20,
        "candidateOnly": True,
    }
    monkeypatch.setattr(sys, "argv", ["chunks", "verify", str(manifest), str(output)])
    chunks.main()
    assert (
        capsys.readouterr().out == "Verified 20 current cells; semantic head remains unapproved\n"
    )
    assert len(native) == 4


def test_cli_requires_output(manifest: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["chunks", "collect", str(manifest)])
    with pytest.raises(SystemExit) as error:
        chunks.main()
    assert error.value.code == 2


def test_first_boundary_and_dependency_conflict_are_explicit(
    manifest: Path, tmp_path: Path
) -> None:
    raw = json.loads(manifest.read_text())
    raw["chunks"][0]["boundary"] = "budget"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="First chunk"):
        chunks.load_manifest(manifest)
    raw["chunks"][0]["boundary"] = "first"
    second = tmp_path / "second.json"
    config = json.loads(second.read_text())
    config["inputs"][str(tmp_path / "model.bin")] = "a" * 64
    second.write_text(json.dumps(config), encoding="utf-8")
    raw["chunks"][1]["config"]["sha256"] = file_digest(second)
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="Conflicting"):
        chunks.load_manifest(manifest)


@pytest.mark.parametrize("constant", ["MAX_CHUNKS", "MAX_INPUTS", "MAX_TOTAL_CELLS"])
def test_global_manifest_budgets_remain_bounded(
    manifest: Path, monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    monkeypatch.setattr(chunks, constant, 1)
    with pytest.raises(ReplayError):
        chunks.load_manifest(manifest)


def test_native_must_be_executable_before_any_collection(manifest: Path, tmp_path: Path) -> None:
    (tmp_path / "nativeExecutable").chmod(0o600)
    with pytest.raises(ReplayError, match="not executable"):
        chunks.collect(manifest, tmp_path / "collected")


def test_linked_evidence_is_never_adopted(manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "collected"
    output.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (output / "chunk-0000").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        chunks.collect(manifest, output)


@pytest.mark.parametrize("output", [Path.home(), chunks.WORK_ROOT, chunks.WORK_ROOT.parent])
def test_chunk_outputs_cannot_escape_repository_scratch(manifest: Path, output: Path) -> None:
    with pytest.raises(ReplayError, match="repository edits/work"):
        chunks.collect(manifest, output)


def test_stat_snapshot_rejects_nonregular_inputs(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="regular"):
        chunks._stamp(tmp_path)


@pytest.mark.parametrize("mode", ["symlink", "file", "exhausted"])
def test_immutable_run_ledger_rejects_links_and_budget_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    output = tmp_path / "collected"
    output.mkdir()
    runs = output / "runs"
    if mode == "symlink":
        runs.symlink_to(tmp_path, target_is_directory=True)
    else:
        runs.mkdir()
        (runs / "run-0000").write_bytes(b"occupied")
    if mode == "exhausted":
        (runs / "run-0000").unlink()
        (runs / "run-0000").mkdir()
        monkeypatch.setattr(chunks, "MAX_RUNS", 1)
    with pytest.raises(ReplayError):
        chunks._run_directory(output)


def test_wrong_reuse_provenance_basename_rejected(manifest: Path, tmp_path: Path) -> None:
    existing = tmp_path / "receipt.json"
    existing.write_bytes(b"{}")
    raw = json.loads(manifest.read_text())
    raw["chunks"][0]["reuse"] = {"path": str(existing), "sha256": file_digest(existing)}
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="provenance"):
        chunks.load_manifest(manifest)


def test_resume_namespace_pin_and_summary_are_immutable(
    manifest: Path, tmp_path: Path, native: list[tuple[str, ...]]
) -> None:
    output = tmp_path / "collected"
    chunks.collect(manifest, output)
    (output / "summary.json").write_bytes(b"{}")
    with pytest.raises(ReplayError, match="Completed summary changed"):
        chunks.collect(manifest, output)
    (output / "manifest-inputs.json").write_bytes(b"{}")
    with pytest.raises(ReplayError, match="Resume namespace"):
        chunks.collect(manifest, output)
    assert len(native) == 4
