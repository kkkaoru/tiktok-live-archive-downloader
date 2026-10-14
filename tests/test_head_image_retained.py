"""Historical upper pixels keep exact raw clocks and partial support semantics."""

import copy
import json
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import batched
from pathlib import Path

import pytest
from test_head_image_support import frame as frame

from replay import head_image_collection as collection
from replay import head_image_retained as retained
from replay.core import ReplayError
from replay.fcp_visible_arc import VisibleArc
from replay.foreground_observations import MAX_SAMPLES
from replay.head_image_plan import ImageFrameCell
from replay.recordings import record
from replay.stages import file_digest


@dataclass(frozen=True)
class Fixture:
    config: Path
    output: Path
    descriptor: Path
    source: Path
    archive: Path

    def binding(self) -> dict[str, object]:
        return {
            "config": pin(self.config),
            "provenance": pin(self.output / "provenance.json"),
            "producerEvidence": pin(self.descriptor),
        }

    def read(self) -> retained.RetainedUpperPixels:
        return retained.read_image_binding(
            self.binding(), source_hash=file_digest(self.source), width=100, height=200
        )


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: dict[str, object]) -> Fixture:
    monkeypatch.setattr(retained, "ROOT", tmp_path)
    code = tmp_path / "src" / "replay"
    code.mkdir(parents=True)
    _ = [(code / name).write_bytes(name.encode()) for name in retained.READER_FILES]
    monkeypatch.setattr(collection, "__file__", str(code / "head_image_collection.py"))
    work = tmp_path / "edits" / "work"
    work.mkdir(parents=True)
    source, native, sandbox, archive = (
        work / name for name in ("source", "native", "sandbox", "stages.bytes")
    )
    _ = [path.write_bytes(path.name.encode()) for path in (source, native, sandbox)]
    native.chmod(0o700)
    archive.write_bytes(b"raise RuntimeError('archival code must never execute')\n")
    config = work / "config.json"
    declared = {str(path): file_digest(path) for path in (source, native, sandbox)}
    declared.update(
        {str(code / name): file_digest(code / name) for name in retained.PRODUCER_FILES}
    )
    declared[str(code / "stages.py")] = file_digest(archive)
    write(
        config,
        {
            "version": 1,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": file_digest(native),
            "sandbox": str(sandbox),
            "sandboxSHA256": file_digest(sandbox),
            "width": 100,
            "height": 200,
            "cells": [
                ImageFrameCell(
                    Fraction(index, 30),
                    Fraction(index + 1, 30),
                    None if index == 8 else (20, 110, 20, 20),
                ).encoded()
                for index in range(10)
            ],
            "inputs": declared,
        },
    )
    current = collection.load_job(config)
    historical = current.inputs | {str(code / "stages.py"): file_digest(archive)}
    job = replace(current, inputs=historical)
    output = work / "observed"
    output.mkdir()
    batches = tuple(batched((cell for cell in job.cells if cell.face is not None), MAX_SAMPLES))

    def batch(index: int, cells: tuple[ImageFrameCell, ...]) -> Path:
        directory = output / f"batch-{index:04d}"
        directory.mkdir()
        write(directory / "request.json", job.request(cells))

        def row(cell: ImageFrameCell) -> dict[str, object]:
            result = copy.deepcopy(frame)
            result["requested"] = result["actual"] = {
                "value": cell.start.numerator,
                "timescale": cell.start.denominator,
            }
            if cell.start == Fraction(9, 30):
                head = record(result["headTop"])
                support = record(head["upperSupport"])
                support["status"] = "canvasClipped"
            return result

        write(directory / "response.json", [row(cell) for cell in cells])
        (directory / "stderr.log").write_bytes(b"")
        return directory

    directories = [batch(index, cells) for index, cells in enumerate(batches)]
    assert len(directories) == 2
    original, _arc = collection._receipt(job, output)
    write(output / "inputs.json", historical)
    write(output / "provenance.json", original)
    descriptor = work / "descriptor.json"
    write(
        descriptor,
        {
            "version": 1,
            "protocol": retained.PROTOCOL,
            "archives": [
                {
                    "sourcePath": str(code / "stages.py"),
                    "path": str(archive),
                    "sha256": file_digest(archive),
                }
            ],
            "measurement": {"config": pin(config), "provenance": pin(output / "provenance.json")},
        },
    )
    return Fixture(config, output, descriptor, source, archive)


def test_archived_code_keeps_exact_cells_and_partial_raw_status(fixture: Fixture) -> None:
    provenance = (fixture.output / "provenance.json").read_bytes()
    with pytest.raises(ReplayError, match="provenance differs"):
        collection.read_arc(fixture.config, fixture.output)
    verified = fixture.read()
    assert len(verified.arc.cells) == 10 and len(verified.batches) == 2
    assert verified.arc.cells[8].visible_top is None
    assert verified.arc.cells[9].visible_top is None
    assert verified.arc.cells[9].unknown == 3
    assert verified.arc.cells[0].visible_top == 100
    assert verified.arc.cells[0].start == Fraction(0)
    assert verified.arc.cells[-1].end == Fraction(1, 3)
    raw = retained.read_response(verified.batches[0])
    assert isinstance(raw, list)
    assert record(record(raw[0])["headTop"])["status"] == "clippedSupport"
    receipt = verified.receipt()
    assert receipt["semanticHeadVerified"] is False
    assert receipt["captionPlacementApproved"] is False
    assert receipt["historicalExecutionReissued"] is False
    assert receipt["supportedFrames"] == 8
    assert verified.producer_code[str(retained.ROOT / "src/replay/stages.py")] == file_digest(
        fixture.archive
    )
    assert receipt["historicalProducerCode"] != receipt["currentReaderCode"]
    assert (fixture.output / "provenance.json").read_bytes() == provenance
    assert verified.arc.inputs is verified.inputs


@pytest.mark.parametrize(
    "target",
    [
        "source",
        "native",
        "sandbox",
        "archive",
        "request",
        "response",
        "inputs",
        "diagnostic",
        "reader",
        "planCode",
    ],
)
def test_changed_protected_bytes_cannot_reuse_upper_pixels(fixture: Fixture, target: str) -> None:
    binding = fixture.binding()
    source_hash = file_digest(fixture.source)
    paths = {
        "source": fixture.source,
        "native": fixture.source.with_name("native"),
        "sandbox": fixture.source.with_name("sandbox"),
        "archive": fixture.archive,
        "request": fixture.output / "batch-0000/request.json",
        "response": fixture.output / "batch-0000/response.json",
        "inputs": fixture.output / "inputs.json",
        "diagnostic": fixture.output / "batch-0000/stderr.log",
        "reader": retained.ROOT / "src/replay/head_image_support.py",
        "planCode": retained.ROOT / "src/replay/fcp_caption_retarget.py",
    }
    path = paths[target]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, ReplayError)):
        retained.read_image_binding(binding, source_hash=source_hash, width=100, height=200)


@pytest.mark.parametrize(
    "patch",
    [
        {"nativeExitCode": True},
        {"frames": 1},
        {"candidateOnly": False},
        {"semanticHeadVerified": True},
        {"captionPlacementApproved": True},
        {"extra": True},
        {"batches": []},
    ],
)
def test_forged_provenance_cannot_promote_support(
    fixture: Fixture, patch: dict[str, object]
) -> None:
    path = fixture.output / "provenance.json"
    write(path, record(json.loads(path.read_text())) | patch)
    descriptor = record(json.loads(fixture.descriptor.read_text()))
    descriptor["measurement"] = {"config": pin(fixture.config), "provenance": pin(path)}
    write(fixture.descriptor, descriptor)
    with pytest.raises((ValueError, ReplayError)):
        fixture.read()


@pytest.mark.parametrize(
    "mode",
    [
        "missingEvidence",
        "extraBinding",
        "wrongMeasurement",
        "badProvName",
        "wrongGeometry",
        "wrongSource",
        "relative",
        "outside",
        "badPin",
        "unexecutable",
    ],
)
def test_explicit_binding_cannot_escape_original_artifact(fixture: Fixture, mode: str) -> None:
    binding = fixture.binding()
    if mode == "missingEvidence":
        binding.pop("producerEvidence")
    elif mode == "extraBinding":
        binding["headVerified"] = True
    elif mode == "wrongMeasurement":
        descriptor = record(json.loads(fixture.descriptor.read_text()))
        descriptor["measurement"] = {
            "config": pin(fixture.config),
            "provenance": pin(fixture.config),
        }
        write(fixture.descriptor, descriptor)
        binding = fixture.binding()
    elif mode == "badProvName":
        binding["provenance"] = pin(fixture.config)
    elif mode == "relative":
        binding["config"] = {"path": "config.json", "sha256": "a" * 64}
    elif mode == "outside":
        path = retained.ROOT / "outside.json"
        path.write_bytes(fixture.config.read_bytes())
        binding["config"] = pin(path)
    elif mode == "badPin":
        binding["config"] = {"path": str(fixture.config), "sha256": "a" * 64}
    elif mode == "unexecutable":
        fixture.source.with_name("native").chmod(0o600)
    with pytest.raises((ValueError, ReplayError)):
        retained.read_image_binding(
            binding,
            source_hash="a" * 64 if mode == "wrongSource" else file_digest(fixture.source),
            width=101 if mode == "wrongGeometry" else 100,
            height=200,
        )


def test_raw_response_is_rechecked_when_seed_consumer_reads_it(fixture: Fixture) -> None:
    verified = fixture.read()
    batch = verified.batches[0]
    batch.response.write_bytes(batch.response.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact changed"):
        retained.read_response(batch)


def test_terminal_boundary_catches_change_after_measurement_validation(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = collection._receipt

    def changed(job: collection.ImageJob, output: Path) -> tuple[dict[str, object], VisibleArc]:
        result = original(job, output)
        fixture.source.write_bytes(b"changed after raw measurement validation")
        return result

    monkeypatch.setattr(collection, "_receipt", changed)
    with pytest.raises(ReplayError, match="changed"):
        fixture.read()


def test_new_archive_pin_cannot_replace_original_producer_code(fixture: Fixture) -> None:
    fixture.archive.write_bytes(b"another producer")
    descriptor = record(json.loads(fixture.descriptor.read_text()))
    descriptor["archives"] = [
        {
            "sourcePath": str(retained.ROOT / "src/replay/stages.py"),
            "path": str(fixture.archive),
            "sha256": file_digest(fixture.archive),
        }
    ]
    write(fixture.descriptor, descriptor)
    with pytest.raises(ValueError, match="original upper-pixel code pin"):
        fixture.read()


def test_receipt_is_exclusive_and_old_receipts_are_unchanged(fixture: Fixture) -> None:
    verified = fixture.read()
    path = fixture.output / "retained.json"
    retained.write_receipt(path, verified)
    assert record(json.loads(path.read_text()))["deliveryComplete"] is False
    with pytest.raises(FileExistsError):
        retained.write_receipt(path, verified)
    with pytest.raises(ValueError, match="existing repository work parents"):
        retained.write_receipt(retained.ROOT / "not-work.json", verified)


def test_cli_uses_pinned_consumer_config(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    path = fixture.output / "consumer.json"
    write(
        path,
        {
            "binding": fixture.binding(),
            "sourceSHA256": file_digest(fixture.source),
            "width": 100,
            "height": 200,
        },
    )
    receipt = fixture.output / "cli.json"
    monkeypatch.setattr(
        sys, "argv", ["retained", str(path), str(receipt), "--sha256", file_digest(path)]
    )
    retained.main()
    assert record(record(json.loads(receipt.read_text()))["inputs"])[str(path)] == file_digest(path)
    write(path, {"skip": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retained",
            str(path),
            str(fixture.output / "invalid.json"),
            "--sha256",
            file_digest(path),
        ],
    )
    with pytest.raises(ValueError, match="config fields differ"):
        retained.main()


@pytest.mark.parametrize("location", ["outside", "not-work", "relative"])
def test_cli_rejects_unscoped_config_before_any_read(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    path = {
        "outside": str(retained.ROOT.parent / "outside.json"),
        "not-work": str(retained.ROOT / "config.json"),
        "relative": "edits/work/config.json",
    }[location]

    def unexpected_read(_path: Path, _digest: str, _limit: int = 0) -> object:
        pytest.fail("Unscoped consumer config must be refused before opening it")

    monkeypatch.setattr(retained, "_json", unexpected_read)
    monkeypatch.setattr(
        sys, "argv", ["retained", path, str(fixture.output / "never.json"), "--sha256", "0" * 64]
    )
    with pytest.raises(ValueError, match="repository"):
        retained.main()
