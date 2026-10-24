"""Synthetic integrated candidate jobs; native execution is always mocked."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from replay import caption_linguistics
from replay import caption_measured_job as job
from replay import foreground_collection as collection
from replay import foreground_retained as retained
from replay import tracking_observations as storage
from replay.caption_io import _array, _fields, _string
from replay.caption_native_layout import NativeCaptionLayout
from replay.core import ReplayError
from replay.motion_native_envelope import CalibratedMotionEnvelope
from replay.stages import StageError, file_digest


def fake_native(command: tuple[str, ...], output: Path, timeout: int) -> None:
    assert command[0] == "/usr/bin/env"
    assert command[-3:-1] == ("measure-media", "caption_text_measure")
    assert 1 <= timeout <= 60
    assert command[1].startswith("TMPDIR=")
    assert command[2].startswith("TMP=")
    assert command[3].startswith("TEMP=")
    assert command[4].startswith("XDG_CACHE_HOME=")
    request = _fields(json.loads(Path(command[-1]).read_text(encoding="utf-8")))
    texts = tuple(_string(value) for value in _array(request["texts"], 16))
    storage._write(
        output,
        {
            "motionRenderVerified": False,
            "font": {
                "postScriptName": "Synthetic",
                "family": "Synthetic",
                "face": "Regular",
                "pointSize": 7.5,
                "lineAdvance": 10,
                "lineAscent": 9,
                "renderedAscent": 8,
            },
            "bounds": [
                {
                    "text": text,
                    "width": len(text) * 10 + 4,
                    "height": 12,
                    "lineWidths": [len(text) * 10 + 4],
                }
                for text in texts
            ],
        },
    )


@pytest.fixture(autouse=True)
def isolated_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        caption_linguistics, "_atomic_ends", lambda text: frozenset(range(1, len(text) + 1))
    )
    monkeypatch.setattr(caption_linguistics, "semantic_break", lambda previous, following: False)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must never start a native process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(job, "execute_command", fake_native)
    monkeypatch.setattr(job, "ROOT", tmp_path)
    # The real pinned dictionary stays reachable from tmp_path (over-wide words are tokenized).
    dictionary = caption_linguistics.DICTIONARY_PATH.resolve()
    monkeypatch.setattr(caption_linguistics, "DICTIONARY_PATH", dictionary)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        caption_linguistics,
        "boundaries",
        lambda text: (
            tuple(range(2, len(text) + 1, 2)),
            tuple(range(1, len(text) + 1)),
        ),
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    work = tmp_path / "edits" / "work"
    work.mkdir(parents=True)
    source, native = work / "source.bin", work / "native"
    source.write_bytes(b"synthetic source; no media")
    native.write_bytes(b"synthetic executable; never execute")
    native.chmod(0o700)
    config = work / "foreground.json"
    storage._write(
        config,
        {
            "version": 1,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": file_digest(native),
            "clock": {"pts": [100, 101, 107], "timeBase": "1/10", "lastDuration": 2},
            "width": 100,
            "height": 200,
        },
    )
    observations = work / "observations"
    batch = observations / "batch-0000"
    batch.mkdir(parents=True)
    foreground_job = collection.load_job(config)
    storage._write(batch / "request.json", foreground_job.request((100, 101, 107)))
    storage._write(
        batch / "response.json",
        [
            {
                "requested": {"value": tick, "timescale": 10},
                "actual": {"value": tick, "timescale": 10},
                "width": 100,
                "height": 200,
                "observationPresent": True,
                "instanceLabels": [1],
                "unionMask": {
                    "width": 100,
                    "height": 200,
                    "nonzeroPixels": 20,
                    "nonzeroBounds": {"x": 0, "y": 30, "width": 100, "height": 170},
                },
                "maskThreshold8Bit": 1,
                "captionPlacementApproved": False,
            }
            for tick in (100, 101, 107)
        ],
    )
    (batch / "stderr.log").write_bytes(b"")
    foreground_inputs = collection._bindings(foreground_job)
    storage._write(observations / "inputs.json", foreground_inputs)
    provenance = observations / "provenance.json"
    storage._write(provenance, collection._receipt(foreground_job, observations, foreground_inputs))
    template, evidence = work / "template.moti", work / "calibration-evidence.json"
    template.write_bytes(b"synthetic template; no Motion project")
    evidence.write_bytes(b'{"synthetic":true}')
    calibration = work / "calibration.json"
    storage._write(
        calibration,
        {
            "version": 1,
            "canvas": [100, 200],
            "padding": 2,
            "style": {
                "font_name": "Synthetic",
                "ass_font_size": 10,
                "border": 1,
                "rim": 0.5,
                "motion_line_advance": 14,
            },
            "nativeFont": {
                "postScriptName": "Synthetic",
                "family": "Synthetic",
                "face": "Regular",
                "pointSize": 7.5,
                "lineAdvance": 10,
                "lineAscent": 9,
                "renderedAscent": 8,
            },
            "template": {"path": str(template), "sha256": file_digest(template)},
            "evidence": [{"path": str(evidence), "sha256": file_digest(evidence)}],
        },
    )
    tokens = work / "tokens.json"
    storage._write(
        tokens,
        {
            "version": 1,
            "clock": "output",
            "tokens": [
                {"text": "aa", "start": 0, "end": 0.15, "utterance": "a"},
                {"text": "bb", "start": 0.2, "end": 0.35, "utterance": "b"},
                {"text": "cc", "start": 0.4, "end": 0.5, "utterance": "b"},
            ],
        },
    )
    manifest = work / "manifest.json"
    storage._write(
        manifest,
        {
            "version": 1,
            "tokens": {"path": str(tokens), "sha256": file_digest(tokens)},
            "spans": [
                {
                    "sourceStart": {"value": 100, "timescale": 10},
                    "outputStart": {"value": 0, "timescale": 1},
                    "duration": {"value": 1, "timescale": 5},
                },
                {
                    "sourceStart": {"value": 104, "timescale": 10},
                    "outputStart": {"value": 1, "timescale": 5},
                    "duration": {"value": 2, "timescale": 5},
                },
            ],
            "foreground": [
                {
                    "config": {"path": str(config), "sha256": file_digest(config)},
                    "provenance": {"path": str(provenance), "sha256": file_digest(provenance)},
                }
            ],
            "native": {"path": str(native), "sha256": file_digest(native)},
            "calibration": {"path": str(calibration), "sha256": file_digest(calibration)},
            "outputFrameRate": 30,
            "region": [0, 0, 50, 200],
            "centerX": 25,
            "clearance": 4,
        },
    )
    return manifest, work / "candidate-run"


def test_retained_descriptor_connects_to_real_planner(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, directory = inputs
    monkeypatch.setattr(retained, "ROOT", tmp_path)
    archive = manifest.parent / "stages.bytes"
    archive.write_bytes(Path(collection.__file__).with_name("stages.py").read_bytes())
    descriptor = manifest.parent / "retained.json"
    storage._write(
        descriptor,
        {
            "version": 1,
            "protocol": "foreground-collection-v1",
            "archives": [
                {"role": "stages.py", "path": str(archive), "sha256": file_digest(archive)}
            ],
        },
    )
    raw = _fields(json.loads(manifest.read_bytes()))
    reference = _fields(_array(raw["foreground"], 32)[0])
    reference["retainedVerification"] = {"path": str(descriptor), "sha256": file_digest(descriptor)}
    raw["foreground"] = [reference]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    result = job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert result.is_file()
    receipts = _fields(json.loads((directory / "retained-verification.json").read_bytes()))
    assert _fields(_array(receipts["windows"], 32)[0])["historicalExecutionReissued"] is False
    assert _fields(json.loads(result.read_bytes()))["deliveryComplete"] is False


def test_job_batches_native_phrase_alternatives(inputs: tuple[Path, Path]) -> None:
    manifest, directory = inputs
    raw = _fields(json.loads(manifest.read_text(encoding="utf-8")))
    tokens_path = manifest.parent / "tokens.json"
    tokens_path.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "tokens": [{"text": "abcdefghijkl", "start": 0, "end": 0.1, "utterance": "one"}],
            }
        ),
        encoding="utf-8",
    )
    raw["tokens"] = {"path": str(tokens_path), "sha256": file_digest(tokens_path)}
    raw["region"] = [15, 0, 85, 200]
    raw["centerX"] = 50
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    candidate = job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert candidate.is_file()
    assert len(tuple((directory / "native").glob("*.response.json"))) == 2


def test_integrated_vfr_cut_mapping_and_private_candidate(
    inputs: tuple[Path, Path],
) -> None:
    manifest, directory = inputs
    result = job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    report = _fields(json.loads(result.read_text(encoding="utf-8")))
    assert report["candidates"] == [
        {"cue": {"text": "aa", "start": 0.0, "end": 0.2}, "x": 25.0, "y": 24.0},
        {"cue": {"text": "bb", "start": 0.2, "end": 0.4}, "x": 25.0, "y": 24.0},
        {"cue": {"text": "bbcc", "start": 0.4, "end": 0.58}, "x": 25.0, "y": 24.0},
    ]
    assert report["canvas"] == [100, 200]
    assert report["candidateOnly"] is True
    assert report["editorCorrespondenceVerified"] is False
    assert report["completeMotionSupportVerified"] is False
    assert report["motionRenderVerified"] is False
    assert report["semanticPlacementApproved"] is False
    assert report["captionGeometryApproved"] is False
    assert report["aaaContrastApproved"] is False
    assert report["deliveryComplete"] is False
    assert result.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / "scratch").stat().st_mode & 0o777 == 0o700
    assert (directory / "cache").stat().st_mode & 0o777 == 0o700
    assert (directory / "attempt.json").is_file()
    assert (directory / "native" / "batch-000000.request.json").is_file()
    assert (directory / "native" / "batch-000000.response.json").is_file()
    bindings = _fields(report["inputs"])
    assert bindings[str(manifest)] == file_digest(manifest)
    assert bindings[str(manifest.parent / "tokens.json")] == file_digest(
        manifest.parent / "tokens.json"
    )
    assert bindings[str(manifest.parent / "observations" / "provenance.json")] == file_digest(
        manifest.parent / "observations" / "provenance.json"
    )


def test_pinned_existing_source_can_remain_outside_checkout(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = inputs[0]
    root = manifest.parent
    source = tmp_path / "existing-original.bin"
    source.write_bytes(b"synthetic original on separate storage")
    config = root / "foreground.json"
    raw_config = _fields(json.loads(config.read_text(encoding="utf-8")))
    raw_config["source"] = str(source)
    raw_config["sourceSHA256"] = file_digest(source)
    config.write_text(json.dumps(raw_config), encoding="utf-8")
    foreground_job = collection.load_job(config)
    observations = root / "observations"
    (observations / "batch-0000" / "request.json").write_text(
        json.dumps(foreground_job.request((100, 101, 107))),
        encoding="utf-8",
    )
    bindings = collection._bindings(foreground_job)
    (observations / "inputs.json").write_text(json.dumps(bindings), encoding="utf-8")
    provenance = observations / "provenance.json"
    provenance.write_text(
        json.dumps(collection._receipt(foreground_job, observations, bindings)),
        encoding="utf-8",
    )
    raw = _fields(json.loads(manifest.read_text(encoding="utf-8")))
    raw["foreground"] = [
        {
            "config": {"path": str(config), "sha256": file_digest(config)},
            "provenance": {"path": str(provenance), "sha256": file_digest(provenance)},
        }
    ]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(job, "ROOT", root)
    monkeypatch.chdir(root)
    work = root / "edits" / "work"
    work.mkdir(parents=True)
    result = job.plan(manifest, work / "new-run", manifest_sha256=file_digest(manifest))
    report = _fields(json.loads(result.read_text(encoding="utf-8")))
    assert report["candidateOnly"] is True
    assert report["deliveryComplete"] is False
    assert source.read_bytes() == b"synthetic original on separate storage"
    assert _fields(report["inputs"])[str(source)] == file_digest(source)


def test_cli_and_existing_attempt_are_non_overwriting(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, directory = inputs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "caption_measured_job",
            str(manifest),
            str(directory),
            "--manifest-sha256",
            file_digest(manifest),
        ],
    )
    job.main()
    before = (directory / "candidate.json").read_bytes()
    with pytest.raises(FileExistsError):
        job.main()
    assert (directory / "candidate.json").read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("extra", "unrecognized"),
        ("outputFrameRate", 29.97),
        ("outputFrameRate", 121),
        ("region", [0, 0, 50]),
        ("region", [0, 0, 101, 200]),
        ("region", [0, 0, 0, 200]),
        ("centerX", 50),
        ("clearance", -1),
        ("foreground", []),
        ("spans", []),
        ("tokens", {"path": "relative.json", "sha256": "0" * 64}),
        ("native", {"path": "/outside-checkout/native", "sha256": "0" * 64}),
        ("calibration", {}),
    ],
)
def test_manifest_validation_precedes_output_creation(
    inputs: tuple[Path, Path], field: str, value: object
) -> None:
    manifest, directory = inputs
    raw = _fields(json.loads(manifest.read_text(encoding="utf-8")))
    raw[field] = value
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


@pytest.mark.parametrize(
    "value",
    [
        {"version": True, "clock": "output", "tokens": []},
        {"version": 1, "clock": "source", "tokens": []},
        {"version": 1, "clock": "output", "tokens": []},
        {
            "version": 1,
            "clock": "output",
            "tokens": [{"text": "aa", "start": True, "end": 0.1, "utterance": "a"}],
        },
        {
            "version": 1,
            "clock": "output",
            "tokens": [{"text": "aa", "start": 0.2, "end": 0.1, "utterance": "a"}],
        },
        {
            "version": 1,
            "clock": "output",
            "tokens": [{"text": "aa", "start": 0, "end": 0.1, "utterance": "a", "rate": 1}],
        },
    ],
)
def test_strict_output_token_document(inputs: tuple[Path, Path], value: object) -> None:
    manifest, directory = inputs
    tokens = manifest.parent / "tokens.json"
    tokens.write_text(json.dumps(value), encoding="utf-8")
    raw = _fields(json.loads(manifest.read_text(encoding="utf-8")))
    raw["tokens"] = {"path": str(tokens), "sha256": file_digest(tokens)}
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


@pytest.mark.parametrize(
    "value",
    [
        {"value": 1, "timescale": 0},
        {"value": True, "timescale": 1},
        {"value": 1, "timescale": 1, "nominalFPS": 30},
        "1/2",
        0.5,
    ],
)
def test_retained_clocks_require_exact_rational_objects(
    inputs: tuple[Path, Path], value: object
) -> None:
    manifest, directory = inputs
    raw = _fields(json.loads(manifest.read_text(encoding="utf-8")))
    spans = _array(raw["spans"], 3000)
    raw["spans"] = [
        dict(_fields(spans[0]), duration=value),
        spans[1],
    ]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


@pytest.mark.parametrize(
    "contents",
    [
        b'{"version":1,"version":1}',
        b'{"version":NaN}',
        b'{"version":Infinity}',
        b'{"version":',
        b"[]",
    ],
)
def test_strict_json(inputs: tuple[Path, Path], contents: bytes) -> None:
    manifest, directory = inputs
    manifest.write_bytes(contents)
    with pytest.raises(ValueError):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


def test_manifest_size_bound(inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, directory = inputs
    monkeypatch.setattr(storage, "CONFIG_BYTES", 32)
    with pytest.raises(ValueError, match="byte budget"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


def test_reference_size_bound(inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, directory = inputs
    monkeypatch.setattr(job, "MAX_INPUT_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


@pytest.mark.parametrize("pin", ["invalid", "0" * 64])
def test_manifest_pin_is_required(inputs: tuple[Path, Path], pin: str) -> None:
    manifest, directory = inputs
    with pytest.raises(ValueError):
        job.plan(manifest, directory, manifest_sha256=pin)
    assert not directory.exists()


def test_stale_independent_token_pin(inputs: tuple[Path, Path]) -> None:
    manifest, directory = inputs
    tokens = manifest.parent / "tokens.json"
    tokens.write_bytes(tokens.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Pinned candidate input changed"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


def test_working_directory_must_be_checkout(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, directory = inputs
    monkeypatch.chdir(manifest.parent)
    with pytest.raises(ValueError, match="checkout root"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


def test_output_must_be_work_storage(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    manifest, _directory = inputs
    with pytest.raises(ValueError, match="under edits/work"):
        job.plan(manifest, tmp_path / "public-output", manifest_sha256=file_digest(manifest))
    assert not (tmp_path / "public-output").exists()


def test_linked_output_is_rejected(inputs: tuple[Path, Path]) -> None:
    manifest, directory = inputs
    linked = manifest.parent / "linked"
    linked.symlink_to(manifest.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked tracking output"):
        job.plan(manifest, linked / directory.name, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


def test_foreground_reference_requires_receipt_name(inputs: tuple[Path, Path]) -> None:
    manifest, directory = inputs
    raw = _fields(json.loads(manifest.read_text(encoding="utf-8")))
    foreground = _fields(_array(raw["foreground"], 32)[0])
    raw["foreground"] = [dict(foreground, provenance=raw["tokens"])]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance.json"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not directory.exists()


def test_unknown_foreground_is_not_a_layout_fallback(inputs: tuple[Path, Path]) -> None:
    manifest, directory = inputs
    observations = manifest.parent / "observations"
    response = observations / "batch-0000" / "response.json"
    frames = _array(json.loads(response.read_text(encoding="utf-8")), 8)
    frames[0] = dict(
        _fields(frames[0]), observationPresent=False, instanceLabels=None, unionMask=None
    )
    response.write_text(json.dumps(frames), encoding="utf-8")
    foreground_job = collection.load_job(manifest.parent / "foreground.json")
    provenance = observations / "provenance.json"
    provenance.write_text(
        json.dumps(
            collection._receipt(foreground_job, observations, collection._bindings(foreground_job))
        ),
        encoding="utf-8",
    )
    raw = _fields(json.loads(manifest.read_text(encoding="utf-8")))
    foreground = _fields(_array(raw["foreground"], 32)[0])
    raw["foreground"] = [
        dict(foreground, provenance={"path": str(provenance), "sha256": file_digest(provenance)})
    ]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="support is unknown"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert (directory / "attempt.json").is_file()
    assert not (directory / "candidate.json").exists()
    assert not tuple((directory / "native").glob("*.request.json"))


def test_conflicting_pin_cannot_replace_an_earlier_binding(inputs: tuple[Path, Path]) -> None:
    manifest, directory = inputs
    bindings = {str(manifest): "0" * 64}
    with pytest.raises(ValueError, match="Conflicting candidate input bindings"):
        job._pin({"path": str(manifest), "sha256": file_digest(manifest)}, bindings)
    assert bindings[str(manifest)] == "0" * 64
    assert not directory.exists()


def test_failed_publication_retains_pending_not_success(
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directory = inputs

    def fail_link(source: Path, destination: Path) -> None:
        assert source.name == "candidate.pending.json"
        assert destination.name == "candidate.json"
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    pending = directory / "candidate.pending.json"
    assert pending.is_file()
    assert pending.stat().st_mode & 0o777 == 0o600
    assert _fields(json.loads(pending.read_text(encoding="utf-8")))["deliveryComplete"] is False
    assert not (directory / "candidate.json").exists()


def test_source_change_after_serialization_still_prevents_publication(
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directory = inputs
    original_save = NativeCaptionLayout._save

    def save_then_change_source(path: Path, value: object) -> None:
        original_save(path, value)
        if path.name == "candidate.pending.json":
            (manifest.parent / "source.bin").write_bytes(b"changed synthetic source")

    monkeypatch.setattr(NativeCaptionLayout, "_save", staticmethod(save_then_change_source))
    with pytest.raises((ValueError, ReplayError, StageError)):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert (directory / "candidate.pending.json").is_file()
    assert not (directory / "candidate.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "manifest.json",
        "tokens.json",
        "source.bin",
        "native",
        "calibration.json",
        "calibration-evidence.json",
        "template.moti",
        "observations/provenance.json",
        "observations/batch-0000/response.json",
    ],
)
def test_mutated_inputs_stop_success_and_preserve_measurements(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    manifest, directory = inputs

    def mutate(command: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(command, output, timeout)
        target = manifest.parent / name
        target.write_bytes(target.read_bytes() + b" ")

    monkeypatch.setattr(job, "execute_command", mutate)
    with pytest.raises((ValueError, ReplayError, StageError)):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert (directory / "attempt.json").is_file()
    assert (directory / "native" / "batch-000000.request.json").is_file()
    assert (directory / "native" / "batch-000000.response.json").is_file()
    assert not (directory / "candidate.json").exists()


def test_native_timeout_retains_partial_evidence(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, directory = inputs

    def timeout(command: tuple[str, ...], output: Path, seconds: int) -> None:
        storage._write(output, {"incomplete": True})
        raise subprocess.TimeoutExpired(command, seconds)

    monkeypatch.setattr(job, "execute_command", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert (directory / "native" / "batch-000000.request.json").is_file()
    assert json.loads(
        (directory / "native" / "batch-000000.response.json").read_text(encoding="utf-8")
    ) == {"incomplete": True}
    assert not (directory / "candidate.json").exists()


def test_native_font_mismatch_is_not_accepted(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, directory = inputs

    def mismatch(command: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(command, output, timeout)
        raw = _fields(json.loads(output.read_text(encoding="utf-8")))
        raw["font"] = dict(_fields(raw["font"]), pointSize=8)
        output.write_text(json.dumps(raw), encoding="utf-8")

    monkeypatch.setattr(job, "execute_command", mismatch)
    with pytest.raises(ValueError, match="Fresh native font metrics"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not (directory / "candidate.json").exists()


def test_native_call_budget(inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, directory = inputs
    monkeypatch.setattr(job, "MAX_NATIVE_CALLS", 0)
    with pytest.raises(ValueError, match="execution budget"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert (directory / "native" / "batch-000000.request.json").is_file()
    assert not (directory / "candidate.json").exists()


def test_deadline_before_native_execution(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, directory = inputs
    monkeypatch.setattr(job, "JOB_SECONDS", 0)
    with pytest.raises(ValueError, match="execution budget"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert not (directory / "candidate.json").exists()


def test_deadline_after_planning_prevents_success(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, directory = inputs
    clock = [0.0]
    monkeypatch.setattr(job, "monotonic", lambda: clock[0])
    original_verify = CalibratedMotionEnvelope.verify_inputs
    calls = 0

    def expire(envelope: CalibratedMotionEnvelope) -> dict[str, str]:
        nonlocal calls
        result = original_verify(envelope)
        calls += 1
        if calls == 2:
            clock[0] = 301.0
        return result

    monkeypatch.setattr(CalibratedMotionEnvelope, "verify_inputs", expire)
    with pytest.raises(ValueError, match="planning deadline"):
        job.plan(manifest, directory, manifest_sha256=file_digest(manifest))
    assert (directory / "native" / "batch-000000.response.json").is_file()
    assert not (directory / "candidate.json").exists()
