"""Synthetic batch boundaries and real readonly domains; no Apple applications."""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from test_caption_resource_inputs import Chain

from replay import (
    caption_linguistics,
    caption_measured_job,
    caption_resource_inputs,
    digest_store,
    foreground_collection,
    stages,
    workflow,
)
from replay import (
    measurement_digest_batch as batch,
)
from replay.core import ReplayError
from replay.input_hash_cache import Entry, InputHashCache, input_hash_context
from replay.recordings import record

pytest_plugins = ["test_caption_resource_inputs"]


@dataclass
class Case:
    configuration: Path
    source: Path
    native: Path
    job: Path
    output: Path

    def load(self) -> batch.Batch:
        return batch.load(self.configuration, stages.file_digest(self.configuration))


@dataclass
class CaptionCase:
    batch: batch.Batch
    manifest: dict[str, object]
    candidate: dict[str, object]

    def publish(self) -> Path:
        output = self.batch.units[0].output
        output.mkdir()
        path = output / "candidate.json"
        write(path, self.candidate)
        return path


class Holder(ModuleType):
    def __init__(self) -> None:
        super().__init__("replay.synthetic_scope_holder")
        self.file_digest = stages.file_digest


def write(path: Path, value: object) -> batch.Reference:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return batch.Reference(path, stages.file_digest(path))


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": stages.file_digest(path)}


@pytest.fixture
def case(tmp_path: Path) -> Case:
    source, native = tmp_path / "source.bin", tmp_path / "synthetic-native"
    source.write_bytes(b"synthetic bytes")
    native.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[3]).read_text())\n"
        "print(json.dumps([{'requested': s, 'actual': s, 'width': 16, 'height': 16, "
        "'observationPresent': False, 'instanceLabels': None, 'unionMask': None, "
        "'maskThreshold8Bit': 1, 'captionPlacementApproved': False} "
        "for s in request['samples']]))\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    job = tmp_path / "foreground.json"
    write(
        job,
        {
            "version": 1,
            "source": str(source),
            "sourceSHA256": stages.file_digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": stages.file_digest(native),
            "clock": {"pts": [0, 1], "timeBase": "1/30", "lastDuration": 1},
            "width": 16,
            "height": 16,
        },
    )
    paths = (
        set(batch.code_paths())
        | {batch.ROOT / "src/replay" / name for name in foreground_collection.CODE_FILES}
        | {source, native, job}
    )
    configuration, output = tmp_path / "batch.json", tmp_path / "foreground-output"
    write(
        configuration,
        {
            "version": 1,
            "inputs": [reference(path) for path in sorted(paths)],
            "units": [
                {
                    "kind": "foreground",
                    "configuration": reference(job),
                    "output": str(output),
                    "resources": None,
                }
            ],
            "store": str(tmp_path / "store"),
            "state": str(tmp_path / "state"),
            "receipt": str(tmp_path / "receipt.json"),
        },
    )
    return Case(configuration, source, native, job, output)


@pytest.fixture
def caption_case(case: Case, chain: Chain, monkeypatch: pytest.MonkeyPatch) -> CaptionCase:
    foreground_collection.observe(case.job, case.output)
    root = case.configuration.parent
    tokens = root / "tokens.json"
    write(
        tokens,
        {
            "version": 1,
            "clock": "output",
            "tokens": [{"text": "abc", "start": 0, "end": 0.03, "utterance": "synthetic"}],
        },
    )
    chain_config = record(json.loads(chain.config.read_text(encoding="utf-8")))
    motion = record(chain_config["motion"])
    evidence = record(motion["resourceEvidence"])
    resource_path = root / "font-resources.json"
    write(resource_path, {"motion": motion, "evidence": evidence})
    calibration = root / "calibration.json"
    write(
        calibration,
        {"template": reference(chain.root / "template.moti"), "evidence": [reference(tokens)]},
    )
    manifest_path = root / "caption-manifest.json"
    manifest: dict[str, object] = {
        "version": 1,
        "tokens": reference(tokens),
        "spans": [
            {
                "sourceStart": {"value": 0, "timescale": 30},
                "outputStart": {"value": 0, "timescale": 30},
                "duration": {"value": 2, "timescale": 30},
            }
        ],
        "foreground": [
            {
                "config": reference(case.job),
                "provenance": reference(case.output / "provenance.json"),
            }
        ],
        "native": reference(case.native),
        "calibration": reference(calibration),
        "outputFrameRate": 30,
        "region": [0, 0, 16, 16],
        "centerX": 8,
        "clearance": 1,
    }
    write(manifest_path, manifest)
    dictionary = root / "dictionary.bin"
    dictionary.write_bytes(b"synthetic morphology asset")
    monkeypatch.setattr(caption_linguistics, "DICTIONARY_PATH", dictionary)
    monkeypatch.setattr(caption_linguistics, "DICTIONARY_SHA256", stages.file_digest(dictionary))
    resources = caption_resource_inputs.ResourceInputs()
    caption_resource_inputs._font_evidence(resources, motion, evidence, {"abc"})
    foreground_inputs = record(
        json.loads((case.output / "provenance.json").read_text(encoding="utf-8"))
    )
    paths = (
        set(batch.code_paths())
        | {Path(name) for name in record(foreground_inputs["inputs"])}
        | set(resources.pins)
        | {
            manifest_path,
            resource_path,
            tokens,
            calibration,
            dictionary,
            chain.root / "template.moti",
            case.output / "provenance.json",
            case.output / "batch-0000/request.json",
            case.output / "batch-0000/response.json",
        }
        | {batch.ROOT / "src/replay" / name for name in caption_measured_job.CODE_FILES}
    )
    configuration = root / "caption-batch.json"
    output = root / "caption-output"
    write(
        configuration,
        {
            "version": 1,
            "inputs": [reference(path) for path in sorted(paths)],
            "units": [
                {
                    "kind": "caption",
                    "configuration": reference(manifest_path),
                    "output": str(output),
                    "resources": reference(resource_path),
                }
            ],
            "store": str(root / "caption-store"),
            "state": str(root / "caption-state"),
            "receipt": str(root / "caption-receipt.json"),
        },
    )
    loaded = batch.load(configuration, stages.file_digest(configuration))
    candidate: dict[str, object] = {
        "version": 1,
        "candidateOnly": True,
        "inputs": {str(item.path): item.sha256 for item in loaded.inputs},
        "canvas": [16, 16],
        "policy": {"frame_rate": 30},
        "coordinateSystem": "top-left canvas; calibrated text origin",
        "candidates": [{"cue": {"text": "abc", "start": 0, "end": 0.03}, "x": 8, "y": 2}],
        "editorCorrespondenceVerified": False,
        "completeMotionSupportVerified": False,
        "motionRenderVerified": False,
        "semanticPlacementApproved": False,
        "captionGeometryApproved": False,
        "aaaContrastApproved": False,
        "deliveryComplete": False,
    }
    return CaptionCase(loaded, manifest, candidate)


def test_original_caption_readonly_domain_accepts_unapproved_measured_data(
    caption_case: CaptionCase,
) -> None:
    path = caption_case.publish()
    result = batch.verify_unit(caption_case.batch, 0)
    assert result["kind"] == "caption"
    assert result["provisional"] is True
    assert path.exists()
    assert not caption_case.batch.receipt.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"candidates": []},
        {"canvas": [17, 16]},
        {"policy": {"frame_rate": 60}},
        {"motionRenderVerified": True},
    ],
)
def test_original_caption_invalid_domain_cannot_be_accepted(
    caption_case: CaptionCase, change: dict[str, object]
) -> None:
    caption_case.candidate.update(change)
    caption_case.publish()
    with pytest.raises((ValueError, ReplayError)):
        batch.verify_unit(caption_case.batch, 0)
    assert not caption_case.batch.receipt.exists()


def test_caption_closure_growth_cannot_admit_undeclared_media(caption_case: CaptionCase) -> None:
    path = caption_case.batch.configuration.path.parent / "undeclared.bin"
    path.write_bytes(b"unknown source")
    inputs = record(caption_case.candidate["inputs"])
    inputs[str(path)] = stages.file_digest(path)
    caption_case.candidate["inputs"] = inputs
    caption_case.publish()
    with pytest.raises(ReplayError, match="undeclared preexisting"):
        batch.verify_unit(caption_case.batch, 0)


def test_caption_generated_growth_is_bounded_metadata_and_byte_verified(
    caption_case: CaptionCase,
) -> None:
    caption_case.publish()
    generated = caption_case.batch.units[0].output / "native-response.json"
    generated.write_bytes(b"generated")
    inputs = record(caption_case.candidate["inputs"])
    inputs[str(generated)] = stages.file_digest(generated)
    caption_case.candidate["inputs"] = inputs
    write(caption_case.batch.units[0].output / "candidate.json", caption_case.candidate)
    assert batch.verify_unit(caption_case.batch, 0)["provisional"] is True
    generated.write_bytes(b"changed response")
    with pytest.raises(ReplayError, match="evidence changed"):
        batch.verify_unit(caption_case.batch, 0)


def test_caption_child_invokes_original_plan_and_restores_context(
    caption_case: CaptionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    original = stages.file_digest

    def plan(manifest: Path, directory: Path, *, manifest_sha256: str) -> Path:
        calls.append(manifest)
        return caption_case.publish()

    monkeypatch.setattr(caption_measured_job, "plan", plan)
    batch.child(caption_case.batch, 0, verify_only=False)
    batch.child(caption_case.batch, 0, verify_only=True)
    assert len(calls) == 1
    assert stages.file_digest is original
    assert not caption_case.batch.receipt.exists()


def test_real_public_workflow_and_foreground_domain_complete_and_resume(case: Case) -> None:
    loaded = case.load()
    result = batch.run(loaded)
    value = record(json.loads(result.read_text(encoding="utf-8")))
    assert value["complete"] is True
    assert value["boundaryScope"] == "outer-batch"
    assert value["independentPerUnitFreshness"] is False
    assert value["qualityApproved"] is False
    assert value["deliveryApproved"] is False
    assert value["physicalIOBytes"] is None
    assert batch.run(loaded, resume=True) == result
    assert (case.output / "batch-0000/response.json").is_file()
    assert result.stat().st_mode & 0o777 == 0o600


def test_real_code_closure_is_bounded_and_includes_both_producers() -> None:
    paths = batch.code_paths()
    assert batch.ROOT / "src/replay/caption_measured_job.py" in paths
    assert batch.ROOT / "src/replay/foreground_collection.py" in paths
    assert batch.ROOT / "src/replay/input_hash_cache.py" in paths
    assert 1 <= len(paths) <= 512


def test_fresh_initial_tamper_prevents_all_child_effects(case: Case) -> None:
    loaded = case.load()
    case.source.write_bytes(b"tampered bytes!")
    with pytest.raises(ReplayError, match="fresh initial"):
        batch.run(loaded)
    assert not loaded.state.exists()
    assert not case.output.exists()
    assert not loaded.receipt.exists()


def test_warm_cache_cannot_replace_fresh_initial_bytes(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()
    monkeypatch.setattr(digest_store, "MINIMUM_BYTES", 4)
    digest_store.cached_digest(loaded.store, case.source)
    case.source.write_bytes(b"changed source")
    with pytest.raises(ReplayError, match="fresh initial"):
        batch.run(loaded)
    assert not case.output.exists()


def test_generated_artifacts_stay_fresh_and_unknown_paths_are_rejected(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()
    case.output.mkdir()
    generated = case.output / "generated.json"
    generated.write_bytes(b"original generated")
    foreign = case.output.parent / "undeclared.bin"
    foreign.write_bytes(b"foreign")
    monkeypatch.setattr(digest_store, "MINIMUM_BYTES", 4)
    with batch._router(loaded, loaded.units[0]) as counts:
        assert stages.file_digest(case.source) == hashlib.sha256(b"synthetic bytes").hexdigest()
        assert stages.file_digest(generated) == hashlib.sha256(b"original generated").hexdigest()
        generated.write_bytes(b"changed generated")
        assert stages.file_digest(generated) == hashlib.sha256(b"changed generated").hexdigest()
        with pytest.raises(ReplayError, match="undeclared"):
            stages.file_digest(foreign)
    assert counts["cacheCalls"] == 1
    assert counts["generatedFreshCalls"] == 2
    assert counts["cacheByteReads"] == 1
    assert counts["cacheBytesRead"] == 15


def test_scoped_aliases_restore_even_on_child_exception(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()
    original = stages.file_digest
    holder = Holder()
    monkeypatch.setitem(sys.modules, holder.__name__, holder)
    with (
        pytest.raises(ValueError, match="synthetic failure"),
        batch._router(loaded, loaded.units[0]),
    ):
        assert stages.file_digest is not original
        assert holder.file_digest is not original
        raise ValueError("synthetic failure")
    assert stages.file_digest is original
    assert holder.file_digest is original


def test_child_cache_identity_invalidation_rejects_mutation(case: Case) -> None:
    loaded = case.load()
    with batch._router(loaded, loaded.units[0]):
        case.source.write_bytes(b"changed source")
        with pytest.raises(ReplayError, match="identity changed"):
            stages.file_digest(case.source)


def test_child_cache_corruption_cannot_acquire_acceptance(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()

    def wrong(_store: Path, _path: Path, *, fresh: bool = False) -> str:
        return "0" * 64

    monkeypatch.setattr(digest_store, "cached_digest", wrong)
    with batch._router(loaded, loaded.units[0]), pytest.raises(ReplayError, match="pin changed"):
        stages.file_digest(case.source)
    assert not loaded.receipt.exists()


def test_original_invalid_foreground_domain_is_not_accepted(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()
    foreground_collection.observe(case.job, case.output)
    (case.output / "batch-0000/stderr.log").write_bytes(b"decoder warning")
    with pytest.raises(ValueError, match="diagnostics"):
        batch.verify_unit(loaded, 0)
    assert not loaded.receipt.exists()


def test_completed_outputs_are_reconciled_by_domain_without_rerender(case: Case) -> None:
    loaded = case.load()
    foreground_collection.observe(case.job, case.output)
    response = (case.output / "batch-0000/response.json").read_bytes()
    batch.run(loaded, resume=True)
    assert (case.output / "batch-0000/response.json").read_bytes() == response


def test_partial_output_is_preserved_without_blind_rerender(case: Case) -> None:
    loaded = case.load()
    case.output.mkdir()
    (case.output / "partial.json").write_bytes(b"partial evidence")
    with pytest.raises(stages.StageError):
        batch.run(loaded, resume=True)
    assert (case.output / "partial.json").read_bytes() == b"partial evidence"
    assert not loaded.receipt.exists()


def test_child_failure_retains_provisional_artifacts_only(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()

    def failed(_plan: Path, *, state_directory: Path, targets: tuple[str, ...] = ()) -> None:
        case.output.mkdir()
        (case.output / "partial.json").write_bytes(b"failed child")
        raise stages.StageError("synthetic child failure")

    monkeypatch.setattr(workflow, "run_plan", failed)
    with pytest.raises(stages.StageError, match="synthetic child"):
        batch.run(loaded)
    assert not loaded.receipt.exists()
    assert (case.output / "partial.json").read_bytes() == b"failed child"


def test_terminal_mutation_after_valid_domains_prevents_complete_receipt(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()
    original = batch.verify_unit

    def changed(value: batch.Batch, number: int) -> dict[str, object]:
        result = original(value, number)
        case.source.write_bytes(b"terminal source")
        return result

    monkeypatch.setattr(batch, "verify_unit", changed)
    with pytest.raises(ReplayError, match="changed"):
        batch.run(loaded)
    assert not loaded.receipt.exists()


def test_resume_rechecks_source_and_completed_output(case: Case) -> None:
    loaded = case.load()
    batch.run(loaded)
    old = loaded.receipt.read_bytes()
    (case.output / "batch-0000/response.json").write_bytes(b"altered output")
    with pytest.raises((ValueError, ReplayError)):
        batch.run(loaded, resume=True)
    assert loaded.receipt.read_bytes() == old


def test_occupied_receipt_does_not_authorize_automatic_reentry(case: Case) -> None:
    loaded = case.load()
    batch.run(loaded)
    with pytest.raises(ReplayError, match="explicit readonly resume"):
        batch.run(loaded)


def test_fresh_boundary_counts_are_independent_of_warm_crossprocess_cache(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = case.load()
    monkeypatch.setattr(digest_store, "MINIMUM_BYTES", 4)
    cold: list[Path] = []
    original_hash = digest_store.hash_file

    def hash_file(path: Path) -> str:
        if path == case.source:
            cold.append(path)
        return original_hash(path)

    monkeypatch.setattr(digest_store, "hash_file", hash_file)
    with batch._router(loaded, loaded.units[0]):
        stages.file_digest(case.source)
        stages.file_digest(case.source)
    assert cold == [case.source]
    fresh: list[bool] = []
    original_read = InputHashCache._read

    def read(cache: InputHashCache, path: Path, entry: Entry, *, fresh: bool) -> str:
        if path == case.source:
            observed.append(fresh)
        return original_read(cache, path, entry, fresh=fresh)

    observed = fresh
    monkeypatch.setattr(InputHashCache, "_read", read)
    cache = batch._scope(loaded)
    with input_hash_context(cache):
        assert stages.file_digest(case.source) == hashlib.sha256(b"synthetic bytes").hexdigest()
        cache.verify()
    assert fresh == [True, False, True]
    assert cold == [case.source]


@pytest.mark.parametrize(
    "change", [{"version": True}, {"version": 2}, {"extra": 1}, {"units": []}, {"inputs": []}]
)
def test_invalid_batch_metadata_never_produces_units(case: Case, change: dict[str, object]) -> None:
    raw = record(json.loads(case.configuration.read_text(encoding="utf-8")))
    write(case.configuration, raw | change)
    with pytest.raises(ReplayError):
        case.load()
    assert not case.output.exists()


@pytest.mark.parametrize(
    "invalid",
    [None, {}, {"path": "relative", "sha256": "0" * 64}, {"path": "/path", "sha256": "bad"}],
)
def test_invalid_reference_shapes_are_rejected(invalid: object) -> None:
    with pytest.raises(ReplayError):
        batch._reference(invalid)


@pytest.mark.parametrize("value", [None, "relative", "/absolute/../escape", "bad\x00path"])
def test_path_aliases_and_escaping_syntax_are_rejected(value: object) -> None:
    with pytest.raises(ReplayError):
        batch._path(value)


def test_unlisted_missing_and_unrelated_input_roles_are_rejected(case: Case) -> None:
    raw = record(json.loads(case.configuration.read_text(encoding="utf-8")))
    inputs = raw["inputs"]
    assert isinstance(inputs, list)
    raw["inputs"] = [value for value in inputs if record(value)["path"] != str(case.source)]
    write(case.configuration, raw)
    with pytest.raises(ReplayError, match="transitive pin"):
        case.load()


def test_unknown_child_number_never_runs_a_producer(case: Case) -> None:
    loaded = case.load()
    with pytest.raises(ReplayError, match="Unknown"):
        batch.child(loaded, -1, verify_only=False)
    with pytest.raises(ReplayError, match="Unknown"):
        batch.verify_unit(loaded, 8)
    assert not case.output.exists()


def test_cli_runs_the_same_generic_parent_api(
    case: Case, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measurement_digest_batch",
            "run",
            str(case.configuration),
            stages.file_digest(case.configuration),
        ],
    )
    batch.main()
    assert str(case.configuration.parent / "receipt.json") in capsys.readouterr().out


def test_metadata_is_byte_bound_nonempty_regular_and_bounded(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ReplayError, match="SHA256"):
        batch._document(batch.Reference(case.configuration, "0" * 64))
    monkeypatch.setattr(batch, "MAX_JSON", 4)
    with pytest.raises(ReplayError, match="budget"):
        batch._document(batch.Reference(case.configuration, stages.file_digest(case.configuration)))


def test_generated_metadata_is_exclusive_and_bounded(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = case.configuration.parent / "result.json"
    batch._write(path, {"retained": True})
    with pytest.raises(FileExistsError):
        batch._write(path, {"retained": False})
    monkeypatch.setattr(batch, "MAX_JSON", 4)
    with pytest.raises(ReplayError, match="budget"):
        batch._write(path.parent / "oversize.json", {"large": "value"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"retained": True}


def test_two_real_child_processes_share_only_provisional_large_input_cache(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = case.load()
    with case.source.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024)
    job_data = record(json.loads(case.job.read_text(encoding="utf-8")))
    write(case.job, job_data | {"sourceSHA256": stages.file_digest(case.source)})
    raw = record(json.loads(case.configuration.read_text(encoding="utf-8")))
    raw["inputs"] = [
        reference(item.path) for item in base.inputs if item.path != case.configuration
    ]
    raw["units"] = [
        {
            "kind": "foreground",
            "configuration": reference(case.job),
            "output": str(case.output),
            "resources": None,
        },
        {
            "kind": "foreground",
            "configuration": reference(case.job),
            "output": str(case.output.parent / "second-output"),
            "resources": None,
        },
    ]
    write(case.configuration, raw)
    loaded = case.load()
    observed: list[bool] = []
    original_read = InputHashCache._read

    def read(cache: InputHashCache, path: Path, entry: Entry, *, fresh: bool) -> str:
        if path == case.source:
            observed.append(fresh)
        return original_read(cache, path, entry, fresh=fresh)

    monkeypatch.setattr(InputHashCache, "_read", read)
    batch.run(loaded)
    first = record(json.loads((case.output / "cache-observation.json").read_text(encoding="utf-8")))
    second = record(
        json.loads(
            (case.output.parent / "second-output/cache-observation.json").read_text(
                encoding="utf-8"
            )
        )
    )
    first_counts, second_counts = record(first["counts"]), record(second["counts"])
    first_bytes, second_bytes = first_counts["cacheBytesRead"], second_counts["cacheBytesRead"]
    assert isinstance(first_bytes, int) and isinstance(second_bytes, int)
    assert first_bytes - second_bytes == 64 * 1024 * 1024
    assert sum(observed) == 2
    assert first["physicalIOBytes"] is None
    assert first["provisional"] is True


def test_duplicate_and_recursive_pins_are_rejected(case: Case) -> None:
    raw = record(json.loads(case.configuration.read_text(encoding="utf-8")))
    inputs = raw["inputs"]
    assert isinstance(inputs, list)
    write(case.configuration, raw | {"inputs": [*inputs, inputs[0]]})
    with pytest.raises(ReplayError, match="Duplicate"):
        case.load()


@pytest.mark.parametrize(
    "change",
    [{"kind": "unknown"}, {"extra": 1}, {"resources": {"path": "/bad", "sha256": "0" * 64}}],
)
def test_unit_roles_cannot_escape_the_original_backend(
    case: Case, change: dict[str, object]
) -> None:
    raw = record(json.loads(case.configuration.read_text(encoding="utf-8")))
    units = raw["units"]
    assert isinstance(units, list)
    raw["units"] = [record(units[0]) | change]
    write(case.configuration, raw)
    with pytest.raises(ReplayError):
        case.load()


@pytest.mark.parametrize("change", ["nested", "input", "missing-parent", "unrelated"])
def test_storage_roles_and_exact_closure_are_required(case: Case, change: str) -> None:
    raw = record(json.loads(case.configuration.read_text(encoding="utf-8")))
    if change == "nested":
        raw["state"] = str(case.output / "nested")
    elif change == "input":
        raw["receipt"] = str(case.job)
    elif change == "missing-parent":
        raw["store"] = str(case.output.parent / "missing/store")
    else:
        foreign = case.output.parent / "unrelated.bin"
        foreign.write_bytes(b"extra")
        inputs = raw["inputs"]
        assert isinstance(inputs, list)
        raw["inputs"] = [*inputs, reference(foreign)]
    write(case.configuration, raw)
    with pytest.raises(ReplayError):
        case.load()


def test_failed_scope_never_rewrites_occupied_workflow(case: Case) -> None:
    loaded = case.load()
    loaded.state.mkdir()
    (loaded.state / "workflow.json").write_bytes(b'{"different":true}')
    with pytest.raises(ReplayError, match="workflow differs"):
        batch.run(loaded, resume=True)
    assert (loaded.state / "workflow.json").read_bytes() == b'{"different":true}'


def test_busy_batch_lease_does_not_launch_units(case: Case) -> None:
    loaded = case.load()
    with stages.execution_lock(loaded.state, "measurement-batch") as acquired:
        assert acquired
        with pytest.raises(ReplayError, match="Another process"):
            batch.run(loaded, resume=True)
    assert not case.output.exists()


def test_changed_outer_receipt_cannot_be_reapproved_by_resume(case: Case) -> None:
    loaded = case.load()
    batch.run(loaded)
    saved = record(json.loads(loaded.receipt.read_text(encoding="utf-8")))
    write(loaded.receipt, saved | {"qualityApproved": True})
    with pytest.raises(ReplayError, match="receipt differs"):
        batch.run(loaded, resume=True)


def test_links_empty_inputs_and_outside_work_outputs_reject(case: Case) -> None:
    link = case.output.parent / "link"
    link.symlink_to(case.source)
    with pytest.raises(ReplayError, match="canonical"):
        batch._path(str(link))
    empty = case.output.parent / "empty"
    empty.touch()
    with pytest.raises(ReplayError, match="nonempty"):
        batch._identity(empty)
    with pytest.raises(ReplayError, match="work storage"):
        batch._path(str(batch.ROOT / "outside-work"), work=True)


def test_duplicate_and_nonfinite_json_reject_before_production(case: Case) -> None:
    reference = write(case.output.parent / "duplicate.json", {})
    reference.path.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ReplayError, match="Duplicate"):
        batch._document(batch.Reference(reference.path, stages.file_digest(reference.path)))
    reference.path.write_bytes(b'{"x":Infinity}')
    with pytest.raises(ReplayError, match="Non-finite"):
        batch._document(batch.Reference(reference.path, stages.file_digest(reference.path)))


def test_generated_nonmetadata_is_not_a_new_cache_allowlist(case: Case) -> None:
    loaded = case.load()
    case.output.mkdir()
    media = case.output / "unknown.mp4"
    media.write_bytes(b"not generated metadata")
    with (
        batch._router(loaded, loaded.units[0]),
        pytest.raises(ReplayError, match="bounded metadata"),
    ):
        stages.file_digest(media)


def test_readonly_cli_and_child_cli_use_the_same_domains(
    case: Case, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loaded = case.load()
    batch.run(loaded)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measurement_digest_batch",
            "verify",
            str(case.configuration),
            loaded.configuration.sha256,
        ],
    )
    batch.main()
    assert '"complete": true' in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measurement_digest_batch",
            "child-verify",
            str(case.configuration),
            loaded.configuration.sha256,
            "0",
        ],
    )
    batch.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measurement_digest_batch",
            "child-run",
            str(case.configuration),
            loaded.configuration.sha256,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        batch.main()


def test_retained_caption_producer_archives_remain_data_only(caption_case: CaptionCase) -> None:
    loaded = caption_case.batch
    archive = loaded.configuration.path.parent / "producer-archive.py"
    original = batch.ROOT / "src/replay/foreground_collection.py"
    archive.write_bytes(original.read_bytes())
    descriptor = loaded.configuration.path.parent / "retained-reader.json"
    write(
        descriptor,
        {
            "version": 1,
            "protocol": "foreground-collection-v1",
            "archives": [{"role": "foreground_collection.py", **reference(archive)}],
        },
    )
    foreground = caption_case.manifest["foreground"]
    assert isinstance(foreground, list)
    caption_case.manifest["foreground"] = [
        record(foreground[0]) | {"retainedVerification": reference(descriptor)}
    ]
    write(loaded.units[0].configuration.path, caption_case.manifest)
    raw = record(json.loads(loaded.configuration.path.read_text(encoding="utf-8")))
    raw["inputs"] = [
        reference(item.path) for item in loaded.inputs if item.path != loaded.configuration.path
    ] + [reference(archive), reference(descriptor)]
    units = raw["units"]
    assert isinstance(units, list)
    raw["units"] = [
        record(value) | {"configuration": reference(loaded.units[0].configuration.path)}
        for value in units
    ]
    write(loaded.configuration.path, raw)
    refreshed = batch.load(loaded.configuration.path, stages.file_digest(loaded.configuration.path))
    caption_case.candidate["inputs"] = {str(item.path): item.sha256 for item in refreshed.inputs}
    caption_case.publish()
    assert batch.verify_unit(refreshed, 0)["provisional"] is True
    assert archive.read_bytes() == original.read_bytes()


def test_output_parent_replacement_cannot_redirect_publication(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = case.output.parent / "publication"
    parent.mkdir()
    outside = case.output.parent / "other"
    outside.mkdir()
    original = os.open

    def opened(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original(path, flags, mode, dir_fd=dir_fd)
        if isinstance(path, (Path, str)) and Path(path) == parent:
            parent.rename(parent.with_name("retained-publication"))
            parent.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(os, "open", opened)
    with pytest.raises(ReplayError, match="canonical"):
        batch._write(parent / "receipt.json", {"complete": True})
    assert not (outside / "receipt.json").exists()
    assert not (parent.with_name("retained-publication") / "receipt.json").exists()
