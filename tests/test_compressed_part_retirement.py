"""Synthetic lossy-part retirement and consumer restart, never private recordings."""

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import compressed_part_delivery as delivery
from replay import compressed_part_retirement as retirement
from replay import whole_technical_batch as metadata
from replay.core import ReplayError
from replay.native_export_receipt import save_new
from replay.recordings import record
from replay.stages import execution_lock
from replay.whole_technical_batch import Reference

REAL_ROOT = metadata.ROOT
REAL_RUN = subprocess.run


@dataclass
class Case:
    work: Path
    request: Path
    config: dict[str, object]
    source: Path
    movie: Path
    recipe: Path
    disposition: Path
    owner_state: Path
    acceptance: dict[str, object]
    intent: Path
    result: Path


def write(path: Path, raw: dict[str, object]) -> Reference:
    path.write_text(json.dumps(raw), encoding="utf-8")
    return Reference(path, hashlib.sha256(path.read_bytes()).hexdigest())


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    root = tmp_path / "checkout"
    work = root / "edits/work/part-retirement"
    work.mkdir(parents=True)
    monkeypatch.setattr(retirement, "ROOT", root)
    monkeypatch.setattr(metadata, "ROOT", root)
    state, owner = work / "state", work / "owner"
    state.mkdir()
    owner.mkdir()
    source, movie = work / "source.mov", work / "replacement.mp4"
    source.write_bytes(b"synthetic original ProRes bytes")
    movie.write_bytes(b"synthetic lossy HEVC bytes")
    clean, original, voice, motion, font = (
        work / name
        for name in ("clean.mov", "original.mkv", "voice.wav", "caption.moti", "font.ttf")
    )
    created = [
        path.write_bytes(b"retained native ingredient")
        for path in (clean, original, voice, motion, font)
    ]
    assert len(created) == 5
    xml = work / "part.fcpxml"
    xml.write_text(
        f'<fcpxml><resources><asset><media-rep src="{clean.as_uri()}"/></asset>'
        f'<asset><media-rep src="{voice.as_uri()}"/></asset></resources></fcpxml>',
        encoding="utf-8",
    )
    partition = write(
        work / "partition.json", {"version": 1, "parts": [{**ref(xml), "globalStart": "0s"}]}
    )
    cleanup = write(work / "cleanup.json", {"kind": "synthetic-local-repair", "complete": True})
    proof = write(
        work / "frame-proof.json",
        {
            "version": 1,
            "part": 1,
            "movie": ref(source),
            "partition": partition.json(),
            "observed": {"frames": 5},
        },
    )
    completion = write(
        work / "native-completion.json",
        {
            "version": 1,
            "kind": "verified-native-part",
            "part": 1,
            "movie": ref(source),
            "frameProof": proof.json(),
            "completionVerified": True,
            "shareObservedIdle": True,
            "fullAVDecodeExitCode": 0,
            "inputs": {str(source): ref(source)["sha256"], str(proof.path): proof.sha256},
        },
    )
    evidence = write(
        work / "hevc-evidence.json",
        {"kind": "synthetic-compressed-part-evidence", "source": ref(source), "movie": ref(movie)},
    )
    planner = write(work / "planner.json", {"version": 1, "part": 1})
    ingredients = [ref(path) for path in (clean, original, voice, motion, font, xml)]
    recipe = write(
        work / "recipe.json",
        {
            "version": 1,
            "kind": "native-part-rebuild-recipe",
            "part": 1,
            "source": ref(source),
            "cleanSource": ref(clean),
            "sourceOriginal": ref(original),
            "partXML": ref(xml),
            "partition": partition.json(),
            "processedVoice": ref(voice),
            "motionTemplate": ref(motion),
            "fontAsset": ref(font),
            "cleanupEvidence": cleanup.json(),
            "inputs": [*ingredients, partition.json(), cleanup.json()],
            "outputPolicy": "new-nonoverwriting-revision",
        },
    )
    disposition = write(
        work / "disposition.json",
        {
            "version": 1,
            "source": ref(source),
            "owner": "synthetic-owner",
            "ownerReleased": True,
            "observedAtNs": time.time_ns(),
            "references": {
                name: {
                    "scope": "synthetic bounded reference check",
                    "resolved": True,
                    "requiresMaterializedSource": False,
                    "evidence": [planner.json()],
                }
                for name in ("current", "queued", "resume", "FCP", "Viewer", "diagnostic")
            },
        },
    )
    instruction = work / "authorized-cleanup.txt"
    instruction.write_text(
        "Delete duplicate verified originals within the approved scope.", encoding="utf-8"
    )
    config: dict[str, object] = {
        "version": 1,
        "part": 1,
        "source": ref(source),
        "sourceBytes": source.stat().st_size,
        "planner": planner.json(),
        "native": {
            "completion": completion.json(),
            "frameProof": proof.json(),
            "publication": None,
            "stagingRetirement": None,
        },
        "recipe": recipe.json(),
        "disposition": disposition.json(),
        "state": str(state),
        "leases": [str(owner)],
        "approval": {
            "instruction": ref(instruction),
            "scope": "verified-hevc-part-source",
            "deletionAuthorized": True,
        },
    }
    request = work / "request.json"
    write(request, config)
    acceptance: dict[str, object] = {
        "version": 1,
        "kind": "accepted-compressed-part",
        "part": 1,
        "source": ref(source),
        "movie": ref(movie),
        "evidence": evidence.json(),
        "accepted": True,
        "inputs": [ref(source), ref(movie), evidence.json(), planner.json(), proof.json()],
        "reasons": [],
        "qualityWindows": [{"ssim": 0.99, "psnr": 42.0}],
        "videoConfiguration": {"codec_name": "hevc"},
        "deliveryApproved": False,
    }
    monkeypatch.setattr(delivery, "part_acceptance", lambda config, *, number: acceptance)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, b"", b""),
    )
    return Case(
        work,
        request,
        config,
        source,
        movie,
        recipe.path,
        disposition.path,
        owner,
        acceptance,
        work / "intent.json",
        work / "result.json",
    )


def test_prepare_execute_and_consumer_restart_preserve_historical_proof(case: Case) -> None:
    original = Reference(case.source, ref(case.source)["sha256"])
    native = case.work / "native-completion.json"
    old_bytes = native.read_bytes()
    assert retirement.prepare(case.request, case.intent) == case.intent
    assert case.source.exists()
    assert retirement.execute(case.request, case.intent, case.result) == case.result
    assert not case.source.exists()
    result = read(case.result)
    assert result["kind"] == "retired-prores-part"
    assert result["masterMaterialized"] is False
    assert result["historicalNativePartVerified"] is True
    assert result["byteIdenticalReplacement"] is False
    assert result["rebuildRequired"] is True
    assert result["unlinkObserved"] is True
    assert result["interruptedAbsenceRecovered"] is False
    assert "historicalWholeVerified" not in result
    verified = retirement.verify(
        Reference(case.result, ref(case.result)["sha256"]),
        source=original,
        movie=Reference(case.movie, ref(case.movie)["sha256"]),
        part=1,
    )
    assert verified.rebuild_required is True
    assert original not in verified.inventory
    assert native.read_bytes() == old_bytes
    result_bytes = case.result.read_bytes()
    assert retirement.execute(case.request, case.intent, case.result) == case.result
    assert case.result.read_bytes() == result_bytes


def test_interruption_after_unlink_recovers_absence_without_second_unlink(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Reference(case.source, ref(case.source)["sha256"])
    retirement.prepare(case.request, case.intent)
    saved = save_new

    def interrupt(path: Path, raw: dict[str, object]) -> None:
        if path == case.result:
            raise OSError("synthetic power interruption after unlink")
        saved(path, raw)

    with monkeypatch.context() as context:
        context.setattr(retirement, "save_new", interrupt)
        with pytest.raises(OSError, match="interruption"):
            retirement.execute(case.request, case.intent, case.result)
    assert not case.source.exists()
    assert not case.result.exists()
    assert case.result.with_suffix(".json.operation.json").exists()
    monkeypatch.setattr(retirement, "_unlink", lambda *args: pytest.fail("must not unlink twice"))
    retirement.execute(case.request, case.intent, case.result)
    assert read(case.result)["unlinkObserved"] is False
    assert read(case.result)["interruptedAbsenceRecovered"] is True
    assert (
        retirement.verify(
            Reference(case.result, ref(case.result)["sha256"]),
            source=original,
            movie=Reference(case.movie, ref(case.movie)["sha256"]),
            part=1,
        ).rebuild_required
        is True
    )


def test_interruption_before_unlink_keeps_reserved_operation_and_can_resume(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    retirement.prepare(case.request, case.intent)
    with monkeypatch.context() as context:
        context.setattr(
            retirement, "_unlink", lambda *args: (_ for _ in ()).throw(OSError("before unlink"))
        )
        with pytest.raises(OSError, match="before unlink"):
            retirement.execute(case.request, case.intent, case.result)
    assert case.source.exists()
    assert case.result.with_suffix(".json.operation.json").exists()
    retirement.execute(case.request, case.intent, case.result)
    assert not case.source.exists()
    assert read(case.result)["unlinkObserved"] is True


@pytest.mark.parametrize(
    "field,value", [("part", 0), ("part", True), ("sourceBytes", 0), ("version", 2)]
)
def test_invalid_request_integer_is_rejected(case: Case, field: str, value: object) -> None:
    case.config[field] = value
    write(case.request, case.config)
    with pytest.raises(ReplayError):
        retirement.load_request(case.request)


@pytest.mark.parametrize("field", ["scope", "deletionAuthorized", "instruction"])
def test_approval_is_exact_and_required(case: Case, field: str) -> None:
    record(case.config["approval"]).pop(field)
    write(case.request, case.config)
    with pytest.raises(ReplayError, match="approved scope"):
        retirement.load_request(case.request)


@pytest.mark.parametrize(
    "field,value", [("accepted", False), ("kind", "candidate"), ("part", 2), ("version", True)]
)
def test_unaccepted_or_wrong_part_replacement_cannot_retire(
    case: Case, field: str, value: object
) -> None:
    case.acceptance[field] = value
    with pytest.raises(ReplayError, match="measured part acceptance"):
        retirement.prepare(case.request, case.intent)
    assert case.source.exists()
    assert not case.intent.exists()


def test_acceptance_must_bind_source_movie_and_evidence(case: Case) -> None:
    case.acceptance["inputs"] = [ref(case.movie)]
    with pytest.raises(ReplayError, match="omits its source"):
        retirement.prepare(case.request, case.intent)


def test_open_source_is_refused(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, b"p123\nf4\n", b""),
    )
    with pytest.raises(ReplayError, match="open or"):
        retirement.prepare(case.request, case.intent)
    assert case.source.exists()


def test_unknown_fd_observation_is_refused(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, b"", b"permission denied"),
    )
    with pytest.raises(ReplayError, match="unknown"):
        retirement.prepare(case.request, case.intent)


def test_active_workflow_lease_preserves_source(case: Case) -> None:
    with execution_lock(case.owner_state, ".workflow") as acquired:
        assert acquired
        with pytest.raises(ReplayError, match="workflow is active"):
            retirement.prepare(case.request, case.intent)
    assert case.source.exists()


def test_active_retirement_lease_preserves_source(case: Case) -> None:
    request = retirement.load_request(case.request)
    name = hashlib.sha256(str(case.source).encode()).hexdigest()
    with execution_lock(request.state, name) as acquired:
        assert acquired
        with pytest.raises(ReplayError, match="already owned"):
            retirement.prepare(case.request, case.intent)


@pytest.mark.parametrize("scope", ["current", "queued", "resume", "FCP", "Viewer", "diagnostic"])
def test_every_reference_category_must_release_materialized_source(case: Case, scope: str) -> None:
    data = read(case.disposition)
    record(record(data["references"])[scope])["requiresMaterializedSource"] = True
    case.config["disposition"] = write(case.disposition, data).json()
    write(case.request, case.config)
    with pytest.raises(ReplayError, match="unresolved"):
        retirement.prepare(case.request, case.intent)


def test_stale_disposition_blocks_new_deletion_but_not_completed_provenance(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Reference(case.source, ref(case.source)["sha256"])
    retirement.prepare(case.request, case.intent)
    retirement.execute(case.request, case.intent, case.result)
    instant = time.time_ns()
    monkeypatch.setattr(time, "time_ns", lambda: instant + 2 * retirement.MAX_AGE_NS)
    assert (
        retirement.verify(
            Reference(case.result, ref(case.result)["sha256"]),
            source=original,
            movie=Reference(case.movie, ref(case.movie)["sha256"]),
            part=1,
        ).rebuild_required
        is True
    )


def test_stale_disposition_refuses_new_intent(case: Case) -> None:
    raw = read(case.disposition)
    raw["observedAtNs"] = time.time_ns() - 2 * retirement.MAX_AGE_NS
    case.config["disposition"] = write(case.disposition, raw).json()
    write(case.request, case.config)
    with pytest.raises(ReplayError, match="stale"):
        retirement.prepare(case.request, case.intent)


def test_changed_source_after_prepare_cannot_be_deleted(case: Case) -> None:
    retirement.prepare(case.request, case.intent)
    case.source.write_bytes(b"changed source bytes")
    with pytest.raises(ReplayError, match="identity differs"):
        retirement.execute(case.request, case.intent, case.result)
    assert case.source.exists()
    assert not case.result.exists()


def test_missing_source_without_reserved_operation_is_not_a_completed_delete(case: Case) -> None:
    retirement.prepare(case.request, case.intent)
    case.source.unlink()
    with pytest.raises(ReplayError, match="without this reserved"):
        retirement.execute(case.request, case.intent, case.result)


def test_changed_retained_hevc_blocks_execute_and_restart(case: Case) -> None:
    retirement.prepare(case.request, case.intent)
    case.movie.write_bytes(b"changed HEVC")
    with pytest.raises(ReplayError, match="bytes changed"):
        retirement.execute(case.request, case.intent, case.result)
    assert case.source.exists()


def test_missing_native_recipe_input_blocks_execute(case: Case) -> None:
    retirement.prepare(case.request, case.intent)
    (case.work / "clean.mov").unlink()
    with pytest.raises(FileNotFoundError):
        retirement.execute(case.request, case.intent, case.result)
    assert case.source.exists()


def test_recipe_must_retain_clean_editable_font_voice_and_source(case: Case) -> None:
    raw = read(case.recipe)
    raw["inputs"] = [ref(case.work / "clean.mov")]
    case.config["recipe"] = write(case.recipe, raw).json()
    write(case.request, case.config)
    with pytest.raises(ReplayError, match="omits required"):
        retirement.prepare(case.request, case.intent)


def test_recipe_must_retain_every_xml_media_reference(case: Case) -> None:
    xml = case.work / "part.fcpxml"
    xml.write_text(
        "<fcpxml><resources><asset><media-rep "
        f'src="{(case.work / "unretained.mov").as_uri()}"/>'
        "</asset></resources></fcpxml>",
        encoding="utf-8",
    )
    raw = read(case.recipe)
    raw["partXML"] = ref(xml)
    partition = case.work / "partition.json"
    write(partition, {"version": 1, "parts": [ref(xml)]})
    raw["partition"] = ref(partition)
    frame_proof = case.work / "frame-proof.json"
    proof = read(frame_proof)
    proof["partition"] = ref(partition)
    record(case.config["native"])["frameProof"] = write(frame_proof, proof).json()
    completion_path = case.work / "native-completion.json"
    completion = read(completion_path)
    completion["frameProof"] = ref(frame_proof)
    record(completion["inputs"])[str(frame_proof)] = ref(frame_proof)["sha256"]
    record(case.config["native"])["completion"] = write(completion_path, completion).json()
    case.acceptance["inputs"] = [
        ref(case.source),
        ref(case.movie),
        record(case.acceptance["evidence"]),
        record(case.config["planner"]),
        ref(frame_proof),
    ]
    raw["inputs"] = [
        ref(xml),
        ref(partition),
        *[
            item
            for item in retirement._rows(raw["inputs"])
            if record(item)["path"] not in (str(xml), str(partition))
        ],
    ]
    case.config["recipe"] = write(case.recipe, raw).json()
    write(case.request, case.config)
    with pytest.raises(ReplayError, match="unretained"):
        retirement.prepare(case.request, case.intent)


def test_changed_native_completion_is_not_historical_success(case: Case) -> None:
    native = case.work / "native-completion.json"
    raw = read(native)
    raw["completionVerified"] = False
    record(case.config["native"])["completion"] = write(native, raw).json()
    write(case.request, case.config)
    with pytest.raises(ReplayError, match="independent completion"):
        retirement.prepare(case.request, case.intent)


def test_symlink_source_and_hardlink_source_are_refused(case: Case) -> None:
    twin = case.work / "alias.mov"
    twin.hardlink_to(case.source)
    with pytest.raises(ReplayError, match="one regular"):
        retirement.prepare(case.request, case.intent)
    twin.unlink()
    saved = case.work / "saved.mov"
    case.source.rename(saved)
    case.source.symlink_to(saved)
    with pytest.raises(ReplayError, match="symlinks"):
        retirement.load_request(case.request)


def test_result_provenance_never_accepts_reappeared_original(case: Case) -> None:
    original = Reference(case.source, ref(case.source)["sha256"])
    retirement.prepare(case.request, case.intent)
    retirement.execute(case.request, case.intent, case.result)
    case.source.write_bytes(b"ordinary rerender must use a new revision")
    with pytest.raises(ReplayError, match="present"):
        retirement.verify(
            Reference(case.result, ref(case.result)["sha256"]),
            source=original,
            movie=Reference(case.movie, ref(case.movie)["sha256"]),
            part=1,
        )


def test_occupied_operation_with_wrong_intent_is_preserved(case: Case) -> None:
    retirement.prepare(case.request, case.intent)
    operation = case.result.with_suffix(".json.operation.json")
    write(operation, {"kind": "unrelated operation"})
    old = operation.read_bytes()
    with pytest.raises(ReplayError, match="Occupied"):
        retirement.execute(case.request, case.intent, case.result)
    assert operation.read_bytes() == old
    assert case.source.exists()


def test_existing_intent_is_not_overwritten(case: Case) -> None:
    case.intent.write_bytes(b"old evidence")
    with pytest.raises(ReplayError, match="absent"):
        retirement.prepare(case.request, case.intent)
    assert case.intent.read_bytes() == b"old evidence"


@pytest.mark.parametrize("mode", ["prepare", "execute", "verify"])
def test_cli_lifecycle_and_thin_dispatch(case: Case, mode: str) -> None:
    if mode != "prepare":
        retirement.prepare(case.request, case.intent)
    if mode == "verify":
        retirement.execute(case.request, case.intent, case.result)
    args = [mode, str(case.request), str(case.intent if mode == "prepare" else case.result)]
    if mode == "execute":
        args.extend(["--intent", str(case.intent)])
    assert retirement.main(args) == 0


def test_cli_requires_explicit_intent_and_rejects_prepare_intent(case: Case) -> None:
    with pytest.raises(SystemExit) as missing:
        retirement.main(["execute", str(case.request), str(case.result)])
    assert missing.value.code == 1
    with pytest.raises(SystemExit) as unwanted:
        retirement.main(
            ["prepare", str(case.request), str(case.intent), "--intent", str(case.intent)]
        )
    assert unwanted.value.code == 1


def test_published_native_original_is_verified_without_relabeling_hevc(case: Case) -> None:
    published = case.work / "published.mov"
    published.write_bytes(case.source.read_bytes())
    native = record(case.config["native"])
    old_proof = case.work / "frame-proof.json"
    publication_intent = write(
        case.work / "publication-intent.json",
        {
            "version": 1,
            "source": ref(case.source),
            "destination": str(published),
            "completion": native["completion"],
        },
    )
    published_proof = write(
        case.work / "published-proof.json",
        {
            **read(old_proof),
            "movie": ref(published),
            "sourceFrameProof": ref(old_proof),
            "publicationIntent": publication_intent.json(),
        },
    )
    publication = write(
        case.work / "publication.json",
        {
            "version": 1,
            "kind": "published-native-part",
            "source": ref(case.source),
            "movie": ref(published),
            "completion": native["completion"],
            "frameProof": published_proof.json(),
            "intent": publication_intent.json(),
            "sourcePreserved": True,
        },
    )
    native["publication"] = publication.json()
    native["frameProof"] = published_proof.json()
    case.config["source"] = ref(published)
    case.acceptance["source"] = ref(published)
    case.acceptance["inputs"] = [
        ref(published),
        ref(case.movie),
        record(case.acceptance["evidence"]),
        record(case.config["planner"]),
    ]
    recipe = read(case.recipe)
    recipe["source"] = ref(published)
    case.config["recipe"] = write(case.recipe, recipe).json()
    disposition = read(case.disposition)
    disposition["source"] = ref(published)
    case.config["disposition"] = write(case.disposition, disposition).json()
    write(case.request, case.config)
    retirement.prepare(case.request, case.intent)
    retirement.execute(case.request, case.intent, case.result)
    assert not published.exists()
    assert case.source.exists()
    assert read(case.result)["byteIdenticalReplacement"] is False


@pytest.mark.parametrize(
    "changed",
    [
        "kind",
        "masterMaterialized",
        "byteIdenticalReplacement",
        "rebuildRequired",
        "part",
        "deliveryApproved",
    ],
)
def test_forged_retired_flags_do_not_pass_consumer(case: Case, changed: str) -> None:
    original = Reference(case.source, ref(case.source)["sha256"])
    retirement.prepare(case.request, case.intent)
    retirement.execute(case.request, case.intent, case.result)
    raw = read(case.result)
    raw[changed] = "invalid"
    write(case.result, raw)
    with pytest.raises(ReplayError, match="provenance"):
        retirement.verify(
            Reference(case.result, ref(case.result)["sha256"]),
            source=original,
            movie=Reference(case.movie, ref(case.movie)["sha256"]),
            part=1,
        )


def test_post_retirement_changed_recipe_and_replacement_are_not_currently_verified(
    case: Case,
) -> None:
    original = Reference(case.source, ref(case.source)["sha256"])
    replacement = Reference(case.movie, ref(case.movie)["sha256"])
    retirement.prepare(case.request, case.intent)
    retirement.execute(case.request, case.intent, case.result)
    case.movie.write_bytes(b"changed replacement after retirement")
    with pytest.raises(ReplayError, match="bytes changed"):
        retirement.verify(
            Reference(case.result, ref(case.result)["sha256"]),
            source=original,
            movie=replacement,
            part=1,
        )


def test_retained_inventory_cannot_include_original_path(case: Case) -> None:
    retirement.prepare(case.request, case.intent)
    raw = read(case.intent)
    raw["inventory"] = [*retirement._rows(raw["inventory"]), ref(case.source)]
    write(case.intent, raw)
    with pytest.raises(ReplayError, match="retired source"):
        retirement.execute(case.request, case.intent, case.result)
    assert case.source.exists()


def test_changed_parent_identity_refuses_unlink(case: Case) -> None:
    retirement.prepare(case.request, case.intent)
    raw = read(case.intent)
    raw["parentIdentity"] = [0, 0]
    write(case.intent, raw)
    with pytest.raises(ReplayError, match="parent changed"):
        retirement.execute(case.request, case.intent, case.result)
    assert case.source.exists()


def test_correct_inode_but_wrong_sha_refuses_unlink(case: Case) -> None:
    request = retirement.load_request(case.request)
    wrong_source = Reference(case.source, "0" * 64)
    wrong = retirement.Request(
        request.reference,
        request.part,
        wrong_source,
        request.source_bytes,
        request.planner,
        request.native,
        request.recipe,
        request.disposition,
        request.state,
        request.leases,
        request.instruction,
    )
    parent = case.source.parent.stat()
    with pytest.raises(ReplayError, match="bytes changed"):
        retirement._unlink(
            wrong,
            {
                "sourceIdentity": list(retirement._identity(case.source)),
                "parentIdentity": [parent.st_dev, parent.st_ino],
            },
        )
    assert case.source.exists()


def test_interrupted_absence_can_be_reconciled_after_owner_observation_ages(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    retirement.prepare(case.request, case.intent)
    with monkeypatch.context() as context:
        context.setattr(
            retirement,
            "save_new",
            lambda path, raw: (
                save_new(path, raw)
                if path != case.result
                else (_ for _ in ()).throw(OSError("interrupted"))
            ),
        )
        with pytest.raises(OSError):
            retirement.execute(case.request, case.intent, case.result)
    instant = time.time_ns()
    monkeypatch.setattr(time, "time_ns", lambda: instant + 2 * retirement.MAX_AGE_NS)
    retirement.execute(case.request, case.intent, case.result)
    assert read(case.result)["interruptedAbsenceRecovered"] is True


@pytest.mark.parametrize(
    "value", [None, "https://example.invalid/media.mov", "file://elsewhere/media.mov"]
)
def test_native_xml_references_must_be_local(value: str | None) -> None:
    with pytest.raises(ReplayError):
        retirement._xml_media(value)


@pytest.mark.parametrize(
    "value", [[], [1, 2, 3], [1, 2, 0, 4, 5, 1], [1, 2, 3, 4, 5, 2], [True, 2, 3, 4, 5, 1]]
)
def test_invalid_filesystem_identity_is_not_accepted(value: object) -> None:
    with pytest.raises(ReplayError):
        retirement._identity_value(value)


@pytest.mark.parametrize("suffix", [".m4v", ".mov", ".M4V"])
def test_accepted_native_hevc_intermediate_container_can_retire(case: Case, suffix: str) -> None:
    source = Reference(case.source, ref(case.source)["sha256"])
    movie = case.movie.with_suffix(suffix)
    case.movie.rename(movie)
    evidence = case.work / "hevc-evidence.json"
    raw = read(evidence)
    raw["movie"] = ref(movie)
    write(evidence, raw)
    case.acceptance["movie"] = ref(movie)
    case.acceptance["evidence"] = ref(evidence)
    case.acceptance["inputs"] = [
        ref(case.source),
        ref(movie),
        ref(evidence),
        record(case.config["planner"]),
    ]
    retirement.prepare(case.request, case.intent)
    retirement.execute(case.request, case.intent, case.result)
    verified = retirement.verify(
        Reference(case.result, ref(case.result)["sha256"]),
        source=source,
        movie=Reference(movie, ref(movie)["sha256"]),
        part=1,
    )
    assert verified.rebuild_required is True
    assert not case.source.exists()
    assert movie.exists()


def test_unsupported_intermediate_container_cannot_retire(case: Case) -> None:
    movie = case.movie.with_suffix(".avi")
    case.movie.rename(movie)
    case.acceptance["movie"] = ref(movie)
    with pytest.raises(ReplayError, match="supported intermediate container"):
        retirement.prepare(case.request, case.intent)
    assert case.source.exists()
    assert not case.intent.exists()


@pytest.fixture
def configured_python(case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Reference:
    runtime = tmp_path / "runtime"
    binary = runtime / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic configured executable")
    binary.chmod(0o700)
    alias = retirement.ROOT / ".venv/bin/python"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(binary)
    patch = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    (retirement.ROOT / ".python-version").write_text(patch + "\n", encoding="ascii")
    monkeypatch.setattr(sys, "base_prefix", str(runtime))

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert kwargs.get("stdin") == subprocess.DEVNULL
        if args == [str(binary), "--version"]:
            return subprocess.CompletedProcess(args, 0, f"Python {patch}\n".encode(), b"")
        assert args[0] == "/usr/sbin/lsof"
        return subprocess.CompletedProcess(args, 1, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    return Reference(binary, ref(binary)["sha256"])


def test_configured_executable_evidence_survives_retirement_and_resume(
    case: Case, configured_python: Reference
) -> None:
    source = retirement.load_request(case.request).source
    with pytest.raises(ReplayError, match="authorized scope"):
        retirement._ref(configured_python.json())
    case.acceptance["inputs"] = [
        *retirement._rows(case.acceptance["inputs"]),
        configured_python.json(),
    ]
    retirement.prepare(case.request, case.intent)
    retirement.execute(case.request, case.intent, case.result)
    verified = retirement.verify(
        Reference(case.result, ref(case.result)["sha256"]),
        source=source,
        movie=Reference(case.movie, ref(case.movie)["sha256"]),
        part=1,
    )
    assert configured_python in verified.inventory
    assert verified.rebuild_required


@pytest.mark.parametrize(
    "failure", ["missing", "regular-alias", "other-python", "nonexec", "patch", "sha"]
)
def test_external_python_evidence_requires_exact_current_configuration(
    case: Case, configured_python: Reference, failure: str
) -> None:
    alias = retirement.ROOT / ".venv/bin/python"
    target = configured_python.path
    if failure == "missing":
        alias.unlink()
    elif failure == "regular-alias":
        alias.unlink()
        alias.write_bytes(b"not a configured symlink")
    elif failure == "other-python":
        other = target.with_name(target.name + "-other")
        other.write_bytes(target.read_bytes())
        other.chmod(0o700)
        alias.unlink()
        alias.symlink_to(other)
        target = other
    elif failure == "nonexec":
        target.chmod(0o600)
    elif failure == "patch":
        (retirement.ROOT / ".python-version").write_text("0.0.0\n", encoding="ascii")
    else:
        target.write_bytes(b"changed binary")
    value = {"path": str(target), "sha256": configured_python.sha256}
    with pytest.raises(ReplayError, match="Configured Python executable evidence"):
        retirement._evidence_ref(value)
    assert case.source.exists()


@pytest.mark.parametrize(
    "failure", ["returncode", "version", "stderr", "timeout", "target-race", "alias-race"]
)
def test_executable_observation_and_identity_failures_refuse_evidence(
    configured_python: Reference, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert args == [str(configured_python.path), "--version"]
        assert kwargs["timeout"] == 5
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 5)
        patch = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if failure == "target-race":
            configured_python.path.write_bytes(b"changed during observation")
        if failure == "alias-race":
            alias = retirement.ROOT / ".venv/bin/python"
            alias.unlink()
            alias.symlink_to(configured_python.path)
        return subprocess.CompletedProcess(
            args,
            1 if failure == "returncode" else 0,
            b"Python 0.0.0\n" if failure == "version" else f"Python {patch}\n".encode(),
            b"unexpected stderr" if failure == "stderr" else b"",
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ReplayError, match="Configured Python executable evidence"):
        retirement._evidence_ref(configured_python.json())


def test_executable_evidence_never_expands_source_or_recipe_paths(
    case: Case, configured_python: Reference
) -> None:
    with pytest.raises(ReplayError, match="authorized scope"):
        retirement._path(str(configured_python.path), source=True)
    with pytest.raises(ReplayError, match="authorized scope"):
        retirement._ref(configured_python.json())
    raw = read(case.recipe)
    raw["fontAsset"] = configured_python.json()
    raw["inputs"] = [*retirement._rows(raw["inputs"]), configured_python.json()]
    write(case.recipe, raw)
    case.config["recipe"] = ref(case.recipe)
    write(case.request, case.config)
    with pytest.raises(ReplayError, match="authorized scope"):
        retirement.prepare(case.request, case.intent)
    assert case.source.exists()


def test_current_real_interpreter_binds_selected_acceptance_without_media(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retirement, "ROOT", REAL_ROOT)
    monkeypatch.setattr(metadata, "ROOT", REAL_ROOT)
    binary = (REAL_ROOT / ".venv/bin/python").resolve(strict=True)
    configured = Reference(binary, ref(binary)["sha256"])

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if args == [str(binary), "--version"]:
            return REAL_RUN(
                args,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=5,
                cwd=REAL_ROOT,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        assert args[0] == "/usr/sbin/lsof"
        return subprocess.CompletedProcess(args, 1, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    case.acceptance["inputs"] = [*retirement._rows(case.acceptance["inputs"]), configured.json()]
    loaded = retirement.load_request(case.request)
    assert configured in retirement._accepted(loaded, case.acceptance)
    retirement.prepare(case.request, case.intent)
    assert read(case.intent)["acceptance"] == case.acceptance
    assert case.source.exists()


def test_acceptance_never_replaces_prores_with_itself(case: Case) -> None:
    case.acceptance["movie"] = ref(case.source)
    with pytest.raises(ReplayError, match="distinct"):
        retirement.prepare(case.request, case.intent)
    assert case.source.exists()
