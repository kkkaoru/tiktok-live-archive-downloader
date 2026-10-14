import json
import os
import sys
from pathlib import Path

import pytest

from replay import fcp_native_progress as progress
from replay import part_master_publication as publication
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source.mov"
    source.write_bytes(b"verified master bytes")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"verified":true}', encoding="utf-8")
    movie = {"path": str(source), "sha256": file_digest(source)}
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "version": 1,
                "part": 1,
                "movie": movie,
                "partition": {"sha256": "preserved"},
                "observed": {"frames": [0, 1]},
            }
        ),
        encoding="utf-8",
    )
    completion = tmp_path / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "verified-native-part",
                "part": 1,
                "completionVerified": True,
                "shareObservedIdle": True,
                "fullAVDecodeExitCode": 0,
                "movie": movie,
                "inputs": {str(evidence): file_digest(evidence)},
                "frameProof": {"path": str(proof), "sha256": file_digest(proof)},
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "ssd"
    destination.mkdir()
    config = tmp_path / "publish.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "completion": {"path": str(completion), "sha256": file_digest(completion)},
                "destination": str(destination / "part.mov"),
                "receiptOutput": str(tmp_path / "published.json"),
                "frameProofOutput": str(tmp_path / "ssd-proof.json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: os.statvfs_result(
            (4096, 4096, 100_000_000, 90_000_000, 90_000_000, 1, 1, 1, 0, 255)
        ),
    )
    return config


def test_copy_keeps_source_and_rebinds_only_verified_frame_proof(plan: Path) -> None:
    output = publication.publish(plan)
    assert output.name == "published.json"
    assert (plan.parent / "source.mov").read_bytes() == b"verified master bytes"
    assert (plan.parent / "ssd/part.mov").read_bytes() == b"verified master bytes"
    assert not (plan.parent / "ssd/.part.mov.publication-partial").exists()
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["sourcePreserved"] is True
    assert receipt["qualityApproved"] is False
    proof = json.loads((plan.parent / "ssd-proof.json").read_text(encoding="utf-8"))
    assert proof["partition"] == {"sha256": "preserved"}
    assert proof["observed"] == {"frames": [0, 1]}
    assert Path(proof["movie"]["path"]).parent.name == "ssd"
    assert proof["sourceFrameProof"]["path"].endswith("/proof.json")
    assert publication.verify_published(output, plan.parent / "ssd-proof.json", part=1) == (
        plan.parent / "completion.json",
        plan.parent / "published.intent.json",
    )
    with pytest.raises(ReplayError, match="absent"):
        publication.publish(plan)


@pytest.mark.parametrize("change", ["kind", "source", "proof", "intent", "part"])
def test_published_binding_rejects_changes(plan: Path, change: str) -> None:
    output = publication.publish(plan)
    raw = json.loads(output.read_text(encoding="utf-8"))
    if change == "kind":
        raw["kind"] = "unverified"
    elif change == "source":
        raw["source"] = {}
    elif change in {"proof", "intent"}:
        key = "frameProof" if change == "proof" else "intent"
        path = Path(raw[key]["path"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["movie" if change == "proof" else "destination"] = "changed"
        path.write_text(json.dumps(data), encoding="utf-8")
        raw[key]["sha256"] = file_digest(path)
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        publication.verify_published(
            output, plan.parent / "ssd-proof.json", part=2 if change == "part" else 1
        )


@pytest.mark.parametrize("retire", [False, True])
def test_published_progress_uses_original_completion(
    plan: Path, monkeypatch: pytest.MonkeyPatch, retire: bool
) -> None:
    output = publication.publish(plan)
    state = plan.parent / "progress.json"
    state.write_text(
        json.dumps(
            {
                "version": 1,
                "libraryAndMasterRoot": str(plan.parent / "ssd"),
                "parts": [{"part": 1, "master": "part.mov", "state": "rendering"}],
            }
        ),
        encoding="utf-8",
    )
    assembly = plan.parent / "assembly.json"
    assembly.write_text("{}", encoding="utf-8")
    config = plan.parent / "progress-config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "part": 1,
                "progress": {"path": str(state), "sha256": file_digest(state)},
                "assembly": {"path": str(assembly), "sha256": file_digest(assembly)},
                "publication": {"path": str(output), "sha256": file_digest(output)},
                "receiptPath": str(plan.parent / "progress-receipt.json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(progress, "verify", lambda path, *, part: plan.parent / "ssd-proof.json")
    if retire:
        monkeypatch.setattr(
            publication, "verify_part", lambda path, *, part: plan.parent / "ssd-proof.json"
        )
        retirement_config = plan.parent / "retirement-config.json"
        retirement_config.write_text(
            json.dumps(
                {
                    "version": 1,
                    "retireStaging": True,
                    "publication": {"path": str(output), "sha256": file_digest(output)},
                    "assembly": {"path": str(assembly), "sha256": file_digest(assembly)},
                    "receiptOutput": str(plan.parent / "retirement.json"),
                }
            )
        )
        retired = publication.prepare_retirement(retirement_config)
        (plan.parent / "source.mov").unlink()
        raw = json.loads(config.read_text())
        raw["retirement"] = {"path": str(retired), "sha256": file_digest(retired)}
        config.write_text(json.dumps(raw))
    progress.update(config)
    assert json.loads(state.read_text(encoding="utf-8"))["parts"][0]["state"] == "complete"


@pytest.mark.parametrize(
    "change",
    [
        "exists",
        "symlink",
        "partial",
        "receipt",
        "intent",
        "same",
        "space",
        "version",
        "unverified",
        "input",
        "source",
        "proof",
    ],
)
def test_rejects_collisions_incomplete_and_changed_inputs(
    plan: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    config = json.loads(plan.read_text(encoding="utf-8"))
    if change == "exists":
        Path(config["destination"]).write_bytes(b"keep")
    elif change == "symlink":
        Path(config["destination"]).symlink_to(plan.parent / "missing")
    elif change == "partial":
        (plan.parent / "ssd/.part.mov.publication-partial").write_bytes(b"keep")
    elif change == "receipt":
        Path(config["receiptOutput"]).write_bytes(b"keep")
    elif change == "intent":
        (plan.parent / "published.intent.json").write_bytes(b"keep")
    elif change == "same":
        config["destination"] = str(plan.parent / "source.mov")
    elif change == "space":
        monkeypatch.setattr(
            os,
            "statvfs",
            lambda path: os.statvfs_result((4096, 4096, 100, 0, 0, 1, 1, 1, 0, 255)),
        )
    elif change == "version":
        config["version"] = 2
    elif change == "input":
        (plan.parent / "evidence.json").write_bytes(b"changed")
    elif change == "source":
        (plan.parent / "source.mov").write_bytes(b"changed")
    else:
        completion = Path(config["completion"]["path"])
        raw = json.loads(completion.read_text(encoding="utf-8"))
        if change == "unverified":
            raw["completionVerified"] = False
        else:
            raw["part"] = 2
        completion.write_text(json.dumps(raw), encoding="utf-8")
        config["completion"]["sha256"] = file_digest(completion)
    plan.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError, RuntimeError)):
        publication.publish(plan)
    assert (plan.parent / "source.mov").exists()


def test_changed_copy_retains_failure_evidence(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = file_digest

    def digest(path: Path) -> str:
        if path.name.endswith("publication-partial"):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(publication, "file_digest", digest)
    with pytest.raises(ReplayError, match="changed"):
        publication.publish(plan)
    assert (plan.parent / "published.intent.json").exists()
    assert (plan.parent / "ssd/.part.mov.publication-partial").exists()
    assert not (plan.parent / "ssd/part.mov").exists()


def test_publish_race_never_overwrites(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.link

    def collide(source: Path, destination: Path) -> None:
        destination.write_bytes(b"concurrent writer")
        original(source, destination)

    monkeypatch.setattr(os, "link", collide)
    with pytest.raises(FileExistsError):
        publication.publish(plan)
    assert (plan.parent / "ssd/part.mov").read_bytes() == b"concurrent writer"
    assert not (plan.parent / "published.json").exists()


def test_cli(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["part-master-publication", str(plan)])
    publication.main()
    assert (plan.parent / "published.json").is_file()


@pytest.fixture
def retirement_plan(plan: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = publication.publish(plan)
    assembly = plan.parent / "assembly.json"
    assembly.write_text("{}", encoding="utf-8")
    config = plan.parent / "retire.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "retireStaging": True,
                "publication": {"path": str(output), "sha256": file_digest(output)},
                "assembly": {"path": str(assembly), "sha256": file_digest(assembly)},
                "receiptOutput": str(plan.parent / "retirement.json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        publication, "verify_part", lambda path, *, part: plan.parent / "ssd-proof.json"
    )
    return config


def test_retirement_requires_explicit_proof_and_preserves_history(retirement_plan: Path) -> None:
    root = retirement_plan.parent
    original = (root / "published.json").read_bytes()
    receipt = publication.prepare_retirement(retirement_plan)
    assert (root / "source.mov").exists()
    assert json.loads(receipt.read_text())["deletionPerformed"] is False
    (root / "source.mov").unlink()
    with pytest.raises(FileNotFoundError):
        publication.verify_published(root / "published.json", root / "ssd-proof.json", part=1)
    publication.verify_published(
        root / "published.json", root / "ssd-proof.json", part=1, retirement=receipt
    )
    assert (root / "published.json").read_bytes() == original
    verify = root / "verify-retirement.json"
    verify.write_text(
        json.dumps(
            {"version": 1, "retirement": {"path": str(receipt), "sha256": file_digest(receipt)}}
        )
    )
    publication.verify_retirement(verify)


@pytest.mark.parametrize(
    "target",
    [
        "source.mov",
        "ssd/part.mov",
        "completion.json",
        "evidence.json",
        "assembly.json",
        "published.json",
        "retire.json",
    ],
)
def test_retired_inputs_still_reject_mutation(retirement_plan: Path, target: str) -> None:
    receipt = publication.prepare_retirement(retirement_plan)
    (retirement_plan.parent / target).write_bytes(b"changed")
    with pytest.raises((ReplayError, ValueError)):
        publication.verify_published(
            retirement_plan.parent / "published.json",
            retirement_plan.parent / "ssd-proof.json",
            part=1,
            retirement=receipt,
        )


def test_retirement_never_accepts_missing_destination(retirement_plan: Path) -> None:
    receipt = publication.prepare_retirement(retirement_plan)
    (retirement_plan.parent / "ssd/part.mov").unlink()
    with pytest.raises(FileNotFoundError):
        publication.verify_published(
            retirement_plan.parent / "published.json",
            retirement_plan.parent / "ssd-proof.json",
            part=1,
            retirement=receipt,
        )


@pytest.mark.parametrize(
    "change",
    [
        "kind",
        "source",
        "movie",
        "completion",
        "intent",
        "frameProof",
        "sourceIdentity",
        "destinationIdentity",
        "part",
        "assembly",
    ],
)
def test_unrelated_retirement_is_rejected(retirement_plan: Path, change: str) -> None:
    receipt = publication.prepare_retirement(retirement_plan)
    raw = json.loads(receipt.read_text())
    raw[change] = None
    receipt.write_text(json.dumps(raw))
    with pytest.raises((ReplayError, ValueError)):
        publication.verify_published(
            retirement_plan.parent / "published.json",
            retirement_plan.parent / "ssd-proof.json",
            part=1,
            retirement=receipt,
        )


@pytest.mark.parametrize(
    "change", ["version", "intent", "part", "existing", "collision", "assembly"]
)
def test_retirement_preflight_rejects_invalid_intent(
    retirement_plan: Path, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.loads(retirement_plan.read_text())
    if change == "version":
        raw["version"] = 2
    elif change == "intent":
        raw["retireStaging"] = False
    elif change == "existing":
        Path(raw["receiptOutput"]).write_text("{}")
    elif change == "collision":
        raw["receiptOutput"] = str(retirement_plan.parent / "source.mov")
    elif change == "assembly":
        monkeypatch.setattr(
            publication,
            "verify_part",
            lambda path, *, part: retirement_plan.parent / "wrong-proof.json",
        )
    else:
        proof = retirement_plan.parent / "ssd-proof.json"
        document = json.loads(proof.read_text())
        document["part"] = 0
        proof.write_text(json.dumps(document))
        published = Path(raw["publication"]["path"])
        pub = json.loads(published.read_text())
        pub["frameProof"]["sha256"] = file_digest(proof)
        published.write_text(json.dumps(pub))
        raw["publication"]["sha256"] = file_digest(published)
    retirement_plan.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        publication.prepare_retirement(retirement_plan)


def test_retirement_cli(retirement_plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["publication", str(retirement_plan), "--prepare-retirement"])
    publication.main()
    receipt = retirement_plan.parent / "retirement.json"
    verify = retirement_plan.parent / "verify.json"
    verify.write_text(
        json.dumps(
            {"version": 1, "retirement": {"path": str(receipt), "sha256": file_digest(receipt)}}
        )
    )
    monkeypatch.setattr(sys, "argv", ["publication", str(verify), "--verify-retirement"])
    publication.main()
