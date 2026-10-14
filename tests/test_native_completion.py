"""Synthetic receipt-contract tests; no real native render or quality approval."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.edit_completion import REQUIRED_STAGES, assess_delivery
from replay.native_completion import REFERENCES, Evidence, assess
from replay.recordings import record


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Case:
    root: Path
    source: Path
    movie: Path
    directory: Path
    data: dict[str, dict[str, object]]

    @property
    def index(self) -> Path:
        return self.movie.parent.parent / "metadata/native-evidence.json"

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.json"

    def save(self, name: str) -> None:
        self.path(name).write_text(json.dumps(self.data[name]), encoding="utf-8")

    def refresh(self) -> None:
        self.save("plan")
        plan_hash = digest(self.path("plan"))
        self.data["encoding-verified"]["planSHA256"] = plan_hash
        self.data["alpha-layout-audit"]["planSHA256"] = plan_hash
        self.save("encoding-verified")
        self.save("alpha-layout-audit")
        self.save("alpha-submit.request")
        self.save("alpha-submit.response")
        self.data["manifest"]["parts"] = [{"plan": str(self.path("plan")), "planSHA256": plan_hash}]
        self.save("manifest")
        manifest_hash = digest(self.path("manifest"))
        self.data["sourceEncoding"]["manifestSHA256"] = manifest_hash
        self.save("sourceEncoding")
        old_hash = digest(self.path("sourceEncoding"))
        self.data["audio"]["wholeVerificationSHA256"] = old_hash
        self.save("audio")
        self.data["encoding"].update(
            {
                "manifestSHA256": manifest_hash,
                "sourceVerificationSHA256": old_hash,
                "audioVerificationSHA256": digest(self.path("audio")),
            }
        )
        self.data["captions"]["manifestSHA256"] = manifest_hash
        for name in ("cuts", "masks", "quality"):
            self.data[name]["inputs"] = {
                str(self.path("manifest")): manifest_hash,
                str(self.source): digest(self.source),
            }
        self.save("quality")
        self.data["masks"]["qualityEvidence"] = {
            "path": str(self.path("quality").relative_to(self.root)),
            "sha256": digest(self.path("quality")),
        }
        for name in REFERENCES:
            self.save(name)
        index: dict[str, object] = {
            "version": 1,
            "backend": "motion-compressor",
            "recording": "replay-1",
            "configurationSHA256": digest(self.root / "edit-config.json"),
            "sourceSHA256": digest(self.source),
        }
        index.update(
            {
                name: {
                    "path": str(self.path(name).relative_to(self.root)),
                    "sha256": digest(self.path(name)),
                }
                for name in REFERENCES
            }
        )
        self.index.write_text(json.dumps(index), encoding="utf-8")

    def check(self) -> dict[str, str]:
        return assess(self.index, delivery=self.movie, root=self.root, source=self.source)


@pytest.fixture
def case(tmp_path: Path) -> Case:
    root = tmp_path / "edits/replay-1"
    directory = root / "evidence/native"
    directory.mkdir(parents=True)
    delivery = root / "revisions/native/delivery"
    delivery.mkdir(parents=True)
    (delivery.parent / "metadata").mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic original")
    movie = delivery / "2024-01-01_09-00-00_JST_replay-1-review.mov"
    movie.write_bytes(b"synthetic native movie")
    movie.with_suffix(".srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nSynthetic\n", encoding="utf-8"
    )
    old = directory / "old.mov"
    old.write_bytes(b"synthetic old movie")
    voice = directory / "voice.wav"
    voice.write_bytes(b"synthetic voice")
    xml = directory / "project.fcpxml"
    xml.write_text("<fcpxml/>", encoding="utf-8")
    motion = directory / "captions.moti"
    motion.write_bytes(b"synthetic motion")
    alpha = directory / "alpha.mov"
    alpha.write_bytes(b"synthetic alpha")
    (root / "edit-config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "replay_id": "1",
                "source": str(source),
                "source_size": source.stat().st_size,
                "engine": str(tmp_path),
                "executor_scope": str(tmp_path),
                "stages": dict.fromkeys(REQUIRED_STAGES, True),
            }
        ),
        encoding="utf-8",
    )
    encoding: dict[str, object] = {
        "frames": 90,
        "rate": 30,
        "audioSamples": 132300,
        "fullDecodeVerified": True,
        "clocksVerified": True,
        "audioPCMIdentical": True,
    }
    speech = {"level_delta_db_p50": -0.1, "level_delta_db_p05": -0.5, "correlation": 0.999}
    data: dict[str, dict[str, object]] = {
        "manifest": {"totalFrames": 90, "rate": 30},
        "encoding": {
            **encoding,
            "outputBytes": movie.stat().st_size,
            "outputSHA256": digest(movie),
            "editedVoice": str(voice),
            "editedVoiceSHA256": digest(voice),
            "videoPayloadIdentical": True,
        },
        "sourceEncoding": {
            **encoding,
            "output": str(old),
            "outputBytes": old.stat().st_size,
            "outputSHA256": digest(old),
        },
        "audio": {
            "status": "candidate-audio-verified",
            "policy": {
                "minimum_instrumental_drop_db": 10.0,
                "maximum_speech_loss_db": 1.5,
                "quiet_instrumental_db": -60.0,
            },
            "minimumSpeechCorrelation": 0.995,
            "musicGatePassed": True,
            "speechGatePassed": True,
            "candidateAudioOnly": True,
            "candidateVoice": str(voice),
            "candidateVoiceSHA256": digest(voice),
            "sourceVideoSHA256": digest(old),
            "speech": speech,
            "candidateSpeechComparedToOriginalFirstPass": speech,
            "musicProbes": [
                {
                    "start_seconds": 0,
                    "before_dbfs": -30,
                    "after_dbfs": -42,
                    "drop_db": 12,
                    "passed": True,
                }
            ],
        },
        "cuts": {
            "selectionPlanMatchesProtectedCuts": True,
            "selectedFrames": 90,
            "sourceFrames": 120,
            "removedFrames": 30,
        },
        "masks": {
            "recordedExecutionMatchesPlan": True,
            "selectedFrames": 90,
            "selectedMaskFrames": 30,
            "missingMaskFrames": 0,
            "extraMaskFrames": 0,
            "includedEmptyDetections": 0,
            "sourceCaptionQualityApproved": True,
        },
        "quality": {
            "measuredFrames": 30,
            "technicallyInvalidFrames": 0,
            "maximumResidualFraction": 0.001,
            "maximumUnstableFraction": 0.01,
            "readableFrames": 0,
        },
        "captions": {
            "srtSHA256": digest(movie.with_suffix(".srt")),
            "cueCount": 1,
            "outputFrames": 90,
            "rate": 30,
            "textConserved": True,
            "nativeFrameAligned": True,
        },
        "fcp": {
            "importObserved": True,
            "noModalErrorObserved": True,
            "xml": str(xml),
            "xmlSHA256": digest(xml),
        },
        "plan": {
            "startFrame": 0,
            "durationFrames": 90,
            "rate": 30,
            "motion": str(motion),
            "motionSHA256": digest(motion),
            "alphaRequest": str(directory / "alpha-submit.request.json"),
            "captions": [{"start": "0", "end": "1", "lines": ["Synthetic"]}],
        },
        "encoding-verified": {
            "frames": 90,
            "rate": 30,
            "fullDecodeVerified": True,
            "clocksVerified": True,
            "output": str(movie),
            "outputSHA256": digest(movie),
            "outputBytes": movie.stat().st_size,
        },
        "alpha-layout-audit": {
            "boundsAndGapsVerified": True,
            "fullCaptionStates": 1,
            "sampleCount": 2,
            "alphaSHA256": digest(alpha),
        },
        "alpha-submit.request": {"sourcePath": str(motion), "bundleID": "com.apple.CompressorApp"},
        "alpha-submit.response": {
            "ok": True,
            "data": {"structuredContent": {"exitStatus": 0, "outputPath": str(alpha)}},
        },
    }
    result = Case(root, source, movie, directory, data)
    result.refresh()
    return result


def test_synthetic_native_contract_and_completion_dispatch(case: Case) -> None:
    assert case.check() == {}
    assert assess_delivery(case.movie, root=case.root) == {}


def test_recorded_mask_execution_does_not_infer_quality(case: Case) -> None:
    case.data["masks"]["sourceCaptionQualityApproved"] = False
    case.refresh()
    assert case.check() == {
        "source_caption_mask": (
            "native source-mask accounting passed; independent removal quality remains unverified"
        )
    }


@pytest.mark.parametrize(
    ("name", "field", "value", "stage"),
    [
        ("masks", "missingMaskFrames", 1, "source_caption_mask"),
        ("masks", "includedEmptyDetections", 1, "source_caption_mask"),
        ("masks", "extraMaskFrames", False, "source_caption_mask"),
        ("masks", "selectedMaskFrames", 0, "source_caption_mask"),
        ("quality", "maximumResidualFraction", 0.006, "source_caption_mask"),
        ("quality", "maximumUnstableFraction", 0.021, "source_caption_mask"),
        ("quality", "readableFrames", 1, "source_caption_mask"),
        ("quality", "technicallyInvalidFrames", 1, "source_caption_mask"),
        ("cuts", "removedFrames", 0, "silence_cut"),
        ("cuts", "selectionPlanMatchesProtectedCuts", False, "silence_cut"),
        ("captions", "cueCount", 2, "transcription"),
        ("captions", "textConserved", False, "transcription"),
        ("fcp", "importObserved", False, "dynamic_captions"),
        ("audio", "musicGatePassed", False, "bgm_removal"),
        ("audio", "policy", {}, "bgm_removal"),
        (
            "audio",
            "speech",
            {"level_delta_db_p50": -3, "level_delta_db_p05": -4, "correlation": 0.999},
            "bgm_removal",
        ),
        (
            "audio",
            "speech",
            {"level_delta_db_p50": 0, "level_delta_db_p05": 0, "correlation": 0.9},
            "bgm_removal",
        ),
        (
            "audio",
            "musicProbes",
            [
                {
                    "start_seconds": 0,
                    "before_dbfs": -30,
                    "after_dbfs": -35,
                    "drop_db": 5,
                    "passed": True,
                }
            ],
            "bgm_removal",
        ),
        (
            "audio",
            "musicProbes",
            [
                {
                    "start_seconds": 60,
                    "before_dbfs": -30,
                    "after_dbfs": -42,
                    "drop_db": 12,
                    "passed": True,
                }
            ],
            "bgm_removal",
        ),
        (
            "audio",
            "musicProbes",
            [
                {
                    "start_seconds": 0,
                    "before_dbfs": -30,
                    "after_dbfs": -42,
                    "drop_db": 13,
                    "passed": True,
                }
            ],
            "bgm_removal",
        ),
        (
            "audio",
            "musicProbes",
            [
                {
                    "start_seconds": 0,
                    "before_dbfs": -30,
                    "after_dbfs": -42,
                    "drop_db": 12,
                    "passed": True,
                }
            ]
            * 2,
            "bgm_removal",
        ),
    ],
)
def test_stage_failures(case: Case, name: str, field: str, value: object, stage: str) -> None:
    case.data[name][field] = value
    case.refresh()
    assert set(case.check()) == {stage}


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("quality", "measuredFrames", 29),
        ("quality", "maximumResidualFraction", None),
        ("encoding", "fullDecodeVerified", False),
        ("encoding", "videoPayloadIdentical", False),
        ("encoding", "frames", True),
        ("encoding", "frames", 89),
        ("encoding", "editedVoiceSHA256", "wrong"),
        ("plan", "durationFrames", 89),
        ("plan", "startFrame", 1),
        ("plan", "startFrame", False),
        ("plan", "motionSHA256", "wrong"),
        ("plan", "reuseAlphaResponse", "bad.json"),
        ("encoding-verified", "fullDecodeVerified", False),
        ("alpha-layout-audit", "boundsAndGapsVerified", False),
        ("alpha-submit.response", "ok", False),
        ("plan", "captions", []),
        ("plan", "captions", [{"start": 0, "end": "1", "lines": ["Synthetic"]}]),
        ("plan", "captions", [{"start": "0", "end": "1", "lines": [42]}]),
        ("plan", "captions", [{"start": "0", "end": "4", "lines": ["Synthetic"]}]),
        ("plan", "captions", [{"start": "1/31", "end": "1", "lines": ["Synthetic"]}]),
        (
            "audio",
            "speech",
            {"level_delta_db_p50": float("nan"), "level_delta_db_p05": 0, "correlation": 1},
        ),
    ],
)
def test_invalid_common_evidence_is_fail_closed(
    case: Case, name: str, field: str, value: object
) -> None:
    case.data[name][field] = value
    case.refresh()
    with pytest.raises(ReplayError):
        case.check()
    assert set(assess_delivery(case.movie, root=case.root)) == set(REQUIRED_STAGES)


def test_changed_srt_rejected_even_when_its_receipt_is_rehashed(case: Case) -> None:
    case.movie.with_suffix(".srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nAltered\n", encoding="utf-8"
    )
    case.data["captions"]["srtSHA256"] = digest(case.movie.with_suffix(".srt"))
    case.refresh()
    assert set(case.check()) == {"transcription"}


def test_changed_part_plan_without_rebinding_is_rejected(case: Case) -> None:
    case.path("plan").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="part plan changed"):
        case.check()


def test_changed_receipt_does_not_fall_back_to_legacy(case: Case) -> None:
    case.path("audio").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="receipt changed"):
        case.check()
    assert set(assess_delivery(case.movie, root=case.root)) == set(REQUIRED_STAGES)


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", True), ("recording", "replay-2"), ("extra", 1), ("sourceSHA256", "bad")],
)
def test_index_identity(case: Case, field: str, value: object) -> None:
    data = record(json.loads(case.index.read_text(encoding="utf-8")))
    data[field] = value
    case.index.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="index does not match"):
        case.check()


@pytest.mark.parametrize(
    "descriptor", [{}, {"path": "../../foreign.json", "sha256": "x"}, {"path": "", "sha256": "x"}]
)
def test_invalid_reference(case: Case, descriptor: dict[str, object]) -> None:
    data = record(json.loads(case.index.read_text(encoding="utf-8")))
    data["masks"] = descriptor
    case.index.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        case.check()


def test_linked_receipt_rejected(case: Case) -> None:
    original = case.path("cuts")
    target = original.with_name("real-cuts.json")
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(ReplayError, match="linked"):
        case.check()


def test_non_object_receipt_rejected(case: Case) -> None:
    case.path("cuts").write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError, match="object"):
        case.check()


def test_private_account_storage_is_never_hashed(case: Case) -> None:
    private = case.root.parents[1] / "private"
    private.mkdir()
    secret = private / "synthetic.secret"
    secret.write_bytes(b"not a credential")
    store = Evidence(case.root, case.source)
    with pytest.raises(ReplayError, match="private account storage"):
        store.digest(secret)


def test_outside_storage_refused(case: Case, tmp_path: Path) -> None:
    store = Evidence(case.root / "deeper", case.source)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"synthetic")
    with pytest.raises(ReplayError, match="authorized storage"):
        store.digest(outside)


def test_bad_input_binding_rejected(case: Case) -> None:
    case.data["cuts"]["inputs"] = {
        str(case.path("manifest")): digest(case.path("manifest")),
        str(case.source): "wrong",
    }
    case.save("cuts")
    data = record(json.loads(case.index.read_text(encoding="utf-8")))
    data["cuts"] = {
        "path": str(case.path("cuts").relative_to(case.root)),
        "sha256": digest(case.path("cuts")),
    }
    case.index.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="input changed"):
        case.check()


def test_changes_during_verification_are_rejected(case: Case) -> None:
    store = Evidence(case.root, case.source)
    store.digest(case.source)
    case.source.write_bytes(b"changed during verification")
    with pytest.raises(ReplayError, match="changed during verification"):
        store.verify_unchanged()


def test_bare_quality_approval_cannot_pass(case: Case) -> None:
    case.data["masks"].pop("qualityEvidence")
    case.save("masks")
    data = record(json.loads(case.index.read_text(encoding="utf-8")))
    data["masks"] = {
        "path": str(case.path("masks").relative_to(case.root)),
        "sha256": digest(case.path("masks")),
    }
    case.index.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="Invalid native evidence reference"):
        case.check()


def test_quality_cannot_bind_only_some_processed_inputs(case: Case) -> None:
    case.data["quality"]["inputs"] = {str(case.source): digest(case.source)}
    case.save("quality")
    case.data["masks"]["qualityEvidence"] = {
        "path": str(case.path("quality").relative_to(case.root)),
        "sha256": digest(case.path("quality")),
    }
    case.save("masks")
    data = record(json.loads(case.index.read_text(encoding="utf-8")))
    data["masks"] = {
        "path": str(case.path("masks").relative_to(case.root)),
        "sha256": digest(case.path("masks")),
    }
    case.index.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="does not bind every processed input"):
        case.check()


def test_explicit_hash_identical_alpha_reuse(case: Case) -> None:
    case.data["plan"]["reuseAlphaResponse"] = str(case.path("alpha-submit.response"))
    case.refresh()
    assert case.check() == {}


def test_configured_storage_root_is_authorized(case: Case, tmp_path: Path) -> None:
    external = tmp_path / "ssd"
    external.mkdir()
    evidence = external / "receipt.json"
    evidence.write_text('{"ok": true}', encoding="utf-8")
    store = Evidence(case.root, case.source, external)
    path, data = store.reference({"path": "storage:receipt.json", "sha256": digest(evidence)})
    assert path == external / "receipt.json"
    assert data == {"ok": True}


def test_storage_reference_without_storage_root_is_refused(case: Case) -> None:
    store = Evidence(case.root, case.source)
    with pytest.raises(ReplayError, match="unconfigured storage root"):
        store.reference({"path": "storage:receipt.json", "sha256": "0" * 64})


def test_storage_reference_cannot_escape_storage_root(case: Case, tmp_path: Path) -> None:
    external = tmp_path / "ssd"
    external.mkdir()
    store = Evidence(case.root, case.source, external)
    with pytest.raises(ReplayError, match="belong to this recording"):
        store.reference({"path": "storage:../outside.json", "sha256": "0" * 64})
