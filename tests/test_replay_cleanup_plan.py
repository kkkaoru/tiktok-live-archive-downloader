import hashlib
import json
from pathlib import Path

import pytest

from replay import replay_cleanup_plan as plan
from replay import workflow


def document(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def cleanup_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    source = document(tmp_path / "source.mov", {"synthetic": True})
    native = document(tmp_path / "native", {"synthetic": True})
    (tmp_path / "native").chmod(0o700)
    model = document(tmp_path / "model", {"synthetic": True})
    clock = document(
        tmp_path / "clock.json",
        {
            "streams": [{"time_base": "1/1000"}],
            "frames": [
                {"pts": tick, "duration": 40}
                for tick in (0, 33, 73, 106, 146, 179, 219, 252, 292, 325)
            ],
        },
    )
    metadata = document(
        tmp_path / "metadata.json",
        {
            "sourceSHA256": source["sha256"],
            "frameClock": clock,
            "streams": [{"codec_type": "video", "width": 720, "height": 1280}],
        },
    )
    path = tmp_path / "cleanup_request.json"
    document(
        path,
        {
            "version": 1,
            "source": source,
            "native": native,
            "decoder": native,
            "model": model,
            "frameClock": clock,
            "sourceMetadata": metadata,
            "profileSamples": 10,
            "chunkFrames": 3,
            "workers": 2,
            "nativeTimescale": 60000,
        },
    )
    return path


@pytest.fixture
def prepared(cleanup_request: Path) -> Path:
    output = cleanup_request.parent / "prepared"
    plan.prepare(cleanup_request, output)
    return output


@pytest.fixture
def observed(cleanup_request: Path, prepared: Path) -> Path:
    data = json.loads(cleanup_request.read_text())
    profile = prepared / "profile"
    profile.mkdir()
    document(
        profile / "inputs.json",
        {
            "source": data["source"]["path"],
            "backend": "native-cli",
            "collectorSHA256": plan._pin(Path(plan.__file__).with_name("subtitle_observations.py"))[
                "sha256"
            ],
            "parserSHA256": plan._pin(Path(plan.__file__).with_name("caption_io.py"))["sha256"],
            "planSHA256": plan._pin(prepared / "profile-times.json")["sha256"],
            "sourceSHA256": data["source"]["sha256"],
            "nativeSHA256": data["native"]["sha256"],
            "nativeExecutable": data["native"]["path"],
            "measurementTool": "video_text_recognize",
            "fullFrame": True,
        },
    )
    frames = [
        {
            "requestedTimeSeconds": tick,
            "actualTimeSeconds": tick,
            "width": 720,
            "height": 1280,
            "lines": [
                {
                    "text": "字幕" + text,
                    "confidence": 0.9,
                    "region": {"x": 100, "y": 500, "width": 520, "height": 32},
                }
            ],
        }
        for tick, text in zip(plan.load(cleanup_request).times, "春夏秋冬雨雲花風月空", strict=True)
    ]
    first = document(profile / "0000.result.json", frames[:8])
    second = document(profile / "0001.result.json", frames[8:])
    document(
        profile / "complete.json",
        {
            "observations": 10,
            "results": {"0000.result.json": first["sha256"], "0001.result.json": second["sha256"]},
        },
    )
    return prepared


def test_vfr_prepare_resume_and_original_workflow_check(
    cleanup_request: Path, prepared: Path
) -> None:
    before = (prepared / "workflow.json").read_bytes()
    plan.prepare(cleanup_request, prepared)
    assert before == (prepared / "workflow.json").read_bytes()
    assert json.loads((prepared / "physical-clock.json").read_text()) == {
        "pts": [0, 33, 73, 106, 146, 179, 219, 252, 292, 325],
        "timeBase": "1/1000",
        "lastDuration": 40,
    }
    assert (
        workflow.main(
            [str(prepared / "workflow.json"), "--state", str(prepared / "state"), "--check"]
        )
        == 0
    )


def test_profile_handoff_runs_original_factory(cleanup_request: Path, observed: Path) -> None:
    assert plan.main(["handoff", str(cleanup_request), str(observed)]) == 0
    result = json.loads((observed / "classification.json").read_text())
    assert result["canvas"] == [720, 1280]
    assert result["wholeDetectionVerified"] is False
    assert result["removalQualityApproved"] is False
    assert (
        json.loads((observed / "cleanup" / "plan.json").read_text())["wholeSourcePlanned"] is True
    )
    assert (
        workflow.main(
            [
                str(observed / "cleanup" / "worker-00.json"),
                "--state",
                str(observed / "state"),
                "--check",
            ]
        )
        == 0
    )
    plan.handoff(cleanup_request, observed)


@pytest.mark.parametrize(
    "change", [{"version": 2}, {"workers": True}, {"profileSamples": 513}, {"chunkFrames": 0}]
)
def test_request_budget_rejected(cleanup_request: Path, change: dict[str, object]) -> None:
    data = json.loads(cleanup_request.read_text())
    data.update(change)
    document(cleanup_request, data)
    with pytest.raises(ValueError):
        plan.load(cleanup_request)


@pytest.mark.parametrize("value", [None, "relative", "/wrong/../path", "bad\x00path"])
def test_path_refused(value: object) -> None:
    with pytest.raises(ValueError):
        plan.reference({"path": value, "sha256": "a" * 64})


def test_bad_digest() -> None:
    with pytest.raises(ValueError):
        plan.reference({"path": "/canonical", "sha256": "unknown"})


@pytest.mark.parametrize("role", ["source", "native", "decoder", "model"])
def test_missing_role_fails_before_creation(cleanup_request: Path, role: str) -> None:
    data = json.loads(cleanup_request.read_text())
    data[role]["path"] = str(cleanup_request.parent / "absent")
    document(cleanup_request, data)
    with pytest.raises(ValueError, match="Missing cleanup"):
        plan.prepare(cleanup_request, cleanup_request.parent / "never-created")
    assert not (cleanup_request.parent / "never-created").exists()


@pytest.mark.parametrize("role", ["source", "native", "decoder", "model"])
@pytest.mark.parametrize("kind", ["empty", "directory", "symlink"])
def test_nonregular_or_empty_role_refused(cleanup_request: Path, role: str, kind: str) -> None:
    data = json.loads(cleanup_request.read_text())
    invalid = cleanup_request.parent / "invalid"
    if kind == "directory":
        invalid.mkdir()
    elif kind == "symlink":
        invalid.symlink_to(cleanup_request)
    else:
        invalid.touch()
    data[role]["path"] = str(invalid)
    document(cleanup_request, data)
    with pytest.raises(ValueError, match="regular file|canonical path"):
        plan.prepare(cleanup_request, cleanup_request.parent / "never-created")
    assert not (cleanup_request.parent / "never-created").exists()


def test_native_grid_rejects_rounding(cleanup_request: Path) -> None:
    data = json.loads(cleanup_request.read_text())
    data["nativeTimescale"] = 25
    document(cleanup_request, data)
    with pytest.raises(ValueError, match="exactly representable"):
        plan.load(cleanup_request)


def test_missing_profile_is_not_synthesized(cleanup_request: Path, prepared: Path) -> None:
    with pytest.raises(FileNotFoundError):
        plan.profile(cleanup_request, prepared)
    assert not (prepared / "classification.json").exists()


def test_original_handoff_rechecks_actual_media(cleanup_request: Path, observed: Path) -> None:
    (cleanup_request.parent / "source.mov").write_bytes(b"changed synthetic source")
    with pytest.raises(ValueError, match="Source media changed"):
        plan.handoff(cleanup_request, observed)


def test_unobserved_native_grid_refuses_whole_handoff(
    cleanup_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = json.loads(cleanup_request.read_text())
    data.update(profileSamples=1, nativeTimescale=25)
    document(cleanup_request, data)
    output = cleanup_request.parent / "sampled"
    plan.prepare(cleanup_request, output)
    # Isolate the already-covered profile producer to reach the independent
    # whole-clock guard; it must run before the original full-source planner.
    monkeypatch.setattr(plan, "profile", lambda path, output: None)
    with pytest.raises(ValueError, match="Full-source OCR PTS"):
        plan.handoff(cleanup_request, output)
    assert not (output / "cleanup").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"sourceSHA256": "0" * 64},
        {"streams": None},
        {"streams": []},
        {"streams": [{"codec_type": "video", "width": 0, "height": 100}]},
    ],
)
def test_measured_metadata_required(cleanup_request: Path, change: dict[str, object]) -> None:
    data = json.loads(cleanup_request.read_text())
    path = cleanup_request.parent / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(change)
    data["sourceMetadata"] = document(path, metadata)
    document(cleanup_request, data)
    with pytest.raises(ValueError):
        plan.load(cleanup_request)


def test_nonzero_origin_rejected_before_output(cleanup_request: Path) -> None:
    data = json.loads(cleanup_request.read_text())
    clock = json.loads((cleanup_request.parent / "clock.json").read_text())
    clock["frames"][0]["pts"] = 1
    data["frameClock"] = document(cleanup_request.parent / "clock.json", clock)
    document(cleanup_request, data)
    with pytest.raises(ValueError, match="zero-origin"):
        plan.prepare(cleanup_request, cleanup_request.parent / "out")
    assert not (cleanup_request.parent / "out").exists()


def test_checkpoint_changed_refuses_resume(cleanup_request: Path, prepared: Path) -> None:
    document(prepared / "profile-times.json", [0])
    with pytest.raises(ValueError, match="checkpoint changed"):
        plan.prepare(cleanup_request, prepared)


@pytest.mark.parametrize(
    "artifact,value,match",
    [
        ("request-pin.json", {}, "request changed"),
        ("physical-clock.json", {}, "clock checkpoint"),
        ("code-pins.json", [], "code checkpoint"),
    ],
)
def test_changed_proof(
    cleanup_request: Path, observed: Path, artifact: str, value: object, match: str
) -> None:
    document(observed / artifact, value)
    with pytest.raises(ValueError, match=match):
        plan.verify_profile(cleanup_request, observed)


def test_changed_code_pin(cleanup_request: Path, observed: Path) -> None:
    pins = json.loads((observed / "code-pins.json").read_text())
    pins[0]["sha256"] = "0" * 64
    document(observed / "code-pins.json", pins)
    with pytest.raises(ValueError, match="producer code"):
        plan.verify_profile(cleanup_request, observed)


def test_missing_code_binding_refused(cleanup_request: Path, observed: Path) -> None:
    pins = json.loads((observed / "code-pins.json").read_text())
    document(observed / "code-pins.json", pins[1:])
    with pytest.raises(ValueError, match="code closure"):
        plan.verify_profile(cleanup_request, observed)


def test_collector_provenance_required(cleanup_request: Path, observed: Path) -> None:
    path = observed / "profile" / "inputs.json"
    data = json.loads(path.read_text())
    data["collectorSHA256"] = "0" * 64
    document(path, data)
    with pytest.raises(ValueError, match="collector provenance"):
        plan.verify_profile(cleanup_request, observed)


@pytest.mark.parametrize("field,value", [("actualTimeSeconds", 0.01), ("width", 1080)])
def test_wrong_actual_ocr_refused(
    cleanup_request: Path, observed: Path, field: str, value: float
) -> None:
    batch = observed / "profile" / "0000.result.json"
    frames = json.loads(batch.read_text())
    frames[0][field] = value
    changed = document(batch, frames)
    receipt = observed / "profile" / "complete.json"
    data = json.loads(receipt.read_text())
    data["results"]["0000.result.json"] = changed["sha256"]
    document(receipt, data)
    with pytest.raises(ValueError, match="clock/canvas"):
        plan.verify_profile(cleanup_request, observed)


def test_sampling_one_and_different_geometry(cleanup_request: Path) -> None:
    data = json.loads(cleanup_request.read_text())
    data["profileSamples"] = 1
    metadata = json.loads((cleanup_request.parent / "metadata.json").read_text())
    metadata["streams"][0].update(width=1920, height=1080)
    data["sourceMetadata"] = document(cleanup_request.parent / "metadata.json", metadata)
    document(cleanup_request, data)
    loaded = plan.load(cleanup_request)
    assert loaded.times == (0.0,)
    assert (loaded.width, loaded.height) == (1920, 1080)


def test_no_metadata_escape(
    cleanup_request: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    with pytest.raises(ValueError, match="inside checkout"):
        plan.load(cleanup_request)
