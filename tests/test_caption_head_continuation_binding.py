import json
from pathlib import Path

import pytest
from test_caption_consumer_binding import native as native_template
from test_caption_consumer_binding import ref, root, write
from test_caption_native_continuation import config

from replay import caption_head_continuation_binding as binder
from replay import caption_native_continuation, caption_reflow_verify, clock_contract, workflow
from replay.core import ReplayError

__all__ = ["config", "native_template", "root"]


@pytest.fixture
def bound_request(config: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = config.parent
    prepared = base / "prepared"
    caption_native_continuation.prepare(config, prepared)
    voice = base / "voice.wav"
    voice.write_bytes(b"synthetic, never decoded")
    voice_receipt = write(base / "voice-receipt.json", {})
    clock = write(
        base / "clock-config.json",
        {
            "artifacts": {
                "outputAudio": ref(voice),
                "receipt": ref(voice_receipt),
                "cuts": ref(base / "cuts.json"),
                "audioClock": ref(base / "clock.json"),
            }
        },
    )
    clock_receipt = write(base / "clock-receipt.json", {"outputFrames": 30})
    reflow = write(base / "replacement-reflow.json", {"settings": {}})
    metrics = write(
        base / "replacement-metrics.json", json.loads((base / "metrics.json").read_text())
    )
    face_config = write(
        base / "face-config.json",
        {"source": str(base / "source.bin"), "sourceSHA256": ref(base / "source.bin")["sha256"]},
    )
    face = write(base / "face-index.json", {})
    clean = write(base / "clean.json", {})
    template = write(
        base / "head.json",
        {
            "version": 1,
            "name": "Synthetic",
            "faceIndex": ref(face),
            "cleanPaths": {
                "assemblyManifest": str(clean),
                "assemblyReceipt": str(clean),
                "cleanIndex": str(clean),
            },
        },
    )
    monkeypatch.setattr(binder, "face_references", lambda path: ((face_config, base),))
    monkeypatch.setattr(
        binder, "clean_source", lambda value: (base / "source.bin", base / "source.bin", 1)
    )
    monkeypatch.setattr(clock_contract, "verify", lambda path: {"outputFrames": 30})
    monkeypatch.setattr(caption_reflow_verify, "verify", lambda *args, **kwargs: 1)
    return write(
        base / "request.json",
        {
            "version": 1,
            "nativeConfig": ref(config),
            "nativePreparationReceipt": str(prepared / "receipt.json"),
            "headTemplate": ref(template),
            "artifactPaths": {
                "voice": str(voice),
                "voiceReceipt": str(voice_receipt),
                "clockConfig": str(clock),
                "clockReceipt": str(clock_receipt),
                "reflow": str(reflow),
                "measurementReference": str(metrics),
                "segmentIndex": str(base / "recognition-inputs/segments.json"),
            },
        },
    )


def test_explicit_replacement_and_normal_graph(bound_request: Path) -> None:
    output = bound_request.parent / "bound-head"
    assert binder.main([str(bound_request), str(output)]) == 0
    config = json.loads((output / "head-config.json").read_text())
    assert config["reflow"]["path"].endswith("replacement-reflow.json")
    assert config["headReferencePolicy"] == "automatic-face-relative"
    assert config["collectLayoutFailures"] is True
    assert "referenceHeadTop" not in config
    assert (
        workflow.main(
            [
                str(output / "workflow.json"),
                "--state",
                str(bound_request.parent / "head-state"),
                "--check",
            ]
        )
        == 0
    )
    assert not (output / "render").exists()
    receipt = json.loads((output / "binding-receipt.json").read_text())
    assert receipt["deliveryComplete"] is False
    assert receipt["observationsRecovered"] is False


@pytest.mark.parametrize("field,value", [("version", 2), ("artifactPaths", {})])
def test_invalid_schema(bound_request: Path, field: str, value: object) -> None:
    data = json.loads(bound_request.read_text())
    data[field] = value
    write(bound_request, data)
    with pytest.raises(ReplayError):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_wrong_clock(bound_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clock_contract, "verify", lambda path: {"outputFrames": 31})
    with pytest.raises(ReplayError, match="Voice clock"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_wrong_voice(bound_request: Path) -> None:
    data = json.loads(bound_request.read_text())
    data["artifactPaths"]["voice"] = str(bound_request.parent / "source.bin")
    write(bound_request, data)
    with pytest.raises(ReplayError, match="different selected artifacts"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_wrong_cleanup(bound_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        binder,
        "clean_source",
        lambda value: (bound_request.parent / "wrong", bound_request.parent / "source.bin", 1),
    )
    with pytest.raises(ReplayError, match="Cleanup belongs"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_input_changes_during_verification(
    bound_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def verify(path: Path) -> dict[str, object]:
        write(bound_request.parent / "replacement-metrics.json", {"changed": True})
        return {"outputFrames": 30}

    monkeypatch.setattr(clock_contract, "verify", verify)
    with pytest.raises(ReplayError):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_token_mapping_required(bound_request: Path) -> None:
    write(
        bound_request.parent / "replacement-reflow.json",
        {"settings": {"tokenSources": {"map": "hash"}}},
    )
    with pytest.raises(ReplayError, match="requires its token mapping"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_token_mapping_preserved(bound_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = json.loads(bound_request.read_text())
    source_map = write(bound_request.parent / "tokens-map.json", {})
    data["artifactPaths"]["tokenSources"] = str(source_map)
    write(bound_request, data)
    write(
        bound_request.parent / "replacement-reflow.json",
        {"settings": {"tokenSources": {"map": "hash"}}},
    )
    monkeypatch.setattr(binder, "load_sources", lambda path, entries: ((), {"map": "hash"}))
    binder.prepare(bound_request, bound_request.parent / "bound-head")
    output = json.loads((bound_request.parent / "bound-head/head-config.json").read_text())
    assert output["tokenSources"]["path"].endswith("tokens-map.json")


def _edit_template(request: Path, change: dict[str, object]) -> None:
    data = json.loads(request.read_text())
    template = Path(data["headTemplate"]["path"])
    value = json.loads(template.read_text())
    value.update(change)
    write(template, value)
    data["headTemplate"] = ref(template)
    write(request, data)


def test_incomplete_cleanup_lineage(bound_request: Path) -> None:
    _edit_template(bound_request, {"cleanPaths": {"cleanIndex": str(bound_request)}})
    with pytest.raises(ReplayError, match="Incomplete cleanup lineage"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_blank_project_name(bound_request: Path) -> None:
    _edit_template(bound_request, {"name": "  "})
    with pytest.raises(ReplayError, match="explicit project name"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_face_observations_of_another_source(bound_request: Path) -> None:
    write(bound_request.parent / "face-config.json", {"source": "other", "sourceSHA256": "0" * 64})
    with pytest.raises(ReplayError, match="another source"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")


def test_invalid_template_schema(bound_request: Path) -> None:
    _edit_template(bound_request, {"version": 2})
    with pytest.raises(ReplayError, match="Invalid head template"):
        binder.prepare(bound_request, bound_request.parent / "bound-head")
