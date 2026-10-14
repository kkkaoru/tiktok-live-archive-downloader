"""Candidate calibration evidence never substitutes for delivery approval."""

import json
import sys
from pathlib import Path

import pytest
from test_delivery_caption_audit import config as delivery_config
from test_motion_calibration_verify import bounds as bounds
from test_motion_calibration_verify import calibration as calibration
from test_motion_calibration_verify import config as geometry_config
from test_motion_calibration_verify import font as font
from test_motion_calibration_verify import md5 as md5
from test_motion_calibration_verify import probe as probe
from test_motion_calibration_verify import raw_plan as raw_plan
from test_motion_caption import sequence_template as sequence_template
from test_motion_caption import text_template as text_template

from replay import delivery_caption_audit, motion_calibration_verify
from replay import motion_geometry_candidate as module
from replay.delivery_render import Config
from replay.recordings import record
from replay.stages import file_digest

__all__ = ["delivery_config", "geometry_config"]


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def settings(geometry_config: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = geometry_config.parent
    asset = root / "font.ttf"
    asset.write_bytes(b"synthetic font identity")
    historical = record(json.loads((root / "calibration").read_bytes()))
    evidence = historical["evidence"]
    assert isinstance(evidence, list)
    historical["evidence"] = [*evidence, ref(asset)]
    save(root / "calibration", historical)
    receipt = root / "geometry-receipt.json"
    motion_calibration_verify.verify(geometry_config, receipt)
    native = root / "native-tool"
    native.write_bytes(b"synthetic executable")
    path = root / "candidate-config.json"
    save(
        path,
        {
            "geometryConfig": ref(geometry_config),
            "geometryReceipt": ref(receipt),
            "native": ref(native),
            "fontAsset": ref(asset),
        },
    )

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert command[0] == "/usr/bin/env"
        assert command[-3:-1] == ("measure-media", "caption_text_measure")
        assert timeout == 60
        log.write_bytes((root / "paddedResponse").read_bytes())

    monkeypatch.setattr(module, "execute_command", execute)
    return path


def test_new_seven_key_envelope_is_provisional(settings: Path) -> None:
    path = module.prepare(settings, settings.parent / "new")
    result = record(json.loads(path.read_bytes()))
    assert set(result) == {
        "version",
        "canvas",
        "style",
        "nativeFont",
        "padding",
        "template",
        "evidence",
    }
    assert result["version"] == 1
    assert result["padding"] == 24
    scope = record(json.loads((path.parent / "scope.json").read_bytes()))
    assert scope["measuredStrings"] == ["A"]
    assert scope["freshNativeBoundsMatched"] is True
    assert scope["freshNativeFontMetricsMatched"] is True
    assert scope["fontAssetResolutionVerified"] is False
    assert scope["colourApproved"] is False
    assert scope["pixelGradientHoldApproved"] is False
    assert scope["captionGeometryApproved"] is False
    assert scope["deliveryComplete"] is False
    measured = record(json.loads((path.parent / "envelopes.json").read_bytes()))
    assert measured["envelopes"] == [
        {"text": "A", "bounds": {"left": -25.0, "top": -114.0, "right": 25.0, "bottom": 16.0}}
    ]
    with pytest.raises(FileExistsError):
        module.prepare(settings, path.parent)


@pytest.mark.parametrize(
    "field,value", [("extra", 1), ("native", {}), ("native", {"path": "no-such", "sha256": "x"})]
)
def test_invalid_configuration(settings: Path, field: str, value: object) -> None:
    raw = record(json.loads(settings.read_bytes()))
    raw[field] = value
    save(settings, raw)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.prepare(settings, settings.parent / "bad")


def test_pin_changed(settings: Path) -> None:
    (settings.parent / "native-tool").write_bytes(b"changed executable")
    with pytest.raises(ValueError, match="reference changed"):
        module.prepare(settings, settings.parent / "bad")


@pytest.mark.parametrize("change", ["failed", "unbound", "asset"])
def test_invalid_geometry_evidence(settings: Path, change: str) -> None:
    raw = record(json.loads(settings.read_bytes()))
    path = settings.parent / "geometry-receipt.json"
    receipt = record(json.loads(path.read_bytes()))
    if change == "failed":
        receipt["geometryContained"] = False
    elif change == "unbound":
        record(receipt["inputs"]).pop(str(settings.parent / "config.json"))
    else:
        asset = settings.parent / "other.ttf"
        asset.write_bytes(b"unbound font")
        raw["fontAsset"] = ref(asset)
    save(path, receipt)
    raw["geometryReceipt"] = ref(path)
    save(settings, raw)
    with pytest.raises(ValueError, match="geometry-only|does not bind|Font asset"):
        module.prepare(settings, settings.parent / "bad")


@pytest.mark.parametrize("change", ["bounds", "font"])
def test_fresh_mismatch(settings: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        response = record(json.loads((settings.parent / "paddedResponse").read_bytes()))
        if change == "bounds":
            response["bounds"] = [{"text": "A", "width": 51, "height": 130, "lineWidths": [51]}]
        else:
            record(response["font"])["pointSize"] = 81
        save(log, response)

    monkeypatch.setattr(module, "execute_command", execute)
    with pytest.raises(ValueError, match="Fresh native"):
        module.prepare(settings, settings.parent / "bad")
    assert not (settings.parent / "bad/envelope.json").exists()


def test_cli(settings: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = settings.parent / "cli"
    monkeypatch.setattr(sys, "argv", ["candidate", str(settings), str(output)])
    module.main()
    assert (output / "envelope.json").is_file()


def test_formal_consumer_does_not_accept_geometry_scope(
    settings: Path, delivery_config: Config, tmp_path: Path
) -> None:
    envelope = module.prepare(settings, settings.parent / "candidate")
    scope = record(json.loads((envelope.parent / "scope.json").read_bytes()))
    # Even a candidate flag cannot turn a geometry receipt into a render receipt.
    scope["candidateOnly"] = True
    save(delivery_config.work / "alpha-jobs-state/alpha-0000-0000.json", scope)
    old = tmp_path / "old.srt"
    old.write_text("old", encoding="utf-8")
    result = delivery_caption_audit.audit(
        delivery_config.path,
        old,
        tmp_path / "formal-audit",
        phase="audit",
        cache=tmp_path / "audit-cache",
        ffmpeg=delivery_config.path,
        ffprobe=delivery_config.path,
    )
    assert result["accepted"] is False
    assert record(result["checks"])["fullAlphaCoverage"] is False
