"""Synthetic native responses and calibration artifacts; no apps or recordings."""

import json
import runpy
import sys
from pathlib import Path

import pytest

from replay import motion_native_envelope as module
from replay import stages
from replay.caption_native_layout import NativeCaptionLayout, NativeCaptionStyle
from replay.core import ReplayError
from replay.motion_native_envelope import CalibratedMotionEnvelope
from replay.recordings import record
from replay.stages import file_digest
from replay.vertical_captions import Box


def fake_native(argv: tuple[str, ...], output: Path, timeout: int) -> None:
    assert argv[1:3] == ("measure-media", "caption_text_measure")
    assert timeout == 60
    request = record(json.loads(Path(argv[-1]).read_text(encoding="utf-8")))
    texts = request["texts"]
    assert isinstance(texts, list)
    strings = [text for text in texts if isinstance(text, str)]
    assert len(strings) == len(texts)
    output.write_text(
        json.dumps(
            {
                "motionRenderVerified": False,
                "font": {
                    "postScriptName": "Synthetic",
                    "family": "Synthetic",
                    "face": "Regular",
                    "pointSize": 7.5,
                    "lineAdvance": 10.0,
                    "lineAscent": 9.0,
                    "renderedAscent": 8.0,
                },
                "bounds": [
                    {
                        "text": text,
                        "width": len(text) * 10 + 4,
                        "height": 12,
                        "lineWidths": [len(text) * 10 + 4],
                    }
                    for text in strings
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def calibration(tmp_path: Path) -> Path:
    template, evidence = tmp_path / "template.moti", tmp_path / "evidence.json"
    template.write_bytes(b"synthetic template, not Motion XML")
    evidence.write_bytes(b'{"synthetic":true}')
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def layout(tmp_path: Path, calibration: Path) -> NativeCaptionLayout:
    executable = tmp_path / "native"
    executable.write_bytes(b"synthetic native executable")
    return NativeCaptionLayout(
        executable=executable,
        directory=tmp_path / "metrics",
        style=NativeCaptionStyle("Synthetic", 10, 1, 0.5, 14),
        width=50,
        height=26,
        evidence=(calibration.parent / "evidence.json",),
        execute=fake_native,
    )


@pytest.fixture
def envelope(layout: NativeCaptionLayout, calibration: Path) -> CalibratedMotionEnvelope:
    return CalibratedMotionEnvelope(
        layout,
        calibration=calibration,
        calibration_sha256=file_digest(calibration),
        canvas=(100, 200),
    )


def test_signed_origin_uses_each_measurement_and_observed_motion_leading(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
) -> None:
    assert envelope("ab") == Box(-12, -10, 12, 2)
    assert envelope("ab\ncd") == Box(-12, -10, 12, 16)
    assert envelope("ab\ncd") == Box(-12, -10, 12, 16)
    assert len(tuple(layout.directory.glob("*.response.json"))) == 2
    assert len(envelope.verify_inputs()) == 13
    assert envelope.canvas == (100, 200)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("extra", 1),
        ("padding", True),
        ("padding", -1),
        ("padding", float("inf")),
        ("padding", 10**1000),
        ("padding", 65),
        ("canvas", [101, 200]),
        ("canvas", [True, 200]),
        ("canvas", [0, 200]),
        ("canvas", [9000, 200]),
        ("canvas", [100]),
        ("style", {}),
        ("nativeFont", {}),
        ("evidence", []),
        ("evidence", {}),
        ("evidence", [{}] * 33),
        ("template", {}),
    ],
)
def test_invalid_calibration_fields_fail(
    layout: NativeCaptionLayout,
    calibration: Path,
    field: str,
    value: object,
) -> None:
    raw = record(json.loads(calibration.read_text(encoding="utf-8")))
    raw[field] = value
    calibration.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        CalibratedMotionEnvelope(
            layout,
            calibration=calibration,
            calibration_sha256=file_digest(calibration),
            canvas=(100, 200),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("postScriptName", "Other"),
        ("family", ""),
        ("face", 5),
        ("face", "x" * 201),
        ("face", "a\0b"),
        ("pointSize", 0),
        ("renderedAscent", False),
        ("lineAdvance", 11),
    ],
)
def test_invalid_or_mismatched_font_fails(
    layout: NativeCaptionLayout,
    calibration: Path,
    field: str,
    value: object,
) -> None:
    raw = record(json.loads(calibration.read_text(encoding="utf-8")))
    record(raw["nativeFont"])[field] = value
    calibration.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        CalibratedMotionEnvelope(
            layout,
            calibration=calibration,
            calibration_sha256=file_digest(calibration),
            canvas=(100, 200),
        )


def test_style_mismatch_fails(layout: NativeCaptionLayout, calibration: Path) -> None:
    raw = record(json.loads(calibration.read_text(encoding="utf-8")))
    record(raw["style"])["rim"] = 1
    calibration.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="style"):
        CalibratedMotionEnvelope(
            layout,
            calibration=calibration,
            calibration_sha256=file_digest(calibration),
            canvas=(100, 200),
        )


def test_alias_reference_fails(layout: NativeCaptionLayout, calibration: Path) -> None:
    raw = record(json.loads(calibration.read_text(encoding="utf-8")))
    raw["evidence"] = [raw["template"]]
    calibration.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="aliases"):
        CalibratedMotionEnvelope(
            layout,
            calibration=calibration,
            calibration_sha256=file_digest(calibration),
            canvas=(100, 200),
        )


def test_stale_calibration_pin_fails(layout: NativeCaptionLayout, calibration: Path) -> None:
    with pytest.raises(ValueError, match="changed"):
        CalibratedMotionEnvelope(
            layout, calibration=calibration, calibration_sha256="0" * 64, canvas=(100, 200)
        )


def test_changed_template_stops_verification(
    envelope: CalibratedMotionEnvelope,
    calibration: Path,
) -> None:
    envelope("ab")
    (calibration.parent / "template.moti").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        envelope.verify_inputs()


def test_changed_response_stops_verification(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
) -> None:
    envelope("ab")
    (layout.directory / "batch-000000.response.json").write_bytes(b"{}")
    with pytest.raises(ReplayError, match="inputs changed"):
        envelope.verify_inputs()


def test_fresh_native_font_must_match_calibration(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def different_font(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(argv, output, timeout)
        raw = record(json.loads(output.read_text(encoding="utf-8")))
        record(raw["font"])["pointSize"] = 8
        output.write_text(json.dumps(raw), encoding="utf-8")

    monkeypatch.setattr(layout, "_execute", different_font)
    with pytest.raises(ValueError, match="Fresh native font metrics"):
        envelope("ab")


def test_missing_font_response_fails(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
) -> None:
    layout.measure("ab")
    response = layout.directory / "batch-000000.response.json"
    response.rename(layout.directory / "retained-response.json")
    with pytest.raises(FileNotFoundError):
        envelope("ab")


def test_prewarmed_cache_must_have_calibrated_request_settings(
    layout: NativeCaptionLayout,
    calibration: Path,
) -> None:
    layout.style = NativeCaptionStyle("Synthetic", 10, 2, 0.5, 14)
    layout.measure("ab")
    layout.style = NativeCaptionStyle("Synthetic", 10, 1, 0.5, 14)
    with pytest.raises(ValueError, match="Native request settings differ"):
        CalibratedMotionEnvelope(
            layout,
            calibration=calibration,
            calibration_sha256=file_digest(calibration),
            canvas=(100, 200),
        )


def test_failed_attempt_gap_cannot_hide_later_response(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        raise RuntimeError("synthetic process failure before response")

    assert envelope("ab") == Box(-12, -10, 12, 2)
    monkeypatch.setattr(layout, "_execute", fail)
    with pytest.raises(RuntimeError, match="synthetic process failure"):
        layout.measure("failed")
    monkeypatch.setattr(layout, "_execute", fake_native)
    layout.measure("cd")
    with pytest.raises(ValueError, match="lacks completed provenance"):
        envelope("cd")
    with pytest.raises(ValueError, match="lacks completed provenance"):
        envelope.verify_inputs()
    assert sorted(p.name for p in layout.directory.glob("*.response.json")) == [
        "batch-000000.response.json",
        "batch-000002.response.json",
    ]


def test_unbound_cache_dimensions_are_not_font_evidence(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(layout, "measure", lambda text: (24.0, 12.0))
    with pytest.raises(ValueError, match="evidence is missing"):
        envelope("ab")


def test_unexpected_request_schema_fails(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envelope, "_producer_bytes", lambda path: b'{"extra":true}')
    with pytest.raises(ValueError, match="Invalid calibrated native measurement request"):
        envelope._check_request(layout.directory / "batch-000000.request.json")


def test_response_budget_is_bounded(
    envelope: CalibratedMotionEnvelope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "MAX_CACHE", 0)
    with pytest.raises(ValueError, match="response budget"):
        envelope("ab")


@pytest.mark.parametrize(
    "field,value",
    [
        ("style", NativeCaptionStyle("Synthetic", 10, 2, 0.5, 14)),
        ("directory", Path("/changed-directory")),
        ("executable", Path("/changed-executable")),
    ],
)
def test_mutated_native_configuration_fails_before_measurement(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.setattr(layout, field, value)
    with pytest.raises(ValueError, match="configuration changed"):
        envelope("ab")
    with pytest.raises(ValueError, match="configuration changed"):
        envelope.verify_inputs()


def test_reference_byte_budget(
    layout: NativeCaptionLayout,
    calibration: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "MAX_EVIDENCE_BYTES", 2)
    with pytest.raises(ValueError, match="byte budget"):
        CalibratedMotionEnvelope(
            layout,
            calibration=calibration,
            calibration_sha256=file_digest(calibration),
            canvas=(100, 200),
        )


def test_cli_writes_bound_candidates_not_approvals(
    layout: NativeCaptionLayout,
    calibration: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = calibration.parent / "cli"
    monkeypatch.setattr(stages, "execute_command", fake_native)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion_native_envelope",
            str(layout.executable),
            str(calibration),
            str(output),
            "--calibration-sha256",
            file_digest(calibration),
            "--canvas",
            "100",
            "200",
            "--text",
            "ab",
            "--text",
            "ab\ncd",
        ],
    )
    runpy.run_path(module.__file__, run_name="__main__")
    report = record(json.loads((output / "envelopes.json").read_text(encoding="utf-8")))
    assert report["envelopes"] == [
        {"text": "ab", "bounds": {"left": -12, "top": -10, "right": 12, "bottom": 2}},
        {"text": "ab\ncd", "bounds": {"left": -12, "top": -10, "right": 12, "bottom": 16}},
    ]
    assert report["motionRenderVerified"] is False
    assert report["captionGeometryApproved"] is False
    assert report["deliveryComplete"] is False
    assert (output / "envelopes.json").stat().st_mode & 0o777 == 0o600
    assert "inputs" in report


def test_cli_text_budget_precedes_output_creation(
    layout: NativeCaptionLayout,
    calibration: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = calibration.parent / "cli"
    monkeypatch.setattr(module, "MAX_PLAN_TEXTS", 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion_native_envelope",
            str(layout.executable),
            str(calibration),
            str(output),
            "--calibration-sha256",
            file_digest(calibration),
            "--canvas",
            "100",
            "200",
            "--text",
            "ab",
        ],
    )
    with pytest.raises(ValueError, match="text budget"):
        module.main()
    assert not output.exists()


def test_conflicting_response_binding_fails(
    envelope: CalibratedMotionEnvelope,
    layout: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope("ab")
    inputs = layout.verify_inputs()
    inputs[str(layout.directory / "batch-000000.response.json")] = "0" * 64
    monkeypatch.setattr(layout, "verify_inputs", lambda: inputs)
    with pytest.raises(ValueError, match="inputs changed"):
        envelope.verify_inputs()
