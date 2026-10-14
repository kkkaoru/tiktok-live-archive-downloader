import json
from fractions import Fraction
from pathlib import Path

import pytest
from test_caption_visibility_candidate import candidate as candidate

from replay import caption_visibility_candidate as candidate_visibility
from replay import fcp_caption_visibility as visibility
from replay.fcp_caption_srt import caption_cues
from replay.fcp_project import (
    Caption,
    CaptionStyle,
    CaptionTemplate,
    Cut,
    MediaInfo,
    Timeline,
    build_fcpxml,
)
from replay.stages import file_digest


@pytest.fixture
def binding(candidate: tuple[Path, Path, Path]) -> tuple[visibility.VisibilityBinding, Path]:
    config, overlay, _ = candidate
    candidate_visibility.run(config, overlay)
    manifest = overlay.parent / "binding.json"
    manifest.write_text(
        json.dumps(
            {
                "config": {"path": str(config), "sha256": file_digest(config)},
                "overlay": {"path": str(overlay), "sha256": file_digest(overlay)},
            }
        ),
        encoding="utf-8",
    )
    result = visibility.read_binding(
        json.loads(manifest.read_text(encoding="utf-8")),
        reflow_inputs={str(overlay): file_digest(overlay)},
    )
    return result, manifest


def test_explicit_binding_retains_original_and_false_approvals(
    binding: tuple[visibility.VisibilityBinding, Path],
) -> None:
    actual, _ = binding
    assert actual.regions == ((1.0, 4.2),)
    assert actual.proof["speechAccuracyVerified"] is False
    assert actual.proof["completeCaptionsVerified"] is False
    assert actual.proof["deliveryComplete"] is False
    assert len(actual.inputs) == 8


def test_reflow_must_pin_exact_visible_input(candidate: tuple[Path, Path, Path]) -> None:
    config, overlay, _ = candidate
    candidate_visibility.run(config, overlay)
    with pytest.raises(ValueError, match="reflow must pin"):
        visibility.read_binding(
            {
                "config": {"path": str(config), "sha256": file_digest(config)},
                "overlay": {"path": str(overlay), "sha256": file_digest(overlay)},
            },
            reflow_inputs={},
        )


def test_changed_overlay_is_not_accepted_even_with_new_pin(
    candidate: tuple[Path, Path, Path],
) -> None:
    config, overlay, _ = candidate
    candidate_visibility.run(config, overlay)
    overlay.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        visibility.read_binding(
            {
                "config": {"path": str(config), "sha256": file_digest(config)},
                "overlay": {"path": str(overlay), "sha256": file_digest(overlay)},
            },
            reflow_inputs={str(overlay): file_digest(overlay)},
        )


def test_head_hold_is_cleared_after_repage_without_retiming_other_words(
    binding: tuple[visibility.VisibilityBinding, Path],
) -> None:
    actual, _ = binding
    assert visibility.apply_binding(
        [
            {"text": "はい", "start": 0.0, "end": 1.08, "scale": {"factor": 1}},
            {"text": "です", "start": 5.0, "end": 5.28},
        ],
        actual,
    ) == [
        {"text": "はい", "start": 0.0, "end": 1.0, "scale": {"factor": 1}},
        {"text": "です", "start": 5.0, "end": 5.28},
    ]


@pytest.mark.parametrize(
    "start,end",
    [(Fraction(1), Fraction(2)), (Fraction(0), Fraction(11, 10)), (Fraction(4), Fraction(5))],
)
def test_title_crossing_hidden_region_is_not_allowed(
    binding: tuple[visibility.VisibilityBinding, Path], start: Fraction, end: Fraction
) -> None:
    actual, _ = binding
    with pytest.raises(ValueError, match="intersects"):
        visibility.interval_audit(((start, end, "語"),), actual)


def test_exact_display_boundaries_preserve_timestamp_representation(
    binding: tuple[visibility.VisibilityBinding, Path],
) -> None:
    actual, _ = binding
    audit = visibility.interval_audit(
        ((Fraction(0), Fraction(1), "前"), (Fraction(21, 5), Fraction(5), "後")), actual
    )
    assert audit["titleIntervals"] == 2
    assert audit["displayRegions"] == [{"start": "1", "end": "21/5"}]
    assert audit["recognizedClocksChanged"] is False
    assert audit["visibleTitleIntervalsVerified"] is True
    assert audit["nativeRenderedCaptionVisibilityVerified"] is False
    assert audit["completeCaptionsVerified"] is False


def test_native_fraction_clock_matches_float_recognition_without_extra_shift() -> None:
    actual = visibility.VisibilityBinding(((1 / 30, 4 / 30),), {}, {})
    audit = visibility.interval_audit(
        ((Fraction(0), Fraction(1, 30), "前"), (Fraction(2, 15), Fraction(1), "後")), actual
    )
    assert audit["displayRegions"] == [{"start": "1/30", "end": "2/15"}]
    assert audit["recognizedClocksChanged"] is False


@pytest.mark.parametrize("rate", [24, 30, 60])
@pytest.mark.parametrize("text_objects", [2, 3])
def test_subframe_exclusion_uses_exact_end_through_actual_fcp_writer_and_cuts(
    tmp_path: Path, rate: int, text_objects: int
) -> None:
    actual = visibility.VisibilityBinding(((5.01, 5.5),), {}, {})
    raw = [{"text": "一行目\n二行目", "start": 4.95, "end": 5.08, "x": 540, "bottom": 420}]
    clipped = visibility.apply_binding(raw, actual, rate=rate)
    assert clipped == [
        {"text": "一行目\n二行目", "start": 4.95, "end": Fraction(5), "x": 540, "bottom": 420}
    ]
    assert raw == [{"text": "一行目\n二行目", "start": 4.95, "end": 5.08, "x": 540, "bottom": 420}]
    source = tmp_path / "synthetic.mp4"
    timeline = Timeline(
        rate,
        1080,
        1920,
        (
            Cut(source, Fraction(100), Fraction(0), Fraction(3)),
            Cut(source, Fraction(200), Fraction(3), Fraction(3)),
        ),
        (),
        (
            Caption(
                Fraction(str(clipped[0]["start"])),
                Fraction(str(clipped[0]["end"])),
                540,
                420,
                ("一行目", "二行目"),
                "#005A9C",
            ),
        ),
        CaptionStyle(82, "#005A9C", 3, "#FFFFFF", 1, "#000000"),
    )
    output, _ = build_fcpxml(
        timeline,
        name="Synthetic explicit visibility",
        sources={source: MediaInfo(Fraction(300), True, False)},
        template=CaptionTemplate(
            "Synthetic", "synthetic", "file:///synthetic.moti", text_objects=text_objects
        ),
    )
    xml = tmp_path / "synthetic.fcpxml"
    xml.write_bytes(output)
    assert caption_cues(xml)[0][1] == Fraction(5)
    assert caption_cues(xml)[0][2] == "一行目\n二行目"
    assert visibility.audit_xml(xml, actual)["visibleTitleIntervalsVerified"] is True


def test_subframe_exclusion_without_complete_frame_is_rejected() -> None:
    actual = visibility.VisibilityBinding(((1.01, 2),), {}, {})
    with pytest.raises(ValueError, match="no complete display frame"):
        visibility.apply_binding([{"text": "語", "start": 1.005, "end": 1.08}], actual)


@pytest.mark.parametrize("rate", [0, 121, True])
def test_unbounded_or_boolean_frame_rate_is_rejected(rate: int) -> None:
    actual = visibility.VisibilityBinding(((1, 2),), {}, {})
    with pytest.raises(ValueError, match="bounded integer"):
        visibility.apply_binding([{"text": "語", "start": 0, "end": 1.08}], actual, rate=rate)


def test_frame_exact_exclusion_retains_an_existing_single_frame() -> None:
    actual = visibility.VisibilityBinding(((1, 2),), {}, {})
    assert visibility.apply_binding([{"text": "語", "start": 0.96, "end": 1.08}], actual) == [
        {"text": "語", "start": 0.96, "end": Fraction(1)}
    ]


def test_onset_just_after_last_frame_boundary_is_not_moved_to_create_a_frame() -> None:
    actual = visibility.VisibilityBinding(((1, 2),), {}, {})
    raw = [{"text": "語", "start": 29 / 30, "end": 1.08}]
    with pytest.raises(ValueError, match="no complete display frame"):
        visibility.apply_binding(raw, actual)
    assert raw == [{"text": "語", "start": 29 / 30, "end": 1.08}]


@pytest.fixture
def xml(binding: tuple[visibility.VisibilityBinding, Path]) -> Path:
    _, manifest = binding
    path = manifest.parent / "semantic.fcpxml"
    path.write_text(
        '<fcpxml><library><event><project><sequence duration="6s" tcStart="0s"><spine>'
        '<asset-clip offset="0s" start="100s" duration="3s">'
        '<title offset="100s" start="3600s" duration="1s">'
        "<text>前</text><text>前</text><text>前</text></title>"
        '</asset-clip><asset-clip offset="3s" start="200s" duration="3s">'
        '<title offset="1006/5s" start="3600s" duration="1s"><text>後</text></title>'
        "</asset-clip></spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    return path


def test_final_xml_audit_composes_source_jumps_into_output_clock(
    binding: tuple[visibility.VisibilityBinding, Path], xml: Path
) -> None:
    actual, _ = binding
    audit = visibility.audit_xml(xml, actual)
    assert audit["titleIntervals"] == 2
    assert audit["visibleTitleIntervalsVerified"] is True
    assert audit["deliveryComplete"] is False


def test_final_xml_detects_reintroduced_hold(
    binding: tuple[visibility.VisibilityBinding, Path], xml: Path
) -> None:
    actual, _ = binding
    xml.write_text(
        xml.read_text(encoding="utf-8").replace('duration="1s"', 'duration="27/25s"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="intersects"):
        visibility.audit_xml(xml, actual)


def test_cli_independently_audits_explicit_head_xml(
    binding: tuple[visibility.VisibilityBinding, Path],
    xml: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actual, manifest = binding
    reflow = manifest.parent / "reflow.json"
    reflow.write_text(json.dumps({"inputs": actual.inputs}), encoding="utf-8")
    assert visibility.main([str(manifest), str(reflow), str(xml)]) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["visibleTitleIntervalsVerified"] is True
    assert audit["nativeRenderedCaptionVisibilityVerified"] is False
