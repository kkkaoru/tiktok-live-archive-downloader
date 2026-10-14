import json
import runpy
import sys

import pytest

from replay import caption_contrast
from replay.caption_contrast import (
    SampledContrast,
    audit_sampled_contrast,
    contrast_ratio,
    relative_luminance,
    require_white_contrast,
)
from replay.core import ReplayError


def test_black_and_white_have_extreme_relative_luminance() -> None:
    assert relative_luminance((0.0, 0.0, 0.0)) == 0.0
    assert relative_luminance((1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_white_against_black_is_21_in_either_order() -> None:
    assert contrast_ratio((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)) == pytest.approx(21.0)
    assert contrast_ratio((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == pytest.approx(21.0)


def test_identical_colours_have_ratio_one() -> None:
    assert contrast_ratio((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) == 1.0


def test_linear_segment_includes_the_0_04045_threshold() -> None:
    assert relative_luminance((0.04045, 0.04045, 0.04045)) == pytest.approx(
        0.0031308049536, rel=1e-9
    )
    assert relative_luminance((0.02, 0.0, 0.0)) == pytest.approx(0.00032910216718, rel=1e-9)


def test_power_segment_above_threshold() -> None:
    assert relative_luminance((0.5, 0.5, 0.5)) == pytest.approx(0.21404114, abs=1e-8)


def test_specified_single_fill_ratio_against_white() -> None:
    assert contrast_ratio((1.0, 1.0, 1.0), (0.0, 90 / 255, 156 / 255)) == pytest.approx(
        7.1367, abs=1e-4
    )


def test_wrong_component_count_is_rejected() -> None:
    with pytest.raises(ReplayError, match="exactly three components"):
        relative_luminance((0.0, 0.0))
    with pytest.raises(ReplayError, match="exactly three components"):
        relative_luminance((0.0, 0.0, 0.0, 0.0))


def test_non_finite_and_out_of_range_components_are_rejected() -> None:
    with pytest.raises(ReplayError, match="finite and within 0..1"):
        relative_luminance((float("nan"), 0.0, 0.0))
    with pytest.raises(ReplayError, match="finite and within 0..1"):
        relative_luminance((0.0, float("inf"), 0.0))
    with pytest.raises(ReplayError, match="finite and within 0..1"):
        relative_luminance((0.0, 0.0, 1.01))
    with pytest.raises(ReplayError, match="finite and within 0..1"):
        contrast_ratio((1.0, 1.0, 1.0), (-0.01, 0.0, 0.0))


def test_audit_passes_with_unrounded_group_minima() -> None:
    report = audit_sampled_contrast(
        upper=((0.0, 0.0, 0.0), (0.0, 90 / 255, 156 / 255)),
        lower=((0.0, 0.0, 0.0),),
        intermediate=((0.0, 90 / 255, 156 / 255),),
    )
    assert isinstance(report, SampledContrast)
    assert report.upper_minimum == pytest.approx(7.1367, abs=1e-4)
    assert report.lower_minimum == pytest.approx(21.0)
    assert report.intermediate_minimum == pytest.approx(7.1367, abs=1e-4)
    assert report.failures == ()


def test_audit_names_upper_and_lower_failures() -> None:
    report = audit_sampled_contrast(
        upper=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        lower=((0.5, 0.5, 0.5),),
        intermediate=((0.0, 0.0, 0.0),),
    )
    assert report.upper_minimum == 1.0
    assert report.lower_minimum == pytest.approx(3.9767, abs=1e-3)
    assert report.intermediate_minimum == pytest.approx(21.0)
    assert report.failures == ("upper", "lower")


def test_audit_names_intermediate_failure_only() -> None:
    report = audit_sampled_contrast(
        upper=((0.0, 0.0, 0.0),),
        lower=((0.0, 0.0, 0.0),),
        intermediate=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    )
    assert report.failures == ("intermediate",)


def test_audit_rejects_lower_and_intermediate_colours_that_only_meet_aa() -> None:
    report = audit_sampled_contrast(
        upper=((0.0, 0.0, 0.0),),
        lower=((0x76 / 255, 0x76 / 255, 0x76 / 255),),
        intermediate=((0x77 / 255, 0x77 / 255, 0x77 / 255),),
    )
    assert report.lower_minimum == pytest.approx(4.54, abs=0.01)
    assert report.intermediate_minimum == pytest.approx(4.48, abs=0.01)
    assert report.failures == ("lower", "intermediate")


def test_audit_rejects_empty_upper_group() -> None:
    with pytest.raises(ReplayError, match="No sampled upper caption pixels"):
        audit_sampled_contrast(upper=(), lower=((0.0, 0.0, 0.0),), intermediate=((0.0, 0.0, 0.0),))


def test_audit_rejects_empty_lower_group() -> None:
    with pytest.raises(ReplayError, match="No sampled lower caption pixels"):
        audit_sampled_contrast(upper=((0.0, 0.0, 0.0),), lower=(), intermediate=((0.0, 0.0, 0.0),))


def test_audit_rejects_empty_intermediate_group() -> None:
    with pytest.raises(ReplayError, match="No sampled intermediate caption pixels"):
        audit_sampled_contrast(upper=((0.0, 0.0, 0.0),), lower=((0.0, 0.0, 0.0),), intermediate=())


def test_selected_srgb_colour_is_checked_against_white() -> None:
    assert require_white_contrast(((0.0, 90 / 255, 156 / 255),)) == pytest.approx(
        (7.1367,), abs=1e-4
    )


@pytest.mark.parametrize("color", [(0.7, 0.83, 0.87), (0.86, 0.93, 0.95), (0.35, 0.35, 0.35)])
def test_any_bright_stop_is_rejected_without_rounding(color: tuple[float, float, float]) -> None:
    with pytest.raises(ReplayError, match="below 7:1"):
        require_white_contrast(((0.0, 0.0, 0.0), color))


def test_selection_requires_colours() -> None:
    with pytest.raises(ReplayError, match="at least one"):
        require_white_contrast(())


def test_boolean_colour_is_not_numeric() -> None:
    with pytest.raises(ReplayError, match="finite"):
        require_white_contrast(((True, 0.0, 0.0),))


def test_explicit_encoded_srgb_is_not_also_assumed_linear() -> None:
    assert require_white_contrast(((0.3, 0.3, 0.3),))[0] > 7


def test_selection_cli_reports_every_ratio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["caption-contrast", "#005A9C", "000000"])
    runpy.run_path(caption_contrast.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert report["ratios"] == pytest.approx([7.1367, 21.0], abs=1e-4)
    assert report["passed"] is True
    assert report["renderedContrastVerified"] is False


def test_selection_cli_rejects_low_contrast(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["caption-contrast", "#005A9C", "#DAEDF2"])
    with pytest.raises(SystemExit, match="1"):
        caption_contrast.main()
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_rendered_near_black_cannot_substitute_for_selected_cyan() -> None:
    assert (
        caption_contrast.sampled_colour_correspondence(
            ((36 / 255, 85 / 255, 106 / 255),), ((1 / 255, 4 / 255, 8 / 255),)
        )
        is False
    )


def test_sample_correspondence_allows_one_8bit_code_step() -> None:
    assert (
        caption_contrast.sampled_colour_correspondence(
            ((36 / 255, 85 / 255, 106 / 255),), ((37 / 255, 84 / 255, 107 / 255),)
        )
        is True
    )
    assert (
        caption_contrast.sampled_colour_correspondence(
            ((36 / 255, 85 / 255, 106 / 255),), ((38 / 255, 85 / 255, 106 / 255),)
        )
        is False
    )


def test_sample_correspondence_rejects_unpaired_stops() -> None:
    with pytest.raises(ReplayError, match="one-to-one"):
        caption_contrast.sampled_colour_correspondence(
            ((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        )


@pytest.mark.parametrize("sample", [(), ((float("nan"), 0.0, 0.0),), ((1.0, 1.0, 1.0),)])
def test_sample_correspondence_requires_valid_aaa_samples(
    sample: tuple[tuple[float, float, float], ...],
) -> None:
    with pytest.raises(ReplayError):
        caption_contrast.sampled_colour_correspondence(((0.0, 0.0, 0.0),), sample)


@pytest.mark.parametrize("rendered", ["#010408", "#FFFFFF"])
def test_selection_cli_rejects_wrong_or_bright_rendered_colour(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], rendered: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["caption-contrast", "#24556A", "--rendered-colors", rendered])
    with pytest.raises(SystemExit, match="1"):
        caption_contrast.main()
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["suppliedSampleCorrespondence"] is False
    assert report["renderedContrastVerified"] is False


def test_selection_cli_checks_samples_without_claiming_render_verification(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["caption-contrast", "#24556A", "--rendered-colors", "#24556A"]
    )
    caption_contrast.main()
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["suppliedSampleCorrespondence"] is True
    assert report["renderedContrastVerified"] is False


@pytest.mark.parametrize("color", ["FFF", "gg0000", "#00000000"])
def test_selection_cli_rejects_invalid_hex(monkeypatch: pytest.MonkeyPatch, color: str) -> None:
    monkeypatch.setattr(sys, "argv", ["caption-contrast", color])
    with pytest.raises(SystemExit, match="2"):
        caption_contrast.main()
