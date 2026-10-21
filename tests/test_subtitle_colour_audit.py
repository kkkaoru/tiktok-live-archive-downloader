"""Audited frames of real (lavfi) videos give exact unchanged-caption-colour counts."""

import subprocess
from pathlib import Path

import pytest

from replay import subtitle_colour_audit as audit

SIZE = (160, 120)
BOX = (20, 40, 50, 50)


def _video(path: Path, *, pink: bool) -> Path:
    graph = "color=c=gray:s=160x120:r=30:d=1"
    if pink:
        graph += ",drawbox=x=20:y=40:w=30:h=10:color=0xFF69B4:t=fill"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", graph, "-pix_fmt", "yuv444p",
         "-c:v", "libx264", "-qp", "0", "-y", str(path)],
        check=True,
        timeout=60,
    )  # fmt: skip
    return path


def _boxes() -> list[list[tuple[int, int, int, int]]]:
    return [[BOX] for _ in range(30)]


def test_unchanged_caption_colour_inside_the_box_is_counted(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    counts = audit.colour_counts(
        ffmpeg="ffmpeg", source=original, source_start=0.0, cleaned=original,
        indices=[0, 7, 29], boxes=_boxes(), size=SIZE,
    )  # fmt: skip
    assert counts == [300, 300, 300]


def test_repaired_frames_count_zero(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    cleaned = _video(tmp_path / "cleaned.mp4", pink=False)
    counts = audit.colour_counts(
        ffmpeg="ffmpeg", source=original, source_start=0.0, cleaned=cleaned,
        indices=[3, 10], boxes=_boxes(), size=SIZE,
    )  # fmt: skip
    assert counts == [0, 0]


def test_source_offset_seeks_before_counting(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    counts = audit.colour_counts(
        ffmpeg="ffmpeg", source=original, source_start=0.5, cleaned=original,
        indices=[0], boxes=_boxes(), size=SIZE,
    )  # fmt: skip
    assert counts == [300]


def test_indices_must_be_ascending_and_unique(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    with pytest.raises(ValueError, match="ascending"):
        audit.colour_counts(
            ffmpeg="ffmpeg", source=original, source_start=0.0, cleaned=original,
            indices=[5, 5], boxes=_boxes(), size=SIZE,
        )  # fmt: skip


def test_truncated_decode_is_rejected(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    with pytest.raises(ValueError, match="before every audited frame"):
        audit.colour_counts(
            ffmpeg="ffmpeg", source=original, source_start=0.0, cleaned=original,
            indices=[0, 60], boxes=[[BOX]] * 61, size=SIZE,
        )  # fmt: skip


def test_decoder_failure_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="before every audited frame"):
        audit.colour_counts(
            ffmpeg="ffmpeg", source=tmp_path / "missing.mp4", source_start=0.0,
            cleaned=tmp_path / "missing.mp4", indices=[0], boxes=_boxes(), size=SIZE,
        )  # fmt: skip


def test_selector_groups_arithmetic_runs_and_single_frames() -> None:
    assert audit._selector([0, 5, 10, 899]) == ("between(n\\,0\\,10)*not(mod(n-0\\,5))+eq(n\\,899)")


def test_selector_handles_a_single_index() -> None:
    assert audit._selector([7]) == "eq(n\\,7)"


def test_long_audit_runs_decode_with_a_compact_expression(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    counts = audit.colour_counts(
        ffmpeg="ffmpeg", source=original, source_start=0.0, cleaned=original,
        indices=list(range(0, 30, 2)), boxes=_boxes(), size=SIZE,
    )  # fmt: skip
    assert counts == [300] * 15


def test_source_pink_counts_measure_caption_colour_inside_each_line_box(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    counts = audit.source_pink_counts(
        ffmpeg="ffmpeg", source=original, source_start=0.0,
        lines=[
            (7, (20.0, 40.0, 30.0, 10.0)),
            (7, (100.0, 40.0, 30.0, 10.0)),
            (3, (20.0, 40.0, 5.0, 5.0)),
        ],
        size=SIZE,
    )  # fmt: skip
    assert counts == [300, 0, 36]


def test_source_pink_counts_need_at_least_one_line(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    with pytest.raises(ValueError, match="No lines"):
        audit.source_pink_counts(
            ffmpeg="ffmpeg", source=original, source_start=0.0, lines=[], size=SIZE
        )


def test_band_rows_and_reused_bands_give_the_full_frame_counts(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    # An odd top: the crop really starts one row higher, as the band audit's decode does.
    band = (7, 76)
    bands = list(audit._frames("ffmpeg", original, 0.0, [0, 9], SIZE, band))
    counts = audit.colour_counts(
        ffmpeg="ffmpeg", source=original, source_start=0.0, cleaned=original,
        indices=[0, 9], boxes=_boxes(), size=SIZE, band=band, cleaned_bands=bands,
    )  # fmt: skip
    assert [frame.shape for frame in bands] == [(76, 160, 3), (76, 160, 3)]
    assert counts == [300, 300]


def test_source_bands_decoded_ahead_replace_the_source_decode(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    band = (7, 76)
    bands = list(audit._frames("ffmpeg", original, 0.0, [0, 9], SIZE, band))
    counts = audit.colour_counts(
        ffmpeg="ffmpeg", source=tmp_path / "missing.mp4", source_start=0.0,
        cleaned=tmp_path / "missing.mp4", indices=[0, 9], boxes=_boxes(), size=SIZE, band=band,
        cleaned_bands=bands, source_bands=bands,
    )  # fmt: skip
    assert counts == [300, 300]


def test_boxes_near_the_band_edge_are_counted_on_full_frames(tmp_path: Path) -> None:
    original = _video(tmp_path / "original.mp4", pink=True)
    counts = audit.colour_counts(
        ffmpeg="ffmpeg", source=original, source_start=0.0, cleaned=original,
        indices=[0], boxes=_boxes(), size=SIZE, band=(30, 40), cleaned_bands=[],
        source_bands=[],
    )  # fmt: skip
    assert counts == [300]


def test_a_crop_starts_on_an_even_row() -> None:
    assert (audit.crop_origin(7), audit.crop_origin(8)) == (6, 8)


def test_boxes_inside_the_band_leave_the_reach_on_both_sides() -> None:
    assert audit.inside((6, 76), [[(20, 38, 50, 50)]]) is True
    assert audit.inside((6, 76), [[(20, 37, 50, 50)]]) is False
    assert audit.inside((6, 76), [[(20, 40, 50, 51)]]) is False
