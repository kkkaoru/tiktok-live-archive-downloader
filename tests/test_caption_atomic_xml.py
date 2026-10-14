from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_caption_linguistics import synthetic_dictionary as synthetic_dictionary
from test_fcp_caption_retarget import calibration as calibration
from test_fcp_caption_retarget import faces as faces
from test_fcp_caption_retarget import template as template

from replay.caption_native_paging import assemble, paged_cues
from replay.caption_timing import SpokenToken, display_time
from replay.delivery_captions import Cut
from replay.fcp_caption_retarget import Cue, build, selected_spans
from replay.fcp_caption_srt import caption_cues
from replay.fcp_project import CaptionTemplate, MediaInfo
from replay.fcp_tracking_caption import FacePositionSeries, TrackingCalibration, TrackingPosition


class SyntheticMeasurements:
    width = 200.0
    expected_point_size: float | None = 10.0

    def prepare_lines(self, texts: Iterable[str]) -> None:
        tuple(texts)

    def measure(self, text: str) -> tuple[float, float]:
        return max(len(line) * 10 + 4 for line in text.splitlines()), 12.0


@pytest.mark.parametrize("assembled", [False, True])
def test_recognized_fragments_to_exact_xml_with_segment_offset_and_source_gap(
    tmp_path: Path,
    faces: FacePositionSeries,
    calibration: TrackingCalibration,
    template: CaptionTemplate,
    assembled: bool,
) -> None:
    tokens: tuple[SpokenToken, ...] = (
        SpokenToken("フ", 0.1, 0.15, "/synthetic/0001.raw.json:cut-1"),
        SpokenToken("ォ", 0.15, 0.2, "/synthetic/0002.raw.json:cut-1"),
        SpokenToken("ロ", 0.2, 0.25, "/synthetic/0002.raw.json:cut-1"),
        SpokenToken("ワ", 0.25, 0.3, "/synthetic/0002.raw.json:cut-1"),
        SpokenToken("ー", 0.3, 0.4, "/synthetic/0002.raw.json:cut-1"),
        SpokenToken("空", 1.2, 1.4, "/synthetic/0002.raw.json:cut-2"),
    )
    if assembled:
        tokens = tuple(
            replace(token, utterance=f"000000000000:{index}:cut-{2 if index == 5 else 1}")
            for index, token in enumerate(tokens)
        )
    cues, hidden, widened = paged_cues(
        tokens, layout=SyntheticMeasurements(), rate=30, duration=Fraction(2)
    )
    assert (hidden, widened) == (0, 0)
    offset = Fraction(1, 3)
    parts: tuple[dict[str, object], ...] = (
        {
            "captions": [
                {
                    "text": cue.text,
                    "start": float(display_time(cue.start) + offset),
                    "end": float(display_time(cue.end) + offset),
                }
                for cue in cues
            ]
        },
    )
    rows = assemble(parts, rate=30)
    assembled_cues = tuple(
        Cue(
            display_time(float(str(row["start"]))),
            display_time(float(str(row["end"]))),
            str(row["text"]),
        )
        for row in rows
    )
    xml = build(
        assembled_cues,
        selected_spans((Cut(60, 90, 0), Cut(120, 150, 30)), rate=30, source_window=None),
        source=tmp_path / "synthetic.mp4",
        media=MediaInfo(Fraction(5), True, False),
        template=template,
        faces=replace(
            faces,
            positions=(*faces.positions, TrackingPosition(Fraction(5), 320, 720)),
            coverage_end=Fraction(6),
        ),
        calibration=calibration,
        rate=30,
        name="Synthetic atomic caption clock",
    )
    target = tmp_path / "atomic.fcpxml"
    target.write_bytes(xml)
    assert caption_cues(target) == (
        (Fraction(13, 30), Fraction(5, 6), "フォロワー"),
        (Fraction(23, 15), Fraction(11, 6), "空"),
    )
