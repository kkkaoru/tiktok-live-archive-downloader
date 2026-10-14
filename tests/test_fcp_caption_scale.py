"""Uniform scale survives reflow, head geometry and native XML without retiming."""

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_fcp_caption_retarget import reflow_document as reflow_document
from test_fcp_caption_retarget import template as template
from test_fcp_head_near import paired as paired
from test_fcp_pre_layout import geometry as geometry
from test_fcp_pre_layout import internal_geometry as internal_geometry

from replay.caption_scale import CaptionScale, scale_fields
from replay.fcp_caption_retarget import Cue, RetainedSpan, build, read_reflow
from replay.fcp_caption_srt import caption_cues
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_pre_layout import GeometryValues, geometry_factor
from replay.fcp_project import CaptionTemplate, MediaInfo
from replay.fcp_tracking_caption import TrackingCalibration


@pytest.mark.parametrize("provenance", ["indivisible-word", "recognition-anchor-group"])
def test_scale_reflow_to_native_xml_preserves_text_clock_and_bottom_anchor(
    tmp_path: Path,
    reflow_document: dict[str, object],
    template: CaptionTemplate,
    paired: FaceScaleSeries,
    provenance: str,
    geometry: GeometryValues,
    internal_geometry: GeometryValues,
) -> None:
    geometry = internal_geometry
    template = replace(template, pre_layout_geometry=geometry)
    scale = CaptionScale(400, 400, 200, 400, 4, 0.5, provenance)
    reflow_document["captions"] = [
        {"start": 0, "end": 2, "text": "こぉぉぉ", **scale_fields(scale)}
    ]
    path = tmp_path / "reflow.json"
    path.write_text(json.dumps(reflow_document), encoding="utf-8")
    spans = (RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),)
    cues = read_reflow(path, spans, source=Path("/source.mp4"), source_sha256="source-sha")
    assert cues == (Cue(Fraction(0), Fraction(2), "こぉぉぉ", scale),)
    content = build(
        cues,
        spans,
        source=tmp_path / "media.mp4",
        media=MediaInfo(Fraction(2), True, False),
        template=template,
        faces=paired.bottoms,
        scale_faces=paired,
        measure_caption=lambda text: (400, 400),
        calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1)),
        rate=30,
        name="Synthetic scale",
    )
    output = tmp_path / "scaled.fcpxml"
    output.write_bytes(content)
    assert caption_cues(output) == ((Fraction(0), Fraction(2), "こぉぉぉ"),)
    root = ET.fromstring(content)
    transform = root.find(".//title/adjust-transform")
    assert transform is not None
    assert transform.get("scale") is None
    title = root.find(".//title")
    assert title is not None
    assert geometry_factor(title, geometry) == 0.5
    first = transform.find(".//keyframe")
    assert first is not None
    assert first.get("value") == "0 25.2083333333"
