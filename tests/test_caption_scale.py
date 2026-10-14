"""Scale evidence keeps normal cues compatible and rejects forged geometry."""

import math
from dataclasses import asdict

import pytest

from replay.caption_scale import CaptionScale, read_scale, scale_fields
from replay.core import ReplayError


def test_scale_roundtrip() -> None:
    scale = CaptionScale(400, 40, 100, 20, 8, 0.5)
    assert scale.factor == 0.25
    assert read_scale(scale_fields(scale)) == scale
    assert read_scale({}) is None
    assert scale_fields(None) == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 0),
        ("width", math.nan),
        ("height", True),
        ("fragments", 1),
        ("fragments", 2.0),
        ("maximum_fragment_seconds", 3),
        ("available_width", 1000),
    ],
)
def test_invalid_scale_evidence(field: str, value: object) -> None:
    raw = asdict(CaptionScale(400, 40, 100, 40, 8, 0.5))
    raw.pop("provenance")
    raw[field] = value
    with pytest.raises(ReplayError):
        read_scale({"indivisibleScale": raw})


def test_unknown_scale_field_rejected() -> None:
    with pytest.raises(ReplayError, match="fields"):
        read_scale({"indivisibleScale": {"factor": 0.5}})


def test_anchor_scale_has_distinct_serialized_provenance() -> None:
    scale = CaptionScale(400, 40, 100, 40, 8, 0.5, "recognition-anchor-group")
    fields = scale_fields(scale)
    assert set(fields) == {"anchoredGroupScale"}
    assert read_scale(fields) == scale
    with pytest.raises(ReplayError, match="Ambiguous"):
        read_scale({"indivisibleScale": {}, "anchoredGroupScale": {}})
