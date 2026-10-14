import json
import struct
from fractions import Fraction
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.fcp_visible_arc import _points, read_visible_arc
from replay.stages import file_digest


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def fixture(tmp_path: Path, fault: str) -> dict[str, object]:
    image = tmp_path / "frame.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"0" * 8 + struct.pack(">II", 100 if fault != "image" else 90, 200)
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"reviewerKind": "agent-visual-review", "visibleArcReviewed": fault != "review"})
    )
    config = tmp_path / "config.json"
    points = [[10, 50], [20, 52], [30, 53]]
    config.write_text(
        json.dumps(
            {
                "sourceMediaSHA256": "source",
                "timeBase": "1/30",
                "sourcePTS": 0,
                "sourcePNG": pin(image),
                "points": points,
                "visibleArcReview": pin(review),
                "donors": [{"sourcePTS": 1, "sourcePNG": pin(image)}],
            }
        )
    )
    tracks = tmp_path / "tracks.json"
    tracks.write_text(
        json.dumps(
            {
                "configSHA256": file_digest(config),
                "timeBase": "1/30",
                "seedPTS": 0,
                "visibleArcReviewed": True,
                "headContourComplete": False,
                "placementApproved": fault == "approval",
                "observations": []
                if fault == "donors"
                else [
                    {
                        "sourcePTS": 2 if fault == "pts" else 1,
                        "points": points,
                        "supported": [True, True, fault != "partial"]
                        if fault != "flags"
                        else [1, True, True],
                    }
                ],
            }
        )
    )
    return {"config": pin(config), "tracks": pin(tracks)}


@pytest.mark.parametrize(
    "fault",
    ["none", "partial", "image", "review", "approval", "donors", "pts", "flags", "source", "clock"],
)
def test_bound_current_frames_do_not_extend_or_forge_detection(tmp_path: Path, fault: str) -> None:
    binding = fixture(tmp_path, fault)
    times: tuple[Fraction, ...] = (Fraction(0), Fraction(1, 30), Fraction(2, 30))
    if fault == "clock":
        times = (Fraction(0), Fraction(2, 30))
    if fault not in {"none", "partial"}:
        with pytest.raises((ReplayError, ValueError)):
            read_visible_arc(
                binding,
                source_hash="different" if fault == "source" else "source",
                times=times,
                width=100,
                height=200,
            )
        return
    arc = read_visible_arc(binding, source_hash="source", times=times, width=100, height=200)
    assert arc.at(Fraction(-1)) is None
    assert arc.at(Fraction(2, 30)) is None
    assert arc.at(Fraction(2, 30), ending=True) is not None
    first = arc.at(Fraction(0))
    second = arc.at(Fraction(1, 30))
    assert first is not None and first.visible_top == 50
    assert second is not None
    assert second.visible_top == (None if fault == "partial" else 50)
    assert second.unknown == (1 if fault == "partial" else 0)
    assert arc.covers(Fraction(0), Fraction(1, 30))
    assert not arc.covers(Fraction(-1), Fraction(1, 30))
    assert not arc.covers(Fraction(0), Fraction(3, 30))
    assert arc.covers(Fraction(0), Fraction(2, 30)) == (fault == "none")


@pytest.mark.parametrize(
    "value", [[], [[1, 2]], [[True, 2]] * 3, [[float("nan"), 2]] * 3, ["bad"] * 3]
)
def test_invalid_arc_points_rejected(value: object) -> None:
    with pytest.raises(ReplayError):
        _points(value, 100, 200)


def test_clipped_points_remain_unknown() -> None:
    assert _points([None, [-1, 5], [10, 20]], 100, 200) == (None, None, (10, 20))
