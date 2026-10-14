import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay.fcp_measured import read_measured_captions
from replay.recordings import record


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "canvas": [1080, 1920],
                "coordinateSystem": "top-left canvas; calibrated text origin",
                "policy": {"frame_rate": 30},
                "candidates": [
                    {"cue": {"text": "あ\nい", "start": 0.1, "end": 0.9}, "x": 540, "y": 500}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_read_measured_preserves_text_clocks_and_origin(candidate: Path) -> None:
    result = read_measured_captions(
        candidate,
        sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        width=1080,
        height=1920,
        rate=30,
        duration=Fraction(1),
    )
    assert len(result) == 1
    assert result[0].cue.text == "あ\nい"
    assert (result[0].cue.start, result[0].cue.end, result[0].x, result[0].y) == (
        0.1,
        0.9,
        540,
        500,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("version", 2),
        ("candidateOnly", False),
        ("canvas", [1920, 1080]),
        ("coordinateSystem", "ASS bottom"),
        ("policy", {"frame_rate": 24}),
        ("candidates", []),
        ("candidates", None),
    ],
)
def test_read_measured_refuses_wrong_contract(candidate: Path, field: str, value: object) -> None:
    raw = record(json.loads(candidate.read_text(encoding="utf-8")))
    raw[field] = value
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        read_measured_captions(
            candidate,
            sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(1),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", ""),
        ("text", "a\0b"),
        ("text", "a\nb\nc"),
        ("text", "a\n"),
        ("text", "a" * 32769),
        ("start", -1),
        ("start", 0.9),
        ("end", 2),
        ("end", float("nan")),
        ("end", True),
    ],
)
def test_read_measured_refuses_invalid_cue(candidate: Path, field: str, value: object) -> None:
    raw = record(json.loads(candidate.read_text(encoding="utf-8")))
    cue: dict[str, object] = {"text": "a", "start": 0.1, "end": 0.9}
    cue[field] = value
    raw["candidates"] = [{"cue": cue, "x": 540, "y": 500}]
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        read_measured_captions(
            candidate,
            sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(1),
        )


@pytest.mark.parametrize(("x", "y"), [(-1, 500), (1081, 500), (540, -1), (540, 1921)])
def test_read_measured_refuses_outside_origin(candidate: Path, x: float, y: float) -> None:
    raw = record(json.loads(candidate.read_text(encoding="utf-8")))
    raw["candidates"] = [{"cue": {"text": "a", "start": 0, "end": 1}, "x": x, "y": y}]
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="origin"):
        read_measured_captions(
            candidate,
            sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(1),
        )


def test_read_measured_refuses_changed_bytes(candidate: Path) -> None:
    with pytest.raises(ValueError):
        read_measured_captions(
            candidate, sha256="0" * 64, width=1080, height=1920, rate=30, duration=Fraction(1)
        )


def test_read_measured_refuses_reversed_order(candidate: Path) -> None:
    raw = record(json.loads(candidate.read_text(encoding="utf-8")))
    raw["candidates"] = [
        {"cue": {"text": "a", "start": 0.5, "end": 0.9}, "x": 540, "y": 500},
        {"cue": {"text": "b", "start": 0.1, "end": 0.5}, "x": 540, "y": 500},
    ]
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="clock"):
        read_measured_captions(
            candidate,
            sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(1),
        )
