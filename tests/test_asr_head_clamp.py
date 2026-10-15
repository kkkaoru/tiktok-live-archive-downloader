"""A word that begins in the partial first frame keeps its delivered part; nothing is dropped."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import asr_head_clamp as clamp
from replay.core import ReplayError

START = Fraction("0.901")


def label(text: str, start: float, end: float) -> dict[str, object]:
    return {"text": text, "start": start, "end": end, "provenance": "w:0"}


def test_the_first_boundary_is_the_first_whole_frame_after_the_audio_start() -> None:
    assert clamp.first_boundary(Fraction("0.901"), fps=30) == Fraction(28, 30)
    assert clamp.first_boundary(Fraction("0.9"), fps=30) == Fraction(27, 30)
    assert clamp.first_boundary(Fraction(0), fps=30) == Fraction(0)


def test_invalid_frame_rates_and_audio_starts_are_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid frame rate"):
        clamp.first_boundary(Fraction(1), fps=0)
    with pytest.raises(ReplayError, match="Invalid frame rate"):
        clamp.first_boundary(Fraction(-1), fps=30)


def test_a_head_label_is_clamped_to_the_first_delivered_boundary_with_provenance() -> None:
    labels = [label("開始", 0.0, 0.42), label("でき", 0.42, 1.1)]
    result, clamps = clamp.clamp_head(labels, audio_start=START, offset=START, fps=30)
    assert result[0]["start"] == 0.032333334
    assert result[0]["end"] == 0.42
    assert result[0]["headClamp"] == {
        "originalStart": 0.0,
        "firstDeliveredBoundary": pytest.approx(28 / 30),
        "droppedSeconds": pytest.approx(0.032333333),
        "rule": "sub-frame head onset not delivered; label kept from the first whole frame",
    }
    assert result[1] == labels[1]
    assert clamps == [{"index": 0, "text": "開始", "droppedSeconds": pytest.approx(0.032333333)}]


def test_the_input_labels_are_not_modified() -> None:
    labels = [label("開始", 0.0, 0.42)]
    clamp.clamp_head(labels, audio_start=START, offset=START, fps=30)
    assert labels == [label("開始", 0.0, 0.42)]


def test_a_label_that_starts_exactly_on_the_boundary_is_untouched() -> None:
    on_boundary = {"text": "あ", "start": float(Fraction(28, 30) - START), "end": 0.5}
    later = label("い", 0.6, 0.9)
    result, clamps = clamp.clamp_head(
        [on_boundary, later], audio_start=Fraction(28, 30), offset=Fraction(28, 30), fps=30
    )
    assert result == [on_boundary, later]
    assert clamps == []


def test_a_label_wholly_inside_the_removed_head_is_refused() -> None:
    with pytest.raises(ReplayError, match="removed entirely"):
        clamp.clamp_head([label("あ", 0.0, 0.03)], audio_start=START, offset=START, fps=30)


def test_a_label_that_ends_exactly_on_the_boundary_is_refused() -> None:
    labels = [label("あ", 0.0, float(Fraction(28, 30) - START))]
    with pytest.raises(ReplayError, match="removed entirely"):
        clamp.clamp_head(labels, audio_start=START, offset=START, fps=30)


def test_a_point_label_in_the_head_is_refused() -> None:
    with pytest.raises(ReplayError, match="removed entirely"):
        clamp.clamp_head([label("あ", 0.01, 0.01)], audio_start=START, offset=START, fps=30)


def test_a_label_that_starts_before_the_decoded_audio_is_refused() -> None:
    with pytest.raises(ReplayError, match="starts before the decoded audio"):
        clamp.clamp_head([label("あ", -0.1, 0.5)], audio_start=START, offset=START, fps=30)


def test_an_empty_label_list_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid label count"):
        clamp.clamp_head([], audio_start=START, offset=START, fps=30)


def test_the_command_line_writes_a_new_file_with_clamps_and_parent(tmp_path: Path) -> None:
    source = tmp_path / "labels.json"
    source.write_text(
        json.dumps({"version": 1, "labels": [label("開始", 0.0, 0.42), label("でき", 0.42, 1.1)]}),
        encoding="utf-8",
    )
    output = tmp_path / "clamped.json"
    assert clamp.main([str(source), str(output), "--audio-start", "0.901", "--fps", "30"]) == 0
    document = json.loads(output.read_text())
    assert document["parentLabelsFile"] == str(source)
    assert [c["text"] for c in document["headClamps"]] == ["開始"]
    assert document["labels"][0]["headClamp"]["originalStart"] == 0.0
    assert json.loads(source.read_text())["labels"][0]["start"] == 0.0


def test_the_command_line_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "labels.json"
    source.write_text(json.dumps({"labels": [label("あ", 0.5, 1.0)]}), encoding="utf-8")
    output = tmp_path / "out.json"
    output.write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        clamp.main([str(source), str(output), "--audio-start", "0.901", "--fps", "30"])


def test_the_rounded_start_never_falls_below_the_boundary() -> None:
    exact = Fraction(28, 30) - START
    assert Fraction(str(clamp.not_below(exact))) >= exact
    assert Fraction(str(clamp.not_below(Fraction(1, 3)))) >= Fraction(1, 3)
    assert clamp.not_below(Fraction(1, 4)) == 0.25
