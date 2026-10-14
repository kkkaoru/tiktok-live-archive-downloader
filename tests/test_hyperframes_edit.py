"""Synthetic, offline checks for the private parallel editing project."""

import json
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.hyperframes_edit import Clip, _clips, _probe, _seconds, build, composition, parse_srt


def test_cut_and_caption_mapping_escapes_markup() -> None:
    cues = parse_srt(
        "1\n00:00:01,000 --> 00:00:03,000\n<one>\n\n2\n00:00:05,000 --> 00:00:06,000\n続き\n"
    )
    page = composition("assets/source.mp4", (Clip(2, 4), Clip(5, 7)), cues, width=1080, height=1920)
    assert 'data-duration="4.000"' in page
    assert 'id="shot-1"' in page and 'id="sound-1"' in page
    assert 'data-media-start="5.000"' in page
    assert 'data-start="0.000" data-duration="1.000" data-track-index="20">&lt;one&gt;' in page
    assert 'data-start="2.000" data-duration="1.000" data-track-index="20">続き' in page
    assert "<one>" not in page
    assert 'window.__timelines["main"]' in page


def test_srt_clock_validation() -> None:
    assert _seconds("02:03:04,005") == 7384.005
    with pytest.raises(ReplayError, match="timestamp"):
        _seconds("00:00:00.000")
    with pytest.raises(ReplayError, match="clock"):
        _seconds("00:60:00,000")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("oops", "Malformed SRT cue"),
        ("1\n0 --> 1\nx", "timestamp"),
        ("1\n00:00:01,000 - 00:00:02,000\nx", "interval"),
        ("1\n00:00:02,000 --> 00:00:01,000\nx", "content or interval"),
        ("1\n00:00:00,000 --> 00:00:01,000\n   ", "content or interval"),
        (
            "1\n00:00:03,000 --> 00:00:04,000\na\n\n2\n00:00:01,000 --> 00:00:02,000\nb",
            "ordered",
        ),
    ],
)
def test_reject_bad_srt(text: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        parse_srt(text)


def test_clip_validation() -> None:
    assert _clips([[0, 1], [2.5, 3]], 3) == (Clip(0, 1), Clip(2.5, 3))
    with pytest.raises(ReplayError, match="source clip"):
        Clip(2, 1)
    with pytest.raises(ReplayError, match="source clip"):
        Clip(float("nan"), 1)
    with pytest.raises(ReplayError, match="1-3000"):
        _clips([], 10)
    with pytest.raises(ReplayError, match="numeric"):
        _clips([[True, 1]], 10)
    with pytest.raises(ReplayError, match="beyond"):
        _clips([[0, 11]], 10)


def test_caption_budget() -> None:
    cue = parse_srt("1\n00:00:00,000 --> 00:00:01,000\na")
    with pytest.raises(ReplayError, match="Too many captions"):
        composition("assets/source.mp4", (Clip(0, 1),) * 2501, cue, width=640, height=360)


def test_probe_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        stdout = json.dumps(
            {
                "format": {"duration": "3"},
                "streams": [
                    {"codec_type": "audio"},
                    {"codec_type": "video", "width": 640, "height": 360},
                ],
            }
        )

    monkeypatch.setattr("replay.hyperframes_edit.subprocess.run", lambda *a, **k: Result())
    assert _probe(Path("example.mp4")) == (3, 640, 360)

    class BadResult:
        stdout = '{"format":{"duration":"3"},"streams":[]}'

    monkeypatch.setattr("replay.hyperframes_edit.subprocess.run", lambda *a, **k: BadResult())
    with pytest.raises(ReplayError, match="video stream"):
        _probe(Path("example.mp4"))


def test_build_scoped_private_nonoverwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("replay.hyperframes_edit._probe", lambda _: (10.0, 640, 360))
    edits = tmp_path / "edits"
    edits.mkdir()
    original = tmp_path / "original.mp4"
    original.write_bytes(b"original")
    srt = tmp_path / "captions.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nCaption", encoding="utf-8")
    project = edits / "attempt"
    gsap = tmp_path / "gsap.min.js"
    gsap.write_bytes(b"window.gsap={};")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(original),
                "project": str(project),
                "clips": [[0, 3]],
                "srt": str(srt),
                "gsap": str(gsap),
                "caption_offset": 1,
            }
        ),
        encoding="utf-8",
    )
    assert build(config) == project
    assert (project / "assets/source.mp4").resolve() == original
    assert (project / "assets/gsap.min.js").read_bytes() == b"window.gsap={};"
    assert original.read_bytes() == b"original"
    page = (project / "index.html").read_text(encoding="utf-8")
    assert page.count("Caption") == 1
    assert 'id="caption-0" class="clip caption" data-start="0.000"' in page
    with pytest.raises(ReplayError, match="overwrite"):
        build(config)


def test_reject_unscoped_and_missing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    edits = tmp_path / "edits"
    edits.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    config = tmp_path / "config.json"
    gsap = tmp_path / "gsap.min.js"
    gsap.write_bytes(b"window.gsap={};")
    base = {
        "version": 1,
        "source": str(source),
        "project": str(tmp_path / "outside"),
        "clips": [[0, 1]],
        "srt": None,
        "gsap": str(gsap),
    }
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ReplayError, match="edits"):
        build(config)
    base["project"] = str(edits / "new")
    base["srt"] = str(tmp_path / "missing.srt")
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ReplayError, match="SRT"):
        build(config)
    base["srt"] = None
    base["version"] = 2
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ReplayError, match="version"):
        build(config)
    base["version"] = 1
    base["caption_offset"] = -1
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ReplayError, match="caption offset"):
        build(config)
    base["caption_offset"] = 0
    base["source"] = str(tmp_path / "missing.mp4")
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ReplayError, match="local file"):
        build(config)
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="configuration"):
        build(config)
