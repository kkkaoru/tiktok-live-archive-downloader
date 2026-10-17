"""Sampled-chunk collaborators run real ffmpeg and a fake native OCR tool on synthetic media."""

import json
import subprocess
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import subtitle_sampled_chunk as chunk
from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle
from replay.subtitle_cleanup_plan import CaptionSearchProfile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_profile import SubtitleBands

WIDTH = 320
HEIGHT = 240
CLOCK = VideoFrameClock(tuple(range(0, 40 * 3000, 3000)), Fraction(1, 90000), 3000)


def _video(path: Path, *, change: bool) -> Path:
    second = "white" if change else "black"
    graph = (
        f"color=c=black:s={WIDTH}x{HEIGHT}:r=30:d=1[a];"
        f"color=c={second}:s={WIDTH}x{HEIGHT}:r=30:d=1[b];[a][b]concat=n=2:v=1:a=0"
    )
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", graph, "-pix_fmt", "yuv420p",
         "-y", str(path)],
        check=True,
        timeout=60,
    )  # fmt: skip
    return path


def _bar(batch: NDArray[np.float32]) -> NDArray[np.float32]:
    count, _, height, width = batch.shape
    maps = np.zeros((count, 1, height, width), dtype=np.float32)
    maps[:, :, int(height * 0.40) : int(height * 0.50), width // 8 : width // 2] = 0.95
    return maps


def test_band_triggers_bracket_the_visible_change(tmp_path: Path) -> None:
    video = _video(tmp_path / "change.mp4", change=True)
    found = chunk.band_triggers(video, CLOCK, canvas_width=WIDTH, envelope=(0, 80, WIDTH, 40))
    assert found == (29, 30)


def test_band_triggers_are_empty_for_a_static_band(tmp_path: Path) -> None:
    video = _video(tmp_path / "static.mp4", change=False)
    found = chunk.band_triggers(video, CLOCK, canvas_width=WIDTH, envelope=(0, 80, WIDTH, 40))
    assert found == ()


def test_band_triggers_reject_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decode"):
        chunk.band_triggers(
            tmp_path / "absent.mp4", CLOCK, canvas_width=WIDTH, envelope=(0, 80, WIDTH, 40)
        )


FAKE_NATIVE = """#!{python}
import json, sys
from pathlib import Path

request = json.loads(Path(sys.argv[3]).read_text())
samples = request["samples"]
frames = [
    {{
        "requestedTimeSeconds": s["value"] / s["timescale"],
        "actualTimeSeconds": s["value"] / s["timescale"],
        "width": request["width"],
        "height": request["height"],
        "lines": [{{"text": "こんにちは世界です", "confidence": 0.9,
                    "region": {{"x": 20, "y": 90, "width": 200, "height": 30}}}}],
    }}
    for s in samples
]
directory = Path(request["checkpointDirectory"])
for number in range(0, len(frames), 8):
    (directory / f"block-{{number // 8}}.json").write_text(json.dumps(frames[number:number + 8]))
    (directory / f"block-{{number // 8}}-clocks.json").write_text("{{}}")
sys.exit(int(request.get("fail", 0)))
"""


def _native(path: Path) -> Path:
    path.write_text(FAKE_NATIVE.format(python=sys.executable), encoding="utf-8")
    path.chmod(0o700)
    return path


def test_session_ocr_reads_every_sampled_frame_across_children(tmp_path: Path) -> None:
    native = _native(tmp_path / "native")
    indices = list(range(0, 40))
    long_clock = VideoFrameClock(tuple(range(0, 70 * 3000, 3000)), Fraction(1, 90000), 3000)
    found = chunk.session_ocr(
        native,
        tmp_path / "source.mp4",
        long_clock,
        indices + list(range(40, 70)),
        canvas=(WIDTH, HEIGHT),
        work=tmp_path / "work",
        root=tmp_path,
    )
    assert sorted(found) == list(range(70))
    assert found[69].detections[0].text == "こんにちは世界です"
    assert (tmp_path / "work" / "child-001" / "request.json").is_file()


LIMITED_NATIVE = FAKE_NATIVE.replace(
    'samples = request["samples"]\n',
    'samples = request["samples"]\n'
    "if len(samples) > 2:\n"
    '    sys.stderr.write("Native output exceeded its bounded capture limit.")\n'
    "    sys.exit(1)\n",
)


def test_session_ocr_halves_a_child_that_exceeds_the_capture_limit(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.write_text(LIMITED_NATIVE.format(python=sys.executable), encoding="utf-8")
    native.chmod(0o700)
    found = chunk.session_ocr(
        native,
        tmp_path / "source.mp4",
        CLOCK,
        [0, 1, 2, 3, 4],
        canvas=(WIDTH, HEIGHT),
        work=tmp_path / "work",
        root=tmp_path,
    )
    assert sorted(found) == [0, 1, 2, 3, 4]
    assert found[4].detections[0].text == "こんにちは世界です"
    assert (
        (tmp_path / "work" / "child-000" / "stderr.log")
        .read_text(encoding="utf-8")
        .startswith("Native output exceeded")
    )


def test_session_ocr_reports_a_single_frame_over_the_limit(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.write_text(
        "#!/bin/sh\necho 'Native output exceeded its bounded capture limit.' >&2\nexit 1\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    with pytest.raises(ValueError, match="session failed"):
        chunk.session_ocr(
            native,
            tmp_path / "source.mp4",
            CLOCK,
            [0, 1],
            canvas=(WIDTH, HEIGHT),
            work=tmp_path / "work",
            root=tmp_path,
        )


def test_session_ocr_reports_a_failed_native_tool(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    native.chmod(0o700)
    with pytest.raises(ValueError, match="session failed"):
        chunk.session_ocr(
            native,
            tmp_path / "source.mp4",
            CLOCK,
            [0, 1],
            canvas=(WIDTH, HEIGHT),
            work=tmp_path / "work",
            root=tmp_path,
        )


def test_session_ocr_rejects_a_short_native_answer(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    native.chmod(0o700)
    with pytest.raises(ValueError, match="cover every"):
        chunk.session_ocr(
            native,
            tmp_path / "source.mp4",
            CLOCK,
            [0, 1],
            canvas=(WIDTH, HEIGHT),
            work=tmp_path / "work",
            root=tmp_path,
        )


def test_detector_regions_use_polygons_touching_the_envelope(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", change=False)
    short = VideoFrameClock(CLOCK.pts[:10], CLOCK.time_base, CLOCK.last_duration)
    regions = chunk.detector_regions(
        video, short, canvas=(WIDTH, HEIGHT), envelope=(0, 90, WIDTH, 30), network=_bar
    )
    assert len(regions) == 10
    region = regions[0]
    assert region is not None
    assert region[1] < 120
    assert region[3] > 90


def test_detector_regions_are_empty_when_the_envelope_is_elsewhere(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", change=False)
    short = VideoFrameClock(CLOCK.pts[:4], CLOCK.time_base, CLOCK.last_duration)
    regions = chunk.detector_regions(
        video, short, canvas=(WIDTH, HEIGHT), envelope=(0, 200, WIDTH, 20), network=_bar
    )
    assert regions == [None, None, None, None]


def test_detector_regions_reject_a_short_decode(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", change=False)
    longer = VideoFrameClock(tuple(range(0, 100 * 3000, 3000)), Fraction(1, 90000), 3000)
    with pytest.raises(ValueError, match="ended early"):
        chunk.detector_regions(
            video, longer, canvas=(WIDTH, HEIGHT), envelope=(0, 90, WIDTH, 30), network=_bar
        )


def test_plan_chunk_joins_the_collaborators_and_times_each_stage() -> None:
    profile = CaptionSearchProfile(
        SubtitleBands(WIDTH, HEIGHT, primary=(3,), secondary=()), (0, 90, WIDTH, 40)
    )
    seconds = [float(pts * CLOCK.time_base) for pts in CLOCK.pts]
    line = TextDetection("こんにちは世界です", 0.9, TextRectangle(20, 90, 200, 30))

    def real_ocr(indices: Sequence[int]) -> dict[int, FrameObservation]:
        return {
            index: FrameObservation(seconds[index], seconds[index], WIDTH, HEIGHT, (line,))
            for index in indices
        }

    result = chunk.plan_chunk(
        clock=CLOCK,
        profile=profile,
        stride=13,
        radius=1,
        triggers=lambda: (5,),
        ocr=real_ocr,
        regions=lambda: [None] * 40,
    )
    sampling = result["sampling"]
    assert isinstance(sampling, dict)
    assert sampling["sampledFrames"] == [0, 5, 13, 26, 39]
    assert sampling["triggerFrames"] == [5]
    stages = result["stageSeconds"]
    assert isinstance(stages, dict)
    assert set(stages) == {"triggers", "ocr", "detectorRegions"}
    assert json.dumps(result)


BAND_NATIVE = FAKE_NATIVE.replace(
    'samples = request["samples"]\n',
    'samples = request["samples"]\n'
    'if request["height"] > 100:\n'
    '    sys.stderr.write("Native output exceeded its bounded capture limit.")\n'
    "    sys.exit(1)\n",
)


def test_session_ocr_reads_a_dense_single_frame_through_its_caption_band(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", change=False)
    native = tmp_path / "native"
    native.write_text(BAND_NATIVE.format(python=sys.executable), encoding="utf-8")
    native.chmod(0o700)
    found = chunk.session_ocr(
        native,
        video,
        CLOCK,
        [3],
        canvas=(WIDTH, HEIGHT),
        work=tmp_path / "work",
        root=tmp_path,
        band=(100, 80),
    )
    line = found[3].detections[0]
    assert (found[3].width, found[3].height, line.rectangle.y) == (WIDTH, HEIGHT, 190.0)
    assert found[3].requested_time == pytest.approx(0.1)
    assert json.loads((tmp_path / "work" / "band-000" / "band.json").read_text()) == {
        "index": 3,
        "sourcePTS": 9000,
        "top": 100,
        "height": 80,
    }


def test_session_ocr_refuses_a_band_that_still_exceeds_the_limit(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", change=False)
    native = tmp_path / "native"
    native.write_text(
        "#!/bin/sh\necho 'Native output exceeded its bounded capture limit.' >&2\nexit 1\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    with pytest.raises(ValueError, match="Caption band exceeded"):
        chunk.session_ocr(
            native,
            video,
            CLOCK,
            [3],
            canvas=(WIDTH, HEIGHT),
            work=tmp_path / "work",
            root=tmp_path,
            band=(100, 80),
        )


def test_session_ocr_reports_a_band_that_cannot_be_cut(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.write_text(
        "#!/bin/sh\necho 'Native output exceeded its bounded capture limit.' >&2\nexit 1\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    with pytest.raises(ValueError, match="could not be cut"):
        chunk.session_ocr(
            native,
            tmp_path / "absent.mp4",
            CLOCK,
            [3],
            canvas=(WIDTH, HEIGHT),
            work=tmp_path / "work",
            root=tmp_path,
            band=(100, 80),
        )


def test_ocr_band_widens_the_envelope_inside_the_canvas() -> None:
    assert chunk.ocr_band((0, 1299, 1077, 142), (1080, 1920)) == (1219, 302)
    assert chunk.ocr_band((0, 20, 100, 40), (320, 240)) == (0, 140)
