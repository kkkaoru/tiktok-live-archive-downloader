"""Chat comment boxes are found on caption-free frames and filled; other pixels are kept."""

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from replay import gpu_lease
from replay import source_comment_clean as clean
from replay.core import ReplayError
from replay.source_lead_in import sha256, video_pts

NAVY = (60, 20, 15)
LAVENDER = (230, 170, 160)
STTN: dict[str, object] = {"directory": "/sttn", "checkpointSHA256": "c" * 64,
                           "architectureSHA256": "a" * 64}  # fmt: skip


class Flat:
    """A stand-in STTN tiler predicting one flat grey."""

    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        return np.full((len(frames), rows.stop - rows.start, frames.shape[2], 3), 40, np.float32)


def scene() -> NDArray[np.uint8]:
    """A 216x384 navy frame with a lavender figure (light, tinted) on the right."""
    frame = np.full((384, 216, 3), NAVY, np.uint8)
    cv2.circle(frame, (150, 150), 60, LAVENDER, -1)
    return frame


def bubble(frame: NDArray[np.uint8], x0: int, y0: int, x1: int, y1: int) -> NDArray[np.uint8]:
    """A light grey box with dark text strokes."""
    cv2.rectangle(frame, (x0, y0), (x1, y1), (215, 215, 215), -1)
    for x in range(x0 + 6, x1 - 8, 12):
        cv2.line(frame, (x, y0 + 6), (x + 6, y1 - 6), (40, 40, 40), 2)
    return frame


def test_the_intent_is_optional_and_strict(tmp_path: Path) -> None:
    assert clean.intent(tmp_path) is False
    (tmp_path / "cleanup.json").write_text(
        '{"version": 1, "remove": "comment-boxes"}', encoding="utf-8"
    )
    assert clean.intent(tmp_path) is True
    (tmp_path / "cleanup.json").write_text('{"version": 1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="cleanup.json must be"):
        clean.intent(tmp_path)


def test_the_copy_lives_beside_the_delivery_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(storage_root=tmp_path / "deliveries/youtube-x")
    monkeypatch.setattr(clean, "load_config", lambda _root: config)
    found = clean.location(tmp_path / "youtube-x", tmp_path / "v/a.mp4")
    assert found == tmp_path / "deliveries/source-comment-clean/youtube-x/a.mp4"
    config.storage_root = None
    with pytest.raises(ReplayError, match="no bound storage"):
        clean.location(tmp_path / "youtube-x", tmp_path / "v/a.mp4")


def test_a_grey_text_box_is_found_and_the_tinted_figure_is_not() -> None:
    frame = bubble(scene(), 0, 40, 120, 70)
    assert clean.comment_boxes(frame, []) == ([(0, 40, 121, 71)], [(0, 40, 121, 71)])
    assert clean.comment_boxes(scene(), []) == ([], [])


def test_a_grey_box_without_text_or_away_from_an_edge_is_not_a_comment() -> None:
    frame = scene()
    cv2.rectangle(frame, (30, 200), (130, 230), (215, 215, 215), -1)
    assert clean.comment_boxes(frame, []) == ([], [])
    assert clean.comment_boxes(frame, [(30, 200, 131, 231)]) == ([], [])


def test_a_sliver_cut_by_the_top_counts_only_after_a_full_box_near_it() -> None:
    frame = scene()
    cv2.rectangle(frame, (0, 0), (60, 5), (215, 215, 215), -1)
    assert clean.comment_boxes(frame, []) == ([], [])
    assert clean.comment_boxes(frame, [(0, 20, 90, 50)]) == ([(0, 0, 61, 6)], [])
    assert clean.comment_boxes(frame, [(0, 300, 90, 330)]) == ([], [])


def test_a_box_panned_out_at_the_left_edge_counts_when_a_line_high() -> None:
    frame = scene()
    cv2.rectangle(frame, (0, 100), (4, 130), (215, 215, 215), -1)
    assert clean.comment_boxes(frame, []) == ([(0, 100, 5, 131)], [])
    tall = scene()
    cv2.rectangle(tall, (0, 100), (4, 200), (215, 215, 215), -1)
    assert clean.comment_boxes(tall, []) == ([], [])


def test_an_edge_cut_grey_streak_in_a_tinted_scene_is_not_a_box() -> None:
    frame = scene()
    cv2.rectangle(frame, (0, 100), (4, 130), (215, 215, 215), -1)
    frame[100:131:4, 0:5] = (40, 80, 140)
    assert clean.comment_boxes(frame, []) == ([], [])
    assert clean.comment_boxes(frame, [(0, 100, 90, 130)]) == ([], [])


def test_a_fill_far_from_its_surroundings_is_refilled_spatially() -> None:
    frame = np.full((60, 60, 3), NAVY, np.uint8)
    frame[20:40, 20:40] = 230
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    refilled = clean.guarded(frame.copy(), mask)
    assert int(np.abs(refilled[25:35, 25:35].astype(int) - NAVY).max()) <= 3
    near = np.full((60, 60, 3), NAVY, np.uint8)
    near[20:40, 20:40] = (70, 25, 20)
    assert np.array_equal(clean.guarded(near.copy(), mask), near)


def test_scanning_keeps_recent_full_boxes_for_slivers() -> None:
    full = bubble(scene(), 0, 100, 90, 130)
    sliver = scene()
    cv2.rectangle(sliver, (0, 100), (4, 130), (215, 215, 215), -1)
    found = clean.scan(iter([full, sliver, scene()]))
    assert found == [[(0, 100, 91, 131)], [(0, 100, 5, 131)], []]


def test_masks_join_neighbouring_frames_and_grow_by_the_margin() -> None:
    boxes: list[list[clean.Box]] = [[], [], [], [(10, 10, 20, 20)], [], [], []]
    found = clean.masks(boxes, 0, 7, (40, 40))
    assert found.any(axis=(1, 2)).tolist() == [False, True, True, True, True, True, False]
    assert np.flatnonzero(found[1].any(axis=0)).tolist() == list(range(4, 26))


def test_bands_cover_masked_rows_at_most_one_tile_high() -> None:
    mask = np.zeros((2, 1080, 8), dtype=bool)
    mask[0, 100:150] = True
    mask[1, 400:900] = True
    assert clean.bands(mask) == [(76, 174), (376, 559), (559, 741), (741, 924)]


def test_fill_changes_only_masked_pixels() -> None:
    chunk = np.stack([bubble(scene(), 0, 40, 120, 70), scene()])
    mask = clean.masks([[(0, 40, 121, 71)], []], 0, 2, (384, 216))
    filled = clean.fill(chunk.copy(), mask, Flat())
    grown = cv2.dilate(mask[0].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    assert int(filled[0][grown].max()) < 100
    assert np.array_equal(filled[0][~grown], chunk[0][~grown])
    assert np.array_equal(filled[1][~grown], chunk[1][~grown])


def test_timing_gives_a_millisecond_clock_one_line_for_its_repeating_steps() -> None:
    found = clean.timing(
        [0, 528, 1056, 1600, 2128, 2656, 3200, 3728, 4256, 4800, 5328, 5856], Fraction(30)
    )
    assert found == (
        "0.000000 setpts expr 0+528*floor((N-0+2)/3)+528*floor((N-0+1)/3)+544*floor((N-0+0)/3);\n"
    )


def test_timing_starts_a_line_where_the_steps_break() -> None:
    found = clean.timing(
        [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 99000, 99512, 100024,
         100536, 101048, 101560, 102072, 102584, 103096, 103608, 104120, 104632],
        Fraction(30),
    )  # fmt: skip
    assert found == (
        "0.000000 setpts expr 0+512*floor((N-0+0)/1);\n"
        "0.383333 setpts expr 99000+512*floor((N-12+0)/1);\n"
    )


def test_timing_gives_a_single_frame_its_pts() -> None:
    assert clean.timing([7], Fraction(30)) == "0.000000 setpts expr 7;\n"


def test_a_quote_in_the_timing_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="quote"):
        clean.encoder(
            tmp_path / "s.mp4", tmp_path / "o.mp4", tmp_path / "it's.txt", {"time_base": "1/30"},
            (2, 2), rate=Fraction(30), ffmpeg="ffmpeg",
        )  # fmt: skip


def movie(tmp_path: Path) -> Path:
    """2 s of a 216x384 scene, a comment box during the second half, frame steps 528/544/16000."""
    directory = tmp_path / "vertical"
    directory.mkdir()
    picture = tmp_path / "scene.png"
    cv2.imwrite(str(picture), scene())
    output = directory / "2026-10-22_23-30-59_JST_youtube-x.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "30", "-t", "2",
         "-i", str(picture), "-f", "lavfi", "-i", "sine=duration=2:sample_rate=48000",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                "drawbox=x=0:y=40:w=121:h=31:color=0xd7d7d7:t=fill:enable='gte(t,1)',"
                "drawbox=x=8:y=48:w=100:h=3:color=0x282828:t=fill:enable='gte(t,1)',"
                "drawbox=x=8:y=60:w=100:h=3:color=0x282828:t=fill:enable='gte(t,1)',"
                "settb=1/16000,setpts='16*round(N*100/3)'",
         "-fps_mode", "passthrough", "-enc_time_base", "1/16000", "-video_track_timescale",
         "16000", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "aac", "-shortest",
         str(output)],
        check=True,
    )  # fmt: skip
    return output


def gray(path: Path) -> NDArray[np.uint8]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "extractplanes=y", "-f", "rawvideo",
         "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, 384, 216)


def test_boxes_are_filled_frames_keep_their_pts_and_the_copy_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = movie(tmp_path)
    loaded: list[str] = []

    def inpainter(directory: Path, *, checkpoint_sha: str, architecture_sha: str) -> Flat:
        loaded.append(f"{directory} {checkpoint_sha[:1]} {architecture_sha[:1]}")
        return Flat()

    monkeypatch.setattr("replay.source_comment_clean.subtitle_sttn.Inpainter", inpainter)
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "leases")
    # Two chunks with boxes: two leases, one model load.
    monkeypatch.setattr(clean, "CHUNK", 20)
    output = tmp_path / "clean/2026-10-22_23-30-59_JST_youtube-x.mp4"
    made = clean.prepared(
        source, output, sttn=STTN, ffmpeg="ffmpeg", ffprobe="ffprobe", lease="youtube-x"
    )
    assert made == output
    assert loaded == ["/sttn c a"]
    assert video_pts(output, "ffprobe") == video_pts(source, "ffprobe")
    before, after = gray(source), gray(output)
    assert int(np.abs(after[:28].astype(int) - before[:28]).max()) <= 3
    changed = np.abs(after[30:, 45:65, 40:80].astype(int) - before[30:, 45:65, 40:80])
    assert changed.mean(axis=(1, 2)).min() > 100
    receipt = json.loads((tmp_path / "clean/comment-clean.json").read_text(encoding="utf-8"))
    assert sorted(receipt) == [
        "codeVersion", "detector", "evidenceSHA256", "frames", "framesWithBoxes", "output",
        "outputSHA256", "source", "sourceSHA256", "sourceSize", "sttn",
    ]  # fmt: skip
    evidence = json.loads((tmp_path / "clean/comment-boxes.json").read_text(encoding="utf-8"))
    assert sum(1 for row in evidence["frames"] if row) == 29
    assert receipt["evidenceSHA256"] == sha256(tmp_path / "clean/comment-boxes.json")
    assert (receipt["frames"], receipt["framesWithBoxes"]) == (59, 29)
    assert receipt["outputSHA256"] == sha256(output)
    assert receipt["codeVersion"] == "comment-clean-v4"
    again = clean.prepared(
        source, output, sttn=STTN, ffmpeg="/missing", ffprobe="ffprobe", lease="youtube-x"
    )
    assert (again, loaded) == (output, ["/sttn c a"])
    boxes_path = tmp_path / "clean/comment-boxes.json"
    scanned = boxes_path.read_bytes()
    boxes_path.unlink()
    rescanned = clean.prepared(
        source, output, sttn=STTN, ffmpeg="ffmpeg", ffprobe="ffprobe", lease="youtube-x"
    )
    assert (rescanned, boxes_path.read_bytes(), loaded) == (output, scanned, ["/sttn c a"])
    stale = scanned.replace(b'"edgeMin": 3', b'"edgeMin": 4')
    boxes_path.write_bytes(stale)
    with pytest.raises(ReplayError, match="no longer matches"):
        clean.prepared(
            source, output, sttn=STTN, ffmpeg="/missing", ffprobe="ffprobe", lease="youtube-x"
        )
    assert boxes_path.read_bytes() == stale
    boxes_path.write_bytes(scanned)
    output.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="no longer matches"):
        clean.prepared(
            source, output, sttn=STTN, ffmpeg="ffmpeg", ffprobe="ffprobe", lease="youtube-x"
        )


def rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boxes: list[list[clean.Box]] | None
) -> tuple[Path, Path, list[list[clean.Box]], list[str]]:
    """A 20-frame-chunk render of the test movie; (source, output, boxes, GPU events)."""
    source = movie(tmp_path)
    events: list[str] = []

    @contextmanager
    def lease(_root: Path, replay_id: str) -> Iterator[None]:
        events.append(f"lease {replay_id}")
        yield

    def inpainter() -> Flat:
        events.append("model")
        return Flat()

    monkeypatch.setattr(clean, "CHUNK", 20)
    monkeypatch.setattr(gpu_lease, "lease", lease)
    script, output = tmp_path / "timing.txt", tmp_path / "out.mp4"
    script.write_text(clean.timing(video_pts(source, "ffprobe"), Fraction(30)), encoding="utf-8")
    stream: dict[str, object] = {"time_base": "1/16000", "color_space": "bt709"}
    encode = clean.encoder(
        source, output, script, stream, (384, 216), rate=Fraction(30), ffmpeg="ffmpeg"
    )
    found = clean.render(
        source, boxes, inpainter, shape=(384, 216), encode=encode, ffmpeg="ffmpeg", lease="r"
    )
    return source, output, found, events


def test_the_reused_chunk_buffer_fills_only_chunks_with_boxes_under_a_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boxes: list[list[clean.Box]] = [[] for _ in range(59)]
    boxes[30:35] = [[(0, 40, 121, 71)]] * 5
    source, output, found, events = rendered(tmp_path, monkeypatch, boxes)
    assert found is boxes
    assert events == ["lease r", "model"]
    assert video_pts(output, "ffprobe") == video_pts(source, "ffprobe")
    before, after = gray(source).astype(int), gray(output).astype(int)
    assert int(np.abs(after[:28] - before[:28]).max()) <= 3
    assert np.abs(after[31:35, 45:65, 40:80] - before[31:35, 45:65, 40:80]).mean() > 100
    assert int(np.abs(after[40:] - before[40:]).max()) <= 3


def test_a_render_without_boxes_scans_while_it_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, found, events = rendered(tmp_path, monkeypatch, None)
    assert found == clean.scan(clean.frames_of(clean.decoder(source, ffmpeg="ffmpeg"), (384, 216)))
    assert (len(found), sum(1 for row in found if row)) == (59, 29)
    assert events == ["lease r", "model", "lease r", "model"]
    assert video_pts(output, "ffprobe") == video_pts(source, "ffprobe")
    before, after = gray(source).astype(int), gray(output).astype(int)
    assert int(np.abs(after[:28] - before[:28]).max()) <= 3
    assert np.abs(after[30:, 45:65, 40:80] - before[30:, 45:65, 40:80]).mean() > 100


def test_box_evidence_is_reused_redone_when_stale_and_refused_when_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scans: list[str] = []
    monkeypatch.setattr(clean, "frames_of", lambda _c, _s: iter([scene()]))

    def scan(_frames: object) -> list[list[clean.Box]]:
        scans.append("scan")
        return [[(1, 2, 3, 4)]]

    monkeypatch.setattr(clean, "scan", scan)
    path = tmp_path / "comment-boxes.json"
    first = clean.evidence_for(tmp_path, path, "a", (4, 4), ffmpeg="ffmpeg")
    again = clean.evidence_for(tmp_path, path, "a", (4, 4), ffmpeg="/missing")
    assert (first, again, scans) == ([[(1, 2, 3, 4)]], [[(1, 2, 3, 4)]], ["scan"])
    clean.evidence_for(tmp_path, path, "b", (4, 4), ffmpeg="ffmpeg")
    assert scans == ["scan", "scan"]
    path.write_text(
        json.dumps({"sourceSHA256": "b", "detector": clean.settings(), "frames": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="Malformed"):
        clean.evidence_for(tmp_path, path, "b", (4, 4), ffmpeg="ffmpeg")


def test_a_failed_decode_is_an_error(tmp_path: Path) -> None:
    command = ["ffmpeg", "-v", "error", "-i", str(tmp_path / "missing.mp4"), "-f", "rawvideo", "-"]
    with pytest.raises(ReplayError, match="failed while decoding"):
        list(clean.frames_of(command, (2, 2)))


def test_the_copy_needs_its_own_directory_and_every_frame_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = movie(tmp_path)
    with pytest.raises(ReplayError, match="its own directory"):
        clean.prepared(
            source, source, sttn=STTN, ffmpeg="ffmpeg", ffprobe="ffprobe", lease="youtube-x"
        )

    monkeypatch.setattr(clean, "cached_boxes", lambda _path, _digest: [[]])
    output = tmp_path / "clean" / source.name
    with pytest.raises(ReplayError, match="does not cover every frame"):
        clean.prepared(
            source, output, sttn=STTN, ffmpeg="ffmpeg", ffprobe="ffprobe", lease="youtube-x"
        )
    monkeypatch.setattr(clean, "cached_boxes", lambda _path, _digest: [[] for _ in range(59)])
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "leases")
    monkeypatch.setattr(
        "replay.source_comment_clean.video_pts",
        lambda movie, probe: video_pts(movie, probe)[: 58 if "partial" in movie.name else 59],
    )
    with pytest.raises(ReplayError, match="moved a source frame"):
        clean.prepared(
            source, output, sttn=STTN, ffmpeg="ffmpeg", ffprobe="ffprobe", lease="youtube-x"
        )


def test_the_command_cleans_a_recording_with_the_intent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    defaults = tmp_path / "defaults.json"
    defaults.write_text(json.dumps({"sttn": STTN}), encoding="utf-8")
    root = tmp_path / "youtube-x"
    root.mkdir()
    arguments = [str(root), str(tmp_path / "s.mp4"), "--defaults", str(defaults)]
    with pytest.raises(ReplayError, match="has no cleanup.json"):
        clean.main(arguments)
    (root / "cleanup.json").write_text(
        '{"version": 1, "remove": "comment-boxes"}', encoding="utf-8"
    )
    calls: list[tuple[Path, Path, str]] = []
    monkeypatch.setattr(clean, "location", lambda _r, source: tmp_path / "out" / source.name)

    def prepared(source: Path, output: Path, **options: object) -> Path:
        calls.append((source, output, str(options["lease"])))
        return output

    monkeypatch.setattr(clean, "prepared", prepared)
    assert clean.main(arguments) == 0
    assert calls == [(tmp_path / "s.mp4", tmp_path / "out/s.mp4", "youtube-x")]
    assert capsys.readouterr().out == f"{tmp_path / 'out/s.mp4'}\n"


def test_a_partial_light_smear_is_refilled_spatially() -> None:
    frame = np.full((60, 60, 3), NAVY, np.uint8)
    frame[:, 45:] = LAVENDER
    frame[20:40, 20:40] = NAVY
    frame[20:25, 20:40] = 235
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    refilled = clean.guarded(frame.copy(), mask)
    assert int(refilled[20:25, 25:35].max()) < 120
