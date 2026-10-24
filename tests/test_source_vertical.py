"""A wide source is cropped around its tracked character by a smooth, still-when-still camera."""

import json
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replay import source_vertical as vertical
from replay.core import ReplayError
from replay.source_lead_in import sha256, video_pts

YOLO: dict[str, object] = {
    "weights": "/w.pt",
    "sha256": "w" * 64,
    "configDirectory": "/c",
    "step": 2,
}


def gradient(tmp_path: Path) -> Path:
    """A 2 s 240x180 picture whose luma is its column, with a tone."""
    output = tmp_path / "2026-10-22_23-30-59_JST_youtube-x.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "nullsrc=size=240x180:rate=30:d=2",
         "-f", "lavfi", "-i", "sine=duration=2:sample_rate=48000",
         "-vf", "geq=lum='X':cb=128:cr=128", "-c:v", "libx264", "-qp", "0",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(output)],
        check=True,
    )  # fmt: skip
    return output


def evidence(source: Path, digest: str, faces: Sequence[object]) -> dict[str, object]:
    return {
        "source": {"path": str(source), "sha256": digest},
        "weights": {"path": "/w.pt", "sha256": "w" * 64},
        "step": 2,
        "frames": [{"frame": 2 * index, "face": face} for index, face in enumerate(faces)],
    }


def left_columns(movie: Path) -> list[int]:
    """Each frame's crop offset, read back from its mean luma (its column, 102 wide)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(movie), "-vf", "extractplanes=y", "-f", "rawvideo",
         "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 180, 102).astype(np.float64)
    return [int(value) for value in np.rint(frames.mean(axis=(1, 2)) - 50.5)]


def test_the_intent_is_optional_and_strict(tmp_path: Path) -> None:
    assert vertical.intent(tmp_path) is None
    (tmp_path / "reframe.json").write_text(
        '{"version": 1, "aspect": [9, 16], "follow": "character"}', encoding="utf-8"
    )
    assert vertical.intent(tmp_path) == (9, 16)
    (tmp_path / "reframe.json").write_text(
        '{"version": 1, "aspect": [3, 4], "follow": "character"}', encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="reframe.json must be"):
        vertical.intent(tmp_path)
    (tmp_path / "reframe.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError, match="reframe.json must be"):
        vertical.intent(tmp_path)


def test_the_crop_keeps_the_height_at_an_even_width() -> None:
    assert vertical.crop_width(1920, 1080, (9, 16)) == 608
    assert vertical.crop_width(320, 180, (9, 16)) == 102
    with pytest.raises(ReplayError, match="not wider"):
        vertical.crop_width(1080, 1920, (9, 16))


def test_face_centres_come_from_the_evidence_boxes() -> None:
    found = vertical.face_centres(
        {"frames": [{"frame": 0, "face": [10, 0, 30, 5, 0.9]}, {"frame": 2, "face": None}]}
    )
    assert found == [(0, 20.0, 0.9), (2, None, 0.0)]
    with pytest.raises(ReplayError, match="no frames"):
        vertical.face_centres({"frames": []})
    with pytest.raises(ReplayError, match="Malformed face"):
        vertical.face_centres({"frames": [{"frame": 0, "face": [1, 2]}]})
    with pytest.raises(ReplayError, match="Malformed frame"):
        vertical.face_centres({"frames": [{"frame": "0", "face": None}]})


def test_missing_faces_hold_the_last_position_and_the_first_fills_the_lead() -> None:
    centres: list[tuple[int, float | None]] = [(0, None), (2, 10.0), (4, None), (6, 30.0)]
    found = vertical.targets(centres, 8, width=100, window=1)
    assert found == [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 30.0, 30.0]


def test_a_single_stray_detection_is_dropped_by_the_median() -> None:
    centres: list[tuple[int, float | None]] = [
        (0, 10.0),
        (1, 10.0),
        (2, 90.0),
        (3, 10.0),
        (4, 10.0),
    ]
    found = vertical.targets(centres, 5, width=100, window=3)
    assert found == [10.0, 10.0, 10.0, 10.0, 10.0]


def test_a_video_without_faces_stays_centred() -> None:
    assert vertical.targets([(0, None)], 2, width=100, window=1) == [50.0, 50.0]


def test_a_still_or_slightly_moving_character_gives_a_still_frame() -> None:
    target = [200.0, 205.0, 195.0, 210.0, 190.0, 200.0]
    assert vertical.camera(target, width=400, crop=100, rate=10) == [150, 150, 150, 150, 150, 150]


def test_the_crop_never_leaves_the_frame() -> None:
    assert vertical.camera([0.0, 0.0], width=400, crop=100, rate=10) == [0, 0]
    assert vertical.camera([400.0, 400.0], width=400, crop=100, rate=10) == [300, 300]


def test_a_move_eases_in_caps_its_speed_and_stops_without_overshoot() -> None:
    offsets = vertical.camera([100.0] + [300.0] * 59, width=400, crop=100, rate=10)
    steps = np.diff(offsets)
    assert offsets[:8] == [50, 52, 56, 62, 70, 80, 90, 100]
    assert offsets[19:27] == [220, 230, 238, 244, 248, 250, 250, 250]
    assert offsets[-1] == 250
    assert max(offsets) == 250
    assert int(steps.min()) == 0
    assert int(steps.max()) == 10


def test_commands_move_the_crop_half_a_frame_before_each_changed_frame() -> None:
    found = vertical.commands([4, 4, 6, 8], [0, 1000, 2000, 3000], Fraction(1, 30000))
    assert found == "0.000000 crop x 4;\n0.050000 crop x 6;\n0.083333 crop x 8;\n"


def test_a_quote_in_the_command_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="quote"):
        vertical.render_command(
            tmp_path / "s.mp4", tmp_path / "o.mp4", tmp_path / "it's.txt", {"time_base": "1/30"},
            {}, crop=(2, 2, 0), ffmpeg="ffmpeg",
        )  # fmt: skip


def test_the_tracker_runs_the_pinned_detector_into_a_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> None:
        commands.append(command)
        Path(command[5]).write_text("{}", encoding="utf-8")

    monkeypatch.setattr("replay.source_vertical.subprocess.run", run)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "track.json"
    tools = {"ffmpeg": "/ffmpeg", "ffprobe": "/ffprobe"}
    vertical.track(
        tmp_path / "s.mp4",
        output,
        frames=60,
        python="/py",
        yolo=YOLO,
        tools=tools,
    )
    assert output.read_text(encoding="utf-8") == "{}"
    assert commands == [
        ["/py", "-B", "-m", "replay.yolo_track", str(tmp_path / "s.mp4"),
         str(tmp_path / "partial-track.json"), "--weights", "/w.pt", "--weights-sha256", "w" * 64,
         "--config-directory", "/c", "--work", str(tmp_path.resolve() / "edits"), "--end", "60",
         "--step", "2", "--ffmpeg", "/ffmpeg", "--ffprobe", "/ffprobe"]
    ]  # fmt: skip


def test_the_crop_follows_the_character_exactly_keeping_pts_and_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = gradient(tmp_path)
    digest = sha256(source)
    tracked: list[int] = []

    def track(_source: Path, output: Path, *, frames: int, **_k: object) -> None:
        tracked.append(frames)
        faces = [[20, 0, 40, 10, 0.9]] * 8 + [None] * 2 + [[190, 0, 210, 10, 0.9]] * 20
        output.write_text(json.dumps(evidence(source, digest, faces)), encoding="utf-8")

    monkeypatch.setattr(vertical, "track", track)
    monkeypatch.setattr(
        vertical,
        "detect_bodies",
        lambda _s, frames, _t, **_k: [{"frame": frame, "bodies": []} for frame in frames],
    )
    store = tmp_path / "store"
    output = vertical.prepared(
        source, store, ffmpeg="ffmpeg", ffprobe="ffprobe", python="/py", yolo=YOLO, native="/native"
    )
    assert output == store / source.name
    assert tracked == [60]
    assert video_pts(output, "ffprobe") == video_pts(source, "ffprobe")
    commands = (store / "crop-commands.txt").read_text(encoding="utf-8").splitlines()
    measured = left_columns(output)
    faces: list[tuple[int, float | None, float]] = [(2 * index, 30.0, 0.9) for index in range(8)]
    faces += [(16, None, 0.0), (18, None, 0.0)]
    faces += [(20 + 2 * index, 200.0, 0.9) for index in range(20)]
    points, _kinds = vertical.gated(faces, jump=51, confirm=8, bodies={}, crop=102)
    planned = vertical.camera(
        vertical.targets(points, 60, width=240, window=9), width=240, crop=102, rate=30
    )
    assert measured == planned
    assert measured[:18] == [0] * 18
    assert measured[-1] == 64
    assert len(commands) == len(set(measured))
    receipt = json.loads((store / "vertical.json").read_text(encoding="utf-8"))
    assert receipt["crop"] == {"width": 102, "height": 180, "aspect": [9, 16]}
    assert receipt["codeVersion"] == "vertical-reframe-v5"
    assert receipt["samples"] == {
        "reliable": 0.7, "fallback": 0.0, "extrapolated": 0.3, "longestGapSeconds": 0.6
    }  # fmt: skip
    assert receipt["frames"] == 60
    assert receipt["outputSHA256"] == sha256(output)
    again = vertical.prepared(
        source,
        store,
        ffmpeg="/missing",
        ffprobe="ffprobe",
        python="/py",
        yolo=YOLO,
        native="/native",
    )
    assert again == output
    assert tracked == [60]
    output.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="no longer matches"):
        vertical.prepared(
            source,
            store,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            python="/py",
            yolo=YOLO,
            native="/native",
        )


def test_matching_evidence_is_reused_and_stale_evidence_is_redone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked: list[Path] = []
    monkeypatch.setattr(vertical, "track", lambda _s, output, **_k: tracked.append(output))
    found = tmp_path / "track.json"
    found.write_text(json.dumps(evidence(tmp_path, "a" * 64, [])), encoding="utf-8")
    tools = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
    kept = vertical.evidence_for(
        tmp_path,
        tmp_path,
        "a" * 64,
        frames=1,
        python="/py",
        yolo=YOLO,
        tools=tools,
    )
    assert (kept, tracked) == (found, [])
    redone = vertical.evidence_for(
        tmp_path,
        tmp_path,
        "b" * 64,
        frames=1,
        python="/py",
        yolo=YOLO,
        tools=tools,
    )
    assert (redone, tracked, found.exists()) == (found, [found], False)


def test_the_source_directory_and_moved_frames_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = gradient(tmp_path)
    with pytest.raises(ReplayError, match="apart from its source"):
        vertical.prepared(
            source,
            tmp_path,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            python="/py",
            yolo=YOLO,
            native="/native",
        )
    digest = sha256(source)
    monkeypatch.setattr(vertical, "detect_bodies", lambda *_a, **_k: [])
    monkeypatch.setattr(
        vertical,
        "track",
        lambda _s, output, **_k: output.write_text(
            json.dumps(evidence(source, digest, [None])), encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "replay.source_vertical.video_pts",
        lambda movie, probe: video_pts(movie, probe)[: 59 if "partial" in movie.name else 60],
    )
    with pytest.raises(ReplayError, match="moved a source frame"):
        vertical.prepared(
            source,
            tmp_path / "store",
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            python="/py",
            yolo=YOLO,
            native="/native",
        )
    assert (tmp_path / "store" / source.name).exists() is False


def test_a_profile_face_leaves_gaps_and_a_weak_face_is_a_fallback() -> None:
    faces: list[tuple[int, float | None, float]] = [
        (0, 100.0, 0.9), (2, None, 0.0), (4, None, 0.0), (6, 104.0, 0.3), (8, 106.0, 0.9)
    ]  # fmt: skip
    points, kinds = vertical.gated(faces, jump=50, confirm=3, bodies={}, crop=100)
    assert points == [(0, 100.0), (2, None), (4, None), (6, 104.0), (8, 106.0)]
    assert kinds == ["reliable", "extrapolated", "extrapolated", "fallback", "reliable"]


def test_a_spurious_far_detection_is_a_gap() -> None:
    faces: list[tuple[int, float | None, float]] = [
        (0, 100.0, 0.9), (1, 400.0, 0.9), (2, 102.0, 0.9), (3, 390.0, 0.9), (4, 101.0, 0.9)
    ]  # fmt: skip
    points, kinds = vertical.gated(faces, jump=50, confirm=2, bodies={}, crop=100)
    assert points == [(0, 100.0), (1, None), (2, 102.0), (3, None), (4, 101.0)]
    assert kinds == ["reliable", "extrapolated", "reliable", "extrapolated", "reliable"]


def test_a_real_move_is_followed_once_confirmed() -> None:
    faces: list[tuple[int, float | None, float]] = [
        (0, 100.0, 0.9), (1, 400.0, 0.9), (2, None, 0.0), (3, 402.0, 0.9), (4, 404.0, 0.9),
        (5, 900.0, 0.9), (6, 405.0, 0.9),
    ]  # fmt: skip
    points, kinds = vertical.gated(faces, jump=50, confirm=3, bodies={}, crop=100)
    assert points == [
        (0, 100.0), (1, None), (2, None), (3, None), (4, 404.0), (5, None), (6, 405.0)
    ]  # fmt: skip
    assert kinds == [
        "reliable", "extrapolated", "extrapolated", "extrapolated", "reliable", "extrapolated",
        "reliable",
    ]  # fmt: skip


def test_a_gap_at_the_start_takes_the_first_reliable_position() -> None:
    faces: list[tuple[int, float | None, float]] = [
        (0, None, 0.0), (2, None, 0.0), (4, 200.0, 0.9), (6, 210.0, 0.9)
    ]  # fmt: skip
    points, kinds = vertical.gated(faces, jump=50, confirm=3, bodies={}, crop=100)
    assert vertical.targets(points, 8, width=400, window=1) == [
        200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 210.0, 210.0
    ]  # fmt: skip
    assert kinds == ["extrapolated", "extrapolated", "reliable", "reliable"]


def test_a_long_gap_holds_the_last_position_and_is_reported() -> None:
    faces: list[tuple[int, float | None, float]] = [(0, 120.0, 0.9)]
    faces += [(index, None, 0.0) for index in range(1, 301)]
    faces += [(301, 122.0, 0.4)]
    points, kinds = vertical.gated(faces, jump=50, confirm=3, bodies={}, crop=100)
    found = vertical.targets(points, 302, width=400, window=1)
    assert (found[0], found[150], found[300], found[301]) == (120.0, 120.0, 120.0, 122.0)
    shares = vertical.sample_shares(kinds, seconds_per_sample=0.5)
    assert shares == {
        "reliable": 0.0033, "fallback": 0.0033, "extrapolated": 0.9934, "longestGapSeconds": 150.0
    }  # fmt: skip


def test_a_left_facing_profile_keeps_the_face_edge_in_the_crop() -> None:
    assert vertical.facing((724.0, 1406.0), 1124.0, reach=274.0, least=30.0) == 1061.0


def test_a_right_facing_profile_keeps_the_face_edge_in_the_crop() -> None:
    assert vertical.facing((500.0, 1200.0), 760.0, reach=274.0, least=30.0) == 843.0


def test_a_figure_centred_on_the_last_face_is_kept_centred() -> None:
    assert vertical.facing((700.0, 1300.0), 1010.0, reach=274.0, least=30.0) == 1000.0
    assert vertical.facing((700.0, 1300.0), None, reach=274.0, least=30.0) == 1000.0


def test_a_narrow_figure_draws_the_crop_halfway_from_the_last_face_to_its_centre() -> None:
    assert vertical.facing((900.0, 1100.0), 1200.0, reach=274.0, least=30.0) == 1100.0


def test_faceless_samples_with_a_figure_are_fallbacks_facing_from_the_last_face() -> None:
    faces: list[tuple[int, float | None, float]] = [
        (0, 1124.0, 0.9), (2, None, 0.0), (4, None, 0.0), (6, 1120.0, 0.9)
    ]  # fmt: skip
    points, kinds = vertical.gated(
        faces, jump=304, confirm=8, bodies={2: (724.0, 1406.0)}, crop=608
    )
    assert points == [(0, 1124.0), (2, 1060.8), (4, None), (6, 1120.0)]
    assert kinds == ["reliable", "fallback", "extrapolated", "reliable"]


def test_figures_are_looked_for_on_faceless_samples_half_a_second_apart() -> None:
    faces: list[tuple[int, float | None, float]] = [
        (0, None, 0.0), (2, 5.0, 0.9), (4, None, 0.0), (16, None, 0.0), (18, None, 0.0),
        (32, None, 0.0),
    ]  # fmt: skip
    assert vertical.body_frames(faces, every=15) == [0, 16, 32]


def test_the_figure_detector_runs_in_batches_of_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[str] = []

    def run(command: list[str], **_k: object) -> SimpleNamespace:
        request = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        requests.append(f"{command[:3]} {len(request['samples'])}")
        rows = [{"bodies": [{"x": 1, "y": 2, "width": 3, "height": 4}]}] * len(request["samples"])
        return SimpleNamespace(stdout=json.dumps(rows))

    monkeypatch.setattr("replay.source_vertical.subprocess.run", run)
    seconds = [index / 10 for index in range(20)]
    found = vertical.detect_bodies(
        tmp_path / "s.mp4", list(range(10)), seconds, native="/native", directory=tmp_path
    )
    assert requests == [
        "['/native', 'measure-media', 'video_head_detect'] 8",
        "['/native', 'measure-media', 'video_head_detect'] 2",
    ]
    assert found[9] == {"frame": 9, "bodies": [{"x": 1, "y": 2, "width": 3, "height": 4}]}
    assert (tmp_path / "bodies-request.json").exists() is False
    monkeypatch.setattr(
        "replay.source_vertical.subprocess.run", lambda *_a, **_k: SimpleNamespace(stdout="[]")
    )
    with pytest.raises(ReplayError, match="different number"):
        vertical.detect_bodies(
            tmp_path / "s.mp4", [0], seconds, native="/native", directory=tmp_path
        )


def test_the_largest_figure_gives_the_span() -> None:
    spans = vertical.body_spans(
        {"frames": [
            {"frame": 4, "bodies": [{"x": 10, "y": 0, "width": 5, "height": 5},
                                    {"x": 100, "y": 0, "width": 50, "height": 90}]},
            {"frame": 6, "bodies": []},
        ]}
    )  # fmt: skip
    assert spans == {4: (100.0, 150.0)}
    with pytest.raises(ReplayError, match="Malformed figure"):
        vertical.body_spans({"frames": {}})
    with pytest.raises(ReplayError, match="Malformed figure"):
        vertical.body_spans({"frames": [{"frame": 1, "bodies": None}]})


def test_figure_evidence_is_reused_only_for_the_same_source_and_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[int]] = []

    def detect(_s: Path, frames: list[int], _t: list[float], **_k: object) -> list[object]:
        calls.append(frames)
        return []

    monkeypatch.setattr(vertical, "detect_bodies", detect)
    seconds = [0.0, 0.1]
    first = vertical.bodies_for(tmp_path, tmp_path, "a", frames=[0], seconds=seconds, native="/n")
    again = vertical.bodies_for(tmp_path, tmp_path, "a", frames=[0], seconds=seconds, native="/n")
    assert (first, again, calls) == (tmp_path / "bodies.json", first, [[0]])
    vertical.bodies_for(tmp_path, tmp_path, "a", frames=[0, 1], seconds=seconds, native="/n")
    assert calls == [[0], [0, 1]]
    assert json.loads(first.read_text(encoding="utf-8")) == {
        "sourceSHA256": "a", "requested": [0, 1], "frames": []
    }  # fmt: skip


def test_the_facing_strength_sets_how_far_toward_the_face_side_the_crop_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vertical, "FACING_STRENGTH", 1.0)
    assert vertical.facing((724.0, 1406.0), 1124.0, reach=274.0, least=30.0) == 998.0
    monkeypatch.setattr(vertical, "FACING_STRENGTH", 0.0)
    assert vertical.facing((724.0, 1406.0), 1124.0, reach=274.0, least=30.0) == 1124.0


def test_frames_placed_from_a_figure_are_marked_until_a_face_returns() -> None:
    points: list[tuple[int, float | None]] = [(0, 5.0), (2, 9.0), (4, None), (6, 5.0)]
    kinds = ["reliable", "fallback", "extrapolated", "reliable"]
    found = vertical.figure_frames(points, kinds, 8)
    assert found == [False, False, True, True, True, True, False, False]


def test_a_figure_moves_the_crop_within_the_face_dead_zone() -> None:
    target = [1124.0] * 3 + [1061.0] * 40
    still = vertical.camera(target, width=1920, crop=608, rate=30)
    loose = vertical.camera(target, width=1920, crop=608, rate=30, loose=[False] * 3 + [True] * 40)
    assert (still[0], still[-1]) == (820, 820)
    assert (loose[0], loose[-1]) == (820, 756)


def test_the_median_spans_one_and_a_half_seconds_of_samples() -> None:
    assert vertical.median_window(30.0, 2) == 23
    assert vertical.median_window(60.0, 2) == 45


def test_short_bursts_of_stray_positions_no_longer_move_the_crop() -> None:
    centres: list[tuple[int, float | None]] = [
        (2 * index, 1200.0 if index % 40 in range(20, 25) else 900.0) for index in range(300)
    ]
    short = vertical.camera(
        vertical.targets(centres, 600, width=1920, window=9), width=1920, crop=608, rate=30
    )
    wide = vertical.camera(
        vertical.targets(centres, 600, width=1920, window=vertical.median_window(30.0, 2)),
        width=1920, crop=608, rate=30,
    )  # fmt: skip
    assert len(set(short)) > 1
    assert set(wide) == {596}


def test_a_real_move_is_caught_up_within_two_seconds() -> None:
    centres: list[tuple[int, float | None]] = [
        (2 * index, 600.0 if index < 50 else 1300.0) for index in range(150)
    ]
    offsets = vertical.camera(
        vertical.targets(centres, 300, width=1920, window=vertical.median_window(30.0, 2)),
        width=1920, crop=608, rate=30,
    )  # fmt: skip
    assert (offsets[99], offsets[120], offsets[150], offsets[299]) == (296, 580, 996, 996)


def test_a_copy_made_with_other_camera_settings_is_kept_while_its_inputs_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = gradient(tmp_path)
    digest = sha256(source)

    def track(_source: Path, output: Path, **_k: object) -> None:
        output.write_text(json.dumps(evidence(source, digest, [None] * 30)), encoding="utf-8")

    monkeypatch.setattr(vertical, "track", track)
    monkeypatch.setattr(vertical, "detect_bodies", lambda *_a, **_k: [])
    store = tmp_path / "store"
    output = vertical.prepared(
        source, store, ffmpeg="ffmpeg", ffprobe="ffprobe", python="/py", yolo=YOLO, native="/n"
    )
    receipt = store / "vertical.json"
    made = json.loads(receipt.read_text(encoding="utf-8"))
    made["camera"]["medianSeconds"], made["codeVersion"] = 0.5, "vertical-reframe-v4"
    receipt.write_text(json.dumps(made), encoding="utf-8")
    again = vertical.prepared(
        source, store, ffmpeg="/missing", ffprobe="ffprobe", python="/py", yolo=YOLO, native="/n"
    )
    assert again == output
    assert "Reused vertical copy made by vertical-reframe-v4 with camera" in capsys.readouterr().err
    unnamed = {key: value for key, value in made.items() if key != "camera"}
    receipt.write_text(json.dumps(unnamed), encoding="utf-8")
    with pytest.raises(ReplayError, match="names no camera constants"):
        vertical.prepared(
            source, store, ffmpeg="/missing", ffprobe="ffprobe", python="/py", yolo=YOLO,
            native="/n",
        )  # fmt: skip
    receipt.write_text(json.dumps({**made, "camera": {"medianSeconds": 0.5}}), encoding="utf-8")
    with pytest.raises(ReplayError, match="names no camera constants"):
        vertical.prepared(
            source, store, ffmpeg="/missing", ffprobe="ffprobe", python="/py", yolo=YOLO,
            native="/n",
        )  # fmt: skip
    made["sourceSize"] = 1
    receipt.write_text(json.dumps(made), encoding="utf-8")
    with pytest.raises(ReplayError, match="no longer matches"):
        vertical.prepared(
            source, store, ffmpeg="ffmpeg", ffprobe="ffprobe", python="/py", yolo=YOLO,
            native="/n",
        )  # fmt: skip


def test_a_profile_still_draws_the_crop_midway_toward_its_face() -> None:
    faces: list[tuple[int, float | None, float]] = [(2 * index, 1124.0, 0.9) for index in range(30)]
    faces += [(60 + 2 * index, None, 0.0) for index in range(270)]
    bodies = {60 + 2 * index: (724.0, 1406.0) for index in range(0, 270, 8)}
    points, kinds = vertical.gated(faces, jump=304, confirm=8, bodies=bodies, crop=608)
    offsets = vertical.camera(
        vertical.targets(points, 600, width=1920, window=vertical.median_window(30.0, 2)),
        width=1920, crop=608, rate=30, loose=vertical.figure_frames(points, kinds, 600),
    )  # fmt: skip
    assert (offsets[0], offsets[-1]) == (820, 756)
