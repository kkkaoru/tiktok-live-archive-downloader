"""Caption lines come from caption pixels; frames are kept only with a trusted face label."""

import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay import yolo_joint_dataset as joint
from replay.core import ReplayError
from replay.yolo_labels import PixelBox

PINK = (180, 80, 255)  # BGR, HSV hue ~157
WHITE = (250, 250, 250)


def blank() -> np.ndarray:
    return np.full((1138, 640, 3), 90, dtype=np.uint8)


def english_line(image: np.ndarray, *, left: int, top: int, glyphs: int) -> None:
    """Pink glyphs 8 px wide with 4 px gaps, 14 px tall."""
    for index in range(glyphs):
        x = left + 12 * index
        cv2.rectangle(image, (x, top), (x + 7, top + 13), PINK, -1)


def japanese_line(image: np.ndarray, *, left: int, top: int, glyphs: int) -> None:
    """White glyphs with a 1 px pink outline, 16 px squares with 3 px gaps."""
    for index in range(glyphs):
        x = left + 19 * index
        cv2.rectangle(image, (x, top), (x + 15, top + 15), PINK, 1)
        cv2.rectangle(image, (x + 1, top + 1), (x + 14, top + 14), WHITE, -1)


def test_a_pink_english_line_is_one_box() -> None:
    image = blank()
    english_line(image, left=200, top=800, glyphs=20)
    assert joint.caption_lines(image) == [(198, 798, 438, 816)]


def test_a_white_line_with_a_pink_outline_is_one_box() -> None:
    image = blank()
    japanese_line(image, left=100, top=700, glyphs=10)
    assert joint.caption_lines(image) == [(98, 698, 289, 718)]


def test_two_lines_stay_two_boxes() -> None:
    image = blank()
    japanese_line(image, left=100, top=700, glyphs=10)
    english_line(image, left=150, top=730, glyphs=10)
    assert joint.caption_lines(image) == [(98, 698, 289, 718), (148, 728, 268, 746)]


def test_words_split_by_a_gap_are_one_line() -> None:
    image = blank()
    english_line(image, left=50, top=800, glyphs=5)
    english_line(image, left=130, top=800, glyphs=5)
    assert joint.caption_lines(image) == [(48, 798, 188, 816)]


def test_boxes_of_one_row_merge_within_one_and_a_half_line_heights() -> None:
    assert joint.merge_row([(100, 10, 200, 30), (225, 12, 300, 30), (400, 10, 500, 30)]) == [
        (100, 10, 300, 30),
        (400, 10, 500, 30),
    ]


def test_boxes_of_different_rows_do_not_merge() -> None:
    assert joint.merge_row([(100, 10, 200, 30), (100, 40, 200, 60)]) == [
        (100, 10, 200, 30),
        (100, 40, 200, 60),
    ]


def test_caption_coloured_text_above_the_caption_zone_is_ignored() -> None:
    image = blank()
    english_line(image, left=200, top=300, glyphs=20)
    assert joint.caption_lines(image) == []


def test_white_without_a_pink_outline_and_tiny_pink_marks_are_not_captions() -> None:
    image = blank()
    cv2.rectangle(image, (100, 700), (400, 716), WHITE, -1)
    cv2.rectangle(image, (500, 900), (506, 905), PINK, -1)
    assert joint.caption_lines(image) == []


def test_a_tall_pink_shape_is_not_a_text_line() -> None:
    image = blank()
    cv2.rectangle(image, (100, 700), (400, 800), PINK, -1)
    assert joint.caption_lines(image) == []


def test_label_rows_scale_the_canvas_face_and_add_caption_lines() -> None:
    image = blank()
    english_line(image, left=200, top=800, glyphs=20)
    rows = joint.label_rows(image, PixelBox(270.0, 480.0, 540.0, 540.0), canvas=(1080, 1920))
    assert rows == [
        "0 0.500000 0.390549 0.500000 0.281195",
        "1 0.496875 0.709139 0.375000 0.015817",
    ]


def test_a_face_outside_the_frame_is_refused() -> None:
    with pytest.raises(ReplayError, match="outside the frame"):
        joint.label_rows(blank(), PixelBox(2000.0, 10.0, 50.0, 50.0), canvas=(1080, 1920))


def manifest(tmp_path: Path, recordings: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"output": str(tmp_path / "out"), "recordings": recordings}), encoding="utf-8"
    )
    return path


def recording(name: str = "a", role: str = "train", step: object = 60) -> dict[str, object]:
    return {"name": name, "source": "s.mp4", "faceResponses": "r.txt", "step": step, "role": role}


def test_a_manifest_names_recordings_and_the_output(tmp_path: Path) -> None:
    (tmp_path / "r.txt").write_text("one.json\ntwo.json\n", encoding="utf-8")
    item = recording() | {"faceResponses": str(tmp_path / "r.txt")}
    recordings, output = joint.parse_manifest(manifest(tmp_path, [item]))
    assert output == tmp_path / "out"
    assert recordings == [
        joint.Recording(
            "a", Path("s.mp4"), (Path("one.json"), Path("two.json")), 60, "train",
            ("detection", "tracking"),
        )
    ]  # fmt: skip


def test_detection_only_faces_drop_tracked_frames(tmp_path: Path) -> None:
    region = {"x": 10, "y": 20, "width": 30, "height": 40}
    frames = [
        {"method": "detection", "faces": [{"region": region, "confidence": 0.5}]},
        {"method": "tracking", "faces": [{"region": region, "confidence": 0.99}]},
    ]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(frames), encoding="utf-8")
    assert joint.trusted_faces([path, path], ["detection"]) == {
        0: PixelBox(10.0, 20.0, 30.0, 40.0),
        2: PixelBox(10.0, 20.0, 30.0, 40.0),
    }


def test_a_face_response_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="list of frames"):
        joint.trusted_faces([path], ["detection"])


@pytest.mark.parametrize(
    ("recordings", "message"),
    [
        ("x", "recordings list"),
        (["x"], "must be an object"),
        ([recording(name="a-b")], "plain identifier"),
        ([recording() | {"source": 1}], "needs a source"),
        ([recording() | {"faceResponses": None}], "exactly one of windowIndex"),
        ([recording(step=0)], "positive step"),
        ([recording(role="other")], "known role"),
        ([recording() | {"faceMethods": ["guess"]}], "faceMethods"),
    ],
)
def test_an_invalid_manifest_is_refused(tmp_path: Path, recordings: object, message: str) -> None:
    (tmp_path / "r.txt").write_text("", encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        joint.parse_manifest(manifest(tmp_path, recordings))


def test_a_window_index_names_each_windows_response(tmp_path: Path) -> None:
    index = tmp_path / "window-index.json"
    index.write_text(
        json.dumps({"windows": [{"evidence": "/w/0000"}, {"evidence": "/w/0001"}]}),
        encoding="utf-8",
    )
    item = recording() | {"faceResponses": None, "windowIndex": str(index)}
    del item["faceResponses"]
    recordings, _ = joint.parse_manifest(manifest(tmp_path, [item]))
    assert recordings[0].face_responses == (
        Path("/w/0000/response.json"),
        Path("/w/0001/response.json"),
    )


def test_a_manifest_may_keep_vision_detections_only(tmp_path: Path) -> None:
    (tmp_path / "r.txt").write_text("", encoding="utf-8")
    item = recording() | {"faceResponses": str(tmp_path / "r.txt"), "faceMethods": ["detection"]}
    recordings, _ = joint.parse_manifest(manifest(tmp_path, [item]))
    assert recordings[0].face_methods == ("detection",)


def test_an_empty_window_index_is_refused(tmp_path: Path) -> None:
    index = tmp_path / "window-index.json"
    index.write_text(json.dumps({"windows": []}), encoding="utf-8")
    item = {"name": "a", "source": "s.mp4", "windowIndex": str(index), "step": 1, "role": "test"}
    with pytest.raises(ReplayError, match="lists no windows"):
        joint.parse_manifest(manifest(tmp_path, [item]))


def test_duplicate_recording_names_and_a_missing_output_are_refused(tmp_path: Path) -> None:
    (tmp_path / "r.txt").write_text("", encoding="utf-8")
    item = recording() | {"faceResponses": str(tmp_path / "r.txt")}
    with pytest.raises(ReplayError, match="unique"):
        joint.parse_manifest(manifest(tmp_path, [item, item]))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"recordings": []}), encoding="utf-8")
    with pytest.raises(ReplayError, match="output directory"):
        joint.parse_manifest(path)


def frames_with_captions() -> Iterator[tuple[int, np.ndarray]]:
    first, second = blank(), blank()
    english_line(first, left=200, top=800, glyphs=20)
    yield 0, first
    yield 60, second
    yield 120, blank()


def test_only_frames_with_a_face_label_are_written(tmp_path: Path) -> None:
    item = joint.Recording("rec", Path("s.mp4"), (), 60, "train")
    face = PixelBox(270.0, 480.0, 540.0, 540.0)
    counts = joint.write_recording(
        item, frames_with_captions(), {0: face, 60: face}, tmp_path, canvas=(1080, 1920)
    )
    assert counts == {"frames": 2, "captionLines": 1, "framesWithCaption": 1}
    assert sorted(path.name for path in (tmp_path / "images" / "rec").iterdir()) == [
        "rec_0000000.jpg",
        "rec_0000060.jpg",
    ]
    assert (tmp_path / "labels" / "rec" / "rec_0000060.txt").read_text(encoding="utf-8") == (
        "0 0.500000 0.390549 0.500000 0.281195\n"
    )


def test_the_data_description_lists_images_per_role(tmp_path: Path) -> None:
    (tmp_path / "images" / "a").mkdir(parents=True)
    (tmp_path / "images" / "a" / "a_1.jpg").write_bytes(b"")
    (tmp_path / "images" / "b").mkdir(parents=True)
    (tmp_path / "images" / "b" / "b_1.jpg").write_bytes(b"")
    recordings = [
        joint.Recording("a", Path("s"), (), 1, "train"),
        joint.Recording("b", Path("s"), (), 1, "test"),
    ]
    assert joint.data_yaml(tmp_path, recordings) == (
        f"path: {tmp_path}\ntrain: {tmp_path / 'train.txt'}\ntest: {tmp_path / 'test.txt'}\n"
        "names:\n  0: avatar_face\n  1: source_caption\n"
    )
    assert (tmp_path / "train.txt").read_text(encoding="utf-8") == f"{tmp_path}/images/a/a_1.jpg\n"
    assert (tmp_path / "val.txt").read_text(encoding="utf-8") == ""


def response(tmp_path: Path) -> Path:
    frames = [
        {"method": "detection", "faces": [{"region": {"x": 270, "y": 480, "width": 540,
                                                      "height": 540}, "confidence": 0.8}]},
    ] + [{"method": "missing", "faces": []}] * 120  # fmt: skip
    path = tmp_path / "response.json"
    path.write_text(json.dumps(frames), encoding="utf-8")
    (tmp_path / "r.txt").write_text(f"{path}\n", encoding="utf-8")
    return tmp_path / "r.txt"


def test_build_writes_images_labels_and_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = response(tmp_path)
    item = recording() | {"faceResponses": str(responses)}
    monkeypatch.setattr(joint, "probe_video", lambda source, ffprobe: (1080, 1920, 30))
    monkeypatch.setattr(joint, "video_frames", lambda source, **options: frames_with_captions())
    summary = joint.main([str(manifest(tmp_path, [item])), "--jobs", "2"])
    assert summary == 0
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8")) == {
        "a": {"role": "train", "step": 60, "frames": 1, "captionLines": 1, "framesWithCaption": 1}
    }
    assert (tmp_path / "out" / "data.yaml").is_file()


def test_build_refuses_an_existing_output_no_recording_and_no_faces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([{"method": "missing", "faces": []}]), encoding="utf-8")
    (tmp_path / "r.txt").write_text(f"{empty}\n", encoding="utf-8")
    item = recording() | {"faceResponses": str(tmp_path / "r.txt")}
    with pytest.raises(ReplayError, match="no recording"):
        joint.build(manifest(tmp_path, []))
    monkeypatch.setattr(joint, "probe_video", lambda source, ffprobe: (1080, 1920, 30))
    with pytest.raises(ReplayError, match="no trusted face box"):
        joint.build(manifest(tmp_path, [item]))
    with pytest.raises(ReplayError, match="overwrite"):
        joint.build(manifest(tmp_path, [item]))
