import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import native_render
from replay.core import ReplayError
from replay.fcp_project import Caption, CaptionStyle, Cut, Mask, Timeline
from replay.native_render import Encoding, PartWindow, batches, part_request, part_windows
from replay.recordings import record
from replay.stages import StageError, file_digest

STYLE = CaptionStyle(66, "#6CD4FF", 6, "#FFFFFF", 3, "#000000")
SOURCE = Path("/media/source.mp4")
TIMELINE = Timeline(
    30,
    1080,
    1920,
    (
        Cut(SOURCE, Fraction(10), Fraction(0), Fraction(3, 2)),
        Cut(SOURCE, Fraction(40), Fraction(3, 2), Fraction(1, 2)),
    ),
    (),
    (
        Caption(Fraction(0), Fraction(1, 2), 540.0, 554.0, ("一",), "#6CD4FF"),
        Caption(Fraction(1, 2), Fraction(2), 540.0, 554.0, ("二", "三"), "#6CD4FF"),
    ),
    STYLE,
)
FIRST = PartWindow(0, Fraction(0), Fraction(1))
SECOND = PartWindow(1, Fraction(1), Fraction(2))
ASS = (
    "Dialogue: 2,0:00:00.00,0:00:01.00,vertical,,0,0,0,,"
    "{\\an2\\pos(540,554)\\bord0\\1c&HFFD46C&}一\n"
)


def plan_part(start_frame: int, frames: int, start: float) -> dict[str, object]:
    return {
        "startFrame": start_frame,
        "durationFrames": frames,
        "captionBurn": {
            "ass": "a.ass",
            "style": {"fontSize": 66, "fill": "#6CD4FF", "whiteBorder": 6, "blackRim": 3},
        },
        "request": {
            "recipe": {
                "clips": [
                    {
                        "sourcePath": "/media/source.mp4",
                        "selection": {"startSeconds": start, "durationSeconds": 2, "rate": 1},
                    }
                ],
                "video": {"width": 1080, "height": 1920, "frameRate": 30, "resizeMode": "fit"},
            }
        },
    }


def test_part_windows_follow_planned_frames() -> None:
    assert part_windows(
        [{"startFrame": 0, "durationFrames": 45}, {"startFrame": 45, "durationFrames": 15}], 30
    ) == [
        PartWindow(0, Fraction(0), Fraction(3, 2)),
        PartWindow(1, Fraction(3, 2), Fraction(2)),
    ]


def test_part_windows_reject_invalid_parts() -> None:
    with pytest.raises(ReplayError, match="Invalid render plan part"):
        part_windows([1], 30)
    with pytest.raises(ReplayError, match="Invalid render plan frames"):
        part_windows([{"startFrame": 0, "durationFrames": True}], 30)


def test_first_part_request_trims_cuts_and_keeps_masks_and_captions_in_window() -> None:
    assert part_request(TIMELINE, FIRST, Path("/out"), Encoding()) == {
        "recipe": {
            "clips": [
                {
                    "sourcePath": "/media/source.mp4",
                    "selection": {"startSeconds": 10.0, "durationSeconds": 1.0, "rate": 1},
                }
            ],
            "video": {
                "width": 1080,
                "height": 1920,
                "frameRate": 30,
                "resizeMode": "fit",
                "encoding": {
                    "codec": "hevc",
                    "averageBitRate": 4500000,
                    "allowFrameReordering": True,
                },
                "styledCaptions": [
                    {
                        "text": "一",
                        "startSeconds": 0.0,
                        "endSeconds": 0.5,
                        "x": 540.0,
                        "bottom": 554.0,
                    },
                    {
                        "text": "二\n三",
                        "startSeconds": 0.5,
                        "endSeconds": 1.0,
                        "x": 540.0,
                        "bottom": 554.0,
                    },
                ],
                "captionAppearance": {
                    "fontName": "HiraginoSans-W6",
                    "assFontSize": 66,
                    "fill": "#6CD4FF",
                    "border": 6,
                    "borderColor": "#FFFFFF",
                    "rim": 3,
                    "rimColor": "#000000",
                },
            },
        },
        "outputDirectory": "/out",
        "outputName": "part-000000.mp4",
    }


def test_second_part_request_shifts_to_local_time_across_a_cut() -> None:
    request = part_request(TIMELINE, SECOND, Path("/out"), Encoding("h264", 6000000, False))
    assert request["outputName"] == "part-000001.mp4"
    assert request["recipe"] == {
        "clips": [
            {
                "sourcePath": "/media/source.mp4",
                "selection": {"startSeconds": 11.0, "durationSeconds": 0.5, "rate": 1},
            },
            {
                "sourcePath": "/media/source.mp4",
                "selection": {"startSeconds": 40.0, "durationSeconds": 0.5, "rate": 1},
            },
        ],
        "video": {
            "width": 1080,
            "height": 1920,
            "frameRate": 30,
            "resizeMode": "fit",
            "encoding": {
                "codec": "h264",
                "averageBitRate": 6000000,
                "allowFrameReordering": False,
            },
            "styledCaptions": [
                {
                    "text": "二\n三",
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "x": 540.0,
                    "bottom": 554.0,
                }
            ],
            "captionAppearance": {
                "fontName": "HiraginoSans-W6",
                "assFontSize": 66,
                "fill": "#6CD4FF",
                "border": 6,
                "borderColor": "#FFFFFF",
                "rim": 3,
                "rimColor": "#000000",
            },
        },
    }


def test_part_without_masks_or_captions_has_only_clips_and_encoding() -> None:
    timeline = Timeline(
        30, 1080, 1920, (Cut(SOURCE, Fraction(5), Fraction(0), Fraction(1)),), (), (), None
    )
    assert part_request(timeline, FIRST, Path("/out"), Encoding())["recipe"] == {
        "clips": [
            {
                "sourcePath": "/media/source.mp4",
                "selection": {"startSeconds": 5.0, "durationSeconds": 1.0, "rate": 1},
            }
        ],
        "video": {
            "width": 1080,
            "height": 1920,
            "frameRate": 30,
            "resizeMode": "fit",
            "encoding": {"codec": "hevc", "averageBitRate": 4500000, "allowFrameReordering": True},
        },
    }


def test_captions_need_a_style_and_one_fill_per_part() -> None:
    unstyled = Timeline(30, 1080, 1920, TIMELINE.cuts, (), TIMELINE.captions, None)
    with pytest.raises(ReplayError, match="need a caption style"):
        part_request(unstyled, FIRST, Path("/out"), Encoding())
    mixed = Timeline(
        30,
        1080,
        1920,
        TIMELINE.cuts,
        (),
        (
            Caption(Fraction(0), Fraction(1, 2), 540.0, 554.0, ("一",), "#6CD4FF"),
            Caption(Fraction(1, 2), Fraction(1), 540.0, 554.0, ("二",), "#FFFFFF"),
        ),
        STYLE,
    )
    with pytest.raises(ReplayError, match="one fill color per part"):
        part_request(mixed, FIRST, Path("/out"), Encoding())


def test_a_different_fill_in_another_part_is_allowed() -> None:
    timeline = Timeline(
        30,
        1080,
        1920,
        TIMELINE.cuts,
        (),
        (
            Caption(Fraction(0), Fraction(1, 2), 540.0, 554.0, ("一",), "#6CD4FF"),
            Caption(Fraction(3, 2), Fraction(2), 540.0, 554.0, ("二",), "#FFFFFF"),
        ),
        STYLE,
    )
    request = part_request(timeline, SECOND, Path("/out"), Encoding())
    assert request["recipe"] == {
        "clips": [
            {
                "sourcePath": "/media/source.mp4",
                "selection": {"startSeconds": 11.0, "durationSeconds": 0.5, "rate": 1},
            },
            {
                "sourcePath": "/media/source.mp4",
                "selection": {"startSeconds": 40.0, "durationSeconds": 0.5, "rate": 1},
            },
        ],
        "video": {
            "width": 1080,
            "height": 1920,
            "frameRate": 30,
            "resizeMode": "fit",
            "encoding": {"codec": "hevc", "averageBitRate": 4500000, "allowFrameReordering": True},
            "styledCaptions": [
                {
                    "text": "二",
                    "startSeconds": 0.5,
                    "endSeconds": 1.0,
                    "x": 540.0,
                    "bottom": 554.0,
                }
            ],
            "captionAppearance": {
                "fontName": "HiraginoSans-W6",
                "assFontSize": 66,
                "fill": "#FFFFFF",
                "border": 6,
                "borderColor": "#FFFFFF",
                "rim": 3,
                "rimColor": "#000000",
            },
        },
    }


def test_batches_keep_order_and_respect_the_tool_limit() -> None:
    requests: list[dict[str, object]] = [{"n": 1}, {"n": 2}, {"n": 3}]
    assert list(batches(requests, 2)) == [[{"n": 1}, {"n": 2}], [{"n": 3}]]
    assert list(batches(requests)) == [[{"n": 1}, {"n": 2}, {"n": 3}]]
    with pytest.raises(ReplayError, match="Batch size must be 1-32"):
        list(batches(requests, 33))


def test_cli_writes_batch_payloads_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "a.ass").write_text(ASS, encoding="utf-8")
    plan = tmp_path / "render-plan.json"
    plan.write_text(json.dumps([plan_part(0, 30, 10), plan_part(30, 30, 20)]), encoding="utf-8")
    output = tmp_path / "batches.json"
    assert native_render.main([str(plan), str(output), "--render-output", "/out"]) == 0
    assert capsys.readouterr().out == '{"parts": 2, "batches": 1}\n'
    payloads = json.loads(output.read_text(encoding="utf-8"))
    assert [item["outputName"] for item in payloads[0]["items"]] == [
        "part-000000.mp4",
        "part-000001.mp4",
    ]
    assert payloads[0]["concurrency"] == 3
    assert payloads[0]["items"][1]["recipe"]["clips"] == [
        {
            "sourcePath": "/media/source.mp4",
            "selection": {"startSeconds": 20.0, "durationSeconds": 1.0, "rate": 1},
        }
    ]
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        native_render.main([str(plan), str(output), "--render-output", "/out"])


@pytest.fixture
def foreground_inputs(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "a.ass").write_text(ASS, encoding="utf-8")
    plan = tmp_path / "render-plan.json"
    plan.write_text(json.dumps([plan_part(0, 30, 10), plan_part(30, 30, 20)]), encoding="utf-8")
    first, second = tmp_path / "first.mov", tmp_path / "second.mov"
    first.write_bytes(b"synthetic identity fixture, not decoded media: first")
    second.write_bytes(b"synthetic identity fixture, not decoded media: second")
    manifest = tmp_path / "foreground.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "renderPlanSHA256": file_digest(plan),
                "parts": [
                    {
                        "startFrame": 0,
                        "durationFrames": 30,
                        "path": str(first),
                        "sha256": file_digest(first),
                    },
                    {
                        "startFrame": 30,
                        "durationFrames": 30,
                        "path": str(second),
                        "sha256": file_digest(second),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return plan, manifest


def test_foreground_cli_maps_each_part_without_duplicate_captions(
    foreground_inputs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    plan, manifest = foreground_inputs
    output = tmp_path / "foreground-batches.json"
    assert (
        native_render.main(
            [
                str(plan),
                str(output),
                "--render-output",
                "/out",
                "--foreground-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    payloads = json.loads(output.read_text(encoding="utf-8"))
    first, second = payloads[0]["items"]
    assert first["recipe"]["video"]["foregroundVideoPath"] == str(tmp_path / "first.mov")
    assert second["recipe"]["video"]["foregroundVideoPath"] == str(tmp_path / "second.mov")
    assert "styledCaptions" not in first["recipe"]["video"]
    assert "captionAppearance" not in second["recipe"]["video"]
    assert second["recipe"]["clips"] == [
        {
            "sourcePath": "/media/source.mp4",
            "selection": {
                "startSeconds": 20.0,
                "durationSeconds": 1.0,
                "rate": 1,
            },
        }
    ]


def test_foreground_does_not_mutate_editable_timeline_or_add_blur() -> None:
    request = part_request(TIMELINE, SECOND, Path("/out"), Encoding(), foreground=Path("/part.mov"))
    video = record(record(request["recipe"])["video"])
    assert video["foregroundVideoPath"] == "/part.mov"
    assert "masks" not in video
    assert "styledCaptions" not in video
    assert "captionAppearance" not in video
    assert TIMELINE.captions[0].lines == ("一",)
    assert TIMELINE.style == CaptionStyle(66, "#6CD4FF", 6, "#FFFFFF", 3, "#000000")


def test_fixed_blur_is_rejected_even_with_foreground() -> None:
    timeline = replace(TIMELINE, masks=(Mask(Fraction(0), Fraction(1), 0, 10, 100, 40, 64),))
    for foreground in (None, Path("/part.mov")):
        with pytest.raises(ReplayError, match="detected local repair"):
            part_request(timeline, FIRST, Path("/out"), Encoding(), foreground=foreground)


@pytest.mark.parametrize("path", [Path("part.mov"), Path("/bad\0.mov")])
def test_foreground_request_refuses_nonlocal_paths(path: Path) -> None:
    with pytest.raises(ReplayError, match="absolute local path"):
        part_request(TIMELINE, FIRST, Path("/out"), Encoding(), foreground=path)


@pytest.mark.parametrize(
    "patch",
    [
        {"version": True},
        {"version": 2},
        {"extra": 1},
        {"renderPlanSHA256": "changed"},
        {"parts": None},
        {"parts": []},
    ],
)
def test_foreground_manifest_rejects_changed_plan_and_incomplete_mapping(
    foreground_inputs: tuple[Path, Path],
    patch: dict[str, object],
    tmp_path: Path,
) -> None:
    plan, manifest = foreground_inputs
    data = record(json.loads(manifest.read_text(encoding="utf-8")))
    data.update(patch)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "refused.json"
    with pytest.raises(ReplayError, match="Foreground manifest"):
        native_render.main(
            [
                str(plan),
                str(output),
                "--render-output",
                "/out",
                "--foreground-manifest",
                str(manifest),
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "patch",
    [
        {"startFrame": True},
        {"startFrame": 30},
        {"durationFrames": True},
        {"durationFrames": 60},
        {"extra": 1},
        {"path": 3},
        {"path": "relative.mov"},
        {"path": "/bad\0.mov"},
        {"sha256": None},
        {"sha256": "changed"},
    ],
)
def test_foreground_manifest_rejects_invalid_part_fields(
    foreground_inputs: tuple[Path, Path],
    patch: dict[str, object],
) -> None:
    plan, manifest = foreground_inputs
    data = record(json.loads(manifest.read_text(encoding="utf-8")))
    parts = data["parts"]
    assert isinstance(parts, list)
    record(parts[0]).update(patch)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="Foreground"):
        native_render.load_foregrounds(manifest, plan=plan, windows=[FIRST, SECOND], rate=30)


@pytest.mark.parametrize("payload", [None, [], {"version": 1}])
def test_foreground_manifest_rejects_invalid_schema(
    foreground_inputs: tuple[Path, Path],
    payload: object,
) -> None:
    plan, manifest = foreground_inputs
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReplayError, match="does not match"):
        native_render.load_foregrounds(manifest, plan=plan, windows=[FIRST, SECOND], rate=30)


@pytest.mark.parametrize("kind", ["linked", "oversized", "missing"])
def test_foreground_manifest_requires_bounded_regular_storage(
    foreground_inputs: tuple[Path, Path],
    tmp_path: Path,
    kind: str,
) -> None:
    plan, manifest = foreground_inputs
    other = tmp_path / "other.json"
    if kind == "linked":
        other.symlink_to(manifest)
    elif kind == "oversized":
        other.write_bytes(b" " * 1_048_577)
    with pytest.raises(ReplayError, match="bounded regular"):
        native_render.load_foregrounds(other, plan=plan, windows=[FIRST, SECOND], rate=30)


def test_foreground_modified_or_linked_media_is_not_accepted(
    foreground_inputs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    plan, manifest = foreground_inputs
    first = tmp_path / "first.mov"
    first.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="fingerprint differs"):
        native_render.load_foregrounds(manifest, plan=plan, windows=[FIRST, SECOND], rate=30)
    first.unlink()
    first.symlink_to(tmp_path / "second.mov")
    with pytest.raises(StageError, match="regular stage artifact"):
        native_render.load_foregrounds(manifest, plan=plan, windows=[FIRST, SECOND], rate=30)


def test_cli_rejects_relative_output_and_non_list_plans(tmp_path: Path) -> None:
    plan = tmp_path / "render-plan.json"
    plan.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="relative render directory"):
        native_render.main([str(plan), str(tmp_path / "a.json"), "--render-output", "out"])
    with pytest.raises(ReplayError, match="Render plan is empty"):
        native_render.main([str(plan), str(tmp_path / "b.json"), "--render-output", "/out"])
