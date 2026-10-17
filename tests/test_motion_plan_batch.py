"""Synthetic title-plan bundles verify safe, resumable Motion expansion."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import motion_finish, motion_plan_batch, motion_stages
from replay.core import ReplayError


def _first_title_part(value: dict[str, object]) -> dict[str, object]:
    parts = value["titleParts"]
    assert isinstance(parts, list)
    part = parts[0]
    assert isinstance(part, dict)
    return part


def _bundle() -> dict[str, object]:
    part = {
        "version": 1,
        "width": 1280,
        "height": 720,
        "rate": 30,
        "frames": 12,
        "offsetSeconds": 0.0,
        "offsetFrame": 0,
        "expectedText": "Title",
        "placements": [
            {"text": "字幕", "startFrame": 0, "endFrame": 6, "x": 4, "y": 9},
            {"text": "表示", "startFrame": 6, "endFrame": 12, "x": 4, "y": 9},
        ],
    }
    second = {
        **part,
        "offsetSeconds": 0.4,
        "offsetFrame": 12,
        "placements": [{"text": "位置", "startFrame": 0, "endFrame": 12, "x": 4, "y": 9}],
    }
    return {
        "version": 1,
        "canvas": {"width": 1280, "height": 720, "rate": 30},
        "slot": {"centerX": 884, "left": 497, "right": 1271, "width": 774},
        "fontSize": 104,
        "parts": 2,
        "placements": 3,
        "titleParts": [part, second],
    }


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    template = root / "template.moti"
    bundle = root / "plan.json"
    output = root / "expanded"
    template.write_text("synthetic template", encoding="utf-8")
    bundle.write_text(json.dumps(_bundle(), ensure_ascii=False), encoding="utf-8")
    return template, bundle, output


def _style_template() -> str:
    stops = "".join(
        "<parameter factoryID='4'><parameter id='3'>"
        + "".join(
            f"<parameter id='{index}'><curve value='{value}'/></parameter>"
            for index, value in enumerate(rgb, start=1)
        )
        + "</parameter></parameter>"
        for rgb in ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    )
    style = (
        "<style><parameter id='14'><parameter id='17'><parameter id='1'>"
        + stops
        + "</parameter></parameter></parameter></style>"
    )
    return (
        "<ozml><scene><layer><scenegroup>"
        f"<scenenode factoryID='16' name='Caption Face'>{style}</scenenode>"
        "<scenenode factoryID='16' name='Caption Outer Rim'><style/></scenenode>"
        "</scenegroup></layer></scene></ozml>"
    )


def test_reads_selected_gradient_from_nested_face_title() -> None:
    assert motion_plan_batch._style_controls(_style_template()) == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )


def test_expands_parts_with_safe_slot_and_resumable_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, bundle, output = _inputs(tmp_path)
    monkeypatch.setattr(
        motion_plan_batch, "_style_controls", lambda _: ((0.1, 0.2, 0.3), (0.2, 0.3, 0.4))
    )
    monkeypatch.setattr(
        motion_stages,
        "prepare",
        lambda _template, plan: json.dumps(plan, ensure_ascii=False, sort_keys=True),
    )
    monkeypatch.setattr(motion_finish, "finish_sequence", lambda source, *, style: source)

    manifest = motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))
    first = json.loads((output / "part-0000.moti").read_text(encoding="utf-8"))
    layout = json.loads(manifest.read_text(encoding="utf-8"))["layout"]
    assert first["coordinateSpace"] == "canvas-top-left"
    assert {item["x"] for item in first["placements"]} == {884.0}
    assert {item["y"] for item in first["placements"]} == {640.0}
    assert layout["horizontalCenterOutput"] == 1326.0
    assert layout["verticalAnchorOutput"] == 960.0
    first_bytes = (output / "part-0000.moti").read_bytes()
    assert (
        motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))
        == manifest
    )
    assert (output / "part-0000.moti").read_bytes() == first_bytes
    receipt = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(receipt["completed"]) == 1
    assert receipt["layout"]["partCount"] == 1
    assert receipt["motionRenderVerified"] is False


def test_coalesces_overlapping_recognized_units_without_inventing_clocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, bundle, output = _inputs(tmp_path)
    value = _bundle()
    title_parts = value["titleParts"]
    assert isinstance(title_parts, list)
    first = title_parts[0]
    second = title_parts[1]
    assert isinstance(first, dict) and isinstance(second, dict)
    first["frames"] = 5
    first["placements"] = [{"text": "語", "startFrame": 1, "endFrame": 5, "x": 0, "y": 0}]
    second["offsetFrame"] = 4
    second["offsetSeconds"] = 4 / 30
    second["frames"] = 5
    second["placements"] = [{"text": "句", "startFrame": 0, "endFrame": 3, "x": 0, "y": 0}]
    value["titleParts"] = [first, second]
    value["parts"] = 2
    value["placements"] = 2
    bundle.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        motion_plan_batch,
        "_style_controls",
        lambda _: ((0.1, 0.2, 0.3), (0.2, 0.3, 0.4)),
    )
    monkeypatch.setattr(
        motion_stages,
        "prepare",
        lambda _template, plan: json.dumps(plan, ensure_ascii=False, sort_keys=True),
    )
    monkeypatch.setattr(
        motion_finish,
        "finish_sequence",
        lambda source, *, style: source,
    )

    motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))
    result = json.loads((output / "part-0000.moti").read_text(encoding="utf-8"))
    assert result["placements"] == [
        {"text": "語句", "startFrame": 0, "endFrame": 6, "x": 884.0, "y": 640.0}
    ]


def test_anchors_multiline_blocks_by_measured_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, bundle, output = _inputs(tmp_path)
    value = _bundle()
    part = _first_title_part(value)
    part["placements"] = [
        {"text": "一行", "startFrame": 0, "endFrame": 4, "x": 0, "y": 0},
        {"text": "二行\n目", "startFrame": 4, "endFrame": 8, "x": 0, "y": 0},
        {"text": "三\n行\n目", "startFrame": 8, "endFrame": 12, "x": 0, "y": 0},
    ]
    part["frames"] = 12
    value["titleParts"] = [part]
    value["parts"] = 1
    value["placements"] = 3
    bundle.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        motion_plan_batch,
        "_style_controls",
        lambda _: ((0.1, 0.2, 0.3), (0.2, 0.3, 0.4)),
    )
    monkeypatch.setattr(
        motion_stages,
        "prepare",
        lambda _template, plan: json.dumps(plan, ensure_ascii=False, sort_keys=True),
    )
    monkeypatch.setattr(motion_finish, "finish_sequence", lambda source, *, style: source)
    motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))
    result = json.loads((output / "part-0000.moti").read_text(encoding="utf-8"))
    assert [item["y"] for item in result["placements"]] == [640.0, 548.0, 456.0]


def test_recomputes_anchor_when_merging_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, bundle, output = _inputs(tmp_path)
    value = _bundle()
    part = _first_title_part(value)
    part["placements"] = [
        {"text": "上\n", "startFrame": 0, "endFrame": 6, "x": 0, "y": 0},
        {"text": "下", "startFrame": 5, "endFrame": 12, "x": 0, "y": 0},
    ]
    part["frames"] = 12
    value["titleParts"] = [part]
    value["parts"] = 1
    value["placements"] = 2
    bundle.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        motion_plan_batch,
        "_style_controls",
        lambda _: ((0.1, 0.2, 0.3), (0.2, 0.3, 0.4)),
    )
    monkeypatch.setattr(
        motion_stages,
        "prepare",
        lambda _template, plan: json.dumps(plan, ensure_ascii=False, sort_keys=True),
    )
    monkeypatch.setattr(motion_finish, "finish_sequence", lambda source, *, style: source)
    motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))
    result = json.loads((output / "part-0000.moti").read_text(encoding="utf-8"))
    assert result["placements"] == [
        {"text": "上\n下", "startFrame": 0, "endFrame": 12, "x": 884.0, "y": 548.0}
    ]


def test_rejects_bad_anchor_geometry() -> None:
    ratio = Fraction(8, 9)
    assert (
        motion_plan_batch._anchored_y(
            "あ", height=720, anchor_y_ratio=ratio, font_size=104.0, output_height=1080
        )
        == 640.0
    )
    with pytest.raises(ReplayError):
        motion_plan_batch._anchored_y(
            "あ", height=720, anchor_y_ratio=ratio, font_size=0.0, output_height=1080
        )
    with pytest.raises(ReplayError):
        motion_plan_batch._anchored_y(
            "", height=720, anchor_y_ratio=Fraction(8, 9), font_size=104.0, output_height=1080
        )


def test_rejects_changed_inputs_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template, bundle, output = _inputs(tmp_path)
    monkeypatch.setattr(
        motion_plan_batch, "_style_controls", lambda _: ((0.1, 0.2, 0.3), (0.2, 0.3, 0.4))
    )
    monkeypatch.setattr(
        motion_stages,
        "prepare",
        lambda _template, plan: json.dumps(plan, ensure_ascii=False, sort_keys=True),
    )
    monkeypatch.setattr(motion_finish, "finish_sequence", lambda source, *, style: source)
    motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))
    template.write_text("changed template", encoding="utf-8")
    with pytest.raises(ReplayError, match="inputs or implementation changed"):
        motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))


@pytest.mark.parametrize("ratio", [Fraction(0), Fraction(1)])
def test_rejects_out_of_bounds_anchor(tmp_path: Path, ratio: Fraction) -> None:
    template, bundle, output = _inputs(tmp_path)
    with pytest.raises(ReplayError, match="vertical-anchor ratio"):
        motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=ratio)


def test_rejects_inconsistent_counts_and_duplicate_json_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, bundle, output = _inputs(tmp_path)
    monkeypatch.setattr(
        motion_plan_batch,
        "_style_controls",
        lambda _: ((0.1, 0.2, 0.3), (0.2, 0.3, 0.4)),
    )
    monkeypatch.setattr(
        motion_stages,
        "prepare",
        lambda _template, plan: json.dumps(plan, ensure_ascii=False, sort_keys=True),
    )
    monkeypatch.setattr(
        motion_finish,
        "finish_sequence",
        lambda source, *, style: source,
    )
    value = _bundle()
    value["placements"] = 99
    bundle.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReplayError, match="placement total"):
        motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))
    bundle.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="duplicate keys"):
        motion_plan_batch.expand(template, bundle, output, anchor_y_ratio=Fraction(8, 9))


def test_regular_rejects_symlinked_missing_empty_and_oversized_inputs(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(ReplayError, match="symlinks"):
        motion_plan_batch._regular(link, limit=100)
    with pytest.raises(ReplayError, match="unavailable"):
        motion_plan_batch._regular(tmp_path / "absent.json", limit=100)
    with pytest.raises(ReplayError, match="bounded regular"):
        motion_plan_batch._regular(empty, limit=100)
    with pytest.raises(ReplayError, match="bounded regular"):
        motion_plan_batch._regular(target, limit=1)
    assert motion_plan_batch._regular(target, limit=100) == target


def test_mapping_integer_and_number_validate_their_inputs() -> None:
    with pytest.raises(ReplayError, match="plan x fields"):
        motion_plan_batch._mapping({"a": 1}, name="x", keys={"b"})
    with pytest.raises(ReplayError, match="plan frames"):
        motion_plan_batch._integer(True, name="frames")
    with pytest.raises(ReplayError, match="plan frames"):
        motion_plan_batch._integer(-1, name="frames")
    with pytest.raises(ReplayError, match="plan y"):
        motion_plan_batch._number(True, name="y")
    with pytest.raises(ReplayError, match="plan y"):
        motion_plan_batch._number(float("nan"), name="y")
    assert motion_plan_batch._number(3, name="y") == 3.0


def test_main_reports_an_error_and_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "motion",
            str(tmp_path / "t"),
            str(tmp_path / "b"),
            str(tmp_path / "o"),
            "--anchor-y-ratio",
            "8/9",
        ],
    )
    assert motion_plan_batch.main() == 1
    assert capsys.readouterr().out.startswith("MOTION_BATCH_ERROR")


def test_main_prints_the_manifest_path_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(motion_plan_batch, "expand", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        "sys.argv",
        [
            "motion",
            str(tmp_path / "t"),
            str(tmp_path / "b"),
            str(tmp_path / "o"),
            "--anchor-y-ratio",
            "8/9",
        ],
    )
    assert motion_plan_batch.main() == 0
    assert capsys.readouterr().out == f"{manifest}\n"
