import hashlib
import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_project
from replay.core import ReplayError
from replay.fcp_project import (
    VERTICAL_1080,
    Caption,
    CaptionChain,
    CaptionStyle,
    CaptionTemplate,
    Cut,
    Mask,
    MediaInfo,
    ProjectReport,
    Timeline,
    ass_clock,
    build_fcpxml,
    caption_chains,
    caption_events,
    caption_template,
    caption_title,
    chain_reveal_samples,
    frame_span,
    framed_masks,
    load_timeline,
    parse_ass_captions,
    probe_media,
    rational,
    replace_source,
    set_caption_reveal,
    smooth_chain_reveal_samples,
)
from replay.vertical_captions import HeadSample

ASS = (
    "[Script Info]\n"
    "Dialogue: 0,0:00:00.00,0:00:01.00,vertical,,0,0,0,,{\\an2\\pos(540,554)\\bord9}あ\n"
    "Dialogue: 2,0:00:00.00,0:00:01.00,vertical,,0,0,0,,"
    "{\\an2\\pos(540,554)\\bord0\\shad0\\1c&HFFD46C&}一行目\\N二行目\n"
    "Dialogue: 2,0:00:01.00,0:00:01.00,vertical,,0,0,0,,"
    "{\\an2\\pos(540,554)\\bord0\\1c&HFFD46C&}零\n"
    "Comment: ignored\n"
)
STYLE = CaptionStyle(66, "#6CD4FF", 6, "#FFFFFF", 3, "#000000")
TEMPLATE = CaptionTemplate(
    "Caption", "~/Titles.localized/Caption.localized/Caption.moti", "file:///t/Caption.moti"
)
TWO_TEXT_OBJECTS = (
    '<ozml version="5.13"><scene><layer>'
    '<scenenode name="Caption Face" factoryID="16"/>'
    '<scenenode name="Caption Outer Rim" factoryID="16"/>'
    "</layer></scene></ozml>"
)
INSTALLED = "Movies/Motion Templates.localized/Titles.localized/Replay.localized/Cap.localized"
TYPE_ON = '<parameter id="2"><parameter id="319"><parameter id="321"/></parameter></parameter>'
# Remapped IDs; the rim object and its target both precede the face.
REVEAL_OBJECTS = (
    '<ozml version="5.13"><scene><publishSettings><version>2</version>'
    '<target object="3" channel="./5/9/14/15" name="Caption Fill Mode"/>'
    '<target object="7" channel="./2/319/321" name="Caption Reveal Rim"/>'
    '<target object="3" channel="./2/319/321" name="Caption Reveal Face"/>'
    '</publishSettings><layer id="41">'
    f'<scenenode name="Caption Outer Rim" id="7" factoryID="16">{TYPE_ON}</scenenode>'
    f'<scenenode name="Caption Face" id="3" factoryID="16">{TYPE_ON}</scenenode>'
    "</layer></scene></ozml>"
)
REVEAL_TEMPLATE = replace(TEMPLATE, reveal_keys=("9999/41/3/2/319/321", "9999/41/7/2/319/321"))
REVEAL_CAPTIONS = (
    Caption(Fraction(0), Fraction(1, 3), 540, 554, ("a",), "#6CD4FF"),
    Caption(Fraction(1, 3), Fraction(11, 30), 540, 554, ("ab",), "#6CD4FF"),
    Caption(Fraction(11, 30), Fraction(2, 3), 540, 554, ("ab", "c"), "#6CD4FF"),
    Caption(Fraction(1, 6), Fraction(1, 2), 540, 1100, ("x",), "#6CD4FF"),
    Caption(Fraction(7, 10), Fraction(1), 540, 554, ("ab",), "#6CD4FF"),
    Caption(Fraction(1), Fraction(3, 2), 540, 554, ("abcd",), "#6CD4FF"),
    Caption(Fraction(3, 2), Fraction(2), 540, 554, ("z",), "#6CD4FF"),
)


def test_native_display_clock_does_not_add_a_frame_for_float_noise() -> None:
    assert fcp_project._seconds(0.7666666666666667) == Fraction(23, 30)
    assert fcp_project._seconds(0.7667) == Fraction(7667, 10000)
    events, hidden = caption_events(
        (
            Caption(
                fcp_project._seconds(109 / 150),
                fcp_project._seconds(23 / 30),
                0,
                0,
                ("synthetic",),
                "#005A9C",
            ),
        ),
        30,
    )
    assert hidden == 0
    assert [(event.start, event.end) for event in events] == [(Fraction(11, 15), Fraction(23, 30))]


def part(
    start_frame: int,
    frames: int,
    clips: list[dict[str, object]],
    *,
    masks: object = None,
    burn: object = None,
    width: int = 1080,
) -> dict[str, object]:
    video: dict[str, object] = {
        "width": width,
        "height": 1920,
        "frameRate": 30,
        "resizeMode": "fit",
    }
    if masks is not None:
        video["masks"] = masks
    entry: dict[str, object] = {
        "startFrame": start_frame,
        "durationFrames": frames,
        "request": {"recipe": {"clips": clips, "video": video}},
    }
    if burn is not None:
        entry["captionBurn"] = burn
    return entry


def clip(start: float, duration: float, rate: float = 1) -> dict[str, object]:
    return {
        "sourcePath": "/media/source.mp4",
        "selection": {"startSeconds": start, "durationSeconds": duration, "rate": rate},
    }


def blur(
    start: float, end: float, *, opacity: float = 1, radius: float | None = 64
) -> dict[str, object]:
    mask: dict[str, object] = {
        "startSeconds": start,
        "endSeconds": end,
        "region": {"x": 0, "y": 1320, "width": 1080, "height": 120},
        "opacity": opacity,
    }
    if radius is not None:
        mask["blurRadius"] = radius
    return mask


BURN = {
    "ass": "captions/a.ass",
    "style": {"fontSize": 66, "fill": "#6CD4FF", "whiteBorder": 6, "blackRim": 3},
}


@pytest.fixture
def render(tmp_path: Path) -> Path:
    (tmp_path / "captions").mkdir()
    (tmp_path / "captions/a.ass").write_text(ASS, encoding="utf-8")
    return tmp_path


def installed_template(root: Path, text: str = TWO_TEXT_OBJECTS) -> Path:
    directory = root / INSTALLED
    directory.mkdir(parents=True)
    path = directory / "Cap.moti"
    path.write_text(text, encoding="utf-8")
    return path


def test_ass_fill_events_become_captions_with_rgb_fill_and_offset() -> None:
    captions = parse_ass_captions(ASS, Fraction(10))
    assert captions == [
        Caption(Fraction(10), Fraction(11), 540.0, 554.0, ("一行目", "二行目"), "#6CD4FF")
    ]


def test_ass_clock_rejects_malformed_values() -> None:
    assert ass_clock("1:02:03.50") == Fraction(7447, 2)
    with pytest.raises(ReplayError, match="Invalid ASS clock"):
        ass_clock("02:03")


def test_ass_fill_events_need_position_color_and_nonblank_lines() -> None:
    with pytest.raises(ReplayError, match="lacks a position"):
        parse_ass_captions("Dialogue: 2,0:00:00.00,0:00:01.00,s,,0,0,0,,{\\bord0}x")
    with pytest.raises(ReplayError, match="blank line"):
        parse_ass_captions(
            "Dialogue: 2,0:00:00.00,0:00:01.00,s,,0,0,0,,{\\bord0\\pos(1,2)\\1c&H000000&}a\\N "
        )


def test_timeline_trims_parts_to_planned_frames_and_reads_masks_and_captions(
    render: Path,
) -> None:
    plan = [
        part(0, 45, [clip(10, 1), clip(20, 2)], masks=[blur(0.5, 2), blur(1.6, 1.8)], burn=BURN),
        part(45, 15, [clip(30, 1)]),
    ]
    timeline = load_timeline(plan, render)
    assert timeline.cuts == (
        Cut(Path("/media/source.mp4"), Fraction(10), Fraction(0), Fraction(1)),
        Cut(Path("/media/source.mp4"), Fraction(20), Fraction(1), Fraction(1, 2)),
        Cut(Path("/media/source.mp4"), Fraction(30), Fraction(3, 2), Fraction(1, 2)),
    )
    assert timeline.masks == (
        Mask(Fraction(1, 2), Fraction(3, 2), 0.0, 1320.0, 1080.0, 120.0, 64.0),
    )
    assert timeline.captions == (
        Caption(Fraction(0), Fraction(1), 540.0, 554.0, ("一行目", "二行目"), "#6CD4FF"),
    )
    assert timeline.style == STYLE
    assert timeline.duration == Fraction(2)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ([], "Render plan is empty"),
        ({}, "Render plan is empty"),
        ([part(1, 30, [clip(0, 1)])], "contiguous"),
        ([part(0, 30, [clip(0, 1)]), part(30, 30, [clip(0, 1)], width=720)], "one fit canvas"),
        ([part(0, 30, [clip(0, 1, rate=2)])], "unretimed"),
        ([part(0, 60, [clip(0, 1)])], "shorter than the planned frames"),
        ([part(0, 30, [clip(0, 1)], masks=[blur(0, 1, opacity=0.5)])], "opaque blur"),
        ([part(0, 30, [clip(0, 1)], masks=[blur(0, 1, radius=None)])], "opaque blur"),
        ([part(0, 30, [clip(0, 1)], masks={"x": 1})], "Invalid mask list"),
        (
            [part(0, 30, [clip(0, 1)], masks=[{"opacity": True, "region": {}}])],
            "Invalid mask opacity",
        ),
    ],
)
def test_timeline_rejects_unsupported_plans(plan: object, message: str, render: Path) -> None:
    with pytest.raises(ReplayError, match=message):
        load_timeline(plan, render)


def test_timeline_rejects_mixed_caption_styles(render: Path) -> None:
    other = {
        "ass": "captions/a.ass",
        "style": {"fontSize": 66, "fill": "#6CD4FF", "whiteBorder": 4, "blackRim": 3},
    }
    plan = [part(0, 30, [clip(0, 1)], burn=BURN), part(30, 30, [clip(1, 1)], burn=other)]
    with pytest.raises(ReplayError, match="one caption style"):
        load_timeline(plan, render)


def test_frame_span_keeps_exactly_the_displaying_frames() -> None:
    assert frame_span(Fraction(1, 100), Fraction(1, 10), 30) == (Fraction(1, 30), Fraction(1, 10))
    assert frame_span(Fraction(0), Fraction(97, 100), 30) == (Fraction(0), Fraction(1))
    assert frame_span(Fraction(101, 100), Fraction(102, 100), 30) is None


def test_caption_events_join_part_splits_and_count_frameless_captions() -> None:
    captions = [
        Caption(Fraction(1, 2), Fraction(1), 540, 554, ("あ",), "#6CD4FF"),
        Caption(Fraction(0), Fraction(149, 300), 540, 554, ("あ",), "#6CD4FF"),
        Caption(Fraction(101, 100), Fraction(102, 100), 540, 554, ("い",), "#6CD4FF"),
        Caption(Fraction(1), Fraction(2), 540, 900, ("あ",), "#6CD4FF"),
        Caption(Fraction(2), Fraction(5, 2), 540, 554, ("あ",), "#6CD4FF"),
    ]
    assert caption_events(captions, 30) == (
        [
            Caption(Fraction(0), Fraction(1), 540, 554, ("あ",), "#6CD4FF"),
            Caption(Fraction(1), Fraction(2), 540, 900, ("あ",), "#6CD4FF"),
            Caption(Fraction(2), Fraction(5, 2), 540, 554, ("あ",), "#6CD4FF"),
        ],
        1,
    )


def test_caption_chains_keep_every_prefix_growth_stage_and_its_clock() -> None:
    events = [
        Caption(Fraction(0), Fraction(1, 2), 540, 554, ("あ",), "#6CD4FF"),
        Caption(Fraction(1, 2), Fraction(1), 540, 554, ("あい",), "#6CD4FF"),
        Caption(Fraction(1), Fraction(3, 2), 540, 554, ("あいう、",), "#6CD4FF"),
        Caption(Fraction(3, 2), Fraction(2), 540, 554, ("あいう、", "えお"), "#6CD4FF"),
    ]
    assert caption_chains(events) == (
        CaptionChain(
            (
                Caption(Fraction(0), Fraction(1, 2), 540, 554, ("あ",), "#6CD4FF"),
                Caption(Fraction(1, 2), Fraction(1), 540, 554, ("あい",), "#6CD4FF"),
                Caption(Fraction(1), Fraction(3, 2), 540, 554, ("あいう、",), "#6CD4FF"),
                Caption(Fraction(3, 2), Fraction(2), 540, 554, ("あいう、", "えお"), "#6CD4FF"),
            )
        ),
    )


def test_caption_chains_restart_on_reset_and_rebalanced_line_breaks() -> None:
    events = [
        Caption(Fraction(0), Fraction(1, 2), 540, 554, ("ab", "cd"), "#6CD4FF"),
        Caption(Fraction(1, 2), Fraction(1), 540, 554, ("abc", "de"), "#6CD4FF"),
        Caption(Fraction(1), Fraction(3, 2), 540, 554, ("x",), "#6CD4FF"),
        Caption(Fraction(3, 2), Fraction(2), 540, 554, ("x",), "#6CD4FF"),
    ]
    assert caption_chains(events) == (
        CaptionChain((Caption(Fraction(0), Fraction(1, 2), 540, 554, ("ab", "cd"), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(1, 2), Fraction(1), 540, 554, ("abc", "de"), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(1), Fraction(3, 2), 540, 554, ("x",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(3, 2), Fraction(2), 540, 554, ("x",), "#6CD4FF"),)),
    )


def test_caption_chains_never_bridge_gaps_or_overlaps() -> None:
    events = [
        Caption(Fraction(0), Fraction(1, 2), 540, 554, ("あ",), "#6CD4FF"),
        Caption(Fraction(16, 30), Fraction(1), 540, 554, ("あい",), "#6CD4FF"),
        Caption(Fraction(29, 30), Fraction(2), 540, 554, ("あいう",), "#6CD4FF"),
    ]
    assert caption_chains(events) == (
        CaptionChain((Caption(Fraction(0), Fraction(1, 2), 540, 554, ("あ",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(8, 15), Fraction(1), 540, 554, ("あい",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(29, 30), Fraction(2), 540, 554, ("あいう",), "#6CD4FF"),)),
    )


def test_caption_chains_restart_when_position_or_fill_changes() -> None:
    events = [
        Caption(Fraction(0), Fraction(1, 2), 540, 554, ("あ",), "#6CD4FF"),
        Caption(Fraction(1, 2), Fraction(1), 540, 900, ("あい",), "#6CD4FF"),
        Caption(Fraction(1), Fraction(3, 2), 500, 900, ("あいう",), "#6CD4FF"),
        Caption(Fraction(3, 2), Fraction(2), 500, 900, ("あいうえ",), "#FFFFFF"),
    ]
    assert caption_chains(events) == (
        CaptionChain((Caption(Fraction(0), Fraction(1, 2), 540, 554, ("あ",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(1, 2), Fraction(1), 540, 900, ("あい",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(1), Fraction(3, 2), 500, 900, ("あいう",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(3, 2), Fraction(2), 500, 900, ("あいうえ",), "#FFFFFF"),)),
    )


def test_caption_chains_link_stacked_rows_separately_and_refuse_ambiguous_tails() -> None:
    events = [
        Caption(Fraction(0), Fraction(1), 540, 554, ("あ",), "#6CD4FF"),
        Caption(Fraction(0), Fraction(1), 540, 900, ("か",), "#6CD4FF"),
        Caption(Fraction(0), Fraction(1), 540, 900, ("か",), "#6CD4FF"),
        Caption(Fraction(1), Fraction(2), 540, 554, ("あい",), "#6CD4FF"),
        Caption(Fraction(1), Fraction(2), 540, 900, ("かき",), "#6CD4FF"),
    ]
    assert caption_chains(events) == (
        CaptionChain(
            (
                Caption(Fraction(0), Fraction(1), 540, 554, ("あ",), "#6CD4FF"),
                Caption(Fraction(1), Fraction(2), 540, 554, ("あい",), "#6CD4FF"),
            )
        ),
        CaptionChain((Caption(Fraction(0), Fraction(1), 540, 900, ("か",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(0), Fraction(1), 540, 900, ("か",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(1), Fraction(2), 540, 900, ("かき",), "#6CD4FF"),)),
    )


def test_caption_chains_follow_frameless_stages_and_part_split_rejoins() -> None:
    captions = [
        Caption(Fraction(0), Fraction(51, 100), 540, 554, ("あ",), "#6CD4FF"),
        Caption(Fraction(51, 100), Fraction(52, 100), 540, 554, ("あい",), "#6CD4FF"),
        Caption(Fraction(52, 100), Fraction(1), 540, 554, ("あいう",), "#6CD4FF"),
        Caption(Fraction(1), Fraction(3, 2), 540, 554, ("あいう",), "#6CD4FF"),
        Caption(Fraction(3, 2), Fraction(2), 540, 554, ("あいうえ",), "#6CD4FF"),
    ]
    events, hidden = caption_events(captions, 30)
    assert hidden == 1
    assert caption_chains(events) == (
        CaptionChain(
            (
                Caption(Fraction(0), Fraction(8, 15), 540, 554, ("あ",), "#6CD4FF"),
                Caption(Fraction(8, 15), Fraction(3, 2), 540, 554, ("あいう",), "#6CD4FF"),
                Caption(Fraction(3, 2), Fraction(2), 540, 554, ("あいうえ",), "#6CD4FF"),
            )
        ),
    )


def test_caption_chains_compare_unnormalized_code_points_and_keep_them_verbatim() -> None:
    events = [
        Caption(Fraction(0), Fraction(1, 2), 540, 554, ("\u304b",), "#6CD4FF"),
        Caption(Fraction(1, 2), Fraction(1), 540, 554, ("\u304b\u3099",), "#6CD4FF"),
        Caption(Fraction(1), Fraction(3, 2), 540, 554, ("\u304b",), "#6CD4FF"),
        Caption(Fraction(3, 2), Fraction(2), 540, 554, ("\u304c",), "#6CD4FF"),
        Caption(Fraction(2), Fraction(5, 2), 540, 554, ("\U0001f469",), "#6CD4FF"),
        Caption(Fraction(5, 2), Fraction(3), 540, 554, ("\U0001f469\u200d\U0001f4bb",), "#6CD4FF"),
    ]
    assert caption_chains(events) == (
        CaptionChain(
            (
                Caption(Fraction(0), Fraction(1, 2), 540, 554, ("\u304b",), "#6CD4FF"),
                Caption(Fraction(1, 2), Fraction(1), 540, 554, ("\u304b\u3099",), "#6CD4FF"),
            )
        ),
        CaptionChain((Caption(Fraction(1), Fraction(3, 2), 540, 554, ("\u304b",), "#6CD4FF"),)),
        CaptionChain((Caption(Fraction(3, 2), Fraction(2), 540, 554, ("\u304c",), "#6CD4FF"),)),
        CaptionChain(
            (
                Caption(Fraction(2), Fraction(5, 2), 540, 554, ("\U0001f469",), "#6CD4FF"),
                Caption(
                    Fraction(5, 2),
                    Fraction(3),
                    540,
                    554,
                    ("\U0001f469\u200d\U0001f4bb",),
                    "#6CD4FF",
                ),
            )
        ),
    )


def test_chain_reveal_keys_hold_exact_frames_and_merge_one_frame_stage_keys() -> None:
    chain = CaptionChain(
        (
            Caption(Fraction(0), Fraction(1, 3), 540, 554, ("a",), "#6CD4FF"),
            Caption(Fraction(1, 3), Fraction(11, 30), 540, 554, ("ab",), "#6CD4FF"),
            Caption(Fraction(11, 30), Fraction(2, 3), 540, 554, ("ab", "c"), "#6CD4FF"),
        )
    )
    assert chain_reveal_samples(chain, 30) == (
        (Fraction(0), Fraction(1, 8)),
        (Fraction(3, 10), Fraction(1, 8)),
        (Fraction(1, 3), Fraction(3, 8)),
        (Fraction(11, 30), Fraction(7, 8)),
    )


def test_chain_reveal_counts_grapheme_clusters_and_newlines() -> None:
    events = [
        Caption(Fraction(0), Fraction(1, 2), 540, 554, ("\U0001f469\u200d\U0001f4bb",), "#6CD4FF"),
        Caption(
            Fraction(1, 2), Fraction(1), 540, 554, ("\U0001f469\u200d\U0001f4bbe\u0301",), "#6CD4FF"
        ),
        Caption(
            Fraction(1),
            Fraction(3, 2),
            540,
            554,
            ("\U0001f469\u200d\U0001f4bbe\u0301", "\U0001f600"),
            "#6CD4FF",
        ),
    ]
    chains = caption_chains(events)
    assert len(chains) == 1
    assert chain_reveal_samples(chains[0], 30) == (
        (Fraction(0), Fraction(1, 8)),
        (Fraction(7, 15), Fraction(1, 8)),
        (Fraction(1, 2), Fraction(3, 8)),
        (Fraction(29, 30), Fraction(3, 8)),
        (Fraction(1), Fraction(7, 8)),
    )


@pytest.mark.parametrize(
    ("chain", "message"),
    [
        (CaptionChain(()), "no stages"),
        (
            CaptionChain(
                (
                    Caption(Fraction(0), Fraction(1, 2), 540, 554, ("ax",), "#6CD4FF"),
                    Caption(Fraction(1, 2), Fraction(1), 540, 554, ("abc",), "#6CD4FF"),
                )
            ),
            "not a prefix",
        ),
        (
            CaptionChain(
                (
                    Caption(Fraction(0), Fraction(1, 2), 540, 554, ("a",), "#6CD4FF"),
                    Caption(Fraction(2, 3), Fraction(1), 540, 554, ("ab",), "#6CD4FF"),
                )
            ),
            "contiguous",
        ),
        (
            CaptionChain(
                (
                    Caption(Fraction(0), Fraction(1, 2), 540, 554, ("\u304b",), "#6CD4FF"),
                    Caption(Fraction(1, 2), Fraction(1), 540, 554, ("\u304b\u3099",), "#6CD4FF"),
                )
            ),
            "splits a composed character",
        ),
    ],
)
def test_chain_reveal_refuses_invalid_stages(chain: CaptionChain, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        chain_reveal_samples(chain, 30)


def test_masks_are_snapped_to_frames_and_frameless_masks_counted() -> None:
    masks = [
        Mask(Fraction(1, 2), Fraction(51, 100), 0, 1320, 1080, 120, 64),
        Mask(Fraction(101, 100), Fraction(102, 100), 0, 1320, 1080, 120, 64),
    ]
    assert framed_masks(masks, 30) == (
        [Mask(Fraction(1, 2), Fraction(8, 15), 0, 1320, 1080, 120, 64)],
        1,
    )


def test_caption_title_is_one_plain_text_template_instance_per_text_object() -> None:
    one = caption_title(
        Caption(Fraction(0), Fraction(1), 540, 1100, ("あ",), "#6CD4FF"),
        STYLE,
        VERTICAL_1080,
        title_ref="t1",
    )
    assert ET.tostring(one, encoding="unicode") == (
        '<title ref="t1"><text>あ</text><text>あ</text>'
        '<adjust-transform position="0 -6.6406" /></title>'
    )
    two = caption_title(
        Caption(Fraction(0), Fraction(1), 540, 554, ("a", "b"), "#6CD4FF"),
        STYLE,
        VERTICAL_1080,
        title_ref="t9",
    )
    assert ET.tostring(two, encoding="unicode") == (
        '<title ref="t9"><text>a\nb</text><text>a\nb</text>'
        '<adjust-transform position="0 25.2344" /></title>'
    )


def test_caption_reveal_keys_both_text_objects_linearly_before_their_text() -> None:
    title = caption_title(
        Caption(Fraction(0), Fraction(4), 540, 1100, ("あ",), "#6CD4FF"),
        STYLE,
        VERTICAL_1080,
        title_ref="t1",
    )
    # A one-shot iterator: validation must not consume the samples twice.
    samples = iter([(Fraction(0), Fraction(1, 4)), (Fraction(119, 30), Fraction(3, 4))])
    set_caption_reveal(title, REVEAL_TEMPLATE, samples=samples)
    assert ET.tostring(title, encoding="unicode") == (
        '<title ref="t1">'
        '<param name="Caption Reveal Face" key="9999/41/3/2/319/321"><keyframeAnimation>'
        '<keyframe time="3600s" value="0.25" interp="linear" curve="linear" />'
        '<keyframe time="108119/30s" value="0.75" interp="linear" curve="linear" />'
        "</keyframeAnimation></param>"
        '<param name="Caption Reveal Rim" key="9999/41/7/2/319/321"><keyframeAnimation>'
        '<keyframe time="3600s" value="0.25" interp="linear" curve="linear" />'
        '<keyframe time="108119/30s" value="0.75" interp="linear" curve="linear" />'
        "</keyframeAnimation></param>"
        '<text>あ</text><text>あ</text><adjust-transform position="0 -6.6406" /></title>'
    )


def test_caption_reveal_values_are_exact_fixed_point_text() -> None:
    title = ET.fromstring('<title ref="t1"><text>a</text></title>')
    set_caption_reveal(
        title,
        replace(TEMPLATE, reveal_keys=("9999/1/2/2/319/321", "9999/1/3/2/319/321")),
        samples=[
            (Fraction(0), Fraction(0)),
            (Fraction(1, 30), Fraction(1, 20000)),
            (Fraction(1, 15), Fraction(1, 3)),
            (Fraction(1, 10), Fraction(1)),
        ],
    )
    face = title.find("param[@name='Caption Reveal Face']/keyframeAnimation")
    assert face is not None
    assert [(key.attrib["time"], key.attrib["value"]) for key in face] == [
        ("3600s", "0"),
        ("108001/30s", "0.00005"),
        ("54001/15s", "0.33333333333333331"),
        ("36001/10s", "1"),
    ]


def test_caption_reveal_refuses_templates_without_controls_before_mutating() -> None:
    title = ET.fromstring('<title ref="t1"><text>a</text></title>')
    with pytest.raises(ReplayError, match="no validated reveal controls"):
        set_caption_reveal(title, TEMPLATE, samples=[(Fraction(0), Fraction(1))])
    assert ET.tostring(title, encoding="unicode") == '<title ref="t1"><text>a</text></title>'


@pytest.mark.parametrize(
    "keys",
    [
        ("9999/41/3/2/319/321", "9999/41/3/2/319/321"),
        ("9999/41/3/2/319/322", "9999/41/7/2/319/321"),
        ("9999/41/x/2/319/321", "9999/41/7/2/319/321"),
        ("41/3/2/319/321", "9999/41/7/2/319/321"),
    ],
)
def test_caption_reveal_refuses_malformed_template_keys_before_mutating(
    keys: tuple[str, str],
) -> None:
    title = ET.fromstring('<title ref="t1"><text>a</text></title>')
    with pytest.raises(ReplayError, match="reveal keys are malformed"):
        set_caption_reveal(
            title, replace(TEMPLATE, reveal_keys=keys), samples=[(Fraction(0), Fraction(1))]
        )
    assert ET.tostring(title, encoding="unicode") == '<title ref="t1"><text>a</text></title>'


def test_caption_reveal_refuses_an_existing_matching_control() -> None:
    title = ET.fromstring(
        '<title ref="t1"><param name="Other" key="9999/41/7/2/319/321" value="1"/>'
        "<text>a</text></title>"
    )
    with pytest.raises(ReplayError, match="already has caption reveal controls"):
        set_caption_reveal(title, REVEAL_TEMPLATE, samples=[(Fraction(0), Fraction(1))])
    assert ET.tostring(title, encoding="unicode") == (
        '<title ref="t1"><param name="Other" key="9999/41/7/2/319/321" value="1" />'
        "<text>a</text></title>"
    )


def test_caption_reveal_refuses_a_title_without_text() -> None:
    title = ET.fromstring('<title ref="t1"><adjust-transform position="0 0"/></title>')
    with pytest.raises(ReplayError, match="no text to reveal"):
        set_caption_reveal(title, REVEAL_TEMPLATE, samples=[(Fraction(0), Fraction(1))])
    assert ET.tostring(title, encoding="unicode") == (
        '<title ref="t1"><adjust-transform position="0 0" /></title>'
    )


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([], "at least one sample"),
        ([(Fraction(-1, 30), Fraction(0))], "nonnegative and strictly increasing"),
        ([(Fraction(0), Fraction(0)), (Fraction(0), Fraction(1))], "strictly increasing"),
        ([(Fraction(1), Fraction(0)), (Fraction(1, 2), Fraction(1))], "strictly increasing"),
        ([(Fraction(0), Fraction(-1, 10))], "between 0 and 1"),
        ([(Fraction(0), Fraction(0)), (Fraction(1), Fraction(11, 10))], "between 0 and 1"),
    ],
)
def test_caption_reveal_rejects_invalid_samples_without_mutating(
    samples: list[tuple[Fraction, Fraction]], message: str
) -> None:
    title = ET.fromstring('<title ref="t1"><text>a</text><text>a</text></title>')
    with pytest.raises(ReplayError, match=message):
        set_caption_reveal(title, REVEAL_TEMPLATE, samples=samples)
    assert ET.tostring(title, encoding="unicode") == (
        '<title ref="t1"><text>a</text><text>a</text></title>'
    )


def test_project_effect_is_unchanged_by_template_reveal_keys() -> None:
    document = build_fcpxml(BASE, name="x", sources=SOURCES, template=REVEAL_TEMPLATE)[0]
    root = ET.fromstring(document)
    effect = root.find("resources/effect[@name='Caption']")
    assert effect is not None
    assert effect.attrib == {
        "id": "r3",
        "name": "Caption",
        "uid": "~/Titles.localized/Caption.localized/Caption.moti",
        "src": "file:///t/Caption.moti",
    }
    assert list(root.iter("param")) == []
    assert [ET.tostring(title, encoding="unicode") for title in root.iter("title")] == [
        '<title ref="r3" lane="10" offset="10s" name="Caption 1" start="3600s" duration="1s">'
        '<text>あ</text><text>あ</text><adjust-transform position="0 -6.6406" /></title>',
        '<title ref="r3" lane="11" offset="21/2s" name="Caption 2" start="3600s" duration="1s">'
        '<text>い</text><text>い</text><adjust-transform position="0 21.7969" /></title>',
    ]


@pytest.fixture
def three_object_document() -> str:
    return REVEAL_OBJECTS.replace(
        "</publishSettings>",
        '<target object="9" channel="./2/319/321" name="Caption Reveal Keyline"/>'
        "</publishSettings>",
    ).replace(
        "</layer>",
        f'<scenenode name="Caption Keyline Face" id="9" factoryID="16">{TYPE_ON}'
        "</scenenode></layer>",
    )


def test_three_object_project_preserves_each_text_and_position(
    tmp_path: Path, three_object_document: str
) -> None:
    template = caption_template(installed_template(tmp_path, three_object_document))
    assert template.text_objects == 3
    assert template.reveal_keys == (
        "9999/41/3/2/319/321",
        "9999/41/7/2/319/321",
        "9999/41/9/2/319/321",
    )
    root = ET.fromstring(build_fcpxml(BASE, name="three", sources=SOURCES, template=template)[0])
    assert [node.text for node in root.findall(".//title/text")] == [
        "あ",
        "あ",
        "あ",
        "い",
        "い",
        "い",
    ]
    assert root.findall(".//title/adjust-transform")[0].get("position") == "0 -6.6406"
    assert root.findall(".//title/text-style") == []
    first = root.findall(".//title")[0]
    set_caption_reveal(first, template, samples=[(Fraction(0), Fraction(1))])
    assert [node.get("key") for node in first.findall("param")] == [
        "9999/41/3/2/319/321",
        "9999/41/7/2/319/321",
        "9999/41/9/2/319/321",
    ]
    assert [node.get("time") for node in first.findall("param/keyframeAnimation/keyframe")] == [
        "3600s",
        "3600s",
        "3600s",
    ]
    assert [node.get("value") for node in first.findall("param/keyframeAnimation/keyframe")] == [
        "1",
        "1",
        "1",
    ]


def test_three_object_template_refuses_missing_keyline(
    tmp_path: Path, three_object_document: str
) -> None:
    text = three_object_document.replace(
        '<target object="9" channel="./2/319/321" name="Caption Reveal Keyline"/>', ""
    )
    with pytest.raises(ReplayError, match="face, rim and keyline"):
        caption_template(installed_template(tmp_path, text))


def test_three_object_template_refuses_aliased_keyline(
    tmp_path: Path, three_object_document: str
) -> None:
    text = three_object_document.replace('target object="9"', 'target object="3"')
    with pytest.raises(ReplayError, match="different text objects"):
        caption_template(installed_template(tmp_path, text))


def test_three_object_reveal_refuses_partial_keys_before_mutation() -> None:
    title = ET.fromstring("<title><text>x</text><text>x</text><text>x</text></title>")
    template = replace(REVEAL_TEMPLATE, text_objects=3)
    with pytest.raises(ReplayError, match="malformed"):
        set_caption_reveal(title, template, samples=[(Fraction(0), Fraction(1))])
    assert title.findall("param") == []


@pytest.mark.parametrize("count", [0, 1, 4, True])
def test_text_object_count_is_bounded(count: int) -> None:
    with pytest.raises(ReplayError, match="two or three"):
        replace(TEMPLATE, text_objects=count)
    with pytest.raises(ReplayError, match="two or three"):
        caption_title(
            REVEAL_CAPTIONS[0], STYLE, fcp_project.VERTICAL_1080, title_ref="t", text_objects=count
        )


def test_installed_template_reference_matches_final_cut_pro_form(tmp_path: Path) -> None:
    path = installed_template(tmp_path)
    template = caption_template(path)
    assert template.name == "Cap"
    assert template.reveal_keys is None
    assert template.uid == "~/Titles.localized/Replay.localized/Cap.localized/Cap.moti"
    assert template.src.startswith("file:///")
    assert template.src.endswith(
        "/Movies/Motion%20Templates.localized/Titles.localized/Replay.localized/"
        "Cap.localized/Cap.moti"
    )


def test_ordinary_templates_have_no_reveal_keys(tmp_path: Path) -> None:
    assert caption_template(installed_template(tmp_path)).reveal_keys is None
    unrelated = installed_template(
        tmp_path / "unrelated",
        "<ozml><scene><publishSettings><version>2</version>"
        '<target object="3" channel="./2/319/321" name="Caption Fill Mode"/>'
        '</publishSettings><layer id="41">'
        '<scenenode id="3" factoryID="16"/><scenenode id="7" factoryID="16"/>'
        "</layer></scene></ozml>",
    )
    assert caption_template(unrelated).reveal_keys is None


def test_reveal_keys_follow_publish_target_names_not_object_order(tmp_path: Path) -> None:
    template = caption_template(installed_template(tmp_path, REVEAL_OBJECTS))
    assert template.reveal_keys == ("9999/41/3/2/319/321", "9999/41/7/2/319/321")
    assert template.name == "Cap"


def test_reveal_keys_use_each_text_objects_own_layer(tmp_path: Path) -> None:
    split = REVEAL_OBJECTS.replace(
        '</scenenode><scenenode name="Caption Face"',
        '</scenenode></layer><layer id="52"><scenenode name="Caption Face"',
    )
    assert caption_template(installed_template(tmp_path, split)).reveal_keys == (
        "9999/52/3/2/319/321",
        "9999/41/7/2/319/321",
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            REVEAL_OBJECTS.replace(
                'object="7" channel="./2/319/321"', 'object="7" channel="./2/319/322"'
            ),
            "Unsupported caption reveal channel",
        ),
        (
            REVEAL_OBJECTS.replace('object="7" channel=', 'object="7a" channel='),
            "object ID must be numeric",
        ),
        (REVEAL_OBJECTS.replace('object="7" channel=', "channel="), "object ID must be numeric"),
        (REVEAL_OBJECTS.replace('<layer id="41">', '<layer id="L41">'), "layer ID must be numeric"),
        (REVEAL_OBJECTS.replace('<layer id="41">', "<layer>"), "layer ID must be numeric"),
        (
            REVEAL_OBJECTS.replace('object="7" channel=', 'object="8" channel='),
            "exactly one text object",
        ),
        (
            REVEAL_OBJECTS.replace('object="7" channel=', 'object="9" channel=').replace(
                "</layer>", '<scenenode id="9" factoryID="13"/></layer>'
            ),
            "exactly one text object",
        ),
        (
            REVEAL_OBJECTS.replace(
                "</layer>", '</layer><layer id="42"><scenenode id="7" factoryID="13"/></layer>'
            ),
            "exactly one text object",
        ),
        (
            REVEAL_OBJECTS.replace(
                f'id="7" factoryID="16">{TYPE_ON}',
                'id="7" factoryID="16"><parameter id="2"><parameter id="319"/></parameter>',
            ),
            "lacks the unique Type On control",
        ),
        (
            REVEAL_OBJECTS.replace(
                "</publishSettings>",
                '<target object="7" channel="./2/319/321" name="Caption Reveal Face"/>'
                "</publishSettings>",
            ),
            "exactly one face and one rim target",
        ),
        (
            REVEAL_OBJECTS.replace(
                '<target object="7" channel="./2/319/321" name="Caption Reveal Rim"/>', ""
            ),
            "exactly one face and one rim target",
        ),
        (
            REVEAL_OBJECTS.replace('object="7" channel=', 'object="3" channel='),
            "different text objects",
        ),
    ],
)
def test_reveal_keys_reject_malformed_or_partial_controls(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(ReplayError, match=message):
        caption_template(installed_template(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        REVEAL_OBJECTS.replace('<parameter id="2">', '<parameter id="2"/><parameter id="2">'),
        REVEAL_OBJECTS.replace('<parameter id="319">', '<parameter id="319"/><parameter id="319">'),
        REVEAL_OBJECTS.replace(
            '<parameter id="321"/>', '<parameter id="321"/><parameter id="321"/>'
        ),
    ],
)
def test_reveal_keys_reject_ambiguous_parameter_paths(tmp_path: Path, text: str) -> None:
    with pytest.raises(ReplayError, match="unique Type On control"):
        caption_template(installed_template(tmp_path, text))


def test_installed_template_accepts_motions_empty_declaration(tmp_path: Path) -> None:
    path = installed_template(tmp_path, "<!DOCTYPE ozxmlscene>" + TWO_TEXT_OBJECTS)
    assert caption_template(path).name == "Cap"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '<ozml><scene><layer><scenenode factoryID="16"/></layer></scene></ozml>',
            "exactly two or three text objects",
        ),
        ('<!DOCTYPE x [<!ENTITY a "b">]>' + TWO_TEXT_OBJECTS, "declarations are not allowed"),
        ("<ozml>", "Invalid Motion XML"),
        ("<motion/>", "Expected a Motion ozml document"),
    ],
)
def test_installed_template_rejects_unverified_structure(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(ReplayError, match=message):
        caption_template(installed_template(tmp_path, text))


def test_installed_template_rejects_oversized_xml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = installed_template(tmp_path)
    monkeypatch.setattr(fcp_project, "MAX_XML_CHARACTERS", 10)
    with pytest.raises(ReplayError, match="size limit"):
        caption_template(path)


def test_installed_template_rejects_paths_outside_installed_titles(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="absolute .moti"):
        caption_template(Path("Cap.moti"))
    with pytest.raises(ReplayError, match="absolute .moti"):
        caption_template(tmp_path / "Cap.motn")
    with pytest.raises(ReplayError, match="not an installed file: Cap.moti"):
        caption_template(tmp_path / "Cap.moti")
    loose = tmp_path / "Loose.moti"
    loose.write_text(TWO_TEXT_OBJECTS, encoding="utf-8")
    with pytest.raises(ReplayError, match="inside Motion Templates.localized"):
        caption_template(loose)
    generator = tmp_path / "Motion Templates.localized/Generators.localized/G.moti"
    generator.parent.mkdir(parents=True)
    generator.write_text(TWO_TEXT_OBJECTS, encoding="utf-8")
    with pytest.raises(ReplayError, match="installed title"):
        caption_template(generator)
    link = tmp_path / "Link.moti"
    link.symlink_to(installed_template(tmp_path))
    with pytest.raises(ReplayError, match="not an installed file: Link.moti"):
        caption_template(link)


def test_rationals_use_exact_seconds() -> None:
    assert rational(Fraction(0)) == "0s"
    assert rational(Fraction(3)) == "3s"
    assert rational(Fraction(1, 30)) == "1/30s"


BASE = Timeline(
    rate=30,
    width=1080,
    height=1920,
    cuts=(
        Cut(Path("/media/source.mp4"), Fraction(10), Fraction(0), Fraction(1)),
        Cut(Path("/media/source.mp4"), Fraction(20), Fraction(1), Fraction(1)),
    ),
    masks=(),
    captions=(
        Caption(Fraction(0), Fraction(1), 540, 1100, ("あ",), "#6CD4FF"),
        Caption(Fraction(1, 2), Fraction(3, 2), 540, 554, ("い",), "#6CD4FF"),
        Caption(Fraction(161, 100), Fraction(162, 100), 540, 554, ("う",), "#6CD4FF"),
    ),
    style=STYLE,
)
SOURCES = {Path("/media/source.mp4"): MediaInfo(Fraction(100), True, False)}
VOICE = (Path("/media/delivery.mp4"), MediaInfo(Fraction(2), True, True))


def test_head_placement_maps_source_cuts_and_changes_only_caption_coordinates() -> None:
    timeline = replace(
        BASE,
        captions=(
            replace(BASE.captions[0], start=Fraction(1, 4), end=Fraction(1, 2), x=300),
            replace(BASE.captions[1], start=Fraction(5, 4), end=Fraction(3, 2), x=700),
        ),
    )
    result = fcp_project.place_captions_above_heads(
        timeline, (HeadSample(10.25, 900), HeadSample(20.25, 800))
    )
    assert [(caption.x, caption.bottom) for caption in result.captions] == [(540, 851), (540, 751)]
    assert replace(result, captions=timeline.captions) == timeline
    assert [
        replace(caption, x=old.x, bottom=old.bottom)
        for caption, old in zip(result.captions, timeline.captions, strict=True)
    ] == list(timeline.captions)
    xml, report = build_fcpxml(result, name="Head placement", sources=SOURCES, template=TEMPLATE)
    titles = ET.fromstring(xml).findall(".//title/adjust-transform")
    assert len(titles) == 2
    assert all(title.get("position") != "0 0" for title in titles)
    assert report.mask_clips == 0


@pytest.mark.parametrize(
    "timeline",
    [
        replace(BASE, rate=24),
        replace(BASE, width=1920),
        replace(BASE, style=None),
        replace(BASE, style=replace(STYLE, ass_size=52)),
        replace(BASE, cuts=()),
    ],
)
def test_head_placement_refuses_uncalibrated_canvas_style_and_sources(timeline: Timeline) -> None:
    with pytest.raises(ReplayError, match="calibrated vertical style"):
        fcp_project.place_captions_above_heads(timeline, ())


@pytest.mark.parametrize(
    "cuts",
    [
        (replace(BASE.cuts[0], source_start=Fraction(1, 100)), BASE.cuts[1]),
        (BASE.cuts[0], replace(BASE.cuts[1], output_start=Fraction(2))),
    ],
)
def test_head_placement_refuses_unaligned_or_discontinuous_cuts(cuts: tuple[Cut, ...]) -> None:
    with pytest.raises(ReplayError, match="frame-aligned cuts"):
        fcp_project.place_captions_above_heads(replace(BASE, cuts=cuts), ())


def test_authorized_top_overlap_preserves_canvas_source_cuts_and_caption_text() -> None:
    timeline = replace(BASE, captions=BASE.captions[:1])
    result = fcp_project.place_captions_above_heads(
        timeline,
        (HeadSample(10.0, 24.0),),
        allow_top_overlap=True,
    )
    assert [(caption.x, caption.bottom) for caption in result.captions] == [(540, 554)]
    assert replace(result, captions=timeline.captions) == timeline
    assert replace(result.captions[0], bottom=1100) == timeline.captions[0]


def test_caption_enlargement_preserves_footage_and_native_clocks() -> None:
    timeline = replace(BASE, captions=BASE.captions[:1])
    result = fcp_project.place_captions_above_heads(
        timeline,
        (HeadSample(10.0, 24.0),),
        allow_top_overlap=True,
        caption_scale=1.3,
    )
    assert result.style is not None
    assert (result.style.ass_size, result.style.border, result.style.rim) == pytest.approx(
        (85.8, 7.8, 3.9)
    )
    assert [(caption.x, caption.bottom) for caption in result.captions] == [(540, 607)]
    assert replace(result, style=timeline.style, captions=timeline.captions) == timeline
    assert replace(result.captions[0], bottom=1100) == timeline.captions[0]
    xml, report = build_fcpxml(result, name="Larger captions", sources=SOURCES, template=TEMPLATE)
    assert ET.fromstring(xml).find(".//title/adjust-transform") is not None
    assert report.mask_clips == 0


def test_head_placement_refuses_overlapping_captions_and_missing_geometry() -> None:
    with pytest.raises(ReplayError, match="ordered captions"):
        fcp_project.place_captions_above_heads(BASE, ())
    with pytest.raises(ReplayError, match="no observed head"):
        fcp_project.place_captions_above_heads(replace(BASE, captions=BASE.captions[:1]), ())


def test_source_audio_stays_on_each_source_cut_and_never_on_blur_duplicates() -> None:
    sources = {Path("/media/source.mp4"): MediaInfo(Fraction(100), True, True, audio_channels=2)}
    document, report = build_fcpxml(
        BASE,
        name="Original audio diagnostic",
        sources=sources,
        source_audio=True,
        template=TEMPLATE,
    )
    root = ET.fromstring(document)
    clips = root.findall("library/event/project/sequence/spine/asset-clip")
    assert [c.attrib for c in clips] == [
        {
            "ref": "r2",
            "name": "Cut 1",
            "offset": "0s",
            "start": "10s",
            "duration": "1s",
            "srcEnable": "all",
            "tcFormat": "NDF",
        },
        {
            "ref": "r2",
            "name": "Cut 2",
            "offset": "1s",
            "start": "20s",
            "duration": "1s",
            "srcEnable": "all",
            "tcFormat": "NDF",
        },
    ]
    assert root.findall(".//spine/asset-clip/asset-clip") == []
    assert len(root.findall("resources/asset")) == 1
    assert report.frames == 60
    assert root.findall(".//spine/asset-clip/audio-channel-source") == []
    assert [
        node.attrib for node in root.findall(".//asset-clip[@lane='1']/audio-channel-source")
    ] == []


def test_source_channels_are_deactivated_without_muting_connected_voice() -> None:
    other = Path("/media/mono.mp4")
    timeline = replace(BASE, cuts=(BASE.cuts[0], replace(BASE.cuts[1], source=other)))
    document, report = build_fcpxml(
        timeline,
        name="Isolated voice",
        sources={
            Path("/media/source.mp4"): MediaInfo(Fraction(100), True, True, audio_channels=2),
            other: MediaInfo(Fraction(100), True, True, audio_channels=1),
        },
        voice=VOICE,
        template=TEMPLATE,
        group_captions=True,
    )
    root = ET.fromstring(document)
    assert report.frames == 60
    assert [node.attrib for node in root.findall(".//spine/asset-clip/audio-channel-source")] == [
        {"srcCh": "1, 2", "role": "dialogue.dialogue-1", "active": "0"},
        {"srcCh": "1", "role": "dialogue.dialogue-1", "active": "0"},
    ]
    assert [
        node.attrib for node in root.findall(".//asset-clip[@lane='1']/audio-channel-source")
    ] == []
    assert root.findall(".//asset-clip[@lane='-1']/audio-channel-source") == []
    assert len(root.findall(".//asset-clip[@name='Voice']")) == 2
    assert len(root.findall(".//title")) == 2
    assert [child.tag for child in root.findall(".//asset-clip[@name='Cut 1']")[0]] == [
        "ref-clip",
        "ref-clip",
        "asset-clip",
        "audio-channel-source",
    ]
    assert root.findall(".//asset-clip[@lane='1']") == []


def test_default_without_voice_deactivates_original_audio_too() -> None:
    document, _ = build_fcpxml(
        replace(BASE, masks=(), captions=(), style=None),
        name="Silent",
        sources={Path("/media/source.mp4"): MediaInfo(Fraction(100), True, True, audio_channels=2)},
    )
    root = ET.fromstring(document)
    assert [node.attrib for node in root.findall(".//audio-channel-source")] == [
        {"srcCh": "1, 2", "role": "dialogue.dialogue-1", "active": "0"},
        {"srcCh": "1, 2", "role": "dialogue.dialogue-1", "active": "0"},
    ]
    assert root.findall(".//asset-clip[@name='Voice']") == []


def test_unknown_source_channels_cannot_silently_enable_original_audio() -> None:
    with pytest.raises(ReplayError, match="requires a verified audio channel count"):
        build_fcpxml(
            BASE,
            name="Unknown channels",
            sources={Path("/media/source.mp4"): MediaInfo(Fraction(100), True, True)},
            source_audio=False,
            template=TEMPLATE,
        )


def test_explicit_original_audio_needs_no_duplicate_mask_channel_muting() -> None:
    document, _ = build_fcpxml(
        BASE,
        name="Original audio",
        sources={Path("/media/source.mp4"): MediaInfo(Fraction(100), True, True)},
        source_audio=True,
        template=TEMPLATE,
    )
    root = ET.fromstring(document)
    assert [clip.get("srcEnable") for clip in root.findall(".//spine/asset-clip")] == ["all", "all"]
    assert root.findall(".//audio-channel-source") == []
    assert root.findall(".//asset-clip[@lane='1']") == []


@pytest.mark.parametrize("channels", [True, 0, -1, 65])
def test_media_info_rejects_invalid_channel_counts(channels: int) -> None:
    with pytest.raises(ReplayError, match="Invalid audio channel count"):
        MediaInfo(Fraction(1), True, True, audio_channels=channels)


def test_media_info_rejects_channels_without_an_audio_stream() -> None:
    with pytest.raises(ReplayError, match="Invalid audio channel count"):
        MediaInfo(Fraction(1), True, False, audio_channels=2)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("with_voice", [False, True])
def test_resource_ids_share_one_namespace_and_preserve_reference_targets(
    grouped: bool, with_voice: bool
) -> None:
    second = Path("/media/other.mp4")
    timeline = replace(BASE, cuts=(BASE.cuts[0], replace(BASE.cuts[1], source=second)))
    document, _ = build_fcpxml(
        timeline,
        name="Resource identity",
        sources={**SOURCES, second: MediaInfo(Fraction(100), True, False)},
        voice=VOICE if with_voice else None,
        template=TEMPLATE,
        group_captions=grouped,
    )
    root = ET.fromstring(document)
    resources = root.findall("resources/*")
    ids = [element.attrib["id"] for element in resources]
    assert ids == [f"r{index}" for index in range(1, len(resources) + 1)]
    by_id = {element.attrib["id"]: element for element in resources}
    assert {
        by_id[value].tag
        for element in root.iter()
        for key, value in element.attrib.items()
        if key == "format"
    } == {"format"}
    assert {
        (element.tag, by_id[element.attrib["ref"]].tag)
        for element in root.iter()
        if "ref" in element.attrib
    } == (
        {("asset-clip", "asset"), ("title", "effect")}
        | ({("ref-clip", "media")} if grouped else set())
    )
    assert {by_id[t.attrib["ref"]].get("uid") for t in root.iter("title")} == {
        "~/Titles.localized/Caption.localized/Caption.moti"
    }
    assert list(root.iter("filter-video")) == []
    assert [
        by_id[c.attrib["ref"]].findall("media-rep")[0].get("src")
        for c in root.findall("library/event/project/sequence/spine/asset-clip")
    ] == ["file:///media/source.mp4", "file:///media/other.mp4"]


def test_original_audio_cannot_masquerade_as_output_clock_voice() -> None:
    with pytest.raises(ReplayError, match="cannot be mixed"):
        build_fcpxml(
            BASE, name="x", sources=SOURCES, voice=VOICE, source_audio=True, template=TEMPLATE
        )
    with pytest.raises(ReplayError, match="without an audio stream"):
        build_fcpxml(BASE, name="x", sources=SOURCES, source_audio=True, template=TEMPLATE)
    with pytest.raises(ReplayError, match="output clock"):
        build_fcpxml(
            BASE,
            name="x",
            sources=SOURCES,
            voice=(Path("/media/original.mp4"), MediaInfo(Fraction(100), True, True)),
            template=TEMPLATE,
        )


def test_project_contains_editable_cuts_template_captions_and_voice_without_blur() -> None:
    document, report = build_fcpxml(
        BASE, name="Replay & edit", sources=SOURCES, voice=VOICE, template=TEMPLATE
    )
    assert report == ProjectReport(
        cuts=2,
        masks=0,
        mask_clips=0,
        captions=3,
        titles=2,
        caption_chains=2,
        frames=60,
        hidden_masks=0,
        hidden_captions=1,
    )
    text = document.decode("utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n')
    root = ET.fromstring(document)
    assert [item.attrib for item in root.findall("resources/*")] == [
        {
            "id": "r1",
            "frameDuration": "1/30s",
            "width": "1080",
            "height": "1920",
            "colorSpace": "1-1-1 (Rec. 709)",
        },
        {
            "id": "r2",
            "name": "source",
            "start": "0s",
            "duration": "100s",
            "hasVideo": "1",
            "hasAudio": "0",
        },
        {
            "id": "r3",
            "name": "delivery",
            "start": "0s",
            "duration": "2s",
            "hasVideo": "1",
            "hasAudio": "1",
        },
        {
            "id": "r4",
            "name": "Caption",
            "uid": "~/Titles.localized/Caption.localized/Caption.moti",
            "src": "file:///t/Caption.moti",
        },
    ]
    clips = root.findall("library/event/project/sequence/spine/asset-clip")
    assert [clip.attrib["offset"] for clip in clips] == ["0s", "1s"]
    event = root.find("library/event")
    assert event is not None
    assert event.attrib == {"name": "Replay & edit"}
    assert root.findall(".//asset-clip[@lane='1']") == []
    assert root.findall(".//filter-video") == []
    assert "Gaussian" not in text
    assert [ET.tostring(title, encoding="unicode") for title in root.iter("title")] == [
        '<title ref="r4" lane="10" offset="10s" name="Caption 1" start="3600s" duration="1s">'
        '<text>あ</text><text>あ</text><adjust-transform position="0 -6.6406" /></title>',
        '<title ref="r4" lane="11" offset="21/2s" name="Caption 2" start="3600s" duration="1s">'
        '<text>い</text><text>い</text><adjust-transform position="0 21.7969" /></title>',
    ]
    assert root.find(".//text-style") is None
    voice = root.findall(".//asset-clip[@lane='-1']")
    assert [(item.attrib["offset"], item.attrib["start"]) for item in voice] == [
        ("10s", "0s"),
        ("20s", "1s"),
    ]


def test_caption_compounds_preserve_lanes_and_titles_extending_past_their_cut() -> None:
    document, report = build_fcpxml(
        BASE,
        name="Grouped",
        sources=SOURCES,
        voice=VOICE,
        template=TEMPLATE,
        group_captions=True,
    )
    root = ET.fromstring(document)
    refs = root.findall("library/event/project/sequence/spine/asset-clip/ref-clip")
    assert [item.attrib for item in refs] == [
        {
            "ref": "r5",
            "lane": "10",
            "name": "Captions captions-1-10",
            "offset": "10s",
            "start": "0s",
            "duration": "1s",
            "srcEnable": "video",
        },
        {
            "ref": "r6",
            "lane": "11",
            "name": "Captions captions-1-11",
            "offset": "10s",
            "start": "0s",
            "duration": "3/2s",
            "srcEnable": "video",
        },
    ]
    assert root.findall("library/event/project/sequence/spine/asset-clip/title") == []
    gaps = root.findall("resources/media/sequence/spine/gap")
    assert [item.attrib for item in gaps] == [
        {"name": "Caption timing", "offset": "0s", "start": "0s", "duration": "1s"},
        {"name": "Caption timing", "offset": "0s", "start": "0s", "duration": "3/2s"},
    ]
    titles = root.findall("resources/media/sequence/spine/gap/title")
    assert [
        (t.attrib["offset"], t.attrib["lane"], t.attrib["start"], t.attrib["duration"])
        for t in titles
    ] == [("0s", "1", "3600s", "1s"), ("1/2s", "1", "3600s", "1s")]
    assert [t.findtext("text") for t in titles] == ["あ", "い"]
    assert [t.tag for gap in gaps for t in gap] == ["title", "title"]
    assert root.findall(".//asset-clip[@lane='1']") == []
    assert len(root.findall(".//asset-clip[@lane='-1']")) == 2
    assert report.titles == 2
    assert report.frames == 60


def test_caption_compounds_keep_growth_stages_and_rebase_each_cut_separately() -> None:
    timeline = replace(
        BASE,
        captions=(
            Caption(Fraction(0), Fraction(1, 2), 540, 554, ("あ",), "#6CD4FF"),
            Caption(Fraction(1, 2), Fraction(1), 540, 554, ("あい",), "#6CD4FF"),
            Caption(Fraction(1), Fraction(2), 540, 554, ("う",), "#6CD4FF"),
        ),
    )
    document, report = build_fcpxml(
        timeline, name="Growth", sources=SOURCES, template=TEMPLATE, group_captions=True
    )
    root = ET.fromstring(document)
    assert [m.attrib["id"] for m in root.findall("resources/media")] == [
        "r4",
        "r5",
    ]
    refs = root.findall("library/event/project/sequence/spine/asset-clip/ref-clip")
    assert [r.attrib["offset"] for r in refs] == ["10s", "20s"]
    titles = root.findall("resources/media/sequence/spine/gap/title")
    assert [(t.attrib["offset"], t.attrib["duration"], t.findtext("text")) for t in titles] == [
        ("0s", "1/2s", "あ"),
        ("1/2s", "1/2s", "あい"),
        ("0s", "1s", "う"),
    ]
    assert report == ProjectReport(
        cuts=2,
        masks=0,
        mask_clips=0,
        captions=3,
        titles=3,
        caption_chains=2,
        frames=60,
        hidden_masks=0,
        hidden_captions=0,
    )
    assert root.findall(".//text-style") == []


def test_project_without_captions_needs_no_template() -> None:
    document, report = build_fcpxml(replace(BASE, captions=(), masks=()), name="x", sources=SOURCES)
    assert report == ProjectReport(
        cuts=2,
        masks=0,
        mask_clips=0,
        captions=0,
        titles=0,
        caption_chains=0,
        frames=60,
        hidden_masks=0,
        hidden_captions=0,
    )
    assert ET.fromstring(document).find("resources/effect[@name='Caption']") is None


def test_smooth_project_animates_growing_and_singleton_titles() -> None:
    timeline = replace(
        BASE,
        masks=(),
        captions=(
            Caption(Fraction(0), Fraction(1, 2), 540, 554, ("A",), "#6CD4FF"),
            Caption(Fraction(1, 2), Fraction(1), 540, 554, ("AB",), "#6CD4FF"),
            Caption(Fraction(1), Fraction(2), 540, 554, ("終",), "#6CD4FF"),
        ),
    )
    document, report = build_fcpxml(
        timeline,
        name="x",
        sources=SOURCES,
        template=REVEAL_TEMPLATE,
        motion_reveal=True,
        motion_smooth=True,
    )
    titles = ET.fromstring(document).findall(".//title")
    assert report.titles == 2
    assert len(titles) == 2
    assert [
        k.attrib
        for k in titles[0].findall("param[@name='Caption Reveal Face']/keyframeAnimation/keyframe")
    ] == [
        {"time": "3600s", "value": "0", "interp": "ease", "curve": "smooth"},
        {"time": "54004/15s", "value": "0.5", "interp": "ease", "curve": "smooth"},
        {"time": "7201/2s", "value": "0.5", "interp": "ease", "curve": "smooth"},
        {"time": "108023/30s", "value": "1", "interp": "ease", "curve": "smooth"},
    ]
    assert [
        k.attrib
        for k in titles[1].findall("param[@name='Caption Reveal Rim']/keyframeAnimation/keyframe")
    ] == [
        {"time": "3600s", "value": "0", "interp": "ease", "curve": "smooth"},
        {"time": "54004/15s", "value": "1", "interp": "ease", "curve": "smooth"},
    ]


def test_smooth_project_needs_reveal_mode() -> None:
    with pytest.raises(ReplayError, match="require motion_reveal"):
        build_fcpxml(BASE, name="x", sources=SOURCES, template=REVEAL_TEMPLATE, motion_smooth=True)


def test_motion_reveal_emits_one_keyed_title_per_chain_in_overlap_lanes() -> None:
    timeline = replace(BASE, masks=(), captions=REVEAL_CAPTIONS)
    document, report = build_fcpxml(
        timeline, name="x", sources=SOURCES, template=REVEAL_TEMPLATE, motion_reveal=True
    )
    assert report == ProjectReport(
        cuts=2,
        masks=0,
        mask_clips=0,
        captions=7,
        titles=4,
        caption_chains=4,
        frames=60,
        hidden_masks=0,
        hidden_captions=0,
    )
    root = ET.fromstring(document)
    assert [ET.tostring(title, encoding="unicode") for title in root.iter("title")] == [
        '<title ref="r3" lane="10" offset="10s" name="Caption 1" start="3600s" duration="2/3s">'
        '<param name="Caption Reveal Face" key="9999/41/3/2/319/321"><keyframeAnimation>'
        '<keyframe time="3600s" value="0.125" interp="linear" curve="linear" />'
        '<keyframe time="36003/10s" value="0.125" interp="linear" curve="linear" />'
        '<keyframe time="10801/3s" value="0.375" interp="linear" curve="linear" />'
        '<keyframe time="108011/30s" value="0.875" interp="linear" curve="linear" />'
        "</keyframeAnimation></param>"
        '<param name="Caption Reveal Rim" key="9999/41/7/2/319/321"><keyframeAnimation>'
        '<keyframe time="3600s" value="0.125" interp="linear" curve="linear" />'
        '<keyframe time="36003/10s" value="0.125" interp="linear" curve="linear" />'
        '<keyframe time="10801/3s" value="0.375" interp="linear" curve="linear" />'
        '<keyframe time="108011/30s" value="0.875" interp="linear" curve="linear" />'
        "</keyframeAnimation></param>"
        "<text>ab\nc</text><text>ab\nc</text>"
        '<adjust-transform position="0 25.2344" /></title>',
        '<title ref="r3" lane="11" offset="61/6s" name="Caption 2" start="3600s" duration="1/3s">'
        '<text>x</text><text>x</text><adjust-transform position="0 -6.6406" /></title>',
        '<title ref="r3" lane="10" offset="107/10s" name="Caption 3" start="3600s" '
        'duration="4/5s">'
        '<param name="Caption Reveal Face" key="9999/41/3/2/319/321"><keyframeAnimation>'
        '<keyframe time="3600s" value="0.375" interp="linear" curve="linear" />'
        '<keyframe time="54004/15s" value="0.375" interp="linear" curve="linear" />'
        '<keyframe time="36003/10s" value="0.875" interp="linear" curve="linear" />'
        "</keyframeAnimation></param>"
        '<param name="Caption Reveal Rim" key="9999/41/7/2/319/321"><keyframeAnimation>'
        '<keyframe time="3600s" value="0.375" interp="linear" curve="linear" />'
        '<keyframe time="54004/15s" value="0.375" interp="linear" curve="linear" />'
        '<keyframe time="36003/10s" value="0.875" interp="linear" curve="linear" />'
        "</keyframeAnimation></param>"
        '<text>abcd</text><text>abcd</text><adjust-transform position="0 21.7969" /></title>',
        '<title ref="r3" lane="10" offset="41/2s" name="Caption 4" start="3600s" duration="1/2s">'
        '<text>z</text><text>z</text><adjust-transform position="0 21.7969" /></title>',
    ]


def test_default_project_keeps_one_unkeyed_title_per_event_beside_chain_counts() -> None:
    timeline = replace(BASE, masks=(), captions=REVEAL_CAPTIONS)
    document, report = build_fcpxml(timeline, name="x", sources=SOURCES, template=REVEAL_TEMPLATE)
    assert report == ProjectReport(
        cuts=2,
        masks=0,
        mask_clips=0,
        captions=7,
        titles=7,
        caption_chains=4,
        frames=60,
        hidden_masks=0,
        hidden_captions=0,
    )
    root = ET.fromstring(document)
    assert [(t.attrib["duration"], t.findtext("text")) for t in root.iter("title")] == [
        ("1/3s", "a"),
        ("1/3s", "x"),
        ("1/30s", "ab"),
        ("3/10s", "ab\nc"),
        ("3/10s", "ab"),
        ("1/2s", "abcd"),
        ("1/2s", "z"),
    ]
    assert root.findall(".//param[@name='Caption Reveal Face']") == []


def test_motion_reveal_keys_travel_unchanged_into_caption_compounds() -> None:
    timeline = replace(BASE, masks=(), captions=REVEAL_CAPTIONS[4:6])
    document, report = build_fcpxml(
        timeline,
        name="x",
        sources=SOURCES,
        template=REVEAL_TEMPLATE,
        group_captions=True,
        motion_reveal=True,
    )
    assert (report.titles, report.caption_chains) == (1, 1)
    root = ET.fromstring(document)
    titles = root.findall("resources/media/sequence/spine/gap/title")
    assert [(t.attrib["offset"], t.attrib["duration"]) for t in titles] == [("7/10s", "4/5s")]
    assert [
        (key.attrib["time"], key.attrib["value"])
        for key in titles[0].findall("param[@name='Caption Reveal Rim']/keyframeAnimation/*")
    ] == [("3600s", "0.375"), ("54004/15s", "0.375"), ("36003/10s", "0.875")]


@pytest.mark.parametrize(
    "captions",
    [
        (
            Caption(Fraction(0), Fraction(1, 2), 540, 554, ("\u304b",), "#6CD4FF"),
            Caption(Fraction(1, 2), Fraction(1), 540, 554, ("\u304b\u3099",), "#6CD4FF"),
        ),
        (
            Caption(Fraction(0), Fraction(1, 2), 540, 554, ("\U0001f469",), "#6CD4FF"),
            Caption(
                Fraction(1, 2), Fraction(1), 540, 554, ("\U0001f469\u200d\U0001f4bb",), "#6CD4FF"
            ),
        ),
    ],
)
def test_motion_reveal_refuses_stages_splitting_a_final_grapheme(
    captions: tuple[Caption, ...],
) -> None:
    timeline = replace(BASE, masks=(), captions=captions)
    assert len(caption_chains(captions)) == 1
    assert (
        build_fcpxml(timeline, name="x", sources=SOURCES, template=REVEAL_TEMPLATE)[1].titles == 2
    )
    with pytest.raises(ReplayError, match="splits a composed character"):
        build_fcpxml(
            timeline, name="x", sources=SOURCES, template=REVEAL_TEMPLATE, motion_reveal=True
        )


@pytest.mark.parametrize(
    ("project", "template"),
    [(BASE, TEMPLATE), (replace(BASE, captions=()), None)],
)
def test_motion_reveal_refuses_templates_without_validated_controls(
    project: Timeline, template: CaptionTemplate | None
) -> None:
    with pytest.raises(ReplayError, match="validated reveal controls"):
        build_fcpxml(project, name="x", sources=SOURCES, template=template, motion_reveal=True)


@pytest.mark.parametrize(
    ("project", "sources", "voice", "template", "message"),
    [
        (replace(BASE, width=720), SOURCES, None, TEMPLATE, "only calibrated"),
        (BASE, {}, None, TEMPLATE, "Missing source duration"),
        (
            BASE,
            {Path("/media/source.mp4"): MediaInfo(Fraction(100), False, True)},
            None,
            TEMPLATE,
            "no video stream: source.mp4",
        ),
        (replace(BASE, style=None), SOURCES, None, TEMPLATE, "caption style"),
        (BASE, SOURCES, None, None, "installed caption template"),
        (
            replace(
                BASE,
                captions=(
                    Caption(Fraction(0), Fraction(1), 540, 1100, ("あ",), "#6CD4FF"),
                    Caption(Fraction(1), Fraction(2), 540, 1100, ("い",), "#FFFFFF"),
                ),
            ),
            SOURCES,
            None,
            TEMPLATE,
            "per-caption fill colours",
        ),
        (
            replace(BASE, masks=(Mask(Fraction(0), Fraction(1), 0, 0, 10, 10, 8),)),
            SOURCES,
            None,
            TEMPLATE,
            "Source-caption blur masks were removed; use verified local repair",
        ),
        (
            BASE,
            SOURCES,
            (Path("/media/vocals.wav"), MediaInfo(Fraction(3), False, True)),
            TEMPLATE,
            "output clock",
        ),
        (
            BASE,
            SOURCES,
            (Path("/media/silent.mp4"), MediaInfo(Fraction(2), True, False)),
            TEMPLATE,
            "no audio stream: silent.mp4",
        ),
    ],
)
def test_project_rejects_uncalibrated_or_incomplete_inputs(
    project: Timeline,
    sources: dict[Path, MediaInfo],
    voice: tuple[Path, MediaInfo] | None,
    template: CaptionTemplate | None,
    message: str,
) -> None:
    with pytest.raises(ReplayError, match=message):
        build_fcpxml(project, name="x", sources=sources, voice=voice, template=template)


def test_voice_within_one_frame_of_the_output_clock_is_accepted() -> None:
    near = (Path("/media/delivery.mp4"), MediaInfo(Fraction(61, 30), False, True))
    document = build_fcpxml(BASE, name="x", sources=SOURCES, voice=near, template=TEMPLATE)[0]
    voice = ET.fromstring(document).find("resources/asset[@id='r3']")
    assert voice is not None
    assert voice.attrib["duration"] == "61/30s"


@pytest.mark.parametrize(
    "offsets,with_voice",
    [
        ((), True),
        ((Fraction(0),), False),
        ((Fraction(-1), Fraction(0)), True),
        ((Fraction(0), Fraction(2)), True),
    ],
)
def test_voice_selection_requires_bounded_per_cut_clock(
    offsets: tuple[Fraction, ...],
    with_voice: bool,
) -> None:
    with pytest.raises(ReplayError, match="Voice selection"):
        build_fcpxml(
            BASE,
            name="x",
            sources=SOURCES,
            template=TEMPLATE,
            voice=VOICE if with_voice else None,
            voice_offsets=offsets,
        )


def test_replace_source_points_every_cut_at_an_explicit_copy(tmp_path: Path) -> None:
    original = tmp_path / "original.mp4"
    original.write_bytes(b"v")
    replaced = replace_source(BASE, original)
    assert replaced.cuts == (
        Cut(original, Fraction(10), Fraction(0), Fraction(1)),
        Cut(original, Fraction(20), Fraction(1), Fraction(1)),
    )


def test_replace_source_refuses_ambiguous_plans_and_non_files(tmp_path: Path) -> None:
    original = tmp_path / "original.mp4"
    original.write_bytes(b"v")
    mixed = replace(
        BASE,
        cuts=(
            Cut(Path("/media/a.mp4"), Fraction(0), Fraction(0), Fraction(1)),
            Cut(Path("/media/b.mp4"), Fraction(0), Fraction(1), Fraction(1)),
        ),
    )
    with pytest.raises(ReplayError, match="single-source plan"):
        replace_source(mixed, original)
    with pytest.raises(ReplayError, match="regular file: missing.mp4"):
        replace_source(BASE, tmp_path / "missing.mp4")
    link = tmp_path / "link.mp4"
    link.symlink_to(original)
    with pytest.raises(ReplayError, match="regular file: link.mp4"):
        replace_source(BASE, link)


def one_second_media(path: Path) -> MediaInfo:
    has_audio = path.suffix == ".mp4"
    return MediaInfo(Fraction(1), True, has_audio, audio_channels=2 if has_audio else None)


@pytest.mark.parametrize("lines", [("あ",), ("あ", "い")])
def test_measured_origin_does_not_use_ass_font_baseline(lines: tuple[str, ...]) -> None:
    caption = Caption(Fraction(0), Fraction(1), 540, 999, lines, "#FFFFFF", motion_origin_y=500)
    title = caption_title(
        caption, replace(STYLE, ass_size=120.12), VERTICAL_1080, title_ref="t", text_objects=3
    )
    transform = title.find("adjust-transform")
    assert transform is not None
    assert transform.get("position") == "0 23.9583"
    assert len(title.findall("text")) == 3


@pytest.mark.parametrize("origin", [float("nan"), -1, 1921, True])
def test_measured_origin_refuses_invalid_y(origin: float) -> None:
    caption = replace(REVEAL_CAPTIONS[0], motion_origin_y=origin)
    with pytest.raises(ReplayError):
        caption_title(caption, STYLE, VERTICAL_1080, title_ref="t")


def test_measured_and_ass_coordinates_do_not_merge() -> None:
    first = Caption(Fraction(0), Fraction(1), 540, 500, ("a",), "#FFFFFF", motion_origin_y=500)
    legacy = replace(first, start=Fraction(1), end=Fraction(2), motion_origin_y=None)
    assert len(caption_events((first, legacy), 30)[0]) == 2
    assert len(caption_chains((first, replace(legacy, lines=("ab",))))) == 2
    same = replace(legacy, motion_origin_y=500)
    events, hidden = caption_events((first, same), 30)
    assert len(events) == 1
    assert events[0].end == Fraction(2)
    assert events[0].motion_origin_y == 500
    assert hidden == 0
    assert len(caption_chains((first, replace(same, lines=("ab",))))) == 1


def test_legacy_head_placement_refuses_measured_origin() -> None:
    timeline = replace(BASE, captions=(replace(BASE.captions[0], motion_origin_y=500),))
    with pytest.raises(ReplayError, match="cannot use ASS"):
        fcp_project.place_captions_above_heads(timeline, ())


@pytest.mark.parametrize("burn", [BURN, None, {"style": BURN["style"]}])
def test_cli_connects_pinned_measured_origins(render: Path, burn: dict[str, object] | None) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)], burn=burn)]), encoding="utf-8")
    candidate = render / "measured.json"
    candidate.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "canvas": [1080, 1920],
                "coordinateSystem": "top-left canvas; calibrated text origin",
                "policy": {"frame_rate": 30},
                "candidates": [
                    {"cue": {"text": "新しい\n字幕", "start": 0.1, "end": 0.9}, "x": 540, "y": 500}
                ],
            }
        ),
        encoding="utf-8",
    )
    output = render / "measured.fcpxml"
    args = [
        str(plan),
        str(output),
        "--name",
        "Measured",
        "--caption-template",
        str(installed_template(render)),
        "--measured-captions",
        str(candidate),
        "--measured-captions-sha256",
        hashlib.sha256(candidate.read_bytes()).hexdigest(),
    ]
    if burn is None:
        with pytest.raises(ReplayError, match="template caption style"):
            fcp_project.main(args, probe=one_second_media)
        assert not output.exists()
        return
    assert fcp_project.main(args, probe=one_second_media) == 0
    title = ET.parse(output).getroot().find(".//title")
    assert title is not None
    assert [node.text for node in title.findall("text")] == ["新しい\n字幕", "新しい\n字幕"]
    transform = title.find("adjust-transform")
    assert transform is not None
    assert transform.get("position") == "0 23.9583"
    assert title.get("offset") == "1/10s"
    assert title.get("duration") == "4/5s"


@pytest.mark.parametrize("option", ["--measured-captions", "--measured-captions-sha256"])
def test_cli_requires_both_measured_arguments(render: Path, option: str) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)], burn=BURN)]), encoding="utf-8")
    with pytest.raises(ReplayError, match="both a path and SHA-256"):
        fcp_project.main([str(plan), str(render / "new.fcpxml"), "--name", "x", option, "missing"])


def test_cli_writes_a_new_project_and_reports_counts(
    render: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)], burn=BURN)]), encoding="utf-8")
    template = installed_template(render)
    output = render / "project.fcpxml"
    code = fcp_project.main(
        [
            str(plan),
            str(output),
            "--name",
            "Edit",
            "--render-dir",
            str(render),
            "--caption-template",
            str(template),
            "--group-captions",
        ],
        probe=one_second_media,
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "cuts": 1,
        "masks": 0,
        "mask_clips": 0,
        "captions": 1,
        "titles": 1,
        "caption_chains": 1,
        "frames": 30,
        "hidden_masks": 0,
        "hidden_captions": 0,
    }
    effect = ET.parse(output).getroot().find("resources/effect[@name='Cap']")
    assert effect is not None
    assert effect.attrib["uid"] == "~/Titles.localized/Replay.localized/Cap.localized/Cap.moti"
    assert len(ET.parse(output).getroot().findall("resources/media")) == 1
    with pytest.raises(ReplayError, match="Refusing to write"):
        fcp_project.main([str(plan), str(output), "--name", "Edit"], probe=one_second_media)


def test_cli_refuses_captions_without_an_installed_template(render: Path) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)], burn=BURN)]), encoding="utf-8")
    output = render / "project.fcpxml"
    with pytest.raises(ReplayError, match="installed caption template"):
        fcp_project.main([str(plan), str(output), "--name", "Edit"], probe=one_second_media)
    assert not output.exists()


def test_cli_motion_reveal_reports_one_title_per_chain(
    render: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (render / "captions/a.ass").write_text(
        "Dialogue: 2,0:00:00.00,0:00:00.50,vertical,,0,0,0,,"
        "{\\an2\\pos(540,554)\\bord0\\1c&HFFD46C&}あ\n"
        "Dialogue: 2,0:00:00.50,0:00:01.00,vertical,,0,0,0,,"
        "{\\an2\\pos(540,554)\\bord0\\1c&HFFD46C&}あい\n",
        encoding="utf-8",
    )
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)], burn=BURN)]), encoding="utf-8")
    output = render / "reveal.fcpxml"
    code = fcp_project.main(
        [
            str(plan),
            str(output),
            "--name",
            "Edit",
            "--caption-template",
            str(installed_template(render, REVEAL_OBJECTS)),
            "--motion-reveal",
        ],
        probe=one_second_media,
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "cuts": 1,
        "masks": 0,
        "mask_clips": 0,
        "captions": 2,
        "titles": 1,
        "caption_chains": 1,
        "frames": 30,
        "hidden_masks": 0,
        "hidden_captions": 0,
    }
    face = ET.parse(output).getroot().find(".//title/param[@name='Caption Reveal Face']")
    assert face is not None
    assert [(key.attrib["time"], key.attrib["value"]) for key in face.iter("keyframe")] == [
        ("3600s", "0.25"),
        ("54007/15s", "0.25"),
        ("7201/2s", "0.75"),
    ]


def test_cli_source_audio_explicitly_preserves_original_audio(render: Path) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)])]), encoding="utf-8")
    output = render / "original-audio.fcpxml"
    assert (
        fcp_project.main(
            [str(plan), str(output), "--name", "Diagnostic", "--source-audio"],
            probe=one_second_media,
        )
        == 0
    )
    spine_clip = ET.parse(output).getroot().find("library/event/project/sequence/spine/asset-clip")
    assert spine_clip is not None
    assert spine_clip.get("srcEnable") == "all"


def test_cli_smooth_forwards_the_mode_without_implicitly_enabling_reveal(render: Path) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)], burn=BURN)]), encoding="utf-8")
    output = render / "smooth.fcpxml"
    with pytest.raises(ReplayError, match="require motion_reveal"):
        fcp_project.main(
            [str(plan), str(output), "--name", "Edit", "--motion-smooth"], probe=one_second_media
        )
    assert not output.exists()


def test_cli_motion_reveal_refuses_an_ordinary_template_without_writing(render: Path) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)], burn=BURN)]), encoding="utf-8")
    output = render / "project.fcpxml"
    arguments = [str(plan), str(output), "--name", "Edit", "--motion-reveal"]
    with pytest.raises(ReplayError, match="validated reveal controls"):
        fcp_project.main(
            [*arguments, "--caption-template", str(installed_template(render))],
            probe=one_second_media,
        )
    assert not output.exists()


def test_cli_adds_voice_audio_and_replaces_the_source(
    render: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = render / "render-plan.json"
    plan.write_text(json.dumps([part(0, 30, [clip(0, 1)])]), encoding="utf-8")
    original = render / "original.mov"
    original.write_bytes(b"v")
    output = render / "voice.fcpxml"
    fcp_project.main(
        [
            str(plan),
            str(output),
            "--name",
            "Edit",
            "--audio",
            str(render / "delivery.mp4"),
            "--source-video",
            str(original),
        ],
        probe=one_second_media,
    )
    capsys.readouterr()
    root = ET.parse(output).getroot()
    voice = root.find("resources/asset[@id='r3']")
    assert voice is not None
    assert voice.attrib["hasAudio"] == "1"
    source = root.find("resources/asset[@id='r2']")
    assert source is not None
    assert source.attrib["name"] == "original"
    assert source.attrib["hasAudio"] == "0"


def test_sparse_smooth_reveal_holds_and_shortens_only_short_ramps() -> None:
    chain = CaptionChain(REVEAL_CAPTIONS[:3])
    assert smooth_chain_reveal_samples(chain, 30) == (
        (Fraction(0), Fraction(0)),
        (Fraction(4, 15), Fraction(1, 4)),
        (Fraction(1, 3), Fraction(1, 4)),
        (Fraction(11, 30), Fraction(1, 2)),
        (Fraction(19, 30), Fraction(1)),
    )


def test_smooth_singleton_is_not_a_pop_in() -> None:
    assert smooth_chain_reveal_samples(CaptionChain((REVEAL_CAPTIONS[6],)), 30) == (
        (Fraction(0), Fraction(0)),
        (Fraction(4, 15), Fraction(1)),
    )


@pytest.mark.parametrize("rate,ramp", [(0, 8), (30, 0), (30, 31), (True, 1), (30, True)])
def test_smooth_invalid_policy(rate: int, ramp: int) -> None:
    with pytest.raises(ReplayError, match="Invalid smooth"):
        smooth_chain_reveal_samples(CaptionChain(REVEAL_CAPTIONS[:3]), rate, ramp_frames=ramp)


def test_smooth_final_prefix_reaches_full_on_a_visible_frame() -> None:
    stage = replace(REVEAL_CAPTIONS[0], end=Fraction(2, 30))
    assert smooth_chain_reveal_samples(CaptionChain((stage,)), 30) == (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 30), Fraction(1)),
    )
    with pytest.raises(ReplayError, match="two visible frames"):
        smooth_chain_reveal_samples(CaptionChain((replace(stage, end=Fraction(1, 30)),)), 30)


def test_smooth_empty_chain() -> None:
    with pytest.raises(ReplayError, match="empty chain"):
        smooth_chain_reveal_samples(CaptionChain(()), 30)


def test_smooth_rejects_gap() -> None:
    with pytest.raises(ReplayError, match="contiguous"):
        smooth_chain_reveal_samples(CaptionChain((REVEAL_CAPTIONS[0], REVEAL_CAPTIONS[4])), 30)


def test_smooth_rejects_subframe_stage() -> None:
    stage = replace(REVEAL_CAPTIONS[0], end=Fraction(1, 31))
    with pytest.raises(ReplayError, match="frame grid"):
        smooth_chain_reveal_samples(CaptionChain((stage,)), 30)


def test_smooth_rejects_non_growing_prefix() -> None:
    first = REVEAL_CAPTIONS[0]
    second = replace(first, start=first.end, end=Fraction(1))
    with pytest.raises(ReplayError, match="strictly grow"):
        smooth_chain_reveal_samples(CaptionChain((first, second)), 30)


def test_smooth_rejects_split_grapheme() -> None:
    first = replace(REVEAL_CAPTIONS[0], lines=("か",))
    second = replace(REVEAL_CAPTIONS[1], lines=("か\u3099",))
    with pytest.raises(ReplayError, match="composed character"):
        smooth_chain_reveal_samples(CaptionChain((first, second)), 30)


def test_fcp_smooth_keys_are_explicit_and_both_objects_match() -> None:
    title = ET.fromstring("<title><text>ab</text><text>ab</text></title>")
    set_caption_reveal(
        title,
        REVEAL_TEMPLATE,
        samples=((Fraction(0), Fraction(0)), (Fraction(4, 15), Fraction(1))),
        smooth=True,
    )
    assert [point.attrib for point in title.findall("param/keyframeAnimation/keyframe")] == [
        {"time": "3600s", "value": "0", "interp": "ease", "curve": "smooth"},
        {"time": "54004/15s", "value": "1", "interp": "ease", "curve": "smooth"},
        {"time": "3600s", "value": "0", "interp": "ease", "curve": "smooth"},
        {"time": "54004/15s", "value": "1", "interp": "ease", "curve": "smooth"},
    ]


@pytest.mark.parametrize("smooth", [1])
def test_fcp_bad_smoothing_does_not_mutate_title(smooth: bool) -> None:
    title = ET.fromstring("<title><text>ab</text></title>")
    with pytest.raises(ReplayError, match="boolean"):
        set_caption_reveal(title, REVEAL_TEMPLATE, samples=(), smooth=smooth)
    assert title.findall("param") == []


def test_probe_reads_ffprobe_duration_and_stream_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        assert options["timeout"] == 60
        assert command[-1] == "/media/a.mp4"
        assert command[4] == "format=duration:stream=codec_type,channels,duration_ts,time_base"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"streams":[{"codec_type":"video"},{"codec_type":"audio","channels":2}],'
            '"format":{"duration":"12.5"}}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert probe_media(Path("/media/a.mp4")) == MediaInfo(
        Fraction(25, 2), True, True, audio_channels=2
    )


def test_probe_accepts_a_video_without_audio_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"streams":[{"codec_type":"video"}],"format":{"duration":"1"}}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert probe_media(Path("/media/a.mp4")) == MediaInfo(Fraction(1), True, False)


@pytest.mark.parametrize(
    "clock,expected",
    [
        ({"duration_ts": 1471, "time_base": "1/44100"}, Fraction(1471, 44100)),
        ({}, Fraction(932, 27941)),
    ],
)
def test_audio_probe_preserves_sample_ticks_or_legacy_container(
    monkeypatch: pytest.MonkeyPatch,
    clock: dict[str, object],
    expected: Fraction,
) -> None:
    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "audio", "channels": 2, **clock}],
                    "format": {"duration": "0.033356"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert probe_media(Path("/media/voice.wav")) == MediaInfo(
        expected,
        False,
        True,
        audio_channels=2,
    )


@pytest.mark.parametrize(
    "clock",
    [
        {"duration_ts": True, "time_base": "1/44100"},
        {"duration_ts": 0, "time_base": "1/44100"},
        {"duration_ts": 1471},
        {"duration_ts": 1471, "time_base": "1/0"},
        {"duration_ts": 1471, "time_base": "-1/44100"},
        {"duration_ts": 1471, "time_base": "N/A"},
    ],
)
def test_audio_probe_refuses_invalid_explicit_clock(
    monkeypatch: pytest.MonkeyPatch,
    clock: dict[str, object],
) -> None:
    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "audio", "channels": 2, **clock}],
                    "format": {"duration": "1"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ReplayError, match="Cannot read media duration"):
        probe_media(Path("/media/voice.wav"))


@pytest.mark.parametrize(
    "streams",
    [
        '[{"codec_type":"audio"}]',
        '[{"codec_type":"audio","channels":null}]',
        '[{"codec_type":"audio","channels":true}]',
        '[{"codec_type":"audio","channels":"2"}]',
        '[{"codec_type":"audio","channels":1.5}]',
        '[{"codec_type":"audio","channels":0}]',
        '[{"codec_type":"audio","channels":-1}]',
        '[{"codec_type":"audio","channels":65}]',
        '[{"codec_type":"audio","channels":1},{"codec_type":"audio","channels":2}]',
    ],
)
def test_probe_refuses_missing_or_ambiguous_audio_channels(
    monkeypatch: pytest.MonkeyPatch, streams: str
) -> None:
    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"streams":' + streams + ',"format":{"duration":"1"}}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ReplayError, match="Cannot read media duration"):
        probe_media(Path("/media/a.mp4"))


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        '{"streams":[],"format":{}}',
        '{"streams":[],"format":{"duration":"N/A"}}',
        '{"streams":{},"format":{"duration":"1"}}',
        "[]",
    ],
)
def test_probe_rejects_unreadable_media(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        assert options["check"] is False
        return subprocess.CompletedProcess(command, 1, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ReplayError, match="Cannot read media duration: a.mp4"):
        probe_media(Path("/media/a.mp4"))
