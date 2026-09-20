from xml.etree import ElementTree

import pytest

from replay.fcpxml_chapters import XMLChapter, partition_chapters


def test_small_project_needs_no_arbitrary_chapter_split() -> None:
    assert partition_chapters(90, lambda first, end: b"<fcpxml><resources/></fcpxml>") == (
        XMLChapter(0, 90, b"<fcpxml><resources/></fcpxml>", 2),
    )


def test_actual_node_count_drives_contiguous_frame_partition() -> None:
    def build(first: int, end: int) -> bytes:
        return b"<fcpxml>" + b"<gap/>" * ((end - first) * 13000) + b"</fcpxml>"

    chapters = partition_chapters(4, build)
    assert tuple(
        map(lambda chapter: (chapter.first_frame, chapter.end_frame, chapter.elements), chapters)
    ) == ((0, 2, 26001), (2, 4, 26001))


def test_actual_utf8_byte_limit_drives_partition() -> None:
    def build(first: int, end: int) -> bytes:
        return b"<fcpxml>" + b"x" * ((end - first) * 3 * 1024 * 1024) + b"</fcpxml>"

    chapters = partition_chapters(3, build)
    assert tuple(map(lambda chapter: (chapter.first_frame, chapter.end_frame), chapters)) == (
        (0, 1),
        (1, 3),
    )


@pytest.mark.parametrize("frames", [0, -1, True, 648001])
def test_invalid_timeline(frames: int) -> None:
    with pytest.raises(ValueError, match="timeline"):
        partition_chapters(frames, lambda first, end: b"<fcpxml/>")


@pytest.mark.parametrize(
    "content", [b'<!DOCTYPE fcpxml SYSTEM "remote"><fcpxml/>', b'<!ENTITY x "text"><fcpxml/>']
)
def test_no_external_declarations(content: bytes) -> None:
    with pytest.raises(ValueError, match="entities"):
        partition_chapters(30, lambda first, end: content)


def test_wrong_document_kind() -> None:
    with pytest.raises(ValueError, match="Expected FCPXML"):
        partition_chapters(30, lambda first, end: b"<other/>")


def test_malformed_generated_document_stops() -> None:
    with pytest.raises(ElementTree.ParseError):
        partition_chapters(30, lambda first, end: b"<fcpxml>")


def test_single_frame_over_budget_is_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="Single frame"):
        partition_chapters(1, lambda first, end: b"<fcpxml>" + b"<gap/>" * 50000 + b"</fcpxml>")
