import pytest

from replay.core import ReplayError
from replay.motion_font_lines import placement_lines


def test_preserves_actual_prefix_lines_and_spaces() -> None:
    assert placement_lines(["ab", "abc\ndef", "abc\nde", " ab "]) == (
        "ab",
        "abc",
        "def",
        "de",
        " ab ",
    )


@pytest.mark.parametrize(
    "texts",
    [
        [],
        [""],
        ["a\n"],
        ["a\nb\nc\nd"],
        ["a\tb"],
        ["a\rb"],
        ["a\u2028b"],
        ["a" * 121],
        ["😀" * 61],
        [str(i) for i in range(129)],
        [str(i) + "x" * 100 for i in range(50)],
    ],
)
def test_rejects_unsupported_or_unbounded_contexts(texts: list[str]) -> None:
    with pytest.raises(ReplayError):
        placement_lines(texts)


def test_three_lines_follow_selected_limit() -> None:
    assert placement_lines(["a\nb\nc"]) == ("a", "b", "c")
