import pytest

from replay.caption_io import mlx_native_pieces
from replay.caption_repair import ClockLabel
from replay.caption_review import refine_coarse_native_label


def test_supplied_fine_clock_inside_coarse_word_is_not_interpolation() -> None:
    result = refine_coarse_native_label(
        ClockLabel("好", 2, 6, "old"), (ClockLabel("好", 5.8, 6, "measured"),)
    )
    assert result == ClockLabel("好", 5.8, 6, "measured")


def test_already_fine_clock_is_unchanged() -> None:
    assert (
        refine_coarse_native_label(
            ClockLabel("好", 2, 2.2, "old"), (ClockLabel("好", 2, 2.1, "new"),)
        )
        is None
    )


@pytest.mark.parametrize(
    "candidate",
    [
        ClockLabel("好", 1.9, 2.1, "new"),
        ClockLabel("好", 5.9, 6.1, "new"),
        ClockLabel("好き", 5.8, 6, "new"),
        ClockLabel("好", 5, 5, "new"),
        ClockLabel("好", 2, 5, "new"),
    ],
)
def test_no_fabricated_or_changed_word(candidate: ClockLabel) -> None:
    assert refine_coarse_native_label(ClockLabel("好", 2, 6, "old"), (candidate,)) is None


def test_ambiguous_repetition_remains_unresolved() -> None:
    assert (
        refine_coarse_native_label(
            ClockLabel("好", 2, 6, "old"),
            (ClockLabel("好", 2, 2.1, "new"), ClockLabel("好", 5.8, 6, "new")),
        )
        is None
    )


def test_raw_positive_piece_keeps_its_clock_next_to_an_unresolved_point() -> None:
    pieces = mlx_native_pieces(
        {
            "language": "ja",
            "text": "っ好",
            "segments": [
                {
                    "text": "っ好",
                    "start": 0.2,
                    "end": 0.6,
                    "words": [
                        {"word": "っ", "start": 0.3, "end": 0.3, "probability": 0.8},
                        {"word": "好", "start": 0.3, "end": 0.4, "probability": 0.9},
                    ],
                }
            ],
        },
        duration=1,
    )
    assert tuple(
        map(lambda piece: (piece.text, piece.start, piece.end, piece.piece_count), pieces)
    ) == (("っ", 0.3, 0.3, 1), ("好", 0.3, 0.4, 1))


def test_raw_piece_access_still_validates_full_text() -> None:
    with pytest.raises(ValueError, match="conserve"):
        mlx_native_pieces({"language": "ja", "text": "changed", "segments": []}, duration=1)


def test_resource_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        refine_coarse_native_label(
            ClockLabel("好", 2, 6, "old"), (ClockLabel("好", 5.8, 6, "new"),) * 100001
        )
