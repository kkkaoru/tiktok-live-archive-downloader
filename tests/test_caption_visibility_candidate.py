import json
from pathlib import Path

import pytest

from replay import caption_visibility_candidate as visibility
from replay.caption_timing import CaptionPolicy, SpokenToken, make_captions
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(visibility, "ROOT", tmp_path)
    work = tmp_path / "edits" / "work"
    work.mkdir(parents=True)
    labels, cuts, projected = work / "labels.json", work / "cuts.json", work / "projected.json"
    labels.write_text('{"labels": "original unchanged"}', encoding="utf-8")
    cuts.write_text('{"cuts": "original unchanged"}', encoding="utf-8")
    projected.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "boundaryConflicts": [],
                "labels": {"path": str(labels), "sha256": file_digest(labels)},
                "cuts": {"path": str(cuts), "sha256": file_digest(cuts)},
                "tokens": [
                    {"text": "はい", "start": 0.0, "end": 0.3, "utterance": "before"},
                    {"text": "フォ", "start": 1.0, "end": 1.1, "utterance": "000000000000:0:cut-1"},
                    {"text": "ロワ", "start": 1.1, "end": 4.1, "utterance": "000000000000:1:cut-1"},
                    {
                        "text": "ーさん",
                        "start": 4.1,
                        "end": 4.2,
                        "utterance": "000000000000:2:cut-1",
                    },
                    {"text": "です", "start": 5.0, "end": 5.2, "utterance": "after"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = work / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "operation": "overlay",
                "projectedTokens": {"path": str(projected), "sha256": file_digest(projected)},
                "targets": [
                    {"utterance": "000000000000:1:cut-1", "reason": "native timing remains coarse"}
                ],
            }
        ),
        encoding="utf-8",
    )
    return config, work / "overlay.json", projected


def test_explicit_overlay_preserves_original_and_wholeword(
    candidate: tuple[Path, Path, Path],
) -> None:
    config, output, projected = candidate
    original = projected.read_bytes()
    assert visibility.run(config, output) == 2
    assert visibility.run(config, output, verify=True) == 2
    assert projected.read_bytes() == original
    actual = json.loads(output.read_text(encoding="utf-8"))
    assert actual["tokens"] == [
        {"text": "はい", "start": 0.0, "end": 0.3, "utterance": "region:0:before"},
        {"text": "です", "start": 5.0, "end": 5.2, "utterance": "region:1:after"},
    ]
    assert actual["candidateVisibility"]["regions"] == [
        {
            "first": 1,
            "stop": 4,
            "start": 1.0,
            "end": 4.2,
            "tokens": [
                {"text": "フォ", "start": 1.0, "end": 1.1, "utterance": "000000000000:0:cut-1"},
                {"text": "ロワ", "start": 1.1, "end": 4.1, "utterance": "000000000000:1:cut-1"},
                {"text": "ーさん", "start": 4.1, "end": 4.2, "utterance": "000000000000:2:cut-1"},
            ],
        }
    ]
    assert actual["candidateOnly"] is True
    assert actual["candidateVisibility"]["completeCaptionsVerified"] is False
    assert actual["candidateVisibility"]["speechAccuracyVerified"] is False


@pytest.mark.parametrize(
    "boundary",
    [
        "scene2:000000000000:1:cut-1",
        "000000000000:1:cut-2",
        "speaker2:000000000000:1:cut-1",
        "unknown-owner",
    ],
)
def test_real_boundaries_are_not_expanded(boundary: str) -> None:
    regions = visibility.lexical_regions(
        (
            SpokenToken("フォ", 0, 0.1, "000000000000:0:cut-1"),
            SpokenToken("ロワ", 0.1, 3.1, boundary),
        ),
        (boundary,),
    )
    assert regions == (visibility.VisibilityRegion(1, 2, 0.1, 3.1),)


def test_real_gap_does_not_expand_word() -> None:
    assert visibility.lexical_regions(
        (
            SpokenToken("フォ", 0, 0.1, "000000000000:0:cut-1"),
            SpokenToken("ロワ", 1, 4, "000000000000:1:cut-1"),
        ),
        ("000000000000:1:cut-1",),
    ) == (visibility.VisibilityRegion(1, 2, 1, 4),)


def test_ordinary_caption_path_keeps_precision_guard_and_never_hides() -> None:
    with pytest.raises(ValueError, match="too coarse"):
        make_captions(
            (SpokenToken("フォロワーさん", 0, 4, "a"),),
            policy=CaptionPolicy(linguistic_boundaries=True),
        )


def test_overlapping_hidden_regions_are_not_silently_merged() -> None:
    with pytest.raises(ValueError, match="overlap"):
        visibility.lexical_regions(
            (SpokenToken("はい", 0, 2, "speaker1"), SpokenToken("うん", 1, 3, "speaker2")),
            ("speaker1", "speaker2"),
        )


@pytest.mark.parametrize("targets", [(), ("missing",), ("a", "a")])
def test_explicit_unique_target_required(targets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        visibility.lexical_regions((SpokenToken("はい", 0, 1, "a"),), targets)


def test_duplicate_original_owner_rejected() -> None:
    with pytest.raises(ValueError, match="one exact"):
        visibility.lexical_regions(
            (SpokenToken("はい", 0, 1, "a"), SpokenToken("ね", 1, 2, "a")), ("a",)
        )


def test_large_observed_unit_is_not_unbounded() -> None:
    with pytest.raises(ValueError, match="bounded wholeword"):
        visibility.lexical_regions((SpokenToken("長音", 0, 61, "a"),), ("a",))


def test_all_text_cannot_be_hidden() -> None:
    with pytest.raises(ValueError, match="all recognized"):
        visibility.visible_tokens(
            (SpokenToken("はい", 0, 1, "a"),), (visibility.VisibilityRegion(0, 1, 0, 1),)
        )


def test_hold_clear_preserves_onsets_text_layout_and_outside_clock() -> None:
    assert visibility.clip_holds(
        [
            {"text": "はい", "start": 0.3, "end": 1.08, "scale": {"factor": 1}},
            {"text": "です", "start": 4.2, "end": 5.2},
        ],
        ((1.0, 4.2),),
    ) == [
        {"text": "はい", "start": 0.3, "end": 1.0, "scale": {"factor": 1}},
        {"text": "です", "start": 4.2, "end": 5.2},
    ]


def test_no_fabricated_onset_or_hidden_cue_allowed() -> None:
    with pytest.raises(ValueError, match="starts inside"):
        visibility.clip_holds([{"text": "語", "start": 1.1, "end": 2}], ((1, 3),))


@pytest.mark.parametrize(
    "cue",
    [
        {"text": " ", "start": 0, "end": 1},
        {"text": "語", "start": 1, "end": 1},
        {"text": "語", "start": -1, "end": 1},
    ],
)
def test_invalid_caption_rejected(cue: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        visibility.clip_holds([cue], ((2, 3),))


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"version": True},
        {"operation": "automatic"},
        {"operation": "clip"},
        {"targets": []},
        {"targets": [{"utterance": "missing", "reason": "explicit"}]},
        {"targets": [{"utterance": "000000000000:1:cut-1", "reason": " "}]},
    ],
)
def test_invalid_overlay_config_rejected(
    candidate: tuple[Path, Path, Path], change: dict[str, object]
) -> None:
    config, output, _ = candidate
    raw = json.loads(config.read_text(encoding="utf-8"))
    config.write_text(json.dumps({**raw, **change}), encoding="utf-8")
    with pytest.raises(ValueError):
        visibility.run(config, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "change", [{"clock": "source"}, {"boundaryConflicts": [1]}, {"candidateVisibility": {}}]
)
def test_invalid_projection_rejected(
    candidate: tuple[Path, Path, Path], change: dict[str, object]
) -> None:
    config, output, projected = candidate
    raw = json.loads(projected.read_text(encoding="utf-8"))
    projected.write_text(json.dumps({**raw, **change}), encoding="utf-8")
    manifest = json.loads(config.read_text(encoding="utf-8"))
    manifest["projectedTokens"]["sha256"] = file_digest(projected)
    config.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="conflict-free"):
        visibility.run(config, output)


def test_changed_original_pin_rejected(candidate: tuple[Path, Path, Path]) -> None:
    config, output, projected = candidate
    projected.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="hash changed"):
        visibility.run(config, output)


def test_existing_output_preserved(candidate: tuple[Path, Path, Path]) -> None:
    config, output, _ = candidate
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="absent"):
        visibility.run(config, output)
    assert output.read_text(encoding="utf-8") == "preserve"


def test_tampered_candidate_rejected(candidate: tuple[Path, Path, Path]) -> None:
    config, output, _ = candidate
    visibility.run(config, output)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        visibility.run(config, output, verify=True)


@pytest.fixture
def clipping(candidate: tuple[Path, Path, Path]) -> tuple[Path, Path, Path]:
    config, overlay, _ = candidate
    visibility.run(config, overlay)
    reflow = overlay.parent / "reflow.json"
    reflow.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "inputs": {str(overlay): file_digest(overlay)},
                "layout": {"style": 20},
                "captions": [
                    {"text": "はい", "start": 0.0, "end": 1.08},
                    {"text": "です", "start": 5.0, "end": 5.28},
                ],
            }
        ),
        encoding="utf-8",
    )
    clip_config = overlay.parent / "clip-config.json"
    clip_config.write_text(
        json.dumps(
            {
                "version": 1,
                "operation": "clip",
                "overlayConfig": {"path": str(config), "sha256": file_digest(config)},
                "overlay": {"path": str(overlay), "sha256": file_digest(overlay)},
                "reflow": {"path": str(reflow), "sha256": file_digest(reflow)},
            }
        ),
        encoding="utf-8",
    )
    return clip_config, overlay.parent / "clipped.json", reflow


def test_measured_hold_clipping_requires_exact_candidate_binding(
    clipping: tuple[Path, Path, Path],
) -> None:
    config, output, reflow = clipping
    original = reflow.read_bytes()
    assert visibility.run(config, output) == 2
    assert visibility.run(config, output, verify=True) == 2
    assert reflow.read_bytes() == original
    actual = json.loads(output.read_text(encoding="utf-8"))
    assert actual["captions"] == [
        {"text": "はい", "start": 0.0, "end": 1.0},
        {"text": "です", "start": 5.0, "end": 5.28},
    ]
    assert actual["layout"] == {"style": 20}
    assert actual["completeCaptionsVerified"] is False
    assert actual["deliveryComplete"] is False


@pytest.mark.parametrize("change", [{"version": 2}, {"candidateOnly": False}, {"inputs": {}}])
def test_wrong_reflow_identity_rejected(
    clipping: tuple[Path, Path, Path], change: dict[str, object]
) -> None:
    config, output, reflow = clipping
    raw = json.loads(reflow.read_text(encoding="utf-8"))
    reflow.write_text(json.dumps({**raw, **change}), encoding="utf-8")
    manifest = json.loads(config.read_text(encoding="utf-8"))
    manifest["reflow"]["sha256"] = file_digest(reflow)
    config.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Measured reflow"):
        visibility.run(config, output)


def test_wrong_overlay_proof_rejected(clipping: tuple[Path, Path, Path]) -> None:
    config, output, _ = clipping
    raw = json.loads(config.read_text(encoding="utf-8"))
    overlay = Path(raw["overlay"]["path"])
    overlay.write_text("{}", encoding="utf-8")
    raw["overlay"]["sha256"] = file_digest(overlay)
    config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="overlay differs"):
        visibility.run(config, output)


def test_cli_explicit_candidate(
    candidate: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, output, _ = candidate
    assert visibility.main([str(config), str(output)]) == 0
    assert "delivery unapproved" in capsys.readouterr().out
