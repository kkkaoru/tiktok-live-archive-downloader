"""Synthetic overlap adoption; no model or real recording access."""

import json
from pathlib import Path

import pytest

from replay.asr_label_refine import _read, main, replace_block, run
from replay.caption_repair import ClockLabel
from replay.stages import file_digest


def save(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def config(tmp_path: Path) -> Path:
    source = save(tmp_path / "source.json", {"synthetic": True})
    inputs = save(
        tmp_path / "inputs.json",
        {
            "version": 1,
            "samples": 1440000,
            "sampleRate": 16000,
            "windowSamples": 960000,
            "strideSamples": 480000,
            "clock": "input-pcm-not-projected",
            "inputs": {source["path"]: source["sha256"]},
        },
    )
    raw = save(
        tmp_path / "raw.json",
        {
            "language": "ja",
            "text": "新語",
            "segments": [
                {
                    "start": 1,
                    "end": 3,
                    "text": "新語",
                    "words": [
                        {"word": "新", "start": 1, "end": 2, "probability": 0.9},
                        {"word": "語", "start": 2, "end": 3, "probability": 0.9},
                    ],
                }
            ],
        },
    )
    clock = save(
        tmp_path / "clock.json",
        {
            "startSample": 480000,
            "endSample": 1440000,
            "sampleRate": 16000,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": raw["sha256"],
        },
    )
    complete = save(
        tmp_path / "complete.json",
        {
            "windows": 2,
            "rawSHA256": ["a" * 64, raw["sha256"]],
            "deliveryComplete": False,
        },
    )
    labels = save(
        tmp_path / "labels.json",
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "labels": [
                {"text": "前", "start": 30, "end": 30.5, "provenance": "old:0"},
                {"text": "粗", "start": 29, "end": 33, "provenance": "old:1"},
                {"text": "後", "start": 34, "end": 35, "provenance": "old:2"},
            ],
            "parentSHA256": {
                item["path"]: item["sha256"] for item in (inputs, raw, clock, complete)
            },
            "humanReviewed": False,
        },
    )
    path = tmp_path / "config.json"
    save(
        path,
        {
            "version": 1,
            "artifacts": {
                "labels": labels,
                "inputs": inputs,
                "raw": raw,
                "clock": clock,
                "complete": complete,
            },
            "replacement": {
                "firstLabel": 1,
                "stopLabel": 2,
                "startSample": 496000,
                "endSample": 528000,
            },
        },
    )
    return path


def test_adoption_keeps_neighbors_and_one_offset(config: Path, tmp_path: Path) -> None:
    original = (tmp_path / "labels.json").read_bytes()
    output = tmp_path / "candidate.json"
    assert main([str(config), str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["labels"] == [
        {"text": "前", "start": 30.0, "end": 30.5, "provenance": "old:0"},
        {"text": "新", "start": 31.0, "end": 32.0, "provenance": "000000480000:0"},
        {"text": "語", "start": 32.0, "end": 33.0, "provenance": "000000480000:1"},
        {"text": "後", "start": 34.0, "end": 35.0, "provenance": "old:2"},
    ]
    assert result["refinement"]["removed"] == [
        {"text": "粗", "start": 29.0, "end": 33.0, "provenance": "old:1"},
    ]
    assert result["onsetReversals"] == 0
    assert result["refinement"]["sourceProjectionApplications"] == 0
    assert result["humanReviewed"] is False
    assert result["deliveryComplete"] is False
    assert (tmp_path / "labels.json").read_bytes() == original
    with pytest.raises(ValueError, match="new output"):
        run(config, output)


@pytest.mark.parametrize(
    ("first", "stop", "start", "end"),
    [
        (-1, 1, 0, 16000),
        (0, 0, 0, 16000),
        (0, 2, 0, 16000),
        (0, 1, -1, 16000),
        (0, 1, 0, 0),
        (0, 1, 0, 976000),
        (True, 1, 0, 16000),
    ],
)
def test_invalid_selection(first: int, stop: int, start: int, end: int) -> None:
    with pytest.raises(ValueError, match="bounded"):
        replace_block(
            (ClockLabel("旧", 0, 1, "old"),),
            (),
            first=first,
            stop=stop,
            start_sample=start,
            end_sample=end,
        )


@pytest.mark.parametrize("alternative", [(), (ClockLabel("跨", 0.5, 2, "new"),)])
def test_empty_or_split_word(alternative: tuple[ClockLabel, ...]) -> None:
    with pytest.raises(ValueError, match="empty|splits"):
        replace_block(
            (ClockLabel("旧", 1, 2, "old"),),
            alternative,
            first=0,
            stop=1,
            start_sample=16000,
            end_sample=32000,
        )


@pytest.mark.parametrize(
    ("alternative", "message"),
    [
        ((ClockLabel("新", 0.5, 2, "new"),), "left neighbor"),
        ((ClockLabel("新", 1, 3.5, "new"),), "right neighbor"),
        ((ClockLabel("新", 2, 2.5, "new1"), ClockLabel("逆", 1, 2.5, "new2")), "reverses"),
    ],
)
def test_rejects_boundary_overlap_and_reversal(
    alternative: tuple[ClockLabel, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace_block(
            (ClockLabel("前", 0, 1, "a"), ClockLabel("旧", 0, 3, "b"), ClockLabel("後", 3, 4, "c")),
            alternative,
            first=1,
            stop=2,
            start_sample=0,
            end_sample=64000,
        )


def test_points_preserved() -> None:
    result, selected = replace_block(
        (ClockLabel("旧", 1, 1, "old"),),
        (ClockLabel("点", 1, 1, "native"),),
        first=0,
        stop=1,
        start_sample=16000,
        end_sample=32000,
    )
    assert result == selected == (ClockLabel("点", 1, 1, "native"),)


def test_nested_alternative_cannot_cross_right_neighbor() -> None:
    with pytest.raises(ValueError, match="right neighbor"):
        replace_block(
            (ClockLabel("旧", 1, 3, "old"), ClockLabel("後", 4, 6, "right")),
            (ClockLabel("長", 1, 5, "a"), ClockLabel("短", 2, 3, "b")),
            first=0,
            stop=1,
            start_sample=16000,
            end_sample=96000,
        )


@pytest.mark.parametrize("old", [ClockLabel("遠", 3, 4, "old"), ClockLabel("長", 0, 90, "old")])
def test_no_unrelated_or_unbounded_removal(old: ClockLabel) -> None:
    with pytest.raises(ValueError, match="Removed labels"):
        replace_block(
            (old,),
            (ClockLabel("新", 1, 2, "new"),),
            first=0,
            stop=1,
            start_sample=16000,
            end_sample=32000,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "Unsupported"),
        ("artifacts", {}, "bindings"),
        ("replacement", {"firstLabel": "1"}, "integer"),
    ],
)
def test_bad_config(config: Path, tmp_path: Path, field: str, value: object, message: str) -> None:
    data = json.loads(config.read_text())
    data[field] = value
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        run(config, tmp_path / "out.json")


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    [
        ("complete", "windows", 0, "Incomplete"),
        ("clock", "startSample", 1, "Incomplete"),
        ("clock", "sampleRate", 8000, "bound"),
        ("labels", "parentSHA256", {}, "bound"),
    ],
)
def test_invalid_evidence(
    config: Path, tmp_path: Path, artifact: str, field: str, value: object, message: str
) -> None:
    data = json.loads(config.read_text())
    path = Path(data["artifacts"][artifact]["path"])
    document = json.loads(path.read_text())
    document[field] = value
    data["artifacts"][artifact] = save(path, document)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        run(config, tmp_path / "out.json")


def test_large_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.asr_label_refine.MAX_BYTES", 1)
    path = tmp_path / "large.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="byte budget"):
        _read(path)
