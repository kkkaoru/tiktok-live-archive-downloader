import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from replay import subtitle_eligibility
from replay.local_asr import fingerprint
from replay.recordings import record


@pytest.fixture
def manifest(tmp_path: Path, monkeypatch: MonkeyPatch) -> dict[str, object]:
    monkeypatch.chdir(tmp_path)
    provenance = tmp_path / "producer.json"
    provenance.write_text('{"sourceSHA256":"source"}', encoding="utf-8")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            [
                {
                    "width": 8,
                    "height": 8,
                    "requestedTimeSeconds": time,
                    "actualTimeSeconds": time,
                    "lines": [],
                }
                for time in range(8)
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        '[{"width":8,"height":8,"requestedTimeSeconds":8,"actualTimeSeconds":8,"lines":[]}]',
        encoding="utf-8",
    )
    return {
        "version": 1,
        "kind": "source-caption-ocr-batches",
        "sourceSHA256": "source",
        "batches": [
            {
                "ocr": {"path": str(path), "sha256": fingerprint(path)},
                "ocrProvenance": {"path": str(provenance), "sha256": fingerprint(provenance)},
            }
            for path in (first, second)
        ],
    }


def test_nine_observations_keep_global_order(manifest: dict[str, object]) -> None:
    bindings: dict[str, str] = {}
    frames = subtitle_eligibility._observations(
        manifest, producer={"sourceSHA256": "source"}, bindings=bindings
    )
    assert tuple(frame.actual_time for frame in frames) == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert len(bindings) == 3


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", True),
        ("version", 2),
        ("kind", "other"),
        ("sourceSHA256", "different"),
        ("batches", []),
        ("batches", None),
        ("batches", [None]),
    ],
)
def test_invalid_manifest(manifest: dict[str, object], key: str, value: object) -> None:
    manifest[key] = value
    with pytest.raises(ValueError):
        subtitle_eligibility._observations(
            manifest, producer={"sourceSHA256": "source"}, bindings={}
        )


@pytest.mark.parametrize("times", [(7, 7), (9, 8), (6, 6)])
def test_duplicate_reverse_or_substituted_clock(
    manifest: dict[str, object], tmp_path: Path, times: tuple[int, int]
) -> None:
    path = tmp_path / "second.json"
    path.write_text(
        json.dumps(
            [
                {
                    "width": 8,
                    "height": 8,
                    "lines": [],
                    "requestedTimeSeconds": times[0],
                    "actualTimeSeconds": times[1],
                }
            ]
        ),
        encoding="utf-8",
    )
    batches = manifest["batches"]
    assert isinstance(batches, list)
    record(batches[1])["ocr"] = {"path": str(path), "sha256": fingerprint(path)}
    with pytest.raises(ValueError, match="clocks"):
        subtitle_eligibility._observations(
            manifest, producer={"sourceSHA256": "source"}, bindings={}
        )


def test_different_media_provenance(manifest: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "foreign.json"
    path.write_text('{"sourceSHA256":"foreign"}', encoding="utf-8")
    batches = manifest["batches"]
    assert isinstance(batches, list)
    record(batches[1])["ocrProvenance"] = {"path": str(path), "sha256": fingerprint(path)}
    with pytest.raises(ValueError, match="provenance"):
        subtitle_eligibility._observations(
            manifest, producer={"sourceSHA256": "source"}, bindings={}
        )


def test_missing_batch(manifest: dict[str, object], tmp_path: Path) -> None:
    (tmp_path / "second.json").unlink()
    with pytest.raises(ValueError, match="evidence"):
        subtitle_eligibility._observations(
            manifest, producer={"sourceSHA256": "source"}, bindings={}
        )


@pytest.mark.parametrize("budget", [0, 8])
def test_global_budget(manifest: dict[str, object], monkeypatch: MonkeyPatch, budget: int) -> None:
    monkeypatch.setattr(subtitle_eligibility, "MAX_FRAMES", budget)
    with pytest.raises(ValueError):
        subtitle_eligibility._observations(
            manifest, producer={"sourceSHA256": "source"}, bindings={}
        )


@pytest.mark.parametrize("count", [0, 9])
def test_each_batch_is_nonempty_and_at_most_eight(
    manifest: dict[str, object], tmp_path: Path, count: int
) -> None:
    path = tmp_path / "second.json"
    path.write_text(
        json.dumps(
            [
                {
                    "width": 8,
                    "height": 8,
                    "lines": [],
                    "requestedTimeSeconds": index + 8,
                    "actualTimeSeconds": index + 8,
                }
                for index in range(count)
            ]
        ),
        encoding="utf-8",
    )
    batches = manifest["batches"]
    assert isinstance(batches, list)
    record(batches[1])["ocr"] = {"path": str(path), "sha256": fingerprint(path)}
    with pytest.raises(ValueError):
        subtitle_eligibility._observations(
            manifest, producer={"sourceSHA256": "source"}, bindings={}
        )
