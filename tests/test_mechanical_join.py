from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from replay import delivery_render, mechanical_join, native_join
from replay.core import ReplayError

DIGEST_A = hashlib.sha256(b"alpha").hexdigest()
DIGEST_B = hashlib.sha256(b"beta").hexdigest()


def _movie(path: Path, data: bytes) -> mechanical_join.ManifestPart:
    path.write_bytes(data)
    return mechanical_join.pin_part(path, 30)


def _document(parts: object) -> dict[str, object]:
    return {"version": 1, "parts": parts}


def _row(path: str, frames: object, digest: object) -> dict[str, object]:
    return {"sourcePath": path, "durationFrames": frames, "sha256": digest}


def test_parse_manifest_returns_ordered_parts() -> None:
    parts = mechanical_join.parse_manifest(
        _document([_row("/a.mov", 30, DIGEST_A), _row("/b.mov", 60, DIGEST_B)])
    )
    assert parts == (
        mechanical_join.ManifestPart(Path("/a.mov"), 30, DIGEST_A),
        mechanical_join.ManifestPart(Path("/b.mov"), 60, DIGEST_B),
    )


def test_parse_manifest_accepts_more_than_the_workflow_array_cap() -> None:
    rows = [_row(f"/part-{index}.mov", 30, DIGEST_A) for index in range(919)]
    assert len(mechanical_join.parse_manifest(_document(rows))) == 919


def test_parse_manifest_rejects_extra_top_level_key() -> None:
    with pytest.raises(ReplayError, match="exactly version and parts"):
        mechanical_join.parse_manifest({"version": 1, "parts": [], "extra": 1})


def test_parse_manifest_rejects_non_object() -> None:
    with pytest.raises(ReplayError, match="exactly version and parts"):
        mechanical_join.parse_manifest([])


def test_parse_manifest_rejects_unknown_version() -> None:
    with pytest.raises(ReplayError, match="Unsupported join manifest version"):
        mechanical_join.parse_manifest({"version": 2, "parts": [_row("/a", 1, DIGEST_A)]})


def test_parse_manifest_rejects_boolean_version() -> None:
    with pytest.raises(ReplayError, match="Unsupported join manifest version"):
        mechanical_join.parse_manifest({"version": True, "parts": [_row("/a", 1, DIGEST_A)]})


def test_parse_manifest_rejects_empty_parts() -> None:
    with pytest.raises(ReplayError, match="outside its budget"):
        mechanical_join.parse_manifest(_document([]))


def test_parse_manifest_rejects_non_list_parts() -> None:
    with pytest.raises(ReplayError, match="outside its budget"):
        mechanical_join.parse_manifest(_document({}))


def test_parse_manifest_rejects_too_many_parts() -> None:
    rows = [_row(f"/part-{index}.mov", 1, DIGEST_A) for index in range(4097)]
    with pytest.raises(ReplayError, match="outside its budget"):
        mechanical_join.parse_manifest(_document(rows))


def test_parse_manifest_rejects_part_with_extra_key() -> None:
    row = _row("/a.mov", 1, DIGEST_A) | {"extra": 1}
    with pytest.raises(ReplayError, match="exactly path, frames and digest"):
        mechanical_join.parse_manifest(_document([row]))


def test_parse_manifest_rejects_non_object_part() -> None:
    with pytest.raises(ReplayError, match="exactly path, frames and digest"):
        mechanical_join.parse_manifest(_document(["/a.mov"]))


def test_parse_manifest_rejects_relative_path() -> None:
    with pytest.raises(ReplayError, match="must be absolute"):
        mechanical_join.parse_manifest(_document([_row("a.mov", 1, DIGEST_A)]))


def test_parse_manifest_rejects_empty_path() -> None:
    with pytest.raises(ReplayError, match="nonempty text"):
        mechanical_join.parse_manifest(_document([_row("", 1, DIGEST_A)]))


def test_parse_manifest_rejects_zero_frames() -> None:
    with pytest.raises(ReplayError, match="positive integers"):
        mechanical_join.parse_manifest(_document([_row("/a.mov", 0, DIGEST_A)]))


def test_parse_manifest_rejects_boolean_frames() -> None:
    with pytest.raises(ReplayError, match="positive integers"):
        mechanical_join.parse_manifest(_document([_row("/a.mov", True, DIGEST_A)]))


def test_parse_manifest_rejects_short_digest() -> None:
    with pytest.raises(ReplayError, match="lowercase SHA-256"):
        mechanical_join.parse_manifest(_document([_row("/a.mov", 1, "abc")]))


def test_parse_manifest_rejects_uppercase_digest() -> None:
    with pytest.raises(ReplayError, match="lowercase SHA-256"):
        mechanical_join.parse_manifest(_document([_row("/a.mov", 1, DIGEST_A.upper())]))


def test_parse_manifest_rejects_repeated_path() -> None:
    rows = [_row("/a.mov", 1, DIGEST_A), _row("/a.mov", 2, DIGEST_B)]
    with pytest.raises(ReplayError, match="repeats a part path"):
        mechanical_join.parse_manifest(_document(rows))


def test_pin_part_records_current_digest(tmp_path: Path) -> None:
    part = _movie(tmp_path / "a.mov", b"alpha")
    assert part == mechanical_join.ManifestPart(tmp_path / "a.mov", 30, DIGEST_A)


def test_pin_part_rejects_symlink(tmp_path: Path) -> None:
    _movie(tmp_path / "real.mov", b"alpha")
    (tmp_path / "link.mov").symlink_to(tmp_path / "real.mov")
    with pytest.raises(ReplayError, match="regular immutable"):
        mechanical_join.pin_part(tmp_path / "link.mov", 30)


def test_pin_part_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="regular immutable"):
        mechanical_join.pin_part(tmp_path / "missing.mov", 30)


def test_write_manifest_round_trips(tmp_path: Path) -> None:
    parts = [_movie(tmp_path / "a.mov", b"alpha"), _movie(tmp_path / "b.mov", b"beta")]
    path = mechanical_join.write_manifest(tmp_path / "manifest.json", parts)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "parts": [
            {"sourcePath": str(tmp_path / "a.mov"), "durationFrames": 30, "sha256": DIGEST_A},
            {"sourcePath": str(tmp_path / "b.mov"), "durationFrames": 30, "sha256": DIGEST_B},
        ],
    }


def test_write_manifest_never_overwrites(tmp_path: Path) -> None:
    part = _movie(tmp_path / "a.mov", b"alpha")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        mechanical_join.write_manifest(tmp_path / "manifest.json", [part])


def test_write_manifest_rejects_invalid_part_before_writing(tmp_path: Path) -> None:
    bad = mechanical_join.ManifestPart(Path("relative.mov"), 30, DIGEST_A)
    with pytest.raises(ReplayError, match="must be absolute"):
        mechanical_join.write_manifest(tmp_path / "manifest.json", [bad])
    assert not (tmp_path / "manifest.json").exists()


def test_join_manifest_verifies_then_joins_in_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _movie(tmp_path / "b.mov", b"beta")
    second = _movie(tmp_path / "a.mov", b"alpha")
    manifest = mechanical_join.write_manifest(tmp_path / "manifest.json", [first, second])
    calls: list[tuple[tuple[native_join.PartFile, ...], int, Path, Path | None]] = []

    def fake_join(
        parts: tuple[native_join.PartFile, ...], *, rate: int, output: Path, audio: Path | None
    ) -> int:
        calls.append((parts, rate, output, audio))
        return 7

    monkeypatch.setattr(native_join, "join", fake_join)
    result = mechanical_join.join_manifest(manifest, tmp_path / "out.mp4", 30)
    assert result == 7
    assert calls == [
        (
            (
                native_join.PartFile(tmp_path / "b.mov", 30),
                native_join.PartFile(tmp_path / "a.mov", 30),
            ),
            30,
            tmp_path / "out.mp4",
            None,
        )
    ]


def test_join_manifest_stops_when_a_part_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = _movie(tmp_path / "a.mov", b"alpha")
    manifest = mechanical_join.write_manifest(tmp_path / "manifest.json", [part])
    (tmp_path / "a.mov").write_bytes(b"tampered")
    monkeypatch.setattr(native_join, "join", lambda *args, **kwargs: pytest.fail("joined"))
    with pytest.raises(ReplayError, match="Pinned native part changed: a.mov"):
        mechanical_join.join_manifest(manifest, tmp_path / "out.mp4", 30)


def test_join_manifest_stops_when_a_part_became_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = _movie(tmp_path / "a.mov", b"alpha")
    manifest = mechanical_join.write_manifest(tmp_path / "manifest.json", [part])
    (tmp_path / "other.mov").write_bytes(b"alpha")
    (tmp_path / "a.mov").unlink()
    (tmp_path / "a.mov").symlink_to(tmp_path / "other.mov")
    monkeypatch.setattr(native_join, "join", lambda *args, **kwargs: pytest.fail("joined"))
    with pytest.raises(ReplayError, match="regular immutable"):
        mechanical_join.join_manifest(manifest, tmp_path / "out.mp4", 30)


def test_join_manifest_rejects_symlinked_manifest(tmp_path: Path) -> None:
    part = _movie(tmp_path / "a.mov", b"alpha")
    real = mechanical_join.write_manifest(tmp_path / "real.json", [part])
    (tmp_path / "link.json").symlink_to(real)
    with pytest.raises(ReplayError, match="must be a regular file"):
        mechanical_join.join_manifest(tmp_path / "link.json", tmp_path / "out.mp4", 30)


def test_join_manifest_rejects_oversized_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = _movie(tmp_path / "a.mov", b"alpha")
    manifest = mechanical_join.write_manifest(tmp_path / "manifest.json", [part])
    monkeypatch.setattr(mechanical_join, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(ReplayError, match="byte budget"):
        mechanical_join.join_manifest(manifest, tmp_path / "out.mp4", 30)


def test_join_stage_input_list_is_constant_for_919_parts(tmp_path: Path) -> None:
    stage = mechanical_join.join_stage(
        tmp_path / "manifest.json",
        tmp_path / "out.mp4",
        30,
        ["python", "-m", "replay.mechanical_join"],
        [tmp_path / "cuts.json", tmp_path / "chunks.json"],
    )
    assert stage == {
        "name": "join",
        "command": [
            "python",
            "-m",
            "replay.mechanical_join",
            "join-manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "out.mp4"),
            "--rate",
            "30",
        ],
        "inputs": [
            str(tmp_path / "manifest.json"),
            str(tmp_path / "cuts.json"),
            str(tmp_path / "chunks.json"),
        ],
        "outputs": [str(tmp_path / "out.mp4")],
    }


def test_main_passes_arguments_to_join_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = _movie(tmp_path / "a.mov", b"alpha")
    manifest = mechanical_join.write_manifest(tmp_path / "manifest.json", [part])
    seen: list[tuple[Path, Path | None]] = []

    def fake_join(
        parts: tuple[native_join.PartFile, ...], *, rate: int, output: Path, audio: Path | None
    ) -> int:
        seen.append((output, audio))
        return 0

    monkeypatch.setattr(native_join, "join", fake_join)
    code = mechanical_join.main(
        [
            "join-manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.mp4"),
            "--rate",
            "30",
            "--audio",
            str(tmp_path / "voice.wav"),
        ]
    )
    assert code == 0
    assert seen == [(tmp_path / "out.mp4", tmp_path / "voice.wav")]


def _config(tmp_path: Path) -> delivery_render.Config:
    work = tmp_path / "work"
    storage = tmp_path / "storage"
    work.mkdir()
    storage.mkdir()
    other = tmp_path / "other"
    return delivery_render.Config(
        path=other,
        recording="r",
        captions=other,
        work=work,
        storage=storage,
        rate=30,
        width=1080,
        height=1920,
        cuts=tmp_path / "cuts.json",
        chunks=tmp_path / "chunks.json",
        template=other,
        expected_text="",
        finish=(),
        alpha_preset=other,
        encode_preset=other,
        native=other,
        encoding={},
        audio_source=other,
        audio_clock=other,
        source=other,
        delivery_root=other,
        final_preset=other,
        calibration=other,
        viewer_root=other,
        cut_evidence=(tmp_path / "evidence.json",),
    )


def _render(tmp_path: Path, name: str, data: bytes, count: int) -> Path:
    request = tmp_path / name / "render.request.json"
    request.parent.mkdir()
    movie = tmp_path / f"{name}.mov"
    movie.write_bytes(data)
    response = {"outputPath": str(movie), "frameCount": count}
    request.with_name("render.response.json").write_text(json.dumps(response), encoding="utf-8")
    return request


def test_generate_join_writes_manifest_and_constant_size_workflow(tmp_path: Path) -> None:
    config = _config(tmp_path)
    parts = [
        (_render(tmp_path, "p1", b"alpha", 30), 30),
        (_render(tmp_path, "p2", b"beta", 60), 60),
    ]
    workflow = mechanical_join.generate_join(parts, config)
    assert workflow == config.work / "join-workflow.json"
    document = json.loads(workflow.read_text(encoding="utf-8"))
    stage = document["stages"][0]
    assert stage["inputs"] == [
        str(config.work / "join-manifest.json"),
        str(tmp_path / "cuts.json"),
        str(tmp_path / "chunks.json"),
        str(tmp_path / "evidence.json"),
    ]
    assert stage["outputs"] == [str(config.storage / "joined-video.mp4")]
    assert stage["timeout_seconds"] == 7200
    manifest = json.loads((config.work / "join-manifest.json").read_text(encoding="utf-8"))
    assert [row["durationFrames"] for row in manifest["parts"]] == [30, 60]
    assert [row["sha256"] for row in manifest["parts"]] == [DIGEST_A, DIGEST_B]


def test_generate_join_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    parts = [(_render(tmp_path, "p1", b"alpha", 29), 30)]
    with pytest.raises(ReplayError, match="frame count differs"):
        mechanical_join.generate_join(parts, config)


def test_classic_join_fits_exactly_at_the_workflow_array_cap() -> None:
    assert mechanical_join.needs_manifest(253, 0) is False


def test_manifest_join_is_needed_one_part_beyond_the_cap() -> None:
    assert mechanical_join.needs_manifest(254, 0) is True


def test_cut_evidence_counts_toward_the_cap() -> None:
    assert mechanical_join.needs_manifest(250, 4) is True
