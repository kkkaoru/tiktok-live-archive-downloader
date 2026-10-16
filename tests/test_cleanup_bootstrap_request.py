"""The bootstrap request carries recording-specific pins and the reference's policy only."""

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

from replay import cleanup_bootstrap_request as bootstrap
from replay.core import ReplayError

POLICY = {
    "decoder": {"d": 1},
    "model": {"m": 2},
    "profileSamples": 40,
    "nativeTimescale": 600,
    "chunkFrames": 900,
    "workers": 2,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_ffprobe(tmp_path: Path) -> Path:
    path = tmp_path / "ffprobe"
    path.write_text(
        f"#!{sys.executable}\nimport json\n"
        'print(json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "1"}}))\n',
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def setup(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "media.mp4"
    source.write_bytes(b"media")
    (work / "preparation.json").write_text(
        json.dumps({"inputs": [{"path": str(source), "sha256": "feed"}]}), encoding="utf-8"
    )
    (work / "video-clock.json").write_text("clock", encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({**POLICY, "source": "other"}), encoding="utf-8")
    native = tmp_path / "native"
    native.write_bytes(b"native")
    return work, source, reference, native, fake_ffprobe(tmp_path)


def test_the_request_pins_the_recording_and_copies_only_the_policy(tmp_path: Path) -> None:
    work, source, reference, native, ffprobe = setup(tmp_path)
    request = bootstrap.build(work, source, reference, native, ffprobe, tmp_path / "out")
    data = json.loads(request.read_text(encoding="utf-8"))
    assert data["source"] == {"path": str(source), "sha256": "feed"}
    assert data["frameClock"]["sha256"] == digest(b"clock")
    assert data["native"]["sha256"] == digest(b"native")
    assert {key: data[key] for key in POLICY} == POLICY
    assert set(data) == {"version", "source", "frameClock", "sourceMetadata", *POLICY, "native"}
    metadata = json.loads((tmp_path / "out/source-stream-metadata-v1.json").read_text())
    assert metadata["sourceSHA256"] == "feed"
    assert metadata["streams"] == [{"codec_type": "video"}]
    assert data["sourceMetadata"]["sha256"] == digest(
        (tmp_path / "out/source-stream-metadata-v1.json").read_bytes()
    )


def test_an_existing_output_a_missing_clock_or_a_reference_without_policy_is_refused(
    tmp_path: Path,
) -> None:
    work, source, reference, native, ffprobe = setup(tmp_path)
    (tmp_path / "taken").mkdir()
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        bootstrap.build(work, source, reference, native, ffprobe, tmp_path / "taken")
    reference.write_text(json.dumps({"decoder": 1}), encoding="utf-8")
    with pytest.raises(ReplayError, match="lacks policy fields"):
        bootstrap.build(work, source, reference, native, ffprobe, tmp_path / "a")
    (work / "video-clock.json").unlink()
    with pytest.raises(ReplayError, match="no video-clock.json"):
        bootstrap.build(work, source, reference, native, ffprobe, tmp_path / "b")


def test_a_source_the_preparation_does_not_pin_is_refused(tmp_path: Path) -> None:
    work, source, _, _, _ = setup(tmp_path)
    assert bootstrap.source_digest(work / "preparation.json", source) == "feed"
    with pytest.raises(ReplayError, match="does not pin this source"):
        bootstrap.source_digest(work / "preparation.json", tmp_path / "other.mp4")


def test_main_prints_the_request_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    work, source, reference, native, ffprobe = setup(tmp_path)
    code = bootstrap.main(
        [str(work), str(source), str(reference), str(tmp_path / "viamain"),
         "--native", str(native), "--ffprobe", str(ffprobe)]
    )  # fmt: skip
    assert code == 0
    assert capsys.readouterr().out.strip().endswith("BOOTSTRAP-REQUEST-v1.json")
