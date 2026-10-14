import json
import wave
from pathlib import Path

import pytest

from replay import continuous_cut_job as job
from replay.continuous_cuts import write_pcm_spans
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("replay.caption_measured_job.ROOT", tmp_path)
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\xe8\x03" * 4000 + b"\xd0\x07" * 4000)
    clock = tmp_path / "clock.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
                "frames": [{"pts": 0, "nb_samples": 4000}, {"pts": 6000, "nb_samples": 4000}],
            }
        ),
        encoding="utf-8",
    )
    clock_ref = {"path": str(clock), "sha256": file_digest(clock)}
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinateSpace": "native-presentation-seconds",
                "source": {"path": str(source), "sha256": file_digest(source)},
                "clock": clock_ref,
                "evidence": [clock_ref],
                "intervals": [
                    {"start": {"value": 1, "timescale": 4}, "end": {"value": 5, "timescale": 4}}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_keeps_gap_as_duration_difference_not_zero_samples(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        ["continuous-cut-job", str(config), str(output), "--sha256", file_digest(config)],
    )
    job.main()
    result = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert result["spans"] == [[2000, 8000]]
    assert result["sourceSamples"] == 8000
    assert result["selectedSamples"] == 6000
    assert result["nativeIntervalDuration"] == "1"
    assert result["audioDuration"] == "3/4"
    assert result["durationDifference"] == "-1/4"
    assert result["paddingSamples"] == 0
    assert result["resamplingPerformed"] is False
    assert result["fadesApplied"] is False
    assert result["independentPCMComparisonPassed"] is False
    assert result["sourceCorrespondenceVerified"] is False
    assert result["AVSyncVerified"] is False
    assert result["deliveryComplete"] is False
    assert result["inputs"][str(config)] == file_digest(config)
    target = output / "selected.wav"
    assert result["outputSHA256"] == file_digest(target)
    with wave.open(str(target), "rb") as reader:
        assert reader.getparams()[:4] == (1, 2, 8000, 6000)
        assert reader.readframes(6001) == b"\xe8\x03" * 2000 + b"\xd0\x07" * 4000
    assert target.stat().st_mode & 0o777 == 0o600
    assert (output / "selection.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        job.select(config, output, sha256=file_digest(config))


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("version", 2),
        ("coordinateSpace", "output-seconds"),
        ("evidence", None),
        ("evidence", [None] * 129),
        ("intervals", []),
        ("intervals", "bad"),
        ("unexpected", True),
    ],
)
def test_invalid_job_never_creates_output(
    config: Path, tmp_path: Path, key: str, value: object
) -> None:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    data[key] = value
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        job.select(config, tmp_path / "rejected", sha256=file_digest(config))
    assert not (tmp_path / "rejected").exists()


def test_changed_source_rejected(config: Path, tmp_path: Path) -> None:
    with (tmp_path / "source.wav").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Pinned PCM source"):
        job.select(config, tmp_path / "rejected", sha256=file_digest(config))
    assert not (tmp_path / "rejected").exists()


def test_source_budget(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job, "MAX_PCM_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        job.select(config, tmp_path / "rejected", sha256=file_digest(config))


def test_changed_input_after_copy_preserves_partial(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def change(source: Path, output: Path, spans: tuple[tuple[int, int], ...]) -> int:
        count = write_pcm_spans(source, output, spans)
        (tmp_path / "clock.json").write_text("{}", encoding="utf-8")
        return count

    monkeypatch.setattr(job, "write_pcm_spans", change)
    with pytest.raises(ValueError, match="input changed"):
        job.select(config, tmp_path / "partial", sha256=file_digest(config))
    assert (tmp_path / "partial/selected.wav").is_file()
    assert not (tmp_path / "partial/selection.json").exists()


def test_wrong_output_header_is_not_accepted(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong(_source: Path, output: Path, _spans: tuple[tuple[int, int], ...]) -> int:
        with wave.open(str(output), "wb") as writer:
            writer.setparams((1, 2, 8001, 0, "NONE", "not compressed"))
            writer.writeframes(b"\x01\x00" * 6000)
        return 6000

    monkeypatch.setattr(job, "write_pcm_spans", wrong)
    with pytest.raises(ValueError, match="header differs"):
        job.select(config, tmp_path / "partial", sha256=file_digest(config))
    assert not (tmp_path / "partial/selection.json").exists()
