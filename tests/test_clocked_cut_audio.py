import json
import os
import wave
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from replay import clocked_cut_audio as job
from replay.audio_clock import AudioFrameClock
from replay.stages import file_digest


def write_audio(path: Path, samples: int = 8000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(np.arange(samples, dtype="<i2").tobytes())


@pytest.mark.parametrize(
    "pts,end,ratio",
    [
        ((0, 4000), F(1), F(1)),
        ((0, 6000), F(5, 4), F(4, 5)),
        ((0, 2000), F(3, 4), F(4, 3)),
        ((8000, 12000), F(2), F(1)),
    ],
)
def test_real_frame_gaps_overlaps_and_origin(pts: tuple[int, int], end: F, ratio: F) -> None:
    clock = AudioFrameClock(pts, (4000, 4000), 8000, F(1, 8000))
    slots = job.plan_slots(
        clock, ((F(pts[0], 8000), end),), rate=8000, samples=8000, maximum_rate_change=0.4
    )
    assert slots[0].input_samples == 8000
    assert F(8000, slots[0].output_end) == ratio
    assert slots[0].spans == ((0, 8000),)


def test_absolute_rounding_does_not_accumulate() -> None:
    clock = AudioFrameClock((0,), (8000,), 8000, F(1, 8000))
    slots = job.plan_slots(
        clock,
        ((F(0), F(1, 30)), (F(1, 30), F(2, 30)), (F(2, 30), F(1, 10))),
        rate=8000,
        samples=8000,
        maximum_rate_change=0.01,
    )
    assert tuple(slot.output_end for slot in slots) == (267, 533, 800)
    assert tuple(slot.spans for slot in slots) == (((0, 267),), ((267, 533),), ((533, 800),))


def test_overlap_across_cut_boundary_never_reverses_samples() -> None:
    clock = AudioFrameClock((0, 2000), (4000, 4000), 8000, F(1, 8000))
    slots = job.plan_slots(
        clock,
        ((F(0), F(3, 8)), (F(3, 8), F(3, 4))),
        rate=8000,
        samples=8000,
        maximum_rate_change=0.7,
    )
    assert slots[0].spans == ((0, 5000),)
    assert slots[1].spans == ((5000, 8000),)


@pytest.mark.parametrize(
    "intervals,rate,samples,limit",
    [
        ((), 8000, 8000, 0.5),
        (((F(0), F(1)),), 7999, 8000, 0.5),
        (((F(0), F(1)),), 8000, 8002, 0.5),
        (((F(0), F(1)),), 8000, 8000, float("nan")),
        (((F(1), F(2)),), 8000, 8000, 0.5),
        (((F(0), F(1, 100000)),), 8000, 8000, 0.5),
        (((F(0), F(1)), (F(0), F(1))), 8000, 8000, 0.5),
    ],
)
def test_invalid_mapping(
    intervals: tuple[tuple[F, F], ...], rate: int, samples: int, limit: float
) -> None:
    with pytest.raises(ValueError):
        job.plan_slots(
            AudioFrameClock((0,), (8000,), 8000, F(1, 8000)),
            intervals,
            rate=rate,
            samples=samples,
            maximum_rate_change=limit,
        )


def test_extreme_local_change_rejected() -> None:
    with pytest.raises(ValueError, match="maximum local"):
        job.plan_slots(
            AudioFrameClock((0, 6000), (4000, 4000), 8000, F(1, 8000)),
            ((F(0), F(5, 4)),),
            rate=8000,
            samples=8000,
            maximum_rate_change=0.1,
        )


def test_identity_and_partial_window(tmp_path: Path) -> None:
    source, full, part = tmp_path / "source.wav", tmp_path / "full.wav", tmp_path / "part.wav"
    write_audio(source)
    slots = (job.CutSlot(F(0), F(1), 0, 8000, ((0, 8000),)),)
    assert job.render(source, full, slots) == 8000
    assert full.read_bytes() == source.read_bytes()
    assert job.render(source, part, slots, output_start=1000, output_end=2000) == 1000
    with wave.open(str(part), "rb") as reader:
        assert reader.readframes(1000) == np.arange(1000, 2000, dtype="<i2").tobytes()


def test_disjoint_order_preserved_without_mix(tmp_path: Path) -> None:
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    write_audio(source)
    slot = job.CutSlot(F(0), F(1), 0, 6000, ((0, 2000), (4000, 8000)))
    job.render(source, output, (slot,))
    with wave.open(str(output), "rb") as reader:
        actual = np.frombuffer(reader.readframes(6000), dtype="<i2")
    assert np.array_equal(actual, np.concatenate((np.arange(2000), np.arange(4000, 8000))))


@pytest.mark.parametrize("target,expected", [(5, [0, 70, 150, 230, 300]), (3, [17, 150, 283])])
def test_nonidentity_stereo_interpolation(tmp_path: Path, target: int, expected: list[int]) -> None:
    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(
            np.array([[0, 0], [100, -100], [200, -200], [300, -300]], dtype="<i2").tobytes()
        )
    with wave.open(str(source), "rb") as reader:
        actual = job.resampled_block(
            reader, job.CutSlot(F(0), F(1), 0, target, ((0, 4),)), start=0, end=target
        )
    assert actual[:, 0].tolist() == expected
    assert actual[:, 1].tolist() == [-value for value in expected]


def test_resampled_window_matches_full_across_block_boundary(tmp_path: Path) -> None:
    source, full, part = tmp_path / "source.wav", tmp_path / "full.wav", tmp_path / "part.wav"
    write_audio(source, 70000)
    slots = (job.CutSlot(F(0), F(10), 0, 80000, ((0, 70000),)),)
    job.render(source, full, slots)
    job.render(source, part, slots, output_start=65530, output_end=65550)
    with wave.open(str(full), "rb") as reader:
        reader.setpos(65530)
        expected = reader.readframes(20)
    with wave.open(str(part), "rb") as reader:
        assert reader.readframes(20) == expected


def test_rate_conversion_and_half_sample_tie() -> None:
    slots = job.plan_slots(
        AudioFrameClock((0,), (44100,), 44100, F(1, 44100)),
        ((F(0), F(1)),),
        rate=16000,
        samples=16000,
        maximum_rate_change=0,
    )
    assert slots[0].spans == ((0, 16000),)
    assert job.sample_boundary(F(1, 32000), 16000) == 0
    assert job.sample_boundary(F(3, 32000), 16000) == 1


def test_no_source_before_native_origin() -> None:
    clock = AudioFrameClock((8000,), (8000,), 8000, F(1, 8000))
    with pytest.raises(ValueError, match="no source"):
        job.plan_slots(clock, ((F(0), F(1, 2)),), rate=8000, samples=8000, maximum_rate_change=0.5)


def test_pcm24_rejected(tmp_path: Path) -> None:
    source = tmp_path / "24.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 3, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(bytes(24000))
    with pytest.raises(ValueError, match="PCM16"):
        job.render(source, tmp_path / "out.wav", (job.CutSlot(F(0), F(1), 0, 8000, ((0, 8000),)),))


def test_empty_slots_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Empty"):
        job.render(tmp_path / "source.wav", tmp_path / "out.wav", ())


def test_bad_render_window_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="window"):
        job.render(
            tmp_path / "source.wav",
            tmp_path / "out.wav",
            (job.CutSlot(F(0), F(1), 0, 8000, ((0, 8000),)),),
            output_start=8000,
        )


@pytest.mark.parametrize("start,end", [(-1, 2), (0, 0), (0, 8001)])
def test_bad_block(tmp_path: Path, start: int, end: int) -> None:
    source = tmp_path / "source.wav"
    write_audio(source)
    with wave.open(str(source), "rb") as reader, pytest.raises(ValueError):
        job.resampled_block(
            reader, job.CutSlot(F(0), F(1), 0, 8000, ((0, 8000),)), start=start, end=end
        )


def test_truncated_source(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_audio(source, 10)
    with pytest.raises(ValueError, match="Truncated"):
        job.render(source, tmp_path / "out.wav", (job.CutSlot(F(0), F(1), 0, 8000, ((0, 8000),)),))


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("replay.caption_measured_job.ROOT", tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    source, clock, config = (
        tmp_path / "source.wav",
        tmp_path / "clock.json",
        tmp_path / "config.json",
    )
    write_audio(source)
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
                "frames": [{"pts": 0, "nb_samples": 4000}, {"pts": 6000, "nb_samples": 4000}],
            }
        )
    )
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinateSpace": "native-presentation-seconds",
                "source": {"path": str(source), "sha256": file_digest(source)},
                "clock": {"path": str(clock), "sha256": file_digest(clock)},
                "intervals": [
                    {"start": {"value": 0, "timescale": 1}, "end": {"value": 5, "timescale": 4}}
                ],
                "evidence": [{"path": str(clock), "sha256": file_digest(clock)}],
            }
        )
    )
    return config


def test_cli_candidate_receipt(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "edits/work/result"
    monkeypatch.setattr(
        "sys.argv",
        [
            "clocked-cut-audio",
            str(config),
            str(output),
            "--sha256",
            file_digest(config),
            "--maximum-rate-change",
            "0.3",
            "--start-seconds",
            "1/4",
            "--end-seconds",
            "1",
        ],
    )
    job.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outputSamples"] == 6000
    assert receipt["cutSlots"][0]["rateRatio"] == "4/5"
    assert receipt["AVSyncVerified"] is False
    with wave.open(str(output / "candidate.wav"), "rb") as reader:
        assert reader.getnframes() == 6000


@pytest.mark.parametrize(
    "key,value",
    [("version", True), ("coordinateSpace", "other"), ("intervals", None), ("evidence", None)],
)
def test_invalid_config(config: Path, tmp_path: Path, key: str, value: object) -> None:
    data = json.loads(config.read_text())
    data[key] = value
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        job.run(
            config,
            tmp_path / "edits/work/result",
            sha256=file_digest(config),
            maximum_rate_change=0.3,
        )


def test_identity_ignores_reads_but_not_writes(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    path.write_bytes(b"pcm")
    before = job.identity(path)
    status = path.stat()
    os.utime(path, ns=(status.st_atime_ns + 5_000_000_000, status.st_mtime_ns))
    assert job.identity(path) == before
    path.write_bytes(b"pcm2")
    assert job.identity(path) != before
