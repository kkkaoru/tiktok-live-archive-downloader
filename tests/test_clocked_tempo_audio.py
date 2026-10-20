import json
import subprocess
import wave
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from replay import clocked_tempo_audio as job
from replay.clocked_cut_audio import CutSlot
from replay.stages import file_digest

FFMPEG = Path("/opt/homebrew/bin/ffmpeg")


@pytest.mark.parametrize("ratio", [F(1, 16), F(1, 5), F(1, 2), F(1), F(3), F(16)])
def test_tempo_factorization_preserves_overall_rate(ratio: F) -> None:
    factors = [float(item.split("=")[1]) for item in job.tempo_filter(ratio).split(",")]
    product = 1.0
    for factor in factors:
        assert 0.5 <= factor <= 2
        product *= factor
    assert len(factors) <= 4
    assert product == pytest.approx(float(ratio))


@pytest.mark.parametrize("ratio", [F(0), F(-1), F(1, 17), F(17)])
def test_tempo_factorization_rejects_unbounded_rates(ratio: F) -> None:
    with pytest.raises(ValueError, match="ratio"):
        job.tempo_filter(ratio)


@pytest.fixture
def tones(tmp_path: Path) -> Path:
    source = tmp_path / "tones.wav"
    time = np.arange(32000) / 8000
    signal = np.column_stack((np.sin(2 * np.pi * 440 * time), np.sin(2 * np.pi * 660 * time)))
    with wave.open(str(source), "wb") as writer:
        writer.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(np.rint(signal * 20000).astype("<i2").tobytes())
    return source


@pytest.mark.parametrize("target", [16000, 6000, 40000, 2000])
def test_stereo_pitch_preserved_instead_of_sample_rate_pitch_change(
    tones: Path, target: int
) -> None:
    policy = job.TempoPolicy(maximum_tail_error_seconds=F(1, 4))
    with wave.open(str(tones), "rb") as reader:
        raw, measured = job.tempo_cut(
            reader,
            CutSlot(F(1), F(2), 0, target, ((8000, 16000),)),
            ffmpeg=FFMPEG,
            policy=policy,
        )
    result = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
    frequencies = np.fft.rfftfreq(target, 1 / 8000)
    assert abs(frequencies[np.argmax(abs(np.fft.rfft(result[:, 0])))] - 440) < 2
    assert abs(frequencies[np.argmax(abs(np.fft.rfft(result[:, 1])))] - 660) < 2
    assert len(result) == target
    assert measured.filter_applied is True
    assert abs(measured.tail_error_samples) < 8000 * policy.maximum_tail_error_seconds
    assert np.sqrt(np.mean(result[:200].astype(float) ** 2)) > 5000
    assert np.sqrt(np.mean(result[-200:].astype(float) ** 2)) > 5000


def test_identity_is_bit_exact_without_process(
    tones: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Identity must not start a process")

    monkeypatch.setattr("replay.clocked_tempo_audio.subprocess.run", forbidden)
    with wave.open(str(tones), "rb") as reader:
        raw, measured = job.tempo_cut(
            reader,
            CutSlot(F(0), F(1), 0, 8000, ((0, 8000),)),
            ffmpeg=FFMPEG,
            policy=job.TempoPolicy(),
        )
        reader.setpos(0)
        assert raw == reader.readframes(8000)
    assert measured.filter_applied is False
    assert measured.tail_error_samples == 0


@pytest.mark.parametrize("target", [40000, 2000])
def test_chained_localized_sound_stays_in_mapped_center(tmp_path: Path, target: int) -> None:
    rate = 8000
    time = np.arange(32000) / rate
    envelope = np.exp(-(((time - 1.4) / 0.04) ** 2))
    source = tmp_path / "localized.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(
            np.rint(envelope * np.sin(2 * np.pi * 440 * time) * 20000).astype("<i2").tobytes()
        )
    with wave.open(str(source), "rb") as reader:
        if target == 40000:
            # Sparse audio can lose more tail than a continuous tone at the
            # same rate: chain support is not a universal timing guarantee.
            with pytest.raises(ValueError, match="tail discrepancy"):
                job.tempo_cut(
                    reader,
                    CutSlot(F(1), F(2), 0, target, ((8000, 16000),)),
                    ffmpeg=FFMPEG,
                    policy=job.TempoPolicy(maximum_tail_error_seconds=F(1, 4)),
                )
            return
        raw, measured = job.tempo_cut(
            reader,
            CutSlot(F(1), F(2), 0, target, ((8000, 16000),)),
            ffmpeg=FFMPEG,
            policy=job.TempoPolicy(maximum_tail_error_seconds=F(1, 4)),
        )
    energy = np.frombuffer(raw, dtype="<i2").astype(float) ** 2
    centroid = float(np.dot(np.arange(target), energy) / energy.sum()) / rate
    expected = 0.4 * target / rate
    print(
        f"localized target={target}, centroid={centroid}, expected={expected}, "
        f"tail={measured.tail_error_samples}, crop={measured.center_crop_start_sample}:"
        f"{measured.center_crop_end_sample}, available={measured.filter_output_samples}"
    )
    assert abs(centroid - expected) < 0.25
    assert measured.center_crop_end_sample <= measured.filter_output_samples
    assert measured.synthetic_left_context_samples == 0
    assert measured.synthetic_right_context_samples == 0
    assert len(energy) == target


def test_tiny_cut_context_keeps_nonzero_audio(tones: Path) -> None:
    with wave.open(str(tones), "rb") as reader:
        raw, measured = job.tempo_cut(
            reader,
            CutSlot(F(1), F(2), 0, 80, ((8000, 8040),)),
            ffmpeg=FFMPEG,
            policy=job.TempoPolicy(),
        )
    assert len(raw) == 320
    assert np.max(abs(np.frombuffer(raw, dtype="<i2"))) > 10000
    assert measured.source_context_samples == 4000


@pytest.mark.parametrize("span", [(0, 4000), (28000, 32000)])
def test_source_edge_context_is_reported(tones: Path, span: tuple[int, int]) -> None:
    with wave.open(str(tones), "rb") as reader:
        raw, measured = job.tempo_cut(
            reader, CutSlot(F(0), F(1), 0, 8000, (span,)), ffmpeg=FFMPEG, policy=job.TempoPolicy()
        )
    assert len(raw) == 32000
    assert (
        measured.synthetic_left_context_samples + measured.synthetic_right_context_samples == 4000
    )
    assert np.sqrt(np.mean(np.frombuffer(raw, dtype="<i2").astype(float) ** 2)) > 5000


def test_window_is_exact_crop_across_cut_boundary(tones: Path, tmp_path: Path) -> None:
    slots = (
        CutSlot(F(0), F(1), 0, 12000, ((0, 8000),)),
        CutSlot(F(1), F(2), 12000, 24000, ((8000, 16000),)),
    )
    job.render(tones, tmp_path / "full.wav", slots, ffmpeg=FFMPEG, policy=job.TempoPolicy())
    job.render(
        tones,
        tmp_path / "part.wav",
        slots,
        ffmpeg=FFMPEG,
        policy=job.TempoPolicy(),
        output_start=11990,
        output_end=12010,
    )
    with wave.open(str(tmp_path / "full.wav"), "rb") as reader:
        reader.setpos(11990)
        expected = reader.readframes(20)
    with wave.open(str(tmp_path / "part.wav"), "rb") as reader:
        assert reader.readframes(20) == expected


@pytest.mark.parametrize("target", [32000, 30000])
def test_long_slot_is_split_within_budget(
    tones: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: int
) -> None:
    slots = (CutSlot(F(0), F(4), 0, target, ((0, 32000),)),)
    job.render(tones, tmp_path / "whole.wav", slots, ffmpeg=FFMPEG, policy=job.TempoPolicy())
    monkeypatch.setattr(job, "MAX_WINDOW_BYTES", 40000 * 4)
    measured = job.render(
        tones, tmp_path / "split.wav", slots, ffmpeg=FFMPEG, policy=job.TempoPolicy()
    )
    assert [m.output_end_sample for m in measured] == [*range(10000, target, 10000), target]
    with wave.open(str(tmp_path / "split.wav"), "rb") as split:
        assert split.getnframes() == target
        if target == 32000:
            assert split.readframes(target) == (tmp_path / "whole.wav").read_bytes()[44:]


def test_chirp_frequency_tracks_source_time_without_octave_shift(tmp_path: Path) -> None:
    source = tmp_path / "chirp.wav"
    time = np.arange(32000) / 8000
    signal = np.rint(20000 * np.sin(2 * np.pi * (200 * time + 50 * time * time))).astype("<i2")
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(signal.tobytes())
    with wave.open(str(source), "rb") as reader:
        raw, measured = job.tempo_cut(
            reader,
            CutSlot(F(1), F(3), 0, 32000, ((8000, 24000),)),
            ffmpeg=FFMPEG,
            policy=job.TempoPolicy(),
        )
    audio = np.frombuffer(raw, dtype="<i2")
    frequencies = np.fft.rfftfreq(2000, 1 / 8000)
    assert 330 < frequencies[np.argmax(abs(np.fft.rfft(audio[7000:9000])))] < 370
    assert 430 < frequencies[np.argmax(abs(np.fft.rfft(audio[23000:25000])))] < 470
    assert measured.center_crop_end_sample - measured.center_crop_start_sample == 32000


def test_boundary_voiced_bursts_survive_short_cut(tmp_path: Path) -> None:
    source = tmp_path / "bursts.wav"
    time = np.arange(24000) / 8000
    envelope = ((time >= 1) & (time < 1.08)) | ((time >= 1.42) & (time < 1.5))
    signal = np.rint(20000 * np.sin(2 * np.pi * 440 * time) * envelope).astype("<i2")
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(signal.tobytes())
    with wave.open(str(source), "rb") as reader:
        raw, measured = job.tempo_cut(
            reader,
            CutSlot(F(1), F(3, 2), 0, 8000, ((8000, 12000),)),
            ffmpeg=FFMPEG,
            policy=job.TempoPolicy(),
        )
    audio = np.frombuffer(raw, dtype="<i2").astype(float)
    assert np.sqrt(np.mean(audio[:1600] ** 2)) > 3000
    assert np.sqrt(np.mean(audio[-1600:] ** 2)) > 3000
    assert measured.synthetic_left_context_samples == 0


@pytest.mark.parametrize("context,budget", [(F(0), F(1, 10)), (F(3), F(1, 10)), (F(1, 2), F(1))])
def test_invalid_policy(context: F, budget: F) -> None:
    with pytest.raises(ValueError):
        job.TempoPolicy(context, budget)


@pytest.mark.parametrize(
    "slot",
    [
        CutSlot(F(0), F(1), 0, 0, ((0, 4),)),
        CutSlot(F(0), F(1), 0, 8000, ()),
        CutSlot(F(0), F(1), 0, 8000, ((0, 40000),)),
        CutSlot(F(0), F(1), 0, 8000, ((0, 100),)),
    ],
)
def test_invalid_slot(tones: Path, slot: CutSlot) -> None:
    with wave.open(str(tones), "rb") as reader, pytest.raises(ValueError):
        job.tempo_cut(reader, slot, ffmpeg=FFMPEG, policy=job.TempoPolicy())


def test_tail_error_not_silently_padded(tones: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def truncated(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess([], 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("replay.clocked_tempo_audio.subprocess.run", truncated)
    with wave.open(str(tones), "rb") as reader, pytest.raises(ValueError, match="tail discrepancy"):
        job.tempo_cut(
            reader,
            CutSlot(F(0), F(1), 0, 12000, ((0, 8000),)),
            ffmpeg=FFMPEG,
            policy=job.TempoPolicy(),
        )


@pytest.fixture
def config(tones: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("replay.caption_measured_job.ROOT", tmp_path)
    scratch = tmp_path / "edits/work/tmp"
    scratch.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(scratch))
    clock, path = tmp_path / "clock.json", tmp_path / "config.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
                "frames": [{"pts": 0, "nb_samples": 32000}],
            }
        ),
        encoding="utf-8",
    )
    ref = {"path": str(clock), "sha256": file_digest(clock)}
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinateSpace": "native-presentation-seconds",
                "source": {"path": str(tones), "sha256": file_digest(tones)},
                "clock": ref,
                "evidence": [ref],
                "intervals": [
                    {"start": {"value": 0, "timescale": 1}, "end": {"value": 4, "timescale": 1}}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_receipt(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "edits/work/out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tempo",
            str(config),
            str(output),
            "--sha256",
            file_digest(config),
            "--ffmpeg",
            str(FFMPEG),
            "--maximum-rate-change",
            "0.5",
        ],
    )
    job.main()
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["outputSamples"] == 32000
    assert receipt["speechQualityVerified"] is False
    assert receipt["cuts"][0]["filter_applied"] is False


def test_explicit_external_source_keeps_outputs_local(
    config: Path, tones: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr("replay.caption_measured_job.ROOT", checkout)
    with pytest.raises(ValueError, match="checkout"):
        job.source_path(str(tones), None)
    assert job.source_path(str(tones), tmp_path) == tones
    monkeypatch.setattr("replay.caption_measured_job.ROOT", tmp_path)
    data = json.loads(config.read_text())
    data["source"]["sha256"] = "0" * 64
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="binding"):
        job.run(
            config,
            tmp_path / "edits/work/out",
            sha256=file_digest(config),
            ffmpeg=FFMPEG,
            maximum_rate_change=0.5,
            policy=job.TempoPolicy(),
            source_root=tmp_path,
        )
    with pytest.raises(ValueError, match="checkout"):
        job.run(
            config,
            tmp_path.parent / "escape",
            sha256=file_digest(config),
            ffmpeg=FFMPEG,
            maximum_rate_change=0.5,
            policy=job.TempoPolicy(),
            source_root=tmp_path,
        )


def test_authorized_source_symlinks_and_invalid_roots(tones: Path, tmp_path: Path) -> None:
    link = tmp_path / "linked.wav"
    link.symlink_to(tones)
    with pytest.raises(ValueError, match="Linked"):
        job.source_path(str(link), tmp_path)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        job.source_path(str(tones), root_link)
    for root, source in [
        (Path("relative"), str(tones)),
        (tones, str(tones)),
        (tmp_path, "relative.wav"),
        (tmp_path, str(tmp_path)),
        (tmp_path / "other", str(tones)),
    ]:
        with pytest.raises(ValueError):
            job.source_path(source, root)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(ValueError, match="explicit"):
        job.source_path(str(tones), allowed)


@pytest.mark.parametrize("key,value", [("version", True), ("intervals", None), ("evidence", None)])
def test_invalid_config(config: Path, tmp_path: Path, key: str, value: object) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    data[key] = value
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        job.run(
            config,
            tmp_path / "edits/work/out",
            sha256=file_digest(config),
            ffmpeg=FFMPEG,
            maximum_rate_change=0.5,
            policy=job.TempoPolicy(),
        )
