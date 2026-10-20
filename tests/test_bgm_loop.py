"""Loop BGM rebuild: period, per-segment alignment, consensus choice and seamless tiling."""

import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from replay import bgm_loop
from replay.core import ReplayError

RATE = 1000
PERIOD = 12_000
REPEATS = 10


def save(path: Path, samples: np.ndarray) -> Path:
    with bgm_loop.written(path, RATE) as sink:
        sink.writeframes(bgm_loop.pcm(samples))
    return path


def loop() -> np.ndarray:
    return np.random.default_rng(7).normal(0, 3000, (PERIOD, 2))


def recording() -> tuple[np.ndarray, np.ndarray]:
    """Six talked-over repetitions, a 7-frame join jump, a gain change and a foreign noise in the
    first quiet repetition."""
    rng = np.random.default_rng(11)
    music = np.tile(loop(), (REPEATS, 1))
    music[60_000:] *= 0.8
    voice = np.zeros_like(music)
    leak = rng.normal(0, 4000, (72_000, 2))
    voice[:72_000] = leak * 2
    music[:72_000] += leak * (np.arange(72_000) % PERIOD < 9_000)[:, None]
    music[76_000:81_000] += rng.normal(0, 6000, (5_000, 2))
    music = np.concatenate([music[:54_000], np.zeros((7, 2)), music[54_000:-7]])
    return music, voice


def circular_ncc(found: np.ndarray, truth: np.ndarray) -> float:
    a, b = found.mean(axis=1), truth.mean(axis=1)
    lags = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), len(a))
    return float(lags.max() / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_extract_rebuilds_the_loop_from_the_clean_repetitions(tmp_path: Path) -> None:
    music, voice = recording()
    music_path, voice_path = save(tmp_path / "m.wav", music), save(tmp_path / "v.wav", voice)

    receipt = bgm_loop.extract(music_path, voice_path, out=tmp_path / "o.wav", repeats=3)

    with bgm_loop.opened(tmp_path / "o.wav") as out:
        tiled = out.read(0, out.frames)
    assert receipt["periodFrames"] == PERIOD
    assert receipt["durationSeconds"] == 36.0
    assert receipt["uncleanBlockSeconds"] == []
    assert len(tiled) == 3 * PERIOD
    assert circular_ncc(tiled[:PERIOD], loop()) > 0.999
    assert np.array_equal(tiled[:PERIOD], tiled[PERIOD : 2 * PERIOD])
    assert receipt["songStartFrame"] == 0
    assert receipt["sourceRepetitions"] == [7]


def test_a_phase_with_speech_in_every_repetition_is_reported(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    music = np.tile(loop(), (REPEATS, 1))
    voice = np.zeros_like(music)
    starts = np.arange(REPEATS) * PERIOD + 4_000
    mask = np.zeros(len(music), bool)
    mask[(starts[:, None] + np.arange(2_000)).ravel()] = True
    music[mask] += rng.normal(0, 9000, (int(mask.sum()), 2))
    voice[mask] = 1000

    receipt = bgm_loop.extract(
        save(tmp_path / "m.wav", music), save(tmp_path / "v.wav", voice),
        out=tmp_path / "o.wav", repeats=1,
    )  # fmt: skip

    assert receipt["uncleanBlockSeconds"] == [3.0, 4.0, 5.0, 6.0]


def test_coarse_period_prefers_the_fundamental_over_its_multiples() -> None:
    low = np.tile(np.random.default_rng(1).normal(size=600), 20)

    assert bgm_loop.coarse_period(low, 50.0) == pytest.approx(12.0, abs=0.02)


def test_coarse_period_needs_two_repetitions_and_a_repeat() -> None:
    with pytest.raises(ReplayError, match="too short"):
        bgm_loop.coarse_period(np.ones(900), 50.0)
    with pytest.raises(ReplayError, match="No repetition"):
        bgm_loop.coarse_period(np.linspace(0, 1, 5000), 50.0)


def test_offset_finds_a_sub_frame_lag() -> None:
    signal = np.sin(np.arange(3000) * 0.05) + np.random.default_rng(2).normal(0, 0.1, 3000)

    lag, ncc, scale = bgm_loop.offset(signal[1005:1505] * 2, signal[1000:1520])

    assert lag == pytest.approx(-5.0, abs=0.05)
    assert ncc > 0.99
    assert scale == pytest.approx(0.5, abs=0.01)


def test_fractional_delay_reads_between_frames() -> None:
    source = np.stack([np.arange(200.0)] * 2, axis=1)

    shifted = bgm_loop.fractional(source, 0.25, 10)

    assert shifted[:, 0] == pytest.approx(np.arange(32.25, 42.25), abs=0.01)


def test_consensus_removes_a_step_every_repetition_sees_and_rejects_none() -> None:
    offsets = np.array([[0.0, 5.0, 5.0], [1.0, 6.0, np.nan], [2.0, 7.0, 7.0]])

    assert bgm_loop.consensus(offsets).tolist()[0] == [0.0, 0.0, 0.0]
    unseen = np.array([[np.nan, 0.0, 5.0], [np.nan, 1.0, 6.0]])
    assert bgm_loop.consensus(unseen)[:, 1:].tolist() == [[0.0, 0.0], [1.0, 1.0]]
    with pytest.raises(ReplayError, match="No repetition"):
        bgm_loop.consensus(np.full((2, 2), np.nan))


def test_unmeasured_segments_take_the_nearest_measurement_in_time() -> None:
    offsets = np.array([[np.nan, 1.0, np.nan], [np.nan, np.nan, 5.0]])

    assert bgm_loop.nearest_in_time(offsets).tolist() == [[1.0, 1.0, 1.0], [1.0, 5.0, 5.0]]
    assert bgm_loop.nearest_in_time(np.array([[np.nan, 2.0]])).tolist() == [[2.0, 2.0]]


def test_choose_needs_two_agreeing_copies_unless_the_loop_is_silent() -> None:
    stack = np.stack([np.zeros((4, 2)), np.ones((4, 2)), np.ones((4, 2))])

    assert bgm_loop._choose(np.zeros((3, 4, 2))) == (0, 0.0, [0])
    assert bgm_loop._choose(np.ones((1, 4, 2))) == (0, 1.0, [0])
    index, error, members = bgm_loop._choose(stack)
    assert index == 1
    assert error == pytest.approx(0.0)
    assert members == [1, 2]
    assert bgm_loop._choose(stack + np.array([0.0, 0.5])[None, None, :])[2] == [0, 1, 2]


def test_spectral_median_drops_a_sound_carried_by_a_minority_of_copies() -> None:
    rng = np.random.default_rng(5)
    music = rng.normal(0, 1000, (8000, 2))
    leaks = rng.normal(0, 500, (2, 8000, 2))
    copies = np.stack([music, music, music, music + leaks[0], music + leaks[1]])

    rebuilt, short = bgm_loop.spectral_median(copies)

    assert np.abs(rebuilt - music).max() < 300
    assert float(np.sqrt(((rebuilt - music) ** 2).mean())) < 30
    assert short == 0
    assert bgm_loop.spectral_median(copies[3:])[0].tolist() == copies[3].tolist()


def test_spectral_median_uses_only_copies_where_nobody_speaks() -> None:
    rng = np.random.default_rng(6)
    music = rng.normal(0, 1000, (8000, 2))
    damaged = music * 0.3
    copies = np.stack([damaged, damaged, damaged, music, music, music])
    frames = bgm_loop.frame_count(8000)
    levels = np.array([[500.0] * frames] * 3 + [[0.0] * frames] * 3)

    rebuilt, short = bgm_loop.spectral_median(copies, levels, 10.0)
    fallback, few = bgm_loop.spectral_median(copies[:4], levels[:4], 10.0)

    assert float(np.sqrt(((rebuilt - music) ** 2).mean())) < 1
    assert short == 0
    assert float(np.sqrt(((fallback - damaged) ** 2).mean())) < 1
    assert few == frames


def test_parabolic_keeps_edges_and_flat_tops() -> None:
    assert bgm_loop._parabolic(np.array([3.0, 1.0, 0.0]), 0) == 0.0
    assert bgm_loop._parabolic(np.array([1.0, 1.0, 1.0]), 1) == 1.0


def test_where_everyone_talks_each_bin_takes_the_copies_without_voice_there() -> None:
    """Separation strips the music under the voice: half the copies lose the low tone, half the
    high one. A broadband choice keeps three copies missing the same tone; per bin, the copies
    whose voice lies elsewhere still hold it."""
    t = np.arange(8192)
    low = 3000 * np.sin(2 * np.pi * 100 * t / 2048)
    high = 3000 * np.sin(2 * np.pi * 400 * t / 2048)
    music = np.stack([low + high, low + high], axis=1)
    lost_high = np.stack([low, low], axis=1)
    lost_low = np.stack([high, high], axis=1)
    copies = np.stack([lost_high] * 8 + [lost_low] * 8)
    vocals = np.stack([lost_low] * 8 + [lost_high] * 8) / 3
    levels = np.ones((16, bgm_loop.frame_count(8192)))

    broadband, short = bgm_loop.spectral_median(copies, levels, 0.5)
    per_bin, _ = bgm_loop.spectral_median(copies, levels, 0.5, vocals)

    inner = slice(2048, 6144)
    assert short == bgm_loop.frame_count(8192)
    assert circular_ncc(broadband[inner], music[inner]) < 0.75
    assert circular_ncc(per_bin[inner], music[inner]) > 0.999
    assert np.array_equal(
        bgm_loop.spectral_median(copies, np.zeros_like(levels), 0.5, vocals)[0],
        bgm_loop.spectral_median(copies, np.zeros_like(levels), 0.5)[0],
    )


def test_stereo_reads_pad_outside_the_file_and_refuse_mono(tmp_path: Path) -> None:
    path = save(tmp_path / "a.wav", np.array([[1.0, 2.0], [3.0, 4.0]]))
    mono = tmp_path / "mono.wav"
    with wave.open(str(mono), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(RATE)
        sink.writeframes(b"\0\0")

    with bgm_loop.opened(path) as stereo:
        assert stereo.read(-1, 4).tolist() == [[0, 0], [1, 2], [3, 4], [0, 0]]
    with pytest.raises(ReplayError, match="stereo"), bgm_loop.opened(mono):
        pass


def test_stereo_reads_32_bit_pcm_and_round_trips_float(tmp_path: Path) -> None:
    wide = tmp_path / "w.wav"
    with wave.open(str(wide), "wb") as sink:
        sink.setnchannels(2)
        sink.setsampwidth(4)
        sink.setframerate(RATE)
        sink.writeframes(np.array([65536 * 3 + 32768, -65536 * 5], "<i4").tobytes())
    with bgm_loop.written(tmp_path / "f.wav", RATE, floating=True) as sink:
        sink.writeframes(bgm_loop.float_pcm(np.array([[-1.5, 40000.0], [0.001, 0.0]])))

    with bgm_loop.opened(wide) as stereo:
        assert stereo.read(0, 1).tolist() == [[3.5, -5.0]]
    with bgm_loop.opened(tmp_path / "f.wav") as stereo:
        assert stereo.frames == 2
        assert stereo.read(0, 2).tolist() == [
            [-1.5, 40000.0], [pytest.approx(0.001, rel=1e-6), 0.0]
        ]  # fmt: skip
    assert (tmp_path / "f.wav").read_bytes()[:4] == b"RIFF"


def test_a_wav_past_the_riff_size_limit_is_rf64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bgm_loop, "RIFF_SIZE_LIMIT", 64)

    with bgm_loop.written(tmp_path / "big.wav", RATE, floating=True) as sink:
        sink.writeframes(bgm_loop.float_pcm(np.full((3, 2), 2.0)))

    raw = (tmp_path / "big.wav").read_bytes()
    assert raw[:4] == b"RF64"
    assert raw[12:16] == b"ds64"
    with bgm_loop.opened(tmp_path / "big.wav") as stereo:
        assert stereo.frames == 3
        assert stereo.read(2, 2).tolist() == [[2.0, 2.0], [0.0, 0.0]]


def chunk(name: bytes, body: bytes) -> bytes:
    return name + len(body).to_bytes(4, "little") + body + b"\0" * (len(body) % 2)


def riff(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def fmt(tag: int, bits: int, *, channels: int = 2) -> bytes:
    block = channels * bits // 8
    return chunk(b"fmt ", struct.pack("<HHIIHH", tag, channels, RATE, RATE * block, block, bits))


def test_wave_header_variants_are_read_or_refused(tmp_path: Path) -> None:
    extensible = struct.pack("<HHIIHHHHI", 0xFFFE, 2, RATE, RATE * 6, 6, 24, 22, 24, 3)
    cases = {
        "ext.wav": riff(
            chunk(b"LIST", b"odd"),
            chunk(b"fmt ", extensible + struct.pack("<H", 1) + bytes(14)),
            chunk(b"data", bytes([0, 0, 1, 0, 0, 0xFF])),
        ),
        "aiff.wav": b"FORM" + bytes(8),
        "late.wav": riff(chunk(b"data", bytes(4)), fmt(1, 16)),
        "nodata.wav": riff(fmt(1, 16)),
        "byte.wav": riff(fmt(1, 8), chunk(b"data", bytes(2))),
    }
    paths = {name: tmp_path / name for name in cases}
    paths["ext.wav"].write_bytes(cases["ext.wav"])
    paths["aiff.wav"].write_bytes(cases["aiff.wav"])
    paths["late.wav"].write_bytes(cases["late.wav"])
    paths["nodata.wav"].write_bytes(cases["nodata.wav"])
    paths["byte.wav"].write_bytes(cases["byte.wav"])

    with bgm_loop.opened(paths["ext.wav"]) as stereo:
        assert stereo.read(0, 1).tolist() == [[256.0, -256.0]]
    with pytest.raises(ReplayError, match="RIFF or RF64"), bgm_loop.opened(paths["aiff.wav"]):
        pass
    with pytest.raises(ReplayError, match="precedes"), bgm_loop.opened(paths["late.wav"]):
        pass
    with pytest.raises(ReplayError, match="no data"), bgm_loop.opened(paths["nodata.wav"]):
        pass
    with pytest.raises(ReplayError, match="stereo"), bgm_loop.opened(paths["byte.wav"]):
        pass


def test_analysis_and_rebuild_refuse_unusable_inputs(tmp_path: Path) -> None:
    short = save(tmp_path / "s.wav", np.zeros((10, 2)))
    longer = save(tmp_path / "l.wav", np.zeros((20, 2)))
    alignment = bgm_loop.Alignment(10.0, 10.0, 10, np.full((1, 1), np.nan), np.ones(1), 0)

    with bgm_loop.opened(short) as a, bgm_loop.opened(longer) as b:
        with pytest.raises(ReplayError, match="differ"):
            bgm_loop.analysis(a, b)
        with pytest.raises(ReplayError, match="covers loop frame -50"):
            bgm_loop.rebuild(a, np.zeros(1), alignment)


def test_residual_subtracts_and_keeps_each_parts_first_span(tmp_path: Path) -> None:
    mix_a = save(tmp_path / "ma.wav", np.full((6, 2), 10.0))
    voice_a = save(tmp_path / "va.wav", np.full((6, 2), 3.0))
    mix_b = save(tmp_path / "mb.wav", np.full((5, 2), 20.0))
    voice_b = save(tmp_path / "vb.wav", np.full((5, 2), 1.0))

    frames = bgm_loop.residual(
        [(mix_a, voice_a), (mix_b, voice_b)], keep_seconds=0.004, out=tmp_path / "r.wav"
    )

    with bgm_loop.opened(tmp_path / "r.wav") as out:
        assert out.read(0, out.frames)[:, 0].tolist() == [7, 7, 7, 7, 19, 19, 19, 19, 19]
    assert frames == 9
    with pytest.raises(ReplayError, match="No parts"):
        bgm_loop.residual([], keep_seconds=1, out=tmp_path / "x.wav")
    with pytest.raises(ReplayError, match="do not match"):
        bgm_loop.residual([(mix_a, voice_b)], keep_seconds=1, out=tmp_path / "y.wav")


def test_main_runs_both_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    music, voice = recording()
    music_path, voice_path = save(tmp_path / "m.wav", music), save(tmp_path / "v.wav", voice)
    receipt = tmp_path / "receipt.json"

    assert bgm_loop.main(
        ["residual", str(tmp_path / "r.wav"), "--part", str(music_path), str(voice_path),
         "--keep-seconds", "1"]
    ) == 0  # fmt: skip
    assert bgm_loop.main(
        ["extract", str(music_path), str(voice_path), str(tmp_path / "o.wav"), str(receipt),
         "--repeats", "2"]
    ) == 0  # fmt: skip

    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0]) == {"output": str(tmp_path / "r.wav"), "frames": 120_000}
    assert json.loads(lines[1])["durationSeconds"] == 24.0
    assert json.loads(receipt.read_text(encoding="utf-8"))["repeats"] == 2
    with pytest.raises(ReplayError, match="at least 1"):
        bgm_loop.main(["extract", "a", "b", "c", "d", "--repeats", "0"])


def song(silence_at: int) -> np.ndarray:
    """A 12 s loop whose song fades into 1 s of dithered silence ending at frame `silence_at`."""
    rng = np.random.default_rng(9)
    loop = rng.normal(0, 3000, (PERIOD, 2)) * np.linspace(1, 0.2, PERIOD)[:, None]
    quiet = rng.integers(-2, 3, (1000, 2)).astype(float)
    return np.roll(np.concatenate([loop[: PERIOD - 1000], quiet]), silence_at, axis=0)


def test_song_start_is_the_attack_after_the_silence() -> None:
    assert bgm_loop.song_start(song(4000), RATE) == 3990
    assert bgm_loop.song_start(song(11_500), RATE) == 11_490
    assert bgm_loop.song_start(song(5), RATE) == 11_995
    assert bgm_loop.song_start(loop(), RATE) == 0
    assert bgm_loop.song_start(np.zeros((PERIOD, 2)), RATE) == 0


def test_retile_rotates_a_saved_loop_to_its_song_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    master = save(tmp_path / "master.wav", song(4000))

    assert bgm_loop.main(["retile", str(master), str(tmp_path / "o.wav"), "--repeats", "2"]) == 0

    with bgm_loop.opened(tmp_path / "o.wav") as out:
        tiled = out.read(0, out.frames)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "master": str(master), "output": str(tmp_path / "o.wav"),
        "startSeconds": 3.99, "durationSeconds": 24.0, "gainDB": pytest.approx(8.855, abs=1e-3),
    }  # fmt: skip
    assert len(tiled) == 2 * PERIOD
    assert np.abs(tiled[:10]).max() <= 6
    assert np.abs(tiled[10:20]).max() > 250
    assert np.abs(tiled[PERIOD - 990 : PERIOD]).max() <= 6
    assert 20 * np.log10(bgm_loop.true_peak(tiled[:PERIOD]) / 32768) == pytest.approx(
        -1.0, abs=1e-6
    )


def test_true_peak_finds_the_peak_between_samples_and_gain_keeps_fractions(
    tmp_path: Path,
) -> None:
    phase = 2 * np.pi * (np.arange(PERIOD) / 4 + 1 / 8)
    between = np.stack([1000 * np.sin(phase), np.zeros(PERIOD)], axis=1)
    quiet = np.full((PERIOD, 2), 0.25)

    start, gain_db = bgm_loop.tile(quiet, rate=RATE, out=tmp_path / "q.wav", repeats=1)
    silent = bgm_loop.tile(np.zeros((PERIOD, 2)), rate=RATE, out=tmp_path / "z.wav",
                           repeats=1, peak_db=-1.0)  # fmt: skip

    assert np.abs(between).max() == pytest.approx(707.107, abs=1e-3)
    assert bgm_loop.true_peak(between) == pytest.approx(1000.0, abs=1e-6)
    assert (start, gain_db) == (0, 0.0)
    assert silent == (0, 0.0)
    with bgm_loop.opened(tmp_path / "q.wav") as out:
        assert out.read(0, 2).tolist() == [[0.25, 0.25], [0.25, 0.25]]
