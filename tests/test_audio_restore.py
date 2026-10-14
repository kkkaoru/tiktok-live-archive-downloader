"""Synthetic tests for safe, evidence-gated audio revision creation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import audio_restore as restore
from replay.audio_expander import ExpansionVerificationDict
from replay.bgm_separation import (
    AdaptiveMixResult,
    SeparationPolicy,
    SeparationResult,
    SpeechQuality,
)
from replay.core import ReplayError

SAMPLE_RATE = 44100


class _ProbeBackend:
    def __init__(self, quiet: bool = False, weak: bool = False) -> None:
        self.calls = 0
        self.quiet = quiet
        self.weak = weak

    @property
    def sources(self) -> tuple[str, ...]:
        return ("drums", "bass", "other", "vocals")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk * np.float32(0.9)

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        self.calls += 1
        if self.quiet:
            return np.zeros_like(chunk)
        if self.weak:
            return chunk
        return chunk if self.calls % 2 else chunk * np.float32(0.001)


def _fixture_audio() -> NDArray[np.float32]:
    frames = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    mono = 0.2 * np.sin(2.0 * np.pi * 440.0 * frames)
    return np.asarray(np.stack((mono, mono)), dtype=np.float32)


def test_run_and_video_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "12\n", ""),
    )
    completed = restore._run(["ffmpeg"])
    assert completed.returncode == 0
    assert restore._video_frames(Path("video.mp4")) == 12
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "bad media"),
    )
    with pytest.raises(ReplayError, match="bad media"):
        restore._run(["ffmpeg"])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "N/A", ""),
    )
    with pytest.raises(ReplayError, match="frame count"):
        restore._video_frames(Path("video.mp4"))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "0", ""),
    )
    with pytest.raises(ReplayError, match="no decoded frames"):
        restore._video_frames(Path("video.mp4"))


def test_run_reports_missing_program(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(ReplayError, match="install FFmpeg"):
        restore._run(["ffmpeg"])


def test_speech_mask_padding_and_bad_transcript(tmp_path: Path) -> None:
    subtitle = tmp_path / "spoken.srt"
    subtitle.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    mask = restore._speech_mask(subtitle, 4 * SAMPLE_RATE)
    assert mask[: SAMPLE_RATE // 2].sum() == 0
    assert mask[SAMPLE_RATE // 2 : int(2.5 * SAMPLE_RATE)].all()
    subtitle.write_text("not an SRT", encoding="utf-8")
    with pytest.raises(ReplayError, match="transcript"):
        restore._speech_mask(subtitle, SAMPLE_RATE)
    subtitle.write_text("1\n00:00:05,000 --> 00:00:06,000\nlate\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="no speech"):
        restore._speech_mask(subtitle, SAMPLE_RATE)


def test_expander_filter_level_shifts_deep_threshold() -> None:
    audio_filter, gain_db = restore._expander_filter(-71.85, -86.85)
    assert gain_db == pytest.approx(11.85)
    assert audio_filter.startswith("volume=11.85dB,compand=")
    assert "points=-90/-90|-80.00/-89.90|-60.00/-60.00|0/0" in audio_filter
    assert audio_filter.endswith("volume=-11.85dB")


def test_probe_starts_and_music_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    assert restore._probe_starts(SAMPLE_RATE * 20) == (0,)
    assert restore._probe_starts(SAMPLE_RATE * 180) == (
        0,
        SAMPLE_RATE * 60,
        SAMPLE_RATE * 120,
    )
    monkeypatch.setattr(restore, "MUSIC_PROBE_SECONDS", 2)
    audio = np.tile(_fixture_audio(), 5)
    vocals = audio * np.float32(0.9)
    probes = restore._measure_music(audio, vocals)
    assert tuple(probe.start_seconds for probe in probes) == (0.0, 2.0, 4.0)
    assert len(probes) == 3
    expected = 10.0 * np.log10(
        np.mean((audio[:, : 2 * SAMPLE_RATE] - vocals[:, : 2 * SAMPLE_RATE]) ** 2)
    )
    assert probes[0].before_dbfs == pytest.approx(expected)


def test_music_probe_rejects_misshapen_vocal_estimate() -> None:
    with pytest.raises(ReplayError, match="does not align"):
        restore._measure_music(_fixture_audio(), _fixture_audio()[:, :-1])


def test_probe_verification_passes_quiet_and_rejects_weak_removal() -> None:
    speech = SpeechQuality(-0.1, -0.5, 0.999)
    probe = restore.MusicProbe(0.0, -15.0)
    passed = restore._verify_probes(
        (probe,), (restore.MusicProbe(0.0, -30.0),), speech, SeparationPolicy()
    )
    assert passed[0].passed is True
    quiet = restore.MusicProbe(0.0, -80.0)
    untouched = restore._verify_probes((quiet,), (quiet,), speech, SeparationPolicy())
    assert untouched[0].passed is None
    assert untouched[0].after_dbfs == pytest.approx(-80.0)
    introduced = restore._verify_probes(
        (quiet,), (restore.MusicProbe(0.0, -30.0),), speech, SeparationPolicy()
    )
    assert introduced[0].passed is False
    assert "introduced instrumental" in str(introduced[0].reason)
    weak = restore._verify_probes(
        (probe,), (restore.MusicProbe(0.0, -20.0),), speech, SeparationPolicy()
    )
    assert weak[0].passed is False
    assert "below the minimum" in str(weak[0].reason)


def test_expander_is_optional_when_quiet_speech_is_hurt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(restore, "MUSIC_PROBE_SECONDS", 1)
    before = np.tile(_fixture_audio(), 2)
    mask = np.ones(2 * SAMPLE_RATE, dtype=np.bool_)
    probes = (restore.MusicProbe(0.0, -70.0), restore.MusicProbe(1.0, -30.0))
    verified: ExpansionVerificationDict = {
        "passed": True,
        "suppressionDb": 5.0,
        "speechLossDb": 0.0,
        "reason": "Residual suppressed",
    }
    hurt = before.copy()
    hurt[:, :SAMPLE_RATE] *= np.float32(0.3)
    selected, quality = restore._select_expander(
        probes, before, hurt, mask, verified, SeparationPolicy()
    )
    assert selected is False
    assert quality is not None
    assert quality.level_delta_db_p05 < -1.5
    safe = before.copy()
    safe[:, SAMPLE_RATE:] *= np.float32(0.3)
    selected, quality = restore._select_expander(
        probes, before, safe, mask, verified, SeparationPolicy()
    )
    assert selected is True
    assert quality is not None
    assert quality.level_delta_db_p05 == pytest.approx(0.0)
    rejected: ExpansionVerificationDict = {
        "passed": False,
        "suppressionDb": 0.0,
        "speechLossDb": 0.0,
        "reason": "Weak reduction",
    }
    assert restore._select_expander(probes, before, safe, mask, rejected, SeparationPolicy()) == (
        False,
        None,
    )
    assert restore._select_expander(
        (restore.MusicProbe(0.0, -30.0),), before, safe, mask, verified, SeparationPolicy()
    ) == (False, None)


def test_save_report_never_overwrites(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    restore._save_report(report, {"status": "verified"})
    assert '"verified"' in report.read_text(encoding="utf-8")
    with pytest.raises(ReplayError, match="overwrite"):
        restore._save_report(report, {"status": "changed"})


def _mock_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    quiet: bool = False,
    weak: bool = False,
    expander_weak: bool = False,
) -> tuple[Path, Path, _ProbeBackend]:
    source = (
        tmp_path / "edits" / "replay-test" / "revisions" / "old" / "delivery" / "full-review.mp4"
    )
    source.parent.mkdir(parents=True)
    source.touch()
    source.with_suffix(".srt").write_text(
        "1\n00:00:00,000 --> 00:00:00,900\nhello\n", encoding="utf-8"
    )
    revision = tmp_path / "edits" / "replay-test" / "revisions" / "new"
    evidence = revision.parent.parent / "evidence"
    evidence.mkdir()
    (evidence / "source-probe.json").write_text(
        json.dumps({"format": {"filename": "/synthetic/2024-01-01_09-00-00_JST_replay-test.mp4"}}),
        encoding="utf-8",
    )
    audio = _fixture_audio()
    backend = _ProbeBackend(quiet, weak)
    loudness_runs = 0

    def validate(source_path: Path, revision_path: Path) -> tuple[Path, Path, Path]:
        return tmp_path, source_path, revision_path

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal loudness_runs
        if "ebur128" in arguments:
            loudness_runs += 1
            floor = "-50.0" if loudness_runs == 1 else ("-49.0" if expander_weak else "-60.0")
            log = "\n".join(
                (
                    f"[Parsed_ebur128_0 @ 0x1] t: 0.1 M: -20.0 S: {floor}",
                    f"[Parsed_ebur128_0 @ 0x1] t: 0.5 M: -20.0 S: {floor}",
                    f"[Parsed_ebur128_0 @ 0x1] t: 2.0 M: {floor} S: {floor}",
                    f"[Parsed_ebur128_0 @ 0x1] t: 2.5 M: {floor} S: {floor}",
                )
            )
            return subprocess.CompletedProcess(arguments, 0, "", log)
        if arguments[-1] != "-":
            Path(arguments[-1]).write_bytes(b"synthetic output")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
        if path.name == "mlx-vocals.wav":
            gain = 1.0 if quiet else 0.9
            return audio * np.float32(gain), 2
        output_gain = np.float32(10.0 ** (restore.VOICE_OUTPUT_GAIN_DB / 20.0))
        if path.name in ("encoded-audio.wav", "expanded-audio.wav"):
            return audio * output_gain, 2
        if path.name == "encoded-vocals.wav":
            gain = 0.9 if weak or quiet else 0.988 * float(output_gain)
            return audio * np.float32(gain), 2
        return audio.copy(), 2

    def separate(
        input_path: Path,
        output_path: Path,
        config: object,
        _backend: object = None,
    ) -> SeparationResult:
        output_path.touch()
        return SeparationResult(1.0, 0.01, 100.0)

    def write_wav(path: Path, values: NDArray[np.float32]) -> None:
        path.touch()

    monkeypatch.setattr(restore, "_validate_paths", validate)
    monkeypatch.setattr(restore, "_video_frames", lambda path: 100)
    monkeypatch.setattr(restore, "_run", run)
    monkeypatch.setattr(restore, "_read_wav", read_wav)
    monkeypatch.setattr(restore, "_write_wav", write_wav)
    monkeypatch.setattr(restore, "resolve_device", lambda requested, model: "mps")
    monkeypatch.setattr(restore, "_load_backend", lambda model, device: backend)
    monkeypatch.setattr(restore, "separate_vocals", separate)
    return source, revision, backend


def test_restore_delivery_publishes_only_after_verifying_voice_stems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    written: list[NDArray[np.float32]] = []
    monkeypatch.setattr(restore, "apply_wiener_vocal_mask", lambda original, vocals: vocals.copy())
    monkeypatch.setattr(
        restore,
        "adapt_vocal_mix",
        lambda original, vocals, mask: AdaptiveMixResult(vocals.copy(), 1.0, 1.0, 0.0, 0.0, 1.0),
    )

    def write_wav(path: Path, values: NDArray[np.float32]) -> None:
        if path.name == "expander-input.wav":
            written.append(values.copy())
        path.touch()

    monkeypatch.setattr(restore, "_write_wav", write_wav)

    def verify_voice(
        reference: NDArray[np.float32],
        processed: NDArray[np.float32],
        mask: NDArray[np.bool_],
    ) -> SpeechQuality:
        assert np.allclose(reference, _fixture_audio() * np.float32(0.9))
        assert np.allclose(
            processed,
            _fixture_audio() * np.float32(0.988 * 10.0 ** (restore.VOICE_OUTPUT_GAIN_DB / 20.0)),
        )
        assert mask.all()
        return SpeechQuality(0.0, 0.0, 1.0)

    monkeypatch.setattr(restore, "measure_speech_quality", verify_voice)
    report = restore.restore_delivery(source, revision, "cpu")
    assert report["status"] == "verified"
    assert report["model"] == "mlx-community/mel-roformer-mlx"
    assert report["output_voice_gain_db"] == restore.VOICE_OUTPUT_GAIN_DB
    assert report["expander_passed"] is True
    assert report["expander_applied"] is False
    assert report["expander_quiet_speech_quality"] is None
    assert report["expander_pre_gain_db"] == 0.0
    assert report["expander_policy"] == {
        "ratio": 2.0,
        "margin_above_floor_db": 3.0,
        "minimum_speech_clearance_db": 9.0,
        "maximum_threshold_above_floor_db": 60.0,
        "minimum_suppression_db": 3.0,
        "maximum_speech_loss_db": 1.5,
        "speech_padding_seconds": 0.5,
    }
    assert len(written) == 1
    assert np.allclose(
        written[0],
        _fixture_audio() * np.float32(0.9 * 10.0 ** (restore.VOICE_OUTPUT_GAIN_DB / 20.0)),
    )
    assert (revision / "delivery" / "2024-01-01_09-00-00_JST_replay-test-review.mp4").is_file()
    assert (revision / "delivery" / "2024-01-01_09-00-00_JST_replay-test-review.srt").is_file()
    assert (revision / "metadata" / "verification.json").is_file()
    assert not source.exists()
    assert source.with_suffix(".srt").is_file()
    assert (
        revision.parent.parent
        / "evidence"
        / "2024-01-01_09-00-00_JST_replay-test-review-selection.json"
    ).is_file()


def test_restore_delivery_reports_no_music_without_publishing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path, quiet=True)
    report = restore.restore_delivery(source, revision, "cpu")
    assert report["status"] == "no-instrumental-content"
    assert not list((revision / "delivery").glob("*.mp4"))


def test_restore_delivery_stops_on_missing_subtitle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    source.with_suffix(".srt").unlink()
    with pytest.raises(ReplayError, match="SRT sidecar"):
        restore.restore_delivery(source, revision, "cpu")


def test_restore_delivery_rejects_external_transcript_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    subtitle = source.with_suffix(".srt")
    subtitle.unlink()
    outside = tmp_path / "outside.srt"
    outside.write_text("1\n00:00:01,000 --> 00:00:02,000\ntext\n")
    subtitle.symlink_to(outside)
    with pytest.raises(ReplayError, match="Transcript sidecar must remain inside edits"):
        restore.restore_delivery(source, revision, "cpu")
    assert not revision.exists()


def test_restore_delivery_stops_on_speech_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    monkeypatch.setattr(
        restore,
        "adapt_vocal_mix",
        lambda original, vocals, mask: AdaptiveMixResult(vocals, 1.0, 1.0, -2.0, -2.1, 0.9),
    )
    with pytest.raises(ReplayError, match="speech quality"):
        restore.restore_delivery(source, revision, "cpu")
    assert not list((revision / "delivery").glob("*.mp4"))
    assert '"rejected-adaptive-speech-quality"' in (
        revision / "metadata" / "verification.json"
    ).read_text(encoding="utf-8")


def test_restore_delivery_records_encoded_speech_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    monkeypatch.setattr(
        restore,
        "measure_speech_quality",
        lambda original, processed, mask: SpeechQuality(0.0, -2.0, 0.999),
    )
    with pytest.raises(ReplayError, match="Encoded full-video speech"):
        restore.restore_delivery(source, revision, "cpu")
    report = json.loads((revision / "metadata" / "verification.json").read_text(encoding="utf-8"))
    assert report["status"] == "rejected-encoded-speech-quality"
    assert report["music_windows_per_video"] > 0
    assert isinstance(report["music_gate_passed"], bool)
    assert not list((revision / "delivery").glob("*.mp4"))


def test_restore_delivery_skips_weak_expander_without_rejecting_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path, expander_weak=True)
    report = restore.restore_delivery(source, revision, "cpu")
    assert report["status"] == "verified"
    assert report["expander_passed"] is False
    assert report["expander_applied"] is False
    assert (revision / "delivery" / "2024-01-01_09-00-00_JST_replay-test-review.mp4").exists()


def test_restore_delivery_records_music_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path, weak=True)
    with pytest.raises(ReplayError, match="Music verification failed"):
        restore.restore_delivery(source, revision, "cpu")
    report = (revision / "metadata" / "verification.json").read_text(encoding="utf-8")
    assert '"rejected-music-quality"' in report
    assert not list((revision / "delivery").glob("*.mp4"))


def test_listening_approved_revision_has_independent_checks_and_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    original_run = restore._run
    commands: list[list[str]] = []

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        if "stream=duration" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "1.000\n", "")
        return original_run(arguments)

    monkeypatch.setattr(restore, "_run", run)
    report = restore.apply_listening_approved(source, revision, "mps")
    assert report["status"] == "listening-approved"
    assert report["model"] == (
        "mlx-community/mel-roformer-mlx@639de4848eb79a9e29cc8cba3c01170ddf165eef"
    )
    assert report["model_sha256"] == (
        "86eee1472d825bf978bc6b6bb104bb71cb7950c4fea9dd192a8c55ed2db28b27"
    )
    assert report["numeric_music_and_speech_gates"] == (
        "not asserted; user approved MLX by listening"
    )
    assert report["full_audio_video_decode"] == "passed"
    assert any(command[-3:] == ["-f", "null", "-"] for command in commands)
    assert not source.exists()
    assert (revision / "delivery" / "2024-01-01_09-00-00_JST_replay-test-review.mp4").is_file()
    assert (revision / "delivery" / "2024-01-01_09-00-00_JST_replay-test-review.srt").is_file()
    assert json.loads((revision / "metadata" / "verification.json").read_text())["status"] == (
        "listening-approved"
    )


def _earlier_listening_revision(tmp_path: Path, *, status: str = "listening-approved") -> Path:
    revisions = tmp_path / "edits" / "replay-test" / "revisions"
    render = revisions / "plan" / "render-a"
    (render / "delivery").mkdir(parents=True)
    (render / "voice.m4a").write_bytes(b"voice")
    earlier = revisions / "earlier"
    (earlier / "delivery").mkdir(parents=True)
    (earlier / "metadata").mkdir()
    published = earlier / "delivery" / "2024-01-01_09-00-00_JST_replay-test-review.mp4"
    published.write_bytes(b"published")
    (earlier / "metadata" / "verification.json").write_text(
        json.dumps(
            {
                "status": status,
                "full_audio_video_decode": "passed",
                "model_sha256": "86eee1472d825bf978bc6b6bb104bb71cb7950c4fea9dd192a8c55ed2db28b27",
                "delivery": "edits/replay-test/revisions/earlier/delivery/"
                "2024-01-01_09-00-00_JST_replay-test-review.mp4",
                "source": "edits/replay-test/revisions/plan/render-a/delivery/full-review.mp4",
                "separator_realtime_factor": 7.5,
            }
        ),
        encoding="utf-8",
    )
    return published


def test_listening_approved_reuses_vocals_separated_from_identical_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    _earlier_listening_revision(tmp_path)
    _earlier_listening_revision(tmp_path / "unused", status="rejected-runtime")
    original_run = restore._run
    commands: list[list[str]] = []

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        if "stream=duration" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "1.000\n", "")
        if arguments[-3:] == ["-f", "md5", "-"]:
            return subprocess.CompletedProcess(arguments, 0, "MD5=same\n", "")
        return original_run(arguments)

    def separate(*_arguments: object, **_keywords: object) -> object:
        raise AssertionError("separation must be reused")

    monkeypatch.setattr(restore, "_run", run)
    monkeypatch.setattr(restore, "separate_vocals", separate)
    report = restore.apply_listening_approved(source, revision, "mps")
    assert report["status"] == "listening-approved"
    assert report["separation_reused_from"] == (
        "edits/replay-test/revisions/earlier/delivery/2024-01-01_09-00-00_JST_replay-test-review.mp4"
    )
    assert report["separator_realtime_factor"] == 7.5
    mux = next(
        command
        for command in commands
        if str(revision) in command[-1]
        and "-i" in command
        and "copy" in command
        and "-map_metadata" in command
    )
    assert mux[mux.index("-c:a") + 1] == "copy"


def test_listening_approved_separates_when_earlier_audio_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    voice = _earlier_listening_revision(tmp_path).parents[2] / "plan/render-a/voice.m4a"
    original_run = restore._run

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if "stream=duration" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "1.000\n", "")
        if arguments[-3:] == ["-f", "md5", "-"]:
            digest = "MD5=old" if str(voice) in arguments else "MD5=new"
            return subprocess.CompletedProcess(arguments, 0, digest + "\n", "")
        return original_run(arguments)

    monkeypatch.setattr(restore, "_run", run)
    report = restore.apply_listening_approved(source, revision, "mps")
    assert report["separation_reused_from"] is None
    assert report["separator_realtime_factor"] == 100.0


def test_audio_fingerprint_requires_an_md5(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        restore,
        "_run",
        lambda arguments: subprocess.CompletedProcess(arguments, 0, "", ""),
    )
    with pytest.raises(ReplayError, match="fingerprint"):
        restore._audio_fingerprint(tmp_path / "a.mp4")


def test_listening_approved_rejects_runtime_without_publishing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    original_run = restore._run

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if "stream=duration" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "1.000\n", "")
        return original_run(arguments)

    monkeypatch.setattr(restore, "_run", run)
    monkeypatch.setattr(
        restore,
        "_timing_report",
        lambda started, audio_seconds: {
            "processing_wall_seconds": 30.0,
            "processing_minutes_per_audio_hour": 30.0,
        },
    )
    with pytest.raises(ReplayError, match="30-minutes"):
        restore.apply_listening_approved(source, revision)
    assert not list((revision / "delivery").glob("*.mp4"))
    assert json.loads((revision / "metadata" / "verification.json").read_text())["status"] == (
        "rejected-runtime"
    )


def test_listening_approved_rejects_unaligned_aac_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, revision, _ = _mock_delivery(monkeypatch, tmp_path)
    original_run = restore._run

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if "stream=duration" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "2.000\n", "")
        return original_run(arguments)

    monkeypatch.setattr(restore, "_run", run)
    with pytest.raises(ReplayError, match="duration does not match"):
        restore.apply_listening_approved(source, revision)
    assert not list((revision / "delivery").glob("*.mp4"))


def test_path_validation_requires_a_new_sibling_revision(tmp_path: Path) -> None:
    revisions = tmp_path / "edits" / "replay-test" / "revisions"
    source = revisions / "old" / "delivery" / "full-review.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    with pytest.raises(ReplayError, match="broadcast date"):
        restore._validate_paths(source, revisions / "new", tmp_path)
    assert not (revisions / "new").exists()
    evidence = revisions.parent / "evidence"
    evidence.mkdir()
    (evidence / "source-probe.json").write_text(
        json.dumps({"format": {"filename": "/synthetic/2024-01-01_09-00-00_JST_replay-test.mp4"}}),
        encoding="utf-8",
    )
    valid = restore._validate_paths(source, revisions / "new", tmp_path)
    assert valid == (
        tmp_path,
        source,
        revisions / "2024-01-01_09-00-00_JST_replay-test-review-new",
    )
    canonical = revisions / "2024-01-01_09-00-00_JST_replay-test-review-new"
    canonical.mkdir()
    with pytest.raises(ReplayError, match="overwrite"):
        restore._validate_paths(source, revisions / "new", tmp_path)
    canonical.rmdir()
    wrong_source = source.parent / "wrong.mp4"
    wrong_source.touch()
    with pytest.raises(ReplayError, match="processed delivery"):
        restore._validate_paths(wrong_source, revisions / "newer", tmp_path)
    with pytest.raises(ReplayError, match="sibling"):
        restore._validate_paths(source, tmp_path / "edits" / "another-replay" / "new", tmp_path)
    existing = revisions / "existing"
    existing.mkdir()
    with pytest.raises(ReplayError, match="overwrite"):
        restore._validate_paths(source, existing, tmp_path)
    outside = tmp_path.parent / "outside.mp4"
    outside.touch()
    with pytest.raises(ReplayError, match="inside"):
        restore._validate_paths(outside, revisions / "newer", tmp_path)


def test_dated_delivery_blocks_other_revision_families(tmp_path: Path) -> None:
    root = tmp_path / "edits" / "replay-123"
    source = root / "revisions" / "old" / "delivery" / "full-review.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    evidence = root / "evidence"
    evidence.mkdir()
    (evidence / "source-probe.json").write_text(
        '{"format":{"filename":"/original/2024-01-01_09-00-00_JST_replay-123.mp4"}}',
        encoding="utf-8",
    )
    source.with_name("2024-01-01_09-00-00_JST_replay-123-review.mp4").touch()
    with pytest.raises(ReplayError, match="already has a dated delivery"):
        restore._validate_paths(source, root / "revisions" / "another-family", tmp_path)
    assert not (root / "revisions" / "another-family").exists()


def test_cli_stops_safely_and_prints_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        restore, "restore_delivery", lambda source, revision, device: {"status": "verified"}
    )
    assert restore.main(["source.mp4", "revision"]) == 0
    assert '"verified"' in capsys.readouterr().out
    monkeypatch.setattr(
        restore,
        "apply_listening_approved",
        lambda source, revision, device: {"status": "listening-approved"},
    )
    assert restore.main(["source.mp4", "revision", "--listening-approved"]) == 0
    assert '"listening-approved"' in capsys.readouterr().out

    def fail(source: Path, revision: Path, device: str) -> dict[str, object]:
        raise ReplayError("blocked")

    monkeypatch.setattr(restore, "restore_delivery", fail)
    assert restore.main(["source.mp4", "revision"]) == 2
    assert "blocked" in capsys.readouterr().err
