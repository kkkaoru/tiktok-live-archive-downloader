import json
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import partial_audio_edge as edge
from replay.audio_clock import AudioFrameClock
from replay.caption_projection import CutSpan
from replay.caption_repair import ClockLabel
from replay.clocked_tempo_audio import TempoPolicy
from replay.core import ReplayError
from replay.partial_audio_edge import plan, render
from replay.stages import StageError, file_digest


def _write(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 10000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\xff\x7f" + bytes(9899 * 2))
    clock = _write(
        tmp_path / "clock.json",
        {
            "streams": [{"sample_rate": "10000", "time_base": "1/10000"}],
            "frames": [{"pts": 100, "nb_samples": 9900}],
        },
    )
    labels = _write(
        tmp_path / "labels.json",
        {
            "clock": "source-presentation",
            "labels": [{"text": "あ", "start": 0.01, "end": 0.02, "provenance": "observed"}],
            "parentSHA256": {str(clock): file_digest(clock)},
        },
    )
    cuts = _write(
        tmp_path / "cuts.json",
        {
            "version": 1,
            "clock": "source-presentation",
            "fps": 30,
            "sourceFrames": 30,
            "cuts": [{"source_start_frame": 0, "source_end_frame": 30, "output_start_frame": 0}],
            "parentSHA256": {
                str(clock): file_digest(clock),
                str(labels): file_digest(labels),
                str(source): file_digest(source),
            },
        },
    )
    executable = tmp_path / "unused-ffmpeg"
    executable.write_bytes(b"not executed for rate one")
    names = (
        "partial_audio_edge.py",
        "clocked_cut_audio.py",
        "clocked_tempo_audio.py",
        "audio_clock.py",
        "cut_audio_support.py",
        "continuous_quiet_plan.py",
        "caption_projection.py",
    )
    dependencies = {
        str(Path(edge.__file__).with_name(name).resolve()): file_digest(
            Path(edge.__file__).with_name(name)
        )
        for name in names
    }
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    return _write(
        tmp_path / "config.json",
        {
            "version": 1,
            "kind": edge.KIND,
            "internalClockPolicy": edge.POLICY,
            "maximumRateChange": 0.5,
            "dependencies": dependencies,
            "artifacts": {
                "cuts": _ref(cuts),
                "labels": _ref(labels),
                "audioClock": _ref(clock),
                "source": _ref(source),
                "ffmpeg": _ref(executable),
            },
        },
    )


@pytest.mark.parametrize("prepare_only", [False, True])
def test_public_cli_roundtrip(config: Path, prepare_only: bool) -> None:
    output = config.parent / "result"
    before = file_digest(config.parent / "labels.json")
    assert edge.main(["prepare" if prepare_only else "render", str(config), str(output)]) == 0
    assert edge.main(["verify-plan" if prepare_only else "verify", str(config), str(output)]) == 0
    receipt = edge.verify(config, output, prepare_only=prepare_only)
    assert receipt["kind"] == edge.KIND
    assert receipt["outputSamples"] == 10000
    assert receipt["recognitionSourceAnchorsChanged"] is False
    assert file_digest(config.parent / "labels.json") == before
    with pytest.raises(ReplayError, match="fresh"):
        edge.execute(config, output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "legacy"),
        ("internalClockPolicy", "unknown"),
        ("dependencies", {}),
        ("artifacts", {}),
        ("maximumRateChange", True),
    ],
)
def test_cli_rejects_invalid_contract(config: Path, field: str, value: object) -> None:
    data = json.loads(config.read_text())
    data[field] = value
    _write(config, data)
    with pytest.raises(ReplayError):
        edge.execute(config, config.parent / "bad")


@pytest.mark.parametrize("name", ["source.wav", "labels.json", "unused-ffmpeg"])
def test_cli_rejects_changed_pins(config: Path, name: str) -> None:
    with (config.parent / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(StageError):
        edge.execute(config, config.parent / "bad")


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "context-atempo-exact-cut-slot-candidate"),
        ("outputSamples", 9999),
        ("measurements", []),
        ("outputSHA256", "0" * 64),
    ],
)
def test_verifier_rejects_false_receipt(config: Path, field: str, value: object) -> None:
    output = config.parent / "render"
    edge.execute(config, output)
    receipt = json.loads((output / "receipt.json").read_text())
    receipt[field] = value
    _write(output / "receipt.json", receipt)
    with pytest.raises(ReplayError):
        edge.verify(config, output)


def test_invalid_scratch_is_rejected(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TMPDIR")
    with pytest.raises(ReplayError, match="scratch"):
        edge.execute(config, config.parent / "bad")


@pytest.mark.parametrize("dependency", ["producer", "model"])
def test_changed_code_or_model_dependency_is_rejected(config: Path, dependency: str) -> None:
    data = json.loads(config.read_text())
    if dependency == "producer":
        data["dependencies"][str(Path(edge.__file__).resolve())] = "0" * 64
    else:
        model = config.parent / "synthetic-model.bin"
        model.write_bytes(b"fixed synthetic weights")
        data["dependencies"][str(model)] = file_digest(model)
        model.write_bytes(b"changed synthetic weights")
    _write(config, data)
    with pytest.raises(StageError):
        edge.execute(config, config.parent / "bad")


def test_fractional_cell_retains_vowel_and_reports_sample_rounding() -> None:
    result = plan(
        (CutSpan(0, 60, 0),),
        (ClockLabel("あ", 0.473, 0.7, "observed"),),
        AudioFrameClock((473,), (44100,), 44100, Fraction(1, 1000)),
        fps=30,
        rate=44100,
        samples=44100,
        maximum_rate_change=0.5,
    )
    assert result.cuts == (CutSpan(14, 45, 0),)
    assert result.removed == ((0, 14), (45, 60))
    assert result.silence == ((0, 279), (44379, 45570))
    assert result.slots[0].spans == ((0, 44100),)
    assert result.slots[0].output_start == 279
    assert result.slots[0].output_end == 44379
    assert result.boundary_errors == (Fraction(-3, 10), Fraction(-3, 10))
    assert result.output_samples == 45570


def test_impulse_is_placed_after_known_absence_without_losing_source_samples(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "source.wav", tmp_path / "result.wav"
    raw = b"\xff\x7f" + bytes(9898 * 2) + b"\x01\x00"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 10000, 0, "NONE", "not compressed"))
        writer.writeframes(raw)
    prepared = plan(
        (CutSpan(0, 30, 0),),
        (ClockLabel("声", 0.01, 0.02, "raw"),),
        AudioFrameClock((100,), (9900,), 10000, Fraction(1, 10000)),
        fps=30,
        rate=10000,
        samples=9900,
        maximum_rate_change=0.5,
    )
    measured = render(source, output, prepared, ffmpeg=tmp_path / "unused", policy=TempoPolicy())
    with wave.open(str(output), "rb") as reader:
        assert reader.getnframes() == 10000
        assert reader.readframes(100) == bytes(200)
        assert reader.readframes(9900) == raw
    assert measured[0].filter_applied is False


@pytest.mark.parametrize("text,start,end", [("声", 0.1, 0.3), ("点", 0.1, 0.1)])
def test_removed_recognition_and_points_are_never_dropped(
    text: str, start: float, end: float
) -> None:
    with pytest.raises(ReplayError, match="recognized"):
        plan(
            (CutSpan(0, 30, 0),),
            (ClockLabel(text, start, end, "raw"),),
            AudioFrameClock((2000,), (8000,), 10000, Fraction(1, 10000)),
            fps=30,
            rate=10000,
            samples=8000,
            maximum_rate_change=0.5,
        )


def test_speech_in_partial_missing_audio_is_rejected() -> None:
    with pytest.raises(ReplayError, match="outside measured"):
        plan(
            (CutSpan(0, 30, 0),),
            (ClockLabel("声", 0.001, 0.02, "raw"),),
            AudioFrameClock((100,), (9900,), 10000, Fraction(1, 10000)),
            fps=30,
            rate=10000,
            samples=9900,
            maximum_rate_change=0.5,
        )


def test_disjoint_cuts_reset_offsets_and_remove_whole_unsupported_cuts() -> None:
    result = plan(
        (CutSpan(0, 1, 0), CutSpan(6, 12, 1), CutSpan(15, 18, 7), CutSpan(29, 30, 10)),
        (),
        AudioFrameClock((2000,), (5000,), 10000, Fraction(1, 10000)),
        fps=30,
        rate=10000,
        samples=5000,
        maximum_rate_change=0.5,
    )
    assert result.cuts == (CutSpan(6, 12, 0), CutSpan(15, 18, 6))
    assert result.removed == ((0, 1), (29, 30))
    assert result.silence == ()
    assert result.output_samples == 3000


@pytest.mark.parametrize(
    "cuts,fps,message",
    [
        ((), 30, "cut count"),
        ((CutSpan(0, 1, 0),), 0, "frame rate"),
        ((CutSpan(0, 1, 1),), 30, "contiguous"),
        ((CutSpan(0, 1, 0),), 30, "No retained"),
    ],
)
def test_invalid_cut_contracts(cuts: tuple[CutSpan, ...], fps: int, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        plan(
            cuts,
            (),
            AudioFrameClock((2000,), (8000,), 10000, Fraction(1, 10000)),
            fps=fps,
            rate=10000,
            samples=8000,
            maximum_rate_change=0.5,
        )


def test_label_budget() -> None:
    with pytest.raises(ReplayError, match="budget"):
        plan(
            (CutSpan(0, 30, 0),),
            (ClockLabel("a", 0.3, 0.4, "r"),) * 100001,
            AudioFrameClock((0,), (10000,), 10000, Fraction(1, 10000)),
            fps=30,
            rate=10000,
            samples=10000,
            maximum_rate_change=0.5,
        )


@pytest.mark.parametrize(
    "silence,total,message",
    [
        (((0, 101),), 10000, "overlap"),
        ((), 10000, "hole"),
        (((0, 100),), 10001, "extent"),
    ],
)
def test_renderer_rejects_unaccounted_samples(
    tmp_path: Path, silence: tuple[tuple[int, int], ...], total: int, message: str
) -> None:
    prepared = plan(
        (CutSpan(0, 30, 0),),
        (),
        AudioFrameClock((100,), (9900,), 10000, Fraction(1, 10000)),
        fps=30,
        rate=10000,
        samples=9900,
        maximum_rate_change=0.5,
    )
    with pytest.raises(ReplayError, match=message):
        render(
            tmp_path / "absent",
            tmp_path / "out.wav",
            replace(prepared, silence=silence, output_samples=total),
            ffmpeg=tmp_path / "unused",
            policy=TempoPolicy(),
        )


def test_pcm_header_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 10000, 0, "NONE", "not compressed"))
        writer.writeframes(bytes(20000))
    prepared = plan(
        (CutSpan(0, 30, 0),),
        (),
        AudioFrameClock((100,), (9900,), 10000, Fraction(1, 10000)),
        fps=30,
        rate=10000,
        samples=9900,
        maximum_rate_change=0.5,
    )
    with pytest.raises(ReplayError, match="identity"):
        render(
            source, tmp_path / "out.wav", prepared, ffmpeg=tmp_path / "unused", policy=TempoPolicy()
        )
