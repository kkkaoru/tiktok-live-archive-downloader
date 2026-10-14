import json
import shutil
import subprocess
import sys
import types
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import delivery_audio_quality as q
from replay.bgm_separation import SeparationBackend, measure_speech_quality
from replay.caption_measured_job import ROOT
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def wav(path: Path, samples: int = 441, scale: float = 1.0) -> Path:
    values = np.asarray(np.sin(np.arange(samples) * 0.1) * 10000 * scale, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(np.stack((values, values), axis=1).tobytes())
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


class Model:
    calls = 0

    @property
    def sources(self) -> tuple[str, ...]:
        return ("vocals", "instrumental")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        self.calls += 1
        return chunk.copy()

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.zeros_like(chunk)


def loader(data: dict[str, object], pins: dict[str, q.Fingerprint]) -> SeparationBackend:
    return Model()


def runner(arguments: Sequence[str], destination: Path) -> None:
    if "-show_frames" in arguments:
        write_json(
            destination,
            {
                "streams": [
                    {
                        "codec_name": "aac",
                        "sample_rate": "44100",
                        "channels": 2,
                        "time_base": "1/44100",
                        "start_pts": 0,
                        "duration_ts": 441,
                    }
                ],
                "frames": [{"pts": 0, "nb_samples": 441}],
            },
        )
    elif "-show_packets" in arguments:
        write_json(
            destination,
            {
                "packets": [
                    {
                        "pts": -1024,
                        "duration": 1024,
                        "side_data_list": [
                            {
                                "side_data_type": "Skip Samples",
                                "skip_samples": 1024,
                                "discard_padding": 0,
                            }
                        ],
                    },
                    {"pts": 0, "duration": 441},
                ]
            },
        )
    else:
        wav(Path(arguments[-1]))
        destination.write_bytes(b"")


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(q, "ROOT", tmp_path)
    source = wav(tmp_path / "voice.wav")
    baseline = wav(tmp_path / "input.wav", scale=1.5)
    selected = wav(tmp_path / "selected.wav")
    movie = tmp_path / "final.mp4"
    movie.write_bytes(b"synthetic-movie")
    tool = tmp_path / "tool"
    tool.write_bytes(b"tool")
    cuts = write_json(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 1, "output_start_frame": 0}]},
    )
    clock = write_json(
        tmp_path / "clock.json",
        {
            "streams": [{"sample_rate": "44100", "time_base": "1/44100"}],
            "frames": [{"pts": 0, "nb_samples": 441}],
        },
    )
    render = write_json(
        tmp_path / "render.json",
        {"audio": {"source": str(source), "clock": str(clock)}, "cuts": str(cuts), "rate": 100},
    )
    selection = write_json(
        tmp_path / "selection.json",
        {
            "version": 1,
            "output": str(selected),
            "outputSHA256": file_digest(selected),
            "spans": [[0, 441]],
            "inputs": {str(source): file_digest(source), str(clock): file_digest(clock)},
        },
    )
    audit = write_json(
        tmp_path / "audit.json",
        {
            "version": 2,
            "kind": "independent-pcm-selection-clock-audit",
            "comparedBytes": 1764,
            "selectedPCMBytesEqual": True,
            "mismatchedBytes": 0,
            "spans": [[0, 441]],
            "inputs": {
                str(source): file_digest(source),
                str(clock): file_digest(clock),
                str(selected): file_digest(selected),
                str(selection): file_digest(selection),
            },
        },
    )
    asr = write_json(tmp_path / "asr.json", {"words": ["synthetic"]})
    speech = write_json(
        tmp_path / "speech.json",
        {
            "coordinateSpace": "source-pcm-samples",
            "sampleRate": 44100,
            "source": ref(source),
            "asr": ref(asr),
            "intervals": [[0, 441]],
        },
    )
    return write_json(
        tmp_path / "config.json",
        {
            "version": 1,
            "renderConfig": ref(render),
            "finalMovie": ref(movie),
            "selection": ref(selection),
            "pcmAudit": ref(audit),
            "firstPassVoice": ref(source),
            "modelInput": ref(baseline),
            "originalClock": ref(clock),
            "protectedSpeech": ref(speech),
            "cuts": ref(cuts),
            "ffmpeg": ref(tool),
            "ffprobe": ref(tool),
        },
    )


def test_mapping_then_quality(config: Path) -> None:
    output = config.parent / "edits/work/map"
    receipt = q.analyze(config, output, sha256=file_digest(config), phase="mapping", runner=runner)
    data = q.read_json(receipt)
    assert data["accepted"] is False
    assert data["checks"] == {
        "selectedPCMReceiptBound": True,
        "nativeTimesPreserved": True,
        "aacSampleTimeMapping": True,
        "speechGatePassed": None,
        "musicGatePassed": None,
        "inputsStable": True,
    }
    quality = q.analyze(
        config,
        config.parent / "edits/work/quality",
        sha256=file_digest(config),
        phase="quality",
        runner=runner,
        model_loader=loader,
    )
    assert q.read_json(quality)["accepted"] is True
    with pytest.raises(FileExistsError):
        q.analyze(config, output, sha256=file_digest(config), phase="mapping", runner=runner)


def test_quality_aac_extent_blocks_model(config: Path) -> None:
    def bad_runner(arguments: Sequence[str], destination: Path) -> None:
        runner(arguments, destination)
        if destination.name == "decode.stdout":
            wav(Path(arguments[-1]), samples=442)

    receipt = q.analyze(
        config,
        config.parent / "edits/work/bad",
        sha256=file_digest(config),
        phase="quality",
        runner=bad_runner,
    )
    assert q.read_json(receipt)["accepted"] is False
    assert q.read_json(receipt)["checks"] == {
        "selectedPCMReceiptBound": True,
        "nativeTimesPreserved": True,
        "aacSampleTimeMapping": False,
        "speechGatePassed": None,
        "musicGatePassed": None,
        "inputsStable": True,
    }


def test_changed_input_after_analysis(config: Path) -> None:
    def changing(arguments: Sequence[str], destination: Path) -> None:
        runner(arguments, destination)
        if destination.name == "decode.stdout":
            config.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="changed during"):
        q.analyze(
            config,
            config.parent / "edits/work/changed",
            sha256=file_digest(config),
            phase="mapping",
            runner=changing,
        )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("version", True, "version"),
        ("renderConfig", None, "object"),
        ("modelInput", "firstPassVoice", "Independent"),
        ("finalMovie", "firstPassVoice", "Self"),
    ],
)
def test_bad_config(config: Path, key: str, value: object, match: str) -> None:
    data = q.read_json(config)
    data[key] = data[value] if isinstance(value, str) and value in data else value
    write_json(config, data)
    with pytest.raises((ValueError, TypeError, ReplayError), match=match):
        q.analyze(
            config,
            config.parent / "edits/work/bad",
            sha256=file_digest(config),
            phase="mapping",
            runner=runner,
        )


def test_low_level_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        q.fingerprint(tmp_path / "missing")
    with pytest.raises(ValueError, match="absolute"):
        q.reference({"path": "relative", "sha256": "a" * 64}, {})
    path = tmp_path / "input"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="changed"):
        q.reference({"path": str(path), "sha256": "a" * 64}, {})
    with pytest.raises(ValueError, match="Scratch"):
        q.scratch(ROOT / "escape")
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="budget"):
        q.read_json(path)
    with pytest.raises(ValueError, match="Unknown"):
        q.analyze(path, path, sha256="a" * 64, phase="bad")


def test_run_tool(tmp_path: Path) -> None:
    q.run_tool([sys.executable, "-c", "print('ok')"], tmp_path / "result.out")
    assert (tmp_path / "result.out").read_text(encoding="utf-8") == "ok\n"
    with pytest.raises(subprocess.CalledProcessError):
        q.run_tool([sys.executable, "-c", "raise SystemExit(2)"], tmp_path / "failed.out")


def test_wav_read_and_selection_errors(tmp_path: Path) -> None:
    path = wav(tmp_path / "a.wav")
    with wave.open(str(path), "rb") as stream:
        with pytest.raises(ValueError, match="Truncated"):
            q.read_samples(stream, 0, 442)
        with pytest.raises(ValueError, match="exceeds"):
            q.selected_window(stream, ((0, 441),), 0, 442)
    assert (
        q.protected_mask(((0, 100), (200, 300)), ((90, 220),), start=95, count=10).tolist()
        == [True] * 10
    )
    path.write_bytes(path.read_bytes()[:-10])
    with wave.open(str(path), "rb") as stream, pytest.raises(ValueError, match="Truncated"):
        q.read_samples(stream, 0, 441)


def test_window_extent_empty_mask_and_bad_model(tmp_path: Path) -> None:
    source = wav(tmp_path / "source.wav")
    output = wav(tmp_path / "output.wav")
    baseline = wav(tmp_path / "baseline.wav", scale=1.5)
    result = q.quality_windows(
        decoded=output,
        first_pass=source,
        model_input=baseline,
        spans=((0, 441),),
        speech=((442, 443),),
        backend=Model(),
        directory=tmp_path,
        cache_key="abc",
    )
    assert result["speechGatePassed"] is False
    assert result["musicGatePassed"] is True
    with pytest.raises(ValueError, match="extent"):
        q.quality_windows(
            decoded=output,
            first_pass=source,
            model_input=baseline,
            spans=((0, 442),),
            speech=((0, 442),),
            backend=Model(),
            directory=tmp_path,
            cache_key="abc",
        )

    class InvalidModel(Model):
        def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
            return chunk[:, :1]

    with pytest.raises(ValueError, match="invalid vocals"):
        q.quality_windows(
            decoded=output,
            first_pass=source,
            model_input=baseline,
            spans=((0, 441),),
            speech=((0, 441),),
            backend=InvalidModel(),
            directory=tmp_path,
            cache_key="abc",
        )


def test_two_windows_one_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "WINDOW", 441)
    source = wav(tmp_path / "source.wav", samples=1000)
    output = wav(tmp_path / "output.wav", samples=1000, scale=0.5)
    baseline = wav(tmp_path / "baseline.wav", samples=1000, scale=1.5)
    model = Model()
    result = q.quality_windows(
        decoded=output,
        first_pass=source,
        model_input=baseline,
        spans=((0, 1000),),
        speech=((0, 1000),),
        backend=model,
        directory=tmp_path,
        cache_key="abc",
    )
    assert model.calls == 3
    assert result["speechGatePassed"] is False
    assert result["musicGatePassed"] is False


def test_cli(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = write_json(config.parent / "receipt.json", {"accepted": False})
    monkeypatch.setattr(q, "analyze", lambda *args, **kwargs: result)
    assert q.main([str(config), str(config.parent / "out"), "--sha256", "a" * 64]) == 1
    assert "receipt.json" in capsys.readouterr().out


def test_model_binding_offline(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = config.parent / "edits/models/test"
    folder.mkdir(parents=True)
    weights = folder / "model.safetensors"
    weights.write_bytes(b"mock-verified-weights")
    model_config = write_json(folder / "config.json", {"mock": True})
    cache = config.parent / "edits/cache"
    cache.mkdir()
    data: dict[str, object] = {
        "modelWeights": ref(weights),
        "modelConfig": ref(model_config),
        "hfCache": str(cache),
    }
    hub = types.ModuleType("huggingface_hub")
    monkeypatch.setattr(
        hub, "try_to_load_from_cache", lambda *args, **kwargs: str(weights), raising=False
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(q, "MODEL_SHA256", file_digest(weights))
    monkeypatch.setattr(q, "_load_backend", lambda *args: Model())
    assert isinstance(q.load_model(data, {}), Model)
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="resolve"):
        q.load_model(data, {})
    monkeypatch.setattr(q, "MODEL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="substitution"):
        q.load_model(data, {})
    data["hfCache"] = str(config.parent)
    with pytest.raises(ValueError, match="repository-local"):
        q.load_model(data, {})
    data["hfCache"] = None
    with pytest.raises(ValueError, match="hfCache"):
        q.load_model(data, {})


def test_streaming_speech_matches_existing_metric() -> None:
    rng = np.random.default_rng(12)
    original = rng.normal(0, 0.1, (2, 1000)).astype(np.float32)
    processed = original * np.float32(0.93)
    mask = np.arange(1000) % 3 != 0
    delta1, moment1 = q.speech_moments(original[:, :882], processed[:, :882], mask[:882])
    delta2, moment2 = q.speech_moments(original[:, 882:], processed[:, 882:], mask[882:])
    combined = q.combined_speech([delta1, delta2], moment1 + moment2)
    direct = measure_speech_quality(original, processed, mask)
    assert combined is not None
    assert combined.level_delta_db_p50 == pytest.approx(direct.level_delta_db_p50, abs=1e-12)
    assert combined.level_delta_db_p05 == pytest.approx(direct.level_delta_db_p05, abs=1e-12)
    assert combined.correlation == pytest.approx(direct.correlation, abs=1e-12)
    silence = np.zeros((2, 441), dtype=np.float32)
    delta, moment = q.speech_moments(silence, silence, np.ones(441, dtype=np.bool_))
    silent = q.combined_speech([delta], moment)
    assert silent is not None
    assert silent.correlation == 0.0


@pytest.mark.parametrize(
    ("file", "key", "value", "message"),
    [
        ("render.json", "audio", {"source": "bad", "clock": "bad"}, "voice/clock"),
        ("render.json", "cuts", None, "render cuts"),
        ("render.json", "cuts", "/different", "Render cuts"),
        ("selection.json", "inputs", {}, "Selection does not bind"),
        ("audit.json", "inputs", {}, "PCM audit does not bind"),
        ("audit.json", "selectedPCMBytesEqual", False, "byte comparison"),
        ("audit.json", "spans", [[0, 440]], "spans differ"),
        ("audit.json", "version", True, "version 2"),
        ("selection.json", "version", 2, "version 1"),
        ("audit.json", "comparedBytes", 1, "all selected bytes"),
        ("speech.json", "coordinateSpace", "guessed", "coordinates"),
        ("speech.json", "intervals", [[0, 999]], "exceeds source"),
    ],
)
def test_document_binding_failures(
    config: Path, file: str, key: str, value: object, message: str
) -> None:
    path = config.parent / file
    document = q.read_json(path)
    document[key] = value
    write_json(path, document)
    data = q.read_json(config)
    names = {
        "render.json": "renderConfig",
        "selection.json": "selection",
        "audit.json": "pcmAudit",
        "speech.json": "protectedSpeech",
    }
    data[names[file]] = ref(path)
    write_json(config, data)
    with pytest.raises(ValueError, match=message):
        q.analyze(
            config,
            config.parent / "edits/work/invalid",
            sha256=file_digest(config),
            phase="mapping",
            runner=runner,
        )


def test_decode_packet_end_mismatch(tmp_path: Path) -> None:
    def altered(arguments: Sequence[str], destination: Path) -> None:
        runner(arguments, destination)
        if "-show_packets" in arguments:
            write_json(destination, {"packets": [{"pts": 0, "duration": 442}]})

    _, facts = q.decode_aac(
        tmp_path / "movie", tmp_path, ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), runner=altered
    )
    assert facts["packetFrameExtentMatches"] is False
    assert facts["packetEndMinusDecodedEnd"] == "1/44100"


@pytest.mark.parametrize("packet_duration", [441, 1024])
def test_clipped_and_coded_discard_have_same_measured_end(
    tmp_path: Path, packet_duration: int
) -> None:
    def altered(arguments: Sequence[str], destination: Path) -> None:
        runner(arguments, destination)
        if "-show_packets" in arguments:
            write_json(
                destination,
                {
                    "packets": [
                        {
                            "pts": 0,
                            "duration": packet_duration,
                            "side_data_list": [{"discard_padding": 583}],
                        }
                    ]
                },
            )

    _, facts = q.decode_aac(
        tmp_path / "movie", tmp_path, ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), runner=altered
    )
    assert facts["packetFrameExtentMatches"] is True
    assert facts["streamPresentationExtentMatches"] is True
    assert facts["packetEndMinusDecodedEnd"] == "0"
    assert facts["lastPacketDecodedEnd"] == "1/100"
    assert facts["discardPaddingAppliedAgain"] is False


def test_discard_requires_stream_confirmation(tmp_path: Path) -> None:
    def altered(arguments: Sequence[str], destination: Path) -> None:
        runner(arguments, destination)
        if "-show_packets" in arguments:
            write_json(
                destination,
                {
                    "packets": [
                        {
                            "pts": 0,
                            "duration": 1024,
                            "side_data_list": [{"discard_padding": 583}],
                        }
                    ]
                },
            )
        if "-show_frames" in arguments:
            data = q.read_json(destination)
            streams = data["streams"]
            assert isinstance(streams, list)
            record(streams[0])["duration_ts"] = 442
            write_json(destination, data)

    _, facts = q.decode_aac(
        tmp_path / "movie", tmp_path, ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), runner=altered
    )
    assert facts["packetFrameExtentMatches"] is False
    assert facts["streamPresentationExtentMatches"] is False
    assert facts["packetExtentPolicy"] == "packet-duration-presentation"


def test_real_synthetic_aac_priming(tmp_path: Path) -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    source = wav(tmp_path / "source.wav", samples=4410)
    movie = tmp_path / "candidate.m4a"
    q.run_tool(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-c:a",
            "aac",
            str(movie),
        ],
        tmp_path / "encode.stdout",
    )
    decoded, facts = q.decode_aac(
        movie,
        tmp_path,
        ffmpeg=Path(ffmpeg),
        ffprobe=Path(ffprobe),
    )
    assert decoded.is_file()
    assert facts["primingSamples"] == 1024
    assert facts["firstDecodedPTS"] == "0"
    assert facts["audioPCMIdentical"] is False
    assert facts["decodeMatchesFrameClock"] is True


def test_mapping_cache_reused_without_decode(config: Path) -> None:
    receipt = q.analyze(
        config,
        config.parent / "edits/work/cache",
        sha256=file_digest(config),
        phase="mapping",
        runner=runner,
    )
    data = q.read_json(config)
    data["aacEvidence"] = ref(receipt)
    next_config = write_json(config.parent / "quality-config.json", data)

    def forbidden(arguments: Sequence[str], destination: Path) -> None:
        raise AssertionError("A matching AAC cache must not launch decoding")

    result = q.analyze(
        next_config,
        config.parent / "edits/work/reused",
        sha256=file_digest(next_config),
        phase="quality",
        runner=forbidden,
        model_loader=loader,
    )
    assert q.read_json(result)["accepted"] is True
    assert not (result.parent / "final-decoded.wav").exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("kind", "wrong", "Invalid AAC"),
        ("inputs", None, "inputs missing"),
        ("inputs", [], "fingerprints differ"),
        ("artifacts", [], "artifacts missing"),
    ],
)
def test_cache_rejects_mismatched_receipts(
    config: Path, key: str, value: object, message: str
) -> None:
    receipt = q.analyze(
        config,
        config.parent / "edits/work/cache",
        sha256=file_digest(config),
        phase="mapping",
        runner=runner,
    )
    cached = q.read_json(receipt)
    cached[key] = value
    write_json(receipt, cached)
    data = q.read_json(config)
    data["aacEvidence"] = ref(receipt)
    next_config = write_json(config.parent / "next.json", data)
    with pytest.raises(ValueError, match=message):
        q.analyze(
            next_config,
            config.parent / "edits/work/invalid",
            sha256=file_digest(next_config),
            phase="mapping",
            runner=runner,
        )


def test_real_aac_tail_is_distinct_from_native_drift(config: Path) -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    movie = config.parent / "encoded.m4a"
    q.run_tool(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(config.parent / "voice.wav"),
            "-c:a",
            "aac",
            str(movie),
        ],
        config.parent / "encode.stdout",
    )
    data = q.read_json(config)
    data.update(
        finalMovie=ref(movie),
        ffmpeg=ref(Path(ffmpeg).resolve()),
        ffprobe=ref(Path(ffprobe).resolve()),
    )
    write_json(config, data)
    result = q.analyze(
        config, config.parent / "edits/work/real", sha256=file_digest(config), phase="mapping"
    )
    facts = q.read_json(result)
    assert facts["checks"] == {
        "selectedPCMReceiptBound": True,
        "nativeTimesPreserved": True,
        "aacSampleTimeMapping": True,
        "speechGatePassed": None,
        "musicGatePassed": None,
        "inputsStable": True,
    }
    assert facts["accepted"] is False
