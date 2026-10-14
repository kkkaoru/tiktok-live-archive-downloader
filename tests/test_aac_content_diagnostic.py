"""Small bounded PCM fixtures; no recordings, desktop, model or network use."""

import hashlib
import json
import os
import shutil
import subprocess
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import aac_content_diagnostic as content
from replay import aac_presentation_acceptance as aac
from replay.core import ReplayError
from replay.recordings import record


def ref(path: Path, data: bytes) -> aac.Reference:
    path.write_bytes(data)
    return aac.Reference(path, hashlib.sha256(data).hexdigest())


def doc(path: Path, data: dict[str, object]) -> aac.Reference:
    return ref(path, json.dumps(data).encode())


def write_wav(path: Path, values: NDArray[np.int16], rate: int) -> None:
    with wave.open(str(path), "wb") as target:
        target.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(values.astype("<i2").tobytes())


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> content.Request:
    monkeypatch.setattr(aac, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    source_path = work / "voice.wav"
    write_wav(
        source_path,
        np.random.default_rng(12).integers(-4000, 4000, (64000, 2), dtype=np.int16),
        8000,
    )
    source = aac.SourcePCM(
        aac.Reference(source_path, hashlib.sha256(source_path.read_bytes()).hexdigest()),
        8000,
        64000,
    )
    candidate = ref(work / "candidate.mp4", b"synthetic-current-container")
    tool = ref(work / "ffmpeg", b"trusted-fixture")
    original_ref = doc(work / "original.json", {"fixture": True})
    original = aac.ObservationRequest(
        original_ref,
        candidate,
        aac.FileIdentity.from_stat(candidate.path.stat()),
        original_ref,
        original_ref,
        (tool, tool),
        (),
        source,
        None,
        None,
    )
    monkeypatch.setattr(aac, "load_observation_request", lambda reference: original)
    monkeypatch.setattr(
        aac,
        "_original_provenance",
        lambda request: (
            {
                "streams": {
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "sample_rate": "8000",
                            "time_base": "1/8000",
                            "channels": 2,
                        }
                    ]
                }
            },
            {},
        ),
    )
    composition = doc(
        work / "composition.json",
        {
            "version": 1,
            "kind": "video-composition-timing-observation",
            "newDecode": False,
            "wholeSHAFreshlyVerified": False,
            "request": original_ref.json(),
            "candidate": candidate.json(),
            "expectedIdentity": original.identity.json(),
            "aacAssessment": {
                "trackEditMappingAccepted": True,
                "packetDecodedPresentationAccepted": True,
                "decodedStart": "0",
                "decodedEnd": "79/10",
                "decodedSamples": 63200,
                "source": source.reference.json(),
            },
        },
    )
    request = doc(
        work / "request.json",
        {
            "version": 1,
            "kind": "aac-content-diagnostic-request",
            "presentationRequest": original_ref.json(),
            "composition": composition.json(),
            "sourceIdentity": aac.FileIdentity.from_stat(source_path.stat()).json(),
            "comparisonRate": 8000,
            "windowSeconds": 2,
        },
    )
    return content.load(request)


def fake_run(command: list[str], directory: Path, name: str) -> None:
    rate = 8000
    first = 0
    count = 16000
    if name == "resample":
        with wave.open(str(directory / "source.wav"), "rb") as source:
            payload = source.readframes(source.getnframes())
        output = directory / "source-comparison.wav"
    else:
        filter_text = command[command.index("-af") + 1]
        first = int(filter_text.split("start_pts=")[1].split(":")[0])
        count = min(16000, 63200 - first)
        with wave.open(str(directory.parent / "voice.wav"), "rb") as source:
            source.setpos(first)
            payload = source.readframes(count)
        output = directory / "candidate.wav"
    write_wav(output, np.frombuffer(payload, dtype="<i2").reshape(count, 2), rate)
    (directory / f"{name}.stdout").write_bytes(b"")
    (directory / f"{name}.stderr").write_text(
        f"[Parsed_ashowinfo_0] n:0 pts:{first} pts_time:0 fmt:s16 channels:2 "
        f"chlayout:stereo rate:{rate} nb_samples:{count}\n",
        encoding="utf-8",
    )
    (directory / f"{name}.process.json").write_text(
        json.dumps({"command": command, "returncode": 0, "seconds": 0.1}), encoding="utf-8"
    )


def test_real_boundaries_and_readonly_reuse(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "_run", fake_run)
    output = setup.reference.path.parent / "result.json"
    content.collect(setup, output)
    before = output.stat()
    content.verify(setup, output)
    report = record(json.loads(output.read_bytes()))
    rows = report["windows"]
    assert isinstance(rows, list)
    assert record(rows[0])["candidateStartPTS"] == 0
    assert record(rows[1])["candidateStartPTS"] == 24000
    assert record(rows[2])["candidateSamples"] == 15200
    assert record(record(rows[2])["comparison"])["missingSourceSamples"] == 800
    assert report["qualityApproved"] is False
    assert report["syncApproved"] is False
    assert report["deliveryApproved"] is False
    assert output.stat() == before
    with pytest.raises(FileExistsError, match="occupied"):
        content.collect(setup, output)


@pytest.mark.parametrize("rate", [16000, 48000, 96000])
def test_different_rates_project_exact_source_indices(setup: content.Request, rate: int) -> None:
    request = replace(setup, rate=rate, decoded_end=rate * 8 - 80)
    result = content.windows(request)
    assert result[0].source_start == 0
    assert result[1].source_start == 24000
    assert result[2].source_start == 48000
    assert result[2].actual_count == rate * 2 - 80


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("kind", "other"),
        ("comparisonRate", 7999),
        ("comparisonRate", 192001),
        ("windowSeconds", 3),
        ("sourceIdentity", [1]),
    ],
)
def test_request_fields_rejected(setup: content.Request, field: str, value: object) -> None:
    row = record(json.loads(setup.reference.path.read_bytes()))
    row[field] = value
    reference = doc(setup.reference.path.parent / "changed-request.json", row)
    with pytest.raises((ReplayError, ValueError)):
        content.load(reference)


@pytest.mark.parametrize(
    "field,value",
    [
        ("decodedStart", "1/10"),
        ("decodedEnd", "1/3"),
        ("decodedSamples", 20),
        ("trackEditMappingAccepted", False),
        ("packetDecodedPresentationAccepted", False),
        ("source", None),
    ],
)
def test_unknown_or_changed_presentation_rejected(
    setup: content.Request, field: str, value: object
) -> None:
    report = record(json.loads(setup.composition.path.read_bytes()))
    assessment = record(report["aacAssessment"])
    assessment[field] = value
    report["aacAssessment"] = assessment
    composition = doc(setup.composition.path.parent / "changed-composition.json", report)
    request = record(json.loads(setup.reference.path.read_bytes()))
    request["composition"] = composition.json()
    with pytest.raises(ReplayError):
        content.load(doc(setup.reference.path.parent / "changed-request.json", request))


def test_wrong_candidate_context_rejected(setup: content.Request) -> None:
    report = record(json.loads(setup.composition.path.read_bytes()))
    report["candidate"] = None
    composition = doc(setup.composition.path.parent / "wrong-context.json", report)
    request = record(json.loads(setup.reference.path.read_bytes()))
    request["composition"] = composition.json()
    with pytest.raises(ReplayError, match="bound"):
        content.load(doc(setup.reference.path.parent / "changed-request.json", request))


@pytest.mark.parametrize("samples,end", [(16000, 16000), (64000, 65000), (64000, 50000)])
def test_invalid_window_extents(setup: content.Request, samples: int, end: int) -> None:
    source = setup.original.source
    assert source is not None
    original = replace(setup.original, source=replace(source, samples=samples))
    request = replace(setup, original=original, rate=48000, decoded_end=end * 6)
    with pytest.raises(ReplayError):
        content.windows(request)


def test_missing_source_rejected(setup: content.Request) -> None:
    with pytest.raises(ReplayError):
        content.windows(replace(setup, original=replace(setup.original, source=None)))


def test_command_retains_native_pts_and_bounded_preroll(setup: content.Request) -> None:
    window = content.windows(setup)[1]
    vectors = content.commands(setup, window, setup.reference.path.parent)
    assert vectors["decode-window"][vectors["decode-window"].index("-ss") + 1] == "2"
    assert (
        vectors["decode-window"][vectors["decode-window"].index("-af") + 1]
        == "atrim=start_pts=24000:end_pts=40000,ashowinfo"
    )
    assert "-copyts" in vectors["decode-window"]
    assert "-start_at_zero" not in vectors["decode-window"]
    assert "aresample" not in " ".join(vectors["decode-window"])
    assert (
        vectors["resample"][vectors["resample"].index("-af") + 1]
        == "aresample=8000:async=0,ashowinfo"
    )


@pytest.mark.parametrize(
    "text",
    [
        b"",
        b"[Parsed_ashowinfo_0] missing",
        b"[Parsed_ashowinfo_0] n:1 pts:0 channels:2 rate:8000 nb_samples:10",
        b"[Parsed_ashowinfo_0] n:0 pts:1 channels:2 rate:8000 nb_samples:10",
        b"[Parsed_ashowinfo_0] n:0 pts:0 channels:1 rate:8000 nb_samples:10",
        b"[Parsed_ashowinfo_0] n:0 pts:0 channels:2 rate:9000 nb_samples:10",
        b"[Parsed_ashowinfo_0] n:0 pts:0 channels:2 rate:8000 nb_samples:0",
        b"[Parsed_ashowinfo_0] n:0 pts:0 channels:2 rate:8000 nb_samples:65537",
        b"[Parsed_ashowinfo_0] n:0 pts:0 channels:2 rate:8000 nb_samples:11",
    ],
)
def test_clipped_shifted_unknown_or_gapped_trace_rejected(text: bytes) -> None:
    with pytest.raises(ReplayError):
        content.trace(text, rate=8000, start=0, count=10)


def test_trace_keeps_negative_preroll_outside_selected_window() -> None:
    content.trace(
        b"ordinary input preroll pts:-64\n"
        b"[Parsed_ashowinfo_0] n:0 pts:32000 channels:2 rate:8000 nb_samples:5\n"
        b"[Parsed_ashowinfo_0] n:1 pts:32005 channels:2 rate:8000 nb_samples:5\n",
        rate=8000,
        start=32000,
        count=10,
    )


def test_large_log_rejected() -> None:
    with pytest.raises(ReplayError, match="budget"):
        content.trace(bytes(content.MAX_LOG_BYTES + 1), rate=8000, start=0, count=10)


def test_shift_is_comparison_instrument_only() -> None:
    source = np.random.default_rng(9).normal(0, 0.1, (16000, 2))
    candidate = np.concatenate((np.zeros((17, 2)), source[:-17]))
    result = content.compare(source, candidate, rate=8000)
    correspondence = record(result["correspondence"])
    assert correspondence["status"] == "matched"
    assert correspondence["candidateMinusSourceSamples"] == 17
    assert correspondence["appliedToProduct"] is False
    assert result["missingSourceSamples"] == 0


def test_silent_window_is_not_speech_or_sync_approval() -> None:
    result = content.compare(np.zeros((16000, 2)), np.zeros((15900, 2)), rate=8000)
    assert record(result["correspondence"])["reason"] == "reference-low-energy"
    assert record(result["missingSourceLevel"])["exactZeroFraction"] == 1.0
    assert record(result["missingSourceLevel"])["speechClassified"] is False


def test_noisy_unrelated_window_records_failure_without_adjustment() -> None:
    rng = np.random.default_rng(9)
    result = content.compare(
        rng.normal(0, 0.1, (16000, 2)), rng.normal(0, 0.1, (16000, 2)), rate=8000
    )
    assert record(result["correspondence"])["reason"] == "low-correlation"
    assert result["silenceOrSpeechAssumed"] is False


@pytest.mark.parametrize("count", [0, 7999, 16001])
def test_missing_or_excessive_comparison_window(count: int) -> None:
    with pytest.raises(ReplayError):
        content.compare(np.zeros((16000, 2)), np.zeros((count, 2)), rate=8000)


@pytest.mark.parametrize("values", [np.empty((0, 2)), np.full((5, 2), np.nan)])
def test_empty_or_nonfinite_signal_rejected(values: content.PCM) -> None:
    with pytest.raises(ReplayError):
        content.level(values)


def test_source_mutation_stops_before_child(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = setup.original.source
    assert source is not None
    source.reference.path.write_bytes(b"changed")
    monkeypatch.setattr(content, "_run", lambda *args: pytest.fail("child must not run"))
    with pytest.raises(ReplayError, match="identity"):
        content.collect(setup, setup.reference.path.parent / "result.json")


def test_typed_request_cannot_hide_changed_rate(setup: content.Request) -> None:
    with pytest.raises(ReplayError, match="Typed"):
        content._guard(replace(setup, rate=16000))


def test_missing_or_altered_result_cannot_be_reused(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "_run", fake_run)
    output = setup.reference.path.parent / "result.json"
    with pytest.raises(FileNotFoundError):
        content.verify(setup, output)
    content.collect(setup, output)
    row = record(json.loads(output.read_bytes()))
    row["syncApproved"] = True
    output.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ReplayError, match="differs"):
        content.verify(setup, output)


def test_changed_process_clock_is_not_reused(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "_run", fake_run)
    output = setup.reference.path.parent / "result.json"
    content.collect(setup, output)
    (output.parent / "tail/decode-window.stderr").write_bytes(b"no native clock")
    with pytest.raises(ReplayError, match="extent"):
        content.verify(setup, output)


def test_wrong_process_argv_is_rejected(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "_run", fake_run)
    output = setup.reference.path.parent / "result.json"
    content.collect(setup, output)
    path = output.parent / "first/resample.process.json"
    row = record(json.loads(path.read_bytes()))
    row["command"] = ["untrusted"]
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ReplayError, match="provenance"):
        content.verify(setup, output)


def test_partial_outputs_are_retained_without_retry(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        content, "_run", lambda *args: (_ for _ in ()).throw(ReplayError("failed child"))
    )
    output = setup.reference.path.parent / "result.json"
    with pytest.raises(ReplayError, match="failed child"):
        content.collect(setup, output)
    assert (output.parent / "first/source.wav").is_file()
    assert not output.exists()
    with pytest.raises(FileExistsError):
        content.collect(setup, output)


def test_artifact_link_and_budget_rejected(tmp_path: Path) -> None:
    original = tmp_path / "actual"
    original.write_bytes(b"data")
    link = tmp_path / "linked"
    link.symlink_to(original)
    with pytest.raises(ReplayError, match="links"):
        content._pin(link)
    original.write_bytes(bytes(content.MAX_VIEW_BYTES + 1))
    with pytest.raises(ReplayError, match="budget"):
        content._pin(original)


def test_wrong_pcm_header_rejected(tmp_path: Path) -> None:
    path = tmp_path / "small.wav"
    write_wav(path, np.zeros((20, 2), dtype=np.int16), 16000)
    with pytest.raises(ReplayError, match="format"):
        content.pcm(path, rate=8000, count=20)


def test_original_wav_header_rejected_before_convert(setup: content.Request) -> None:
    source = setup.original.source
    assert source is not None
    write_wav(source.reference.path, np.zeros((64000, 2), dtype=np.int16), 16000)
    request = replace(
        setup, source_identity=aac.FileIdentity.from_stat(source.reference.path.stat())
    )
    with pytest.raises(ReplayError, match="header"):
        content._source_view(
            request, content.windows(request)[0], setup.reference.path.parent / "view.wav"
        )


def test_child_failure_records_actual_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(ReplayError, match="failed"):
        content._run(["tool"], tmp_path, "failed")
    assert record(json.loads((tmp_path / "failed.process.json").read_bytes()))["returncode"] == 7


def test_child_timeout_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(command, 120)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        content._run(["tool"], tmp_path, "timeout")
    assert (tmp_path / "timeout.stderr").exists()
    assert not (tmp_path / "timeout.process.json").exists()


def test_real_ffmpeg_synthetic_window_retains_pts_and_resamples_once(
    setup: content.Request, tmp_path: Path
) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.fail("The repository FFmpeg fixture dependency is missing")
    tool = aac.Reference(
        Path(executable).resolve(), hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    )
    source = setup.original.source
    assert source is not None
    original = replace(setup.original, candidate=source.reference, tools=(tool, tool))
    request = replace(setup, original=original)
    window = content.windows(request)[1]
    directory = tmp_path / "real"
    directory.mkdir()
    content._source_view(request, window, directory / "source.wav")
    vectors = content.commands(request, window, directory)
    content._run(vectors["decode-window"], directory, "decode-window")
    content.trace(
        (directory / "decode-window.stderr").read_bytes(), rate=8000, start=24000, count=16000
    )
    assert content.pcm(directory / "candidate.wav", rate=8000, count=16000).shape == (16000, 2)
    converted = replace(request, rate=48000, decoded_end=379200)
    resample = content.commands(converted, content.windows(converted)[1], directory)["resample"]
    content._run(resample, directory, "resample")
    content.trace((directory / "resample.stderr").read_bytes(), rate=48000, start=0, count=96000)
    assert content.pcm(directory / "source-comparison.wav", rate=48000, count=96000).shape == (
        96000,
        2,
    )


def test_cli_success_and_readonly_verify(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "_run", fake_run)
    output = setup.reference.path.parent / "result.json"
    assert (
        content.main(
            ["observe", str(setup.reference.path), str(output), "--sha256", setup.reference.sha256]
        )
        == 0
    )
    assert (
        content.main(
            ["verify", str(setup.reference.path), str(output), "--sha256", setup.reference.sha256]
        )
        == 0
    )


def test_cli_missing_request_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        content.main(
            ["observe", str(tmp_path / "missing"), str(tmp_path / "out"), "--sha256", "a" * 64]
        )
    assert error.value.code == 1


def test_source_fd_change_is_rejected(setup: content.Request) -> None:
    wrong = replace(setup, source_identity=replace(setup.source_identity, inode=1))
    with pytest.raises(ReplayError, match="FD"):
        content._source_view(
            wrong, content.windows(wrong)[0], setup.reference.path.parent / "view.wav"
        )


def test_wrong_audio_clock_guard(setup: content.Request, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        aac, "_original_provenance", lambda request: ({"streams": {"streams": []}}, {})
    )
    with pytest.raises(ReplayError, match="one actual"):
        content._guard(setup)


def test_artifact_race_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"data")
    real_lstat = Path.lstat

    def changed(item: Path) -> os.stat_result:
        if item == path:
            return real_lstat(item.parent)
        return real_lstat(item)

    monkeypatch.setattr(Path, "lstat", changed)
    with pytest.raises(ReplayError):
        content._pin(path)


def test_public_prepare_pins_actual_identity_without_whole_media_digest(
    setup: content.Request,
) -> None:
    output = setup.reference.path.parent / "prepared.json"
    assert (
        content.main(
            [
                "prepare",
                str(setup.original.reference.path),
                str(output),
                "--sha256",
                setup.original.reference.sha256,
                "--composition",
                str(setup.composition.path),
                "--composition-sha256",
                setup.composition.sha256,
            ]
        )
        == 0
    )
    prepared = content.load(content._pin(output))
    assert prepared.source_identity == setup.source_identity
    assert prepared.rate == 8000
    assert prepared.decoded_end == 63200
    with pytest.raises(FileExistsError):
        content.prepare(setup.original.reference, setup.composition, output)


def test_prepare_needs_actual_composition(setup: content.Request) -> None:
    with pytest.raises(SystemExit) as error:
        content.main(
            [
                "prepare",
                str(setup.original.reference.path),
                str(setup.reference.path.parent / "prepared.json"),
                "--sha256",
                setup.original.reference.sha256,
            ]
        )
    assert error.value.code == 1


def test_fractional_original_projection_is_rejected(setup: content.Request) -> None:
    source = setup.original.source
    assert source is not None
    original = replace(setup.original, source=replace(source, samples=64001))
    with pytest.raises(ReplayError, match="fractional"):
        content.windows(replace(setup, original=original, rate=44100, decoded_end=300000))


@pytest.mark.parametrize("field", ["request", "expectedIdentity", "kind", "newDecode"])
def test_unrelated_composition_contract_is_rejected(setup: content.Request, field: str) -> None:
    report = record(json.loads(setup.composition.path.read_bytes()))
    report[field] = None
    composition = doc(setup.composition.path.parent / "bad-context.json", report)
    request = record(json.loads(setup.reference.path.read_bytes()))
    request["composition"] = composition.json()
    with pytest.raises(ReplayError, match="bound"):
        content.load(doc(setup.reference.path.parent / "other-request.json", request))


def test_fractional_missing_boundary_keeps_partial_edge_sample() -> None:
    window = content.Window("tail", 229765410, 88200, 250084800, 96000, 94656)
    assert content.missing_native_range(window, source_rate=44100, comparison_rate=48000) == (
        Fraction(434826, 5),
        86965,
        86966,
    )


def test_different_fractional_missing_boundary_keeps_partial_edge_sample() -> None:
    window = content.Window("tail", 282043020, 88200, 306985600, 96000, 94464)
    assert content.missing_native_range(window, source_rate=44100, comparison_rate=48000) == (
        Fraction(433944, 5),
        86788,
        86789,
    )


@pytest.mark.parametrize("actual", [0, 96000, 96001])
def test_missing_native_extent_cannot_be_invented(actual: int) -> None:
    window = content.Window("tail", 0, 88200, 0, 96000, actual)
    with pytest.raises(ReplayError):
        content.missing_native_range(window, source_rate=44100, comparison_rate=48000)


def test_native_range_rejects_wrong_rate_or_support() -> None:
    window = content.Window("tail", 0, 10, 0, 96000, 94656)
    with pytest.raises(ReplayError, match="support"):
        content.missing_native_range(window, source_rate=44100, comparison_rate=48000)


def test_native_tail_uses_pinned_views_without_another_child(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "_run", fake_run)
    output = setup.reference.path.parent / "result.json"
    content.collect(setup, output)
    parent = content._pin(output)
    monkeypatch.setattr(content, "_run", lambda *args: pytest.fail("tail cannot run a decoder"))
    result = content.native_tail(setup, parent=parent)
    assert result["relativeSourceBoundary"] == "15200"
    assert result["coveringStartSampleFloor"] == 63200
    assert result["whollyUnsupportedStartSampleCeil"] == 63200
    assert record(result["coveringNativeLevel"])["samples"] == 800
    assert result["newDecode"] is False
    assert result["newResample"] is False
    assert result["speechClassified"] is False
    args = [
        str(setup.reference.path),
        str(output.parent / "tail-native.json"),
        "--sha256",
        setup.reference.sha256,
        "--parent",
        str(parent.path),
        "--parent-sha256",
        parent.sha256,
    ]
    assert content.main(["tail-observe", *args]) == 0
    assert content.main(["tail-verify", *args]) == 0
    (output.parent / "tail-native.json").write_bytes(b"{}")
    with pytest.raises(SystemExit):
        content.main(["tail-verify", *args])


def test_native_tail_rejects_changed_parent(
    setup: content.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "_run", fake_run)
    output = setup.reference.path.parent / "result.json"
    content.collect(setup, output)
    data = record(json.loads(output.read_bytes()))
    data["comparisonRate"] = 16000
    parent = doc(output.parent / "bad-parent.json", data)
    with pytest.raises(ReplayError, match="parent"):
        content.native_tail(setup, parent=parent)


def test_native_tail_cli_requires_parent(setup: content.Request) -> None:
    with pytest.raises(SystemExit):
        content.main(
            [
                "tail-observe",
                str(setup.reference.path),
                str(setup.reference.path.parent / "tail-native.json"),
                "--sha256",
                setup.reference.sha256,
            ]
        )
