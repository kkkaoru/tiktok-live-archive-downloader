"""Preparation cannot accidentally start Metal or claim confinement."""

import json
import subprocess
import sys
import wave
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from replay import isolated_metal_asr_probe as probe
from replay.core import ReplayError
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256


def request(root: Path) -> probe.ProbeRequest:
    item = probe.PinnedInput(root / "synthetic", "a" * 64)
    return probe.ProbeRequest(
        item,
        item,
        item,
        replace(item, sha256=DEFAULT_WEIGHTS_SHA256),
        replace(item, sha256=DEFAULT_CONFIG_SHA256),
        16000,
        32000,
        300,
    )


def test_no_dispatch_even_valid_request(tmp_path: Path) -> None:
    with patch("subprocess.Popen") as popen:
        probe.require_dispatch_supported(request(tmp_path), tmp_path)
    popen.assert_not_called()
    assert probe.RAM_RESERVE_BYTES == 6 * 1024**3


def configuration(root: Path) -> dict[str, object]:
    value = request(root)
    inputs = {
        name: {"path": str(pin.path), "sha256": pin.sha256}
        for name, pin in zip(
            ("pcm", "cpuRaw", "cpuClock", "weights", "config"),
            (value.pcm, value.cpu_raw, value.cpu_receipt, value.weights, value.config),
            strict=True,
        )
    }
    provider = (
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/mlx_whisper"
    )
    code = {
        str(Path(probe.__file__).resolve()): "a" * 64,
        str(Path(probe.__file__).with_name("asr_runtime.py").resolve()): "a" * 64,
    }
    code.update(
        {
            str(provider / name): "a" * 64
            for name in ("transcribe.py", "decoding.py", "load_models.py")
        }
    )
    return {
        "inputs": inputs,
        "codeInputs": code,
        "fp16": True,
        "startSample": 16000,
        "sampleCount": 32000,
        "timeoutSeconds": 300,
        "cpuInputs": {"path": str(root / "cpu-inputs.json"), "sha256": "a" * 64},
    }


def test_load_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "request.json"
    data = configuration(tmp_path)
    path.write_text(json.dumps(data))
    assert probe.load_request(path, verify=False)[0].sample_count == 32000
    for key, value in (("fp16", False), ("codeInputs", {}), ("sampleCount", "1")):
        path.write_text(json.dumps(data | {key: value}))
        with pytest.raises(ReplayError):
            probe.load_request(path, verify=False)


def test_load_verified_mock_only(tmp_path: Path) -> None:
    data = configuration(Path.cwd())
    # Each model role needs its own real path; these remain purely synthetic.
    inputs = data["inputs"]
    assert isinstance(inputs, dict)
    for name in ("weights", "config"):
        inputs[name]["path"] = str(Path.cwd() / name)
    with patch.object(probe, "fingerprint") as digest, patch.object(probe, "read") as read:
        read.side_effect = [
            data,
            {
                "device": "cpu",
                "dtypeContract": "provider-default-fp16-unchanged",
                "clock": "input-pcm-not-projected",
                "inputs": {
                    str(Path.cwd() / "synthetic"): "a" * 64,
                    str(Path.cwd() / "weights"): DEFAULT_WEIGHTS_SHA256,
                    str(Path.cwd() / "config"): DEFAULT_CONFIG_SHA256,
                },
            },
            {"startSample": 16000, "endSample": 48000, "sampleRate": 16000, "rawSHA256": "a" * 64},
        ]
        digest.side_effect = (
            ["a" * 64] * 3 + [DEFAULT_WEIGHTS_SHA256, DEFAULT_CONFIG_SHA256] + ["a" * 64] * 6
        )
        probe.load_request(tmp_path / "request", verify=True)


def test_worker_mock_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "edits/work/probe"
    output.mkdir(parents=True)
    value = request(tmp_path)
    with wave.open(str(value.pcm.path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(48000 * 2))
    for key, val in probe.planned_environment(output, tmp_path).items():
        monkeypatch.setenv(key, val)
        if val != "1":
            Path(val).mkdir(exist_ok=True)
    mx = MagicMock()
    mx.stream.return_value = nullcontext()
    provider = MagicMock()
    provider.transcribe.return_value = {"text": "語", "segments": []}
    with (
        patch.object(probe, "load_request", return_value=(value, {})),
        patch.object(probe, "audit_descriptors") as audit,
        patch(
            "replay.isolated_metal_asr_probe.importlib.import_module", side_effect=[mx, provider]
        ),
    ):
        probe.worker(tmp_path / "request", output)
    assert audit.call_count == 4
    assert provider.transcribe.call_args.kwargs["fp16"] is True


def test_run_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    child = MagicMock()
    child.__enter__.return_value = child
    child.wait.return_value = 0
    child.poll.return_value = 0
    with (
        patch.object(probe, "load_request", return_value=(request(tmp_path), {})),
        patch.object(probe, "fingerprint", return_value="a" * 64),
        patch("replay.isolated_metal_asr_probe.subprocess.Popen", return_value=child) as popen,
        patch.object(probe, "verify_result", return_value={"productionApproved": False}),
        patch("replay.isolated_metal_asr_probe.os.killpg") as kill,
    ):
        assert probe.run(tmp_path / "request", tmp_path / "edits/work/ok").exists()
        assert popen.call_args.args[0][0] == "/usr/bin/sandbox-exec"
        child.wait.side_effect = [subprocess.TimeoutExpired("probe", 1), 0]
        child.poll.return_value = None
        with pytest.raises(subprocess.TimeoutExpired):
            probe.run(tmp_path / "request", tmp_path / "edits/work/timeout")
        kill.assert_called_once()


def test_result_and_denial(tmp_path: Path) -> None:
    (tmp_path / "worker.log").write_text("")
    with (
        patch.object(probe, "load_request", return_value=(request(tmp_path), {})),
        patch.object(probe, "fingerprint", return_value="a" * 64),
        patch.object(probe, "read") as read,
    ):
        read.side_effect = [{"text": "a", "segments": []}, {"text": "b", "segments": []}] + [
            {"externalWritableRegular": []}
        ] * 4
        result = probe.verify_result(tmp_path / "request", tmp_path)
        assert result["textEqual"] is False and result["productionApproved"] is False
        (tmp_path / "worker.log").write_text("Sandbox: deny file-write")
        read.side_effect = [{}, {}]
        with pytest.raises(ReplayError, match="denial"):
            probe.verify_result(tmp_path / "request", tmp_path)


@pytest.mark.parametrize("mode", ["--prepare", "--verify", "--worker", ""])
def test_cli_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", str(tmp_path / "request"), str(tmp_path / "out")] + ([mode] if mode else []),
    )
    with (
        patch.object(probe, "load_request"),
        patch.object(probe, "verify_result"),
        patch.object(probe, "worker"),
        patch.object(probe, "run"),
    ):
        probe.main()


@pytest.mark.parametrize(
    "start,count,timeout", [(-1, 1, 1), (0, 0, 1), (0, 960001, 1), (0, 1, 0), (0, 1, 3601)]
)
def test_limits(tmp_path: Path, start: int, count: int, timeout: int) -> None:
    with pytest.raises(ReplayError, match="bounded"):
        replace(
            request(tmp_path), start_sample=start, sample_count=count, timeout_seconds=timeout
        ).validate(tmp_path)


@pytest.mark.parametrize("digest", ["", "A" * 64, "g" * 64])
def test_bad_pin(tmp_path: Path, digest: str) -> None:
    with pytest.raises(ReplayError, match="pinned"):
        probe.PinnedInput(tmp_path / "input", digest).validate(tmp_path)


def test_paths_and_model(tmp_path: Path) -> None:
    with pytest.raises(ReplayError):
        probe.PinnedInput(Path("relative"), "a" * 64).validate(tmp_path)
    with pytest.raises(ReplayError):
        probe.PinnedInput(tmp_path.parent / "outside", "a" * 64).validate(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path.parent)
    with pytest.raises(ReplayError):
        probe.PinnedInput(link, "a" * 64).validate(tmp_path)
    value = request(tmp_path)
    with pytest.raises(ReplayError, match="large-v3"):
        replace(value, weights=value.pcm).validate(tmp_path)


def test_sandbox_and_environment(tmp_path: Path) -> None:
    profile = probe.sandbox_profile(tmp_path)
    assert "(deny file-write*)" in profile
    assert str(tmp_path) in profile and '(literal "/dev/null")' in profile
    env = probe.planned_environment(tmp_path / "edits/work/probe", tmp_path)
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"]
    for root in (Path("relative"), tmp_path / 'bad"root'):
        with pytest.raises(ReplayError):
            probe.sandbox_profile(root)
    with pytest.raises(ReplayError):
        probe.planned_environment(tmp_path.parent, tmp_path)


def test_comparison_preserves_difference_and_never_approves() -> None:
    original = probe.Recognition("語", (1,), (probe.TimedWord("語", 1.0, 2.0),))
    assert probe.compare(original, original).words_equal
    assert not probe.compare(original, original).approved
    changed = probe.Recognition("語。", (1, 2), (probe.TimedWord("語", 1.001, 2.0),))
    result = probe.compare(original, changed)
    assert not result.text_equal and not result.tokens_equal and not result.words_equal
    assert result.cpu_words == original.words and result.metal_words == changed.words


def test_observation_failures_are_not_suppressed(tmp_path: Path) -> None:
    item = request(tmp_path).pcm
    probe.check_observations(
        before=(item,), after=(item,), external_writable=(), sandbox_denials=()
    )
    with pytest.raises(ReplayError, match="changed"):
        probe.check_observations(before=(item,), after=(), external_writable=(), sandbox_denials=())
    for external, denied in ((("outside",), ()), ((), ("denied",))):
        with pytest.raises(ReplayError, match="Confinement"):
            probe.check_observations(
                before=(item,), after=(item,), external_writable=external, sandbox_denials=denied
            )
