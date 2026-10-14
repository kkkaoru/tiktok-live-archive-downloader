"""Separate MPS gate: actual synthetic PCM/benchmark evidence and refused bindings."""

import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_separation_device_benchmark import case as case
from test_separation_device_benchmark import completed

from replay import mps_separation_acceptance as gate
from replay import separation_device_benchmark as benchmark
from replay.core import ReplayError
from replay.native_invocation import durable, identity, pin, read
from replay.recordings import record


@pytest.fixture
def ready(case: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    baseline, root = case
    plan = read(baseline)
    source = Path(str(record(plan["source"])["path"]))
    with wave.open(str(source), "wb") as stream:
        stream.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(
            b"\x01\x00\x02\x00" * 64 + b"\x03\x00\x04\x00" * 64 + b"\x07\x00\x08\x00" * 64
        )
    plan["source"] = {"path": str(source), "identity": identity(source)}
    baseline.write_text(json.dumps(plan))
    durable(
        root / "workflow.json",
        {
            "stages": [
                {
                    "command": [
                        "python",
                        "-m",
                        "replay.separation_device_benchmark",
                        "run",
                        str(baseline),
                        str(root / "cpu"),
                        "--device",
                        "cpu",
                    ]
                }
            ]
        },
    )
    metrics = gate.characteristics(baseline, (0, 64, 128), root / "metrics.json")
    prepared = gate.prepare(baseline, metrics, root / "more")
    prepared_clips = read(prepared)["clips"]
    assert isinstance(prepared_clips, list)
    assert len(prepared_clips) == 2
    selected = read(metrics)["selected"]
    assert isinstance(selected, list)
    plans = [baseline, root / "more/clip-01/request.json", root / "more/clip-02/request.json"]
    clips: list[dict[str, object]] = []
    for index, request in enumerate(plans):
        for role in ("cpu", "mps"):
            completed(request, request.parent / role, role)
        comparison = benchmark.compare(
            request,
            request.parent / "cpu",
            request.parent / "mps",
            request.parent / "comparison.json",
        )
        clip: dict[str, object] = {
            "comparison": pin(comparison),
            "characteristics": selected[index],
            "characterEvidence": pin(metrics),
        }
        if index == 0:
            asr_request = root / "asr-request.json"
            durable(asr_request, {"synthetic": True})
            asr = root / "asr.json"
            durable(
                asr,
                {
                    "request": pin(asr_request),
                    "cpuSeparationWorker": pin(root / "cpu/worker.json"),
                    "mpsSeparationWorker": pin(root / "mps/worker.json"),
                    **{
                        key: True
                        for key in (
                            "rawTextEqual",
                            "orderedSegmentTextEqual",
                            "orderedWordTextEqual",
                            "tokensEqual",
                            "wordTimingExact",
                            "allNativeCollectionsVerified",
                            "sameModelProviderContext",
                        )
                    },
                },
            )
            clip["asrComparison"] = pin(asr)
        clips.append(clip)
    monkeypatch.setattr(gate, "_verify_asr", lambda request, receipt: read(receipt))
    acceptance = root / "gate.json"
    durable(acceptance, {"version": gate.VERSION, "clips": clips})
    return acceptance, root


def test_three_clips_one_asr_and_readonly_verify(ready: tuple[Path, Path]) -> None:
    request, root = ready
    receipt = gate.assess(request, root / "accepted.json")
    result = gate.verify_acceptance(request, receipt)
    assert result["mpsGatePassed"] is True and result["cpuExactGateChanged"] is False
    assert result["asrClips"] == 1
    (root / "cpu/vocals.npy").write_bytes(b"changed")
    with pytest.raises(ReplayError):
        gate.verify_acceptance(request, receipt)


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate",
        "missing-asr",
        "timing",
        "text",
        "two-clips",
        "metrics",
        "wrong-worker",
        "legacy-policy",
        "forged-score",
    ],
)
def test_gate_refusals(ready: tuple[Path, Path], mode: str) -> None:
    request, root = ready
    data = read(request)
    clips = data["clips"]
    assert isinstance(clips, list)
    first = record(clips[0])
    if mode == "duplicate":
        clips[1] = clips[0]
    elif mode == "missing-asr":
        first.pop("asrComparison")
        clips[0] = first
    elif mode in {"timing", "text", "wrong-worker"}:
        asr = read(root / "asr.json")
        asr[
            "wordTimingExact"
            if mode == "timing"
            else "rawTextEqual"
            if mode == "text"
            else "cpuSeparationWorker"
        ] = False
        (root / "asr.json").write_text(json.dumps(asr))
        first["asrComparison"] = pin(root / "asr.json")
        clips[0] = first
    elif mode == "two-clips":
        data["clips"] = clips[:2]
    elif mode == "legacy-policy":
        data["version"] = 1
    elif mode == "metrics":
        first["characteristics"] = {"rms": 999}
        clips[0] = first
    else:
        comp = read(root / "comparison.json")
        comp["sdrDB"] = 999
        (root / "comparison.json").write_text(json.dumps(comp))
        first["comparison"] = pin(root / "comparison.json")
        clips[0] = first
    request.write_text(json.dumps(data))
    if mode in {"timing", "text"}:
        receipt = gate.assess(request, root / "rejected.json")
        assert read(receipt)["mpsGatePassed"] is False
        with pytest.raises(ReplayError):
            gate.verify_acceptance(request, receipt)
    else:
        with pytest.raises(ReplayError):
            gate.assess(request, root / "rejected.json")


def test_original_verifier_seam(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    operation = MagicMock(return_value={"actualContract": True})
    monkeypatch.setitem(
        sys.modules, "replay.separation_vocal_asr_compare", SimpleNamespace(verify=operation)
    )
    assert gate._verify_asr(tmp_path, tmp_path) == {"actualContract": True}
    operation.assert_called_once_with(tmp_path, tmp_path)


@pytest.mark.parametrize("ratio,accepted", [(2.999, False), (3.0, True)])
def test_each_clip_actual_wall_gate(ready: tuple[Path, Path], ratio: float, accepted: bool) -> None:
    request, root = ready
    resources = read(root / "cpu/resources.json")
    resources["wallSeconds"] = ratio * float(str(read(root / "mps/resources.json")["wallSeconds"]))
    (root / "cpu/resources.json").write_text(json.dumps(resources))
    comparison = benchmark.compare(
        root / "request.json", root / "cpu", root / "mps", root / "comparison-new.json"
    )
    data = read(request)
    clips = data["clips"]
    assert isinstance(clips, list)
    first = record(clips[0])
    first["comparison"] = pin(comparison)
    clips[0] = first
    request.write_text(json.dumps(data))
    receipt = gate.assess(request, root / "wall-gate.json")
    assert read(receipt)["mpsGatePassed"] is accepted


@pytest.mark.parametrize("action", ["measure", "prepare", "assess", "verify"])
def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    name = {"measure": "characteristics", "verify": "verify_acceptance"}.get(action, action)
    called = MagicMock()
    monkeypatch.setattr(gate, name, called)
    assert (
        gate.main(
            [action, str(tmp_path), str(tmp_path), "--starts", "0,1,2", "--metrics", str(tmp_path)]
        )
        == 0
    )
    assert called.call_count == 1
