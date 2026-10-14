"""Factory metadata and original mixed producer contracts on synthetic PCM."""

from pathlib import Path

import pytest
from test_mixed_isolated_asr import GPU, Quiet
from test_mixed_isolated_asr import fixture as fixture

from replay import isolated_metal_asr as metal
from replay import metal_asr_request_factory as factory
from replay import mixed_isolated_asr as mixed
from replay.core import ReplayError
from replay.native_invocation import durable, pin
from replay.recordings import record


def test_prepare_and_seal(
    fixture: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, output, cpu = fixture
    old = mixed.load(request)
    monkeypatch.setattr(factory, "DEFAULT_WEIGHTS_SHA256", old.metal.pins[old.metal.weights])
    monkeypatch.setattr(factory, "DEFAULT_CONFIG_SHA256", old.metal.pins[old.metal.config])
    draft = factory.prepare(cpu, old.policy, output.parent / "prepared")
    data = metal._read(draft)
    assert data["ownerCaptures"] is None and data["canDispatch"] is False
    assert (
        metal.load_request(draft.parent / "metal-request.json", verify=False).samples
        == old.metal.samples
    )
    with pytest.raises(ReplayError, match="Occupied"):
        factory.prepare(cpu, old.policy, draft.parent)
    barrier = output.parent / "barrier.json"
    durable(barrier, {"notATerminalClaim": True})
    refs = output.parent / "owners.json"
    owner = dict(old.captures[0])
    owner["stopBarrier"] = pin(barrier)
    durable(refs, {"ownerCaptures": [owner]})
    sealed = factory.seal(draft, refs, output.parent / "sealed.json")
    assert mixed.load(sealed).captures[0]["stopBarrier"] == pin(barrier)
    # Sealing is metadata validation, not a terminal-state or barrier quality approval.
    owner["stopBarrier"] = {"path": str(barrier), "sha256": "0" * 64}
    bad = output.parent / "bad.json"
    durable(bad, {"ownerCaptures": [owner]})
    with pytest.raises(ReplayError):
        factory.seal(draft, bad, output.parent / "bad-sealed.json")


@pytest.mark.parametrize("field", ["device", "sampleRate", "windowSamples", "clock"])
def test_wrong_manifest(fixture: tuple[Path, Path, Path], field: str) -> None:
    request, output, cpu = fixture
    data = metal._read(cpu / "inputs.json")
    data[field] = "invalid"
    (cpu / "inputs.json").write_text(__import__("json").dumps(data))
    with pytest.raises(ReplayError, match="Unsupported"):
        factory.prepare(cpu, mixed.load(request).policy, output)
    assert not output.exists()


def test_trial_and_full_graph(
    fixture: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, output, _ = fixture
    mixed.adopt(request, output, host=Quiet())
    real = mixed.run
    gpu = GPU()

    def run(
        path: Path, dest: Path, *, maximum_new_windows: int | None = None, verify_only: bool = False
    ) -> Path:
        return real(
            path,
            dest,
            runtime=gpu,
            maximum_new_windows=maximum_new_windows,
            verify_only=verify_only,
        )

    monkeypatch.setattr(mixed, "run", run)
    receipt = output.parent / "trial.json"
    factory.trial(request, output, receipt)
    factory.verify_trial(request, output, receipt)
    assert gpu.calls == 1
    for full in (False, True):
        path = factory.workflow(
            request, output, output.parent / f"workflow-{full}.json", complete=full
        )
        from replay.workflow import parse_stages

        assert len(parse_stages(metal._read(path))) == 2
    real(request, output, runtime=gpu)
    complete_receipt = output.parent / "complete-trial.json"
    factory.trial(request, output, complete_receipt)
    assert metal._read(complete_receipt)["collectionComplete"] is True
    partial = record(metal._read(receipt)["result"])
    Path(str(partial["path"])).write_text("{}")
    with pytest.raises(ReplayError):
        factory.verify_trial(request, output, receipt)


@pytest.mark.parametrize(
    "action", ["prepare", "seal", "workflow", "full-workflow", "trial", "verify-trial"]
)
def test_cli(action: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from unittest.mock import Mock

    target = "workflow" if action == "full-workflow" else action.replace("-", "_")
    mock = Mock()
    monkeypatch.setattr(factory, target, mock)
    assert factory.main([action, str(tmp_path), str(tmp_path), str(tmp_path)]) == 0
    assert mock.call_count == 1
