import json
import sys
from pathlib import Path

import pytest

from replay import tracking_batch as batch
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, file_digest
from replay.workflow import parse_stages, run_plan


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic source")
    native = tmp_path / "native"
    native.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "plan=json.loads(Path(sys.argv[3]).read_text())['plan']\n"
        "start=int(plan['startSeconds'])\n"
        "print(json.dumps([{'presentationValue': t, 'presentationTimescale': 1, "
        "'width':16,'height':16,'method':'detection','detectionAttempted':True,"
        "'faces':[{'confidence':0.9,'region':{'x':2,'y':2,'width':8,'height':8}}]} "
        "for t in range(start,start+plan['maximumFrames'])]))\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    clock = tmp_path / "clock.json"
    seeds = tmp_path / "seeds.json"
    eof = tmp_path / "eof.json"
    write(clock, {"pts": [0, 1, 2], "timeBase": "1/1", "lastDuration": 1})
    write(seeds, {"candidateSeedPTS": [0, 1, 2], "timeBase": "1/1"})
    write(eof, {"packets": [{"pts": 2, "duration": 1}]})
    path = tmp_path / "manifest.json"
    write(
        path,
        {
            "version": 1,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": file_digest(native),
            "width": 16,
            "height": 16,
            "redetectIntervalSeconds": 0.5,
            "clock": reference(clock),
            "seeds": reference(seeds),
            "eof": reference(eof),
            "maximumSeconds": "2/1",
        },
    )
    return path


def test_prepare_collect_and_resume(manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "batch"
    plan = batch.prepare(manifest, output)
    assert record(read(output / "window-0000.json")["clock"])["pts"] == [0, 1]
    assert record(read(output / "window-0001.json")["clock"])["pts"] == [1, 2]
    run_plan(plan, state_directory=output / "state")
    response = output / "window-0000" / "response.json"
    before = file_digest(response)
    run_plan(plan, state_directory=output / "state")
    assert file_digest(response) == before
    assert read(output / "window-0000" / "provenance.json")["captionPlacementApproved"] is False
    assert output.stat().st_mode & 0o777 == 0o700
    assert plan.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        batch.prepare(manifest, output)
    response.write_text("changed", encoding="utf-8")
    with pytest.raises(StageError):
        run_plan(plan, state_directory=output / "state")


@pytest.mark.parametrize(
    "changed", ["source.bin", "native", "clock.json", "seeds.json", "eof.json", "manifest.json"]
)
def test_changes_before_first_execution_stop(manifest: Path, tmp_path: Path, changed: str) -> None:
    output = tmp_path / "batch"
    batch.prepare(manifest, output)
    binding = output / "binding.json"
    seal = file_digest(binding)
    (tmp_path / changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        batch.run_window("observe", binding, seal, 0)
    assert not (output / "window-0000").exists()


@pytest.mark.parametrize("changed", ["binding.json", "workflow.json", "window-0000.json"])
def test_changed_preparation_stops(manifest: Path, tmp_path: Path, changed: str) -> None:
    output = tmp_path / "batch"
    batch.prepare(manifest, output)
    binding = output / "binding.json"
    seal = file_digest(binding)
    (output / changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        batch.run_window("observe", binding, seal, 0)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"pts": [], "timeBase": "1/1", "lastDuration": 1},
        {"pts": [0, 1], "timeBase": "1e999999", "lastDuration": 1},
    ],
)
def test_invalid_clock(manifest: Path, tmp_path: Path, value: object) -> None:
    clock = tmp_path / "clock.json"
    write(clock, value)
    write(manifest, read(manifest) | {"clock": reference(clock)})
    with pytest.raises((ValueError, ReplayError)):
        batch.prepare(manifest, tmp_path / "batch")


@pytest.mark.parametrize("packets", [[], [{"pts": 3, "duration": 1}], [{"pts": 2, "duration": 2}]])
def test_wrong_terminal_support(manifest: Path, tmp_path: Path, packets: object) -> None:
    eof = tmp_path / "eof.json"
    write(eof, {"packets": packets})
    write(manifest, read(manifest) | {"eof": reference(eof)})
    with pytest.raises(ValueError, match="packet"):
        batch.prepare(manifest, tmp_path / "batch")


@pytest.mark.parametrize(
    "changes",
    [{"version": True}, {"other": 1}, {"sourceSHA256": "a" * 64}, {"maximumSeconds": "1e999999"}],
)
def test_invalid_manifest(manifest: Path, tmp_path: Path, changes: dict[str, object]) -> None:
    write(manifest, read(manifest) | changes)
    with pytest.raises(ValueError):
        batch.prepare(manifest, tmp_path / "batch")


def test_many_windows_do_not_repeat_other_window_configs(manifest: Path, tmp_path: Path) -> None:
    clock = tmp_path / "clock.json"
    seeds = tmp_path / "seeds.json"
    eof = tmp_path / "eof.json"
    write(clock, {"pts": list(range(80)), "timeBase": "1/1", "lastDuration": 1})
    write(seeds, {"candidateSeedPTS": list(range(80)), "timeBase": "1/1"})
    write(eof, {"packets": [{"pts": 79, "duration": 1}]})
    write(
        manifest,
        read(manifest)
        | {"clock": reference(clock), "seeds": reference(seeds), "eof": reference(eof)},
    )
    output = tmp_path / "many"
    plan = batch.prepare(manifest, output)
    stages = parse_stages(read(plan))
    assert len(stages) == 79
    assert output / "window-0000.json" in stages[0].inputs
    assert output / "window-0001.json" not in stages[0].inputs
    binding = output / "binding.json"
    seal = file_digest(binding)
    (output / "window-0001.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        batch.run_window("observe", binding, seal, 0)
    assert not (output / "window-0000").exists()


def test_evidence_changed_before_prepare(manifest: Path, tmp_path: Path) -> None:
    (tmp_path / "clock.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        batch.prepare(manifest, tmp_path / "batch")


def test_rejects_linked_output(manifest: Path, tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        batch.prepare(manifest, link / "batch")


def test_invalid_action_and_index(manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "batch"
    batch.prepare(manifest, output)
    binding = output / "binding.json"
    with pytest.raises(ValueError, match="action"):
        batch.run_window("unknown", binding, file_digest(binding), 0)
    with pytest.raises(ValueError, match="index"):
        batch.run_window("observe", binding, file_digest(binding), -1)


def test_missing_seed_keeps_raw_evidence(manifest: Path, tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.write_text(
        native.read_text(encoding="utf-8")
        .replace("'method':'detection'", "'method':'missing'")
        .replace("[{'confidence':0.9,'region':{'x':2,'y':2,'width':8,'height':8}}]", "[]"),
        encoding="utf-8",
    )
    write(manifest, read(manifest) | {"nativeSHA256": file_digest(native)})
    output = tmp_path / "batch"
    batch.prepare(manifest, output)
    binding = output / "binding.json"
    with pytest.raises(ValueError, match="seed frame"):
        batch.run_window("observe", binding, file_digest(binding), 0)
    assert (output / "window-0000" / "response.json").exists()
    assert read(output / "window-0000" / "provenance.json")["captionPlacementApproved"] is False


def test_unseeded_cli_keeps_all_missing_contiguous_clock(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "native"
    native.write_text(
        native.read_text(encoding="utf-8")
        .replace("'method':'detection'", "'method':'missing'")
        .replace("[{'confidence':0.9,'region':{'x':2,'y':2,'width':8,'height':8}}]", "[]"),
        encoding="utf-8",
    )
    seeds = tmp_path / "seeds.json"
    write(seeds, {"candidateSeedPTS": [], "timeBase": "1/1"})
    write(
        manifest, read(manifest) | {"nativeSHA256": file_digest(native), "seeds": reference(seeds)}
    )
    output = tmp_path / "unseeded"
    monkeypatch.setattr(
        sys, "argv", ["tracking-batch", "prepare", str(manifest), str(output), "--unseeded"]
    )
    batch.main()
    assert record(read(output / "window-0000.json")["clock"])["pts"] == [0, 1]
    assert record(read(output / "window-0001.json")["clock"])["pts"] == [2]
    assert read(output / "binding.json")["seedPolicy"] == "unseeded-observe"
    run_plan(output / "workflow.json", state_directory=output / "state")
    response = json.loads((output / "window-0001" / "response.json").read_text(encoding="utf-8"))
    assert response == [
        {
            "presentationValue": 2,
            "presentationTimescale": 1,
            "width": 16,
            "height": 16,
            "method": "missing",
            "detectionAttempted": True,
            "faces": [],
        }
    ]
    assert read(output / "window-0001" / "provenance.json")["captionPlacementApproved"] is False
    run_plan(output / "workflow.json", state_directory=output / "state")


def test_empty_seeds_need_explicit_unseeded_policy(manifest: Path, tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.json"
    write(seeds, {"candidateSeedPTS": [], "timeBase": "1/1"})
    write(manifest, read(manifest) | {"seeds": reference(seeds)})
    with pytest.raises(ValueError, match="candidate seeds"):
        batch.prepare(manifest, tmp_path / "batch")


def test_unknown_unseeded_policy_is_rejected(manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "batch"
    batch.prepare(manifest, output, unseeded=True)
    binding = output / "binding.json"
    write(binding, read(binding) | {"seedPolicy": "assume-detected"})
    with pytest.raises(ValueError, match="Invalid tracking binding"):
        batch.run_window("observe", binding, file_digest(binding), 0)


def test_cli_prepare_observe_verify(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cli-batch"
    monkeypatch.setattr(sys, "argv", ["tracking-batch", "prepare", str(manifest), str(output)])
    batch.main()
    binding = output / "binding.json"
    monkeypatch.setattr(
        sys, "argv", ["tracking-batch", "observe", str(binding), file_digest(binding), "0"]
    )
    batch.main()
    monkeypatch.setattr(
        sys, "argv", ["tracking-batch", "verify", str(binding), file_digest(binding), "0"]
    )
    batch.main()
    assert (output / "window-0000" / "provenance.json").exists()


@pytest.mark.parametrize(
    "change", [{"clock": {}}, {"clock": {"path": "relative", "sha256": "a" * 64}}]
)
def test_invalid_evidence_reference(
    manifest: Path, tmp_path: Path, change: dict[str, object]
) -> None:
    write(manifest, read(manifest) | change)
    with pytest.raises(ValueError):
        batch.prepare(manifest, tmp_path / "batch")


def test_workflow_budget_not_widened(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "MAX_PLAN_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        batch.prepare(manifest, tmp_path / "batch")
    assert (tmp_path / "batch" / "window-0000.json").exists()
    assert not (tmp_path / "batch" / "workflow.json").exists()


def test_partial_output_is_not_overwritten(manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "batch"
    batch.prepare(manifest, output)
    partial = output / "window-0000"
    partial.mkdir()
    with pytest.raises(FileExistsError):
        batch.run_window(
            "observe", output / "binding.json", file_digest(output / "binding.json"), 0
        )
    assert partial.exists()
