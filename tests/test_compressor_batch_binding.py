"""Submission provenance tests; no application or real media calls."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_compressor_batch import SUBMITTED, SUCCESS, FakeCompressor, job_file, make_batch
from test_compressor_batch import synthetic_runtime as synthetic_runtime

from replay import compressor_batch as batch
from replay.core import ReplayError


def test_receipt_is_persisted_before_submit_and_id_before_postcheck(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state/alpha-0000.json"

    def runner(arguments: Sequence[str]) -> str:
        assert "-jobpath" in arguments
        saved = json.loads(state.read_text())
        assert saved["status"] == "submitting"
        assert saved["inputs"][0]["sha256"] == batch._pin(tmp_path / "part.moti")["sha256"]
        (tmp_path / "part.moti").write_text("changed")
        return SUBMITTED

    worker = batch.Batch(tmp_path / "state", Path("/Compressor"), "test", runner, lambda _s: None)
    with pytest.raises(ReplayError, match="input changed"):
        batch.run(batch.read_jobs(jobs), worker, in_flight=1)
    saved = json.loads(state.read_text())
    assert saved["jobID"] == "J1"
    assert saved["status"] == "submitted"
    assert not output.exists()
    with pytest.raises(ReplayError, match="input changed"):
        batch.run(batch.read_jobs(jobs), worker, in_flight=1)


def test_pending_cycles_do_not_rehash_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, output = job_file(tmp_path)
    original = batch._pin
    paths: list[Path] = []

    def pin(path: Path) -> dict[str, object]:
        paths.append(path)
        return original(path)

    monkeypatch.setattr(batch, "_pin", pin)
    fake = FakeCompressor(output, ["Pending", "Processing", "Processing", "Successful"])
    assert batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=1) == 1
    assert paths.count(tmp_path / "part.moti") == 2
    assert paths.count(tmp_path / "alpha.setting") == 2


@pytest.mark.parametrize("field", ["source", "preset", "output"])
def test_completed_reuse_refuses_changed_inputs_or_output_config(
    tmp_path: Path, field: str
) -> None:
    jobs, output = job_file(tmp_path)
    worker = make_batch(tmp_path, FakeCompressor(output, ["Successful"]))
    original = batch.read_jobs(jobs)[0]
    batch.run([original], worker, in_flight=1)
    assert batch.run([original], worker, in_flight=1) == 0
    if field == "output":
        changed = batch.Job(
            original.name, original.source, original.preset, output.with_name("new.mov")
        )
    else:
        getattr(original, field).write_bytes(b"changed")
        changed = original
    with pytest.raises(ReplayError, match="changed"):
        batch.run([changed], worker, in_flight=1)


def test_terminal_uses_content_hash_not_stat_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, output = job_file(tmp_path)
    original_stat = batch._stat
    source = tmp_path / "part.moti"
    before = original_stat(source)

    def fixed_stat(path: Path) -> dict[str, int]:
        return before if path == source else original_stat(path)

    def runner(arguments: Sequence[str]) -> str:
        if "-jobpath" in arguments:
            return SUBMITTED
        source.write_bytes(b"mutate")
        output.write_bytes(b"movie")
        return SUCCESS

    monkeypatch.setattr(batch, "_stat", fixed_stat)
    worker = batch.Batch(tmp_path / "state", Path("/Compressor"), "test", runner, lambda _s: None)
    with pytest.raises(ReplayError, match="SHA256 changed"):
        batch.run(batch.read_jobs(jobs), worker, in_flight=1)
    assert json.loads((worker.state / "alpha-0000.json").read_text())["status"] == "submitted"


def test_monitor_only_never_submits_untracked_or_legacy_jobs(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["Successful"])
    worker = make_batch(tmp_path, fake)
    tracked = batch.read_jobs(jobs)[0]
    worker.state.mkdir()
    state = worker.state / "alpha-0000.json"
    state.write_text('{"status":"submitted","jobID":"J1"}')
    output.parent.mkdir()
    new = batch.Job("unsubmitted", tracked.source, tracked.preset, tmp_path / "new.mov")
    assert batch.run([tracked, new], worker, in_flight=1, monitor_only=True) == 1
    assert len(fake.calls) == 1
    assert "-monitor" in fake.calls[0]
    assert not (worker.state / "unsubmitted.json").exists()
    assert "submissionVersion" not in json.loads(state.read_text())
    assert not batch.verify_submission(tracked, json.loads(state.read_text()), rehash=True)


def test_monitor_only_rejects_retry(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    with pytest.raises(ReplayError, match="cannot retry"):
        batch.run(
            batch.read_jobs(jobs),
            make_batch(tmp_path, FakeCompressor(output, [])),
            in_flight=1,
            monitor_only=True,
            retry_controller_failure=True,
        )


def test_direct_submit_refuses_existing_receipt(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    worker = make_batch(tmp_path, FakeCompressor(output, []))
    worker.state.mkdir()
    (worker.state / "alpha-0000.json").write_text('{"status":"submitting"}')
    with pytest.raises(ReplayError, match="must be reconciled"):
        worker.submit(batch.read_jobs(jobs)[0])


@pytest.mark.parametrize("fault", ["empty", "linked", "missing-pins", "bad-digest"])
def test_invalid_bindings_fail_closed(tmp_path: Path, fault: str) -> None:
    jobs, output = job_file(tmp_path)
    job = batch.read_jobs(jobs)[0]
    worker = make_batch(tmp_path, FakeCompressor(output, ["Successful"]))
    if fault == "empty":
        job.source.write_bytes(b"")
    elif fault == "linked":
        target = tmp_path / "target.moti"
        job.source.rename(target)
        job.source.symlink_to(target)
    else:
        batch.run([job], worker, in_flight=1)
        state = worker.state / "alpha-0000.json"
        saved = json.loads(state.read_text())
        if fault == "missing-pins":
            saved["inputs"] = []
        else:
            saved["inputs"][0]["sha256"] = "invalid"
        state.write_text(json.dumps(saved))
    with pytest.raises(ReplayError):
        batch.run([job], worker, in_flight=1)


def test_completed_output_change_stops(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    worker = make_batch(tmp_path, FakeCompressor(output, ["Successful"]))
    batch.run(batch.read_jobs(jobs), worker, in_flight=1)
    output.write_bytes(b"changed movie")
    with pytest.raises(ReplayError, match="output changed"):
        batch.run(batch.read_jobs(jobs), worker, in_flight=1)


def test_snapshot_and_tracked_list_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _jobs, output = job_file(tmp_path)
    monkeypatch.setattr(batch, "MAX_JOBS", 0)
    with pytest.raises(ReplayError, match="job budget"):
        batch.parse_snapshot(SUCCESS)
    with pytest.raises(ReplayError, match="tracked job budget"):
        batch.run(
            [batch.Job("a", tmp_path, tmp_path, output)],
            make_batch(tmp_path, FakeCompressor(output, [])),
            in_flight=1,
        )
    monkeypatch.setattr(batch, "MAX_SNAPSHOT_BYTES", 1)
    with pytest.raises(ReplayError, match="byte budget"):
        batch.parse_snapshot(SUCCESS)


def test_monitor_only_cli_leaves_unsubmitted_jobs_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, [])
    monkeypatch.setattr(batch, "_run", fake)
    assert batch.main([str(jobs), "--state", str(tmp_path / "state"), "--monitor-only"]) == 0
    assert fake.calls == []
    assert json.loads(capsys.readouterr().out) == {
        "finished": 0,
        "monitor_only": True,
        "deferred_ambiguous": [],
    }
    assert not (tmp_path / "state/alpha-0000.json").exists()


def test_empty_existing_receipt_never_authorizes_resubmit(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, [])
    worker = make_batch(tmp_path, fake)
    worker.state.mkdir()
    (worker.state / "alpha-0000.json").write_text("{}")
    with pytest.raises(ReplayError, match="outcome.*unknown"):
        batch.run(batch.read_jobs(jobs), worker, in_flight=1)
    assert fake.calls == []
