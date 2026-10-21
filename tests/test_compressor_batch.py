import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import compressor_batch as batch
from replay import compressor_helper_guard
from replay.core import ReplayError

SUBMITTED = '{ "batch" : { "batchID" : "B1", "jobs" : [ {"jobID" : "J1"} ] }}'
SUCCESS = '[{ "jobStatus" : { "status" : "Successful", "jobid" : "J1" }}]'


@pytest.fixture(autouse=True)
def synthetic_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Older CLI fakes use /Compressor; supply real local bytes for their tool pin."""
    executable = tmp_path / "synthetic-compressor"
    executable.write_bytes(b"mock CLI executable")
    original = batch.runtime_paths

    def paths(tool: Path) -> tuple[Path, ...]:
        return original(executable if tool == Path("/Compressor") else tool)

    monkeypatch.setattr(batch, "runtime_paths", paths)


class FakeCompressor:
    """Records CLI calls and writes the output when a job is first monitored."""

    def __init__(self, output: Path, statuses: list[str]) -> None:
        self.output = output
        self.statuses = statuses
        self.calls: list[list[str]] = []

    def __call__(self, arguments: Sequence[str]) -> str:
        self.calls.append(list(arguments))
        if "-jobpath" in arguments:
            return "objc noise\n" + SUBMITTED
        status = self.statuses.pop(0)
        if status == "Successful":
            self.output.write_bytes(b"movie")
        if status == "[]":
            return "[]"
        return f'[{{ "jobStatus" : {{ "status" : "{status}", "jobid" : "J1" }}}}]'


@pytest.fixture(autouse=True)
def _private_shared_monitor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each test drives its own fake CLI sequence: no snapshot reuse unless a test opts in.
    monkeypatch.setattr(batch, "SHARED_MONITOR", tmp_path / "shared-monitor")
    monkeypatch.setattr(batch, "SHARED_SNAPSHOT_SECONDS", 0.0)


def job_file(tmp_path: Path) -> tuple[Path, Path]:
    source, preset = tmp_path / "part.moti", tmp_path / "alpha.setting"
    source.write_text("motion", encoding="utf-8")
    preset.write_text("preset", encoding="utf-8")
    output = tmp_path / "out" / "alpha.mov"
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "alpha-0000",
                        "source": str(source),
                        "preset": str(preset),
                        "output": str(output),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return jobs, output


def make_batch(tmp_path: Path, fake: FakeCompressor) -> batch.Batch:
    return batch.Batch(tmp_path / "state", Path("/Compressor"), "test", fake, lambda _s: None)


def test_submit_wait_and_record_success(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["[]", "Processing", "Successful"])
    finished = batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=2)
    assert finished == 1
    assert fake.calls[0][1:3] == ["-batchname", "test"]
    saved = json.loads((tmp_path / "state" / "alpha-0000.json").read_text(encoding="utf-8"))
    assert saved["status"] == "Successful"
    assert saved["jobID"] == "J1"
    assert saved["output"] == str(output)
    assert saved["bytes"] == 5
    assert saved["fingerprintMoment"] == "before-submit"
    assert saved["terminalInputsVerified"] is True
    assert batch.verify_submission(batch.read_jobs(jobs)[0], saved, rehash=True)
    assert batch.parse_submission(saved["submitResponse"]) == "J1"
    assert batch.parse_snapshot(Path(saved["monitorResponse"]["path"]).read_text()) == {
        "J1": "Successful"
    }


def test_resume_monitors_recorded_job_without_resubmitting(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "alpha-0000.json").write_text('{"status": "submitted", "jobID": "J1"}')
    output.parent.mkdir()
    fake = FakeCompressor(output, ["Successful"])
    assert batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=1) == 1
    assert fake.calls == [
        [
            "/Compressor",
            "-monitor",
            "-once",
            "-timeout",
            "10",
            "-outputformat",
            "json",
        ]
    ]


def test_completed_jobs_are_skipped(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "alpha-0000.json").write_text('{"status": "Successful", "jobID": "J1"}')
    fake = FakeCompressor(output, [])
    assert batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=1) == 0
    assert fake.calls == []


def test_unknown_submission_outcome_stops(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "alpha-0000.json").write_text('{"status": "submitting"}')
    with pytest.raises(ReplayError, match="unknown"):
        batch.run(
            batch.read_jobs(jobs), make_batch(tmp_path, FakeCompressor(output, [])), in_flight=1
        )


@pytest.mark.parametrize("status", ["Failed", "Failed: job controller is down"])
def test_failed_job_stops(tmp_path: Path, status: str) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, [status])
    with pytest.raises(ReplayError, match="ended as Failed"):
        batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=1)


def test_success_without_output_stops(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["Successful"])
    fake.output = tmp_path / "elsewhere" / "x.mov"
    fake.output.parent.mkdir()
    with pytest.raises(ReplayError, match="without output"):
        batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=1)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    output.parent.mkdir()
    output.write_bytes(b"old")
    with pytest.raises(ReplayError, match="overwrite"):
        batch.run(
            batch.read_jobs(jobs), make_batch(tmp_path, FakeCompressor(output, [])), in_flight=1
        )


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    (tmp_path / "part.moti").unlink()
    with pytest.raises(ReplayError, match="Missing Compressor source"):
        batch.run(
            batch.read_jobs(jobs), make_batch(tmp_path, FakeCompressor(output, [])), in_flight=1
        )


def test_in_flight_limit_is_bounded(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    with pytest.raises(ReplayError, match="In-flight"):
        batch.run(
            batch.read_jobs(jobs), make_batch(tmp_path, FakeCompressor(output, [])), in_flight=0
        )


def test_parse_submission_requires_one_job() -> None:
    with pytest.raises(ReplayError, match="exactly one"):
        batch.parse_submission('{"batch": {"jobs": []}}')


def test_parse_submission_requires_json() -> None:
    with pytest.raises(ReplayError, match="no submission"):
        batch.parse_submission("nothing")


def test_parse_submission_rejects_bad_id() -> None:
    with pytest.raises(ReplayError, match="Invalid Compressor job ID"):
        batch.parse_submission('{"batch": {"jobs": [{"jobID": "bad id"}]}}')


def test_parse_status_variants() -> None:
    assert batch.parse_status(SUCCESS) == "Successful"
    assert batch.parse_status("[]") == "Pending"


def test_parse_status_requires_list() -> None:
    with pytest.raises(ReplayError, match="no job status"):
        batch.parse_status("{}")


def test_parse_status_requires_single_row() -> None:
    with pytest.raises(ReplayError, match="one Compressor job status"):
        batch.parse_status('[{"jobStatus": {}}, {"jobStatus": {}}]')


def test_parse_status_requires_status_text() -> None:
    with pytest.raises(ReplayError, match="status is missing"):
        batch.parse_status('[{"jobStatus": {}}]')


def test_read_jobs_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    row = {"name": "a", "source": "/s", "preset": "/p", "output": "/o"}
    path.write_text(json.dumps({"jobs": [row, row]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="unique"):
        batch.read_jobs(path)


def test_read_jobs_rejects_bad_name(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps({"jobs": [{"name": "../x", "source": "/s", "preset": "/p", "output": "/o"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="Invalid Compressor job name"):
        batch.read_jobs(path)


def test_read_jobs_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": [{"name": "a"}]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="needs name"):
        batch.read_jobs(path)


def test_read_jobs_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text('{"jobs": []}', encoding="utf-8")
    with pytest.raises(ReplayError, match="nonempty"):
        batch.read_jobs(path)


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "jobs.json"
    path.write_text('{"jobs": []}', encoding="utf-8")
    assert batch.main([str(path), "--state", str(tmp_path / "state")]) == 1
    assert "COMPRESSOR_BATCH_ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("opt_in", [False, True])
def test_cli_runs_with_fake_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    opt_in: bool,
) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["Successful"])
    monkeypatch.setattr(batch, "_run", fake)
    options = ["--defer-ambiguous"] if opt_in else []
    assert batch.main([str(jobs), "--state", str(tmp_path / "state"), *options]) == 0
    expected = (
        '{"finished": 1, "completed": 1, "pending": 0, "deferred_ambiguous": []}'
        if opt_in
        else '{"finished": 1}'
    )
    assert capsys.readouterr().out.strip() == expected


def test_default_runner_executes_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="ok\n"),
    )
    assert batch._run(["/bin/echo", "ok"]) == "ok\n"


@pytest.mark.parametrize("ambiguous_names", [("uncertain-1",), ("uncertain-1", "uncertain-2")])
@pytest.mark.parametrize("new_names", [(), ("new-1", "new-2")])
@pytest.mark.parametrize("in_flight", [1, 2])
def test_cli_defers_without_touching_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ambiguous_names: tuple[str, ...],
    new_names: tuple[str, ...],
    in_flight: int,
) -> None:
    monkeypatch.setattr("replay.compressor_batch.time.sleep", lambda _s: None)
    jobs_path, _output = job_file(tmp_path)
    original = batch.read_jobs(jobs_path)[0]
    names = ambiguous_names + new_names
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": name,
                        "source": str(original.source),
                        "preset": str(original.preset),
                        "output": str(tmp_path / f"{name}.mov"),
                    }
                    for name in names
                ]
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state"
    state.mkdir()
    receipts = {name: state / f"{name}.json" for name in ambiguous_names}
    receipt_bytes = b'{ "status": "submitting", "evidence": "retain" }\n'
    receipts["uncertain-1"].write_bytes(receipt_bytes)
    if "uncertain-2" in receipts:
        receipts["uncertain-2"].write_bytes(receipt_bytes)
    calls: list[str] = []
    active: set[str] = set()
    peak = 0
    early_reports: list[str] = []

    def runner(arguments: Sequence[str]) -> str:
        nonlocal peak
        if not calls:
            early_reports.extend(capsys.readouterr().err.splitlines())
        if "-jobpath" in arguments:
            output = Path(arguments[arguments.index("-locationpath") + 1])
            name = output.stem
            calls.append(f"submit:{name}")
            active.add(name)
            peak = max(peak, len(active))
            return json.dumps({"batch": {"jobs": [{"jobID": name}]}})
        names = sorted(active)
        calls.extend(f"monitor:{name}" for name in names)
        active.clear()

        def complete(name: str) -> dict[str, dict[str, str]]:
            (tmp_path / f"{name}.mov").write_bytes(b"synthetic output")
            return {"jobStatus": {"status": "Successful", "jobid": name}}

        return json.dumps([complete(name) for name in names])

    monkeypatch.setattr(batch, "_run", runner)
    assert (
        batch.main(
            [
                str(jobs_path),
                "--state",
                str(state),
                "--defer-ambiguous",
                "--in-flight",
                str(in_flight),
                "--submit-workers",
                "1",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    reports = early_reports + captured.err.splitlines()
    assert [json.loads(line) for line in reports] == [
        {"deferred_ambiguous": name} for name in ambiguous_names
    ]
    assert not calls or len(early_reports) == len(ambiguous_names)
    assert json.loads(captured.out) == {
        "finished": len(new_names),
        "completed": len(new_names),
        "pending": len(ambiguous_names),
        "deferred_ambiguous": list(ambiguous_names),
    }
    assert [call for call in calls if call.startswith("submit:")] == [
        f"submit:{name}" for name in new_names
    ]
    assert [call for call in calls if call.startswith("monitor:")] == [
        f"monitor:{name}" for name in new_names
    ]
    assert [path.read_bytes() for path in receipts.values()] == [receipt_bytes] * len(
        ambiguous_names
    )
    assert peak <= in_flight
    assert active == set()
    assert "quality" not in captured.out
    assert "workflow" not in captured.out


@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize("saved_status", ["submitted", "Successful"])
def test_cli_known_receipts_keep_original_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    opt_in: bool,
    saved_status: str,
) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_text(json.dumps({"status": saved_status, "jobID": "J1"}), encoding="utf-8")
    before = receipt.read_bytes()
    output.parent.mkdir()
    fake = FakeCompressor(output, ["Successful"])
    monkeypatch.setattr(batch, "_run", fake)
    options = ["--defer-ambiguous"] if opt_in else []
    assert batch.main([str(jobs), "--state", str(state), *options]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    expected_finished = 1 if saved_status == "submitted" else 0
    if opt_in:
        assert json.loads(captured.out) == {
            "finished": expected_finished,
            "completed": 1,
            "pending": 0,
            "deferred_ambiguous": [],
        }
    else:
        assert json.loads(captured.out) == {"finished": expected_finished}
    assert all("-jobpath" not in call for call in fake.calls)
    if saved_status == "Successful":
        assert fake.calls == []
        assert receipt.read_bytes() == before


@pytest.mark.parametrize("failure", ["state", "submit", "monitor", "failed"])
def test_deferral_does_not_swallow_other_errors(tmp_path: Path, failure: str) -> None:
    jobs, output = job_file(tmp_path)
    known = batch.read_jobs(jobs)[0]
    uncertain = batch.Job("uncertain", known.source, known.preset, tmp_path / "uncertain.mov")
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "uncertain.json"
    receipt.write_bytes(b'{"status":"submitting"}')
    if failure == "state":
        (state / "alpha-0000.json").write_text("invalid", encoding="utf-8")
    reported: list[str] = []

    def runner(arguments: Sequence[str]) -> str:
        if failure == "submit" or (failure == "monitor" and "-monitor" in arguments):
            raise OSError("synthetic failure")
        if "-jobpath" in arguments:
            return SUBMITTED
        return '[{"jobStatus":{"status":"Failed"}}]'

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, lambda _s: None)
    with pytest.raises((ReplayError, OSError, ValueError)):
        batch.run([uncertain, known], worker, in_flight=1, on_ambiguous=reported.append)
    assert reported == ["uncertain"]
    assert receipt.read_bytes() == b'{"status":"submitting"}'
    assert not output.exists()


def test_default_unknown_blocks_independent_jobs_immediately(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    known = batch.read_jobs(jobs)[0]
    uncertain = batch.Job("uncertain", known.source, known.preset, tmp_path / "uncertain.mov")
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "uncertain.json"
    receipt.write_bytes(b'{"status":"submitting"}')
    fake = FakeCompressor(output, [])
    with pytest.raises(ReplayError, match="Submission outcome of uncertain is unknown"):
        batch.run([known, uncertain], make_batch(tmp_path, fake), in_flight=1)
    assert fake.calls == []
    assert receipt.read_bytes() == b'{"status":"submitting"}'


def test_deferred_run_resumes_known_ids_with_in_flight_bound(tmp_path: Path) -> None:
    jobs_path, output = job_file(tmp_path)
    new = batch.read_jobs(jobs_path)[0]
    uncertain = batch.Job("uncertain", new.source, new.preset, tmp_path / "uncertain.mov")
    submitted = batch.Job("submitted", new.source, new.preset, tmp_path / "submitted.mov")
    completed = batch.Job("completed", new.source, new.preset, tmp_path / "completed.mov")
    state = tmp_path / "state"
    state.mkdir()
    (state / "uncertain.json").write_bytes(b'{"status":"submitting"}')
    (state / "submitted.json").write_bytes(b'{"status":"submitted","jobID":"existing-ID"}')
    (state / "completed.json").write_bytes(b'{"status":"Successful","jobID":"done-ID"}')
    events: list[str] = []

    def runner(arguments: Sequence[str]) -> str:
        if "-jobpath" in arguments:
            events.append("submit:new")
            return SUBMITTED
        job_id = "J1" if "submit:new" in events else "existing-ID"
        events.append(f"monitor:{job_id}")
        target = submitted.output if job_id == "existing-ID" else output
        target.write_bytes(b"synthetic")
        return json.dumps([{"jobStatus": {"status": "Successful", "jobid": job_id}}])

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, lambda _s: None)
    result = batch.run(
        [new, uncertain, submitted, completed],
        worker,
        in_flight=1,
        on_ambiguous=lambda name: events.append(f"defer:{name}"),
    )
    assert type(result) is int
    assert result == 2
    assert events == ["defer:uncertain", "monitor:existing-ID", "submit:new", "monitor:J1"]
    assert (state / "uncertain.json").read_bytes() == b'{"status":"submitting"}'
    assert (state / "completed.json").read_bytes() == b'{"status":"Successful","jobID":"done-ID"}'


def test_submitting_with_id_is_not_silently_deferred(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_bytes(b'{"status":"submitting","jobID":"J1"}')
    fake = FakeCompressor(output, [])
    deferred: list[str] = []
    with pytest.raises(ReplayError, match="unknown"):
        batch.run(
            batch.read_jobs(jobs),
            make_batch(tmp_path, fake),
            in_flight=1,
            on_ambiguous=deferred.append,
        )
    assert fake.calls == []
    assert deferred == []
    assert receipt.read_bytes() == b'{"status":"submitting","jobID":"J1"}'


def test_snapshot_once_per_cycle_retains_missing_ids(tmp_path: Path) -> None:
    jobs_path, output = job_file(tmp_path)
    first = batch.read_jobs(jobs_path)[0]
    second = batch.Job("second", first.source, first.preset, tmp_path / "second.mov")
    state = tmp_path / "state"
    state.mkdir()
    (state / "alpha-0000.json").write_text('{"status":"submitted","jobID":"J1"}')
    (state / "second.json").write_text('{"status":"submitted","jobID":"J2"}')
    output.parent.mkdir()
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        if len(calls) == 1:
            output.write_bytes(b"first")
            return SUCCESS
        if len(calls) == 2:
            return '[{"jobStatus":{"jobid":"J2","status":"Missing"}}]'
        second.output.write_bytes(b"second")
        return '[{"jobStatus":{"jobid":"J2","status":"Successful"}}]'

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, sleeps.append)
    assert batch.run([first, second], worker, in_flight=2) == 2
    assert len(calls) == 3
    assert all(
        call == ["/Compressor", "-monitor", "-once", "-timeout", "10", "-outputformat", "json"]
        for call in calls
    )
    assert sleeps == [batch.POLL_SECONDS, batch.POLL_SECONDS]


@pytest.mark.parametrize(
    "text",
    [
        "no JSON",
        '[{"jobStatus":{"status":"Failed"}}]',
        '[{"jobStatus":{"jobid":"bad id","status":"Failed"}}]',
        '[{"jobStatus":{"jobid":"J1","jobID":"J2","status":"Failed"}}]',
        '[{"jobStatus":{"jobid":"J1"}}]',
        '[{"jobStatus":{"jobid":"J1","status":"Pending"}},'
        '{"jobStatus":{"jobid":"J1","status":"Successful"}}]',
    ],
)
def test_snapshot_rejects_invalid_identity_or_status(text: str) -> None:
    with pytest.raises(ReplayError):
        batch.parse_snapshot(text)


def test_snapshot_flattens_batches_without_inheriting_outcomes() -> None:
    assert batch.parse_snapshot(
        json.dumps(
            [
                {
                    "batchStatus": {
                        "name": "synthetic",
                        "batchid": "B1",
                        "status": "Failed",
                        "jobs": [
                            {"jobStatus": {"batchid": "B1", "jobid": "J1", "status": "Successful"}},
                            {"jobStatus": {"batchid": "B1", "jobid": "J2", "status": "Processing"}},
                        ],
                    }
                },
                {
                    "batchStatus": {
                        "batchid": "B2",
                        "status": "Successful",
                        "jobs": [
                            {"jobStatus": {"batchid": "B2", "jobid": "J3", "status": "Failed"}},
                        ],
                    }
                },
                {"batchStatus": {"batchid": "B3", "jobs": []}},
                {"jobStatus": {"jobID": "J4", "status": "Pending"}},
            ]
        )
    ) == {"J1": "Successful", "J2": "Processing", "J3": "Failed", "J4": "Pending"}


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {},
        {"batchStatus": None},
        {"batchStatus": {"batchid": "B1"}},
        {"batchStatus": {"batchid": "B1", "jobs": {}}},
        {"batchStatus": {"batchid": "bad id", "jobs": []}},
        {"batchStatus": {"batchid": "B1", "batchID": "B2", "jobs": []}},
        {"batchStatus": {"batchid": "B1", "jobs": []}, "jobStatus": {}},
    ],
)
def test_snapshot_rejects_malformed_batch(entry: object) -> None:
    with pytest.raises(ReplayError):
        batch.parse_snapshot(json.dumps([entry]))


@pytest.mark.parametrize(
    "job",
    [
        None,
        {},
        {"jobStatus": None},
        {"batchStatus": {}},
        {"jobStatus": {}, "batchStatus": {}},
        {"jobStatus": {"jobid": "J1", "status": "Failed"}},
        {"jobStatus": {"jobid": "J1", "batchid": "B2", "status": "Failed"}},
        {"jobStatus": {"jobid": "J1", "batchid": "B1", "batchID": "B2", "status": "Failed"}},
        {"jobStatus": {"jobid": "J1", "jobID": "J2", "batchid": "B1", "status": "Failed"}},
        {"jobStatus": {"jobid": "J1", "batchid": "B1"}},
        {"jobStatus": {"jobid": "J1", "batchid": "B1", "status": 42}},
    ],
)
def test_snapshot_rejects_malformed_nested_job(job: object) -> None:
    with pytest.raises(ReplayError):
        batch.parse_snapshot(
            json.dumps([{"batchStatus": {"batchid": "B1", "status": "Failed", "jobs": [job]}}])
        )


@pytest.mark.parametrize("second_batch", ["B1", "B2"])
def test_snapshot_rejects_duplicate_ids_across_batches(second_batch: str) -> None:
    with pytest.raises(ReplayError):
        batch.parse_snapshot(
            json.dumps(
                [
                    {
                        "batchStatus": {
                            "batchid": "B1",
                            "jobs": [
                                {"jobStatus": {"jobid": "J1", "batchid": "B1", "status": "Failed"}}
                            ],
                        }
                    },
                    {
                        "batchStatus": {
                            "batchid": second_batch,
                            "jobs": [
                                {
                                    "jobStatus": {
                                        "jobid": "J1",
                                        "batchid": second_batch,
                                        "status": "Failed",
                                    }
                                }
                            ],
                        }
                    },
                ]
            )
        )


@pytest.mark.parametrize("flat_duplicate", [False, True])
def test_snapshot_rejects_duplicate_nested_and_flat_ids(flat_duplicate: bool) -> None:
    job = {"jobStatus": {"jobid": "J1", "batchid": "B1", "status": "Pending"}}
    jobs = [job] if flat_duplicate else [job, job]
    rows: list[object] = [{"batchStatus": {"batchid": "B1", "jobs": jobs}}]
    if flat_duplicate:
        rows.append(job)
    with pytest.raises(ReplayError, match="Duplicate ID"):
        batch.parse_snapshot(json.dumps(rows))


def test_standalone_status_keeps_job_filter(tmp_path: Path) -> None:
    fake = FakeCompressor(tmp_path / "out.mov", ["Processing"])
    assert make_batch(tmp_path, fake).status("J1") == "Processing"
    assert fake.calls[0][1:5] == ["-monitor", "-jobid", "J1", "-once"]


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_controller_retry_preserves_receipt_and_resumes_alternate_output(
    tmp_path: Path, interrupt: bool, nested: bool
) -> None:
    jobs_path, output = job_file(tmp_path)
    jobs = batch.read_jobs(jobs_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    original = b'{ "status": "submitted", "jobID": "J1", "extra": "retain" }\n'
    receipt.write_bytes(original)
    output.parent.mkdir()
    output.touch()
    retry_output = output.with_name("alpha.retry-1.mov")
    calls: list[list[str]] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        if "-jobpath" in arguments:
            saved = json.loads(receipt.read_text())
            assert saved["status"] == "submitting"
            assert "jobID" not in saved
            assert saved["requested_output"] == str(retry_output)
            assert arguments[arguments.index("-locationpath") + 1] == str(retry_output)
            assert (state / "alpha-0000.attempt-0.receipt.json").read_bytes() == original
            return '{"batch":{"jobs":[{"jobID":"J2"}]}}'
        if len(calls) == 1:
            if nested:
                return json.dumps(
                    [
                        {
                            "batchStatus": {
                                "batchid": "B1",
                                "status": "Failed",
                                "jobs": [
                                    {
                                        "jobStatus": {
                                            "jobid": "J1",
                                            "batchid": "B1",
                                            "status": (
                                                "Failed: 2x “ジョブコントローラがダウンしています”"
                                            ),
                                        }
                                    },
                                    {
                                        "jobStatus": {
                                            "jobid": "OTHER",
                                            "batchid": "B1",
                                            "status": "Processing",
                                        }
                                    },
                                ],
                            }
                        }
                    ]
                )
            return (
                '[{"jobStatus":{"jobid":"J1",'
                '"status":"Failed: ジョブコントローラがダウンしています"}}]'
            )
        if interrupt:
            raise OSError("monitor interrupted")
        retry_output.write_bytes(b"retry movie")
        return '[{"jobStatus":{"jobid":"J2","status":"Successful"}}]'

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, lambda _s: None)
    if interrupt:
        with pytest.raises(OSError, match="interrupted"):
            batch.run(jobs, worker, in_flight=1, retry_controller_failure=True)
        retry_output.write_bytes(b"retry movie")
        resume_calls: list[list[str]] = []

        def resume(arguments: Sequence[str]) -> str:
            resume_calls.append(list(arguments))
            return '[{"jobStatus":{"jobid":"J2","status":"Successful"}}]'

        worker.runner = resume
        assert batch.run(jobs, worker, in_flight=1) == 1
        assert len(resume_calls) == 1
        assert "-monitor" in resume_calls[0]
    else:
        assert batch.run(jobs, worker, in_flight=1, retry_controller_failure=True) == 1
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "Successful"
    assert saved["jobID"] == "J2"
    assert saved["output"] == str(retry_output)
    assert saved["requested_output"] == str(retry_output)
    assert saved["history"][0]["jobID"] == "J1"
    if nested:
        assert saved["history"][0]["status"] == "Failed: 2x “ジョブコントローラがダウンしています”"
    else:
        assert saved["history"][0]["status"] == "Failed: ジョブコントローラがダウンしています"
    assert saved["retry_count"] == 1
    assert output.read_bytes() == b""
    assert (state / "alpha-0000.attempt-0.receipt.json").read_bytes() == original
    assert len([call for call in calls if "-jobpath" in call]) == 1
    assert batch.actual_output(output) == retry_output


@pytest.mark.parametrize(
    "status",
    [
        "Failed: codec failure",
        "Failed",
        "Cancelled",
        "Stopped",
        "Failed: 2x “codec error”",
        "Failed: 2x “ジョブコントローラがダウンしています; codec error”",
        "Failed: 2x “ジョブコントローラがダウンしています” codec error",
        "Failed: codec error 2x “ジョブコントローラがダウンしています”",
        "Failed: 0x “ジョブコントローラがダウンしています”",
        "Failed: 10000x “ジョブコントローラがダウンしています”",
        "Failed: 02x “ジョブコントローラがダウンしています”",
        "Failed: -2x “ジョブコントローラがダウンしています”",
        "Failed: 2.5x “ジョブコントローラがダウンしています”",
        "Failed: ２x “ジョブコントローラがダウンしています”",
        "Failed: 2x “ジョブコントローラがダウンしています",
        "Failed: 2x ジョブコントローラがダウンしています”",
        "Failed: 2x ”ジョブコントローラがダウンしています“",
    ],
)
def test_retry_opt_in_does_not_retry_other_failures(tmp_path: Path, status: str) -> None:
    jobs_path, output = job_file(tmp_path)
    fake = FakeCompressor(output, [status])
    with pytest.raises(ReplayError, match="ended as"):
        batch.run(
            batch.read_jobs(jobs_path),
            make_batch(tmp_path, fake),
            in_flight=1,
            retry_controller_failure=True,
        )
    assert len([call for call in fake.calls if "-jobpath" in call]) == 1


@pytest.mark.parametrize("variant", ["alpha.mov", "alpha.mp4", "alpha.retry-1.mov"])
def test_controller_retry_rejects_nonempty_prior_output(tmp_path: Path, variant: str) -> None:
    jobs_path, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_bytes(b'{"status":"submitted","jobID":"J1"}')
    output.parent.mkdir()
    partial = output.with_name(variant)
    partial.write_bytes(b"partial")
    fake = FakeCompressor(output, ["Failed: job controller is down"])
    with pytest.raises(ReplayError, match="explicit reconciliation"):
        batch.run(
            batch.read_jobs(jobs_path),
            make_batch(tmp_path, fake),
            in_flight=1,
            retry_controller_failure=True,
        )
    assert receipt.read_bytes() == b'{"status":"submitted","jobID":"J1"}'
    assert partial.read_bytes() == b"partial"
    assert len(fake.calls) == 1


def test_retry_submission_interruption_stays_unknown_and_deferred_untouched(tmp_path: Path) -> None:
    jobs_path, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_bytes(b'{"status":"submitted","jobID":"J1"}')
    calls: list[list[str]] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        if "-jobpath" in arguments:
            raise subprocess.TimeoutExpired(arguments, 120)
        return '[{"jobStatus":{"jobID":"J1","status":"Failed: job controller is down"}}]'

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, lambda _s: None)
    jobs = batch.read_jobs(jobs_path)
    with pytest.raises(subprocess.TimeoutExpired):
        batch.run(jobs, worker, in_flight=1, retry_controller_failure=True)
    unknown = receipt.read_bytes()
    assert json.loads(unknown)["status"] == "submitting"
    assert json.loads(unknown)["requested_output"] == str(output.with_name("alpha.retry-1.mov"))
    with pytest.raises(ReplayError, match="unknown"):
        batch.run(jobs, worker, in_flight=1, retry_controller_failure=True)
    deferred: list[str] = []
    assert (
        batch.run(
            jobs, worker, in_flight=1, retry_controller_failure=True, on_ambiguous=deferred.append
        )
        == 0
    )
    assert deferred == ["alpha-0000"]
    assert receipt.read_bytes() == unknown
    assert len(calls) == 2


def test_retry_is_bounded_across_invocations(tmp_path: Path) -> None:
    jobs_path, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["Failed: job controller is down"] * 3)
    worker = make_batch(tmp_path, fake)
    with pytest.raises(ReplayError, match="ended as Failed"):
        batch.run(batch.read_jobs(jobs_path), worker, in_flight=1, retry_controller_failure=True)
    with pytest.raises(ReplayError, match="not permitted"):
        batch.run(batch.read_jobs(jobs_path), worker, in_flight=1, retry_controller_failure=True)
    assert len([call for call in fake.calls if "-jobpath" in call]) == 2


def test_actual_output_refuses_multiple_nonempty_variants(tmp_path: Path) -> None:
    requested = tmp_path / "alpha.mov"
    requested.touch()
    (tmp_path / "alpha.mp4").write_bytes(b"partial")
    (tmp_path / "alpha.retry-1.mov").write_bytes(b"retry")
    with pytest.raises(ReplayError, match="without output"):
        batch.actual_output(requested)


def test_other_id_controller_failure_never_authorizes_retry(tmp_path: Path) -> None:
    jobs_path, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_bytes(b'{"status":"submitted","jobID":"J1"}')
    calls: list[list[str]] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        if len(calls) == 1:
            return '[{"jobStatus":{"jobid":"OTHER","status":"Failed: job controller is down"}}]'
        raise OSError("bounded observation ended")

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, lambda _s: None)
    with pytest.raises(OSError, match="observation ended"):
        batch.run(batch.read_jobs(jobs_path), worker, in_flight=1, retry_controller_failure=True)
    assert len(calls) == 2
    assert all("-monitor" in call for call in calls)
    assert receipt.read_bytes() == b'{"status":"submitted","jobID":"J1"}'
    assert not output.exists()


def test_retry_reconciles_all_ids_before_submission_and_limits_invocation(tmp_path: Path) -> None:
    jobs_path, output = job_file(tmp_path)
    first = batch.read_jobs(jobs_path)[0]
    second = batch.Job("second", first.source, first.preset, tmp_path / "second.mov")
    third = batch.Job("third", first.source, first.preset, tmp_path / "third.mov")
    state = tmp_path / "state"
    state.mkdir()
    (state / "alpha-0000.json").write_text('{"status":"submitted","jobID":"J1"}')
    (state / "second.json").write_text('{"status":"submitted","jobID":"J2"}')
    (state / "third.json").write_text('{"status":"submitted","jobID":"J3"}')
    second.output.write_bytes(b"done")
    calls: list[list[str]] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        if "-jobpath" in arguments:
            assert json.loads((state / "second.json").read_text())["status"] == "Successful"
            return '{"batch":{"jobs":[{"jobID":"J4"}]}}'
        return json.dumps(
            [
                {"jobStatus": {"jobid": "J1", "status": "Failed: job controller is down"}},
                {"jobStatus": {"jobid": "J2", "status": "Successful"}},
                {"jobStatus": {"jobid": "J3", "status": "Failed: job controller is down"}},
            ]
        )

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, lambda _s: None)
    with pytest.raises(ReplayError, match="third ended as Failed"):
        batch.run([first, second, third], worker, in_flight=3, retry_controller_failure=True)
    assert len(calls) == 2
    assert "-monitor" in calls[0]
    assert "-jobpath" in calls[1]
    assert json.loads((state / "alpha-0000.json").read_text())["jobID"] == "J4"
    assert (state / "third.json").read_bytes() == b'{"status":"submitted","jobID":"J3"}'
    assert not output.exists()


@pytest.mark.parametrize("monitor, timeout", [(False, 120), (True, 60)])
def test_direct_cli_uses_bounded_subprocess_run(
    monkeypatch: pytest.MonkeyPatch, monitor: bool, timeout: int
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(arguments: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(arguments, 0, stdout="result")

    monkeypatch.setattr(subprocess, "run", fake_run)
    arguments = ["/Compressor", "-monitor"] if monitor else ["/Compressor", "-jobpath", "/s"]
    assert batch._run(arguments) == "result"
    assert calls == [
        {
            "stdin": subprocess.DEVNULL,
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "check": True,
        }
    ]


def test_cli_controller_retry_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_path, output = job_file(tmp_path)
    fake = FakeCompressor(
        output.with_name("alpha.retry-1.mov"), ["Failed: job controller is down", "Successful"]
    )
    monkeypatch.setattr(batch, "_run", fake)
    monkeypatch.setattr("replay.compressor_batch.time.sleep", lambda _s: None)
    assert (
        batch.main(
            [str(jobs_path), "--state", str(tmp_path / "state"), "--retry-controller-failure"]
        )
        == 0
    )
    assert batch.actual_output(output) == output.with_name("alpha.retry-1.mov")


@pytest.mark.parametrize(
    "status, expected",
    [
        (
            "Failed: ジョブコントローラがダウンしています。\n"
            "ジョブコントローラがダウンしています。",
            True,
        ),
        ("Failed: Job controller is down.", True),
        ("Failed: 2x “ジョブコントローラがダウンしています”", True),
        ("Failed: 1x “job controller is down”", True),
        ("Failed: 9999x “job controller is down”", True),
        ("Failed: “ジョブコントローラがダウンしています”", True),
        ("Failed: 2x job controller is down", True),
        ("Cancelled: job controller is down", False),
        ("Failed: not job controller is down", False),
        ("Failed: job controller is down; codec error", False),
    ],
)
def test_controller_failure_requires_only_recognized_messages(status: str, expected: bool) -> None:
    assert batch._controller_failed(status) is expected


@pytest.mark.parametrize("direct", [False, True])
def test_guard_blocks_before_submission_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direct: bool
) -> None:
    jobs_path, output = job_file(tmp_path)
    fake = FakeCompressor(output, [])
    worker = make_batch(tmp_path, fake)
    worker.helper_guard_directory = tmp_path / "guard"

    def blocked(directory: Path, executable: Path) -> None:
        raise ReplayError("helper guard blocked")

    monkeypatch.setattr(compressor_helper_guard, "check", blocked)
    with pytest.raises(ReplayError, match="helper guard blocked"):
        if direct:
            worker.submit(batch.read_jobs(jobs_path)[0])
        else:
            batch.run(batch.read_jobs(jobs_path), worker, in_flight=1)
    assert fake.calls == []
    assert not (worker.state / "alpha-0000.json").exists()
    assert not output.exists()


def test_guard_each_cycle_and_before_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_path, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["Processing", "Successful"])
    worker = make_batch(tmp_path, fake)
    worker.helper_guard_directory = tmp_path / "guard"
    observations: list[int] = []

    def check(directory: Path, executable: Path) -> None:
        observations.append(len(fake.calls))

    monkeypatch.setattr(compressor_helper_guard, "check", check)
    assert batch.run(batch.read_jobs(jobs_path), worker, in_flight=1) == 1
    assert observations == [0, 0, 2]


def test_cli_guard_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_path, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["Successful"])
    monkeypatch.setattr(batch, "_run", fake)
    directories: list[Path] = []

    def check(directory: Path, executable: Path) -> None:
        directories.append(directory)

    monkeypatch.setattr(compressor_helper_guard, "check", check)
    assert (
        batch.main(
            [
                str(jobs_path),
                "--state",
                str(tmp_path / "state"),
                "--helper-guard-directory",
                str(tmp_path / "guard"),
            ]
        )
        == 0
    )
    assert len(directories) == 2
    assert directories[0] == tmp_path / "guard"


def test_guard_blocks_retry_before_any_receipt_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_path, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_bytes(b'{"status":"submitted","jobID":"J1"}')
    fake = FakeCompressor(output, [])
    worker = make_batch(tmp_path, fake)
    worker.helper_guard_directory = tmp_path / "guard"

    def blocked(directory: Path, executable: Path) -> None:
        raise ReplayError("helper guard blocked")

    monkeypatch.setattr(compressor_helper_guard, "check", blocked)
    with pytest.raises(ReplayError, match="helper guard blocked"):
        worker.retry(batch.read_jobs(jobs_path)[0], "J1", "Failed: job controller is down")
    assert receipt.read_bytes() == b'{"status":"submitted","jobID":"J1"}'
    assert not (state / "alpha-0000.attempt-0.receipt.json").exists()
    assert fake.calls == []


@pytest.mark.parametrize("timeouts", [1, 2, 3])
def test_monitor_timeout_budget_preserves_receipts_and_queue(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    timeouts: int,
) -> None:
    jobs_path, output = job_file(tmp_path)
    known = batch.read_jobs(jobs_path)[0]
    queued = batch.Job("queued", known.source, known.preset, tmp_path / "queued.mov")
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    original = b'{ "status": "submitted", "jobID": "J1", "extra": "retain" }\n'
    receipt.write_bytes(original)
    output.parent.mkdir()
    calls: list[list[str]] = []
    sleeps: list[float] = []
    unchanged: list[bool] = []
    failure = subprocess.TimeoutExpired("private command must not be logged", 60)

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        if len(calls) <= timeouts:
            raise failure
        if "-jobpath" in arguments:
            queued.output.write_bytes(b"synthetic queued")
            return '{"batch":{"jobs":[{"jobID":"J2"}]}}'
        output.write_bytes(b"synthetic known")
        return (
            '[{"jobStatus":{"jobid":"J1","status":"Successful"}},'
            '{"jobStatus":{"jobid":"J2","status":"Successful"}}]'
        )

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(calls) <= timeouts:
            unchanged.append(
                receipt.read_bytes() == original
                and not (state / "queued.json").exists()
                and not output.exists()
                and not queued.output.exists()
            )

    guard_calls: list[int] = []

    def check(directory: Path, executable: Path) -> None:
        guard_calls.append(len(calls))

    monkeypatch.setattr(compressor_helper_guard, "check", check)
    worker = batch.Batch(
        state,
        Path("/Compressor"),
        "test",
        runner,
        sleep,
        helper_guard_directory=tmp_path / "guard",
    )
    if timeouts == 3:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            batch.run([known, queued], worker, in_flight=1)
        assert caught.value is failure
        assert receipt.read_bytes() == original
        assert sorted(path.name for path in state.iterdir()) == ["alpha-0000.json"]
        assert len(calls) == 3
        assert sleeps == [5.0, 15.0]
        assert guard_calls == [0, 1, 2]
        assert not output.exists()
        assert not queued.output.exists()
    else:
        assert batch.run([known, queued], worker, in_flight=1) == 2
        assert json.loads(receipt.read_bytes())["jobID"] == "J1"
        assert len(calls) == timeouts + 3
        assert len([call for call in calls if "-jobpath" in call]) == 1
        assert sleeps == ([5.0] if timeouts == 1 else [5.0, 15.0])
        assert guard_calls == ([0, 1, 2, 2] if timeouts == 1 else [0, 1, 2, 3, 3])
    assert unchanged == ([True] if timeouts == 1 else [True, True])
    assert (
        calls[:timeouts]
        == [["/Compressor", "-monitor", "-once", "-timeout", "10", "-outputformat", "json"]]
        * timeouts
    )
    diagnostic = capsys.readouterr().err
    assert diagnostic == (
        "COMPRESSOR_MONITOR_RETRY attempt=2/3 delay=5s\n"
        if timeouts == 1
        else "COMPRESSOR_MONITOR_RETRY attempt=2/3 delay=5s\n"
        "COMPRESSOR_MONITOR_RETRY attempt=3/3 delay=15s\n"
    )


def test_monitor_retry_guard_blocks_before_next_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def runner(arguments: Sequence[str]) -> str:
        events.append("monitor")
        raise subprocess.TimeoutExpired(arguments, 60)

    def sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds:g}")

    def blocked(directory: Path, executable: Path) -> None:
        events.append("guard")
        raise ReplayError("helper guard blocked")

    monkeypatch.setattr(compressor_helper_guard, "check", blocked)
    worker = batch.Batch(
        tmp_path / "state",
        Path("/Compressor"),
        "test",
        runner,
        sleep,
        helper_guard_directory=tmp_path / "guard",
    )
    with pytest.raises(ReplayError, match="helper guard blocked"):
        worker.snapshot()
    assert events == ["monitor", "sleep:5", "guard"]
    assert not worker.state.exists()


@pytest.mark.parametrize("error", [OSError("failure"), subprocess.CalledProcessError(1, "cli")])
def test_monitor_other_runner_errors_are_not_retried(tmp_path: Path, error: Exception) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        raise error

    worker = batch.Batch(tmp_path / "state", Path("/Compressor"), "test", runner, sleeps.append)
    with pytest.raises(type(error)) as caught:
        worker.snapshot()
    assert caught.value is error
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "response", ["[invalid", '[{"jobStatus":{}}]', SUCCESS.replace("Successful", "Failed")]
)
def test_monitor_parse_and_native_failures_are_not_retried(tmp_path: Path, response: str) -> None:
    jobs_path, _output = job_file(tmp_path)
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        return SUBMITTED if "-jobpath" in arguments else response

    worker = batch.Batch(tmp_path / "state", Path("/Compressor"), "test", runner, sleeps.append)
    with pytest.raises((ReplayError, ValueError)):
        batch.run(batch.read_jobs(jobs_path), worker, in_flight=1)
    assert len(calls) == 2
    assert sleeps == []
    saved = json.loads((worker.state / "alpha-0000.json").read_text())
    assert saved["status"] == "submitted"
    assert saved["jobID"] == "J1"
    assert saved["fingerprintMoment"] == "before-submit"
    assert "terminalInputsVerified" not in saved


@pytest.mark.parametrize("standalone", [False, True])
def test_submission_and_standalone_status_timeout_stay_single_call(
    tmp_path: Path, standalone: bool
) -> None:
    jobs_path, _output = job_file(tmp_path)
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        raise subprocess.TimeoutExpired(arguments, 60)

    worker = batch.Batch(tmp_path / "state", Path("/Compressor"), "test", runner, sleeps.append)
    with pytest.raises(subprocess.TimeoutExpired):
        if standalone:
            worker.status("J1")
        else:
            batch.run(batch.read_jobs(jobs_path), worker, in_flight=1)
    assert len(calls) == 1
    assert sleeps == []
    if standalone:
        assert calls[0] == [
            "/Compressor",
            "-monitor",
            "-jobid",
            "J1",
            "-once",
            "-timeout",
            "10",
            "-outputformat",
            "json",
        ]
    else:
        saved = json.loads((worker.state / "alpha-0000.json").read_text())
        assert saved["status"] == "submitting"
        assert saved["fingerprintMoment"] == "before-submit"
        assert "jobID" not in saved


def test_job_missing_from_every_snapshot_fails_instead_of_waiting_forever(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "alpha-0000.json").write_text('{"status": "submitted", "jobID": "J1"}')
    output.parent.mkdir()
    fake = FakeCompressor(output, ["[]"] * (batch.LOST_POLLS + 5))
    with pytest.raises(ReplayError, match="absent from"):
        batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=1)
    assert len(fake.calls) == batch.LOST_POLLS


def test_purged_job_with_output_is_reported_without_blocking_others(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "alpha-0000.json").write_text('{"status": "submitted", "jobID": "J1"}')
    output.parent.mkdir()
    output.write_bytes(b"movie")
    fake = FakeCompressor(output, ["[]"] * batch.LOST_POLLS)
    with pytest.raises(ReplayError, match="verify the outputs"):
        batch.run(batch.read_jobs(jobs), make_batch(tmp_path, fake), in_flight=1)
    saved = json.loads((state / "alpha-0000.json").read_text(encoding="utf-8"))
    assert saved["status"] == "submitted"


def test_parallel_submit_tracks_every_accepted_job(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    fake = FakeCompressor(output, ["Successful"])
    worker = make_batch(tmp_path, fake)
    worker.submit_workers = 3
    worker.state.mkdir()
    first = batch.read_jobs(jobs)[0]
    accepted, error = worker.submit_many([first])
    assert error is None
    assert accepted[first.name][1] == "J1"
    missing = batch.Job("gone", tmp_path / "none.moti", first.preset, tmp_path / "o.mov", ())
    accepted, error = worker.submit_many([missing])
    assert accepted == {} and isinstance(error, ReplayError)


def test_refill_skips_idle_sleep_but_processing_slots_remain_bounded(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    first = batch.read_jobs(jobs)[0]
    second = batch.Job("second", first.source, first.preset, output.with_name("second.mov"))
    third = batch.Job("third", first.source, first.preset, output.with_name("third.mov"))
    events: list[str] = []
    polls = 0

    def runner(arguments: Sequence[str]) -> str:
        nonlocal polls
        if "-jobpath" in arguments:
            destination = Path(arguments[arguments.index("-locationpath") + 1])
            identity = {"alpha.mov": "J1", "second.mov": "J2", "third.mov": "J3"}[destination.name]
            events.append(f"submit:{destination.name}")
            return json.dumps({"batch": {"jobs": [{"jobID": identity}]}})
        polls += 1
        events.append("monitor")
        if polls == 1:
            output.write_bytes(b"first")
            return (
                '[{"jobStatus":{"jobid":"J1","status":"Successful"}},'
                '{"jobStatus":{"jobid":"J2","status":"Processing"}}]'
            )
        if polls == 2:
            return (
                '[{"jobStatus":{"jobid":"J2","status":"Processing"}},'
                '{"jobStatus":{"jobid":"J3","status":"Processing"}}]'
            )
        second.output.write_bytes(b"second")
        third.output.write_bytes(b"third")
        return (
            '[{"jobStatus":{"jobid":"J2","status":"Successful"}},'
            '{"jobStatus":{"jobid":"J3","status":"Successful"}}]'
        )

    def sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    worker = batch.Batch(tmp_path / "state", Path("/Compressor"), "test", runner, sleep)
    assert batch.run([first, second, third], worker, in_flight=2) == 3
    assert events == [
        "submit:alpha.mov",
        "submit:second.mov",
        "monitor",
        "submit:third.mov",
        "monitor",
        f"sleep:{batch.POLL_SECONDS}",
        "monitor",
    ]
    assert output.read_bytes() == b"first"
    assert second.output.read_bytes() == b"second"
    assert third.output.read_bytes() == b"third"


def test_reappearing_job_resets_consecutive_missing_history_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "LOST_POLLS", 2)
    jobs, output = job_file(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_bytes(b'{"status":"submitted","jobID":"J1"}')
    output.parent.mkdir()
    responses = [
        "[]",
        '[{"jobStatus":{"jobid":"J1","status":"Processing"}}]',
        "[]",
        "Successful",
    ]
    fake = FakeCompressor(output, responses)
    calls = 0
    sleeps: list[float] = []

    def runner(arguments: Sequence[str]) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            fake.calls.append(list(arguments))
            return responses.pop(0)
        return fake(arguments)

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, sleeps.append)
    assert batch.run(batch.read_jobs(jobs), worker, in_flight=1) == 1
    assert calls == 4
    assert sleeps == [batch.POLL_SECONDS] * 3
    assert output.read_bytes() == b"movie"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "Successful"
    assert all("-jobpath" not in arguments for arguments in fake.calls)


def test_purged_history_preserves_bytes_and_receipt_while_refilling_independent_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "LOST_POLLS", 2)
    jobs, output = job_file(tmp_path)
    first = batch.read_jobs(jobs)[0]
    independent = batch.Job(
        "independent", first.source, first.preset, output.with_name("independent.mov")
    )
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "alpha-0000.json"
    receipt.write_bytes(b'{"status":"submitted","jobID":"J1"}')
    output.parent.mkdir()
    output.write_bytes(b"retained movie")
    events: list[str] = []

    def runner(arguments: Sequence[str]) -> str:
        if "-jobpath" in arguments:
            events.append("submit:independent")
            return '{"batch":{"jobs":[{"jobID":"J2"}]}}'
        events.append("monitor")
        if "submit:independent" not in events:
            return "[]"
        independent.output.write_bytes(b"new movie")
        return '[{"jobStatus":{"jobid":"J2","status":"Successful"}}]'

    def sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    worker = batch.Batch(state, Path("/Compressor"), "test", runner, sleep)
    with pytest.raises(ReplayError, match="alpha-0000.*receipts are untouched"):
        batch.run([first, independent], worker, in_flight=1)
    assert events == [
        "monitor",
        f"sleep:{batch.POLL_SECONDS}",
        "monitor",
        "submit:independent",
        "monitor",
    ]
    assert receipt.read_bytes() == b'{"status":"submitted","jobID":"J1"}'
    assert output.read_bytes() == b"retained movie"
    assert independent.output.read_bytes() == b"new movie"
    completed = json.loads((state / "independent.json").read_text(encoding="utf-8"))
    assert completed["status"] == "Successful"
    assert completed["jobID"] == "J2"
    assert completed["terminalInputsVerified"] is True
    assert completed["terminalRuntimeVerified"] is True


def test_parallel_partial_submission_reconciles_accepted_ids_without_resubmitting(
    tmp_path: Path,
) -> None:
    jobs, output = job_file(tmp_path)
    first = batch.read_jobs(jobs)[0]
    missing = batch.Job(
        "missing", tmp_path / "missing.moti", first.preset, output.with_name("missing.mov")
    )
    last = batch.Job("last", first.source, first.preset, output.with_name("last.mov"))
    submitted: list[str] = []
    monitored: list[str] = []

    def runner(arguments: Sequence[str]) -> str:
        if "-jobpath" in arguments:
            destination = Path(arguments[arguments.index("-locationpath") + 1])
            submitted.append(destination.name)
            identity = "J1" if destination.name == "alpha.mov" else "J3"
            return json.dumps({"batch": {"jobs": [{"jobID": identity}]}})
        monitored.append("monitor")
        output.write_bytes(b"first movie")
        last.output.write_bytes(b"last movie")
        return (
            '[{"jobStatus":{"jobid":"J1","status":"Successful"}},'
            '{"jobStatus":{"jobid":"J3","status":"Successful"}}]'
        )

    worker = batch.Batch(tmp_path / "state", Path("/Compressor"), "test", runner, lambda _s: None)
    worker.submit_workers = 3
    with pytest.raises(ReplayError, match="Missing Compressor source"):
        batch.run([first, missing, last], worker, in_flight=3)
    assert sorted(submitted) == ["alpha.mov", "last.mov"]
    assert monitored == []
    assert (
        json.loads((worker.state / "alpha-0000.json").read_text(encoding="utf-8"))["jobID"] == "J1"
    )
    assert json.loads((worker.state / "last.json").read_text(encoding="utf-8"))["jobID"] == "J3"
    assert not (worker.state / "missing.json").exists()
    assert not output.exists()
    assert not last.output.exists()
    assert batch.run([first, last], worker, in_flight=2) == 2
    assert sorted(submitted) == ["alpha.mov", "last.mov"]
    assert monitored == ["monitor"]
    assert output.read_bytes() == b"first movie"
    assert last.output.read_bytes() == b"last movie"


def test_batches_share_one_recent_monitor_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        return '[{"jobStatus":{"jobid":"J1","status":"Successful"}}]'

    monkeypatch.setattr(batch, "SHARED_SNAPSHOT_SECONDS", 60.0)
    shared = tmp_path / "monitor"
    first, second = (
        batch.Batch(tmp_path / name, Path("/Compressor"), "test", runner, lambda _s: None,
                    shared_monitor=shared)
        for name in ("a", "b")
    )  # fmt: skip
    assert first.snapshot() == second.snapshot() == {"J1": "Successful"}
    assert len(calls) == 1
    assert second.monitor_response == first.monitor_response
    monkeypatch.setattr(batch, "SHARED_SNAPSHOT_SECONDS", 0.0)
    second.snapshot()
    assert len(calls) == 2
    (shared / "snapshot.json").write_text("not json", encoding="utf-8")
    monkeypatch.setattr(batch, "SHARED_SNAPSHOT_SECONDS", 60.0)
    first.snapshot()
    assert len(calls) == 3
