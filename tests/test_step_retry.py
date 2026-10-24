"""A step is retried once, and only for a transient failure its attempt wrote."""

import json
from pathlib import Path

import pytest

from replay import failure_notify, step_retry, volume_events
from replay.stages import StageError

CONTROLLER = (
    "COMPRESSOR_BATCH_ERROR Compressor job w002-part-0002 ended as Failed: 2x "
    "“ジョブコントローラがダウンしています”\n"
)
PART = "error: Part 7 of media/2026-10-02_18-30-10_JST_replay-1-review.mp4 failed\n"


@pytest.fixture(autouse=True)
def no_failure_notices(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Failure notices go to a list, never to the desktop or the shared failures record."""
    sent: list[tuple[object, ...]] = []
    monkeypatch.setattr(failure_notify, "notify", lambda *args, **kw: sent.append((*args, kw)))
    return sent


class Step:
    """A fake step: each attempt appends ``producer`` text and fails while ``failures`` last."""

    def __init__(self, logs: Path, producer: list[str], failures: int) -> None:
        self.logs, self.producer, self.failures = logs, producer, failures
        self.runs: list[str] = []
        self.count = 0

    def run_log(self) -> Path:
        self.count += 1
        return self.logs / f"render-{self.count}.run.log"

    def execute(self, log: Path) -> None:
        self.runs.append(log.name)
        produced = self.logs / "caption-render.log"
        with produced.open("a", encoding="utf-8") as stream:
            stream.write(self.producer[len(self.runs) - 1])
        if len(self.runs) <= self.failures:
            log.write_text(
                f"replay.core.ReplayError: Producer failed (1); inspect {produced}\n", "utf-8"
            )
            raise StageError(f"Stage command failed; inspect {log}")
        log.write_text("done\n", "utf-8")


def run(step: Step, waits: list[float]) -> None:
    step_retry.run_with_retry(
        step.execute, step.run_log, logs=step.logs, step="caption-render", sleep=waits.append
    )


def test_a_controller_down_failure_is_retried_once(tmp_path: Path) -> None:
    step = Step(tmp_path, [CONTROLLER, "ok\n"], failures=1)
    waits: list[float] = []
    run(step, waits)
    assert step.runs == ["render-1.run.log", "render-2.run.log"]
    assert waits == [30.0]


def test_an_upload_part_failure_is_retried_once(tmp_path: Path) -> None:
    step = Step(tmp_path, [PART, "ok\n"], failures=1)
    waits: list[float] = []
    run(step, waits)
    assert step.runs == ["render-1.run.log", "render-2.run.log"]


def test_a_second_transient_failure_stops_the_step(tmp_path: Path) -> None:
    step = Step(tmp_path, [CONTROLLER, CONTROLLER], failures=2)
    with pytest.raises(StageError, match="render-2.run.log"):
        run(step, [])
    assert step.runs == ["render-1.run.log", "render-2.run.log"]


def test_other_failures_are_not_retried(tmp_path: Path) -> None:
    step = Step(tmp_path, ["Rendered caption gradient is missing\n"], failures=1)
    waits: list[float] = []
    with pytest.raises(StageError, match="render-1.run.log"):
        run(step, waits)
    assert waits == []


def test_a_transient_failure_of_an_earlier_attempt_does_not_count(tmp_path: Path) -> None:
    (tmp_path / "caption-render.log").write_text(CONTROLLER, "utf-8")  # an older run's failure
    step = Step(tmp_path, ["Output mapping needs explicit reconciliation\n"], failures=1)
    with pytest.raises(StageError):
        run(step, [])
    assert step.runs == ["render-1.run.log"]


def test_a_success_runs_once(tmp_path: Path) -> None:
    step = Step(tmp_path, ["ok\n"], failures=0)
    run(step, [])
    assert step.runs == ["render-1.run.log"]


def test_a_run_log_without_a_producer_or_absent_is_not_transient(tmp_path: Path) -> None:
    log = tmp_path / "step.run.log"
    assert step_retry.transient(log, {}) is False
    log.write_text("Step stopped; its finished work is reused on the next run\n", "utf-8")
    assert step_retry.transient(log, {}) is False


def test_a_replaced_producer_log_is_read_from_its_start(tmp_path: Path) -> None:
    producer = tmp_path / "upload.log"
    producer.write_text(PART, "utf-8")
    log = tmp_path / "upload.run.log"
    log.write_text(f"Producer failed (1); inspect {producer}\n", "utf-8")
    assert step_retry.transient(log, {producer: 10_000}) is True


class Progressing(Step):
    """A step whose every attempt finishes one more receipt before failing."""

    def __init__(self, logs: Path, producer: list[str], failures: int) -> None:
        super().__init__(logs, producer, failures)
        self.done = 0

    def execute(self, log: Path) -> None:
        self.done += 1
        super().execute(log)


def records(tmp_path: Path) -> list[dict[str, object]]:
    paths = sorted(tmp_path.glob("*.failure.json"), key=lambda path: path.name)
    return [json.loads(path.read_text("utf-8")) for path in paths]


def test_a_transient_fault_that_keeps_progressing_is_retried_until_the_cap(tmp_path: Path) -> None:
    step = Progressing(tmp_path, [CONTROLLER] * 5, failures=5)
    waits: list[float] = []
    with pytest.raises(StageError, match="render-4.run.log"):
        step_retry.run_with_retry(
            step.execute, step.run_log, logs=tmp_path, step="caption-render",
            progress=lambda: step.done, sleep=waits.append, clock=lambda: 0.0,
        )  # fmt: skip
    assert waits == [30.0, 60.0, 120.0]
    assert [row["stopReason"] for row in records(tmp_path)][-1] == "3 retries within 3600 s"


def test_a_transient_fault_injected_once_relaunches_and_completes(tmp_path: Path) -> None:
    step = Progressing(tmp_path, [PART, "ok\n"], failures=1)
    step_retry.run_with_retry(
        step.execute, step.run_log, logs=tmp_path, step="upload", progress=lambda: step.done,
        sleep=lambda _s: None,
    )  # fmt: skip
    assert step.runs == ["render-1.run.log", "render-2.run.log"]
    assert [(row["class"], row["retried"]) for row in records(tmp_path)] == [("transient", True)]


def test_a_retry_without_progress_stops(tmp_path: Path) -> None:
    step = Step(tmp_path, [CONTROLLER] * 3, failures=3)
    with pytest.raises(StageError, match="render-2.run.log"):
        run(step, [])
    assert [row["stopReason"] for row in records(tmp_path)] == [
        None,
        "the previous attempt made no progress",
    ]


def test_a_deterministic_failure_is_recorded_and_never_retried(tmp_path: Path) -> None:
    step = Step(tmp_path, ["ReplayError: Audit failed after escalation for chunk 4\n"], failures=1)
    with pytest.raises(StageError):
        run(step, [])
    (row,) = records(tmp_path)
    assert (row["class"], row["subclass"], row["retried"]) == (
        "deterministic", "audit or gate", False,
    )  # fmt: skip
    assert row["cause"] == "ReplayError: Audit failed after escalation"
    assert row["stopReason"] == "deterministic failures are not retried"


def test_an_operator_stop_is_never_retried(tmp_path: Path) -> None:
    log = tmp_path / "render.run.log"
    log.write_text("Step stopped; its finished work is reused on the next run\n", "utf-8")
    assert step_retry.classify(log, {}) == step_retry.Failure(
        "operator", "step stopped (SIGTERM)", "Step stopped; its finished work is reused"
    )


def test_compose_and_mps_failures_are_transient(tmp_path: Path) -> None:
    log = tmp_path / "compose.run.log"
    log.write_text(
        "subprocess.CalledProcessError: Command '['py', '-m', 'replay.apple_pro_apps_call', "
        "'media_edit', 'x']' returned non-zero exit status 1.\n", "utf-8",
    )  # fmt: skip
    assert step_retry.classify(log, {}).sub == "compose media_edit call"
    log.write_text("RuntimeError: MPS backend out of memory (MPS allocated: 2 GiB)\n", "utf-8")
    assert step_retry.classify(log, {}).sub == "MPS out of memory"
    log.write_text("", "utf-8")
    assert step_retry.classify(log, {}) == step_retry.Failure("unknown", "unclassified", "")


def test_receipts_count_stage_and_chunk_receipts(tmp_path: Path) -> None:
    (tmp_path / "a" / "state").mkdir(parents=True)
    (tmp_path / "a" / "state" / "x.json").write_text("{}", "utf-8")
    (tmp_path / "a" / "state" / "x.lock").write_text("", "utf-8")
    (tmp_path / "b" / "chunk-00001").mkdir(parents=True)
    (tmp_path / "b" / "chunk-00001" / "chunk-receipt.json").write_text("{}", "utf-8")
    assert step_retry.receipts((tmp_path / "a", tmp_path / "b", tmp_path / "none")) == 2


def test_a_deterministic_failure_raises_a_notice(
    tmp_path: Path, no_failure_notices: list[tuple[object, ...]]
) -> None:
    step = Step(tmp_path, ["ReplayError: Rendered caption gradient is missing\n"], failures=1)
    with pytest.raises(StageError):
        step_retry.run_with_retry(
            step.execute, step.run_log, logs=tmp_path, step="caption-render", recording="9",
            sleep=lambda _s: None,
        )  # fmt: skip
    assert no_failure_notices == [(
        "9", "caption-render",
        {"cls": "deterministic", "cause": "ReplayError: Rendered caption gradient is missing",
         "action": "needs a person"},
    )]  # fmt: skip


def test_a_stale_state_refusal_is_relaunched_once_at_once(
    tmp_path: Path, no_failure_notices: list[tuple[object, ...]]
) -> None:
    stale = "StageError: Partial outputs require a read-only provenance verifier\n"
    step = Step(tmp_path, [stale, "ok\n"], failures=1)
    waits: list[float] = []
    run(step, waits)
    assert step.runs == ["render-1.run.log", "render-2.run.log"]
    assert waits == []
    again = Step(tmp_path / "again", [stale, stale], failures=2)
    (tmp_path / "again").mkdir()
    with pytest.raises(StageError):
        run(again, waits)
    assert [row["stopReason"] for row in records(tmp_path / "again")] == [
        None,
        "stale failures are not retried",
    ]


def test_an_edit_mid_run_is_deterministic_and_relaunchable_at_once(
    tmp_path: Path, no_failure_notices: list[tuple[object, ...]]
) -> None:
    step = Step(tmp_path, ["EDITED_MID_RUN src/replay/caption_native_paging.py\n"], failures=1)
    with pytest.raises(StageError):
        run(step, [])
    assert no_failure_notices[0][2] == {
        "cls": "deterministic", "cause": "EDITED_MID_RUN", "action": "relaunch now",
    }  # fmt: skip


EIO = "OSError: [Errno 5] Input/output error: '/Volumes/m.2-ssd/movie/x.mov'\n"
RESULT_FOLDER = (
    "COMPRESSOR_BATCH_ERROR Compressor job w028-part-0000 ended as Failed: 3x "
    "“結果フォルダ“/Users/x/caption-parts-v1/moti/w028-part-0000.moti”にアクセスできません”\n"
)
MISSING = "FileNotFoundError: [Errno 2] No such file or directory: '/Volumes/m.2-ssd/x'\n"


def test_a_device_error_is_retried_once_after_the_volume_recovered(tmp_path: Path) -> None:
    step = Step(tmp_path, [EIO, "ok\n"], failures=1)
    recovered: list[tuple[object, ...]] = []

    def recover(before: object, roots: object, *, failed_at: float) -> None:
        recovered.append((before, roots))

    step_retry.run_with_retry(
        step.execute, step.run_log, logs=tmp_path, step="compose", recover=recover,
        receipt_roots=(tmp_path / "work",),
    )  # fmt: skip
    assert step.runs == ["render-1.run.log", "render-2.run.log"]
    assert recovered == [({}, (tmp_path / "work",))]
    rows = [json.loads(path.read_text("utf-8")) for path in tmp_path.glob("*.failure.json")]
    assert [(row["class"], row["subclass"], row["retried"]) for row in rows] == [
        ("volume", "device I/O error", True)
    ]


def test_a_second_volume_failure_needs_a_person(
    tmp_path: Path, no_failure_notices: list[tuple[object, ...]]
) -> None:
    step = Step(tmp_path, [RESULT_FOLDER, RESULT_FOLDER], failures=2)
    with pytest.raises(StageError):
        step_retry.run_with_retry(
            step.execute, step.run_log, logs=tmp_path, step="caption-render",
            recover=lambda *_a, **_k: None,
        )  # fmt: skip
    assert len(step.runs) == 2
    assert no_failure_notices == [
        (
            "",
            "caption-render",
            {
                "cls": "volume",
                "cause": "COMPRESSOR_BATCH_ERROR Compressor job w028-part-0000 ended as Failed: 3x "
                "“結果フォルダ“/Users/x/caption-parts-v1/moti/w028-part-0000.moti”にアクセスできません",
                "action": "needs a person",
            },
        )
    ]


def test_a_missing_file_while_a_volume_detached_counts_as_volume(tmp_path: Path) -> None:
    step = Step(tmp_path, [MISSING, "ok\n"], failures=1)
    ssd = {"/Volumes/m.2-ssd": volume_events.Volume("UUID-A", 5)}
    readings = iter([ssd, {}, ssd])
    step_retry.run_with_retry(
        step.execute, step.run_log, logs=tmp_path, step="compose",
        volumes=lambda: next(readings), recover=lambda *_a, **_k: None,
    )  # fmt: skip
    assert len(step.runs) == 2


def test_a_missing_file_without_a_detach_stays_deterministic(tmp_path: Path) -> None:
    step = Step(tmp_path, [MISSING], failures=1)
    with pytest.raises(StageError):
        step_retry.run_with_retry(
            step.execute, step.run_log, logs=tmp_path, step="compose",
            recover=lambda *_a, **_k: None,
        )  # fmt: skip
    assert len(step.runs) == 1


def test_a_volume_that_did_not_return_stops_the_step(
    tmp_path: Path, no_failure_notices: list[tuple[object, ...]]
) -> None:
    step = Step(tmp_path, [EIO], failures=1)
    with pytest.raises(StageError):
        step_retry.run_with_retry(
            step.execute, step.run_log, logs=tmp_path, step="compose",
            recover=lambda *_a, **_k: "/Volumes/m.2-ssd did not return within 600 s",
        )  # fmt: skip
    rows = [json.loads(path.read_text("utf-8")) for path in tmp_path.glob("*.failure.json")]
    assert [row["stopReason"] for row in rows] == ["/Volumes/m.2-ssd did not return within 600 s"]
    assert len(no_failure_notices) == 1
