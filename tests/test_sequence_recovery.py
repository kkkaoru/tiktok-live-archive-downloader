"""Typed recovery lifecycle simulation; no native calls or processes."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_native_sequence import Fake, MemoryJournal

from replay.native_sequence import Checkpoint, Config, ContractError, run
from replay.sequence_adapters import reference
from replay.sequence_journal import Pin
from replay.sequence_recovery import Boundary, BoundedRecovery, Quit


class Runtime:
    def __init__(self, root: Path) -> None:
        evidence = root / "evidence.json"
        evidence.write_text("{}", encoding="utf-8")
        self.pin = Pin(evidence, reference(evidence)["sha256"])
        candidate = root / "candidate.json"
        candidate.write_text('{"canDispatch":false,"candidateOnly":true}', encoding="utf-8")
        self.future = Pin(candidate, reference(candidate)["sha256"])
        self.before = Boundary(
            "saved project",
            123,
            "birth-one",
            self.pin,
            "memory-pressure",
            self.pin,
            (self.pin,),
            (self.pin,),
            2,
            12000,
            100000,
            True,
            True,
            True,
            True,
            True,
        )
        self.after = replace(
            self.before, pid=456, birth="birth-two", pressure=1, physical_bytes=1500
        )
        self.quit = Quit(self.pin, False, False, True)
        self.forced = Quit(self.pin, True, True, True)
        self.events: list[str] = []

    def observe(self, checkpoint: Checkpoint) -> Boundary:
        return self.before

    def verify_current(self, boundary: Boundary) -> None:
        self.events.append("verify")

    def normal_quit(self, boundary: Boundary) -> Quit:
        self.events.append("quit")
        return self.quit

    def force_quit(self, boundary: Boundary) -> Quit:
        self.events.append("force")
        return self.forced

    def launch_once(self, previous: Boundary) -> Boundary:
        self.events.append("launch")
        return self.after

    def future_data(self, checkpoint: Checkpoint, before: Boundary, after: Boundary) -> Pin:
        self.events.append("data")
        return self.future


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1] / "edits/work"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = Runtime(self.root)
        self.recovery = BoundedRecovery(self.root, self.runtime)
        self.checkpoint = Checkpoint(Config("immutable queue", (3, 4), "fixed-source"), "prepared")

    def test_optional_controller_callback_never_retries_pump(self) -> None:
        journal, steps = MemoryJournal(), Fake()
        steps.allowed = False
        self.assertEqual(
            run((3, 4), journal, steps, recovery=self.recovery).status, "resource-refused"
        )
        self.assertEqual(self.runtime.events, ["verify", "quit", "launch", "verify", "data"])
        self.assertEqual(journal.rows[3].phase, "prepared")
        self.assertEqual(steps.events, ["capacity3"])
        self.recovery.refused(self.checkpoint)
        self.assertEqual(self.runtime.events.count("quit"), 1)
        self.runtime.before = replace(self.runtime.after, physical_bytes=13000, pressure=2)
        self.recovery.refused(self.checkpoint)
        self.assertEqual(self.runtime.events.count("quit"), 1)

    def test_started_uncertain_and_active_lease_never_recover(self) -> None:
        journal, steps = MemoryJournal(), Fake()
        journal.rows[3] = Checkpoint(steps.materialize((3,)), "started")
        self.assertEqual(run((3,), journal, steps, recovery=self.recovery).status, "uncertain")
        self.assertEqual(self.runtime.events, [])
        with self.assertRaisesRegex(ContractError, "started"):
            self.recovery.refused(replace(self.checkpoint, phase="uncertain"))
        steps.allowed, steps.lease = False, True
        self.assertEqual(
            run((4,), journal, steps, recovery=self.recovery).status, "resource-refused"
        )
        self.assertEqual(self.runtime.events, [])

    def test_ineffective_recovery_and_unknown_quit_cannot_repeat(self) -> None:
        self.runtime.after = replace(self.runtime.before, pid=456, birth="birth-two")
        self.recovery.refused(self.checkpoint)
        self.assertNotIn("data", self.runtime.events)
        self.recovery.refused(self.checkpoint)
        self.assertEqual(self.runtime.events.count("quit"), 1)
        newer = self.root / "new-progress.json"
        newer.write_text('{"completed":"next real render"}', encoding="utf-8")
        self.runtime.before = replace(
            self.runtime.before, progress=Pin(newer, reference(newer)["sha256"])
        )
        self.runtime.quit = Quit(self.runtime.pin, True, False, False)
        with self.assertRaisesRegex(ContractError, "force is not allowed"):
            self.recovery.refused(self.checkpoint)
        self.recovery.refused(self.checkpoint)
        self.assertNotIn("force", self.runtime.events)
        self.assertEqual(self.runtime.events.count("quit"), 2)

    def test_force_only_after_failed_normal_unresponsive_and_actual_gone(self) -> None:
        self.runtime.quit = Quit(self.runtime.pin, True, True, False)
        self.recovery.refused(self.checkpoint)
        self.assertEqual(
            self.runtime.events, ["verify", "quit", "force", "launch", "verify", "data"]
        )

    def test_live_old_process_and_wrong_reopened_project_refuse(self) -> None:
        self.runtime.quit = Quit(self.runtime.pin, True, True, False)
        self.runtime.forced = Quit(self.runtime.pin, True, True, False)
        with self.assertRaisesRegex(ContractError, "still present"):
            self.recovery.refused(self.checkpoint)
        self.assertNotIn("launch", self.runtime.events)
        other = self.root / "other-recovery"
        other.mkdir()
        self.runtime.quit = Quit(self.runtime.pin, False, False, True)
        self.runtime.after = replace(self.runtime.after, project="wrong project")
        with self.assertRaisesRegex(ContractError, "exact saved"):
            BoundedRecovery(other, self.runtime).refused(self.checkpoint)

    def test_missing_idle_evidence_invalid_reason_or_root_refuse(self) -> None:
        with self.assertRaisesRegex(ContractError, "canonical"):
            BoundedRecovery(self.root / "missing", self.runtime)
        self.runtime.before = replace(self.runtime.before, no_export=False)
        with self.assertRaisesRegex(ContractError, "non-export"):
            self.recovery.refused(self.checkpoint)
        self.runtime.before = replace(self.runtime.before, no_export=True, reason="guess")
        with self.assertRaisesRegex(ContractError, "observed resource"):
            self.recovery.refused(self.checkpoint)
        self.assertEqual(self.runtime.events, [])

    def test_stale_readback_and_dispatchable_future_data_are_rejected(self) -> None:
        with (
            patch.object(
                self.runtime,
                "verify_current",
                side_effect=ContractError("stale owner or native observation"),
            ),
            self.assertRaisesRegex(ContractError, "stale"),
        ):
            self.recovery.refused(self.checkpoint)
        self.assertEqual(self.runtime.events, [])
        self.runtime.future = self.runtime.pin
        with self.assertRaisesRegex(ContractError, "future DATA only"):
            self.recovery.refused(self.checkpoint)
        self.recovery.refused(self.checkpoint)
        self.assertEqual(self.runtime.events.count("quit"), 1)


if __name__ == "__main__":
    unittest.main()
