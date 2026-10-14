"""Synthetic only: no application, filesystem media, or process effects."""

import unittest

from replay.native_sequence import Checkpoint, Config, ContractError, run


class MemoryJournal:
    def __init__(self) -> None:
        self.rows: dict[int, Checkpoint] = {}

    def get(self, part: int) -> Checkpoint | None:
        return self.rows.get(part)

    def put(self, part: int, old: Checkpoint | None, new: Checkpoint) -> None:
        if self.rows.get(part) != old:
            raise ContractError("CAS mismatch")
        self.rows[part] = new


class Fake:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.allowed = True
        self.uncertain = False
        self.lease = False
        self.sticky = False
        self.digest = "fixed"
        self.retirement = "retirement"
        self.release_proof = "release"
        self.bad_tail = False
        self.crash = False

    def materialize(self, tail: tuple[int, ...]) -> Config:
        return Config(str(tail[0]), tail[::-1] if self.bad_tail else tail, self.digest)

    def validate(self, checkpoint: Checkpoint) -> None:
        if checkpoint.phase in ("published", "retired", "released") and not checkpoint.proof:
            raise ContractError("Missing proof")

    def capacity(self, config: Config) -> bool:
        self.events.append("capacity" + config.identity)
        return self.allowed

    def pump(self, config: Config) -> str | None:
        self.events.append("pump" + config.identity)
        self.lease = True
        if self.crash:
            raise RuntimeError("lost response")
        return None if self.uncertain else "publication"

    def retire(self, config: Config, publication: str) -> str:
        self.events.append("retire" + config.identity)
        return self.retirement

    def close_observation_lease(self) -> None:
        self.events.append("close")
        self.lease = self.sticky

    def leased(self) -> bool:
        return self.lease

    def release(self, current: Config, successor: Config, retirement: str) -> str:
        if self.lease or current.tail[1:] != successor.tail:
            raise ContractError("Unsafe release")
        self.events.append("release" + successor.identity)
        return self.release_proof


class SequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.journal = MemoryJournal()
        self.steps = Fake()

    def test_success_and_reentry(self) -> None:
        for parts in ((4, 5), (8, 11, 16)):
            journal, steps = MemoryJournal(), Fake()
            self.assertEqual(run(parts, journal, steps).status, "complete")
            before = steps.events[:]
            self.assertEqual(run(parts, journal, steps).status, "complete")
            self.assertEqual(before, steps.events)
            self.assertLess(
                steps.events.index("close"), steps.events.index("release" + str(parts[1]))
            )
            self.assertLess(
                steps.events.index("release" + str(parts[1])),
                steps.events.index("pump" + str(parts[1])),
            )

    def test_uncertain_and_crash_never_retry(self) -> None:
        self.steps.uncertain = True
        self.assertEqual(run((1, 2), self.journal, self.steps).status, "uncertain")
        self.steps.uncertain = False
        self.assertEqual(run((1, 2), self.journal, self.steps).status, "uncertain")
        self.assertEqual(self.steps.events.count("pump1"), 1)
        journal, steps = MemoryJournal(), Fake()
        steps.crash = True
        with self.assertRaises(RuntimeError):
            run((1,), journal, steps)
        steps.crash = False
        self.assertEqual(run((1,), journal, steps).status, "uncertain")
        self.assertEqual(steps.events.count("pump1"), 1)

    def test_resource_refusal_is_resumable(self) -> None:
        self.steps.allowed = False
        self.assertEqual(run((1, 2), self.journal, self.steps).status, "resource-refused")
        self.assertNotIn("pump1", self.steps.events)
        self.steps.allowed = True
        self.assertEqual(run((1, 2), self.journal, self.steps).status, "complete")

    def test_leased_release_and_immutable_successor(self) -> None:
        self.steps.sticky = True
        with self.assertRaises(ContractError):
            run((1, 2), self.journal, self.steps)
        self.assertEqual(self.journal.rows[1].phase, "retired")
        self.assertNotIn("pump2", self.steps.events)
        self.steps.sticky = False
        self.journal.rows[2] = Checkpoint(Config("2", (2,), "changed"), "prepared")
        with self.assertRaises(ContractError):
            run((1, 2), self.journal, self.steps)
        self.assertEqual(self.steps.events.count("pump1"), 1)

    def test_release_capacity_refusal_reuses_retirement(self) -> None:
        self.journal.rows[1] = Checkpoint(Config("1", (1, 2), "fixed"), "retired", "proof")
        self.steps.allowed = False
        self.assertEqual(run((1, 2), self.journal, self.steps).status, "resource-refused")
        self.steps.allowed = True
        self.assertEqual(run((1, 2), self.journal, self.steps).status, "complete")
        self.assertNotIn("pump1", self.steps.events)
        self.assertNotIn("retire1", self.steps.events)

    def test_bad_contracts(self) -> None:
        for parts in ((), (1, 1), (0,), tuple(range(1, 1026))):
            with self.assertRaises(ContractError):
                run(parts, MemoryJournal(), Fake())
        self.steps.bad_tail = True
        with self.assertRaises(ContractError):
            run((1, 2), self.journal, self.steps)
        self.steps.bad_tail = False
        self.steps.retirement = ""
        with self.assertRaises(ContractError):
            run((1, 2), self.journal, self.steps)
        self.steps.retirement = "retirement"
        self.steps.release_proof = ""
        with self.assertRaises(ContractError):
            run((1, 2), self.journal, self.steps)


if __name__ == "__main__":
    unittest.main()
