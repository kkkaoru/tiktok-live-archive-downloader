"""Private scratch-only tests. No production API is invoked against real media."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from replay import sequence_journal
from replay.native_sequence import Checkpoint, Config, ContractError
from replay.sequence_adapters import (
    Materialization,
    OriginalPreparation,
    OriginalReleaseFactory,
    ProductionAPI,
    ReleaseInputs,
    materialize,
    reference,
    save_identical,
)
from replay.sequence_journal import DurableJournal, Pin, encode, publish


def pin(path: Path) -> Pin:
    return Pin(path, hashlib.sha256(path.read_bytes()).hexdigest())


class FakePreparation:
    def __init__(self, candidate: Path) -> None:
        self.path = candidate
        self.checked: list[Path] = []

    def candidate(self, request: Path, output: Path) -> Path:
        return self.path

    def reuse_candidate(self, request: Path, output: Path) -> Path:
        return self.path

    def check_queue(self, path: Path) -> None:
        self.checked.append(path)

    def check_pump(self, path: Path) -> None:
        self.checked.append(path)


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1] / "edits/work"
        )
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.input = self.root / "input.json"
        self.input.write_text("{}")
        module = Path(sequence_journal.__file__).resolve()
        self.pins = tuple(
            pin(path)
            for path in (
                module,
                module.with_name("native_sequence.py"),
                module.with_name("sequence_adapters.py"),
                self.input,
            )
        )

    def test_durable_reopen_chain_and_cas(self) -> None:
        journal = DurableJournal(self.root, self.pins)
        config = Config("a", (4, 5), "fixed")
        prepared = Checkpoint(config, "prepared")
        journal.put(4, None, prepared)
        with self.assertRaises(ContractError):
            journal.put(4, None, prepared)
        started = Checkpoint(config, "started")
        journal.put(4, prepared, started)
        reopened = DurableJournal(self.root, self.pins)
        self.assertEqual(reopened.get(4), started)
        with self.assertRaises(ContractError):
            reopened.put(4, started, Checkpoint(config, "released", "fake"))
        path = self.root / "part-4-01.json"
        value = json.loads(path.read_text())
        value["previous"] = "changed"
        path.write_text(json.dumps(value))
        with self.assertRaises(ContractError):
            reopened.get(4)

    def test_changed_pins_gap_and_symlink(self) -> None:
        journal = DurableJournal(self.root, self.pins)
        self.input.write_text('{"changed":true}')
        with self.assertRaises(ContractError):
            journal.get(1)
        self.input.write_text("{}")
        (self.root / "part-1-01.json").write_text("{}")
        with self.assertRaises(ContractError):
            journal.get(1)
        target = self.root / "link"
        target.symlink_to(self.input)
        with self.assertRaises(ContractError):
            publish(target, b"replace")
        self.assertEqual(self.input.read_text(), "{}")

    def test_nonoverwrite_and_tail_materialization(self) -> None:
        candidate = self.root / "candidate.json"
        candidate.write_bytes(
            encode(
                {
                    "candidateOnly": True,
                    "canDispatch": False,
                    "parts": [{"assemblyPart": 4}, {"assemblyPart": 5}],
                    "inputs": [],
                }
            )
        )
        api = FakePreparation(candidate)
        spec = Materialization(
            pin(self.input),
            self.root,
            self.root / "queue.json",
            {"guiGrant": "actual-supplied"},
            self.root / "pump.json",
            {"runtimeEvidence": "actual-supplied"},
            (4, 5),
        )
        first = materialize(spec, api)
        self.assertEqual(materialize(spec, api), first)
        self.assertEqual(first.tail, (4, 5))
        self.assertEqual(len(api.checked), 4)
        with self.assertRaises(ContractError):
            save_identical(spec.queue_path, {"changed": True})
        spec.queue_fields["parts"] = []
        with self.assertRaises(ContractError):
            materialize(spec, api)

    def test_retirement_preflight_and_original_state(self) -> None:
        terminal = self.root / "terminal.json"
        terminal.write_text('{"complete":true}')
        calls: list[list[str]] = []

        def fake(arguments: list[str]) -> int:
            calls.append(arguments)
            return 0

        api = ProductionAPI(workflow_runner=fake)
        result = api.run_retirement(pin(self.input), self.root / "state", terminal)
        self.assertEqual(result, pin(terminal))
        self.assertEqual(
            calls[0], [str(self.input), "--state", str(self.root / "state"), "--check"]
        )
        self.assertEqual(calls[1], calls[0][:-1])
        failed = ProductionAPI(workflow_runner=lambda arguments: 1)
        with self.assertRaises(ContractError):
            failed.run_retirement(pin(self.input), self.root / "state", terminal)

    def test_actual_original_candidate_api_two_tails(self) -> None:
        from replay import fcp_native_queue as original

        xml = self.root / "synthetic.fcpxml"
        xml.write_text("<fcpxml/>")
        actions = self.root / "actions.json"
        actions.write_bytes(
            encode(
                {
                    "parts": [
                        {
                            "part": number,
                            "xml": reference(xml),
                            "duration": duration,
                            "export": {
                                "project": f"Synthetic {number}",
                                "directory": str(self.root),
                                "fileName": f"unique-{number}.mov",
                            },
                        }
                        for number, duration in ((4, "101/25s"), (5, "137/30s"))
                    ]
                }
            )
        )
        request = self.root / "request.json"
        request.write_bytes(
            encode(
                {
                    "version": 1,
                    "kind": "native-queue-preparation-request",
                    "nativeActions": reference(actions),
                    "assemblyParts": [4, 5],
                    "libraryBasename": "Synthetic.fcpbundle",
                    "event": "Synthetic event",
                    "bundleID": "com.apple.FinalCutApp",
                    "backgroundTitle": "Background Tasks",
                }
            )
        )
        api = OriginalPreparation()
        result = api.candidate(request, self.root / "candidate")
        self.assertEqual(
            tuple(part.assembly_part for part in original.validate_candidate(result)), (4, 5)
        )
        before = result.read_bytes()
        self.assertEqual(api.candidate(request, self.root / "candidate").read_bytes(), before)

    def test_release_factory_pins_original_closure(self) -> None:
        names = (
            "preparation",
            "terminal",
            "publication",
            "retirement",
            "postDeleteStatus",
            "retirementAudit",
            "consumer",
        )
        artifacts = {name: self.root / f"{name}.json" for name in names}
        for path in artifacts.values():
            path.write_text("{}")
        output = self.root / "release.json"
        spec = ReleaseInputs(output, artifacts, (self.input,), self.input)
        factory = OriginalReleaseFactory({(4, 5): spec})
        current, successor = Config("a", (4, 5), "a"), Config("b", (5,), "b")
        result = factory.request(current, successor, pin(artifacts["retirement"]), pin(self.input))
        self.assertEqual(result, pin(output))
        self.assertEqual(json.loads(output.read_text())["successorOrdinal"], 1)
        self.assertEqual(
            factory.request(current, successor, pin(artifacts["retirement"]), pin(self.input)),
            result,
        )
        prior = output.read_bytes()
        capacity = self.root / "fresh-capacity.json"
        capacity.write_text('{"capturedAt": 2}')
        retried = factory.request(current, successor, pin(artifacts["retirement"]), pin(capacity))
        self.assertNotEqual(retried.path, output)
        self.assertEqual(output.read_bytes(), prior)
        self.assertEqual(
            factory.request(current, successor, pin(artifacts["retirement"]), pin(capacity)),
            retried,
        )
        artifacts["consumer"].write_text('{"changed": true}')
        with self.assertRaisesRegex(ContractError, "closure changed"):
            factory.request(current, successor, pin(artifacts["retirement"]), pin(capacity))
        with self.assertRaises(ContractError):
            factory.request(
                current, Config("wrong", (6,), "c"), pin(artifacts["retirement"]), pin(self.input)
            )


if __name__ == "__main__":
    unittest.main()
