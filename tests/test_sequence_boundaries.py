"""Real adapter branches with isolated mocked native boundaries; never real apps."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_native_sequence import Fake
from test_sequence_adapters import FakePreparation, pin

from replay import fcp_native_admission as admission
from replay import fcp_native_delivery_pump as pump
from replay import fcp_native_queue as queue
from replay import fcp_native_queue_followup as followup
from replay import sequence_journal
from replay.core import ReplayError
from replay.native_sequence import Checkpoint, Config, ContractError, run
from replay.sequence_adapters import (
    BoundSteps,
    Materialization,
    Member,
    OriginalPreparation,
    OriginalReleaseFactory,
    ProductionAPI,
    main,
    prepare_deferred,
    reference,
)
from replay.sequence_journal import (
    DurableJournal,
    _checkpoint,
    _record,
    bounded_read,
    decode,
    encode,
)


class Host:
    def birth(self, pid: int) -> str:
        return "synthetic-birth"

    def now_ns(self) -> int:
        return 100

    def free(self, path: Path) -> int:
        return 10**12

    def floor(self, path: Path) -> int:
        return 10**10

    def pressure(self) -> int:
        return 1

    def observe(self, path: Path) -> admission.FileState:
        return admission.FileState(False, False, False)


class Transport:
    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        raise AssertionError("No native transport calls are permitted in this suite")


class BoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1] / "edits/work"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "pump.json"
        self.config.write_text("{}")
        self.proof = self.root / "proof.json"
        self.proof.write_text("{}")
        native = queue.Queue(
            self.config,
            "digest",
            "synthetic",
            self.config,
            queue.Identity(123, "synthetic-birth"),
            "com.synthetic",
            self.root,
            self.root,
            (queue.Part("test", self.root / "absent.mov", (), 4, "Synthetic.fcpbundle"),),
            (),
            self.root / "inventory.json",
            100,
            30,
            30,
            400,
            100,
            1,
        )
        self.plan = pump.Pump(
            self.config,
            native,
            self.root,
            {"target": {"pid": 123}, "registry": "registry"},
            {},
            {},
            (),
        )
        self.api = ProductionAPI(loader=lambda path: self.plan)
        self.host, self.transport = Host(), Transport()
        self.lease = self.root / ".fcp-native-queue-owner.json"

    def observe(self, admitted: bool, *, fail: bool = False, wrong: bool = False) -> Path:
        def refresh(
            plan: pump.Pump, transport: queue.Transport, host: admission.Host, name: str
        ) -> None:
            directory = self.root / name
            directory.mkdir()
            value = {
                "admitted": admitted,
                "target": plan.template["target"],
                "registry": "wrong" if wrong else plan.template["registry"],
            }
            (directory / "snapshot.json").write_bytes(encode(value))
            if fail:
                raise ReplayError("explicit admission denial")

        with (
            patch.object(queue, "GUI_LEASE_ROOT", self.root),
            patch.object(pump, "_lease", autospec=True, return_value=(self.lease, "same")),
            patch.object(pump, "_refresh", autospec=True, side_effect=refresh),
        ):
            return self.api.observe_owned_capacity(
                pin(self.config), self.transport, self.host, "fresh"
            )

    def test_fresh_observation_lease_and_explicit_denial(self) -> None:
        result = self.observe(True)
        self.assertTrue(result.is_file())
        self.assertFalse(self.lease.exists())
        (self.root / "fresh" / "snapshot.json").unlink()
        (self.root / "fresh").rmdir()
        result = self.observe(False, fail=True)
        self.assertFalse(json.loads(result.read_text())["admitted"])
        self.assertFalse(self.lease.exists())

    def test_unknown_effect_and_changed_lease_are_preserved(self) -> None:
        with (
            patch.object(queue, "GUI_LEASE_ROOT", self.root),
            patch.object(pump, "_lease", autospec=True, return_value=(self.lease, "same")),
            patch.object(
                pump, "_refresh", autospec=True, side_effect=ReplayError("lost AX response")
            ),
            self.assertRaises(ReplayError),
        ):
            self.api.observe_owned_capacity(pin(self.config), self.transport, self.host, "lost")
        self.assertTrue(self.lease.exists())
        self.lease.unlink()
        with (
            patch.object(queue, "GUI_LEASE_ROOT", self.root),
            patch.object(
                pump,
                "_lease",
                autospec=True,
                side_effect=[(self.lease, "old"), (self.lease, "new")],
            ),
            patch.object(pump, "_refresh", autospec=True),
            self.assertRaises(ContractError),
        ):
            self.api.observe_owned_capacity(pin(self.config), self.transport, self.host, "changed")
        self.assertTrue(self.lease.exists())

    def test_wrong_denial_keeps_lease(self) -> None:
        with self.assertRaises(ReplayError):
            self.observe(False, fail=True, wrong=True)
        self.assertTrue(self.lease.exists())

    def test_original_wrappers_require_canonical_effect_and_pins(self) -> None:
        with (
            patch.object(
                pump, "run", autospec=True, return_value=pump.Result("uncertain", 4, None)
            ),
            self.assertRaises(ContractError),
        ):
            self.api.run_pump(pin(self.config), self.transport, self.host)
        with patch.object(
            pump,
            "run",
            autospec=True,
            return_value=pump.Result("published-candidate", 4, self.proof),
        ) as call:
            self.assertEqual(
                self.api.run_pump(pin(self.config), self.transport, self.host), self.proof
            )
            call.assert_called_once_with(self.plan, transport=self.transport, host=self.host)
        release = followup.Release(4, 5, "next", self.root / "next.mov", ())
        with patch.object(followup, "verify_release", autospec=True, return_value=release):
            self.assertEqual(self.api.verify_release(pin(self.config), self.host), release)
        with self.assertRaises(ContractError):
            self.api.refresh(pin(self.config), self.transport, self.host, "../escape")
        with (
            patch.object(queue, "load", autospec=True, return_value=self.plan.native) as q,
            patch.object(pump, "load", autospec=True, return_value=self.plan) as p,
        ):
            OriginalPreparation().check_queue(self.config)
            OriginalPreparation().check_pump(self.config)
            q.assert_called_once_with(self.config)
            p.assert_called_once_with(self.config)
        outcomes = iter((0, 1))
        failing = ProductionAPI(workflow_runner=lambda args: next(outcomes))
        with self.assertRaises(ContractError):
            failing.run_retirement(pin(self.config), self.root / "state", self.proof)

    def test_bound_callbacks_keep_original_artifacts(self) -> None:
        candidate = self.root / "candidate.json"
        candidate.write_bytes(
            encode(
                {
                    "candidateOnly": True,
                    "canDispatch": False,
                    "parts": [{"assemblyPart": 4}],
                    "inputs": [],
                }
            )
        )
        generated = self.root / "generated.json"
        spec = Materialization(
            pin(self.config), self.root, self.root / "queue.json", {}, generated, {}, (4,)
        )
        member = Member(spec, pin(self.config), self.root / "state", self.proof)
        factory = OriginalReleaseFactory({})
        bound = BoundSteps(
            (member,), self.transport, self.host, factory, self.api, FakePreparation(candidate)
        )
        config = bound.materialize((4,))
        bound.validate(Checkpoint(config, "prepared"))
        with self.assertRaises(ContractError):
            bound._pin(replace(config, identity="wrong"))
        with self.assertRaises(ContractError):
            bound.validate(Checkpoint(config, "published", "{}"))
        publication = encode(reference(self.proof)).decode()
        bound.validate(Checkpoint(config, "published", publication))
        with patch.object(ProductionAPI, "run_pump", autospec=True, return_value=self.proof):
            self.assertEqual(bound.pump(config), publication)
        with patch.object(
            ProductionAPI, "run_retirement", autospec=True, return_value=pin(self.proof)
        ):
            self.assertEqual(bound.retire(config, publication), publication)
        with patch.object(queue, "GUI_LEASE_ROOT", self.root):
            bound.close_observation_lease()
            self.lease.write_text("foreign")
            with self.assertRaises(ContractError):
                bound.close_observation_lease()
        with self.assertRaises(ContractError):
            BoundSteps((member, member), self.transport, self.host, factory)
        with self.assertRaises(ContractError):
            bound.release(config, config, "{}")
        snapshot = self.root / "snapshot.json"
        snapshot.write_text('{"admitted":false}')
        with patch.object(
            ProductionAPI, "observe_owned_capacity", autospec=True, return_value=snapshot
        ):
            self.assertFalse(bound.capacity(config))
            snapshot.write_text('{"admitted":true}')
            self.plan.native.inventory.write_text("different")
            with self.assertRaises(ContractError):
                bound.capacity(config)
            self.plan.native.inventory.write_bytes(snapshot.read_bytes())
            self.assertTrue(bound.capacity(config))
        with (
            patch.object(
                OriginalReleaseFactory, "request", autospec=True, return_value=pin(self.proof)
            ) as request,
            patch.object(
                ProductionAPI,
                "verify_release",
                autospec=True,
                return_value=followup.Release(4, 5, "next", self.root / "next.mov", ()),
            ),
        ):
            self.assertEqual(bound.release(config, config, publication), publication)
            self.assertEqual(request.call_count, 1)

    def test_journal_invalid_shapes_and_generation(self) -> None:
        malformed: tuple[object, ...] = ([], None, 42)
        for value in malformed:
            with self.assertRaises(ContractError):
                _record(value)
        invalid: tuple[object, ...] = ({}, {"config": {}, "phase": "unknown", "proof": ""})
        for value in invalid:
            with self.assertRaises(ContractError):
                _checkpoint(value)
        with self.assertRaises(ContractError):
            DurableJournal(self.root, (pin(self.config),))
        with self.assertRaises(ContractError):
            DurableJournal(self.root, ())
        with self.assertRaises(ContractError):
            DurableJournal(self.root, (pin(self.config), pin(self.config)))
        with self.assertRaises(ContractError):
            DurableJournal._transition(3, None, Checkpoint(Config("x", (4,), "d"), "prepared"))
        with self.assertRaises(ContractError):
            DurableJournal._transition(4, None, Checkpoint(Config("x", (4,), "d"), "released"))

    def test_bounded_reader_detects_growth_replace_symlink_and_duplicate_keys(self) -> None:
        original_read = os.read
        for attack in ("grow", "replace", "symlink"):
            self.config.write_text("{}")
            attacked = False

            def malicious_read(descriptor: int, count: int, selected: str = attack) -> bytes:
                nonlocal attacked
                data = original_read(descriptor, count)
                if not attacked:
                    attacked = True
                    if selected == "grow":
                        with self.config.open("a") as stream:
                            stream.write("x" * 100)
                    else:
                        self.config.unlink()
                        if selected == "symlink":
                            self.config.symlink_to(self.proof)
                        else:
                            self.config.write_text("{}")
                return data

            with (
                patch("replay.sequence_journal.os.read", side_effect=malicious_read),
                patch.object(sequence_journal, "MAX_BYTES", 32),
                self.assertRaises(ContractError),
            ):
                bounded_read(self.config)
            self.config.unlink()
        for data in (b'{"x":1,"x":2}', b'{"nested":{"x":1,"x":2}}', b'{"x":NaN}'):
            with self.assertRaises(ContractError):
                decode(data)

    def test_deferred_cli_two_members_reuse_and_namespace(self) -> None:
        xml = self.root / "test.fcpxml"
        xml.write_text("<fcpxml/>")
        actions = self.root / "actions.json"
        actions.write_bytes(
            encode(
                {
                    "parts": [
                        {
                            "part": number,
                            "xml": reference(xml),
                            "export": {
                                "project": f"Synthetic {number}",
                                "directory": str(self.root),
                                "fileName": f"new{number}.mov",
                            },
                        }
                        for number in (4, 5)
                    ]
                }
            )
        )
        future_registry, future_prefix = self.root / "registry.json", self.root / "prefix.json"
        retired, status = self.root / "retirement.json", self.root / "post.status"
        members: list[dict[str, object]] = []
        pending_admission: dict[str, object] = {}
        stable_evidence = self.root / "viewer-v15-evidence.json"
        stable_evidence.write_text("{}")
        for number, tail in ((4, [4, 5]), (5, [5])):
            root = self.root / str(number)
            root.mkdir()
            prep = root / "prepare.json"
            prep.write_bytes(
                encode(
                    {
                        "version": 1,
                        "kind": "native-queue-preparation-request",
                        "nativeActions": reference(actions),
                        "assemblyParts": tail,
                        "libraryBasename": "Synthetic.fcpbundle",
                        "event": "event",
                        "bundleID": "com.synthetic",
                        "backgroundTitle": "Background Tasks",
                    }
                )
            )
            target = {"pid": 123, "birth": "synthetic-birth", "bundleID": "com.synthetic"}
            grant = root / "grant.json"
            grant.write_bytes(
                encode(
                    {
                        "version": 1,
                        "kind": "native-gui-idle-handoff",
                        "released": True,
                        "eventRoot": str(root),
                        "target": target,
                        "previousOwner": "synthetic",
                    }
                )
            )
            qt, pt = root / "qt.json", root / "pt.json"
            qt.write_bytes(
                encode(
                    {
                        "provider": "synthetic",
                        "executor": reference(xml),
                        "guiGrant": reference(grant),
                        "target": target,
                        "eventRoot": str(root),
                        "bootVolume": str(self.root),
                        "inventory": str(root / "current-inventory.json"),
                        "limits": {
                            "maxCalls": 100,
                            "callSeconds": 30,
                            "observationSeconds": 30,
                            "peakAdditionalBytes": 400,
                            "minimumFreeBytes": 100,
                            "capacity": 1,
                        },
                    }
                )
            )
            registry = root / "owners.json"
            registry.write_bytes(
                encode(
                    {
                        "version": 1,
                        "kind": "native-admission-registry",
                        "generation": str(number),
                        "previous": None,
                        "complete": True,
                        "approvedOutputRoots": [str(self.root)],
                        "owners": [
                            {
                                "id": "native",
                                "process": {"pid": 123, "birth": "synthetic-birth"},
                                "role": "native",
                            }
                        ],
                        "outputs": [
                            {
                                "path": str(self.root / f"new{number}.mov"),
                                "owner": "native",
                                "state": "planned",
                                "reservedAdditionalBytes": 0,
                            }
                        ],
                    }
                )
            )
            admission_data = {
                "version": 1,
                "kind": "native-admission-preparation-request",
                "registry": reference(registry),
                "nativeObservation": None,
                "target": {"pid": 123, "birth": "synthetic-birth"},
                "nativeOwner": "native",
                "releases": [],
                "requested": {
                    "path": str(self.root / f"new{number}.mov"),
                    "peakAdditionalBytes": 400,
                },
                "policy": {
                    "maximumAgeSeconds": 30,
                    "maximumStagingCount": 4,
                    "maximumStagingBytes": 1000000,
                    "maximumActiveRenders": 1,
                    "maximumPressure": 1,
                    "minimumFreeBytes": 100,
                    "bootVolume": str(self.root),
                },
            }
            admission_path = root / "admission.json"
            if number == 4:
                admission_path.write_bytes(encode(admission_data))
            else:
                pending_admission = admission_data
            capability = root / "capability.json"
            capability.write_bytes(
                encode(
                    {
                        "ok": True,
                        "data": {"isError": False, "structuredContent": {"synthetic": True}},
                    }
                )
            )
            runtime = root / "runtime.json"
            runtime.write_bytes(
                encode(
                    {
                        "version": 1,
                        "kind": "native-delivery-runtime-evidence",
                        "executor": reference(xml),
                        "provider": "synthetic",
                        "target": {"pid": 123, "birth": "synthetic-birth"},
                        "capabilities": reference(capability),
                        "scratch": {
                            "roots": [str(root)],
                            "observedPaths": [str(root / "synthetic-cache")],
                        },
                    }
                )
            )
            pt.write_bytes(
                encode(
                    {
                        "outputRoot": str(root),
                        "runtimeEvidence": reference(runtime),
                        "followup": {
                            "assembly": reference(stable_evidence),
                            "publishedAssembly": {"$ref": "prefix"}
                            if number == 5
                            else reference(stable_evidence),
                            "sourceCleanup": reference(stable_evidence),
                            "ffmpeg": reference(xml),
                            "ffprobe": reference(xml),
                            "audioSampleRate": 48000,
                            "audioChannels": 2,
                            "receiptOutput": str(root / "publication.json"),
                        },
                        "admissionTemplate": {"$ref": "registry"}
                        if number == 5
                        else reference(admission_path),
                    }
                )
            )
            members.append(
                {
                    "part": number,
                    "preparationRequest": reference(prep),
                    "candidateRoot": str(root / "candidate"),
                    "queuePath": str(root / "queue.json"),
                    "queueTemplate": reference(qt),
                    "pumpPath": str(root / "pump.json"),
                    "pumpTemplate": reference(pt),
                    "dependencies": {"prefix": str(future_prefix), "registry": str(future_registry)}
                    if number == 5
                    else {},
                    "predecessor": {
                        "part": 4,
                        "retirementConfig": str(retired),
                        "postDeleteStatus": str(status),
                    }
                    if number == 5
                    else None,
                }
            )
        request = self.root / "sequence.json"
        request.write_bytes(
            encode({"version": 1, "kind": "native-sequence-data", "members": members})
        )
        with patch("replay.sequence_adapters.verify_retirement", autospec=True) as verify:
            first = prepare_deferred(pin(request), 4, OriginalPreparation())
            self.assertEqual(first.tail, (4, 5))
            with patch.object(
                queue,
                "prepare_data",
                autospec=True,
                side_effect=AssertionError("Producer rerun forbidden"),
            ):
                for phase in ("prepared", "released"):
                    self.assertEqual(
                        prepare_deferred(pin(request), 4, OriginalPreparation()), first, phase
                    )
            with self.assertRaises(FileNotFoundError):
                prepare_deferred(pin(request), 5, OriginalPreparation())
            self.assertEqual(verify.call_count, 0)
            status.write_text("1")
            with self.assertRaises(ContractError):
                prepare_deferred(pin(request), 5, OriginalPreparation())
            status.write_text("0")
            retired.write_text("{}")
            verify.side_effect = ReplayError("source present; pre-delete is not post-delete")
            with self.assertRaises(ReplayError):
                prepare_deferred(pin(request), 5, OriginalPreparation())
            verify.side_effect = None
            future_prefix.write_text('{"part":4}')
            future_registry.write_bytes(encode(pending_admission))
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "prepare-member",
                            str(request),
                            "--sha256",
                            pin(request).sha256,
                            "--part",
                            "5",
                        ]
                    ),
                    0,
                )
            reply = decode(output.getvalue())
            self.assertIs(reply["canDispatch"], False)
            generated = decode(bounded_read(Path(str(reply["config"]))))
            self.assertEqual(
                _record(generated["followup"])["sourceCleanup"], reference(stable_evidence)
            )
            self.assertEqual(generated["admissionTemplate"], reference(future_registry))

            class DeferredController(Fake):
                def materialize(self, tail: tuple[int, ...]) -> Config:
                    return prepare_deferred(pin(request), tail[0], OriginalPreparation())

            journal_root = self.root / "journal"
            journal_root.mkdir()
            module = Path(sequence_journal.__file__).resolve()
            journal = DurableJournal(
                journal_root,
                tuple(
                    pin(path)
                    for path in (
                        module,
                        module.with_name("native_sequence.py"),
                        module.with_name("sequence_adapters.py"),
                        request,
                    )
                ),
            )
            steps = DeferredController()
            steps.allowed = False
            with patch.object(
                queue,
                "prepare_data",
                autospec=True,
                side_effect=AssertionError("No producer on resume"),
            ):
                self.assertEqual(run((4, 5), journal, steps).status, "resource-refused")
                steps.allowed = True
                self.assertEqual(run((4, 5), journal, steps).status, "complete")
                before = steps.events[:]
                self.assertEqual(run((4, 5), journal, steps).status, "complete")
                self.assertEqual(steps.events, before)
