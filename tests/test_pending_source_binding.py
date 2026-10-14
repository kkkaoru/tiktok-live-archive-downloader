"""Pending publication requires actual upstream completion and immutable pins."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import completion_barrier as barrier
from replay import pending_source_binding as pending
from replay import source_caption_binding as binding
from replay import workflow
from replay.artifact_json import MAX_JSON_BYTES
from replay.caption_consumer_binding import MAX_JSON_BYTES as MAX_CAPTION_RECEIPT_BYTES
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.stages import Stage, file_digest


@pytest.fixture
def pending_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(pending, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    original = work / "original.mov"
    original.write_bytes(b"original")
    clean = work / "clean.mov"
    clean.write_bytes(b"genuine cleanup fixture")
    voice = work / "voice.wav"
    voice.write_bytes(b"voice fixture")
    voice_receipt = work / "voice.json"
    voice_receipt.write_text(json.dumps({"outputSHA256": file_digest(voice)}))
    manifest = work / "manifest.json"
    manifest.write_text("{}")
    assembly = work / "assembly.json"
    assembly.write_text("{}")
    index = work / "index.json"
    index.write_text("{}")
    semantic = work / "semantic.fcpxml"
    semantic.write_bytes(b"synthetic semantic XML")
    child = work / "child.json"
    child.write_text('{"status":"complete"}')
    cleanup_completion = work / "cleanup-completion.json"
    cleanup_completion.write_text(
        json.dumps(
            {
                "version": 1,
                "assemblyReceipt": _reference(assembly),
                "output": _reference(clean),
                "childStageReceipt": _reference(child),
            }
        )
    )
    head_completion = work / "head-completion.json"
    head_completion.write_text(
        json.dumps(
            {
                "semanticXML": _reference(semantic),
                "readyForRender": False,
                "requiredNextStage": "verified-clean-source-substitution",
            }
        )
    )
    path = work / "PENDING.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "pending-actual-pins-not-executable",
                "fixedFields": {
                    "version": 1,
                    "terminalFrameCell": True,
                    "removeSourceMasks": True,
                    "voice": _reference(voice),
                    "voiceReceipt": _reference(voice_receipt),
                },
                "futurePinFields": {
                    "assemblyManifest": str(manifest),
                    "assemblyReceipt": str(assembly),
                    "cleanIndex": str(index),
                    "semanticXML": str(semantic),
                },
                "requiredCompletions": [str(cleanup_completion), str(head_completion)],
                "configOutput": str(work / "config.json"),
                "bindingOutput": str(work / "binding.json"),
                "bindingConsumerKind": "transformed-source-config",
                "readyForRender": False,
            }
        )
    )
    monkeypatch.setattr(binding, "clean_source", lambda config: (original, clean, Fraction(2)))
    return path


def test_bind_verify_nonoverwrite(pending_request: Path) -> None:
    digest = file_digest(pending_request)
    result = pending.bind(pending_request, sha256=digest)
    assert result.name == "binding.pending-source.json"
    assert pending.verify(pending_request, sha256=digest) == result
    assert json.loads(result.read_text())["readyForRender"] is False
    with pytest.raises(ReplayError, match="already exists"):
        pending.bind(pending_request, sha256=digest)


@pytest.mark.parametrize("padding", [32, MAX_JSON_BYTES + 1])
def test_semantic_measurement_receipt_budget(pending_request: Path, padding: int) -> None:
    head = pending_request.with_name("head-completion.json")
    document = json.loads(head.read_text())
    document["measurementLineage"] = "x" * padding
    head.write_text(json.dumps(document))
    digest = file_digest(pending_request)
    result = pending.bind(pending_request, sha256=digest)
    assert pending.verify(pending_request, sha256=digest) == result


def test_cleanup_completion_keeps_ordinary_budget(pending_request: Path) -> None:
    cleanup = pending_request.with_name("cleanup-completion.json")
    document = json.loads(cleanup.read_text())
    document["padding"] = "x" * (MAX_JSON_BYTES + 1)
    cleanup.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="byte budget"):
        pending.bind(pending_request, sha256=file_digest(pending_request))
    assert not pending_request.with_name("config.json").exists()


def test_semantic_completion_exceeding_caption_budget(pending_request: Path) -> None:
    head = pending_request.with_name("head-completion.json")
    with head.open("r+b") as stream:
        stream.truncate(MAX_CAPTION_RECEIPT_BYTES + 1)
    with pytest.raises(ReplayError, match="byte budget"):
        pending.bind(pending_request, sha256=file_digest(pending_request))
    assert not pending_request.with_name("config.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("readyForRender", True),
        ("bindingConsumerKind", "fcpxml"),
        ("kind", "executable"),
        ("requiredCompletions", []),
        ("futurePinFields", {}),
        ("fixedFields", {}),
    ],
)
def test_schema_rejects(pending_request: Path, field: str, value: object) -> None:
    data = json.loads(pending_request.read_text())
    data[field] = value
    pending_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        pending.bind(pending_request, sha256=file_digest(pending_request))
    assert not pending_request.with_name("config.json").exists()


@pytest.mark.parametrize(
    "name", ["semantic.fcpxml", "cleanup-completion.json", "head-completion.json"]
)
def test_missing_producer(pending_request: Path, name: str) -> None:
    pending_request.with_name(name).unlink()
    with pytest.raises(FileNotFoundError):
        pending.bind(pending_request, sha256=file_digest(pending_request))
    assert not pending_request.with_name("config.json").exists()


@pytest.mark.parametrize(
    "name,document",
    [
        ("child.json", {"status": "running"}),
        ("head-completion.json", {"semanticXML": {"path": "/wrong", "sha256": "0" * 64}}),
        ("voice.json", {"outputSHA256": "wrong"}),
        ("cleanup-completion.json", {}),
    ],
)
def test_tampered_completion(pending_request: Path, name: str, document: object) -> None:
    pending_request.with_name(name).write_text(json.dumps(document))
    with pytest.raises(ReplayError):
        pending.bind(pending_request, sha256=file_digest(pending_request))


def test_voice_receipt_correspondence(pending_request: Path) -> None:
    receipt = pending_request.with_name("voice.json")
    receipt.write_text('{"outputSHA256":"wrong"}')
    data = json.loads(pending_request.read_text())
    data["fixedFields"]["voiceReceipt"] = _reference(receipt)
    pending_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="Processed voice"):
        pending.bind(pending_request, sha256=file_digest(pending_request))


@pytest.mark.parametrize(
    "name", ["semantic.fcpxml", "binding.json", "config.json", "binding.pending-source.json"]
)
def test_changed_retained_input_or_output(pending_request: Path, name: str) -> None:
    digest = file_digest(pending_request)
    pending.bind(pending_request, sha256=digest)
    target = pending_request.with_name(name)
    data: object = {} if target.suffix == ".json" else "changed"
    target.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        pending.verify(pending_request, sha256=digest)


def test_input_changed_during_validation(
    pending_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = binding.create_binding

    def changed(cleanup: binding.Cleanup, *, consumer: Path, kind: str, output: Path) -> Path:
        result = original(cleanup, consumer=consumer, kind=kind, output=output)
        pending_request.with_name("head-completion.json").write_text("{}")
        return result

    monkeypatch.setattr(binding, "create_binding", changed)
    with pytest.raises(ReplayError, match="changed during"):
        pending.bind(pending_request, sha256=file_digest(pending_request))
    assert not pending_request.with_name("binding.pending-source.json").exists()


def test_paths_and_request_sha(pending_request: Path) -> None:
    with pytest.raises(ReplayError, match="hash changed"):
        pending.bind(pending_request, sha256="0" * 64)
    with pytest.raises(ReplayError, match="inside checkout"):
        pending._local(str(pending_request.parents[2] / "escape.json"))
    data = json.loads(pending_request.read_text())
    data["bindingOutput"] = data["configOutput"]
    pending_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="distinct"):
        pending.bind(pending_request, sha256=file_digest(pending_request))


def test_domain_provenance_failure(pending_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(config: dict[str, object]) -> tuple[Path, Path, Fraction]:
        raise ReplayError("actual source provenance rejected")

    monkeypatch.setattr(binding, "clean_source", failed)
    with pytest.raises(ReplayError, match="provenance rejected"):
        pending.bind(pending_request, sha256=file_digest(pending_request))


def test_cli(
    pending_request: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["pending", "bind", str(pending_request), "--sha256", file_digest(pending_request)],
    )
    pending.main()
    assert capsys.readouterr().out.endswith("binding.pending-source.json\n")


def test_prepare_graph(pending_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = pending_request.parent
    producer = work / "producer.json"
    producer.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "producer",
                        "timeout_seconds": 60,
                        "inputs": [str(pending_request)],
                        "outputs": [str(work / "future.json")],
                        "command": ["python", "-m", "synthetic"],
                        "verify_command": ["python", "-m", "synthetic", "--verify"],
                    }
                ],
            }
        )
    )
    capture = work / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "workflow": _reference(producer),
                "stage": "producer",
                "checkpoint": str(work / "state/producer.json"),
                "statusPath": str(work / "status"),
            }
        )
    )
    template = work / "template.json"
    template.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "transform",
                        "timeout_seconds": 60,
                        "inputs": [str(work / "config.json"), str(work / "binding.json")],
                        "outputs": [str(work / "clean.fcpxml")],
                        "command": ["python", "synthetic"],
                        "verify_command": ["python", "synthetic", "--verify"],
                    }
                ],
            }
        )
    )
    output = work / "continuation"
    output.mkdir()
    monkeypatch.setattr(barrier, "load", lambda path, digest: None)
    graph = pending.prepare_graph(
        pending_request,
        sha256=file_digest(pending_request),
        captures=(capture,),
        template=template,
        output=output,
    )
    data = json.loads(graph.read_text())
    assert [row["name"] for row in data["stages"]] == ["wait-producer-00", "run-clean-consumer"]
    assert data["stages"][0]["command"][3] == "wait"
    assert data["stages"][0]["verify_command"][3] == "verify"
    child = json.loads((output / "child-workflow.json").read_text())
    assert child["stages"][0]["verify_command"][3] == "verify"
    assert child["stages"][1]["command"] == ["python", "synthetic"]
    assert data["stages"][1]["command"] == data["stages"][1]["verify_command"]
    work.joinpath("semantic.fcpxml").unlink()
    work.joinpath("cleanup-completion.json").unlink()

    def reaches_wait(
        stages: tuple[Stage, ...],
        *,
        graph: dict[str, tuple[str, ...]],
        state: Path,
        source_caption_bindings: dict[str, Path],
    ) -> None:
        assert stages[0].name == "wait-producer-00"
        raise RuntimeError("runtime preflight reached wait")

    monkeypatch.setattr(workflow, "_run_selected", reaches_wait)
    with pytest.raises(RuntimeError, match="runtime preflight reached wait"):
        workflow.run_plan(graph, state_directory=output / "state")
    with pytest.raises(ReplayError, match="empty existing"):
        pending.prepare_graph(
            pending_request,
            sha256=file_digest(pending_request),
            captures=(capture,),
            template=template,
            output=output,
        )


def test_prepare_cli_requires_arguments(
    pending_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["pending", "prepare", str(pending_request), "--sha256", file_digest(pending_request)],
    )
    with pytest.raises(SystemExit) as error:
        pending.main()
    assert error.value.code == 2
