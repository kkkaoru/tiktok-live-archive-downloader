"""Actual clean provenance and exact window bytes remain mandatory for pilots."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
from test_completion_barrier import raw_request as raw_request
from test_completion_barrier import write_request

from replay import fcp_transformed_source as transformed
from replay import post_clean_xml_plan as previews
from replay import source_caption_binding as binding
from replay import workflow
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.stages import Stage, file_digest


@pytest.fixture
def plan_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(previews, "ROOT", tmp_path)
    work = tmp_path / "edits/work/preview"
    work.mkdir(parents=True)
    clean = work / "clean.mov"
    clean.write_bytes(b"synthetic genuine clean bytes")
    source = work / "clean.fcpxml"
    source.write_text(f'''<fcpxml version="1.11"><resources>
      <format id="f" frameDuration="1/30s" width="1080" height="1920"/>
      <asset id="v" hasVideo="1"><media-rep src="{clean.as_uri()}"/></asset>
      </resources><library><event><project name="clean"><sequence format="f" duration="120s">
      <spine><asset-clip ref="v" offset="0s" start="0s" duration="120s">
      <title offset="0s" start="3600s" duration="120s" ref="t"><text>actual phrase</text>
      <text>actual phrase</text><text>actual phrase</text></title>
      </asset-clip></spine></sequence></project></event></library></fcpxml>''')
    source.with_suffix(".source-lineage.json").write_text('{"captionClockEqual":true}')
    candidate = work / "candidate.json"
    candidate.write_text("{}")
    producer = work / "producer.json"
    producer.write_text(json.dumps({"outputDirectory": str(work)}))
    work.joinpath("candidate.mp4").write_bytes(b"genuine candidate")
    index = work / "index.json"
    index.write_text("{}")
    manifest = work / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "index": _reference(index),
                "chunks": [{"candidate": _reference(candidate), "producer": _reference(producer)}],
            }
        )
    )
    receipt = work / "assembly.json"
    receipt.write_text(
        json.dumps({"output": _reference(clean), "probeEvidence": {str(index): file_digest(index)}})
    )
    config = work / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "removeSourceMasks": True,
                "assemblyManifest": _reference(manifest),
                "assemblyReceipt": _reference(receipt),
                "cleanIndex": _reference(index),
            }
        )
    )
    source_binding = work / "source-binding.json"
    source_binding.write_text(
        json.dumps(
            {
                **json.loads(config.read_text()),
                "consumer": {**_reference(config), "kind": "transformed-source-config"},
            }
        )
    )
    completion = work / "clean-completion.json"
    completion.write_text(
        json.dumps({"status": "complete", "outputs": {str(source): file_digest(source)}})
    )
    request = work / "REQUEST.json"
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "post-clean-xml-plan",
                "sourceConfig": str(config),
                "sourceBinding": str(source_binding),
                "sourceXML": str(source),
                "completion": str(completion),
                "windows": [
                    {
                        "start": "0",
                        "end": "60",
                        "output": str(work / "preview-60.fcpxml"),
                        "bindingOutput": str(work / "preview-60.binding.json"),
                    },
                    {
                        "start": "0",
                        "end": "90",
                        "output": str(work / "preview-90.fcpxml"),
                        "bindingOutput": str(work / "preview-90.binding.json"),
                    },
                ],
                "receiptOutput": str(work / "preview-proof.json"),
            }
        )
    )
    monkeypatch.setattr(
        binding, "clean_source", lambda config: (work / "original.mov", clean, Fraction(120))
    )
    monkeypatch.setattr(transformed, "verify", lambda config, source: None)
    return request


def test_actual_window_generation_and_readonly_verify(plan_request: Path) -> None:
    digest = file_digest(plan_request)
    receipt = previews.execute(plan_request, sha256=digest)
    assert receipt.name == "preview-proof.json"
    assert previews.execute(plan_request, sha256=digest, verify=True) == receipt
    proof = json.loads(receipt.read_text())
    assert [row["end"] for row in proof["windows"]] == ["60", "90"]
    assert proof["nativeRenderVerified"] is False
    assert proof["qualityApproved"] is False
    with pytest.raises(ReplayError, match="already exist"):
        previews.execute(plan_request, sha256=digest)


@pytest.mark.parametrize("field,value", [("version", 2), ("windows", []), ("kind", "native-ready")])
def test_schema_rejected(plan_request: Path, field: str, value: object) -> None:
    data = json.loads(plan_request.read_text())
    data[field] = value
    plan_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="schema"):
        previews.execute(plan_request, sha256=file_digest(plan_request))


@pytest.mark.parametrize("start,end", [("bad", "60"), ("0", "1/0"), ("60", "60"), ("-1", "60")])
def test_invalid_clock(plan_request: Path, start: str, end: str) -> None:
    data = json.loads(plan_request.read_text())
    data["windows"][0]["start"] = start
    data["windows"][0]["end"] = end
    plan_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        previews.execute(plan_request, sha256=file_digest(plan_request))


@pytest.mark.parametrize("name", ["config.json", "clean-completion.json", "clean.fcpxml"])
def test_missing_actual_input(plan_request: Path, name: str) -> None:
    plan_request.with_name(name).unlink()
    with pytest.raises(FileNotFoundError):
        previews.execute(plan_request, sha256=file_digest(plan_request))


@pytest.mark.parametrize(
    "name", ["preview-60.fcpxml", "preview-60.binding.json", "preview-proof.json"]
)
def test_output_tamper(plan_request: Path, name: str) -> None:
    digest = file_digest(plan_request)
    previews.execute(plan_request, sha256=digest)
    plan_request.with_name(name).write_text("{}")
    with pytest.raises(ReplayError):
        previews.execute(plan_request, sha256=digest, verify=True)


def test_running_checkpoint_rejected(plan_request: Path) -> None:
    plan_request.with_name("clean-completion.json").write_text('{"status":"running"}')
    with pytest.raises(ReplayError, match="checkpoint"):
        previews.execute(plan_request, sha256=file_digest(plan_request))


def test_domain_lineage_failure(plan_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(config: Path, source: Path) -> None:
        raise ReplayError("clean lineage rejected")

    monkeypatch.setattr(transformed, "verify", reject)
    with pytest.raises(ReplayError, match="lineage rejected"):
        previews.execute(plan_request, sha256=file_digest(plan_request))


def test_source_changes_during_generation(
    plan_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = binding.create_binding

    def changed(cleanup: binding.Cleanup, *, consumer: Path, kind: str, output: Path) -> Path:
        result = original(cleanup, consumer=consumer, kind=kind, output=output)
        plan_request.with_name("clean-completion.json").write_text("{}")
        return result

    monkeypatch.setattr(binding, "create_binding", changed)
    with pytest.raises(ReplayError, match="changed"):
        previews.execute(plan_request, sha256=file_digest(plan_request))
    assert not plan_request.with_name("preview-proof.json").exists()


def test_scope_and_output_overlap(plan_request: Path) -> None:
    with pytest.raises(ReplayError, match="inside checkout"):
        previews._local(str(plan_request.parents[3] / "escape"))
    data = json.loads(plan_request.read_text())
    data["windows"][0]["output"] = data["sourceXML"]
    plan_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="distinct"):
        previews.load(plan_request, sha256=file_digest(plan_request))


def test_prepare_delays_missing_future_inputs(
    plan_request: Path, raw_request: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier_file = Path(str(raw_request["output"])).with_name("barrier-request.json")
    write_request(barrier_file, raw_request)
    plan_request.with_name("config.json").unlink()
    plan_request.with_name("clean.fcpxml").unlink()
    output = plan_request.parent / "prepared"
    output.mkdir()
    graph = previews.prepare(
        plan_request, sha256=file_digest(plan_request), barrier_request=barrier_file, output=output
    )
    assert [row["name"] for row in json.loads(graph.read_text())["stages"]] == [
        "wait-clean-producer",
        "run-preview-consumer",
    ]

    def reaches_wait(
        stages: tuple[Stage, ...],
        *,
        graph: dict[str, tuple[str, ...]],
        state: Path,
        source_caption_bindings: dict[str, Path],
    ) -> None:
        assert stages[0].name == "wait-clean-producer"
        raise RuntimeError("wait reached")

    monkeypatch.setattr(workflow, "_run_selected", reaches_wait)
    with pytest.raises(RuntimeError, match="wait reached"):
        workflow.run_plan(graph, state_directory=output / "state")
    with pytest.raises(ReplayError, match="empty"):
        previews.prepare(
            plan_request,
            sha256=file_digest(plan_request),
            barrier_request=barrier_file,
            output=output,
        )


def test_cli(plan_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["preview", "run", str(plan_request), "--sha256", file_digest(plan_request)]
    )
    previews.main()
    monkeypatch.setattr(
        "sys.argv", ["preview", "prepare", str(plan_request), "--sha256", file_digest(plan_request)]
    )
    with pytest.raises(SystemExit):
        previews.main()
