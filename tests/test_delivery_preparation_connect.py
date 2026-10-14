"""Real producer/workflow contracts; only native/Compressor calls are mocked."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from test_delivery_render import configured, write
from test_motion_caption import sequence_template as sequence_template
from test_motion_caption import text_template as text_template
from test_motion_measured import candidate as candidate

from replay import compressor_batch, delivery_render, motion_dependencies, workflow
from replay import delivery_caption_audit as audit
from replay.core import ReplayError
from replay.stages import file_digest


def test_prepare_workflow_to_compressor_and_caption_history(
    tmp_path: Path, candidate: Path, sequence_template: str
) -> None:
    path = configured(tmp_path)
    data = json.loads(path.read_text())
    data["work"] = str(tmp_path / "new-work")
    data["motion"]["finish"] = [
        "--top",
        "0.1",
        "0.2",
        "0.3",
        "--bottom",
        "0.3",
        "0.4",
        "0.5",
        "--font-name",
        "NotoSansJP-Thin_Bold",
        "--font-size",
        "40",
        "--stroke-scale",
        "0.5",
        "--shadow-scale",
        "1",
    ]
    write(path, data)
    config = delivery_render.load_config(path)
    config.template.write_text(sequence_template, encoding="utf-8")
    (config.captions / "combined-0000.json").write_bytes(candidate.read_bytes())
    config.alpha_preset.write_bytes(b"synthetic preset")
    [export] = delivery_render.motion_export(config, workers=1)
    export_state = export.with_name("export-state")
    assert workflow.main([str(export), "--state", str(export_state), "--check"]) == 0
    assert workflow.main([str(export), "--state", str(export_state)]) == 0
    [prepare] = delivery_render.motion_prepare(config, workers=1)
    prepare_state = prepare.with_name("prepare-state")
    assert workflow.main([str(prepare), "--state", str(prepare_state), "--check"]) == 0
    assert workflow.main([str(prepare), "--state", str(prepare_state)]) == 0
    assert workflow.main([str(prepare), "--state", str(prepare_state)]) == 0
    [job] = compressor_batch.read_jobs(delivery_render.alpha_jobs(config))
    [part] = delivery_render.motion_parts(config)
    stages = workflow.load_stages(prepare)
    required = motion_dependencies.preparation_inputs(
        config=config.path,
        template=config.template,
        plan=part.plan,
        calibration=config.calibration,
    )
    assert set(required) <= set(stages[0].inputs) <= set(stages[1].inputs)
    assert set(stages[0].inputs) == set(job.dependencies)
    assert config.resolved_assets is not None
    assert set(config.resolved_assets) <= set(job.dependencies)
    assert part.plan.parent / "manifest.json" in job.dependencies
    assert config.captions / "combined-0000.json" in job.dependencies
    executable = tmp_path / "Compressor"
    executable.write_bytes(b"mock Compressor, not executed")
    calls: list[str] = []

    def cli(command: Sequence[str]) -> str:
        if "-monitor" in command:
            return '[{"jobStatus":{"jobID":"synthetic-job","status":"Successful"}}]'
        calls.append("submit")
        job.output.parent.mkdir(parents=True, exist_ok=True)
        job.output.write_bytes(b"synthetic output, not a decoded alpha")
        return '{"batch":{"jobs":[{"jobID":"synthetic-job"}]}}'

    state = config.work / "alpha-jobs-state"
    worker = compressor_batch.Batch(state, executable, "synthetic", cli, lambda _: None)
    assert compressor_batch.run([job], worker, in_flight=1) == 1
    assert calls == ["submit"]
    caption_part = audit.Part(job.name, part.plan, job.source, part.offset, part.frames, ())
    saved = state / f"{job.name}.json"
    assert audit.state_binding(
        saved, job=job, movie=job.output, job_id="synthetic-job", inputs=audit.Inputs()
    )
    assert audit.tool_code_binding(saved, config=config, part=caption_part, inputs=audit.Inputs())
    audit.verify_motion_source(config, caption_part, audit.Inputs())
    # Neither file pins nor successful XML preparation certify actual font resolution.
    assert json.loads(saved.read_text())["terminalRuntimeVerified"] is True
    config.resolved_assets[0].write_bytes(b"changed supplier font")
    assert workflow.main([str(prepare), "--state", str(prepare_state)]) == 1
    with pytest.raises(ReplayError):
        audit.tool_code_binding(saved, config=config, part=caption_part, inputs=audit.Inputs())
    assert calls == ["submit"]


@pytest.mark.parametrize(
    "field",
    [
        None,
        {},
        {"fonts": [], "resources": []},
        {"fonts": [{"path": "/missing"}], "resources": []},
        {"fonts": [{"path": "/missing", "sha256": "0" * 64}], "resources": []},
    ],
)
def test_missing_or_invalid_assets_fail_closed(tmp_path: Path, field: object) -> None:
    path = configured(tmp_path)
    data = json.loads(path.read_text())
    data["motion"]["resolvedAssets"] = field
    write(path, data)
    with pytest.raises(ReplayError, match="asset|Assets"):
        delivery_render.motion_prepare(delivery_render.load_config(path), workers=1)
    assert not list((tmp_path / "work/motion").glob("prepare-*.json"))


def test_changed_asset_hash_and_explicit_resource(tmp_path: Path) -> None:
    path = configured(tmp_path)
    data = json.loads(path.read_text())
    resource = tmp_path / "external.bin"
    resource.write_bytes(b"explicit external Motion resource")
    data["motion"]["resolvedAssets"]["resources"] = [
        {"path": str(resource), "sha256": file_digest(resource)}
    ]
    write(path, data)
    config = delivery_render.load_config(path)
    assert config.resolved_assets == (tmp_path / "synthetic-font.bin", resource)
    resource.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="hash changed"):
        delivery_render.load_config(path)
    with pytest.raises(ReplayError, match="supplier evidence"):
        delivery_render.alpha_jobs(replace(config, resolved_assets=None))
