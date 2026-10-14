"""Producer evidence closure, without native/app execution or historical backfill."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from test_caption_resource_consumers import check_caption_consumers
from test_caption_resource_inputs import Chain, ref, write
from test_caption_resource_inputs import chain as chain
from test_delivery_render import configured

from replay import caption_resource_inputs as resources
from replay import compressor_batch, stages, workflow
from replay import delivery_render as render
from replay.core import ReplayError
from replay.recordings import record


def test_producer_binds_both_stages_and_completed_alpha(chain: Chain) -> None:
    base = render.load_config(configured(chain.root / "config-base"))
    new_work = chain.root / "new-attempt"
    manifest = new_work / "motion/prepare-0-0000.json"
    state = new_work / "state"
    data = record(json.loads(chain.config.read_text()))
    record(record(data["motion"])["resourceEvidence"])["preparation"] = {
        "workflow": str(manifest),
        "stateDirectory": str(state),
    }
    config_path = write(chain.root / "new-config.json", data)
    config = replace(
        base,
        path=config_path,
        work=new_work,
        template=chain.root / "template.moti",
        calibration=chain.root / "calibration.json",
        alpha_preset=chain.root / "alpha.setting",
        finish=("--font-name", "Test-Bold", "--font-size", "12"),
        resolved_assets=(chain.font,),
    )
    write(
        new_work / "motion/segment-0000/plans/manifest.json",
        {"parts": [{"path": str(chain.plan), "offsetFrame": 0, "frames": 1}]},
    )
    write(chain.plan.parent / "manifest.json", {"parts": [{"path": str(chain.plan)}]})
    [actual] = render.motion_prepare(config, workers=1)
    assert actual == manifest
    first, finish = workflow.load_stages(actual)
    proof = resources.preparation_inputs(config=config_path, plan=chain.plan)
    assert set(proof.pins) <= set(first.inputs)
    assert set(proof.pins) <= set(finish.inputs)
    assert new_work / "motion/segment-0000/plans/manifest.json" in first.inputs
    assert chain.plan.parent / "manifest.json" in first.inputs
    assert manifest not in first.inputs
    assert not any(path.is_relative_to(state) for path in first.inputs)
    with pytest.raises(FileNotFoundError):
        render.alpha_jobs(config)
    # Simulated outputs/receipts for the real generated definitions, not media approval.
    first.outputs[0].parent.mkdir(parents=True, exist_ok=True)
    first.outputs[0].write_bytes(b"synthetic XML intermediate")
    finish.outputs[0].write_bytes(b"synthetic XML finished")
    write(
        state / f"{first.name}.json",
        {
            "status": "complete",
            "fingerprint": stages._fingerprint(first),
            "outputs": {str(first.outputs[0]): ref(first.outputs[0])["sha256"]},
        },
    )
    write(
        state / f"{finish.name}.json",
        {
            "status": "complete",
            "fingerprint": stages._fingerprint(finish),
            "outputs": {str(finish.outputs[0]): ref(finish.outputs[0])["sha256"]},
        },
    )
    jobs = record(json.loads(render.alpha_jobs(config).read_text()))
    job = resources.rows(jobs["jobs"])[0]
    dependencies = job["dependencies"]
    assert isinstance(dependencies, list)
    assert str(manifest) in dependencies
    assert str(state / f"{first.name}.json") in dependencies
    assert str(state / f"{finish.name}.json") in dependencies
    assert set(map(str, proof.pins)) <= set(dependencies)
    compressor = new_work / "Compressor"
    compressor.write_bytes(b"mock compressor, never executed")
    compressor.chmod(0o700)
    output = new_work / "output.mov"

    def cli(command: Sequence[str]) -> str:
        if "-monitor" in command:
            return '[{"jobStatus":{"jobID":"synthetic","status":"Successful"}}]'
        output.write_bytes(b"synthetic, not rendered media")
        return '{"batch":{"jobs":[{"jobID":"synthetic"}]}}'

    batch = compressor_batch.Batch(new_work / "batch", compressor, "mock", cli, lambda _: None)
    submitted = compressor_batch.Job(
        "caption",
        finish.outputs[0],
        config.alpha_preset,
        output,
        tuple(Path(str(path)) for path in dependencies),
    )
    assert compressor_batch.run([submitted], batch, in_flight=1) == 1
    saved = record(json.loads((new_work / "batch/caption.json").read_text()))
    connected = Chain(config_path, chain.plan, finish.outputs[0], saved, chain.font, new_work)
    check_caption_consumers(connected, config)
    chain.font.write_bytes(b"changed-font")
    with pytest.raises(ReplayError):
        render.alpha_dependencies(config, render.motion_parts(config)[0])


def test_external_assets_cannot_use_legacy_path(chain: Chain) -> None:
    config = render.load_config(configured(chain.root / "config-base"))
    config = replace(config, resolved_assets=(Path("/outside-checkout/font.ttf"),))
    with pytest.raises(ReplayError, match="resourceEvidence"):
        render.preparation_inputs(config, render.motion_parts(config)[0])


def test_workflow_location_must_match_declared_binding(chain: Chain) -> None:
    base = render.load_config(configured(chain.root / "config-base"))
    config = replace(base, path=chain.config, resolved_assets=(chain.font,))
    # Actual plan coverage is validated before the namespace mismatch.
    write(
        config.work / "motion/segment-0000/plans/manifest.json",
        {"parts": [{"path": str(chain.plan), "offsetFrame": 0, "frames": 1}]},
    )
    with pytest.raises(ReplayError, match="workflow differs"):
        render.motion_prepare(config, workers=1)
