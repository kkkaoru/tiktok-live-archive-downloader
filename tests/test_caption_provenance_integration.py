"""Producer to caption consumer using actual XML preparation and FFmpeg raster audit.

The Compressor invocation is mocked, not its producer or caption acceptance logic.
Config/native-part lookups and other domain receipts are synthetic prerequisites;
this is not actual ASR, native compositor or FCP execution.
"""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_delivery_caption_audit import config as config
from test_delivery_caption_audit import constraints as constraints
from test_delivery_caption_audit import part as part
from test_delivery_caption_audit import ref, write_json
from test_delivery_completion_v2 import Fixture, fingerprint
from test_delivery_completion_v2 import delivery as delivery
from test_motion_caption import sequence_template as sequence_template
from test_motion_caption import text_template as text_template

from replay import compressor_batch, motion_dependencies, motion_finish, motion_stages
from replay import delivery_caption_audit as audit
from replay import delivery_completion_v2 as completion
from replay import delivery_verification as verification
from replay.core import ReplayError
from replay.delivery_render import Config
from replay.recordings import record


@pytest.fixture(autouse=True)
def checkout_scope(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    # Real producer code belongs to this checkout, not the unit fixture's tmp_path.
    repository = Path(__file__).resolve().parents[1]
    assert config.work.is_relative_to(repository)
    monkeypatch.setattr(verification, "REPOSITORY", repository)


def test_preparation_dependencies_use_actual_checkout(config: Config, part: audit.Part) -> None:
    dependencies = motion_dependencies.preparation_inputs(
        config=config.path, template=config.template, plan=part.plan, calibration=config.calibration
    )
    assert {
        path
        for path in dependencies
        if not path.is_relative_to(verification.REPOSITORY)
        and path != Path(sys.executable).resolve(strict=True)
    } == set()
    assert config.work.is_relative_to(verification.REPOSITORY)


def refresh_prerequisite(
    fixture: Fixture, key: str, updates: dict[str, object] | None = None
) -> None:
    data = json.loads(fixture.receipts[key].read_text())
    if updates is not None:
        data.update(updates)
    if "inputs" in data:
        data["inputs"] = [fingerprint(Path(row["path"])) for row in data["inputs"]]
    if "inputsAfter" in data:
        data["inputsAfter"] = data["inputs"]
    fixture.change(key, data)


def test_real_producer_to_accepted_caption(
    config: Config,
    part: audit.Part,
    constraints: list[dict[str, object]],
    delivery: Fixture,
    sequence_template: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.fail("FFmpeg and ffprobe required")
    template = config.work / "template.moti"
    template.write_text(sequence_template, encoding="utf-8")
    plan = json.loads(part.plan.read_text())
    plan["expectedText"] = "Title"
    write_json(part.plan, plan)
    manifest = part.plan.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    data["parts"][0].update(ref(part.plan))
    data["frames"] = 90
    write_json(manifest, data)
    write_json(
        config.captions / "segments.json",
        {
            "segments": [{"segment": 0, "offsetFrame": 0, "frames": 90}],
        },
    )
    write_json(
        delivery.root / "cuts.json",
        {
            "cuts": [{"source_start_frame": 30, "source_end_frame": 120, "output_start_frame": 0}],
        },
    )
    render = json.loads(delivery.render.read_text())
    render.update(width=4, height=4)
    write_json(delivery.render, render)
    config = replace(
        config,
        path=delivery.render,
        source=delivery.source,
        cuts=delivery.root / "cuts.json",
        template=template,
        finish=(
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
        ),
    )
    monkeypatch.setattr(audit, "load_config", lambda _: config)
    style = motion_finish.selected_finish_style(
        top=(0.1, 0.2, 0.3),
        bottom=(0.3, 0.4, 0.5),
        font_name="NotoSansJP-Thin_Bold",
        font_size=40.0,
        stroke_scale=0.5,
        shadow_scale=1.0,
    )
    part.source.write_text(
        motion_finish.finish_sequence(
            motion_stages.prepare(sequence_template, plan),
            style=style,
        ),
        encoding="utf-8",
    )
    movie = config.work / "alpha/new-alpha.mov"
    dependencies = motion_dependencies.preparation_inputs(
        config=config.path,
        template=template,
        plan=part.plan,
        calibration=config.calibration,
    )
    job = compressor_batch.Job(part.name, part.source, config.alpha_preset, movie, dependencies)
    write_json(
        config.work / "alpha-jobs.json",
        {
            "jobs": [
                {
                    "name": job.name,
                    "source": str(job.source),
                    "preset": str(job.preset),
                    "output": str(movie),
                    "dependencies": list(map(str, dependencies)),
                }
            ]
        },
    )
    state = config.work / "alpha-jobs-state"
    state.rename(config.work / "old-unbound-state")
    state.mkdir()
    raw = config.work / "synthetic.rgba"
    image = np.zeros((5, 4, 4, 4), dtype=np.uint8)
    image[:2, 1:3, 1:3, :] = 255
    image[3:, 1:3, 1:3, :] = 255
    raw.write_bytes(image.tobytes())

    environment = dict(
        os.environ, TMPDIR=str(config.work), TMP=str(config.work), TEMP=str(config.work)
    )
    environment.pop("FFREPORT", None)

    def mock_cli(command: Sequence[str]) -> str:
        if "-monitor" in command:
            return '[{"jobStatus":{"jobID":"mock-job","status":"Successful"}}]'
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgba",
                "-video_size",
                "4x4",
                "-framerate",
                "30",
                "-i",
                str(raw),
                "-c:v",
                "png",
                "-pix_fmt",
                "rgba",
                str(movie),
            ],
            check=True,
            timeout=20,
            capture_output=True,
            env=environment,
        )
        return '{"batch":{"jobs":[{"jobID":"mock-job"}]}}'

    executable = config.work / "Compressor"
    executable.write_bytes(b"mock executable; runner intercepts CLI")
    executable.chmod(0o700)
    worker = compressor_batch.Batch(state, executable, "synthetic", mock_cli, lambda _: None)
    assert compressor_batch.run([job], worker, in_flight=1) == 1
    assert (
        audit.tool_code_binding(
            state / f"{job.name}.json", config=config, part=part, inputs=audit.Inputs()
        )
        is True
    )
    constraint_path = write_json(
        config.work / "constraints.json",
        {
            "version": 1,
            "kind": "delivery-caption-constraints",
            "parts": {
                part.name: {
                    "plan": ref(part.plan),
                    "inputs": [ref(config.calibration)],
                    "states": constraints,
                }
            },
        },
    )
    labels = write_json(
        config.work / "labels.json",
        {
            "labels": [
                {"text": "あ", "start": 31 / 30, "end": 33 / 30, "provenance": "synthetic-asr"},
                {"text": "あい", "start": 34 / 30, "end": 36 / 30, "provenance": "synthetic-asr"},
            ]
        },
    )
    transcript = write_json(
        config.work / "transcript.json",
        {
            "version": 1,
            "kind": "caption-source-labels",
            "labels": ref(labels),
            "cuts": ref(config.cuts),
            "inputs": [ref(config.source)],
        },
    )
    final = delivery.movie
    request = write_json(
        config.work / "request.json",
        {
            "recipe": {
                "additionalVideo": [
                    {
                        "sourcePath": str(movie),
                        "selection": {
                            "startSeconds": 0.0,
                            "durationSeconds": 5 / 30,
                            "rate": 1,
                        },
                        "offsetSeconds": 1 / 30,
                        "opacity": 1,
                    }
                ]
            }
        },
    )
    response = write_json(config.work / "render.response.json", {"outputPath": str(final)})
    bindings = [
        audit.Inputs().bind(p)
        for p in (
            config.path,
            config.source,
            config.cuts,
            final,
            movie,
            request,
            response,
        )
    ]
    cut = write_json(
        config.work / "cut-execution.json",
        {
            "version": 1,
            "kind": "delivery-cut-execution",
            "accepted": True,
            "selectionPlanMatchesProtectedCuts": True,
            "recordedExecutionMatchesPlan": True,
            "checks": {"completedNativeExecution": True},
            "inputs": bindings,
            "inputsAfter": bindings,
            "actualNativeReceipts": [
                {
                    "actualNativeReceipt": str(response),
                    "receiptSHA256": ref(response)["sha256"],
                    "movie": str(final),
                    "movieSHA256Recorded": ref(final)["sha256"],
                    "movieSHA256AtAudit": ref(final)["sha256"],
                }
            ],
            "selectedFrames": 90,
        },
    )
    monkeypatch.setattr(audit, "rendered_parts", lambda _: [(request, 90)])
    srt = final.with_suffix(".srt")
    srt.write_text(audit.canonical_srt([part], 30), encoding="utf-8")
    result = audit.audit(
        config.path,
        srt,
        config.work / "accepted-audit",
        phase="audit",
        cache=config.work / "cache",
        ffmpeg=Path(ffmpeg),
        ffprobe=Path(ffprobe),
        constraints=constraint_path,
        transcript=transcript,
        final_movie=final,
        cut_execution=cut,
    )
    assert result["accepted"] is True
    assert record(result["checks"])["historicalToolCodeBound"] is True
    assert record(result["measurements"])["missingBindings"] == []
    assert result["wholeDeliveryAccepted"] is False
    # Other domains remain synthetic adapter prerequisites; no caption flags are injected.
    delivery.change("captionAudit", result)
    refresh_prerequisite(delivery, "renderConfig")
    refresh_prerequisite(
        delivery,
        "technical",
        {
            "expected": {"frames": 90, "rate": 30, "width": 4, "height": 4},
        },
    )
    refresh_prerequisite(delivery, "remux")
    refresh_prerequisite(delivery, "cutExecution")
    refresh_prerequisite(delivery, "audioQuality")
    refresh_prerequisite(delivery, "maskQuality")
    refresh_prerequisite(delivery, "fcp")
    repository = Path(__file__).resolve().parents[1]
    assert (
        completion.assess(
            delivery.index,
            delivery=final,
            root=delivery.root,
            source=delivery.source,
            storage=repository,
        )
        == {}
    )
    state_path = state / f"{job.name}.json"
    saved = json.loads(state_path.read_text())
    saved["terminalRuntimeVerified"] = False
    write_json(state_path, saved)
    with pytest.raises(ReplayError, match="terminal runtime"):
        audit.tool_code_binding(state_path, config=config, part=part, inputs=audit.Inputs())
