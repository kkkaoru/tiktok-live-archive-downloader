"""Shared real consumers with synthetic quality and mock-Compressor evidence."""

import json
from dataclasses import replace

import pytest
from test_caption_resource_inputs import Chain, ref
from test_caption_resource_inputs import chain as chain
from test_delivery_render import configured

from replay import delivery_caption_audit as audit
from replay import delivery_completion_v2 as completion
from replay.core import ReplayError
from replay.delivery_render import Config, load_config
from replay.recordings import record


def test_font_receipt_to_caption_and_v2(chain: Chain) -> None:
    config_path = configured(chain.root / "config-fixture")
    config = replace(
        load_config(config_path),
        path=chain.config,
        template=chain.root / "template.moti",
        calibration=chain.root / "calibration.json",
        resolved_assets=(chain.font,),
    )
    check_caption_consumers(chain, config)


def check_caption_consumers(chain: Chain, config: Config) -> None:
    """Exercise both real consumers with a mock successful Compressor submission."""
    part = audit.Part("caption", chain.plan, chain.source, 0, 1, ())
    inputs = audit.Inputs()
    assert audit.tool_code_binding(
        chain.root / "batch/caption.json", config=config, part=part, inputs=inputs
    )
    verified = chain.validate()
    movie = chain.root / "output.mov"
    srt = movie.with_suffix(".srt")
    srt.write_text("synthetic sidecar", encoding="utf-8")
    paths = {*verified.pins, movie, srt}
    data: dict[str, object] = {
        "version": 1,
        "kind": "delivery-caption-audit",
        "accepted": True,
        "limitations": [],
        "inputs": [{**ref(path), "bytes": path.stat().st_size} for path in paths],
        "checks": dict.fromkeys(
            (
                "measuredPlanCorrespondence",
                "orderedFrameTiming",
                "canonicalSRTFrameEquivalent",
                "candidateSRTExactMatch",
                "fullAlphaCoverage",
                "boundsAndGapsVerified",
                "historicalSubmissionBound",
                "recognitionCutProjectionVerified",
                "canonicalSRTActualDisplayVerified",
                "historicalToolCodeBound",
                "finalCompositionVerified",
            ),
            True,
        ),
        "measurements": {
            "wholeVideoFrames": 1,
            "expectedAlphaParts": 1,
            "auditedAlphaParts": 1,
            "canonicalSRT": ref(srt),
            "parts": [
                {
                    "backend": "direct-compressor-cli",
                    "frames": 1,
                    "historicalSubmissionBound": True,
                    "jobID": "synthetic",
                    "plan": ref(chain.plan),
                    "source": ref(chain.source),
                    "producerState": ref(chain.root / "batch/caption.json"),
                    "checks": {
                        "boundsAndGapsVerified": True,
                        "gapsVerified": True,
                        "visibleStatesVerified": True,
                        "fullStateConstraintCoverage": True,
                        "fullCaptionStates": 1,
                        "sampleCount": 1,
                        "constraintFrames": 1,
                        "measuredFrames": 1,
                    },
                }
            ],
        },
    }
    # Simulated external font lives outside this simulated recording checkout.
    store = completion.Evidence(chain.config.parent / "edits/replay-test", chain.source)
    with pytest.raises(ReplayError, match="authorized storage"):
        store.digest(chain.font)
    completion._captions(store, data, render_path=chain.config, movie=movie, frames=1)
    assert store.caption_resources == {}
    with pytest.raises(ReplayError, match="authorized storage"):
        store.digest(chain.font)
    # No other domain inherits caption data privileges after a failure, either.
    data["accepted"] = False
    with pytest.raises(ReplayError):
        completion._captions(store, data, render_path=chain.config, movie=movie, frames=1)
    assert store.caption_resources == {}


def test_missing_evidence_does_not_use_executable_exception(chain: Chain) -> None:
    raw = record(json.loads(chain.config.read_text()))
    record(raw["motion"]).pop("resourceEvidence")
    chain.config.write_text(json.dumps(raw), encoding="utf-8")
    store = completion.Evidence(chain.root / "edits/replay-test", chain.source)
    with pytest.raises(ReplayError):
        store.digest(chain.font)
    assert store.caption_tools == set()
