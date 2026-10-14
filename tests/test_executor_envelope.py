"""Envelope normalization preserves payloads and passes frozen validation."""

import json

import pytest

from replay.caption_io import executor_measurement, ocr_frames
from replay.executor_envelope import normalize_envelope

INNER = {
    "measurement": [
        {
            "width": 1280,
            "height": 720,
            "requestedTimeSeconds": 30.0,
            "actualTimeSeconds": 30.0,
            "lines": [],
        }
    ],
    "playbackPerformed": False,
    "sourceModified": False,
}


def _content_envelope() -> dict[str, object]:
    return {
        "ok": True,
        "data": {"content": [{"type": "text", "text": json.dumps(INNER)}]},
    }


def _structured_envelope() -> dict[str, object]:
    return {"ok": True, "data": {"structuredContent": dict(INNER)}}


def test_normalizes_content_text_envelope() -> None:
    normalized = normalize_envelope(_content_envelope())
    assert executor_measurement(normalized) == INNER["measurement"]
    frames = ocr_frames(normalized)
    assert len(frames) == 1
    assert frames[0].requested_time == 30.0


def test_passes_structured_envelope_through() -> None:
    envelope = _structured_envelope()
    assert normalize_envelope(envelope) is envelope


def test_rejects_failed_and_malformed_envelopes() -> None:
    with pytest.raises(ValueError, match="did not succeed"):
        normalize_envelope({"ok": False, "data": {}})
    with pytest.raises(ValueError, match="single content"):
        normalize_envelope({"ok": True, "data": {"content": []}})
    with pytest.raises(ValueError, match="read-only source contract"):
        changed = json.loads(json.dumps(INNER))
        changed["sourceModified"] = True
        tampered = normalize_envelope(
            {"ok": True, "data": {"content": [{"type": "text", "text": json.dumps(changed)}]}}
        )
        executor_measurement(tampered)
