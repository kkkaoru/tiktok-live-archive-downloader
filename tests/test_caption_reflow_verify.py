import json
import sys
from pathlib import Path

import pytest

from replay import caption_reflow_verify, reflow_producer_evidence
from replay.core import ReplayError
from replay.stages import file_digest


def test_verify_complete_clock_and_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source")
    code = Path(caption_reflow_verify.__file__).with_name("caption_native_paging.py").resolve()
    saved = tmp_path / "producer.bytes"
    saved.write_bytes(code.read_bytes())
    path = tmp_path / "reflow.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "inputs": {str(source): file_digest(source), str(code): file_digest(saved)},
                "coverage": {"start": 0, "end": 2},
                "captions": [{"start": 0.2, "end": 1.5, "text": "recognized phrase"}],
                "layout": {
                    "version": 1,
                    "scope": "horizontal-phrase-layout-only",
                    "pointSize": 83,
                    "paragraphWidth": 972,
                    "maximumLines": 3,
                    "widthMeasurement": (
                        "native-point-size-checked-glyph-bounds-including-border-rim"
                    ),
                    "boundaries": "pinned-vibrato-ipadic-phrase-boundaries",
                    "recognitionClocksChanged": False,
                    "originsRequireRetargeting": True,
                },
            }
        ),
        encoding="utf-8",
    )
    assert caption_reflow_verify.verify(path, source=source, frames=60, rate=30) == 1
    assert (
        caption_reflow_verify.verify(path, source=source, frames=45, rate=30, start_frame=15) == 1
    )
    with pytest.raises(ReplayError, match="start frame"):
        caption_reflow_verify.verify(path, source=source, frames=45, rate=30, start_frame=-1)
    with pytest.raises(ReplayError, match="coverage"):
        caption_reflow_verify.verify(path, source=source, frames=61, rate=30)
    manifest = tmp_path / "producers.json"
    reflow_producer_evidence.prepare(artifact=path, source=code, evidence=saved, output=manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            str(path),
            "--source-media",
            str(source),
            "--frames",
            "60",
            "--producer-evidence",
            str(manifest),
        ],
    )
    caption_reflow_verify.main()
    source.write_bytes(b"changed source")
    with pytest.raises(ReplayError, match="source differ"):
        caption_reflow_verify.verify(path, source=source, frames=60, rate=30)


@pytest.mark.parametrize("frames,rate", [(0, 30), (10_000_001, 30), (1, 0), (1, 61)])
def test_verify_rejects_invalid_output_extent(frames: int, rate: int) -> None:
    with pytest.raises(ReplayError, match="bounded"):
        caption_reflow_verify.verify(
            Path("unused"), source=Path("unused"), frames=frames, rate=rate
        )
