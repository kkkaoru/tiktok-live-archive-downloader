import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from test_caption_native_layout import layout as layout
from test_caption_native_reflow import exact as exact

from replay import caption_joint_fit
from replay import caption_native_paging as paging
from replay import reflow_producer_evidence as producer
from replay.caption_measure_cache import CachedCaptionLayout, CaptionMeasurer
from replay.caption_native_layout import NativeCaptionLayout
from replay.caption_timing import CaptionCue, SpokenToken
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


def test_paging_uses_recognized_clock(exact: NativeCaptionLayout) -> None:
    tokens = tuple(
        SpokenToken(text, i * 0.2, (i + 1) * 0.2, "u") for i, text in enumerate("abcdefghij")
    )
    cues, _, _ = paging.paged_cues(tokens, layout=exact, rate=30, duration=Fraction(3))
    assert cues
    assert all(len(cue.text.split("\n")) <= 3 for cue in cues)
    assert all(
        exact.measure(line)[0] <= exact.width for cue in cues for line in cue.text.split("\n")
    )
    assert all(0 <= cue.start < cue.end <= 3 for cue in cues)
    assert "j" in cues[-1].text
    exact.expected_point_size = None
    with pytest.raises(ReplayError, match="point size"):
        paging.paged_cues(tokens, layout=exact, rate=30, duration=Fraction(3))
    exact.expected_point_size = 10
    with pytest.raises(ReplayError, match="exceeds"):
        paging.paged_cues(tokens, layout=exact, rate=30, duration=Fraction(1))


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"text": 1, "start": 0, "end": 1}],
        [{"text": "a", "start": 1, "end": 0}],
        [{"text": "a", "start": False, "end": 1}],
    ],
)
def test_bad_assembly(rows: list[object]) -> None:
    with pytest.raises(ReplayError):
        paging.assemble(({"captions": rows},), rate=30)


def test_boundary_hold() -> None:
    parts: tuple[dict[str, object], ...] = (
        {"captions": [{"text": "a", "start": 0, "end": 1.05}]},
        {"captions": [{"text": "b", "start": 1, "end": 2}]},
    )
    assert paging.assemble(parts, rate=30)[0]["end"] == 1


def fixture_index(root: Path) -> Path:
    media = root / "original.mp4"
    media.write_bytes(b"synthetic")
    token = root / "tokens-0000.json"
    token.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "tokens": [{"text": "ab", "start": 0.1, "end": 0.5, "utterance": "u"}],
            }
        )
    )
    (root / "combined-0000.json").write_text(
        json.dumps({"inputs": {str(media): file_digest(media), str(token): file_digest(token)}})
    )
    (root / "segments.json").write_text(
        json.dumps(
            {"rate": 30, "segments": [{"segment": 0, "offsetFrame": 30, "frames": 30, "jobs": [0]}]}
        )
    )
    return media


def test_support_audit_uses_pinned_tokens_without_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_index(tmp_path)
    output = tmp_path / "audit.json"
    assert paging.audit_support(tmp_path, output) == output
    report = json.loads(output.read_text())
    assert report["segments"] == [
        {"segment": 0, "tokens": 1, "units": [], "outsideUnitTimeFragments": 0}
    ]
    assert report["measurementsPerformed"] is False
    with pytest.raises(ReplayError, match="new canonical"):
        paging.audit_support(tmp_path, output)
    monkeypatch.setattr(paging, "audit_support", lambda directory, destination: destination)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paging",
            str(tmp_path),
            str(output),
            "native",
            "--source-media",
            "source",
            "--ass-font-size",
            "10",
            "--point-size",
            "10",
            "--border",
            "1",
            "--rim",
            "1",
            "--width",
            "100",
            "--evidence",
            "evidence",
            "--audit-support-only",
        ],
    )
    paging.main()


def test_support_audit_rejects_invalid_index_and_changed_pin(tmp_path: Path) -> None:
    fixture_index(tmp_path)
    (tmp_path / "tokens-0000.json").write_text("changed")
    with pytest.raises(ValueError, match="Pinned foreground evidence changed"):
        paging.audit_support(tmp_path, tmp_path / "audit.json")
    (tmp_path / "segments.json").write_text('{"segments": []}')
    with pytest.raises(ReplayError, match="bounded support"):
        paging.audit_support(tmp_path, tmp_path / "audit.json")


@pytest.mark.parametrize("long", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_run_and_resume(
    tmp_path: Path,
    exact: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
    long: bool,
    fresh: bool,
) -> None:
    media = fixture_index(tmp_path)
    if fresh:
        previous = tmp_path / "combined-0000.json"
        document = json.loads(previous.read_text())
        document.update(
            version=1, kind="recognition-inputs", segment=0, jobs=[0], measurementsPerformed=False
        )
        binding = tmp_path / "recognition-0000.json"
        binding.write_text(json.dumps(document))
        index_path = tmp_path / "segments.json"
        index = json.loads(index_path.read_text())
        index["segments"][0]["recognitionInput"] = {
            "path": str(binding),
            "sha256": file_digest(binding),
        }
        index_path.write_text(json.dumps(index))
        previous.unlink()
    monkeypatch.setattr(paging, "NativeCaptionLayout", lambda **kwargs: exact)
    long_calls: list[object] = []

    def long_factory(**kwargs: object) -> NativeCaptionLayout:
        long_calls.append(kwargs["long_executable"])
        return exact

    monkeypatch.setattr(paging, "LongLineCaptionLayout", long_factory)
    output = tmp_path / "paged"

    def run() -> Path:
        return paging.run(
            tmp_path,
            output,
            executable=exact.executable,
            style=exact.style,
            point_size=10,
            width=24,
            evidence=(),
            source_media=media,
            long_executable=exact.executable if long else None,
        )

    result = run()
    raw = record(json.loads(result.read_text()))
    assert raw["coverage"] == {"start": 1.0, "end": 2.0}
    captions = raw["captions"]
    assert isinstance(captions, list)
    assert record(captions[0])["start"] == 1.1
    assert long_calls == ([exact.executable] if long else [])
    result.unlink()
    assert run().exists()
    result.unlink()
    (tmp_path / "tokens-0000.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        run()


@pytest.mark.parametrize("jobs", [None, [], [-1], [0]])
def test_bad_tokens(tmp_path: Path, jobs: object) -> None:
    with pytest.raises(ReplayError):
        paging.read_tokens(tmp_path, {"jobs": jobs}, {})


def test_empty_token_span_keeps_ten_frames_without_glyph_requests(
    tmp_path: Path, exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = fixture_index(tmp_path)
    token = tmp_path / "tokens-0000.json"
    token.write_text('{"version":1,"clock":"output","tokens":[]}', encoding="utf-8")
    (tmp_path / "combined-0000.json").write_text(
        json.dumps({"inputs": {str(media): file_digest(media), str(token): file_digest(token)}}),
        encoding="utf-8",
    )
    (tmp_path / "segments.json").write_text(
        json.dumps(
            {"rate": 30, "segments": [{"segment": 0, "offsetFrame": 0, "frames": 10, "jobs": [0]}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paging, "NativeCaptionLayout", lambda **kwargs: exact)

    def no_cache(*args: object, **kwargs: object) -> None:
        pytest.fail("Empty recognition must not import glyph cache")

    monkeypatch.setattr(paging, "reuse_measurements", no_cache)
    output = tmp_path / "out"
    result = paging.run(
        tmp_path,
        output,
        executable=exact.executable,
        style=exact.style,
        point_size=10,
        width=24,
        evidence=(),
        source_media=media,
        measurement_cache=tmp_path / "unused-cache",
    )
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["captions"] == []
    assert document["coverage"] == {"start": 0.0, "end": 1 / 3}
    assert document["emptyRecognitionSpans"] == [
        {
            "start": 0.0,
            "end": 1 / 3,
            "recognizedTokens": 0,
            "measuredGlyphs": 0,
            "classification": "no-recognized-caption-tokens",
        }
    ]
    assert not tuple(exact.directory.glob("request-*.json"))
    part = json.loads((output / "part-0000.json").read_text(encoding="utf-8"))
    assert part["measurementCache"] is None
    assert part["recognizedTokens"] == 0


@pytest.mark.parametrize("change", ["index", "extent", "binding", "empty", "symlink"])
def test_run_invalid(tmp_path: Path, exact: NativeCaptionLayout, change: str) -> None:
    media = fixture_index(tmp_path)
    index = tmp_path / "segments.json"
    raw = record(json.loads(index.read_text()))
    output = tmp_path / "out"
    if change == "index":
        raw["rate"] = 24
    elif change == "extent":
        raw["segments"] = [{"segment": 0, "offsetFrame": 0, "frames": 0, "jobs": [0]}]
    elif change == "binding":
        (tmp_path / "combined-0000.json").write_text('{"inputs":{}}')
    elif change == "symlink":
        output.symlink_to(tmp_path, target_is_directory=True)
    index.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        paging.run(
            tmp_path,
            output,
            executable=exact.executable,
            style=exact.style,
            point_size=10,
            width=24,
            evidence=(),
            source_media=media,
            selected=(9,) if change == "empty" else (),
        )


@pytest.mark.parametrize("cached", [True, False])
def test_cli(
    tmp_path: Path, exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    media = fixture_index(tmp_path)
    monkeypatch.setattr(paging, "NativeCaptionLayout", lambda **kwargs: exact)
    monkeypatch.setattr(paging, "LongLineCaptionLayout", lambda **kwargs: exact)
    calls: list[tuple[str, int, int]] = []

    def reuse(
        native: NativeCaptionLayout, *, root: Path, segment: int, evidence: tuple[Path, ...]
    ) -> CachedCaptionLayout:
        calls.append((root.name, segment, len(evidence)))
        return CachedCaptionLayout(native)

    monkeypatch.setattr(paging, "reuse_measurements", reuse)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paging",
            str(tmp_path),
            str(tmp_path / "cli"),
            str(exact.executable),
            "--source-media",
            str(media),
            "--ass-font-size",
            "10",
            "--point-size",
            "10",
            "--border",
            "1",
            "--rim",
            "0.5",
            "--width",
            "24",
            "--evidence",
            str(exact.executable),
        ]
        + (
            [
                "--measurement-cache",
                str(tmp_path / "prior"),
                "--measurement-cache-segment",
                "0",
                "--long-executable",
                str(exact.executable),
            ]
            if cached
            else []
        ),
    )
    paging.main()
    assert calls == ([("prior", 0, 1)] if cached else [])
    assert (tmp_path / "cli/reflow.json").exists()


@pytest.mark.parametrize("segments", [(7,), (0, 0), (True,), (0,)])
def test_invalid_cache_selection(
    tmp_path: Path, exact: NativeCaptionLayout, segments: tuple[int, ...]
) -> None:
    media = fixture_index(tmp_path)
    with pytest.raises(ReplayError, match="cache selection"):
        paging.run(
            tmp_path,
            tmp_path / "out",
            executable=exact.executable,
            style=exact.style,
            point_size=10,
            width=24,
            evidence=(),
            source_media=media,
            measurement_cache=None if segments == (0,) else tmp_path,
            cache_segments=segments,
        )


@pytest.mark.parametrize(
    "message", ["Native token exceeds display budget; test", "unrelated native failure"]
)
def test_collect_planning_failures_preserves_success_and_rejects_unrelated_errors(
    tmp_path: Path, exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    media = fixture_index(tmp_path)
    (tmp_path / "combined-0001.json").write_bytes((tmp_path / "combined-0000.json").read_bytes())
    (tmp_path / "segments.json").write_text(
        json.dumps(
            {
                "rate": 30,
                "segments": [
                    {"segment": 0, "offsetFrame": 30, "frames": 30, "jobs": [0]},
                    {"segment": 1, "offsetFrame": 60, "frames": 30, "jobs": [0]},
                ],
            }
        )
    )
    monkeypatch.setattr(paging, "NativeCaptionLayout", lambda **kwargs: exact)
    calls = 0

    def planned(
        tokens: tuple[SpokenToken, ...],
        *,
        layout: CaptionMeasurer,
        rate: int,
        duration: Fraction,
        split_wide: bool,
    ) -> tuple[tuple[CaptionCue, ...], int, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError(message)
        return (CaptionCue("ab", 0.1, 0.5),), 0, 0

    monkeypatch.setattr(paging, "paged_cues", planned)
    output = tmp_path / "collected"
    with pytest.raises((ValueError, ReplayError)):
        paging.run(
            tmp_path,
            output,
            executable=exact.executable,
            style=exact.style,
            point_size=10,
            width=24,
            evidence=(),
            source_media=media,
            collect_layout_failures=True,
        )
    assert not (output / "reflow.json").exists()
    if message.startswith("Native token"):
        assert calls == 2
        assert (output / "part-0001.json").exists()
        index = json.loads((output / "layout-failures.json").read_text())
        assert index["complete"] is False and index["successfulSegments"] == 1
        assert index["failures"][0]["segment"] == 0
    else:
        assert calls == 1 and not (output / "layout-failures.json").exists()


def test_invalid_token_document(tmp_path: Path) -> None:
    token = tmp_path / "tokens-0000.json"
    for document in (
        {"version": 1, "clock": "source", "tokens": []},
        {"version": 1, "clock": "output", "tokens": None},
    ):
        token.write_text(json.dumps(document))
        with pytest.raises(ReplayError):
            paging.read_tokens(tmp_path, {"jobs": [0]}, {str(token): file_digest(token)})


def test_token_count_upper_bound_still_rejected(tmp_path: Path) -> None:
    token = tmp_path / "tokens-0000.json"
    token.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "tokens": [{"text": "語", "start": 0, "end": 0.1, "utterance": "u"}] * 4097,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="token count"):
        paging.read_tokens(tmp_path, {"jobs": [0]}, {str(token): file_digest(token)})


def test_run_explicit_sources(
    tmp_path: Path, exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = fixture_index(tmp_path)
    token = tmp_path / "tokens-0000.json"
    reference = {"path": str(token), "sha256": file_digest(token)}
    manifest = tmp_path / "adopted.json"
    manifest.write_text(json.dumps({"version": 1, "tokens": reference}))
    combined = tmp_path / "combined-0000.json"
    raw = record(json.loads(combined.read_text()))
    record(raw["inputs"])[str(manifest)] = file_digest(manifest)
    combined.write_text(json.dumps(raw))
    mapping = tmp_path / "sources.json"
    mapping.write_text(
        json.dumps(
            {
                "version": 1,
                "replacements": [
                    {
                        "segment": 0,
                        "job": 0,
                        "tokens": reference,
                        "adoptedManifest": {"path": str(manifest), "sha256": file_digest(manifest)},
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(paging, "NativeCaptionLayout", lambda **kwargs: exact)
    result = paging.run(
        tmp_path,
        tmp_path / "mapped",
        executable=exact.executable,
        style=exact.style,
        point_size=10,
        width=24,
        evidence=(),
        source_media=media,
        token_sources=mapping,
    )
    output = record(json.loads(result.read_text()))
    assert str(mapping) in record(output["inputs"])
    assert output["coverage"] == {"start": 1.0, "end": 2.0}


def test_discarded_recognizer_noise_is_not_measured(exact: NativeCaptionLayout) -> None:
    cues, _, _ = paging.paged_cues(
        (SpokenToken("ab", 0.1, 0.4, "u"), SpokenToken("ー" * 200, 0.5, 0.8, "u")),
        layout=exact,
        rate=30,
        duration=Fraction(1),
    )
    assert len(cues) == 1
    assert cues[0].text == "ab"
    assert cues[0].start == 0.1


def test_resume_preserves_old_producer_bytes_but_requires_current_joint_dependency(
    tmp_path: Path, exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = fixture_index(tmp_path)
    code = tmp_path / "caption_native_paging.py"
    code.write_bytes(b"old producer")
    archive = tmp_path / "producer.bytes"
    archive.write_bytes(b"old producer")
    helper = tmp_path / "reflow_producer_evidence.py"
    helper.write_bytes(b"synthetic evidence helper")
    joint = tmp_path / "caption_joint_fit.py"
    joint.write_bytes(b"original joint dependency")
    monkeypatch.setattr(paging, "__file__", str(code))
    monkeypatch.setattr(producer, "__file__", str(helper))
    monkeypatch.setattr(caption_joint_fit, "__file__", str(joint))
    monkeypatch.setattr(paging, "NativeCaptionLayout", lambda **kwargs: exact)
    output = tmp_path / "resume"
    result = paging.run(
        tmp_path,
        output,
        executable=exact.executable,
        style=exact.style,
        point_size=10,
        width=24,
        evidence=(),
        source_media=media,
    )
    part = output / "part-0000.json"
    original = part.read_bytes()
    assert record(record(json.loads(original))["inputs"])[str(joint)] == file_digest(joint)
    manifest = tmp_path / "part-evidence.json"
    producer.prepare(artifact=part, source=code, evidence=archive, output=manifest)
    original_manifest = manifest.read_bytes()
    result.unlink()
    code.write_bytes(b"new producer")
    result = paging.run(
        tmp_path,
        output,
        executable=exact.executable,
        style=exact.style,
        point_size=10,
        width=24,
        evidence=(),
        source_media=media,
        resume_evidence=((part, manifest),),
    )
    raw = record(json.loads(result.read_text()))
    assert part.read_bytes() == original
    assert record(raw["inputs"])[str(code)] == file_digest(code)
    assert record(raw["inputs"])[str(archive)] == file_digest(archive)
    assert record(raw["inputs"])[str(joint)] == file_digest(joint)
    assert len(record(raw["historicalProducerEvidence"])) == 1
    result.unlink()
    with pytest.raises(ReplayError, match="uniquely"):
        paging.run(
            tmp_path,
            output,
            executable=exact.executable,
            style=exact.style,
            point_size=10,
            width=24,
            evidence=(),
            source_media=media,
            resume_evidence=((part, manifest), (part, manifest)),
        )
    joint.write_bytes(b"changed joint dependency")
    with pytest.raises(ReplayError, match="inputs changed"):
        paging.run(
            tmp_path,
            output,
            executable=exact.executable,
            style=exact.style,
            point_size=10,
            width=24,
            evidence=(),
            source_media=media,
            resume_evidence=((part, manifest),),
        )
    assert part.read_bytes() == original
    assert manifest.read_bytes() == original_manifest
    assert archive.read_bytes() == b"old producer"
    assert not result.exists()


def test_sustained_token_display_is_capped_and_recorded(exact: NativeCaptionLayout) -> None:
    tokens = (SpokenToken("ある", 0.5, 4.3, "u"), SpokenToken("よ", 4.3, 4.5, "u"))
    limited, rows = paging.limit_durations(tokens)
    assert limited[0].start == 0.5 and limited[0].end < 0.5 + 2.8
    assert limited[1] == tokens[1]
    assert rows == [{"clipped": {"text": "ある", "start": 0.5, "end": 4.3, "utterance": "u"},
                     "into": {"text": "ある", "start": 0.5, "end": limited[0].end,
                              "utterance": "u"}}]  # fmt: skip
    with pytest.raises(ValueError, match="too coarse"):
        paging.paged_cues(tokens, layout=exact, rate=30, duration=Fraction(5))
    cues, _, _ = paging.paged_cues(limited, layout=exact, rate=30, duration=Fraction(5))
    assert "".join(cue.text for cue in cues).replace("\n", "") == "あるよ"
