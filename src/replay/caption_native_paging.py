"""Recognized-output-clock paging and resumable measured segment assembly.

Reuses the shared token planner and native glyph measurements. No new ASR,
source-clock projection or inherited caption positioning occurs. Indivisible
fine-ASR words may carry explicit measured uniform-scale evidence.
"""

import argparse
import json
import math
from dataclasses import asdict, replace
from fractions import Fraction
from functools import partial
from pathlib import Path

from replay import (
    caption_anchor_fit,
    caption_indivisible_fit,
    caption_joint_fit,
    caption_line_legality,
    caption_native_inputs,
    caption_scale,
    caption_style_defaults,
    caption_support_identity,
    caption_timing,
    fcp_measured,
    reflow_producer_evidence,
)
from replay.caption_indivisible_fit import IndivisibleFit
from replay.caption_long_layout import LongLineCaptionLayout
from replay.caption_measure_cache import CaptionMeasurer
from replay.caption_measured_job import _token
from replay.caption_measured_plan import PositionedCue
from replay.caption_native_inputs import recognition_pins, segment_binding
from replay.caption_native_layout import NativeCaptionLayout, NativeCaptionStyle
from replay.caption_paging_cache import reuse_measurements
from replay.caption_scale import read_scale, scale_fields
from replay.caption_timing import (
    CaptionCue,
    CaptionPolicy,
    ScaledCaptionCue,
    SpokenToken,
    display_time,
    make_captions,
)
from replay.caption_token_sources import TokenSource, load_sources
from replay.core import ReplayError
from replay.delivery_captions import CLOCK_EPSILON
from replay.fcp_measured import _boundary_hold, drop_retained_hidden_prefixes, ensure_display_frames
from replay.foreground_evidence import _bound_bytes
from replay.motion_stages import _integer
from replay.recordings import record
from replay.stages import file_digest

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_SEGMENTS = 512
MAX_SEGMENT_TOKENS = 4096
save = NativeCaptionLayout._save


def read_tokens(
    directory: Path,
    row: dict[str, object],
    bindings: dict[str, object],
    sources: tuple[TokenSource, ...] = (),
) -> tuple[tuple[SpokenToken, ...], dict[str, str]]:
    """Job token clocks are already local to their segment, never source clocks."""
    jobs = row.get("jobs")
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 128:
        raise ReplayError("Expected bounded recognized job references")
    result: list[SpokenToken] = []
    pins: dict[str, str] = {}
    for job in jobs:
        path = (directory / f"tokens-{_integer(job):04d}.json").resolve()
        for source in sources:
            if source.segment == row.get("segment") and source.job == job:
                pins.update(source.verify(bindings))
                path = source.tokens
        expected = bindings.get(str(path))
        if not isinstance(expected, str):
            raise ReplayError("Existing candidate does not bind its recognized tokens")
        raw = record(json.loads(_bound_bytes(path, expected, MAX_DOCUMENT_BYTES)))
        values = raw.get("tokens")
        if raw.get("version") != 1 or raw.get("clock") != "output" or not isinstance(values, list):
            raise ReplayError("Recognized tokens must use their existing output clock")
        members = tuple(_token(value) for value in values)
        result.extend(members)
        pins[str(path)] = expected
    if len(result) > MAX_SEGMENT_TOKENS:
        raise ReplayError("Segment token count exceeds the bounded planner")
    return tuple(result), pins


def limit_durations(
    tokens: tuple[SpokenToken, ...],
) -> tuple[tuple[SpokenToken, ...], list[dict[str, object]]]:
    """Cap a sustained or pause-spanning token's display at the focused-timing limit.

    The onset and text are kept; only how long the token keeps its caption visible is shortened,
    and every change is recorded (as ``delivery_captions.limit_token_durations`` does).
    """
    maximum = CaptionPolicy().maximum_context_seconds - CLOCK_EPSILON
    limited = tuple(
        replace(token, end=token.start + maximum) if token.end - token.start > maximum else token
        for token in tokens
    )
    rows: list[dict[str, object]] = [
        {"clipped": asdict(old), "into": asdict(new)}
        for old, new in zip(tokens, limited, strict=True)
        if old != new
    ]
    return limited, rows


def paged_cues(
    tokens: tuple[SpokenToken, ...],
    *,
    layout: CaptionMeasurer,
    rate: int,
    duration: Fraction,
    maximum_height: float | None = None,
    split_wide: bool = False,
) -> tuple[tuple[CaptionCue, ...], int, int]:
    """Use the same recognized onsets and phrase-aware reset/hold/frame policy.

    ``split_wide`` replaces a scaled caption (no full-size layout fits) with consecutive
    full-size pages sharing its display time in proportion to their length, for renderers
    that draw every caption at the style's size.
    """
    if layout.expected_point_size is None:
        raise ReplayError("Paging requires an explicit Motion point size")
    if maximum_height is not None and (
        isinstance(maximum_height, bool)
        or not math.isfinite(maximum_height)
        or not 0 < maximum_height <= 16384
    ):
        raise ReplayError("Paging requires a finite measured height budget")
    if any(display_time(token.end) > duration for token in tokens):
        raise ReplayError("Recognized token exceeds its declared segment clock")
    # The shared planner removes nonlexical recognizer loops before measuring.
    # Raw-token prefetch would incorrectly measure discarded recognition noise.

    fit = IndivisibleFit(tokens, layout=layout, maximum_height=maximum_height)

    policy = CaptionPolicy(
        frame_rate=rate,
        linguistic_boundaries=True,
        complete_phrases=True,
        measured_layout=fit.wrap,
        indivisible_text=fit.eligible,
        scoped_layout=fit.wrap_scoped,
        scoped_indivisible=fit.eligible_scoped,
        maximum_lines=caption_style_defaults.MAX_CAPTION_LINES,
        maximum_characters=80,
    )
    raw = make_captions(tokens, policy=policy)
    raw = tuple(
        ScaledCaptionCue(cue.text, cue.start, cue.end, fit.scoped_evidence[cue.text, cue.start])
        if (cue.text, cue.start) in fit.scoped_evidence
        else cue
        for cue in raw
    )
    if split_wide:
        raw = tuple(
            page
            for cue in raw
            for page in (
                split_pages(cue, fit.pages(cue.text))
                if isinstance(cue, ScaledCaptionCue)
                else (cue,)
            )
        )
    positioned = [
        PositionedCue(replace(cue, end=min(cue.end, float(duration))), 540, 0) for cue in raw
    ]
    positioned, hidden = drop_retained_hidden_prefixes(positioned, rate=rate)
    positioned, widened = ensure_display_frames(positioned, rate=rate, duration=duration)
    if any(
        not display_time(item.cue.start) < display_time(item.cue.end) <= duration
        for item in positioned
    ):
        raise ReplayError("Paged caption has no retained display support")
    return tuple(item.cue for item in positioned), hidden, widened


def split_pages(cue: CaptionCue, pages: tuple[str, ...]) -> tuple[CaptionCue, ...]:
    """Consecutive pages over the cue's display time, each lasting in proportion to its length."""
    total = sum(len(page.replace("\n", "")) for page in pages)
    span, done, result = cue.end - cue.start, 0, []
    for page in pages:
        start = cue.start + span * done / total
        done += len(page.replace("\n", ""))
        result.append(CaptionCue(page, start, cue.start + span * done / total))
    return tuple(result)


def assemble(parts: tuple[dict[str, object], ...], *, rate: int) -> list[dict[str, object]]:
    """Join only the existing bounded hold overlap; never sum overlapping segment extents."""
    result: list[PositionedCue] = []
    for part in parts:
        rows = part.get("captions")
        if not isinstance(rows, list):
            raise ReplayError("Paged segment needs caption stages")
        if not rows:
            if (
                type(part.get("recognizedTokens")) is not int
                or part.get("recognizedTokens") != 0
                or part.get("emptyRecognition")
                != {
                    "classification": "no-recognized-caption-tokens",
                    "recognizedTokens": 0,
                    "measuredGlyphs": 0,
                }
            ):
                raise ReplayError("Empty caption segment requires explicit zero-token evidence")
            continue
        members: list[PositionedCue] = []
        for value in rows:
            row = record(value)
            text, start, end = row.get("text"), row.get("start"), row.get("end")
            if (
                not isinstance(text, str)
                or not text.strip()
                or len(text.split("\n")) > 3
                or isinstance(start, bool)
                or not isinstance(start, (float, int))
                or isinstance(end, bool)
                or not isinstance(end, (float, int))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or not 0 <= start < end
            ):
                raise ReplayError("Invalid paged caption")
            if members and members[-1].cue.end > start:
                raise ReplayError("Paged captions overlap inside a segment")
            scale = read_scale(row)
            cue = (
                CaptionCue(text, start, end)
                if scale is None
                else ScaledCaptionCue(text, start, end, scale)
            )
            members.append(PositionedCue(cue, 540, 0))
        if result:
            result[-1] = _boundary_hold(result[-1], members[0], {"hold_seconds": 0.08})
        if result and display_time(result[-1].cue.end) > display_time(members[0].cue.start):
            raise ReplayError("Paged segments overlap beyond the shared caption hold")
        result.extend(members)
    return [
        {
            "start": item.cue.start,
            "end": item.cue.end,
            "text": item.cue.text,
            **scale_fields(item.cue.scale if isinstance(item.cue, ScaledCaptionCue) else None),
        }
        for item in result
    ]


def run(
    directory: Path,
    output: Path,
    *,
    executable: Path,
    style: NativeCaptionStyle,
    point_size: float,
    width: float,
    evidence: tuple[Path, ...],
    source_media: Path,
    selected: tuple[int, ...] = (),
    token_sources: Path | None = None,
    resume_evidence: tuple[tuple[Path, Path], ...] = (),
    measurement_cache: Path | None = None,
    cache_segments: tuple[int, ...] = (),
    collect_layout_failures: bool = False,
    long_executable: Path | None = None,
    split_wide: bool = False,
) -> Path:
    if type(collect_layout_failures) is not bool:
        raise ReplayError("Layout failure collection must be boolean")
    index_path = directory / "segments.json"
    index_digest = file_digest(index_path)
    index = record(json.loads(_bound_bytes(index_path, index_digest, MAX_DOCUMENT_BYTES)))
    entries = index.get("segments")
    rate = _integer(index.get("rate"))
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SEGMENTS or rate != 30:
        raise ReplayError("Expected bounded segment index on the established frame clock")
    known_segments = {_integer(record(value).get("segment")) for value in entries}
    if (
        len(set(cache_segments)) != len(cache_segments)
        or any(type(number) is not int or number not in known_segments for number in cache_segments)
        or (cache_segments and measurement_cache is None)
    ):
        raise ReplayError(
            "Measurement cache selection requires unique existing segments and a root"
        )
    sources, source_pins = (
        load_sources(token_sources, entries) if token_sources is not None else ((), {})
    )
    media = str(source_media.resolve())
    source_digest = file_digest(source_media)
    if output.absolute() != output.resolve():
        raise ReplayError("Paged output must not traverse symlinks")
    output.mkdir(mode=0o700, exist_ok=True)
    resume = {str(part.resolve()): manifest for part, manifest in resume_evidence}
    if len(resume) != len(resume_evidence) or any(
        part.resolve().parent != output.resolve() or not part.is_file()
        for part, _ in resume_evidence
    ):
        raise ReplayError("Resume evidence must uniquely identify existing parts in this output")
    historical: dict[str, object] = {}
    base_inputs = {
        **{
            str(path.resolve()): file_digest(path)
            for path in (
                Path(caption_scale.__file__),
                Path(caption_indivisible_fit.__file__),
                Path(caption_anchor_fit.__file__),
                Path(caption_joint_fit.__file__),
                Path(caption_line_legality.__file__),
                Path(caption_support_identity.__file__),
                Path(caption_native_inputs.__file__),
            )
        },
        **source_pins,
        media: source_digest,
        str(index_path.resolve()): index_digest,
        str(Path(__file__).resolve()): file_digest(Path(__file__)),
        str(Path(caption_timing.__file__).resolve()): file_digest(Path(caption_timing.__file__)),
        str(Path(fcp_measured.__file__).resolve()): file_digest(Path(fcp_measured.__file__)),
    }
    all_inputs = dict(base_inputs)
    settings: dict[str, object] = {"style": asdict(style), "pointSize": point_size, "width": width}
    if token_sources is not None:
        settings["tokenSources"] = source_pins
    parts: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    coverage_start: Fraction | None = None
    coverage_end = Fraction(0)
    for value in entries:
        row = record(value)
        number = _integer(row.get("segment"))
        if selected and number not in selected:
            continue
        offset = Fraction(_integer(row.get("offsetFrame")), rate)
        duration = Fraction(_integer(row.get("frames")), rate)
        if not 0 < duration <= 120 or offset < 0:
            raise ReplayError("Segment extent exceeds shared planner limits")
        if coverage_start is not None and (
            offset > coverage_end or offset < coverage_end - Fraction(1, 10)
        ):
            raise ReplayError("Selected segment coverage is discontinuous")
        combined = segment_binding(directory, row, sources)
        digest = file_digest(combined)
        old = record(json.loads(_bound_bytes(combined, digest, MAX_DOCUMENT_BYTES)))
        old_inputs = record(old.get("inputs"))
        if old_inputs.get(media) != source_digest:
            raise ReplayError("Existing recognition and requested original source differ")
        tokens, token_pins = read_tokens(directory, row, old_inputs, sources)
        token_pins.update(recognition_pins(old, source=media, source_digest=source_digest))
        part_path = output / f"part-{number:04d}.json"
        code_pins: dict[str, str] = {}
        if part_path.exists():
            part = record(json.loads(part_path.read_text(encoding="utf-8")))
            if part.get("settings") != settings or part.get("coverage") != {
                "start": float(offset),
                "end": float(offset + duration),
            }:
                raise ReplayError("Existing paged segment settings changed")
            saved = record(part.get("inputs"))
            manifest = resume.pop(str(part_path.resolve()), None)
            if manifest is not None:
                code_pins = reflow_producer_evidence.read(
                    manifest, artifact=part_path, inputs=saved
                )
                all_inputs.update(
                    reflow_producer_evidence.evidence_inputs(
                        manifest, artifact=part_path, inputs=saved
                    )
                )
                historical[str(part_path.resolve())] = {
                    "manifest": str(manifest.resolve()),
                    "producerInputs": code_pins,
                }
            if (
                any(
                    not isinstance(digest, str)
                    or (
                        name != media
                        and name not in code_pins
                        and file_digest(Path(name)) != digest
                    )
                    for name, digest in saved.items()
                )
                or saved.get(media) != source_digest
            ):
                raise ReplayError("Existing paged segment inputs changed; retain and investigate")
        else:
            factory = (
                NativeCaptionLayout
                if long_executable is None
                else partial(LongLineCaptionLayout, long_executable=long_executable)
            )
            native = factory(
                executable=executable,
                directory=output / f"metrics-{number:04d}",
                style=style,
                width=width,
                height=4096,
                evidence=(*evidence, combined, *map(Path, token_pins)),
                expected_point_size=point_size,
            )
            layout = (
                reuse_measurements(
                    native, root=measurement_cache, segment=number, evidence=evidence
                )
                if tokens
                and measurement_cache is not None
                and (not cache_segments or number in cache_segments)
                else native
            )
            limited, clipped = limit_durations(tokens)
            try:
                cues, hidden, widened = paged_cues(
                    limited, layout=layout, rate=rate, duration=duration, split_wide=split_wide
                )
            except ValueError as error:
                if not collect_layout_failures or not str(error).startswith(
                    (
                        "Native token exceeds display budget;",
                        "Native token timing is too coarse for focused captions",
                    )
                ):
                    raise
                failure_path = output / f"failure-{number:04d}.json"
                save(
                    failure_path,
                    {
                        "version": 1,
                        "segment": number,
                        "error": str(error),
                        "candidateOnly": True,
                        "settings": settings,
                        "inputs": layout.verify_inputs()
                        | base_inputs
                        | token_pins
                        | {str(combined.resolve()): digest},
                    },
                )
                failures.append(
                    {
                        "segment": number,
                        "path": str(failure_path.resolve()),
                        "sha256": file_digest(failure_path),
                    }
                )
                coverage_start = offset if coverage_start is None else min(coverage_start, offset)
                coverage_end = max(coverage_end, offset + duration)
                print(f"FAILED_PAGING_SEGMENT {number}: {error}", flush=True)
                continue
            pins = (
                layout.verify_inputs()
                | base_inputs
                | token_pins
                | {str(combined.resolve()): digest}
            )
            part = {
                "version": 1,
                "settings": settings,
                "candidateOnly": True,
                "inputs": pins,
                "captions": [
                    {
                        "start": float(display_time(cue.start) + offset),
                        "end": float(display_time(cue.end) + offset),
                        "text": cue.text,
                        **scale_fields(cue.scale if isinstance(cue, ScaledCaptionCue) else None),
                    }
                    for cue in cues
                ],
                "coverage": {"start": float(offset), "end": float(offset + duration)},
                "hiddenRetainedPrefixes": hidden,
                "frameVisibilityExtensions": widened,
                "durationLimitedTokens": clipped,
                "recognizedTokens": len(tokens),
                "measurementCache": str(measurement_cache.resolve())
                if tokens
                and measurement_cache is not None
                and (not cache_segments or number in cache_segments)
                else None,
                "historicalRecognitionInputs": old_inputs,
            }
            if not tokens:
                part["emptyRecognition"] = {
                    "classification": "no-recognized-caption-tokens",
                    "recognizedTokens": 0,
                    "measuredGlyphs": 0,
                }
            save(part_path, part)
        for name, digest_value in record(part.get("inputs")).items():
            if not isinstance(digest_value, str):
                raise ReplayError("Invalid paged provenance hash")
            if name not in code_pins:
                all_inputs[name] = digest_value
        all_inputs[str(part_path.resolve())] = file_digest(part_path)
        parts.append(part)
        coverage_start = offset if coverage_start is None else min(coverage_start, offset)
        coverage_end = max(coverage_end, offset + duration)
        print(f"PAGED_SEGMENT {number}", flush=True)
    if failures:
        save(
            output / "layout-failures.json",
            {
                "version": 1,
                "complete": False,
                "failures": failures,
                "successfulSegments": len(parts),
            },
        )
        raise ReplayError("Caption layout failures collected; no complete reflow was published")
    if not parts or coverage_start is None:
        raise ReplayError("No requested segments found")
    if resume:
        raise ReplayError("Resume producer evidence was not consumed by selected parts")
    if historical:
        helper = Path(reflow_producer_evidence.__file__).resolve()
        all_inputs[str(helper)] = file_digest(helper)
    captions = assemble(tuple(parts), rate=rate)
    if file_digest(source_media) != source_digest or file_digest(index_path) != index_digest:
        raise ReplayError("Recognition source/index changed during paging")
    result = output / "reflow.json"
    save(
        result,
        {
            "version": 1,
            "candidateOnly": True,
            "historicalProducerEvidence": historical,
            "inputs": all_inputs,
            "captions": captions,
            "emptyRecognitionSpans": [
                record(part.get("coverage")) | record(part["emptyRecognition"])
                for part in parts
                if "emptyRecognition" in part
            ],
            "coverage": {"start": float(coverage_start), "end": float(coverage_end)},
            "layout": {
                "version": 1,
                "scope": "horizontal-phrase-layout-only",
                "pointSize": point_size,
                "paragraphWidth": width,
                "maximumLines": caption_style_defaults.MAX_CAPTION_LINES,
                "widthMeasurement": "native-point-size-checked-glyph-bounds-including-border-rim",
                "boundaries": "pinned-vibrato-ipadic-phrase-boundaries",
                "recognitionClocksChanged": False,
                "originsRequireRetargeting": True,
                "motionRenderVerified": False,
                "paging": "shared-recognized-output-token-policy",
                "displayUnits": "complete-pos-phrases-at-first-recognized-onset",
                "oversizedPhrasePolicy": "intact-morphemes-at-existing-recognized-onsets",
                "segments": len(parts),
            },
        },
    )
    return result


def audit_support(directory: Path, output: Path) -> Path:
    """Read bounded pinned recognition inputs; perform no native measurements."""
    index_path = directory / "segments.json"
    digest = file_digest(index_path)
    index = record(json.loads(_bound_bytes(index_path, digest, MAX_DOCUMENT_BYTES)))
    entries = index.get("segments")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SEGMENTS:
        raise ReplayError("Expected bounded support audit segments")
    if output.exists() or output.absolute() != output.resolve():
        raise ReplayError("Support audit requires a new canonical output file")
    pins = {str(index_path.resolve()): digest}
    rows: list[dict[str, object]] = []
    for entry in entries:
        row = record(entry)
        binding = segment_binding(directory, row)
        binding_digest = file_digest(binding)
        document = record(json.loads(_bound_bytes(binding, binding_digest, MAX_DOCUMENT_BYTES)))
        tokens, token_pins = read_tokens(directory, row, record(document.get("inputs")))
        pins.update(token_pins)
        pins[str(binding.resolve())] = binding_digest
        occurrences = caption_indivisible_fit.support_occurrences(tokens)
        rows.append(
            {
                "segment": row.get("segment"),
                "tokens": len(tokens),
                "units": [
                    record(json.loads(json.dumps(asdict(item), default=str)))
                    for item in occurrences
                ],
                "outsideUnitTimeFragments": sum(
                    not (
                        Fraction(str(item.unit.start)) <= fragment.start
                        and fragment.end <= Fraction(str(item.unit.end))
                    )
                    for item in occurrences
                    for fragment in item.identity.fragments
                ),
            }
        )
    if any(file_digest(Path(name)) != value for name, value in pins.items()):
        raise ReplayError("Support audit input changed")
    save(
        output,
        {
            "version": 1,
            "kind": "recognition-support-audit",
            "inputs": pins,
            "segments": rows,
            "recognitionClocksChanged": False,
            "measurementsPerformed": False,
            "deliveryVerified": False,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--source-media", type=Path, required=True)
    parser.add_argument("--ass-font-size", type=float, required=True)
    parser.add_argument("--point-size", type=float, required=True)
    parser.add_argument("--border", type=float, required=True)
    parser.add_argument("--rim", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--segment", type=int, action="append", default=[])
    parser.add_argument("--token-sources", type=Path)
    parser.add_argument("--measurement-cache", type=Path)
    parser.add_argument("--long-executable", type=Path)
    parser.add_argument("--measurement-cache-segment", type=int, action="append", default=[])
    parser.add_argument("--collect-layout-failures", action="store_true")
    parser.add_argument("--split-wide", action="store_true", help="page instead of scaling")
    parser.add_argument("--audit-support-only", action="store_true")
    parser.add_argument(
        "--resume-producer-evidence", type=Path, nargs=2, action="append", default=[]
    )
    args = parser.parse_args()
    if args.audit_support_only:
        audit_support(args.directory, args.output)
        return
    run(
        args.directory,
        args.output,
        executable=args.executable,
        style=NativeCaptionStyle(
            caption_style_defaults.FONT_POSTSCRIPT_NAME,
            args.ass_font_size,
            args.border,
            args.rim,
            args.ass_font_size,
        ),
        point_size=args.point_size,
        width=args.width,
        evidence=tuple(args.evidence),
        source_media=args.source_media,
        selected=tuple(args.segment),
        token_sources=args.token_sources,
        resume_evidence=tuple((part, manifest) for part, manifest in args.resume_producer_evidence),
        measurement_cache=args.measurement_cache,
        cache_segments=tuple(args.measurement_cache_segment),
        collect_layout_failures=args.collect_layout_failures,
        long_executable=args.long_executable,
        split_wide=args.split_wide,
    )


if __name__ == "__main__":
    main()
