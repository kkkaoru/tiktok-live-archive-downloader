"""Reflow existing recognition stages using actual point-size-checked native widths.

Only explicit line breaks change. Old origins are discarded; the face-relative
retargeter must independently supply positions for the new layout.
An overlong phrase fails for token-clock replanning instead of inventing times.
"""

import argparse
import json
from fractions import Fraction
from itertools import chain, combinations
from pathlib import Path

from replay import caption_linguistics, caption_style_defaults
from replay.caption_native_layout import NativeCaptionLayout, NativeCaptionStyle
from replay.caption_timing import display_time
from replay.core import ReplayError
from replay.fcp_measured import MAX_CANDIDATE_BYTES, read_measured_captions
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest


def _phrase_lines(text: str) -> tuple[str, ...]:
    points = (0, *caption_linguistics.boundaries(text)[0])
    return tuple(text[start:end] for start, end in combinations(points, 2))


def reflow(
    source: Path,
    output: Path,
    *,
    layout: NativeCaptionLayout,
    duration: Fraction,
    rate: int,
    source_media: Path,
    offset: Fraction = Fraction(0),
    window: tuple[Fraction, Fraction] | None = None,
) -> None:
    """Keep every stage and recognized clock; fit POS phrases within a pixel width.

    Vertical bounds are deliberately not inferred from ASS leading. The height
    returned to the phrase chooser is a line count, with the shared maximum as
    its budget. Downstream face/paragraph geometry requires separate observation.
    """
    if layout.expected_point_size is None:
        raise ReplayError("Reflow requires the actual Motion point-size contract")
    if offset < 0:
        raise ReplayError("Caption segment offset must be nonnegative")
    first, last = window or (Fraction(0), duration)
    if not 0 <= first < last <= duration:
        raise ReplayError("Explicit caption window exceeds segment coverage")
    digest = file_digest(source)
    raw = record(json.loads(_bound_bytes(source, digest, MAX_CANDIDATE_BYTES)))
    canvas = raw.get("canvas")
    if canvas != [1080, 1920]:
        raise ReplayError("Reflow requires the supported portrait canvas")
    cues = read_measured_captions(
        source, sha256=digest, width=1080, height=1920, rate=rate, duration=duration
    )
    cues = tuple(
        item
        for item in cues
        if display_time(item.cue.start) < last and display_time(item.cue.end) > first
    )
    if not cues:
        raise ReplayError("Explicit caption window contains no recognized stages")
    old_inputs = record(raw.get("inputs"))
    media = str(source_media.resolve())
    media_digest = old_inputs.get(media)
    if not isinstance(media_digest, str) or len(media_digest) != 64:
        raise ReplayError("Recognition candidate must bind the explicit original source")
    texts = tuple(dict.fromkeys(item.cue.text.replace("\n", "") for item in cues))
    layout.prepare_lines(texts)
    overflow = tuple(text for text in texts if layout.measure(text)[0] > layout.width)
    layout.prepare_lines(chain.from_iterable(_phrase_lines(text) for text in overflow))

    def horizontal_bounds(text: str) -> tuple[float, float]:
        lines = text.split("\n")
        return max(layout.measure(line)[0] for line in lines), float(len(lines))

    candidates: list[dict[str, object]] = []
    changed = 0
    previous_end = Fraction(0)
    for positioned in cues:
        text = positioned.cue.text
        start = max(first, display_time(positioned.cue.start))
        end = min(last, display_time(positioned.cue.end))
        if start < previous_end:
            raise ReplayError("Overlapping recognition stages require prior hold reconciliation")
        previous_end = end
        try:
            wrapped = caption_linguistics.wrap_measured_lines(
                text.replace("\n", ""),
                measure=horizontal_bounds,
                width=layout.width,
                height=caption_style_defaults.MAX_CAPTION_LINES,
                maximum_lines=caption_style_defaults.MAX_CAPTION_LINES,
                prepare_lines=layout.prepare_lines,
            )
        except caption_linguistics.CaptionLayoutError as error:
            raise caption_linguistics.CaptionLayoutError(
                f"Recognized stage {start}..{end} needs token-clock paging: {text!r}"
            ) from error
        candidates.append(
            {"start": float(start + offset), "end": float(end + offset), "text": wrapped}
        )
        changed += wrapped != text
    inputs = layout.verify_inputs() | {media: media_digest}
    inputs[str(source.resolve())] = digest
    inputs[str(Path(__file__).resolve())] = file_digest(Path(__file__))
    policy = {
        "version": 1,
        "scope": "horizontal-phrase-layout-only",
        "pointSize": layout.expected_point_size,
        "paragraphWidth": layout.width,
        "maximumLines": caption_style_defaults.MAX_CAPTION_LINES,
        "widthMeasurement": "native-point-size-checked-glyph-bounds-including-border-rim",
        "boundaries": "pinned-vibrato-ipadic-phrase-boundaries",
        "changedStages": changed,
        "recognitionClocksChanged": False,
        "explicitWindowClipping": window is not None,
        "originsRequireRetargeting": True,
        "motionRenderVerified": False,
    }
    if file_digest(source) != digest:
        raise ReplayError("Recognition candidate changed during reflow")
    layout._save(
        output,
        {
            "version": 1,
            "candidateOnly": True,
            "inputs": inputs,
            "historicalRecognitionInputs": old_inputs,
            "captions": candidates,
            "coverage": {"start": float(offset + first), "end": float(offset + last)},
            "layout": policy,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--duration", type=Fraction, required=True)
    parser.add_argument("--source-media", type=Path, required=True)
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--offset", type=Fraction, default=Fraction(0))
    parser.add_argument("--window", type=Fraction, nargs=2)
    parser.add_argument("--ass-font-size", type=float, required=True)
    parser.add_argument("--point-size", type=float, required=True)
    parser.add_argument("--border", type=float, required=True)
    parser.add_argument("--rim", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    args = parser.parse_args()
    layout = NativeCaptionLayout(
        executable=args.executable,
        directory=args.measurements,
        style=NativeCaptionStyle(
            caption_style_defaults.FONT_POSTSCRIPT_NAME,
            args.ass_font_size,
            args.border,
            args.rim,
            args.ass_font_size,
        ),
        width=args.width,
        height=4096,
        expected_point_size=args.point_size,
        evidence=tuple(args.evidence),
    )
    reflow(
        args.source,
        args.output,
        layout=layout,
        duration=args.duration,
        rate=args.rate,
        source_media=args.source_media,
        offset=args.offset,
        window=None if args.window is None else (args.window[0], args.window[1]),
    )


if __name__ == "__main__":
    main()
