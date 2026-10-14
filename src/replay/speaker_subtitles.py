"""Render validated speaker-timed captions in three independent landscape lanes."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from replay.caption_timing import punctuation_lines
from replay.core import ReplayError

# ASS colors are &HAABBGGRR. Video is 1920x1080; lanes cannot overlap horizontally.
STYLES = {
    "left": (650, "&H00DC429A"),  # purple; clear of source video live-chat column
    "middle": (1140, "&H0069C847"),  # green
    "right": (1650, "&H00E2CF5A"),  # cyan
    "unassigned": (960, "&H00404040"),  # readable, never invent an identity for overlap
}
# Speaker lanes use solid fills in the single-speaker style: purple, green, light blue.
LANE_FILLS = {
    "left": (0xC4, 0x8C, 0xFF),
    "middle": (0x7C, 0xDC, 0x8C),
    "right": (0x6C, 0xD4, 0xFF),
    "unassigned": (0xC8, 0xC8, 0xC8),
}
# Blue-leaning cyan: specified sRGB contrast against the white border is 7.1367:1.
# Actual rendered color and outline coverage still require independent verification.
SINGLE_FILL = (0x00, 0x5A, 0x9C)
WHITE_BORDER = 5
BLACK_RIM = 2
FONT_SIZE = 38  # 12 wrapped characters plus borders stay clear of adjacent lanes
STACK_GAP = 12
SINGLE_FONT_SIZE = 52
LINE_HEIGHT = 1.25
SINGLE_WRAP = 24
WRAP_ORPHAN_MAXIMUM = 4  # a last wrapped line this short (e.g. "かな?") is rebalanced
_NO_LINE_START = "、。！？!?.,;:…」』）)]｝ー〜～ぁぃぅぇぉっゃゅょァィゥェォッャュョ"
_NO_LINE_END = "「『（([｛"
SINGLE_BOTTOM = 1040
PLAY_RES = (1920, 1080)
SPEAKER_NAMES = {"left": "左", "middle": "中央", "right": "右"}


@dataclass(frozen=True)
class SpeakerCue:
    start: float
    end: float
    speaker: str
    text: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end <= self.start
            or self.speaker not in STYLES
            or not self.text.strip()
        ):
            raise ValueError("Invalid speaker cue")


def read_cues(path: Path) -> tuple[SpeakerCue, ...]:
    """Require per-speaker timing, not guessed character identity from arrival order."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["start", "end", "speaker", "text"]:
            raise ReplayError("Expected CSV columns: start,end,speaker,text")
        try:
            cues = tuple(
                SpeakerCue(float(row["start"]), float(row["end"]), row["speaker"], row["text"])
                for row in reader
            )
        except (ValueError, TypeError, KeyError) as error:
            raise ReplayError("Invalid speaker cue CSV") from error
    if not cues:
        raise ReplayError("Speaker cue CSV is empty")
    return cues


def ass_clock(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    whole, fraction = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02}:{whole:02}.{fraction:02}"


def _serial_events(events: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Represent active caption sets at original boundaries; invent no speech clocks."""
    boundaries: list[tuple[int, bool, int]] = []
    for index, event in enumerate(events):
        first, last = round(event[0] * 100), round(event[1] * 100)
        if last <= first:
            raise ValueError("Caption collapses at subtitle clock precision")
        boundaries.extend(((first, True, index), (last, False, index)))
    active: dict[int, str] = {}
    result: list[tuple[float, float, str]] = []
    previous = 0
    for tick, starts, index in sorted(boundaries):
        if tick > previous and active:
            result.append((previous / 100, tick / 100, "\n".join(active.values())))
        if starts:
            active[index] = events[index][2]
        else:
            del active[index]
        previous = tick
    return result


def multiple_speakers(cues: tuple[SpeakerCue, ...]) -> bool:
    """Names are shown only when at least two identified speakers are present."""
    return len({cue.speaker for cue in cues if cue.speaker in SPEAKER_NAMES}) >= 2


def render_srt(cues: tuple[SpeakerCue, ...], *, serial: bool = False) -> str:
    """Keep native sidecars; label speakers only for identified multi-speaker videos.

    A single or unknown speaker is never named. Serialize active display sets for
    non-overlapping MP4 samples when requested.
    """
    labelled = multiple_speakers(cues)
    events = [
        (
            cue.start,
            cue.end,
            f"[{SPEAKER_NAMES[cue.speaker]}] {cue.text}"
            if labelled and cue.speaker in SPEAKER_NAMES
            else cue.text,
        )
        for cue in sorted(cues, key=lambda item: (item.start, item.speaker))
    ]
    if serial:
        events = _serial_events(events)
    blocks: list[str] = []
    for number, (first, last, text) in enumerate(events, 1):
        start = ass_clock(first).replace(".", ",") + "0"
        end = ass_clock(last).replace(".", ",") + "0"
        if start.index(":") == 1:
            start = "0" + start
        if end.index(":") == 1:
            end = "0" + end
        blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def wrap_text(text: str, width: int = 12, *, balance: bool = False) -> str:
    """Prefer established punctuation boundaries; keep Japanese marks with their word.

    With ``balance`` an over-long chunk is split into lines of nearly equal length
    instead of a full line followed by a remainder.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        chunks = punctuation_lines(paragraph).splitlines()
        if "".join(chunks) != paragraph:
            chunks = [paragraph]  # Preserve external text, including leading marks/spaces.
        line = ""
        for chunk in chunks:
            if line and len(line + chunk) > width:
                lines.append(line)
                line = ""
            while len(chunk) > width:
                cut = width
                if balance or 0 < len(chunk) % width <= WRAP_ORPHAN_MAXIMUM:
                    # Balance the lines so a short tail such as "よ" is never alone.
                    cut = math.ceil(len(chunk) / math.ceil(len(chunk) / width))
                while cut > 1 and (chunk[cut] in _NO_LINE_START or chunk[cut - 1] in _NO_LINE_END):
                    cut -= 1
                lines.append(chunk[:cut])
                chunk = chunk[cut:]
            line += chunk
        if line:
            lines.append(line)
    return "\n".join(lines)


def wrap_two_lines(text: str, width: int) -> str:
    """Keep established wrapping, or rebalance globally within a strict two-line box."""
    if type(width) is not int or not 1 <= width <= 80:
        raise ValueError("Invalid two-line width")
    wrapped = wrap_text(text, width, balance=True)
    if len(wrapped.splitlines()) <= 2:
        return wrapped
    flat = wrapped.replace("\n", "")
    cuts = [
        cut
        for cut in range(max(1, len(flat) - width), min(width, len(flat) - 1) + 1)
        if flat[cut] not in _NO_LINE_START and flat[cut - 1] not in _NO_LINE_END
    ]
    if not cuts:
        raise ValueError("Caption cannot fit two lines without a prohibited break")
    cut = min(cuts, key=lambda value: abs(len(flat) - 2 * value))
    return flat[:cut] + "\n" + flat[cut:]


def bgr(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"&H{blue:02X}{green:02X}{red:02X}&"


def bordered_layers(
    times: str,
    style: str,
    position: str,
    text: str,
    fills: list[tuple[int, int, str]],
    *,
    width: int = PLAY_RES[0],
    border: int = WHITE_BORDER,
    rim: int = BLACK_RIM,
) -> list[str]:
    """Thin black rim outside a thick white border, then the fill on top."""
    events = [
        f"Dialogue: 0,{times},{style},,0,0,0,,"
        f"{{{position}\\bord{border + rim}\\shad0\\3c&H000000&\\1a&HFF&}}{text}",
        f"Dialogue: 1,{times},{style},,0,0,0,,"
        f"{{{position}\\bord{border}\\shad0\\3c&HFFFFFF&\\1a&HFF&}}{text}",
    ]
    for top, bottom, color in fills:
        events.append(
            f"Dialogue: 2,{times},{style},,0,0,0,,"
            f"{{{position}\\bord0\\shad0\\1c{color}\\clip(0,{top},{width},{bottom})}}"
            f"{text}"
        )
    return events


def clean_text(text: str) -> str:
    # Avoid accepting ASS override tags in external transcript text.
    clean = text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    return clean.replace("\r\n", "\n").replace("\r", "\n")


def ass_header(
    styles: list[tuple[str, int]],
    *,
    frame: tuple[int, int] = PLAY_RES,
    border: int = WHITE_BORDER,
) -> str:
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {frame[0]}\nPlayResY: {frame[1]}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
    )
    for name, size in styles:
        header += (
            f"Style: {name},Hiragino Sans,{size},&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,&H00000000,"
            f"-1,0,0,0,100,100,0,0,1,{border},0,2,0,0,0,1\n"
        )
    return header + (
        "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )


class _LaneStack:
    """Place captions that overlap in time within one lane in the lowest free slot."""

    def __init__(self) -> None:
        self.active: list[tuple[float, int, int]] = []

    def bottom(self, start: float, end: float, base: int, height: int) -> int:
        self.active = [item for item in self.active if item[0] > start]
        candidates = sorted({base, *(top - STACK_GAP for _, top, _ in self.active)}, reverse=True)
        # The highest candidate sits above every active caption, so it is always free.
        bottom = next(
            (
                candidate
                for candidate in candidates
                if all(
                    candidate <= other_top - STACK_GAP
                    or candidate - height >= other_bottom + STACK_GAP
                    for _, other_top, other_bottom in self.active
                )
            ),
            candidates[-1],
        )
        self.active.append((end, bottom - height, bottom))
        return bottom


def render_ass(cues: tuple[SpeakerCue, ...]) -> str:
    """Burn captions without speaker names.

    Several identified speakers keep distinct horizontal lanes with solid purple,
    green and light-blue fills. A single or unknown speaker uses one blue-cyan
    bottom-centre caption. Every caption has a thick white border with a thin black
    rim outside it.
    """
    if not multiple_speakers(cues):
        return _render_single(cues)
    events: list[str] = []
    stacks = {speaker: _LaneStack() for speaker in STYLES}
    for cue in sorted(cues, key=lambda item: (item.start, item.speaker)):
        x, _ = STYLES[cue.speaker]
        clean = clean_text(cue.text)
        # ponytail: conservative character-width lanes; shape glyphs if widths need expansion.
        limit = 72 if cue.speaker == "unassigned" else 36
        if len(clean.replace("\n", "")) > limit:
            raise ValueError("Split long speech into shorter speaker cues")
        wrapped = wrap_text(clean)
        lines = wrapped.count("\n") + 1
        height = math.ceil(lines * FONT_SIZE * LINE_HEIGHT)
        base = 825 if cue.speaker == "unassigned" else 940
        y = stacks[cue.speaker].bottom(cue.start, cue.end, base, height)
        fills = [(0, PLAY_RES[1], bgr(LANE_FILLS[cue.speaker]))]
        events.extend(
            bordered_layers(
                f"{ass_clock(cue.start)},{ass_clock(cue.end)}",
                cue.speaker,
                f"\\an2\\pos({x},{y})",
                wrapped.replace("\n", "\\N"),
                fills,
            )
        )
    styles = [(speaker, FONT_SIZE) for speaker in STYLES]
    return ass_header(styles) + "\n".join(events) + "\n"


def _render_single(cues: tuple[SpeakerCue, ...]) -> str:
    events: list[str] = []
    fill = bgr(SINGLE_FILL)
    stack = _LaneStack()
    for cue in sorted(cues, key=lambda item: (item.start, item.end)):
        wrapped = wrap_text(clean_text(cue.text), SINGLE_WRAP)
        height = math.ceil((wrapped.count("\n") + 1) * SINGLE_FONT_SIZE * LINE_HEIGHT)
        bottom = stack.bottom(cue.start, cue.end, SINGLE_BOTTOM, height)
        wrapped = wrapped.replace("\n", "\\N")
        events.extend(
            bordered_layers(
                f"{ass_clock(cue.start)},{ass_clock(cue.end)}",
                "single",
                f"\\an2\\pos({PLAY_RES[0] // 2},{bottom})",
                wrapped,
                [(0, PLAY_RES[1], fill)],
            )
        )
    return ass_header([("single", SINGLE_FONT_SIZE)]) + "\n".join(events) + "\n"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cues",
        type=Path,
        help="start,end,speaker,text CSV; speaker is left/middle/right/unassigned",
    )
    parser.add_argument("output", type=Path, help="new ASS sidecar for a 1920x1080 video")
    parser.add_argument("--srt", type=Path, help="optional matching TranscriptViewer SRT")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".ass" or (args.srt and args.srt.suffix.lower() != ".srt"):
        raise ReplayError("Outputs must be ASS and SRT subtitle files")
    if (
        args.output.exists()
        or args.output.is_symlink()
        or (args.srt and (args.srt.exists() or args.srt.is_symlink()))
    ):
        raise ReplayError("Refusing to overwrite subtitle output")
    cues = read_cues(args.cues)
    text = render_ass(cues)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    if args.srt:
        with args.srt.open("x", encoding="utf-8") as stream:
            stream.write(render_srt(cues))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
