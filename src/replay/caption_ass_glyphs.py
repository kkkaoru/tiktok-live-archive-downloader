"""Measure actual libass ink bounds in a streamed atlas, without saved frame copies."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.caption_phrase_ass import Appearance, render_ass
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import ass_clock, clean_text
from replay.stages import file_digest

WIDTH, HEIGHT = 1920, 1080
ATLAS_RATE = 10
ANCHOR = 540
MAX_LINES = 10_000


def ink_bounds(rgb: NDArray[np.uint8]) -> tuple[int, int, int]:
    """Measure saturated green ink only; antialiased fringe stays outside the core."""
    mask = (rgb[:, :, 1] > 128) & (rgb[:, :, 0] < 32) & (rgb[:, :, 2] < 32)
    y, x = np.nonzero(mask)
    if not len(y):
        raise ReplayError("Atlas line has no resolved font ink")
    return int(y.min()) - ANCHOR, int(y.max()) + 1 - ANCHOR, int(x.max() - x.min() + 1)


def rasterize(
    lines: tuple[str, ...], *, appearance: Appearance, ffmpeg: Path, fonts: Path, output: Path
) -> dict[str, object]:
    """One process renders all text lines; outer workflow owns its timeout/group."""
    if not 1 <= len(lines) <= MAX_LINES or any(not text or "\n" in text for text in lines):
        raise ReplayError("Invalid bounded glyph atlas")
    output.mkdir(mode=0o700)
    atlas = output / "atlas.ass"
    events = [
        f"Dialogue: 0,{ass_clock(index / ATLAS_RATE)},"
        f"{ass_clock((index + 1) / ATLAS_RATE)},caption,,0,0,0,,"
        f"{{\\an2\\pos({WIDTH // 2},{ANCHOR})\\bord0\\shad0\\1c&H00FF00&}}{clean_text(text)}"
        for index, text in enumerate(lines)
    ]
    atlas.write_text(render_ass((), appearance) + "\n".join(events), encoding="utf-8")
    command = (
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "2",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={WIDTH}x{HEIGHT}:r={ATLAS_RATE}:d={len(lines) / ATLAS_RATE}",
        "-vf",
        f"ass=filename={atlas}:fontsdir={fonts}",
        "-frames:v",
        str(len(lines)),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    )
    records: dict[str, object] = {}
    with (
        (output / "render.log").open("xb") as log,
        subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log
        ) as process,
    ):
        if process.stdout is None:
            raise ReplayError("Atlas renderer has no pixel stream")
        for text in lines:
            payload = process.stdout.read(WIDTH * HEIGHT * 3)
            if len(payload) != WIDTH * HEIGHT * 3:
                raise ReplayError("Atlas renderer returned an incomplete frame")
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            top, bottom, width = ink_bounds(rgb)
            records[text] = {
                "inkTopOffsetPx": top,
                "inkBottomOffsetPx": bottom,
                "inkWidthPx": width,
            }
        if process.stdout.read(1) or process.wait(timeout=60) != 0:
            raise ReplayError("Atlas frame count or renderer exit differs")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("font_receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--font-size", type=float, required=True)
    args = parser.parse_args(argv)
    plan = record(json.loads(args.plan.read_text(encoding="utf-8")))
    rows = plan.get("captions")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Expected phrase caption rows")
    lines: list[str] = []
    for raw in rows:
        text = record(raw).get("text")
        if not isinstance(text, str) or not text.strip():
            raise ReplayError("Invalid phrase caption text")
        lines.extend(text.splitlines())
    font = record(json.loads(args.font_receipt.read_text(encoding="utf-8")))
    instance = Path(str(font.get("renderInstance")))
    if font.get("family") != "Noto Sans JP" or file_digest(instance) != font.get(
        "renderInstanceSHA256"
    ):
        raise ReplayError("Font render instance identity differs")
    appearance = Appearance(args.font_size, 960, 1032, 0, 1920, "Noto Sans JP")
    records = rasterize(
        tuple(dict.fromkeys(lines)),
        appearance=appearance,
        ffmpeg=args.ffmpeg,
        fonts=instance.parent,
        output=args.output,
    )
    result = {
        "version": 1,
        "method": "libass-streamed-green-ink-atlas",
        "planSHA256": file_digest(args.plan),
        "fontReceiptSHA256": file_digest(args.font_receipt),
        "fontSize": args.font_size,
        "lineCount": len(records),
        "records": records,
    }
    (args.output / "bounds.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
