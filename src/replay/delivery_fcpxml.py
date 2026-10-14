"""Write an editable Final Cut Pro timeline for a rendered delivery.

The project mirrors the delivered edit: the caption-removed source pieces of
every retained cut on the primary storyline, each Motion-rendered caption part
as a connected clip at its output offset, and the cut voice track as connected
audio. It is generated from the same render configuration as the delivery and
is imported into Final Cut Pro through native Open Document.
"""

from __future__ import annotations

import argparse
import sys
import wave
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import quoteattr

from replay.compressor_batch import actual_output
from replay.core import ReplayError
from replay.delivery_render import (
    Config,
    clips,
    load_config,
    motion_parts,
    read_chunks,
    read_cuts,
)


def _time(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}s" if value.denominator != 1 else f"{value}s"


def _url(path: Path) -> str:
    return "file://" + quote(str(path))


def _video_asset(identifier: str, name: str, path: Path) -> str:
    return (
        f'<asset id="{identifier}" name={quoteattr(name)} start="0s" hasVideo="1" '
        f'hasAudio="0" format="r1"><media-rep kind="original-media" '
        f"src={quoteattr(_url(path))}/></asset>"
    )


def build(config: Config, *, title: str) -> str:
    """Return FCPXML 1.11 text for the configured delivery."""
    rate = config.rate
    pieces = clips(read_cuts(config.cuts), read_chunks(config.chunks), rate)
    total = sum(piece.frames for piece in pieces)
    alpha = [
        (
            actual_output(
                config.storage / "alpha" / f"segment-{p.segment:04d}-part-{p.part:04d}.mov"
            ),
            p,
        )
        for p in motion_parts(config)
    ]
    voice = config.work / "audio-selected" / "selected.wav"
    with wave.open(str(voice), "rb") as audio:
        audio_rate, channels, samples = (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getnframes(),
        )
    if audio_rate <= 0 or not 1 <= channels <= 64 or samples <= 0:
        raise ReplayError("Invalid selected voice format")
    voice_duration = Fraction(samples, audio_rate)
    media = sorted({piece.media for piece in pieces})
    ids = {path: f"r{index + 2}" for index, path in enumerate(media)}
    resources = [
        f'<format id="r1" frameDuration="1/{rate}s" width="{config.width}" '
        f'height="{config.height}" colorSpace="1-1-1 (Rec. 709)"/>'
    ]
    resources += [_video_asset(ids[path], path.parent.name, path) for path in media]
    overlay_ids = {item[0]: f"a{index}" for index, item in enumerate(alpha)}
    resources += [_video_asset(overlay_ids[item[0]], item[0].stem, item[0]) for item in alpha]
    resources.append(
        f'<asset id="voice" name="voice" start="0s" hasVideo="0" hasAudio="1" '
        f'audioSources="1" audioChannels="{channels}" audioRate="{audio_rate}" '
        f'duration="{_time(voice_duration)}">'
        f'<media-rep kind="original-media" src={quoteattr(_url(voice))}/></asset>'
    )
    spine: list[str] = []
    offset = 0
    for index, piece in enumerate(pieces):
        duration = Fraction(piece.frames, rate)
        children = ""
        if index == 0:
            children = (
                f'<asset-clip ref="voice" lane="-1" offset="{_time(piece.start)}" start="0s" '
                f'duration="{_time(voice_duration)}" name="voice"/>'
            )
        children += "".join(
            f'<asset-clip ref="{overlay_ids[path]}" lane="1" '
            f'offset="{_time(piece.start + Fraction(part.offset - offset, rate))}" start="0s" '
            f'duration="{_time(Fraction(part.frames, rate))}" name="captions"/>'
            for path, part in alpha
            if offset <= part.offset < offset + piece.frames
        )
        spine.append(
            f'<asset-clip ref="{ids[piece.media]}" offset="{_time(Fraction(offset, rate))}" '
            f'start="{_time(piece.start)}" duration="{_time(duration)}" '
            f'name="cut {index}">{children}</asset-clip>'
        )
        offset += piece.frames
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
        f'<fcpxml version="1.11"><resources>{"".join(resources)}</resources>'
        f"<library><event name={quoteattr(title)}><project name={quoteattr(title)}>"
        f'<sequence format="r1" duration="{_time(Fraction(total, rate))}" tcStart="0s" '
        f'tcFormat="NDF"><spine>{"".join(spine)}</spine></sequence></project></event>'
        "</library></fcpxml>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args(argv)
    try:
        text = build(load_config(args.config), title=args.title)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (ReplayError, OSError, ValueError, wave.Error, EOFError) as error:
        print(f"DELIVERY_FCPXML_ERROR {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
