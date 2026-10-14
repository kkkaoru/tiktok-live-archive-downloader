"""Extract a sample-exact PCM window for bounded local ASR refinement.

The receipt binds the source clock and bytes; recognition remains unreviewed and
must be projected back through verified recording/cut clocks by the caller.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.local_asr import MAX_SAMPLES, fingerprint

RATE = 16000


def extract(source: Path, output: Path, *, start_sample: int, sample_count: int) -> Path:
    """Write a new window and source-clock receipt, preserving failed attempts."""
    window = AudioWindow(start_sample, sample_count)
    if window.sample_count > MAX_SAMPLES:
        raise ReplayError("Recognition window exceeds 60 seconds")
    before = fingerprint(source)
    with wave.open(str(source), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != RATE
            or audio.getcomptype() != "NONE"
            or window.end_sample > audio.getnframes()
        ):
            raise ReplayError("Expected an in-range mono PCM16/16kHz window")
        audio.setpos(window.start_sample)
        payload = audio.readframes(window.sample_count)
    if len(payload) != window.sample_count * 2:
        raise ReplayError("Truncated recognition window")
    if fingerprint(source) != before:
        raise ReplayError("Source changed while extracting recognition window")
    output.mkdir(mode=0o700, parents=False)
    clip = output / "window.wav"
    with clip.open("xb") as stream:
        clip.chmod(0o600)
        with wave.open(stream, "wb") as audio_out:
            audio_out.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            audio_out.writeframes(payload)
    receipt = output / "window.json"
    with receipt.open("x", encoding="utf-8") as stream:
        receipt.chmod(0o600)
        json.dump(
            {
                "version": 1,
                "source": str(source.resolve()),
                "sourceSHA256": before,
                "startSample": window.start_sample,
                "endSample": window.end_sample,
                "sampleRate": RATE,
                "windowSHA256": fingerprint(clip),
                "clock": "continuous-input-pcm",
                "sourceClockProjected": False,
            },
            stream,
            indent=2,
        )
    return clip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-sample", type=int, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    args = parser.parse_args(argv)
    print(
        extract(
            args.source, args.output, start_sample=args.start_sample, sample_count=args.sample_count
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
