"""Rebuild a looping BGM from a long recording in which speech covers it most of the time.

The music stem is the separation residual (mix minus the separated voice). Wherever someone
talks, that residual still carries leaked speech, so no single stretch of it is clean. A loop
repeats the same samples, though, so every phase of the loop is heard many times. The period
comes from the residual's own autocorrelation and every repetition is aligned to sub-sample
precision per segment. Each short phase block is rebuilt from the aligned repetitions by a
per-bin spectral median, taken per frame over only the copies in which nobody speaks at that
moment: under speech the separation also removes parts of the music, the same way in every
talked-over copy, so those copies agree with each other yet lack it; speech-free copies are
undamaged.
The blocks are crossfaded into one period, which wraps onto itself. When the period holds a
silence (a song that ends and is played again), the track is rotated to start at the attack
after it, so it opens at the song's beginning, and is tiled into a seamless longer track.
A BGM mixed under talk is quiet, so the period is scaled once, still in floating point, until its
oversampled true peak meets a ceiling, and the residual and output are 32-bit float (RF64 past
4 GiB) so neither the scaling nor any intermediate adds rounding noise.
Nothing here depends on a particular recording: the period, alignment, gain, choice and start
all come from measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError

FloatArray = NDArray[np.float64]

CHANNELS = 2
SAMPLE_WIDTH = 2
WORD_BYTES = 4
# Intermediates and outputs are 32-bit float, read back in the same 16-bit units as the inputs.
FLOAT_WIDTH = 4
WAVE_PCM = 1
WAVE_FLOAT = 3
WAVE_EXTENSIBLE = 0xFFFE
# A RIFF size field holds at most this; larger files are RF64 with the sizes in a ds64 chunk.
RIFF_SIZE_LIMIT = 0xFFFFFFFF
# Room reserved after the RIFF header for a ds64 chunk (riff, data, sample count, table length).
DS64_BYTES = 28
INT16_LIMIT = 32767.0
FULL_SCALE = 32768.0
# Leaves headroom for the AAC encoder's own overshoot; inter-sample peaks are measured.
PEAK_CEILING_DB = -1.0
TRUE_PEAK_OVERSAMPLING = 4
ANALYSIS_DECIMATION = 20
ANALYSIS_CHUNK_SECONDS = 60.0
VOICE_FRAME_SECONDS = 0.1
MIN_PERIOD_SECONDS = 10.0
MAX_PERIOD_SECONDS = 1200.0
# A shorter lag whose correlation is this close to the best one is the true period, not a harmonic.
PEAK_RATIO = 0.85
SEGMENT_SECONDS = 10.0
SEARCH_FRAMES = 2048
MIN_ALIGNMENT_NCC = 0.2
BLOCK_SECONDS = 1.0
CROSSFADE_SECONDS = 0.05
CANDIDATES = 32
FRACTIONAL_TAPS = 32
# Relative error to the consensus above which a block is reported as having no clean repetition.
# A block whose best pair of repetitions correlates below 1 - this has no clean copy.
UNCLEAN_ERROR = 0.2
# Copies correlating this well with the best one are aligned; others are misplaced or foreign.
MEMBER_NCC = 0.5
# Voice-stem level below this share of its typical level means nobody speaks there.
VOICE_FREE_RATIO = 0.01
# The fewest members for which a median rejects a leak instead of averaging it in.
MEDIAN_MEMBERS = 3
# Where too few copies are speech-free, each bin takes the copies with the least voice in that
# bin (separation strips music only where it hears voice, which differs bin by bin): a third of
# the copies, so a median of mostly clean ones still rejects the rest, within these bounds.
BIN_MEMBERS = 8
BIN_SHARE = 3
STFT_FRAME = 2048
STFT_HOP = 512
HASH_CHUNK_BYTES = 1 << 22
SILENCE_WINDOW_SECONDS = 0.2
SILENCE_HOP_SECONDS = 0.01
# Quieter than this share of the loop's median level is silence; a sample above the attack share
# after it, clear of the silence's dither and codec noise, is where the song starts again.
SILENCE_RATIO = 0.05
ATTACK_RATIO = 0.25
LEAD_IN_SECONDS = 0.01


@dataclass(frozen=True)
class Alignment:
    """Per repetition and phase segment, the frame offset of the loop from its nominal grid."""

    grid_frames: float
    period_frames: float
    segment_frames: int
    offsets: FloatArray  # repetitions x segments; NaN where a repetition cannot be aligned
    gains: FloatArray  # per repetition, its music level relative to the typical repetition
    reference: int


@dataclass(frozen=True)
class Block:
    start: int
    repetition: int
    error: float
    voice: float
    members: int
    short_frames: int


@dataclass(frozen=True)
class WaveFormat:
    rate: int
    frames: int
    width: int
    floating: bool
    data_offset: int


def _wave_format(stream: BinaryIO) -> WaveFormat:
    """The sample layout and data position of a RIFF or RF64 WAV."""
    magic, _, kind = struct.unpack("<4sI4s", stream.read(12))
    if magic not in (b"RIFF", b"RF64") or kind != b"WAVE":
        raise ReplayError("Expected a RIFF or RF64 WAV")
    long_size: int | None = None
    layout: tuple[int, int, int, int] | None = None
    while header := stream.read(8):
        chunk, size = struct.unpack("<4sI", header)
        body = stream.tell()
        if chunk == b"ds64":
            long_size = struct.unpack("<QQ", stream.read(16))[1]
        elif chunk == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", stream.read(8))
            stream.seek(6, 1)  # byte rate and block align follow from the rest
            (bits,) = struct.unpack("<H", stream.read(2))
            if tag == WAVE_EXTENSIBLE:
                stream.seek(8, 1)  # extension size, valid bits and channel mask
                (tag,) = struct.unpack("<H", stream.read(2))
            layout = (tag, channels, rate, bits)
        elif chunk == b"data":
            if layout is None:
                raise ReplayError("WAV data precedes its format")
            tag, channels, rate, bits = layout
            width = bits // 8
            if channels != CHANNELS or (tag, width) not in _DECODERS:
                raise ReplayError("Expected 16-, 24- or 32-bit PCM or 32-bit float stereo WAV")
            length = long_size if size == RIFF_SIZE_LIMIT and long_size is not None else size
            return WaveFormat(rate, length // (width * CHANNELS), width, tag == WAVE_FLOAT, body)
        stream.seek(body + size + size % 2)
    raise ReplayError("WAV has no data chunk")


def _integers(raw: bytes, width: int) -> FloatArray:
    """Little-endian PCM in 16-bit units: each sample goes into the top bytes of a 32-bit word."""
    data = np.frombuffer(raw, np.uint8).reshape(-1, width)
    words = np.zeros((len(data), WORD_BYTES), np.uint8)
    words[:, WORD_BYTES - width :] = data
    values: FloatArray = words.view("<i4")[:, 0] / float(1 << 16)
    return values


def _floats(raw: bytes, _width: int) -> FloatArray:
    values: FloatArray = np.frombuffer(raw, "<f4") * FULL_SCALE
    return values


_DECODERS = {
    (WAVE_PCM, 2): _integers,
    (WAVE_PCM, 3): _integers,
    (WAVE_PCM, 4): _integers,
    (WAVE_FLOAT, FLOAT_WIDTH): _floats,
}


class Stereo:
    """Random access to a stereo PCM or float WAV in 16-bit units; outside it is silence."""

    def __init__(self, stream: BinaryIO) -> None:
        layout = _wave_format(stream)
        self._stream = stream
        self._layout = layout
        self._decode = _DECODERS[(WAVE_FLOAT if layout.floating else WAVE_PCM, layout.width)]
        self.rate = layout.rate
        self.frames = layout.frames

    def read(self, start: int, count: int) -> FloatArray:
        out = np.zeros((count, CHANNELS))
        first, last = max(start, 0), min(start + count, self.frames)
        if last > first:
            block = self._layout.width * CHANNELS
            self._stream.seek(self._layout.data_offset + first * block)
            raw = self._stream.read((last - first) * block)
            out[first - start : last - start] = self._decode(raw, self._layout.width).reshape(
                -1, CHANNELS
            )
        return out


@contextmanager
def opened(path: Path) -> Iterator[Stereo]:
    with path.open("rb") as stream:
        yield Stereo(stream)


class Sink:
    """Appends interleaved stereo frames to a WAV whose sizes are filled in on close."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def writeframes(self, data: bytes) -> None:
        self._stream.write(data)


@contextmanager
def written(path: Path, rate: int, *, floating: bool = False) -> Iterator[Sink]:
    """A 16-bit PCM or 32-bit float stereo WAV; one past 4 GiB becomes RF64."""
    width = FLOAT_WIDTH if floating else SAMPLE_WIDTH
    tag = WAVE_FLOAT if floating else WAVE_PCM
    block = width * CHANNELS
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sI4s4sI", b"RIFF", 0, b"WAVE", b"JUNK", DS64_BYTES))
        stream.write(bytes(DS64_BYTES))
        stream.write(
            struct.pack(
                "<4sIHHIIHH", b"fmt ", 16, tag, CHANNELS, rate, rate * block, block, 8 * width
            )
        )
        stream.write(struct.pack("<4sI", b"data", 0))
        start = stream.tell()
        yield Sink(stream)
        data = stream.tell() - start  # whole stereo frames: always even, so no pad byte
        riff = stream.tell() - 8
        if riff <= RIFF_SIZE_LIMIT:
            stream.seek(4)
            stream.write(struct.pack("<I", riff))
            stream.seek(start - 4)
            stream.write(struct.pack("<I", data))
            return
        stream.seek(0)
        stream.write(
            struct.pack("<4sI4s4sI", b"RF64", RIFF_SIZE_LIMIT, b"WAVE", b"ds64", DS64_BYTES)
        )
        stream.write(struct.pack("<QQQI", riff, data, data // block, 0))
        stream.seek(start - 4)
        stream.write(struct.pack("<I", RIFF_SIZE_LIMIT))


def pcm(samples: FloatArray) -> bytes:
    return np.clip(np.rint(samples), -INT16_LIMIT - 1, INT16_LIMIT).astype("<i2").tobytes()


def float_pcm(samples: FloatArray) -> bytes:
    """32-bit float samples, full scale 1.0, of samples in 16-bit units; nothing is rounded off
    that a 16- or 24-bit file could hold."""
    return (samples / FULL_SCALE).astype("<f4").tobytes()


def residual(parts: Sequence[tuple[Path, Path]], *, keep_seconds: float, out: Path) -> int:
    """Join `mix - voice` of consecutive overlapping parts; each part keeps its first span."""
    if not parts:
        raise ReplayError("No parts to subtract")
    total = 0
    with opened(parts[0][0]) as first:
        rate = first.rate
    with written(out, rate, floating=True) as sink:
        for index, (mix_path, voice_path) in enumerate(parts):
            with opened(mix_path) as mix, opened(voice_path) as voice:
                if (mix.rate, mix.frames) != (voice.rate, voice.frames) or mix.rate != rate:
                    raise ReplayError(f"{mix_path.name} and {voice_path.name} do not match")
                last = index == len(parts) - 1
                keep = mix.frames if last else min(mix.frames, round(keep_seconds * rate))
                sink.writeframes(float_pcm(mix.read(0, keep) - voice.read(0, keep)))
                total += keep
    return total


def analysis(music: Stereo, voice: Stereo) -> tuple[FloatArray, FloatArray]:
    """A decimated mono music signal and the voice RMS per `VOICE_FRAME_SECONDS`."""
    if (music.rate, music.frames) != (voice.rate, voice.frames):
        raise ReplayError("Music and voice stems differ in rate or length")
    step = round(VOICE_FRAME_SECONDS * music.rate)
    # Whole voice frames and whole decimation groups, so chunks join without a seam.
    chunk = round(ANALYSIS_CHUNK_SECONDS * music.rate) // (step * ANALYSIS_DECIMATION)
    chunk *= step * ANALYSIS_DECIMATION
    low: list[FloatArray] = []
    level: list[FloatArray] = []
    for start in range(0, music.frames, chunk):
        count = min(chunk, music.frames - start)
        mono = music.read(start, -(-count // ANALYSIS_DECIMATION) * ANALYSIS_DECIMATION)
        low.append(mono.mean(axis=1).reshape(-1, ANALYSIS_DECIMATION).mean(axis=1))
        frames = voice.read(start, -(-count // step) * step).mean(axis=1).reshape(-1, step)
        level.append(np.sqrt((frames**2).mean(axis=1)))
    return np.concatenate(low), np.concatenate(level)


def _parabolic(values: FloatArray, index: int) -> float:
    if not 0 < index < len(values) - 1:
        return float(index)
    left, middle, right = values[index - 1], values[index], values[index + 1]
    curvature = left - 2 * middle + right
    return index + (0.5 * (left - right) / curvature if curvature < 0 else 0.0)


def coarse_period(low: FloatArray, rate: float) -> float:
    """The loop period in seconds from the normalized autocorrelation of the decimated music."""
    shortest, longest = round(MIN_PERIOD_SECONDS * rate), round(MAX_PERIOD_SECONDS * rate)
    longest = min(longest, len(low) // 2)
    if longest <= shortest:
        raise ReplayError("The recording is too short to hold two repetitions")
    centred = low - low.mean()
    size = 1 << int(2 * len(centred) - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    lags = np.fft.irfft(spectrum * np.conj(spectrum), size)[: longest + 2]
    lags /= np.arange(len(centred), len(centred) - len(lags), -1) * centred.var()
    window = lags[shortest : longest + 1]
    peaks = [
        index
        for index in range(1, len(window) - 1)
        if window[index] >= window[index - 1] and window[index] > window[index + 1]
    ]
    if not peaks:
        raise ReplayError("No repetition found in the music stem")
    best = max(window[index] for index in peaks)
    first = next(index for index in peaks if window[index] >= PEAK_RATIO * best)
    return (shortest + _parabolic(window, first)) / rate


def offset(reference: FloatArray, candidate: FloatArray) -> tuple[float, float, float]:
    """Sub-frame lag of `reference` inside the longer `candidate`, its normalized peak and scale."""
    size = 1 << int(len(candidate) + len(reference)).bit_length()
    product = np.fft.rfft(candidate, size) * np.conj(np.fft.rfft(reference, size))
    raw = np.fft.irfft(product, size)[: len(candidate) - len(reference) + 1]
    running = np.concatenate([[0.0], np.cumsum(candidate**2)])
    energy = running[len(reference) :] - running[: len(running) - len(reference)]
    norm = np.sqrt(np.maximum(energy, 1e-12) * max(float(reference @ reference), 1e-12))
    ncc = raw / norm
    index = int(np.argmax(ncc))
    scale = float(raw[index]) / max(float(reference @ reference), 1e-12)
    return _parabolic(ncc, index) - (len(candidate) - len(reference)) // 2, float(ncc[index]), scale


def _quietest_repetition(voice: FloatArray, period: float, step: float, repetitions: int) -> int:
    span = max(1, round(period / step))
    energies = [voice[round(k * period / step) :][:span].mean() for k in range(repetitions)]
    return int(np.argmin(energies))


def align(music: Stereo, voice: FloatArray, period_seconds: float) -> Alignment:
    """Measure every repetition's offset per segment against the quietest one, in one frame."""
    period = period_seconds * music.rate
    repetitions = int(music.frames // period)
    segment = round(SEGMENT_SECONDS * music.rate)
    segments = -(-round(period) // segment)
    step = VOICE_FRAME_SECONDS * music.rate
    reference = _quietest_repetition(voice, period, step, repetitions)
    columns = [
        _measure(music, period, repetitions, start=s * segment, reference=reference,
                 length=min(segment, round(period) - s * segment))
        for s in range(segments)
    ]  # fmt: skip
    offsets = nearest_in_time(consensus(np.stack([lags for lags, _ in columns], axis=1)))
    scales = np.stack([scale for _, scale in columns], axis=1)
    # Against itself the reference also matches its own leaks, so its level is unknown: unused.
    scales[reference] = np.nan
    # Leaked speech in the reference inflates its energy alike for every repetition of a segment.
    typical = np.full(segments, np.nan)
    seen = ~np.isnan(scales).all(axis=0)
    typical[seen] = np.nanmedian(scales[:, seen], axis=0)
    ratios = scales / typical
    measured = ~np.isnan(ratios).all(axis=1)
    gains = np.full(repetitions, np.nan)
    gains[measured] = np.nanmedian(ratios[measured], axis=1)
    # The grid comes from a decimated estimate; the residual drift per repetition corrects it.
    rows = offsets[~np.isnan(offsets).all(axis=1)]
    drift = float(np.nanmedian(np.diff(np.nanmedian(rows, axis=1))))
    return Alignment(period, period + drift, segment, offsets, gains, reference)


def _measure(
    music: Stereo, period: float, repetitions: int, *, start: int, reference: int, length: int
) -> tuple[FloatArray, FloatArray]:
    """Offset and scale of every repetition's phase span against the reference repetition."""
    ref = music.read(round(reference * period) + start, length).mean(axis=1)
    lags, scales = np.full(repetitions, np.nan), np.full(repetitions, np.nan)
    for k in range(repetitions):
        got = music.read(round(k * period) + start - SEARCH_FRAMES, length + 2 * SEARCH_FRAMES)
        lag, ncc, scale = offset(ref, got.mean(axis=1))
        if ncc >= MIN_ALIGNMENT_NCC:
            lags[k], scales[k] = lag + round(k * period) - k * period, scale
    return lags, scales


def consensus(offsets: FloatArray) -> FloatArray:
    """Remove the reference's own per-segment steps (a join inside it) seen by every repetition."""
    usable = ~np.all(np.isnan(offsets), axis=1)
    if not usable.any():
        raise ReplayError("No repetition could be aligned")
    rows = offsets[usable]
    seen = ~np.isnan(rows).all(axis=0)
    centred = rows[:, seen] - np.nanmedian(rows, axis=1, keepdims=True)
    common = np.zeros(offsets.shape[1])
    common[seen] = np.nanmedian(centred, axis=0)
    common[seen] -= common[seen][0]
    corrected: FloatArray = offsets - common
    return corrected


def nearest_in_time(offsets: FloatArray) -> FloatArray:
    """Fill unmeasured segments from the closest measured one; rows then columns are time order.

    Offsets only drift slowly or step at a decoding join, so the nearest measurement is right
    except next to a join, where a misaligned copy then simply fails to agree with the others.
    """
    flat = offsets.ravel()
    known = np.flatnonzero(~np.isnan(flat))
    after = np.clip(np.searchsorted(known, np.arange(len(flat))), 1, len(known) - 1)
    before = after - 1
    if len(known) == 1:
        before = after = np.zeros(len(flat), int)
    closer = np.where(
        np.arange(len(flat)) - known[before] <= known[after] - np.arange(len(flat)), before, after
    )
    filled: FloatArray = flat[known[closer]].reshape(offsets.shape)
    return filled


def fractional(source: FloatArray, fraction: float, length: int) -> FloatArray:
    """`length` frames starting `fraction` past the centre tap, by a windowed sinc."""
    taps = np.arange(-FRACTIONAL_TAPS, FRACTIONAL_TAPS + 1)
    kernel = np.sinc(taps - fraction) * np.hanning(len(taps) + 2)[1:-1]
    kernel /= kernel.sum()
    return np.stack(
        [np.correlate(source[:, c], kernel, mode="valid")[:length] for c in range(CHANNELS)], 1
    )


def _extract(music: Stereo, position: float, length: int) -> FloatArray:
    whole = int(np.floor(position))
    raw = music.read(whole - FRACTIONAL_TAPS, length + 2 * FRACTIONAL_TAPS)
    return fractional(raw, position - whole, length)


def _choose(stack: FloatArray) -> tuple[int, float, list[int]]:
    """The candidate agreeing best with another, one minus that correlation, and the members.

    Leaked speech differs between repetitions while the loop does not, so two clean copies agree
    and a copy carrying speech or a foreign sound agrees with nobody. Members are the copies
    aligned with the best one; a misaligned copy correlates with nothing and is left out.
    """
    flat = stack.reshape(len(stack), -1)
    norms = np.linalg.norm(flat, axis=1)
    if norms.max() <= 1e-9:
        return 0, 0.0, [0]
    if len(stack) < 2:
        return 0, 1.0, [0]
    unit = flat / np.maximum(norms, 1e-9)[:, None]
    agreement = unit @ unit.T
    np.fill_diagonal(agreement, -1.0)
    best = agreement.max(axis=1)
    index = int(np.argmax(best))
    members = sorted({index, *np.flatnonzero(agreement[index] >= MEMBER_NCC).tolist()})
    return index, float(1 - best[index]), members


def _frames(signal: FloatArray) -> NDArray[np.complex128]:
    padded = np.pad(signal, (STFT_FRAME, STFT_FRAME + STFT_HOP))
    starts = np.arange(0, len(padded) - STFT_FRAME + 1, STFT_HOP)
    window = np.hanning(STFT_FRAME + 1)[:-1]
    return np.fft.rfft(padded[starts[:, None] + np.arange(STFT_FRAME)] * window, axis=-1)


def _overlap_add(spectrum: NDArray[np.complex128], length: int) -> FloatArray:
    window = np.hanning(STFT_FRAME + 1)[:-1]
    pieces = np.fft.irfft(spectrum, STFT_FRAME, axis=-1) * window
    total = np.zeros(STFT_HOP * (len(pieces) - 1) + STFT_FRAME)
    norm = np.zeros_like(total)
    for index, piece in enumerate(pieces):
        total[index * STFT_HOP : index * STFT_HOP + STFT_FRAME] += piece
        norm[index * STFT_HOP : index * STFT_HOP + STFT_FRAME] += window**2
    rebuilt: FloatArray = (total / np.maximum(norm, 1e-9))[STFT_FRAME : STFT_FRAME + length]
    return rebuilt


def frame_count(length: int) -> int:
    """STFT frames `_frames` yields for `length` samples; frame i is centred at i*hop - frame/2."""
    return (length + STFT_FRAME + STFT_HOP) // STFT_HOP + 1


def _usable(levels: FloatArray, threshold: float) -> tuple[NDArray[np.bool_], int]:
    """Per frame, the speech-free copies, or the quietest few where too few are speech-free;
    also how many frames needed that fallback."""
    free = levels <= threshold
    quietest = np.argsort(np.argsort(levels, axis=0, kind="stable"), axis=0) < MEDIAN_MEMBERS
    short = free.sum(axis=0) < MEDIAN_MEMBERS
    chosen: NDArray[np.bool_] = np.where(short[None, :], quietest, free)
    return chosen, int(short.sum())


def _bin_fallback(
    use: NDArray[np.bool_], levels: FloatArray, threshold: float, vocals: FloatArray
) -> NDArray[np.bool_]:
    """Per bin, the copies to take: in frames with too few speech-free copies, the
    third (at least `MEDIAN_MEMBERS`, at most `BIN_MEMBERS`) with the least voice magnitude in
    that bin; elsewhere the frame's choice."""
    short = (levels <= threshold).sum(axis=0) < MEDIAN_MEMBERS
    magnitude = np.mean(
        [np.abs(np.stack([_frames(copy[:, c]) for copy in vocals])) for c in range(CHANNELS)],
        axis=0,
    )
    take = min(max(len(vocals) // BIN_SHARE, MEDIAN_MEMBERS), BIN_MEMBERS)
    quiet = np.argsort(np.argsort(magnitude, axis=0, kind="stable"), axis=0) < take
    chosen: NDArray[np.bool_] = np.where(short[None, :, None], quiet, use[:, :, None])
    return chosen


def spectral_median(
    copies: FloatArray,
    levels: FloatArray | None = None,
    threshold: float = np.inf,
    vocals: FloatArray | None = None,
) -> tuple[FloatArray, int]:
    """Per time-frequency bin, the median of aligned copies (copies x frames x channels).

    `levels` (copies x STFT frames) is each copy's voice level; per frame only the copies at or
    below `threshold` count, or the quietest `MEDIAN_MEMBERS` when fewer qualify. Real and
    imaginary parts are taken separately, which is exact for identical copies; a sound present
    in fewer than half of the counted copies in a bin is dropped there. With `vocals` (the voice
    stem of each copy), those frames instead count per bin the copies least covered by voice.
    Returns the result and how many frames had too few speech-free copies.
    """
    if len(copies) < MEDIAN_MEMBERS:
        first: FloatArray = copies[0]
        return first, 0
    count = frame_count(copies.shape[1])
    voice = np.zeros((len(copies), count)) if levels is None else levels
    use, short = _usable(voice, threshold)
    mask = use[:, :, None]
    if short and vocals is not None:
        mask = _bin_fallback(use, voice, threshold, vocals)
    channels = []
    for channel in range(copies.shape[2]):
        spectra = np.stack([_frames(copy[:, channel]) for copy in copies])
        real = np.nanmedian(np.where(mask, spectra.real, np.nan), axis=0)
        imag = np.nanmedian(np.where(mask, spectra.imag, np.nan), axis=0)
        channels.append(_overlap_add(real + 1j * imag, copies.shape[1]))
    return np.stack(channels, axis=1), short


def _frame_levels(
    voice: FloatArray, positions: Sequence[float], length: int, step: float
) -> FloatArray:
    """Each copy's loudest voice level around every STFT frame of its window."""
    centres = np.arange(frame_count(length)) * STFT_HOP - STFT_FRAME // 2
    index = (np.asarray(positions)[:, None] + centres[None, :]) / step
    low = np.clip(np.floor(index).astype(int) - 1, 0, len(voice) - 1)
    loudest: FloatArray = np.stack(
        [voice[np.clip(low + d, 0, len(voice) - 1)] for d in range(3)]
    ).max(axis=0)
    return loudest


def _ramp(length: int, fade: int) -> FloatArray:
    weight = np.ones(length)
    rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(2 * fade) + 0.5) / (2 * fade))
    weight[: 2 * fade] = rise
    weight[-2 * fade :] = rise[::-1]
    return weight


def _options(
    music: Stereo, voice: FloatArray, alignment: Alignment, *, start: int, length: int
) -> list[tuple[float, int, float]]:
    """(voice level, repetition, position) of each repetition fully holding one block window."""
    # The window opens one crossfade early; its segment is that of the block it serves.
    segment = min(max(start, 0) // alignment.segment_frames, alignment.offsets.shape[1] - 1)
    step = VOICE_FRAME_SECONDS * music.rate
    options = []
    for k, lag in enumerate(alignment.offsets[:, segment]):
        position = k * alignment.grid_frames + start + lag
        if (
            np.isnan(lag)
            or np.isnan(alignment.gains[k])
            or position < FRACTIONAL_TAPS
            or position + length > music.frames
        ):
            continue
        level = voice[int(position / step) : int(np.ceil((position + length) / step))]
        options.append((float(level.mean()), k, float(position)))
    if not options:
        raise ReplayError(f"No repetition covers loop frame {start}")
    return sorted(options)


def rebuild(
    music: Stereo, voice: FloatArray, alignment: Alignment, voice_stem: Stereo | None = None
) -> tuple[FloatArray, list[Block]]:
    """One clean period: each block a speech-free spectral median of aligned copies, crossfaded."""
    frames = round(alignment.period_frames)
    threshold = VOICE_FREE_RATIO * float(np.median(voice[voice > 0])) if (voice > 0).any() else 0
    step = VOICE_FRAME_SECONDS * music.rate
    block = round(BLOCK_SECONDS * music.rate)
    fade = max(1, round(CROSSFADE_SECONDS * music.rate))
    out = np.zeros((frames, CHANNELS))
    weights = np.zeros(frames)
    chosen: list[Block] = []
    for start in range(0, frames, block):
        length = min(block, frames - start) + 2 * fade
        picked = _options(music, voice, alignment, start=start - fade, length=length)[:CANDIDATES]
        stack = np.stack([_extract(music, position, length) for _, _, position in picked])
        index, error, members = _choose(stack)
        gains = np.array([1 / alignment.gains[picked[member][1]] for member in members])
        levels = _frame_levels(voice, [picked[m][2] for m in members], length, step)
        vocals = (
            None
            if voice_stem is None
            else np.stack([_extract(voice_stem, picked[m][2], length) for m in members])
        )
        rebuilt, short = spectral_median(
            stack[members] * gains[:, None, None], levels, threshold, vocals
        )
        weight = _ramp(length, fade)
        where = (np.arange(length) + start - fade) % frames
        np.add.at(out, where, rebuilt * weight[:, None])
        np.add.at(weights, where, weight)
        chosen.append(Block(start, picked[index][1], error, picked[index][0], len(members), short))
    return out / weights[:, None], chosen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def song_start(loop: FloatArray, rate: int) -> int:
    """The frame where the song begins: the attack after the loop's quietest stretch, if silent.

    A loop without a silent stretch has no audible beginning, so it keeps frame 0.
    """
    mono = np.abs(loop.mean(axis=1))
    window, hop = round(SILENCE_WINDOW_SECONDS * rate), round(SILENCE_HOP_SECONDS * rate)
    wrapped = np.concatenate([mono, mono[:window]]) ** 2
    running = np.concatenate([[0.0], np.cumsum(wrapped)])
    starts = np.arange(0, len(mono), hop)
    levels = np.sqrt((running[starts + window] - running[starts]) / window)
    quietest = int(np.argmin(levels))
    typical = float(np.median(levels))
    if levels[quietest] > SILENCE_RATIO * typical:
        return 0
    centre = starts[quietest] + window // 2
    after = np.roll(mono, -centre) > ATTACK_RATIO * typical
    if not after.any():
        return 0
    return int((centre + int(np.argmax(after)) - round(LEAD_IN_SECONDS * rate)) % len(mono))


def true_peak(loop: FloatArray) -> float:
    """The loop's peak between samples too, as heard when it wraps onto itself (it is periodic)."""
    length = len(loop) * TRUE_PEAK_OVERSAMPLING
    peaks = [
        np.abs(np.fft.irfft(np.fft.rfft(channel), length)).max() * TRUE_PEAK_OVERSAMPLING
        for channel in loop.T
    ]
    return float(max(peaks))


def tile(
    loop: FloatArray, *, rate: int, out: Path, repeats: int, peak_db: float | None = None
) -> tuple[int, float]:
    """Write `repeats` copies of the loop rotated to its song start, scaled so its true peak is
    `peak_db` dBFS when given; return that start frame and the applied gain in dB."""
    start = song_start(loop, rate)
    peak = true_peak(loop)
    gain_db = 0.0 if peak_db is None or peak == 0 else peak_db - 20 * np.log10(peak / FULL_SCALE)
    data = float_pcm(np.roll(loop, -start, axis=0) * 10 ** (gain_db / 20))
    with written(out, rate, floating=True) as sink:
        for _ in range(repeats):
            sink.writeframes(data)
    return start, float(gain_db)


def retile(
    master: Path, *, out: Path, repeats: int, peak_db: float | None = None
) -> dict[str, object]:
    """Re-tile a saved one-period loop so it starts at its song start."""
    with opened(master) as source:
        loop, rate = source.read(0, source.frames), source.rate
    start, gain_db = tile(loop, rate=rate, out=out, repeats=repeats, peak_db=peak_db)
    return {
        "master": str(master),
        "output": str(out),
        "startSeconds": start / rate,
        "durationSeconds": len(loop) * repeats / rate,
        "gainDB": gain_db,
    }


def extract(
    music_path: Path, voice_path: Path, *, out: Path, repeats: int, peak_db: float | None = None
) -> dict[str, object]:
    """Write `repeats` seamless periods of the rebuilt loop and return the receipt."""
    with opened(music_path) as music, opened(voice_path) as voice_stem:
        low, voice = analysis(music, voice_stem)
        period = coarse_period(low, music.rate / ANALYSIS_DECIMATION)
        alignment = align(music, voice, period)
        loop, blocks = rebuild(music, voice, alignment, voice_stem)
        rate = music.rate
    start, gain_db = tile(loop, rate=rate, out=out, repeats=repeats, peak_db=peak_db)
    unclean = [b.start / rate for b in blocks if b.error > UNCLEAN_ERROR]
    return {
        "version": 1,
        "music": str(music_path),
        "musicSHA256": sha256(music_path),
        "voice": str(voice_path),
        "voiceSHA256": sha256(voice_path),
        "output": str(out),
        "outputSHA256": sha256(out),
        "sampleRate": rate,
        "periodSeconds": len(loop) / rate,
        "periodFrames": len(loop),
        "songStartFrame": start,
        "gainDB": gain_db,
        "gridFrames": alignment.grid_frames,
        "measuredPeriodFrames": alignment.period_frames,
        "repeats": repeats,
        "durationSeconds": len(loop) * repeats / rate,
        "referenceRepetition": alignment.reference,
        "alignedRepetitions": int((~np.isnan(alignment.offsets)).any(axis=1).sum()),
        "blocks": len(blocks),
        "medianBlockError": float(np.median([b.error for b in blocks])),
        "maxBlockError": float(max(b.error for b in blocks)),
        "uncleanBlockSeconds": unclean,
        "medianMembers": float(np.median([b.members for b in blocks])),
        "singleCopyBlockSeconds": [b.start / rate for b in blocks if b.members < MEDIAN_MEMBERS],
        "fewSpeechFreeBlockSeconds": [b.start / rate for b in blocks if b.short_frames],
        "sourceRepetitions": sorted({b.repetition for b in blocks}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    subtract = commands.add_parser("residual", help="join mix minus voice of separated parts")
    subtract.add_argument("out", type=Path)
    subtract.add_argument("--part", nargs=2, type=Path, action="append", required=True)
    subtract.add_argument("--keep-seconds", type=float, required=True)
    build = commands.add_parser("extract", help="rebuild the loop and tile it")
    build.add_argument("music", type=Path)
    build.add_argument("voice", type=Path)
    build.add_argument("out", type=Path)
    build.add_argument("receipt", type=Path)
    build.add_argument("--repeats", type=int, default=1)
    build.add_argument("--peak-db", type=float, default=PEAK_CEILING_DB, help="true-peak ceiling")
    again = commands.add_parser("retile", help="re-tile a one-period loop from its song start")
    again.add_argument("master", type=Path)
    again.add_argument("out", type=Path)
    again.add_argument("--repeats", type=int, default=1)
    again.add_argument("--peak-db", type=float, default=PEAK_CEILING_DB, help="true-peak ceiling")
    args = parser.parse_args(argv)
    if args.command == "residual":
        frames = residual(
            [(mix, voice) for mix, voice in args.part], keep_seconds=args.keep_seconds, out=args.out
        )
        print(json.dumps({"output": str(args.out), "frames": frames}))
        return 0
    if args.repeats < 1:
        raise ReplayError("--repeats must be at least 1")
    if args.command == "retile":
        print(
            json.dumps(
                retile(args.master, out=args.out, repeats=args.repeats, peak_db=args.peak_db)
            )
        )
        return 0
    receipt = extract(
        args.music, args.voice, out=args.out, repeats=args.repeats, peak_db=args.peak_db
    )
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    summary = ("periodSeconds", "durationSeconds", "maxBlockError", "uncleanBlockSeconds")
    print(json.dumps({key: receipt[key] for key in summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
