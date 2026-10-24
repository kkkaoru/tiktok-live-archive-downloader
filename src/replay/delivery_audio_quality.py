"""Fail-closed, phased final AAC mapping and independent speech/music measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import wave
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from replay.audio_clock import AudioFrameClock
from replay.bgm_separation import (
    ADAPTIVE_FRAME_SAMPLES,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    MODEL_SHA256,
    SeparationBackend,
    SeparationPolicy,
    SpeechQuality,
    _load_backend,
    measure_speech_quality,
    signal_level_dbfs,
    verify_music_removal,
)
from replay.caption_measured_job import ROOT, _json
from replay.caption_native_layout import NativeCaptionLayout
from replay.delivery_audio_mapping import cut_intervals, integer, placement, spans_from_json
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import _digest

MAX_JSON = 128 * 1024**2
WINDOW = 60 * MODEL_SAMPLE_RATE
MINIMUM_CORRELATION = 0.995
MAX_GAIN_DB = 40.0  # bound on a recorded static voice gain
Runner = Callable[[Sequence[str], Path], None]


class Fingerprint(TypedDict):
    path: str
    sha256: str
    bytes: int


def fingerprint(path: Path) -> Fingerprint:
    if not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024**3:
        raise ValueError("Input missing or exceeds byte budget")
    return {"path": str(path.resolve()), "sha256": file_digest(path), "bytes": path.stat().st_size}


def read_json(path: Path) -> dict[str, object]:
    if not 0 < path.stat().st_size <= MAX_JSON:
        raise ValueError("JSON exceeds byte budget")
    return record(_json(path.read_bytes()))


def reference(value: object, pins: dict[str, Fingerprint]) -> Path:
    row = record(value)
    name = row.get("path")
    if not isinstance(name, str) or not Path(name).is_absolute():
        raise ValueError("Input references require absolute paths")
    path = Path(name).resolve(strict=True)
    pin = fingerprint(path)
    if pin["sha256"] != _digest(row.get("sha256")):
        raise ValueError("Pinned input changed")
    if "bytes" in row and integer(row["bytes"], minimum=1) != pin["bytes"]:
        raise ValueError("Pinned input byte count changed")
    pins[str(path)] = pin
    return path


def scratch(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT / "edits/work"):
        raise ValueError("Scratch/output must remain under checkout edits/work")
    return resolved


def wav_info(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError("Expected uncompressed PCM16 WAV")
        rate, channels, samples = audio.getframerate(), audio.getnchannels(), audio.getnframes()
    if rate != MODEL_SAMPLE_RATE or channels != 2 or not 0 < samples <= rate * 21600:
        raise ValueError("Expected bounded 44100-Hz stereo PCM; no implicit resampling")
    return rate, channels, samples


def run_tool(arguments: Sequence[str], destination: Path) -> None:
    """Tools write only explicit files; ffmpeg/ffprobe have no model/cache runtime."""
    environment = os.environ.copy()
    environment.update({key: str(destination.parent) for key in ("TMPDIR", "TMP", "TEMP")})
    environment.pop("FFREPORT", None)
    with destination.open("xb") as output, destination.with_suffix(".stderr").open("xb") as error:
        subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=error,
            env=environment,
            timeout=7200,
            check=True,
        )


def decode_aac(
    movie: Path, directory: Path, *, ffmpeg: Path, ffprobe: Path, runner: Runner = run_tool
) -> tuple[Path, dict[str, object]]:
    probe = directory / "aac-frames.json"
    runner(
        [
            str(ffprobe),
            "-protocol_whitelist",
            "file",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,time_base,start_pts,duration_ts:frame=pts,nb_samples",
            "-of",
            "json",
            str(movie),
        ],
        probe,
    )
    data = read_json(probe)
    clock = AudioFrameClock.from_probe(data)
    streams = data.get("streams")
    if not isinstance(streams, list):
        raise ValueError("Missing AAC stream")
    stream = record(streams[0])
    if stream.get("codec_name") != "aac" or stream.get("channels") != 2:
        raise ValueError("Final media must have exactly one stereo AAC stream")
    packets = directory / "aac-packets.json"
    runner(
        [
            str(ffprobe),
            "-protocol_whitelist",
            "file",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_packets",
            "-show_entries",
            "packet=pts,duration:packet_side_data=side_data_type,skip_samples,discard_padding",
            "-of",
            "json",
            str(movie),
        ],
        packets,
    )
    packet_rows = read_json(packets).get("packets")
    if not isinstance(packet_rows, list) or not 1 <= len(packet_rows) <= 1_000_000:
        raise ValueError("Invalid AAC packets")
    first, last = record(packet_rows[0]), record(packet_rows[-1])
    decoded = directory / "final-decoded.wav"
    runner(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-protocol_whitelist",
            "file",
            "-i",
            str(movie),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(decoded),
        ],
        directory / "decode.stdout",
    )
    rate, _, samples = wav_info(decoded)
    contiguous = all(
        (right - left) * clock.time_base == Fraction(count, clock.sample_rate)
        for left, right, count in zip(clock.pts, clock.pts[1:], clock.samples, strict=False)
    )
    end = clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
    first_pts, last_pts = first.get("pts"), last.get("pts")
    if type(first_pts) is not int or type(last_pts) is not int:
        raise ValueError("Missing AAC packet PTS")
    first_side, last_side = first.get("side_data_list", []), last.get("side_data_list", [])
    if not isinstance(first_side, list) or not isinstance(last_side, list):
        raise ValueError("Invalid AAC side data")
    skip = sum(integer(record(item).get("skip_samples", 0)) for item in first_side)
    discard = sum(integer(record(item).get("discard_padding", 0)) for item in last_side)
    packet_start = first_pts * clock.time_base + Fraction(skip, clock.sample_rate)
    raw_packet_end = (last_pts + integer(last.get("duration"), minimum=1)) * clock.time_base
    stream_end = (
        integer(stream.get("start_pts")) + integer(stream.get("duration_ts"), minimum=1)
    ) * clock.time_base
    # Some muxers clip packet duration; others retain a full coded packet. Only
    # decoder AND stream evidence can establish which discard representation applies.
    coded_discard = (
        discard > 0
        and raw_packet_end - end == Fraction(discard, clock.sample_rate)
        and stream_end == end
    )
    packet_end = end if coded_discard else raw_packet_end
    return decoded, {
        "sampleRate": rate,
        "decodedSamples": samples,
        "firstDecodedPTS": str(clock.pts[0] * clock.time_base),
        "decodedEnd": str(end),
        "decodedFrameSamples": clock.sample_ends[-1],
        "lastDecodedFrameSamples": clock.samples[-1],
        "contiguousDecodedClock": contiguous,
        "firstPacket": first,
        "lastPacket": last,
        "primingSamples": skip,
        "discardPaddingSamples": discard,
        "discardPaddingAppliedAgain": False,
        "rawPacketEnd": str(raw_packet_end),
        "packetExtentPolicy": (
            "coded-minus-decoder-and-stream-confirmed-discard"
            if coded_discard
            else "packet-duration-presentation"
        ),
        "firstPacketDecodedStart": str(packet_start),
        "lastPacketDecodedEnd": str(packet_end),
        "packetEndMinusDecodedEnd": str(packet_end - end),
        "packetFrameExtentMatches": (
            packet_start == clock.pts[0] * clock.time_base and packet_end == end
        ),
        "streamStartPTS": stream.get("start_pts"),
        "streamDurationTS": stream.get("duration_ts"),
        "streamPresentationExtentMatches": (
            integer(stream.get("start_pts")) * clock.time_base == packet_start
            and stream_end == packet_end
        ),
        "timeBase": str(clock.time_base),
        "decodeMatchesFrameClock": rate == clock.sample_rate and samples == clock.sample_ends[-1],
        "primingPolicy": (
            "Decoder-applied packet skip/discard; no manual padding, trim or offset search"
        ),
        "audioPCMIdentical": False,
    }


def reuse_aac(
    value: object, pins: dict[str, Fingerprint], *, config: Path
) -> tuple[Path, dict[str, object], list[Fingerprint]]:
    """Reuse only immutable decode artifacts bound to the same inputs/tools/code."""
    required = {name: pin for name, pin in pins.items() if name != str(config)}
    receipt = read_json(reference(value, pins))
    if receipt.get("version") != 1 or receipt.get("kind") != "delivery-audio-quality":
        raise ValueError("Invalid AAC cache receipt")
    prior = receipt.get("inputs")
    if not isinstance(prior, list):
        raise ValueError("AAC cache inputs missing")
    bound = {str(record(item).get("path")): record(item) for item in prior}
    if any(bound.get(name) != pin for name, pin in required.items()):
        raise ValueError("AAC cache input/tool/code fingerprints differ")
    raw_artifacts = receipt.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 3:
        raise ValueError("AAC cache artifacts missing")
    artifacts: list[Fingerprint] = []
    for raw in raw_artifacts:
        path = scratch(reference(raw, pins))
        artifacts.append(pins[str(path)])
    decoded = Path(artifacts[0]["path"])
    if decoded.name != "final-decoded.wav":
        raise ValueError("AAC cache decoded artifact missing")
    facts = record(record(receipt.get("measurements")).get("aac"))
    if wav_info(decoded)[2] != facts.get("decodedSamples"):
        raise ValueError("AAC cache sample count differs")
    return decoded, facts, artifacts


def read_samples(reader: wave.Wave_read, start: int, count: int) -> NDArray[np.float32]:
    reader.setpos(start)
    raw = reader.readframes(count)
    if len(raw) != count * 4:
        raise ValueError("Truncated PCM")
    return np.asarray(np.frombuffer(raw, dtype="<i2").reshape(-1, 2).T / 32768.0, dtype=np.float32)


def selected_window(
    reader: wave.Wave_read, spans: tuple[tuple[int, int], ...], start: int, count: int
) -> NDArray[np.float32]:
    output = np.empty((2, count), dtype=np.float32)
    cursor = written = 0
    for low, high in spans:
        first, last = max(start, cursor), min(start + count, cursor + high - low)
        if first < last:
            size = last - first
            output[:, written : written + size] = read_samples(reader, low + first - cursor, size)
            written += size
        cursor += high - low
        if cursor >= start + count:
            break
    if written != count:
        raise ValueError("Reference window exceeds selected samples")
    return output


def protected_mask(
    spans: tuple[tuple[int, int], ...],
    speech: tuple[tuple[int, int], ...],
    *,
    start: int,
    count: int,
) -> NDArray[np.bool_]:
    mask = np.zeros(count, dtype=np.bool_)
    cursor = 0
    speech_ends = tuple(last for _, last in speech)
    for low, high in spans:
        window_low = max(low, low + start - cursor)
        window_high = min(high, low + start + count - cursor)
        index = bisect_right(speech_ends, window_low)
        while window_low < window_high and index < len(speech):
            first, last = speech[index]
            if first >= window_high:
                break
            left, right = max(window_low, first), min(window_high, last)
            mask[cursor + left - low - start : cursor + right - low - start] = True
            index += 1
        cursor += high - low
        if cursor >= start + count:
            break
    return mask


def speech_moments(
    original: NDArray[np.float32], processed: NDArray[np.float32], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sufficient statistics of the existing speech metric on 10ms-aligned windows."""
    size = ADAPTIVE_FRAME_SAMPLES
    padding = (-original.shape[1]) % size
    shape = (2, -1, size)
    before = np.pad(original, ((0, 0), (0, padding))).reshape(shape)
    after = np.pad(processed, ((0, 0), (0, padding))).reshape(shape)
    chosen = np.pad(mask, (0, padding)).reshape(-1, size).any(axis=1)
    left = np.einsum("cft,cft->f", before, before, dtype=np.float64)[chosen]
    right = np.einsum("cft,cft->f", after, after, dtype=np.float64)[chosen]
    deltas = 10 * np.log10((right + 1e-18) / (left + 1e-18))
    x, y = (
        original[:, mask].astype(np.float64).ravel(),
        processed[:, mask].astype(np.float64).ravel(),
    )
    moments = np.array([x.sum(), y.sum(), x @ x, y @ y, x @ y, x.size], dtype=np.float64)
    return deltas, moments


def combined_speech(
    deltas: list[NDArray[np.float64]], moments: NDArray[np.float64]
) -> SpeechQuality | None:
    x, y, xx, yy, xy, count = moments
    if count == 0:
        return None
    variance_x, variance_y = xx - x * x / count, yy - y * y / count
    # Constant/silent protected references cannot establish speech preservation.
    correlation = (
        (xy - x * y / count) / np.sqrt(variance_x * variance_y)
        if variance_x > 0 and variance_y > 0
        else 0.0
    )
    values = np.concatenate(deltas)
    return SpeechQuality(
        float(np.percentile(values, 50)), float(np.percentile(values, 5)), float(correlation)
    )


def quality_windows(
    *,
    decoded: Path,
    first_pass: Path,
    model_input: Path,
    spans: tuple[tuple[int, int], ...],
    speech: tuple[tuple[int, int], ...],
    backend: SeparationBackend,
    directory: Path,
    cache_key: str,
    gain_db: float = 0.0,
    stem_scale: float = 1.0,
) -> dict[str, object]:
    """One persistent backend, bounded 60s windows, independent first-pass reference.

    Every speech-bearing window and the exact full-file statistics must pass.
    Empty-mask windows still receive instrumental inspection. Caller must validate
    packet presentation extent before excluding any decoded terminal codec padding.
    ``gain_db`` is the static voice gain the delivery applied (its pinned loudness record):
    the first-pass voice is compared scaled by it and the instrumental level before removal is
    offset by it, so a level change the delivery made on purpose is not speech or music loss.
    The thresholds are unchanged. ``stem_scale`` is the static output gain separation wrote
    the first-pass stem with (its pinned ``result.json``); the instrumental estimate subtracts
    the stem scaled back to the model input's level, so no voice leaks into it.
    """
    scale = np.float32(10 ** (gain_db / 20))
    total = sum(high - low for low, high in spans)
    if WINDOW % ADAPTIVE_FRAME_SAMPLES:
        raise ValueError("Quality windows must align to the existing 10ms speech metric")
    if wav_info(decoded)[2] < total:
        raise ValueError("AAC sample extent differs; refusing trim/pad for quality comparison")
    policy = SeparationPolicy()
    results: list[dict[str, object]] = []
    speech_passed = music_passed = True
    protected = 0
    deltas: list[NDArray[np.float64]] = []
    moments = np.zeros(6, dtype=np.float64)
    with (
        wave.open(str(decoded), "rb") as final,
        wave.open(str(first_pass), "rb") as voice,
        wave.open(str(model_input), "rb") as mix,
    ):
        for start in range(0, total, WINDOW):
            count = min(WINDOW, total - start)
            unscaled = selected_window(voice, spans, start, count)
            baseline = selected_window(mix, spans, start, count)
            original = unscaled * scale
            candidate = read_samples(final, start, count)
            predicted = backend.extract_vocals(candidate)
            if predicted.shape != candidate.shape or not np.isfinite(predicted).all():
                raise ValueError("Model returned invalid vocals")
            mask = protected_mask(spans, speech, start=start, count=count)
            protected += int(mask.sum())
            stats = measure_speech_quality(original, predicted, mask) if mask.any() else None
            delta, moment = speech_moments(original, predicted, mask)
            deltas.append(delta)
            moments += moment
            speech_ok = stats is None or (
                stats.correlation >= MINIMUM_CORRELATION
                and stats.level_delta_db_p50 >= -policy.maximum_speech_loss_db
                and stats.level_delta_db_p05 >= -policy.maximum_speech_loss_db
            )
            before, after = (
                signal_level_dbfs(baseline - unscaled / np.float32(stem_scale)) + gain_db,
                signal_level_dbfs(candidate - predicted),
            )
            gate = verify_music_removal(
                before, after, 0.0, stats.level_delta_db_p05 if stats is not None else 0.0, policy
            )
            speech_passed &= speech_ok
            music_passed &= gate.passed
            row: dict[str, object] = {
                "startSample": start,
                "endSample": start + count,
                "protectedSamples": int(mask.sum()),
                "speech": asdict(stats) if stats else None,
                "speechGatePassed": speech_ok if stats else None,
                "beforeInstrumentalDBFS": before,
                "afterInstrumentalDBFS": after,
                "music": asdict(gate),
                "cacheFingerprint": cache_key,
            }
            NativeCaptionLayout._save(directory / f"window-{start:010d}.json", row)
            results.append(row)
    full_speech = combined_speech(deltas, moments)
    global_passed = full_speech is not None and (
        full_speech.correlation >= MINIMUM_CORRELATION
        and full_speech.level_delta_db_p50 >= -policy.maximum_speech_loss_db
        and full_speech.level_delta_db_p05 >= -policy.maximum_speech_loss_db
    )
    return {
        "speech": asdict(full_speech) if full_speech is not None else None,
        "candidateSpeechComparedToOriginalFirstPass": (
            asdict(full_speech) if full_speech is not None else None
        ),
        "policy": asdict(policy),
        "voiceGainDb": gain_db,
        "stemOutputScale": stem_scale,
        "minimumSpeechCorrelation": MINIMUM_CORRELATION,
        "speechAggregation": (
            "Exact full-file 10ms percentiles/moments plus every protected 60-second window"
        ),
        "protectedSamples": protected,
        "windows": results,
        "speechGatePassed": speech_passed and global_passed and protected > 0,
        "musicGatePassed": music_passed,
    }


def stem_output_scale(stem: Path, pins: dict[str, Fingerprint]) -> float:
    """The static output gain separation wrote ``stem`` with, from the ``result.json`` beside it
    (pinned here, and required to pin this stem). Stems with no receipt or no ``outputScale``
    (written before the gain existed) are at unit scale."""
    result_path = stem.with_name("result.json")
    if not result_path.is_file():
        return 1.0
    pin = fingerprint(result_path)
    pins[str(result_path)] = pin
    result = read_json(result_path)
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not any(
        record(item).get("path") == str(stem)
        and record(item).get("sha256") == pins[str(stem)]["sha256"]
        for item in outputs
    ):
        raise ValueError("The separation receipt beside the first-pass voice does not pin it")
    if "outputScale" not in result:
        return 1.0
    scale = result["outputScale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1:
        raise ValueError("The separation receipt's outputScale is not a valid gain")
    return float(scale)


def voice_gain(data: dict[str, object], pins: dict[str, Fingerprint]) -> float:
    """The static voice gain (dB) the delivery applied, from its pinned loudness record
    (``caption_compose``'s ``tempo.m4a.loudness.json``); 0 when the delivery has none."""
    if "voiceLoudness" not in data:
        return 0.0
    gain = read_json(reference(data["voiceLoudness"], pins)).get("gainDb")
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not abs(gain) <= MAX_GAIN_DB:
        raise ValueError("The voice loudness record holds no valid gainDb")
    return float(gain)


def load_model(data: dict[str, object], pins: dict[str, Fingerprint]) -> SeparationBackend:
    weights = reference(data.get("modelWeights"), pins)
    config = reference(data.get("modelConfig"), pins)
    cache_value = data.get("hfCache")
    if not isinstance(cache_value, str):
        raise ValueError("Explicit repository-local hfCache required")
    cache = Path(cache_value).resolve(strict=True)
    if (
        not cache.is_relative_to(ROOT / "edits")
        or not weights.is_relative_to(ROOT / "edits/models")
        or not config.is_relative_to(ROOT / "edits/models")
    ):
        raise ValueError("Model/cache must be repository-local")
    if pins[str(weights)]["sha256"] != MODEL_SHA256:
        raise ValueError("Model substitution forbidden")
    os.environ.update(HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    # Verify the same cache resolver used by the existing backend, before loading it.
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(MODEL_NAME, "model.safetensors", revision=MODEL_REVISION)
    if not isinstance(cached, str) or Path(cached).resolve() != weights:
        raise ValueError("Existing backend cache does not resolve the explicitly pinned model")
    if Path(cached).with_name("config.json").resolve(strict=True) != config:
        raise ValueError("Existing backend config does not resolve the explicitly pinned config")
    return _load_backend(MODEL_NAME, "mps")


def analyze(
    config: Path,
    output: Path,
    *,
    sha256: str,
    phase: str,
    runner: Runner = run_tool,
    model_loader: Callable[
        [dict[str, object], dict[str, Fingerprint]], SeparationBackend
    ] = load_model,
) -> Path:
    if phase not in ("mapping", "quality"):
        raise ValueError("Unknown phase")
    output = scratch(output)
    if output.exists():
        raise FileExistsError(output)
    pins: dict[str, Fingerprint] = {}
    config = reference({"path": str(config.resolve()), "sha256": sha256}, pins)
    data = read_json(config)
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Expected version 1 configuration")
    paths = {
        key: reference(data.get(key), pins)
        for key in (
            "renderConfig",
            "finalMovie",
            "selection",
            "pcmAudit",
            "firstPassVoice",
            "modelInput",
            "originalClock",
            "protectedSpeech",
        )
    }
    render, selection, audit = (
        read_json(paths[key]) for key in ("renderConfig", "selection", "pcmAudit")
    )
    if type(selection.get("version")) is not int or selection["version"] != 1:
        raise ValueError("Expected version 1 selection receipt")
    if type(audit.get("version")) is not int or audit["version"] != 2:
        raise ValueError("Expected version 2 independent PCM audit")
    voice_config = record(render.get("audio"))
    if (
        Path(str(voice_config.get("source"))).resolve() != paths["firstPassVoice"]
        or Path(str(voice_config.get("clock"))).resolve() != paths["originalClock"]
    ):
        raise ValueError("Render voice/clock does not match explicit references")
    cuts_name = render.get("cuts")
    if not isinstance(cuts_name, str):
        raise ValueError("Missing render cuts")
    cuts_path = reference(data.get("cuts"), pins)
    if Path(cuts_name).resolve() != cuts_path:
        raise ValueError("Render cuts differ from explicit cuts reference")
    intervals = cut_intervals(read_json(cuts_path), integer(render.get("rate"), minimum=1))
    selected = reference(
        {"path": selection.get("output"), "sha256": selection.get("outputSHA256")}, pins
    )
    selection_pins, audit_pins = record(selection.get("inputs")), record(audit.get("inputs"))
    for key in ("firstPassVoice", "originalClock"):
        name = str(paths[key])
        if selection_pins.get(name) != pins[name]["sha256"]:
            raise ValueError("Selection does not bind source/clock")
    for path in (paths["selection"], paths["firstPassVoice"], paths["originalClock"], selected):
        if audit_pins.get(str(path)) != pins[str(path)]["sha256"]:
            raise ValueError("PCM audit does not bind current selection/source/clock/output")
    if (
        audit.get("kind") != "independent-pcm-selection-clock-audit"
        or audit.get("selectedPCMBytesEqual") is not True
        or type(audit.get("mismatchedBytes")) is not int
        or audit.get("mismatchedBytes") != 0
    ):
        raise ValueError("Pinned independent PCM audit did not pass byte comparison")
    spans = spans_from_json(selection.get("spans"))
    if spans_from_json(audit.get("spans")) != spans:
        raise ValueError("Audit and selection spans differ")
    source_info = wav_info(paths["firstPassVoice"])
    if (
        wav_info(paths["modelInput"]) != source_info
        or paths["modelInput"] == paths["firstPassVoice"]
    ):
        raise ValueError("Independent model-input and first-pass voice clocks required")
    if selected == paths["firstPassVoice"] or paths["finalMovie"] in (
        selected,
        paths["firstPassVoice"],
    ):
        raise ValueError("Self comparison forbidden")
    selected_info = wav_info(selected)
    if selected_info[2] != sum(high - low for low, high in spans):
        raise ValueError("Selected WAV count mismatch")
    if integer(audit.get("comparedBytes"), minimum=1) != selected_info[2] * 4:
        raise ValueError("Pinned PCM audit did not compare all selected bytes")
    clock = AudioFrameClock.from_probe(read_json(paths["originalClock"]))
    source_count_difference = source_info[2] - Fraction(
        clock.sample_ends[-1] * source_info[0], clock.sample_rate
    )
    mapping = placement(clock, spans, rate=source_info[0], cuts=intervals)
    mapping["sourceSamplesMinusClockProjection"] = str(source_count_difference)
    if source_count_difference:
        mapping["nativeTimesPreserved"] = False
    protection = read_json(paths["protectedSpeech"])
    if (
        protection.get("coordinateSpace") != "source-pcm-samples"
        or protection.get("sampleRate") != source_info[0]
    ):
        raise ValueError("Protected speech needs explicit source PCM sample coordinates")
    if reference(protection.get("source"), pins) != paths["firstPassVoice"]:
        raise ValueError("Protected speech source mismatch")
    reference(protection.get("asr"), pins)
    speech = spans_from_json(protection.get("intervals"))
    if speech[-1][1] > source_info[2]:
        raise ValueError("Protected speech exceeds source")
    tools = {key: reference(data.get(key), pins) for key in ("ffmpeg", "ffprobe")}
    for name in (
        "delivery_audio_quality.py",
        "delivery_audio_mapping.py",
        "bgm_separation.py",
        "audio_clock.py",
    ):
        code = Path(__file__).with_name(name).resolve()
        pins[str(code)] = fingerprint(code)
    output.mkdir(mode=0o700, parents=True)
    if "aacEvidence" in data:
        decoded, aac, artifacts = reuse_aac(data["aacEvidence"], pins, config=config)
    else:
        decoded, aac = decode_aac(
            paths["finalMovie"],
            output,
            ffmpeg=tools["ffmpeg"],
            ffprobe=tools["ffprobe"],
            runner=runner,
        )
        artifacts = [
            fingerprint(path)
            for path in (decoded, output / "aac-frames.json", output / "aac-packets.json")
        ]
    decoded_samples = integer(aac["decodedSamples"], minimum=1)
    extra_samples = decoded_samples - selected_info[2]
    aac_ok = (
        aac["decodeMatchesFrameClock"] is True
        and aac["contiguousDecodedClock"] is True
        and aac["firstDecodedPTS"] == "0"
        and aac["streamPresentationExtentMatches"] is True
        and aac["firstPacketDecodedStart"] == "0"
        and aac["lastPacketDecodedEnd"] == str(Fraction(selected_info[2], selected_info[0]))
        and 0 <= extra_samples < integer(aac["lastDecodedFrameSamples"], minimum=1)
    )
    aac["selectedSamples"] = selected_info[2]
    aac["decodedSamplesBeyondSelected"] = extra_samples
    aac["terminalCodecPaddingProven"] = aac_ok and extra_samples > 0
    aac["comparisonExtentPolicy"] = (
        "Use packet-declared presentation extent only when it exactly equals selected PCM; "
        "retain decoded tail unchanged. Native mapping is a separate mandatory check."
    )
    checks: dict[str, object] = {
        "selectedPCMReceiptBound": True,
        "nativeTimesPreserved": mapping["nativeTimesPreserved"],
        "aacSampleTimeMapping": aac_ok,
        "speechGatePassed": None,
        "musicGatePassed": None,
        "inputsStable": False,
    }
    measurements: dict[str, object] = {"mapping": mapping, "aac": aac}
    limitations = [
        "Lossy AAC is not PCM-identical to the selected voice.",
        "Protected speech/first-pass provenance is explicitly supplied, not ASR re-reviewed.",
    ]
    gain_db = voice_gain(data, pins)
    measurements["voiceGainDb"] = gain_db
    stem_scale = stem_output_scale(paths["firstPassVoice"], pins)
    measurements["stemOutputScale"] = stem_scale
    if phase == "quality" and aac_ok:
        backend = model_loader(data, pins)
        # Cache identity includes every input, tool, model and implementation byte binding.
        key = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
        quality = quality_windows(
            decoded=decoded,
            first_pass=paths["firstPassVoice"],
            model_input=paths["modelInput"],
            spans=spans,
            speech=speech,
            backend=backend,
            directory=output,
            cache_key=key,
            gain_db=gain_db,
            stem_scale=stem_scale,
        )
        measurements["quality"] = quality
        checks.update({key: quality[key] for key in ("speechGatePassed", "musicGatePassed")})
    else:
        limitations.append(
            "Speech/music quality not evaluated"
            + (": AAC mapping failed." if phase == "quality" else ".")
        )
    for pin in pins.values():
        if fingerprint(Path(pin["path"])) != pin:
            raise ValueError("Input changed during analysis; no accepted receipt published")
    for artifact in artifacts:
        if fingerprint(Path(artifact["path"])) != artifact:
            raise ValueError("AAC artifact changed during analysis")
    checks["inputsStable"] = True
    receipt = output / "receipt.json"
    NativeCaptionLayout._save(
        receipt,
        {
            "version": 1,
            "kind": "delivery-audio-quality",
            "phase": phase,
            "inputs": list(pins.values()),
            "artifacts": artifacts,
            "checks": checks,
            "measurements": measurements,
            "accepted": all(value is True for value in checks.values()),
            "limitations": limitations,
        },
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--phase", choices=("mapping", "quality"), default="mapping")
    args = parser.parse_args(argv)
    receipt = analyze(args.config, args.output, sha256=args.sha256, phase=args.phase)
    print(receipt)
    return 0 if read_json(receipt)["accepted"] is True else 1


if __name__ == "__main__":
    sys.exit(main())
