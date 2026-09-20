"""Strict boundaries for local Executor and MLX caption measurements."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from math import isfinite

from replay.caption_masks import TextDetection, TextRectangle
from replay.caption_timing import SpokenToken

MAX_MLX_DURATION = 86400
MAX_MLX_SEGMENTS = 50000
MAX_MLX_WORDS = 100000
MAX_MLX_TEXT_BYTES = 4000000


@dataclass(frozen=True, slots=True)
class FrameObservation:
    requested_time: float
    actual_time: float
    width: int
    height: int
    detections: tuple[TextDetection, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MLXWordGroup:
    """Native pieces sharing an onset; a point is NOT a positive speech interval."""

    text: str
    start: float
    end: float
    source_segments: tuple[int, ...]
    piece_count: int
    minimum_probability: float

    def __post_init__(self) -> None:
        if not self.text.strip() or "\0" in self.text or len(self.text) > 2048:
            raise ValueError("Invalid MLX word text")
        if not all(isfinite(value) for value in (self.start, self.end, self.minimum_probability)):
            raise ValueError("MLX word values must be finite")
        if not 0 <= self.start <= self.end or not 0 <= self.minimum_probability <= 1:
            raise ValueError("Invalid MLX word clock or probability")
        if self.piece_count < 1 or not self.source_segments or min(self.source_segments) < 0:
            raise ValueError("MLX word group lacks provenance")

    def spoken_token(self) -> SpokenToken:
        """Fail on unresolved points rather than assigning them invented durations."""
        return SpokenToken(self.text, self.start, self.end, f"mlx:{self.source_segments[-1]}")


def _fields(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or len(value) > 128:
        raise ValueError("Expected a bounded JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("JSON keys must be strings")
        result[key] = item
    return result


def _array(value: object, maximum: int) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError("Expected a bounded JSON array")
    return list(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite JSON number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("JSON number exceeds its numeric budget") from error
    if not isfinite(number):
        raise ValueError("Expected a finite JSON number")
    return number


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected JSON text")
    return value


def _measurement(value: object) -> object:
    response = _fields(value)
    if response.get("ok") is not True:
        raise ValueError("Executor measurement did not succeed")
    data = _fields(response.get("data"))
    error = data.get("isError")
    if error is not None and error is not False:
        raise ValueError("Native measurement returned an error")
    content = _fields(data.get("structuredContent"))
    if content.get("sourceModified") is not False:
        raise ValueError("Native measurement lacks its read-only source contract")
    return content.get("measurement")


def speech_tokens(value: object, *, utterance: str) -> tuple[SpokenToken, ...]:
    """Parse a final response to an explicitly wordTiming-enabled local request."""
    measured = _fields(_measurement(value))
    if measured.get("onDevice") is not True or measured.get("humanReviewed") is not False:
        raise ValueError("Expected local, automatically recognized speech")
    duration = _number(measured.get("durationSeconds"))
    if not 0 < duration <= 60:
        raise ValueError("Invalid bounded speech duration")
    result: list[SpokenToken] = []
    for item in _array(measured.get("segments"), 1000):
        segment = _fields(item)
        start = _number(segment.get("startSeconds"))
        length = _number(segment.get("durationSeconds"))
        if start + length > duration + 1e-9:
            raise ValueError("Native speech exceeds its measured audio duration")
        result.append(
            SpokenToken(
                _string(segment.get("text")), start, min(duration, start + length), utterance
            )
        )
    if sum(len(token.text.encode("utf-8")) for token in result) > 32768:
        raise ValueError("Speech text budget exceeded")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SoundWindow:
    start: float
    end: float
    speech_confidence: float
    music_confidence: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (self.start, self.end, self.speech_confidence, self.music_confidence)
        ):
            raise ValueError("Non-finite sound classification")
        if not 0 <= self.start < self.end <= 60:
            raise ValueError("Invalid sound classification clock")
        if not 0 <= self.speech_confidence <= 1 or not 0 <= self.music_confidence <= 1:
            raise ValueError("Invalid sound classification confidence")


def sound_windows(value: object) -> tuple[SoundWindow, ...]:
    """Validate raw local scores; low scores and empty results are not silence."""
    measured = _fields(_measurement(value))
    duration = _number(measured.get("durationSeconds"))
    if (
        not 0.5 <= duration <= 60
        or measured.get("humanReviewed") is not False
        or _number(measured.get("windowDurationSeconds")) != 0.5
        or _number(measured.get("overlapFactor")) != 0.5
    ):
        raise ValueError("Unexpected sound classification policy")
    result: list[SoundWindow] = []
    for item in _array(measured.get("windows"), 500):
        fields = _fields(item)
        start = _number(fields.get("startSeconds"))
        window = SoundWindow(
            start,
            start + _number(fields.get("durationSeconds")),
            _number(fields.get("speechConfidence")),
            _number(fields.get("musicConfidence")),
        )
        if window.end > duration or (result and window.start < result[-1].start):
            raise ValueError("Sound windows must be ordered and inside the measured input")
        result.append(window)
    if not result:
        raise ValueError("No classification evidence; cannot infer silence")
    return tuple(result)


def ocr_frames(value: object) -> tuple[FrameObservation, ...]:
    """Keep requested and actual frame times distinct, including held-frame cases."""
    result: list[FrameObservation] = []
    for item in _array(_measurement(value), 8):
        frame = _fields(item)
        width = _number(frame.get("width"))
        height = _number(frame.get("height"))
        requested = _number(frame.get("requestedTimeSeconds"))
        actual = _number(frame.get("actualTimeSeconds"))
        if not width.is_integer() or not height.is_integer():
            raise ValueError("Non-integral OCR canvas")
        if min(width, height) <= 0 or width * height > 16_000_000 or min(requested, actual) < 0:
            raise ValueError("Invalid OCR canvas or time")
        detections: list[TextDetection] = []
        for raw_line in _array(frame.get("lines"), 64):
            line = _fields(raw_line)
            region = _fields(line.get("region"))
            detections.append(
                TextDetection(
                    _string(line.get("text")),
                    _number(line.get("confidence")),
                    TextRectangle(
                        _number(region.get("x")),
                        _number(region.get("y")),
                        _number(region.get("width")),
                        _number(region.get("height")),
                    ),
                )
            )
        result.append(
            FrameObservation(requested, actual, int(width), int(height), tuple(detections))
        )
    if sum(len(line.text.encode("utf-8")) for frame in result for line in frame.detections) > 32768:
        raise ValueError("OCR batch text budget exceeded")
    return tuple(result)


def mlx_groups(value: object, *, duration: float) -> tuple[MLXWordGroup, ...]:
    """Conserve Japanese MLX text and native clocks, including unresolved points.

    Only pieces with EXACTLY the same native onset may share an interval union.
    No word is stretched into the next onset or assigned a proportional clock.
    Callers must inspect zero-duration groups; spoken_token() refuses those.
    Segment and finer word clocks are independently bounded by the measured audio;
    word clocks are not clamped into coarser segment clocks. Source indices retain
    the link to the original segment metadata for disagreement audits.
    An empty recognition result is not independent evidence of silence.
    """
    measured = _fields(value)
    if measured.get("language") != "ja" or not 0 < _number(duration) <= MAX_MLX_DURATION:
        raise ValueError("Expected bounded Japanese MLX transcription")
    text = _string(measured.get("text"))
    if len(text.encode("utf-8")) > MAX_MLX_TEXT_BYTES:
        raise ValueError("MLX text budget exceeded")
    result: list[MLXWordGroup] = []
    text_bytes = 0
    for word_count, word in enumerate(_mlx_pieces(measured.get("segments"), duration=duration), 1):
        text_bytes += len(word.text.encode("utf-8"))
        if word_count > MAX_MLX_WORDS or text_bytes > MAX_MLX_TEXT_BYTES:
            raise ValueError("MLX word budget exceeded")
        if result and word.start < result[-1].start:
            raise ValueError("MLX native word onsets are not ordered")
        if result and word.start == result[-1].start:
            word = _merge_mlx_onset(result.pop(), word)
        result.append(word)
    if "".join(text.split()) != "".join("".join(word.text for word in result).split()):
        raise ValueError("MLX words do not conserve the transcript text")
    return tuple(result)


def mlx_native_pieces(value: object, *, duration: float) -> tuple[MLXWordGroup, ...]:
    """Expose validated original pieces for provenance-bound fine-clock selection.

    Points are retained, and shared onsets are not split or apportioned. These
    pieces are evidence, not directly renderable ordered caption groups. Validate
    the complete transcript before returning any individual piece.
    """
    mlx_groups(value, duration=duration)
    return tuple(_mlx_pieces(_fields(value).get("segments"), duration=duration))


def _mlx_pieces(value: object, *, duration: float) -> Iterator[MLXWordGroup]:
    for index, segment in enumerate(_array(value, MAX_MLX_SEGMENTS)):
        yield from _mlx_segment(segment, index=index, duration=duration)


def _mlx_segment(value: object, *, index: int, duration: float) -> Iterator[MLXWordGroup]:
    segment = _fields(value)
    start = _number(segment.get("start"))
    end = _number(segment.get("end"))
    text = _string(segment.get("text"))
    if not 0 <= start <= end <= duration:
        raise ValueError("MLX segment exceeds its measured audio clock")
    words: list[str] = []
    for raw_word in _array(segment.get("words"), MAX_MLX_WORDS):
        fields = _fields(raw_word)
        word = MLXWordGroup(
            text=_string(fields.get("word")),
            start=_number(fields.get("start")),
            end=_number(fields.get("end")),
            source_segments=(index,),
            piece_count=1,
            minimum_probability=_number(fields.get("probability")),
        )
        if word.end > duration:
            raise ValueError("MLX word exceeds its measured audio clock")
        words.append(word.text)
        yield word
    if "".join(text.split()) != "".join("".join(words).split()):
        raise ValueError("MLX words do not conserve the segment text")


def _merge_mlx_onset(previous: MLXWordGroup, current: MLXWordGroup) -> MLXWordGroup:
    segments = previous.source_segments
    if segments[-1] != current.source_segments[-1]:
        segments += current.source_segments
    return replace(
        previous,
        text=previous.text + current.text,
        end=max(previous.end, current.end),
        source_segments=segments,
        piece_count=previous.piece_count + current.piece_count,
        minimum_probability=min(previous.minimum_probability, current.minimum_probability),
    )
