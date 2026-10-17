"""Validate a sampled-OCR (or YOLO-planned) chunk admission before it may feed later stages.

The per-frame eligibility contract needs OCR for every active frame. A sampled admission
has OCR only on the sampled frames; the rest are bracket-filled regions. This validator
binds that honesty: sampling provenance, exact frame clock, region/span membership, an
accepted attempt whose audit passed, and pins matching the declared source. It certifies
internal consistency of the evidence, not removal quality or the absence of text.
"""

from collections.abc import Mapping

from replay.recordings import is_record
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_plan import repair_spans
from replay.subtitle_sampled_observations import FILL_RULE
from replay.subtitle_stream import SubtitleSpan

POLICIES = frozenset(
    {
        "sampled-ocr-plus-detector-regions-v1",
        "sampled-ocr-plus-detector-regions-v2",
        "sampled-ocr-plus-detector-regions-v3",
        "sampled-ocr-plus-detector-regions-v4",
        "sampled-ocr-plus-detector-regions-v5",
    }
)
YOLO_POLICY = "yolo-caption-boxes-v1"
RECEIPT_KIND = "sampled-ocr-repair-chunk"
SAMPLING_KIND = "sampled-ocr-observation"
YOLO_SAMPLING_KIND = "yolo-caption-boxes"
MAX_STRIDE = 30


def _record(value: object, label: str) -> dict[str, object]:
    if not is_record(value):
        raise ValueError(f"Missing sampled eligibility {label}")
    return value


def _integers(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"Invalid sampled eligibility {label}")
    return [item for item in value if type(item) is int]


def _yolo_sampling(sampling: Mapping[str, object], *, count: int) -> list[int]:
    """A YOLO-planned admission read no frame by OCR: no frame may claim to be sampled."""
    if (
        sampling.get("version") != 1
        or sampling.get("rawOcr") is not False
        or sampling.get("interpolated") is not False
        or sampling.get("qualityApproved") is not False
        or sampling.get("frames") != count
        or sampling.get("detectedFrames") != count
    ):
        raise ValueError("Sampling receipt does not honestly describe YOLO detection")
    return []


def _sampling(value: object, *, count: int, policy: object) -> list[int]:
    sampling = _record(value, "sampling receipt")
    if (policy == YOLO_POLICY) is not (sampling.get("kind") == YOLO_SAMPLING_KIND):
        raise ValueError("Admission policy and sampling kind disagree")
    if policy == YOLO_POLICY:
        return _yolo_sampling(sampling, count=count)
    stride = sampling.get("stride")
    if (
        sampling.get("kind") != SAMPLING_KIND
        or sampling.get("fillRule") != FILL_RULE
        or sampling.get("rawOcr") is not False
        or sampling.get("interpolated") is not True
        or sampling.get("qualityApproved") is not False
        or sampling.get("frames") != count
        or type(stride) is not int
        or not 1 <= stride <= MAX_STRIDE
    ):
        raise ValueError("Sampling receipt does not honestly describe sampled OCR")
    sampled = _integers(sampling.get("sampledFrames"), "sampled frames")
    if (
        not sampled
        or sampled != sorted(set(sampled))
        or sampled[0] != 0
        or sampled[-1] != count - 1
    ):
        raise ValueError(
            "Sampled frames must be unique, ordered and include the first and last frame"
        )
    if any(b - a > stride for a, b in zip(sampled, sampled[1:], strict=False)):
        raise ValueError("Sampled frames leave a gap wider than the declared stride")
    return sampled


def _accepted_attempt(
    receipt: Mapping[str, object], pins: Mapping[str, str], source_sha256: str
) -> dict[str, object]:
    if (
        receipt.get("version") != 1
        or receipt.get("kind") != RECEIPT_KIND
        or receipt.get("qualityApproved") is not False
        or receipt.get("deliveryComplete") is not False
        or receipt.get("regionsFromSampling") is not True
        or receipt.get("pins") != dict(pins)
        or pins.get("source") != source_sha256
    ):
        raise ValueError("Chunk receipt identity, pins or approval flags are invalid")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
        raise ValueError("Chunk receipt must list one or two attempts")
    records = [_record(item, "attempt") for item in attempts]
    accepted = [item for item in records if item.get("directory") == receipt.get("accepted")]
    if len(accepted) != 1:
        raise ValueError("Chunk receipt has no unique accepted attempt")
    attempt = accepted[0]
    audited = attempt.get("auditedFrames")
    if (
        attempt.get("passed") is not True
        or attempt.get("residualLines") != []
        or type(audited) is not int
        or audited < 1
        or attempt is not records[-1]
        or receipt.get("escalated") is not (len(attempts) == 2)
    ):
        raise ValueError("Accepted attempt did not pass a non-empty residual-text audit")
    return attempt


def _membership(
    frames: object,
    *,
    clock: VideoFrameClock,
    sampled: list[int],
    spans: tuple[SubtitleSpan, ...],
) -> list[int]:
    """Check every frame record and return the PTS of frames left unknown."""
    if not isinstance(frames, list) or len(frames) != len(clock.pts):
        raise ValueError("Admission frame records differ from the source clock")
    sampled_set = set(sampled)
    covered = {i for span in spans for i in range(span.start_frame, span.end_frame)}
    unknown: list[int] = []
    for index, (item, pts) in enumerate(zip(frames, clock.pts, strict=True)):
        entry = _record(item, "frame record")
        active = entry.get("classification") == "overlay-subtitle"
        if (
            entry.get("sourcePTS") != pts
            or entry.get("observation") != index
            or active is not (index in covered)
            or (active and entry.get("sampled") is not (index in sampled_set))
            or (not active and entry.get("classification") != "unknown")
        ):
            raise ValueError(f"Admission record {index} differs from clock or repair spans")
        if not active:
            unknown.append(pts)
    return unknown


def validate_sampled_eligibility(
    receipt: object,
    admission: object,
    *,
    source_sha256: str,
    clock: VideoFrameClock,
    width: int,
    height: int,
    pins: Mapping[str, str],
) -> tuple[SubtitleSpan, ...]:
    """Return the repair spans when the sampled evidence is internally consistent."""
    attempt = _accepted_attempt(_record(receipt, "chunk receipt"), pins, source_sha256)
    plan = _record(admission, "admission")
    count = len(clock.pts)
    if (
        plan.get("version") != 1
        or plan.get("policy") not in POLICIES | {YOLO_POLICY}
        or plan.get("removalQualityApproved") is not False
        or plan.get("unknownFramesKeptUnmodified") is not True
    ):
        raise ValueError("Admission policy or approval flags are invalid")
    if attempt.get("sampling") != plan.get("sampling"):
        raise ValueError("Admission sampling differs from the accepted attempt")
    sampled = _sampling(plan.get("sampling"), count=count, policy=plan.get("policy"))
    spans = repair_spans(
        plan.get("masks"), frames=count, width=width, height=height, source_clock=clock
    )
    unknown = _membership(plan.get("frames"), clock=clock, sampled=sampled, spans=spans)
    if plan.get("unknownSourcePTS") != unknown:
        raise ValueError("Unknown source PTS list differs from the frame records")
    return spans
