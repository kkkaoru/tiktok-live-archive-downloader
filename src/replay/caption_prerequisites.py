"""Finite metadata diagnostics, never calibration or delivery-constraint approval."""

from fractions import Fraction
from xml.etree import ElementTree as ET

from replay import delivery_caption_audit as audit
from replay.core import ReplayError

MAX_STATES = 7200


def template_differences(historical: str, selected: str) -> list[str]:
    """Compare complete XML structure; only sceneSettings/presetPath is metadata.

    An empty result is necessary, not sufficient, for geometry transfer: resource
    resolution, font identity, producer code and calibration scope still need pins.
    No text, timing, IDs, transforms or style fields are normalized away.
    """
    if max(len(historical), len(selected)) > 8 * 1024 * 1024:
        raise ReplayError("Template comparison exceeds byte budget")
    try:
        roots = [ET.fromstring(text) for text in (historical, selected)]
    except ET.ParseError as error:
        raise ReplayError("Invalid template XML") from error
    result: list[str] = []

    def compare(left: ET.Element, right: ET.Element, path: str) -> None:
        if left.tag != right.tag or left.attrib != right.attrib:
            result.append(path + "/@structure")
        left_text, right_text = left.text or "", right.text or ""
        if len(left) and len(right):
            left_text, right_text = left_text.strip(), right_text.strip()
        if path != "/ozml/scene/sceneSettings/presetPath" and left_text != right_text:
            result.append(path + "/text()")
        if len(left) != len(right):
            result.append(path + "/@children")
        for index, (a, b) in enumerate(zip(left, right, strict=False)):
            suffix = a.tag
            if sum(child.tag == a.tag for child in left) > 1:
                suffix += f"[{index}]"
            compare(a, b, path + "/" + suffix)

    compare(roots[0], roots[1], "/" + roots[0].tag)
    return result


def require_same_geometry(
    historical: str,
    selected: str,
    *,
    old_identity: dict[str, object],
    new_identity: dict[str, object],
) -> None:
    """Conservative precondition, not a transfer issuer or template-gate bypass."""
    required = {"font", "variation", "native", "code", "model", "policy", "resource"}
    if set(old_identity) != required or set(new_identity) != required:
        raise ReplayError("Incomplete independently bound geometry identities")
    if any(value is None for value in old_identity.values()) or old_identity != new_identity:
        raise ReplayError(
            "Geometry identity changed or remains unobserved; new calibration required"
        )
    if template_differences(historical, selected):
        raise ReplayError("Template geometry differs; new calibration required")


def observation_requests(metadata: dict[str, object]) -> dict[str, object]:
    """Deduplicate retained actual PTS support, not a 30fps sampling grid.

    The containing predecessor in each support interval is retained even when its
    PTS precedes interval start. Do not request exclusive-end frames or bridge gaps.
    Input provenance must be verified by the caller before these requests are run.
    """
    states = audit.rows(metadata.get("states"))
    if not 1 <= len(states) <= MAX_STATES:
        raise ReplayError("Invalid observation request state budget")
    windows: list[tuple[Fraction, Fraction, set[Fraction]]] = []
    coverage: list[dict[str, object]] = []
    visible = 0
    for state in states:
        start = audit.integer(state.get("globalStart"))
        end = audit.integer(state.get("globalEnd"))
        if end <= start or (coverage and start < audit.integer(coverage[-1]["globalEnd"])):
            raise ReplayError("Invalid visible output clock")
        support = audit.rows(state.get("sourceSupport"))
        if not support:
            raise ReplayError("Missing actual source support")
        for interval in support:
            windows.append(_interval(interval))
        visible += end - start
        coverage.append(
            {
                key: state[key]
                for key in ("part", "stateIndex", "globalStart", "globalEnd", "sourceSupport")
            }
        )
    if visible != metadata.get("visibleFrames") or len(states) != metadata.get("stateCount"):
        raise ReplayError("Retained visible coverage differs")
    merged: list[tuple[Fraction, Fraction, set[Fraction]]] = []
    for start_time, end_time, points in sorted(windows, key=lambda row: row[0]):
        if merged and start_time <= merged[-1][1]:
            previous_start, previous_end, previous_points = merged.pop()
            merged.append((previous_start, max(end_time, previous_end), previous_points | points))
        else:
            merged.append((start_time, end_time, points))
    return {
        "version": 1,
        "kind": "caption-observation-request",
        "executionAuthorized": False,
        "constraintsGenerated": False,
        "visibleFrames": visible,
        "stateCount": len(states),
        "requiredObservations": [
            "independent-face-box-confidence",
            "independent-head-hair-top-confidence",
        ],
        "missingMustRemainUnresolved": True,
        "sourceIdentity": None,
        "windows": [
            {
                "startSeconds": str(start),
                "endSecondsExclusive": str(end),
                "actualPTSSeconds": [str(point) for point in sorted(points)],
            }
            for start, end, points in merged
        ],
        "coverage": coverage,
    }


def _interval(interval: dict[str, object]) -> tuple[Fraction, Fraction, set[Fraction]]:
    start, end = (_time(interval.get(key)) for key in ("sourceStartSeconds", "sourceEndSeconds"))
    raw = interval.get("observedSourcePTSSeconds")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_STATES:
        raise ReplayError("Missing bounded actual source PTS")
    points = [_time(value) for value in raw]
    if (
        end <= start
        or points != sorted(set(points))
        or points[0] > start
        or points[-1] >= end
        or sum(point < start for point in points) > 1
    ):
        raise ReplayError("Invalid source support interval or containing predecessor")
    return start, end, set(points)


def _time(value: object) -> Fraction:
    if not isinstance(value, str) or len(value) > 80:
        raise ReplayError("Expected retained rational source PTS")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid retained rational source PTS") from error
    if not 0 <= result <= 86400:
        raise ReplayError("Source PTS outside bounded day")
    return result
