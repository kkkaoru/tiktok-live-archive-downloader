"""Candidate correspondence for aspect-preserving CFR proxies, not pixel identity."""

from fractions import Fraction

from replay.core import ReplayError
from replay.delivery_media import FrameClock


def verify_clocks(original: FrameClock, proxy: FrameClock) -> dict[str, object]:
    """Require one-to-one frames with only source-tick quantization, never drift."""
    if len(original.pts) != len(proxy.pts) or original.pts[0] != 0 or proxy.pts[0] != 0:
        raise ReplayError("Proxy correspondence requires equal zero-origin frame counts")
    if len(proxy.pts) < 2:
        raise ReplayError("Proxy correspondence requires at least two frames")
    step = proxy.pts[1] - proxy.pts[0]
    if any(b - a != step for a, b in zip(proxy.pts, proxy.pts[1:], strict=False)):
        raise ReplayError("Proxy correspondence requires a constant presentation grid")
    interval = step * proxy.time_base
    if original.time_base >= interval / 2 or proxy.last_duration != step:
        raise ReplayError("Source precision or proxy terminal duration is unsupported")
    errors = tuple(
        abs(source * original.time_base - target * proxy.time_base)
        for source, target in zip(original.pts, proxy.pts, strict=True)
    )
    if max(errors) > original.time_base / 2:
        raise ReplayError("Proxy clock differs beyond original tick quantization")
    original_end = (original.pts[-1] + original.last_duration) * original.time_base
    proxy_end = (proxy.pts[-1] + proxy.last_duration) * proxy.time_base
    if original_end != proxy_end:
        raise ReplayError("Proxy and original physical extents differ")
    return {
        "frames": len(proxy.pts),
        "originalTimeBase": str(original.time_base),
        "proxyTimeBase": str(proxy.time_base),
        "frameDuration": str(interval),
        "physicalEnd": str(original_end),
        "maximumTimestampDifference": str(max(errors)),
        "clockIdentical": all(error == Fraction(0) for error in errors),
        "mapping": "same-frame-index-with-source-tick-quantization",
        "pixelIdentityVerified": False,
    }
