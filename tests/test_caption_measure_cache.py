import json

import pytest
from test_caption_native_layout import layout as layout
from test_caption_native_reflow import exact as exact

from replay import caption_measure_cache as cache
from replay.caption_native_layout import NativeCaptionLayout
from replay.core import ReplayError
from replay.stages import file_digest


def test_reuse_native_bounds_and_measure_only_new_text(exact: NativeCaptionLayout) -> None:
    exact.measure("ab")
    wrapped = cache.CachedCaptionLayout(exact)
    wrapped.reuse(directory=exact.directory, inputs=exact.verify_inputs())
    assert wrapped.measure("ab\nab") == (24, 26)
    assert len(tuple(exact.directory.glob("*.response.json"))) == 1
    assert wrapped.measure("cd") == (24, 12)
    assert len(tuple(exact.directory.glob("*.response.json"))) == 2
    assert wrapped.verify_inputs()
    with pytest.raises(ReplayError, match="three"):
        wrapped.measure("a\nb\nc\nd")
    response = exact.directory / "batch-000000.response.json"
    response.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="changed"):
        wrapped.verify_inputs()


@pytest.mark.parametrize("change", ["missing", "style", "text", "width", "size", "empty"])
def test_invalid_cache_refused(exact: NativeCaptionLayout, change: str) -> None:
    exact.measure("ab")
    inputs = exact.verify_inputs()
    response = exact.directory / "batch-000000.response.json"
    raw = json.loads(response.read_text(encoding="utf-8"))
    if change == "missing":
        del inputs[str(response)]
    elif change == "style":
        raw["font"]["postScriptName"] = "Other"
    elif change == "text":
        raw["bounds"][0]["text"] = "bad"
    elif change == "width":
        raw["bounds"][0]["lineWidths"] = [1]
    elif change == "size":
        raw["font"]["pointSize"] = 999
    else:
        inputs = {}
    if change not in {"missing", "empty"}:
        response.write_text(json.dumps(raw), encoding="utf-8")
        inputs[str(response)] = file_digest(response)
    with pytest.raises(ReplayError):
        cache.CachedCaptionLayout(exact).reuse(directory=exact.directory, inputs=inputs)


def test_resource_bound_enforced(
    exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapped = cache.CachedCaptionLayout(exact)
    monkeypatch.setattr(cache, "MAX_CACHE", 1)
    with pytest.raises(ReplayError, match="budget"):
        wrapped.prepare_lines(("ab", "cd"))
