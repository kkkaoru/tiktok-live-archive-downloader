"""Validate paging measurement provenance before reusing glyph data only."""

from dataclasses import asdict
from pathlib import Path

from replay import caption_native_layout, caption_style_defaults
from replay.caption_io import _number, _string
from replay.caption_measure_cache import CachedCaptionLayout
from replay.caption_measured_job import _json
from replay.caption_native_layout import MAX_RESPONSE_BYTES, NativeCaptionLayout
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import _checked_output, _digest

MAX_PART_BYTES = 16 * 1024**2
MAX_PINS = 20_000


def _document(path: Path, digest: str, limit: int) -> dict[str, object]:
    if not path.is_absolute() or _checked_output(path) != path or not path.is_file():
        raise ReplayError("Measurement cache requires absolute regular unlinked files")
    return record(_json(_bound_bytes(path, digest, limit)))


def _font_assets(path: Path, digest: str, native: NativeCaptionLayout) -> dict[str, str]:
    if path.suffix != ".json":
        return {}
    data = _document(path, digest, MAX_PART_BYTES)
    if "fonts" not in data:
        return {}
    primary = record(data.get("primary"))
    variations, fonts = primary.get("variations"), data["fonts"]
    if (
        primary.get("postScriptName") != native.style.font_name
        or _number(primary.get("pointSize")) != native.expected_point_size
        or not isinstance(variations, list)
        or len(variations) > 64
        or not isinstance(fonts, list)
        or not 1 <= len(fonts) <= 64
    ):
        raise ReplayError("Cached font identity or variation evidence differs")
    identifiers: set[int] = set()
    for value in variations:
        variation = record(value)
        identifier = variation.get("identifier")
        if type(identifier) is not int or not 0 <= identifier < 2**32 or identifier in identifiers:
            raise ReplayError("Invalid cached font variation axis")
        identifiers.add(identifier)
        _number(variation.get("value"))
    pins: dict[str, str] = {}
    for value in fonts:
        ref = record(value)
        asset, expected = Path(_string(ref.get("path"))), _digest(ref.get("sha256"))
        if (
            not asset.is_absolute()
            or _checked_output(asset) != asset
            or not asset.is_file()
            or not 0 < asset.stat().st_size <= 64 * 1024**2
            or file_digest(asset) != expected
        ):
            raise ReplayError("Cached font asset changed or is invalid")
        pins[str(asset)] = expected
    return pins


def _validate_batches(directory: Path, inputs: dict[str, str]) -> None:
    """Reject ambiguous JSON/numeric types before the existing geometry reader."""
    for name, digest in inputs.items():
        path = Path(name)
        if path.parent != directory or not path.name.endswith(".request.json"):
            continue
        request = _document(path, digest, MAX_RESPONSE_BYTES)
        for field in ("assFontSize", "border", "rim"):
            _number(request.get(field))
        response = path.with_name(path.name.removesuffix(".request.json") + ".response.json")
        response_digest = inputs.get(str(response))
        if response_digest is None:
            raise ReplayError("Measurement cache response is not pinned")
        data = _document(response, response_digest, MAX_RESPONSE_BYTES)
        rows = data.get("bounds")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
            raise ReplayError("Invalid cached measurement rows")
        for value in rows:
            widths = record(value).get("lineWidths")
            if not isinstance(widths, list) or len(widths) != 1:
                raise ReplayError("Invalid cached measurement line widths")
            _number(widths[0])


class PagingMeasurementCache(CachedCaptionLayout):
    """Existing cache mechanics plus independently verified producer/font pins."""

    def __init__(self, native: NativeCaptionLayout, *, provenance: dict[str, str]) -> None:
        super().__init__(native)
        self.provenance = dict(provenance)

    def verify_inputs(self) -> dict[str, str]:
        if any(file_digest(Path(path)) != digest for path, digest in self.provenance.items()):
            raise ReplayError("Paging measurement identity changed")
        return super().verify_inputs() | self.provenance


def reuse_measurements(
    native: NativeCaptionLayout, *, root: Path, segment: int, evidence: tuple[Path, ...]
) -> PagingMeasurementCache:
    """Reuse one prior segment's pinned glyph bounds, never captions or old code.

    Evidence is the current calibration/font/template evidence, excluding ASR
    documents. Incompatible or malformed caches fail explicitly. New text is
    measured by the live NativeCaptionLayout through the shared cache wrapper.
    """
    if type(segment) is not int or not 0 <= segment < 100_000 or not evidence:
        raise ReplayError("Invalid measurement cache segment or evidence")
    evidence = tuple(_checked_output(path) for path in evidence)
    root = _checked_output(root)
    if not root.is_dir():
        raise ReplayError("Measurement cache root must exist")
    part = _checked_output(root / f"part-{segment:04d}.json")
    failed_measurement = not part.exists()
    if failed_measurement:
        part = _checked_output(root / f"failure-{segment:04d}.json")
    digest = file_digest(part)
    data = _document(part, digest, MAX_PART_BYTES)
    if failed_measurement and (
        data.get("version") != 1
        or data.get("segment") != segment
        or data.get("candidateOnly") is not True
        or not isinstance(data.get("error"), str)
    ):
        raise ReplayError("Invalid failed-planning measurement evidence")
    settings = record(data.get("settings"))
    style = record(settings.get("style"))
    for key in ("ass_font_size", "border", "rim", "motion_line_advance"):
        _number(style.get(key))
    if (
        style != asdict(native.style)
        or _number(settings.get("pointSize")) != native.expected_point_size
        or _number(settings.get("width")) != native.width
    ):
        raise ReplayError("Paging measurement settings differ")
    values = record(data.get("inputs"))
    if not 1 <= len(values) <= MAX_PINS:
        raise ReplayError("Paging measurement input count exceeds budget")
    inputs = {name: _digest(value) for name, value in values.items()}
    current = native.verify_inputs()
    specification = Path(caption_native_layout.__file__).resolve()
    identity = (native.executable, specification, *evidence)
    provenance = {str(part): digest}
    assets: dict[str, str] = {}
    for path in identity:
        path = _checked_output(path)
        name = str(path)
        if name not in current or inputs.get(name) != current[name]:
            raise ReplayError("Paging measurement producer or calibration differs")
        provenance[name] = current[name]
        if path in evidence:
            assets.update(_font_assets(path, current[name], native))
    if not assets:
        raise ReplayError("Paging measurement needs pinned font asset and variation evidence")
    provenance.update(assets)
    for module in (Path(__file__).resolve(), Path(caption_style_defaults.__file__).resolve()):
        provenance[str(module)] = file_digest(module)
    wrapped = PagingMeasurementCache(native, provenance=provenance)
    directories = sorted(
        {
            _checked_output(Path(name).parent)
            for name in inputs
            if Path(name).name.endswith(".request.json")
        }
    )
    if not directories:
        raise ReplayError("Measurement cache has no pinned glyph requests")
    for directory in directories:
        _validate_batches(directory, inputs)
        wrapped.reuse(directory=directory, inputs=inputs)
    wrapped.verify_inputs()
    return wrapped
