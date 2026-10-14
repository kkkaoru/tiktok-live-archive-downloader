import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_fcp_head_near import paired as paired

from replay import artifact_json
from replay import fcp_head_refresh as refresh
from replay import tracking_retained as tracking_evidence
from replay.core import ReplayError
from replay.delivery_mask_quality import rows
from replay.fcp_caption_stability import stabilize_caption_y
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_measured import CaptionAnchor
from replay.fcp_project import CaptionTemplate
from replay.fcp_tracking_caption import MissingFaceRange, TrackingCalibration
from replay.fcp_visible_arc import ArcCell, VisibleArc
from replay.head_approx_placement import ApproxCell, ApproxPlan, DerivedMotion
from replay.head_near_placement import FaceSizeObservation
from replay.recordings import record
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob
from replay.tracking_retained import RetainedTracking


@pytest.fixture
def root() -> ET.Element:
    return ET.fromstring(
        '<fcpxml><asset-clip ref="original" offset="0s" start="0s" duration="2s">'
        '<title offset="0s" start="3600s" duration="2s">'
        "<text>本文</text><text>本文</text><text>本文</text>"
        '<adjust-transform position="0 999"><param name="position">'
        '<keyframeAnimation><keyframe time="3600s" value="0 999"/>'
        "</keyframeAnimation></param></adjust-transform></title></asset-clip></fcpxml>"
    )


def test_refresh_missing_multiline_uses_one_measured_bottom(paired: FaceScaleSeries) -> None:
    root = ET.fromstring(
        '<fcpxml><asset-clip ref="original" offset="0s" start="0s" duration="1s">'
        '<title offset="0s" start="3600s" duration="1/4s"><text>甲</text><text>甲</text>'
        '<text>甲</text><adjust-transform position="0 999"/></title>'
        '<title offset="1/4s" start="3600s" duration="1/4s"><text>甲\n乙</text>'
        '<text>甲\n乙</text><text>甲\n乙</text><adjust-transform position="0 999"/></title>'
        '<title offset="1/2s" start="3600s" duration="1/4s"><text>甲\n乙\n丙</text>'
        '<text>甲\n乙\n丙</text><text>甲\n乙\n丙</text><adjust-transform position="0 999"/></title>'
        '<title offset="3/4s" start="3600s" duration="1/4s"><text>丁</text><text>丁</text>'
        '<text>丁</text><adjust-transform position="0 999"/></title></asset-clip></fcpxml>'
    )
    before = refresh._without_positions(root)
    gap = MissingFaceRange(
        Fraction(0), Fraction(1), 30, Fraction(1), "leading-observation-backfill", "missing"
    )
    faces = FaceScaleSeries(
        replace(paired.tops, derived_ranges=(gap,)),
        replace(paired.bottoms, derived_ranges=(gap,)),
    )
    refresh.refresh(
        root,
        faces=faces,
        calibration=TrackingCalibration(Fraction(1), 400, 1, 0, 24, 1920, (0, 400, 100, 100)),
        template=CaptionTemplate("test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM),
        measure=lambda text: (100, 100 + 120 * text.count("\n")),
    )
    assert tuple(
        title.findall(".//keyframe")[0].get("value") for title in root.findall(".//title")
    ) == ("0 31.0416666667", "0 31.0416666667", "0 31.0416666667", "0 31.0416666667")
    assert refresh._without_positions(root) == before
    assert faces.tops.derived_ranges == (gap,)


@pytest.mark.parametrize("fault", ["none", "texts", "empty", "width", "transform"])
def test_refresh_discards_old_y_preserves_media_text_clock(
    root: ET.Element, paired: FaceScaleSeries, fault: str
) -> None:
    title = root.find(".//title")
    assert title is not None
    if fault == "texts":
        title.append(ET.Element("text"))
    if fault == "empty":
        root.clear()
    if fault == "transform":
        transform = title.find("adjust-transform")
        assert transform is not None
        title.remove(transform)
    before = refresh._without_positions(root)
    template = CaptionTemplate("test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM)
    calibration = TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 900, 100, 400))
    if fault != "none":
        with pytest.raises(ReplayError):
            refresh.refresh(
                root,
                faces=paired,
                calibration=calibration,
                template=template,
                measure=lambda text: (2000 if fault == "width" else 100, 100),
            )
        return
    refresh.refresh(
        root,
        faces=paired,
        calibration=calibration,
        template=template,
        measure=lambda text: (100, 100),
    )
    assert refresh._without_positions(root) == before
    key = title.find(".//keyframe")
    assert key is not None
    assert key.get("value") == "0 25.2083333333"


def test_invalid_run_never_creates_output(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"version": 0}), encoding="utf-8")
    with pytest.raises(ReplayError, match="version"):
        refresh.run(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize(
    "policy",
    ["derived-face-pixel-motion-approximation", "bounded-face-pixel-target-y-approximation-v1"],
)
def test_derived_y_one_two_three_one_rows_preserve_bottom_stack_and_raw_face_flags(
    paired: FaceScaleSeries,
    monkeypatch: pytest.MonkeyPatch,
    unknown: bool,
    policy: str,
) -> None:
    document = ET.fromstring(
        '<fcpxml><asset-clip ref="original" offset="0s" start="0s" duration="1s">'
        '<title offset="0s" start="3600s" duration="1/4s"><text>甲</text><text>甲</text>'
        '<text>甲</text><adjust-transform position="0 999"/></title>'
        '<title offset="1/4s" start="3600s" duration="1/4s"><text>甲\n乙</text>'
        '<text>甲\n乙</text><text>甲\n乙</text><adjust-transform position="0 999"/></title>'
        '<title offset="1/2s" start="3600s" duration="1/4s"><text>甲\n乙\n丙</text>'
        '<text>甲\n乙\n丙</text><text>甲\n乙\n丙</text><adjust-transform position="0 999"/></title>'
        '<title offset="3/4s" start="3600s" duration="1/4s"><text>丁</text><text>丁</text>'
        '<text>丁</text><adjust-transform position="0 999"/></title></asset-clip></fcpxml>'
    )
    raw = (None, None, None, None) if unknown else (700, 702, 698, 720)
    plan = ApproxPlan(
        tuple(
            ApproxCell(
                Fraction(i, 4),
                Fraction(i + 1, 4),
                Fraction(i, 4),
                top,
                Fraction(0),
                "face-motion-derived-candidate",
                False,
                top,
            )
            for i, top in enumerate(raw)
        ),
        (),
        policy,
    )
    before = refresh._without_positions(document)

    applied: list[bool] = []

    def onset_filter(root: ET.Element) -> None:
        applied.append(True)
        stabilize_caption_y(root)

    monkeypatch.setattr(refresh, "stabilize_caption_y", onset_filter)
    reports = refresh.refresh(
        document,
        faces=paired,
        calibration=None,
        template=CaptionTemplate("test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM),
        measure=lambda text: (100, 100 + 120 * text.count("\n")),
        derived_motion=DerivedMotion(plan, {}),
    )
    assert refresh._without_positions(document) == before
    assert applied == [True]
    assert {
        len({key.get("value") for key in title.findall(".//keyframe")})
        for title in document.findall(".//title")
    } == {1}
    cells = rows(record(record(reports[0]["approximation"])["plan"])["cells"])
    assert record(cells[0])["currentFaceObserved"] is False
    assert record(cells[0])["measuredCurrentUpperPixels"] is False
    values = tuple(
        title.findall(".//keyframe")[0].get("value") for title in document.findall(".//title")
    )
    if unknown:
        assert values == (
            "0 31.0416666667",
            "0 31.0416666667",
            "0 31.0416666667",
            "0 31.0416666667",
        )
    else:
        assert values == ("0 14.7916666667", "0 14.7916666667", "0 14.7916666667", "0 14.0625")


@pytest.mark.parametrize("fault", ["both", "missing", "coverage"])
def test_derived_backend_is_explicit_and_physically_scoped(
    root: ET.Element,
    paired: FaceScaleSeries,
    fault: str,
) -> None:
    plan = ApproxPlan(
        (
            ApproxCell(
                Fraction(0),
                Fraction(1),
                Fraction(0),
                700,
                Fraction(0),
                "face-motion-derived-candidate",
                True,
                700,
            ),
        ),
        (),
        "derived-face-pixel-motion-approximation",
    )
    with pytest.raises(ReplayError):
        refresh.refresh(
            root,
            faces=paired,
            calibration=None,
            template=CaptionTemplate(
                "test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM
            ),
            measure=lambda text: (100, 100),
            derived_motion=None if fault == "missing" else DerivedMotion(plan, {}),
            current_upper_pixels=VisibleArc((), {}) if fault == "both" else None,
        )


def test_current_visible_arc_only_overrides_exact_missing_cell(
    root: ET.Element, paired: FaceScaleSeries
) -> None:
    gap = (
        MissingFaceRange(Fraction(1), Fraction(2), 30, Fraction(0), "last-known-hold", "missing"),
    )
    faces = replace(
        paired,
        tops=replace(paired.tops, derived_ranges=gap),
        bottoms=replace(paired.bottoms, derived_ranges=gap),
    )
    arc = VisibleArc((ArcCell(Fraction(1), Fraction(3, 2), 700, 4, 3),), {})
    title = root.find(".//title")
    assert title is not None
    title.set("offset", "1s")
    title.set("duration", "1/2s")
    reports = refresh.refresh(
        root,
        faces=faces,
        calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 900, 100, 400)),
        template=CaptionTemplate("test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM),
        measure=lambda text: (100, 100),
        visible_arc=arc,
    )
    assert len(reports) == 1
    assert reports[0]["sourceTime"] == "1"
    assert reports[0]["unknownPoints"] == 3
    assert reports[0]["captionBottom"] == 676
    assert reports[0]["detectedFace"] is False
    assert reports[0]["completeHeadVerified"] is False


@pytest.mark.parametrize("fault", ["none", "both", "empty", "duplicate", "conflict"])
def test_full_measurement_sources_keep_old_pins(tmp_path: Path, fault: str) -> None:
    paths = (tmp_path / "first.json", tmp_path / "second.json")
    for index, path in enumerate(paths):
        path.write_text(
            json.dumps({"inputs": {"glyph": "changed" if fault == "conflict" and index else "old"}})
        )
    pins = [{"path": str(path), "sha256": file_digest(path)} for path in paths]
    config: dict[str, object] = {"parts": pins}
    if fault == "both":
        config["part"] = pins[0]
    elif fault == "empty":
        config["parts"] = []
    elif fault == "duplicate":
        config["parts"] = [pins[0], pins[0]]
    if fault != "none":
        with pytest.raises(ReplayError):
            refresh.measurement_pins(refresh.measurement_sources(config))
    else:
        assert refresh.measurement_pins(refresh.measurement_sources(config)) == {"glyph": "old"}


def test_receipt_budget_is_separate_from_configuration_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"inputs": {"glyph": "hash"}, "extra": "x" * 300}))
    monkeypatch.setattr(artifact_json, "MAX_JSON_BYTES", 128)
    monkeypatch.setattr(refresh, "MAX_MEASUREMENT_RECEIPT_BYTES", 1024)
    assert refresh.measurement_pins((path,)) == {"glyph": "hash"}
    with pytest.raises(ValueError, match="byte budget"):
        artifact_json._json(path)
    monkeypatch.setattr(refresh, "MAX_MEASUREMENT_RECEIPT_BYTES", 128)
    with pytest.raises(ValueError, match="byte budget"):
        refresh.measurement_pins((path,))


@pytest.mark.parametrize("union", [False, True])
def test_measurement_input_counts_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, union: bool
) -> None:
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps({"inputs": {"a": "1", "b": "2"}}))
    second.write_text(json.dumps({"inputs": {"c": "3", "d": "4"}}))
    monkeypatch.setattr(refresh, "MAX_MEASUREMENT_INPUTS", 2 if union else 1)
    with pytest.raises(ReplayError, match="input count budget"):
        refresh.measurement_pins((first, second))


@pytest.mark.parametrize("fault", ["none", "count", "transform"])
def test_placement_change_clock_and_preserved_titles(root: ET.Element, fault: str) -> None:
    before = ET.fromstring(ET.tostring(root))
    title = root.find(".//title")
    assert title is not None
    transform = title.find("adjust-transform")
    assert transform is not None
    if fault == "count":
        root.clear()
    elif fault == "transform":
        title.remove(transform)
    else:
        transform.set("position", "0 10")
    if fault != "none":
        with pytest.raises(ReplayError):
            refresh.placement_changes(before, root)
    else:
        assert refresh.placement_changes(before, root) == [
            {"title": 0, "start": "0", "end": "2", "changed": True}
        ]


@pytest.mark.parametrize("fault", ["none", "source", "parts"])
def test_partition_impact_does_not_require_unchanged_parts_render(
    tmp_path: Path, fault: str
) -> None:
    xml = tmp_path / "source.xml"
    other = tmp_path / "other.xml"
    xml.write_text("xml")
    other.write_text("other")
    source = other if fault == "source" else xml
    pin = {"path": str(xml), "sha256": file_digest(xml)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {"path": str(source), "sha256": file_digest(source)},
                "parts": []
                if fault == "parts"
                else [
                    pin | {"globalStart": "0s", "duration": "2s"},
                    pin | {"globalStart": "2s", "duration": "2s"},
                ],
            }
        )
    )
    changes: list[dict[str, object]] = [{"start": "0", "end": "2", "changed": True}]
    if fault != "none":
        with pytest.raises(ReplayError):
            refresh.partition_impact(manifest, xml, changes, start=Fraction(0))
    else:
        result = refresh.partition_impact(manifest, xml, changes, start=Fraction(0))
        assert [row["changedTitles"] for row in result] == [1, 0]
        assert [row["renderReuseCandidate"] for row in result] == [False, True]


class Measurer:
    def measure(self, text: str) -> tuple[float, float]:
        return 100, 100

    def verify_inputs(self) -> dict[str, str]:
        return {}


class RetainedPixelsStub:
    def __init__(self, arc: VisibleArc) -> None:
        self.arc = arc

    def receipt(self) -> dict[str, object]:
        return {"candidateOnly": True, "semanticHeadVerified": False}


@pytest.mark.parametrize(
    "mode",
    [
        "image",
        "retained",
        "policy",
        "both",
        "absent",
        "extra",
        "reset-count",
        "duplicate",
        "reset-type",
    ],
)
def test_approximate_policy_requires_explicit_seed_kind_and_reset_clocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    job = TrackingJob(
        tmp_path / "config",
        "a" * 64,
        tmp_path / "source",
        tmp_path / "native",
        "b" * 64,
        "c" * 64,
        VideoFrameClock((0, 1), Fraction(1), 1),
        1080,
        1920,
        0.5,
    )
    arc = VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {})
    monkeypatch.setattr(refresh, "read_image_binding", lambda *args, **kwargs: arc)
    monkeypatch.setattr(
        refresh, "read_retained_image_binding", lambda *args, **kwargs: RetainedPixelsStub(arc)
    )
    data: dict[str, object] = {
        "policy": "current-face-y-offset-candidate",
        "image": {},
        "zoomResetTimes": ["1", "2"],
    }
    if mode == "retained":
        del data["image"]
        data["retainedImage"] = {}
    elif mode == "both":
        data["retainedImage"] = {}
    elif mode == "policy":
        data["policy"] = "old-reference"
    elif mode == "absent":
        del data["image"]
    elif mode == "extra":
        data["referenceHeadTop"] = 1
    elif mode == "reset-count":
        monkeypatch.setattr(refresh, "MAX_REFRESH_TITLES", 1)
    elif mode == "duplicate":
        data["zoomResetTimes"] = ["1", "1"]
    elif mode == "reset-type":
        data["zoomResetTimes"] = [True]
    if mode not in ("image", "retained"):
        with pytest.raises(ReplayError):
            refresh.approximate_inputs(data, job=job)
        return
    result, resets, evidence = refresh.approximate_inputs(data, job=job)
    assert result == arc
    assert resets == (Fraction(1), Fraction(2))
    if mode == "retained":
        assert evidence == {"candidateOnly": True, "semanticHeadVerified": False}
    else:
        assert evidence is None


@pytest.mark.parametrize("fault", ["calibration", "two-backends"])
def test_refresh_backend_contract_does_not_silently_substitute(
    root: ET.Element,
    paired: FaceScaleSeries,
    fault: str,
) -> None:
    with pytest.raises(ReplayError):
        refresh.refresh(
            root,
            faces=paired,
            calibration=None,
            template=CaptionTemplate(
                "test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM
            ),
            measure=lambda text: (100, 100),
            current_upper_pixels=VisibleArc((), {}) if fault == "two-backends" else None,
            approximate_pixels=VisibleArc((), {}) if fault == "two-backends" else None,
        )


@pytest.mark.parametrize("fault", ["none", "lineage", "pins", "name", "mutated"])
@pytest.mark.parametrize("long_capability", [False, True])
@pytest.mark.parametrize("backend", ["reference", "image", "approximate", "derived", "uncertain"])
@pytest.mark.parametrize("retained_tracking", [False, True])
def test_run_pinned_inputs_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root: ET.Element,
    paired: FaceScaleSeries,
    fault: str,
    long_capability: bool,
    backend: str,
    retained_tracking: bool,
) -> None:
    image_capability = backend == "image"
    derived_policy = (
        "bounded-face-pixel-target-y-approximation-v1"
        if backend == "uncertain"
        else "derived-face-pixel-motion-approximation"
    )
    document = tmp_path / "source.xml"
    document.write_bytes(ET.tostring(root))
    evidence = tmp_path / "data.json"
    evidence.write_text("{}", encoding="utf-8")
    pin = {"path": str(evidence), "sha256": file_digest(evidence)}
    xml_pin = {"path": str(document), "sha256": file_digest(document)}
    lineage = tmp_path / "lineage.json"
    lineage.write_text(
        json.dumps({"outputXML": xml_pin, "captionClockEqual": fault != "lineage"}),
        encoding="utf-8",
    )
    part = tmp_path / "part.json"
    part.write_text(
        json.dumps({"inputs": {} if fault == "pins" else {"glyph": "hash"}}), encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "xml": xml_pin,
                "lineage": {"path": str(lineage), "sha256": file_digest(lineage)},
                "part": {"path": str(part), "sha256": file_digest(part)},
                "faceIndex": pin,
                "template": pin,
                "nativeExecutable": pin,
                "longExecutable": pin if long_capability else None,
                "measurementReference": pin,
                "start": "0",
                "end": "2",
                "name": "" if fault == "name" else "fresh geometry",
                "currentUpperPixels": {"config": pin, "provenance": pin}
                if image_capability
                else None,
                "approxHead": {
                    "policy": derived_policy
                    if backend in ("derived", "uncertain")
                    else "current-face-y-offset-candidate",
                    "image": {"config": pin, "provenance": pin},
                }
                if backend in ("approximate", "derived", "uncertain")
                else None,
                "retainedTracking": pin if retained_tracking else None,
                **({"referenceTime": "1000", "referenceHeadTop": 1} if image_capability else {}),
            }
        ),
        encoding="utf-8",
    )
    faces = replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 50, 100),
        ),
    )
    monkeypatch.setattr(refresh, "window", lambda *args: root)
    monkeypatch.setattr(refresh, "face_references", lambda *args: [])
    job = TrackingJob(
        evidence,
        file_digest(evidence),
        document,
        evidence,
        file_digest(document),
        file_digest(evidence),
        VideoFrameClock((0, 1), Fraction(1), 1),
        1080,
        1920,
        0.5,
    )
    monkeypatch.setattr(refresh, "read_face_scale", lambda *args, **kwargs: (job, faces))
    monkeypatch.setattr(
        refresh,
        "verify_catalog",
        lambda *args, **kwargs: RetainedTracking(
            (),
            {
                str(Path(tracking_evidence.__file__).with_name(name)): file_digest(
                    Path(tracking_evidence.__file__).with_name(name)
                )
                for name in tracking_evidence.READER_FILES
            },
        ),
    )
    monkeypatch.setattr(refresh, "bounded_image_references", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        refresh,
        "read_motion_binding",
        lambda *args, **kwargs: DerivedMotion(
            ApproxPlan(
                (
                    ApproxCell(
                        Fraction(0),
                        Fraction(1),
                        Fraction(0),
                        700,
                        Fraction(0),
                        "face-motion-derived-candidate",
                        True,
                        700,
                    ),
                    ApproxCell(
                        Fraction(1),
                        Fraction(2),
                        Fraction(1),
                        700,
                        Fraction(0),
                        "face-motion-derived-candidate",
                        False,
                        700,
                    ),
                ),
                (),
                derived_policy,
            ),
            {},
        ),
    )
    monkeypatch.setattr(
        refresh,
        "read_image_binding",
        lambda *args, **kwargs: (
            VisibleArc(
                (
                    ArcCell(Fraction(0), Fraction(1), 700, 3, 0),
                    ArcCell(Fraction(1), Fraction(2), 300, 3, 0),
                ),
                {},
            )
            if backend == "approximate"
            else VisibleArc((ArcCell(Fraction(0), Fraction(2), 700, 3, 0),), {})
        ),
    )
    if fault == "pins":

        def forbidden_face_read(*args: object, **kwargs: object) -> None:
            raise AssertionError("Invalid measurement pins must fail before face collection")

        monkeypatch.setattr(refresh, "read_face_scale", forbidden_face_read)
    monkeypatch.setattr(refresh, "_source_identity", lambda *args: {})

    def resolve(*args: object) -> TrackingCalibration:
        if backend != "reference":
            raise AssertionError("Current pixels must not use historical reference calibration")
        return TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 900, 200, 400))

    monkeypatch.setattr(refresh, "resolve_reference", resolve)

    def make_layout(**kwargs: object) -> Measurer:
        assert kwargs["executable"] == evidence
        assert kwargs["long_executable"] == (evidence if long_capability else None)
        return Measurer()

    monkeypatch.setattr(refresh, "make_layout", make_layout)
    monkeypatch.setattr(
        refresh,
        "caption_template",
        lambda *args, **kwargs: CaptionTemplate(
            "test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM
        ),
    )
    library = ET.SubElement(root, "library")
    ET.SubElement(ET.SubElement(library, "event"), "project")
    if fault == "mutated":

        def mutate(self: Measurer) -> dict[str, str]:
            evidence.write_text("changed", encoding="utf-8")
            return {}

        monkeypatch.setattr(Measurer, "verify_inputs", mutate)
    if fault != "none":
        with pytest.raises(ReplayError):
            refresh.run(config, tmp_path / "output")
        return
    result = refresh.run(config, tmp_path / "output")
    assert result.name == "candidate.fcpxml"
    receipt = json.loads((result.parent / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["captionMediaClockUnchanged"] is True
    assert receipt["headNearVerified"] is False
    if backend in ("derived", "uncertain"):
        if backend == "uncertain":
            assert receipt["placementPolicy"] == "bounded-face-pixel-target-y-approximation-v1"
            assert (
                receipt["derivedFaceMotionEvidence"]["protocol"]
                == "derived-face-motion-uncertain-head-v1"
            )
        else:
            assert receipt["placementPolicy"] == "derived-face-pixel-motion-approximation"
        assert receipt["derivedFaceMotionEvidence"]["measuredCurrentUpperPixels"] is False
        assert receipt["retainedUpperPixelEvidence"] is None
        assert len(receipt["approximateHeadPlans"]) == 1
    elif backend == "approximate":
        assert receipt["placementPolicy"] == "current-face-y-offset-candidate"
        assert len(receipt["approximateHeadPlans"]) == 1
    else:
        assert receipt["placementPolicy"] == (
            "current-source-cell-upper-pixels"
            if image_capability
            else "current-observed-biaxial-scale-no-history"
        )


def test_current_pixels_override_face_box_deformation_without_changing_content(
    root: ET.Element, paired: FaceScaleSeries
) -> None:
    before = refresh._without_positions(root)
    faces = replace(
        paired,
        tops=replace(
            paired.tops, positions=tuple(replace(point, y=900) for point in paired.tops.positions)
        ),
        bottoms=replace(
            paired.bottoms,
            positions=(
                replace(paired.bottoms.positions[0], y=1300),
                replace(paired.bottoms.positions[1], y=1000),
            ),
        ),
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 300, 100),
        ),
    )
    reports = refresh.refresh(
        root,
        faces=faces,
        calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 900, 100, 400)),
        template=CaptionTemplate("test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM),
        measure=lambda text: (100, 100),
        current_upper_pixels=VisibleArc(
            (
                ArcCell(Fraction(0), Fraction(1), 700, 3, 0),
                ArcCell(Fraction(1), Fraction(2), 704, 3, 0),
            ),
            {},
        ),
    )
    assert refresh._without_positions(root) == before
    assert all(key.get("value") == "0 14.7916666667" for key in root.findall(".//keyframe"))
    assert reports[0]["provenance"] == "current-measured-upper-pixels-candidate"
    assert reports[0]["detectedFace"] is True
    assert reports[0]["completeHeadVerified"] is False


@pytest.mark.parametrize("missing_face", [False, True])
def test_current_pixels_stack_all_rows_from_one_bottom(
    paired: FaceScaleSeries, missing_face: bool
) -> None:
    root = ET.fromstring('<fcpxml><asset-clip offset="0s" start="0s" duration="1s"/></fcpxml>')
    clip = root.find("asset-clip")
    assert clip is not None
    for index, count in enumerate((1, 2, 3, 1)):
        title = ET.SubElement(clip, "title", offset=f"{index}/4s", start="3600s", duration="1/4s")
        for _layer in range(3):
            ET.SubElement(title, "text").text = "\n".join(("甲", "乙", "丙")[:count])
        ET.SubElement(title, "adjust-transform", position="0 999")
    before = refresh._without_positions(root)
    gap = MissingFaceRange(
        Fraction(0), Fraction(1), 30, Fraction(1), "leading-observation-backfill", "missing"
    )
    faces = (
        replace(
            paired,
            tops=replace(paired.tops, derived_ranges=(gap,)),
            bottoms=replace(paired.bottoms, derived_ranges=(gap,)),
        )
        if missing_face
        else paired
    )
    reports = refresh.refresh(
        root,
        faces=faces,
        calibration=TrackingCalibration(
            Fraction(1) if missing_face else Fraction(0), 400, 1, 0, 24, 1920, (0, 400, 100, 100)
        ),
        template=CaptionTemplate("test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM),
        measure=lambda text: (100, 100 + 120 * text.count("\n")),
        current_upper_pixels=VisibleArc(
            tuple(
                ArcCell(Fraction(index, 4), Fraction(index + 1, 4), 700, 3, 0) for index in range(4)
            ),
            {},
        ),
    )
    assert refresh._without_positions(root) == before
    assert all(key.get("value") == "0 14.7916666667" for key in root.findall(".//keyframe"))
    assert all(
        report["detectedFace"] is (not missing_face or Fraction(str(report["sourceTime"])) == 1)
        for report in reports
    )
    assert faces.tops.derived_ranges == ((gap,) if missing_face else ())


def test_current_pixels_follow_real_movement_reset_cut_and_never_hold_missing(
    root: ET.Element, paired: FaceScaleSeries
) -> None:
    template = CaptionTemplate("test", "uid", "file:///template", anchor=CaptionAnchor.BOTTOM)
    calibration = TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 900, 100, 400))
    with pytest.raises(ReplayError, match="complete measured title"):
        refresh.refresh(
            root,
            faces=paired,
            calibration=calibration,
            template=template,
            measure=lambda text: (100, 100),
            current_upper_pixels=VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {}),
        )
    with pytest.raises(ReplayError, match="distinct inputs"):
        refresh.refresh(
            root,
            faces=paired,
            calibration=calibration,
            template=template,
            measure=lambda text: (100, 100),
            current_upper_pixels=VisibleArc((), {}),
            visible_arc=VisibleArc((), {}),
        )
    clip = root.find("asset-clip")
    title = root.find(".//title")
    assert clip is not None and title is not None
    clip.set("duration", "1/2s")
    title.set("duration", "1/2s")
    second = ET.fromstring(ET.tostring(clip))
    second.set("offset", "1/2s")
    second.set("start", "1s")
    next_title = second.find("title")
    assert next_title is not None
    next_title.set("offset", "1s")
    root.append(second)
    refresh.refresh(
        root,
        faces=paired,
        calibration=calibration,
        template=template,
        measure=lambda text: (100, 100),
        current_upper_pixels=VisibleArc(
            (
                ArcCell(Fraction(0), Fraction(1, 2), 700, 3, 0),
                ArcCell(Fraction(1), Fraction(3, 2), 703, 3, 0),
            ),
            {},
        ),
    )
    first_key = root.findall(".//title")[0].find(".//keyframe")
    second_key = root.findall(".//title")[1].find(".//keyframe")
    assert first_key is not None and second_key is not None
    assert first_key.get("value") == "0 14.7916666667"
    assert second_key.get("value") == "0 14.6354166667"
    assert all(key.get("value", "").startswith("0 ") for key in root.findall(".//keyframe"))


def test_current_image_window_selection_excludes_distant_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ET.fromstring(
        "<fcpxml><library><event><project><sequence><spine>"
        '<asset-clip start="2s" duration="1/2s"><title offset="2s" duration="1/2s"/>'
        "</asset-clip></spine></sequence></project></event></library></fcpxml>"
    )
    paths = tuple(tmp_path / str(index) for index in range(3))
    jobs = {
        path: TrackingJob(
            path,
            "a",
            path,
            path,
            "a",
            "a",
            VideoFrameClock((index,), Fraction(1), 1),
            1080,
            1920,
            0.5,
        )
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(refresh, "face_references", lambda _path: [(path, path) for path in paths])
    monkeypatch.setattr(refresh, "load_job", lambda path: jobs[path])
    assert refresh.bounded_image_references(tmp_path, root) == [(paths[2], paths[2])]
    monkeypatch.setattr(refresh, "MAX_WINDOWS", 0)
    with pytest.raises(ReplayError, match="window budget"):
        refresh.bounded_image_references(tmp_path, root)


@pytest.mark.parametrize(
    "fault", ["none", "no_pixels", "not_exact", "below_face", "no_size", "held"]
)
def test_current_pixel_conversion_uses_actual_same_pts_face_only(
    paired: FaceScaleSeries, fault: str
) -> None:
    faces = replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 100, 100),
        ),
    )
    arc = VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {})
    if fault == "no_pixels":
        arc = VisibleArc((), {})
    elif fault == "not_exact":
        arc = VisibleArc((ArcCell(Fraction(1, 2), Fraction(1), 700, 3, 0),), {})
    elif fault == "below_face":
        arc = VisibleArc((ArcCell(Fraction(0), Fraction(1), 1000, 3, 0),), {})
    elif fault == "no_size":
        faces = paired
    elif fault == "held":
        gap = MissingFaceRange(
            Fraction(0), Fraction(1), 30, Fraction(1), "leading-observation-backfill", "missing"
        )
        faces = replace(
            faces,
            tops=replace(faces.tops, derived_ranges=(gap,)),
            bottoms=replace(faces.bottoms, derived_ranges=(gap,)),
        )
    if fault != "none":
        with pytest.raises(ReplayError):
            refresh.image_calibration(faces, arc)
    else:
        assert refresh.image_calibration(faces, arc) == TrackingCalibration(
            Fraction(0), 700, 1, 0, 24, 1920, (0, 900, 200, 400)
        )
        assert faces.scale_heights == ()


@pytest.mark.parametrize("fault", [False, True])
def test_cli_verify_binds_current_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: bool
) -> None:
    monkeypatch.setattr(artifact_json, "MAX_JSON_BYTES", 32)
    monkeypatch.setattr(refresh, "MAX_MEASUREMENT_RECEIPT_BYTES", 4096)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "candidate.fcpxml"
    output.write_text("<fcpxml/>", encoding="utf-8")
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "captionMediaClockUnchanged": not fault,
                "inputs": {str(config): file_digest(config)},
                "output": {"path": str(output), "sha256": file_digest(output)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["refresh", str(config), str(tmp_path), "--verify"])
    if fault:
        with pytest.raises(ReplayError, match="receipt"):
            refresh.main()
    else:
        refresh.main()


@pytest.mark.parametrize("fault", ["none", "source", "dimensions"])
def test_source_identity_rejects_different_media_or_proxy_geometry(
    tmp_path: Path, fault: str
) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"media")
    semantic = tmp_path / "semantic.xml"
    uri = (tmp_path / "other.mov" if fault == "source" else source).as_uri()
    width = "720" if fault == "dimensions" else "1080"
    semantic.write_text(
        f'<fcpxml><resources><asset><media-rep src="{uri}"/></asset>'
        f'<format id="r1" width="{width}" height="1920"/></resources>'
        '<library><event><project><sequence format="r1"/></project></event></library></fcpxml>',
        encoding="utf-8",
    )
    job = TrackingJob(
        source,
        "config",
        source,
        source,
        file_digest(source),
        "native",
        VideoFrameClock((0,), Fraction(1, 30), 1),
        1080,
        1920,
        1,
    )
    lineage: dict[str, object] = {
        "semanticXML": {"path": str(semantic), "sha256": file_digest(semantic)}
    }
    if fault != "none":
        with pytest.raises(ReplayError, match="identity"):
            refresh._source_identity(lineage, job)
    else:
        assert len(refresh._source_identity(lineage, job)) == 2
