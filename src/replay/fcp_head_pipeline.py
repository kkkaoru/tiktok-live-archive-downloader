"""Connect verified faces, cached native glyphs and timed paging to semantic FCPXML.

The semantic output retains the original source for a subsequent verified clean
source substitution; it is never itself marked ready for production rendering.
"""

import argparse
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import reflow_producer_evidence
from replay.artifact_json import caption_reflow_json
from replay.caption_measure_cache import CachedCaptionLayout
from replay.caption_native_paging import assemble
from replay.caption_scale import read_scale
from replay.caption_token_sources import load_sources
from replay.core import ReplayError
from replay.delivery_captions import read_cuts
from replay.fcp_caption_retarget import (
    Cue,
    build,
    face_references,
    read_reflow,
    selected_spans,
    tracking_receipt,
)
from replay.fcp_caption_visibility import apply_binding, audit_xml, read_binding
from replay.fcp_head_near import read_face_scale
from replay.fcp_head_quality import title_geometry_provenance
from replay.fcp_measured import CaptionAnchor
from replay.fcp_part_assembly import _json, _pin, _save
from replay.fcp_project import caption_template, probe_media
from replay.frame_cell_extent import terminal_cell_receipt
from replay.head_height_paging import (
    HeadroomPreference,
    make_layout,
    minimum_headroom,
    repage_segment,
    single_line_height,
)
from replay.head_reference import resolve_reference
from replay.motion_stages import _integer
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_retained import verify_catalog, write_receipt


def _text_bounds(
    part: dict[str, object], layout: CachedCaptionLayout
) -> dict[str, tuple[float, float]]:
    rows = part.get("captions")
    if not isinstance(rows, list):
        raise ReplayError("Expected prepared caption rows")
    result: dict[str, tuple[float, float]] = {}
    for value in rows:
        text = record(value).get("text")
        if not isinstance(text, str):
            raise ReplayError("Expected prepared caption text")
        result[text] = layout.measure(text)
    return result


def _clean_config(
    value: object,
    *,
    semantic: Path,
    voice: dict[str, object],
    output: Path,
    terminal_frame_cell: bool = False,
) -> dict[str, str]:
    """Bind a newly produced semantic XML to already pinned cleanup inputs."""
    clean = record(value)
    names = ("assemblyManifest", "assemblyReceipt", "cleanIndex", "voiceReceipt")
    paths = {name: _pin(clean.get(name)) for name in names}
    _save(
        output,
        {
            "version": 1,
            "terminalFrameCell": terminal_frame_cell,
            "removeSourceMasks": True,
            "voice": voice,
            "semanticXML": {"path": str(semantic), "sha256": file_digest(semantic)},
            **{name: clean[name] for name in names},
        },
    )
    _save(
        output.with_name("clean-transform-binding.json"),
        {
            **{name: clean[name] for name in names},
            "version": 1,
            "consumer": {
                "path": str(output),
                "sha256": file_digest(output),
                "kind": "transformed-source-config",
            },
        },
    )
    return {str(path): file_digest(path) for path in paths.values()}


def prepare(config_path: Path, output: Path) -> Path:
    """Create a new bounded, reusable candidate run; every input is pinned."""
    config = _json(config_path)
    if config.get("version") != 1 or output.exists() or output.absolute() != output.resolve():
        raise ReplayError("Expected version 1 and a new canonical candidate directory")
    terminal = config.get("terminalFrameCell", False)
    if not isinstance(terminal, bool):
        raise ReplayError("Terminal-frame-cell policy must be explicit boolean")
    collect = config.get("collectLayoutFailures", False)
    if not isinstance(collect, bool):
        raise ReplayError("Layout failure collection must be explicit boolean")
    resume = record(config.get("resumeParts", {}))
    names = (
        "faceIndex",
        "cuts",
        "reflow",
        "segmentIndex",
        "nativeExecutable",
        "measurementReference",
        "template",
        "voice",
    )
    paths = {name: _pin(config.get(name)) for name in names}
    if "longExecutable" in config:
        paths["longExecutable"] = _pin(config["longExecutable"])
    if "producerEvidence" in config:
        paths["producerEvidence"] = _pin(config["producerEvidence"])
    # Reject oversized/malformed reflow before expensive full-source face checks.
    old = caption_reflow_json(paths["reflow"])
    code = tuple(
        Path(__file__).with_name(name)
        for name in (
            "fcp_head_pipeline.py",
            "fcp_head_near.py",
            "fcp_screen_stack.py",
            "fcp_head_quality.py",
            "head_near_placement.py",
            "head_reference.py",
            "head_height_paging.py",
            "caption_measure_cache.py",
            "caption_native_paging.py",
            "caption_indivisible_fit.py",
            "caption_anchor_fit.py",
            "caption_joint_fit.py",
            "caption_scale.py",
            "caption_long_layout.py",
            "fcp_caption_retarget.py",
            "fcp_caption_stability.py",
            "caption_y_stability.py",
            "fcp_project.py",
            "fcp_pre_layout.py",
            "motion_pre_layout.py",
            "caption_style_defaults.py",
            "artifact_json.py",
            "source_caption_binding.py",
            "fcp_transformed_source.py",
            "fcp_part_assembly.py",
            "frame_cell_extent.py",
            "fcp_caption_visibility.py",
        )
    )
    bound = {
        str(path.resolve()): file_digest(path) for path in (config_path, *paths.values(), *code)
    }
    output.mkdir(parents=True, mode=0o700)
    references = face_references(paths["faceIndex"])
    retained = None
    retained_receipt = None
    if "retainedTracking" in config:
        descriptor = _pin(config["retainedTracking"])
        retained = verify_catalog(descriptor, sha256=file_digest(descriptor), references=references)
        retained_path = output / "retained-tracking-receipt.json"
        write_receipt(retained_path, retained)
        retained_receipt = {"path": str(retained_path), "sha256": file_digest(retained_path)}
        bound.update(retained.inputs)
        bound[str(retained_path)] = file_digest(retained_path)
    job, faces = read_face_scale(
        references, missing_policy="hold", seed_overlap="next", retained=retained
    )
    calibration = resolve_reference(config, faces)
    reference, head = calibration.reference_time, calibration.source_head_top
    spans = selected_spans(read_cuts(paths["cuts"]), rate=30, source_window=None)
    faces = faces.with_current_scale(reference_time=reference)
    # Fresh producers must match current hashes. Only an explicitly supplied
    # historical manifest can authorize the existing code-only exception.
    read_reflow(
        paths["reflow"],
        spans,
        source=job.source,
        source_sha256=job.source_hash,
        producer_evidence=paths.get("producerEvidence"),
    )
    index = _json(paths["segmentIndex"])
    entries = index.get("segments")
    if index.get("rate") != 30 or not isinstance(entries, list) or not 1 <= len(entries) <= 512:
        raise ReplayError("Expected bounded recognized output-clock segment index")
    token_sources, token_source_pins = (
        load_sources(_pin(config["tokenSources"]), entries)
        if "tokenSources" in config
        else ((), {})
    )
    bound.update(token_source_pins)
    pins = record(old.get("inputs"))
    visibility = (
        read_binding(config["candidateVisibility"], reflow_inputs=pins)
        if "candidateVisibility" in config
        else None
    )
    if visibility is not None:
        bound.update(visibility.inputs)
    historical = (
        reflow_producer_evidence.read(
            paths["producerEvidence"], artifact=paths["reflow"], inputs=pins
        )
        if "producerEvidence" in paths
        else {}
    )
    bound.update(
        {
            name: digest
            for name, digest in pins.items()
            if isinstance(digest, str) and name not in historical
        }
    )
    if pins.get(str(paths["segmentIndex"])) != file_digest(paths["segmentIndex"]):
        raise ReplayError("Caption segment index differs from verified reflow")
    parts: list[dict[str, object]] = []
    layouts: list[CachedCaptionLayout] = []
    measurements: dict[str, tuple[float, float]] = {}
    reports: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for value in entries:
        row = record(value)
        number = _integer(row.get("segment"))
        part_path = paths["reflow"].parent / f"part-{number:04d}.json"
        if pins.get(str(part_path)) != file_digest(part_path):
            raise ReplayError("Paged segment differs from verified reflow")
        part = _json(part_path)
        part_inputs = record(part.get("inputs"))
        if any(not isinstance(digest, str) for digest in part_inputs.values()):
            raise ReplayError("Invalid measured segment input hashes")
        layout = make_layout(
            reference=paths["measurementReference"],
            executable=paths["nativeExecutable"],
            output=output / f"metrics-{number:04d}",
            pinned_inputs={
                key: digest for key, digest in part_inputs.items() if isinstance(digest, str)
            },
            cached_directory=paths["reflow"].parent / f"metrics-{number:04d}",
            long_executable=paths.get("longExecutable"),
        )
        offset = Fraction(_integer(row.get("offsetFrame")), 30)
        end = offset + Fraction(_integer(row.get("frames")), 30)
        height = minimum_headroom(
            faces,
            spans,
            start=offset,
            end=end,
            allow_terminal_cell=terminal,
            preference=HeadroomPreference(reference, float(head), single_line_height(part, layout)),
        )
        layouts.append(layout)
        try:
            result, changed = repage_segment(
                captions=paths["segmentIndex"].parent,
                segment=row,
                part=part,
                layout=layout,
                maximum_height=height,
                rate=30,
                token_sources=token_sources,
            )
        except ValueError as error:
            if not collect or not str(error).startswith(
                (
                    "Native token exceeds display budget;",
                    "Native token timing is too coarse for focused captions",
                )
            ):
                raise
            failed = output / f"failure-{number:04d}.json"
            _save(
                failed,
                {
                    "version": 1,
                    "segment": number,
                    "error": str(error),
                    "maximumHeight": height,
                    "candidateOnly": True,
                    "complete": False,
                    "inputs": layout.verify_inputs()
                    | {
                        str(config_path.resolve()): file_digest(config_path),
                        str(part_path): file_digest(part_path),
                    },
                },
            )
            failed_digest = file_digest(failed)
            bound[str(failed)] = failed_digest
            failures.append({"segment": number, "path": str(failed), "sha256": failed_digest})
            print(f"FAILED_HEAD_HEIGHT_SEGMENT {number}: {error}", flush=True)
            continue
        reused: object = None
        if str(number) in resume:
            saved_part = _pin(resume[str(number)])
            previous = resume.pop(str(number))
            if _json(saved_part) == result:
                reused = previous
            bound[str(saved_part)] = file_digest(saved_part)
        _save(output / f"part-{number:04d}.json", result)
        parts.append(result)
        measurements.update(_text_bounds(result, layout))
        reports.append(
            {
                "segment": number,
                "maximumHeight": height,
                "repaged": changed,
                "reusedPreparedPart": reused,
            }
        )
        print(f"HEAD_HEIGHT_SEGMENT {number} repaged={changed}", flush=True)
    if failures:
        _verify_inputs(layouts, bound)
        _save(
            output / "layout-failures.json",
            {
                "version": 1,
                "complete": False,
                "deliveryComplete": False,
                "readyForRender": False,
                "failures": failures,
                "successfulSegments": reports,
                "inputs": bound,
            },
        )
        raise ReplayError("Head layout failures collected; no complete semantic XML was published")
    if resume:
        raise ReplayError("Prepared segment references were not consumed")
    rows = assemble(tuple(parts), rate=30)
    if visibility is not None:
        rows = apply_binding(rows, visibility)
    cues: list[Cue] = []
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            raise ReplayError("Expected assembled caption text")
        cues.append(
            Cue(Fraction(str(row["start"])), Fraction(str(row["end"])), text, read_scale(row))
        )
    semantic = build(
        tuple(cues),
        spans,
        source=job.source,
        media=probe_media(job.source),
        template=caption_template(paths["template"], anchor=CaptionAnchor.BOTTOM),
        faces=faces.bottoms,
        calibration=calibration,
        rate=30,
        name="Scale-aware caption candidate",
        voice=(paths["voice"], probe_media(paths["voice"])),
        scale_faces=faces,
        measure_caption=measurements.__getitem__,
        terminal_frame_cell=terminal,
    )
    destination = output / "semantic.fcpxml"
    with destination.open("xb") as stream:
        stream.write(semantic)
    visibility_audit = None if visibility is None else audit_xml(destination, visibility)
    if "cleanSource" in config:
        bound.update(
            _clean_config(
                config["cleanSource"],
                semantic=destination,
                voice=record(config["voice"]),
                output=output / "clean-transform-config.json",
                terminal_frame_cell=terminal,
            )
        )
    _verify_inputs(layouts, bound)
    _save(
        output / "receipt.json",
        {
            **tracking_receipt(faces.bottoms, spans),
            "inputs": bound,
            "segments": reports,
            "semanticXML": {"path": str(destination), "sha256": file_digest(destination)},
            **({} if retained_receipt is None else {"retainedTracking": retained_receipt}),
            **({} if visibility_audit is None else {"candidateVisibility": visibility_audit}),
            "headPolicy": "face-scale-derived-and-canvas-fitted",
            "titleGeometry": title_geometry_provenance(ET.fromstring(semantic), faces.tops),
            "faceScale": {
                "policy": "current-observed-biaxial-scale-no-history",
                "derivedSamples": len(faces.scale_heights),
                "rawGeometryUnchanged": True,
                "missingObservationsRecovered": False,
            },
            "headReference": {
                "policy": config.get("headReferencePolicy", "explicit"),
                "sourceTime": str(reference),
                "estimatedHeadTop": head,
                "faceTop": faces.observed_reference(reference).top,
                "faceBottom": faces.observed_reference(reference).bottom,
                "semanticHeadDetected": False,
                "referenceROI": calibration.roi,
                "referenceROIMeaning": "full-width-measured-vertical-band"
                if config.get("headReferencePolicy") == "automatic-face-relative"
                else "explicit-reference-rectangle",
            },
            "footprint": "native-glyph-border-rim-plus-observed-Motion-line-advance",
            "shadowFootprintVerified": False,
            "readyForRender": False,
            "requiredNextStage": "verified-clean-source-substitution",
            "terminalFrameCell": terminal_cell_receipt(
                physical_end=faces.tops.coverage_end, frame_duration=Fraction(1, 30)
            )
            if terminal
            else None,
        },
    )
    return destination


def _verify_inputs(layouts: list[CachedCaptionLayout], bound: dict[str, str]) -> None:
    for layout in layouts:
        bound.update(layout.verify_inputs())
    if any(file_digest(Path(path)) != digest for path, digest in bound.items()):
        raise ReplayError("Head placement inputs changed during preparation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(prepare(args.config, args.output.absolute()))


if __name__ == "__main__":
    main()
