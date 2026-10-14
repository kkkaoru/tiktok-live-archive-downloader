import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from replay.caption_native_paging import read_tokens
from replay.caption_token_sources import TokenSource, combined_path, load_sources, main
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def source_map(tmp_path: Path) -> Path:
    tokens = tmp_path / "refined.json"
    tokens.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "tokens": [{"text": "refined", "start": 0.2, "end": 0.4, "utterance": "u"}],
            }
        )
    )
    reference = {"path": str(tokens), "sha256": file_digest(tokens)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "tokens": reference}))
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "replacements": [
                    {
                        "segment": 2,
                        "job": 8,
                        "tokens": reference,
                        "adoptedManifest": {"path": str(manifest), "sha256": file_digest(manifest)},
                    }
                ],
            }
        )
    )
    return path


def test_adopted_tokens_replace_only_explicit_job(source_map: Path) -> None:
    row: dict[str, object] = {"segment": 2, "jobs": [8]}
    sources, pins = load_sources(source_map, [row])
    source = sources[0]
    bindings: dict[str, object] = {
        str(source.tokens): source.token_digest,
        str(source.manifest): source.manifest_digest,
    }
    tokens, used = read_tokens(source_map.parent, row, bindings, sources)
    assert tokens[0].text == "refined"
    assert tokens[0].start == 0.2
    assert len(used) == 2
    assert len(pins) == 2
    with pytest.raises(ReplayError, match="both"):
        source.verify({})


@pytest.mark.parametrize(
    "mutation",
    ["version", "empty", "duplicate", "job", "relative", "digest", "manifest", "clock", "changed"],
)
def test_reject_invalid_adoption(source_map: Path, mutation: str) -> None:
    raw = record(json.loads(source_map.read_text()))
    values = raw["replacements"]
    assert isinstance(values, list)
    replacement = record(values[0])
    if mutation == "version":
        raw["version"] = 2
    elif mutation == "empty":
        raw["replacements"] = []
    elif mutation == "duplicate":
        values.append(values[0])
    elif mutation == "job":
        replacement["job"] = 9
    elif mutation == "relative":
        record(replacement["tokens"])["path"] = "refined.json"
    elif mutation == "digest":
        record(replacement["tokens"])["sha256"] = "x" * 64
    elif mutation == "manifest":
        path = source_map.with_name("manifest.json")
        path.write_text('{"version":1,"tokens":{}}')
        record(replacement["adoptedManifest"])["sha256"] = file_digest(path)
    elif mutation == "clock":
        path = source_map.with_name("refined.json")
        path.write_text('{"version":1,"clock":"source","tokens":[]}')
        ref = record(replacement["tokens"])
        ref["sha256"] = file_digest(path)
        manifest = source_map.with_name("manifest.json")
        manifest.write_text(json.dumps({"version": 1, "tokens": ref}))
        record(replacement["adoptedManifest"])["sha256"] = file_digest(manifest)
    elif mutation == "changed":
        source_map.with_name("refined.json").write_text("changed")
    source_map.write_text(json.dumps(raw))
    with pytest.raises((ReplayError, ValueError)):
        sources, _ = load_sources(source_map, [{"segment": 2, "jobs": [8]}])
        source = sources[0]
        read_tokens(
            source_map.parent,
            {"segment": 2, "jobs": [8]},
            {str(source.tokens): source.token_digest, str(source.manifest): source.manifest_digest},
            sources,
        )


def test_invalid_membership(source_map: Path) -> None:
    with pytest.raises(ReplayError, match="membership"):
        load_sources(source_map, [{"segment": 2, "jobs": None}])


@pytest.fixture
def overlay(source_map: Path) -> tuple[Path, Path, tuple[TokenSource, ...]]:
    sources, _ = load_sources(source_map, [{"segment": 2, "jobs": [8]}])
    source = sources[0]
    original = source_map.with_name("combined-0002.json")
    old = {"inputs": {"retained": "a" * 64}, "captions": [{"text": "old"}], "duration": 6}
    original.write_text(json.dumps(old))
    target = source_map.with_name("overlay.json")
    old["inputs"] = {
        "retained": "a" * 64,
        str(original): file_digest(original),
        str(source.tokens): source.token_digest,
        str(source.manifest): source.manifest_digest,
    }
    target.write_text(json.dumps(old))
    mapping = json.loads(source_map.read_text())
    row = mapping["replacements"][0]
    row["combined"] = {"path": str(target), "sha256": file_digest(target)}
    row["originalCombined"] = {"path": str(original), "sha256": file_digest(original)}
    source_map.write_text(json.dumps(mapping))
    updated, pins = load_sources(source_map, [{"segment": 2, "jobs": [8]}])
    assert pins[str(original)] == file_digest(original)
    assert pins[str(target)] == file_digest(target)
    return original, target, updated


def test_overlay_only_adds_pins_and_preserves_original(
    overlay: tuple[Path, Path, tuple[TokenSource, ...]],
    source_map: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, target, sources = overlay
    before = original.read_bytes()
    assert combined_path(original, 2, sources) == target
    assert combined_path(original, 3, sources) == original
    assert original.read_bytes() == before
    (source_map.parent / "segments.json").write_text(
        json.dumps({"segments": [{"segment": 2, "jobs": [8]}]})
    )
    monkeypatch.setattr(sys, "argv", ["check", str(source_map), str(source_map.parent)])
    main()


@pytest.mark.parametrize("mutation", ["captions", "duration", "extra", "missing", "changed"])
def test_overlay_rejects_content_changes(
    overlay: tuple[Path, Path, tuple[TokenSource, ...]], mutation: str
) -> None:
    original, target, sources = overlay
    data = json.loads(target.read_text())
    if mutation == "missing":
        data["inputs"].pop(str(sources[0].tokens))
    elif mutation == "changed":
        data["inputs"]["retained"] = "b" * 64
    else:
        data[mutation] = 42
    target.write_text(json.dumps(data))
    updated = (replace(sources[0], combined=(target, file_digest(target))),)
    with pytest.raises(ReplayError, match="changed captions"):
        combined_path(original, 2, updated)


def test_overlay_rejects_unpaired_conflicting_and_wrong_original(
    overlay: tuple[Path, Path, tuple[TokenSource, ...]], source_map: Path
) -> None:
    original, target, sources = overlay
    source = sources[0]
    for members in (
        (replace(source, original_combined=None),),
        (replace(source, original_combined=(target, file_digest(target))),),
        (source, replace(source, combined=(target, "a" * 64))),
    ):
        with pytest.raises(ReplayError):
            combined_path(original, 2, members)
    data = json.loads(source_map.read_text())
    del data["replacements"][0]["originalCombined"]
    source_map.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="paired"):
        load_sources(source_map, [{"segment": 2, "jobs": [8]}])


def test_overlay_cannot_rebind_existing_pin(
    overlay: tuple[Path, Path, tuple[TokenSource, ...]],
) -> None:
    original, _, sources = overlay
    data = json.loads(original.read_text())
    data["inputs"][str(sources[0].tokens)] = "b" * 64
    original.write_text(json.dumps(data))
    updated = (replace(sources[0], original_combined=(original, file_digest(original))),)
    with pytest.raises(ReplayError, match="replace existing evidence"):
        combined_path(original, 2, updated)


@pytest.mark.parametrize("invalid", [False, True])
def test_read_only_cli(source_map: Path, monkeypatch: pytest.MonkeyPatch, invalid: bool) -> None:
    root = source_map.parent
    (root / "segments.json").write_text(
        json.dumps({"segments": None if invalid else [{"segment": 2, "jobs": [8]}]})
    )
    sources, _ = load_sources(source_map, [{"segment": 2, "jobs": [8]}])
    source = sources[0]
    (root / "combined-0002.json").write_text(
        json.dumps(
            {
                "inputs": {
                    str(source.tokens): source.token_digest,
                    str(source.manifest): source.manifest_digest,
                }
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["check", str(source_map), str(root)])
    if invalid:
        with pytest.raises(ReplayError, match="index"):
            main()
    else:
        main()
