import hashlib
import json
from pathlib import Path

import pytest

from replay import native_edit_parts as parts
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def request_data() -> dict[str, object]:
    return {
        "outputDirectory": "/synthetic/output",
        "outputName": "candidate.mp4",
        "recipe": {
            "video": {
                "width": 64,
                "height": 96,
                "frameRate": 30,
                "resizeMode": "fit",
                "encoding": {"codec": "hevc"},
            },
            "clips": [
                {
                    "sourcePath": "/synthetic/a.mp4",
                    "selection": {"startSeconds": 3.1, "durationSeconds": 1.2, "rate": 1},
                },
                {
                    "sourcePath": "/synthetic/b.mp4",
                    "selection": {"startSeconds": 8, "durationSeconds": 0.8, "rate": 1},
                },
            ],
            "additionalVideo": [
                {
                    "sourcePath": "/synthetic/alpha.mov",
                    "offsetSeconds": 0.2,
                    "opacity": 0.7,
                    "selection": {"startSeconds": 0.4, "durationSeconds": 1.4, "rate": 1},
                },
                {
                    "sourcePath": "/synthetic/tail.mov",
                    "offsetSeconds": 1.6,
                    "selection": {"startSeconds": 5, "durationSeconds": 0.4, "rate": 1},
                },
                {
                    "sourcePath": "/synthetic/head.mov",
                    "offsetSeconds": 0,
                    "selection": {"startSeconds": 0, "durationSeconds": 0.2, "rate": 1},
                },
            ],
        },
    }


def test_partition_conserves_cuts_layers_and_clocks(request_data: dict[str, object]) -> None:
    before = json.dumps(request_data)
    first, second = parts.partition(request_data)
    assert first[:2] == (0, 36)
    assert second[:2] == (36, 24)
    assert record(first[2]["recipe"])["clips"] == [
        {
            "sourcePath": "/synthetic/a.mp4",
            "selection": {"startSeconds": 3.1, "durationSeconds": 1.2, "rate": 1},
        }
    ]
    assert record(first[2]["recipe"])["additionalVideo"] == [
        {
            "sourcePath": "/synthetic/alpha.mov",
            "offsetSeconds": 0.2,
            "opacity": 0.7,
            "selection": {"startSeconds": 0.4, "durationSeconds": 1.0, "rate": 1},
        },
        {
            "sourcePath": "/synthetic/head.mov",
            "offsetSeconds": 0.0,
            "selection": {"startSeconds": 0.0, "durationSeconds": 0.2, "rate": 1},
        },
    ]
    assert record(second[2]["recipe"])["additionalVideo"] == [
        {
            "sourcePath": "/synthetic/alpha.mov",
            "offsetSeconds": 0.0,
            "opacity": 0.7,
            "selection": {"startSeconds": 1.4, "durationSeconds": 0.4, "rate": 1},
        },
        {
            "sourcePath": "/synthetic/tail.mov",
            "offsetSeconds": 0.4,
            "selection": {"startSeconds": 5.0, "durationSeconds": 0.4, "rate": 1},
        },
    ]
    assert json.dumps(request_data) == before


def test_no_layers_is_not_a_fabricated_caption_pass(request_data: dict[str, object]) -> None:
    recipe = record(request_data["recipe"])
    del recipe["additionalVideo"]
    request_data["recipe"] = recipe
    result = parts.partition(request_data)
    assert record(result[0][2]["recipe"])["additionalVideo"] == []


@pytest.mark.parametrize("value", [True, -1, "0", float("nan"), float("inf"), 86401])
def test_invalid_seconds(value: object) -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        parts._time(value)


@pytest.mark.parametrize("value", [0, -1, True, 2, "1"])
def test_retiming_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="unretimed"):
        parts._selection({"startSeconds": 0, "durationSeconds": 1, "rate": value})


def test_empty_selection() -> None:
    with pytest.raises(ValueError, match="positive"):
        parts._selection({"startSeconds": 0, "durationSeconds": 0, "rate": 1})


@pytest.mark.parametrize(
    "level,key,value",
    [
        ("root", "unknown", True),
        ("recipe", "additionalAudio", []),
        ("recipe", "clips", []),
        ("recipe", "clips", "bad"),
        ("recipe", "additionalVideo", "bad"),
        ("video", "masks", []),
        ("video", "frameRate", True),
        ("video", "frameRate", 0),
        ("video", "frameRate", 61),
    ],
)
def test_unsupported_operations_fail_closed(
    request_data: dict[str, object], level: str, key: str, value: object
) -> None:
    recipe = record(request_data["recipe"])
    video = record(recipe["video"])
    target = {"root": request_data, "recipe": recipe, "video": video}[level]
    target[key] = value
    recipe["video"] = video
    request_data["recipe"] = recipe
    with pytest.raises(ValueError):
        parts.partition(request_data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("transitionInSeconds", 0.1),
        ("unknown", True),
        ("selection", {"startSeconds": 0, "durationSeconds": 0.01, "rate": 1}),
        ("selection", {"startSeconds": 0, "durationSeconds": 601, "rate": 1}),
    ],
)
def test_invalid_cut(request_data: dict[str, object], key: str, value: object) -> None:
    recipe = record(request_data["recipe"])
    recipe["clips"] = [
        {
            "sourcePath": "/synthetic/source.mp4",
            "selection": {"startSeconds": 0, "durationSeconds": 2, "rate": 1},
            key: value,
        }
    ]
    request_data["recipe"] = recipe
    with pytest.raises(ValueError):
        parts.partition(request_data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("offsetSeconds", 2),
        ("offsetSeconds", 0.01),
        ("unknown", 1),
        ("selection", {"startSeconds": 0.01, "durationSeconds": 1, "rate": 1}),
    ],
)
def test_invalid_layer(request_data: dict[str, object], key: str, value: object) -> None:
    recipe = record(request_data["recipe"])
    recipe["additionalVideo"] = [
        {
            "sourcePath": "/synthetic/alpha.mov",
            "selection": {"startSeconds": 0, "durationSeconds": 1, "rate": 1},
            "offsetSeconds": 0,
            key: value,
        }
    ]
    request_data["recipe"] = recipe
    with pytest.raises(ValueError):
        parts.partition(request_data)


def test_export_is_pinned_private_and_non_overwriting(
    tmp_path: Path, request_data: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    media_root = tmp_path / "independent-media"
    request_data["outputDirectory"] = str(media_root)
    request.write_text(json.dumps(request_data), encoding="utf-8")
    output = tmp_path / "parts"
    digest = file_digest(request)
    monkeypatch.setattr(
        "sys.argv", ["native-edit-parts", str(request), str(output), "--sha256", digest]
    )
    parts.main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["renderVerified"] is False
    assert manifest["losslessAssemblyVerified"] is False
    assert manifest["deliveryComplete"] is False
    join_plan = output / "join-plan.json"
    assert json.loads(join_plan.read_text(encoding="utf-8")) == [
        {"durationFrames": 36},
        {"durationFrames": 24},
    ]
    assert manifest["joinPlan"]["sha256"] == file_digest(join_plan)
    assert manifest["parts"][1]["offsetFrame"] == 36
    assert manifest["parts"][1]["frames"] == 24
    first = output / "part-0000" / "request.json"
    data = json.loads(first.read_text(encoding="utf-8"))
    media_directory = Path(data["outputDirectory"])
    assert media_directory.parent.parent == media_root
    assert media_directory.name == "part-0000"
    assert not media_root.exists()
    second_export = tmp_path / "other-parts"
    parts.export(request, second_export, sha256=digest)
    second_data = json.loads(
        (second_export / "part-0000" / "request.json").read_text(encoding="utf-8")
    )
    assert second_data["outputDirectory"] != data["outputDirectory"]
    media_directory.parent.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        parts.export(request, tmp_path / "parts", sha256=digest)
    assert first.stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    assert file_digest(request) == digest
    with pytest.raises(FileExistsError):
        parts.export(request, output, sha256=digest)
    with pytest.raises(ValueError):
        parts.export(request, tmp_path / "stale", sha256="0" * 64)
    monkeypatch.setattr(parts, "file_digest", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="changed"):
        parts.export(request, tmp_path / "changed", sha256=digest)
    assert not (tmp_path / "changed").exists()


def test_export_refuses_existing_media_namespace(
    tmp_path: Path, request_data: dict[str, object]
) -> None:
    media = tmp_path / "media"
    request_data["outputDirectory"] = str(media)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(request_data), encoding="utf-8")
    output = tmp_path / "parts"
    namespace = hashlib.sha256(str(output).encode("utf-8")).hexdigest()
    existing = media / f"parts-{namespace}"
    existing.mkdir(parents=True)
    evidence = existing / "candidate.mp4"
    evidence.write_bytes(b"unchanged synthetic media")
    with pytest.raises(FileExistsError):
        parts.export(request, output, sha256=file_digest(request))
    assert evidence.read_bytes() == b"unchanged synthetic media"
    assert not output.exists()


def test_export_keeps_configuration_inside_repository(
    tmp_path: Path, request_data: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.chdir(repository)
    request = repository / "request.json"
    request.write_text(json.dumps(request_data), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the repository"):
        parts.export(request, tmp_path / "outside-repository", sha256=file_digest(request))
    assert not (tmp_path / "outside-repository").exists()


@pytest.mark.parametrize("destination", [None, "", "relative"])
def test_export_rejects_invalid_media_root(
    tmp_path: Path, request_data: dict[str, object], destination: object
) -> None:
    request_data["outputDirectory"] = destination
    request = tmp_path / "request.json"
    request.write_text(json.dumps(request_data), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute media"):
        parts.export(request, tmp_path / "parts", sha256=file_digest(request))
    assert not (tmp_path / "parts").exists()


def test_export_rejects_redirected_media_root(
    tmp_path: Path, request_data: dict[str, object]
) -> None:
    permitted = tmp_path / "permitted"
    permitted.mkdir()
    (permitted / "redirect").symlink_to(tmp_path / "outside-permitted")
    request_data["outputDirectory"] = str(permitted / "redirect" / "media")
    request = tmp_path / "request.json"
    request.write_text(json.dumps(request_data), encoding="utf-8")
    with pytest.raises(ValueError, match="Linked"):
        parts.export(request, tmp_path / "parts", sha256=file_digest(request))
    assert not (tmp_path / "parts").exists()
