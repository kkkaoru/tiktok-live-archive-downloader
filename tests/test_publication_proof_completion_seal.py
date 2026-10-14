import hashlib
from pathlib import Path

import pytest

from replay import publication_proof_completion_seal as proof
from replay.core import ReplayError
from replay.stages import Stage, _fingerprint, file_digest


def source(path: Path, text: str) -> proof.SourceBytes:
    path.write_text(text, encoding="utf-8")
    return proof.SourceBytes(path, hashlib.sha256(text.encode()).hexdigest())


def review(tmp_path: Path, old: str, new: str, consumer: str) -> proof.SourceReview:
    historical = source(tmp_path / "old.py", old)
    current = source(tmp_path / "new.py", new)
    caller = source(tmp_path / "caller.py", consumer)
    return proof.review_source_transition(
        module="replay.fixture",
        historical=historical,
        current=current,
        authenticated_historical_sha256=historical.sha256,
        consumers=(caller,),
    )


def test_actual_unused_function_body_transition(tmp_path: Path) -> None:
    result = review(
        tmp_path,
        "def used():\n return 1\ndef unused():\n return 2\n",
        "def used():\n return 1\ndef unused():\n return 3\n",
        "from replay import fixture as f\nf.used()\n",
    )
    assert result.used_symbols == ("used",)
    assert result.changed_unused_symbols == ("unused",)
    assert result.json()["historicalBytesExecuted"] is False
    assert result.historical_sha256 != result.current_sha256
    assert result.json()["module"] == "replay.fixture"


def test_imported_used_function_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="actually referenced"):
        review(
            tmp_path,
            "def used():\n return 1",
            "def used():\n return 2",
            "from replay.fixture import used as call\ncall()",
        )


def test_transitively_used_helper_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="actually referenced"):
        review(
            tmp_path,
            "def used():\n return helper()\ndef helper():\n return 1",
            "def used():\n return helper()\ndef helper():\n return 2",
            "import replay.fixture as f\nf.used()",
        )


def test_constant_and_import_change_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="Module execution"):
        review(
            tmp_path, "import math\nVALUE=1", "import math\nVALUE=2", "from replay import fixture"
        )


@pytest.mark.parametrize(
    "caller",
    [
        "from replay.fixture import *",
        "import replay.fixture",
        "from replay import fixture as f\npass_to_elsewhere(f)",
        "from replay import fixture as f\ngetattr(f, 'used')()",
    ],
)
def test_unresolved_consumer_contract_rejected(tmp_path: Path, caller: str) -> None:
    with pytest.raises(ReplayError, match="stronger.*review|static consumer"):
        review(tmp_path, "def used():\n return 1", "def used():\n return 1", caller)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("def unused(x=1):\n return 1", "def unused(x=2):\n return 2"),
        ("@decorate\ndef unused():\n return 1", "@decorate\ndef unused():\n return 2"),
        ("class Unused:\n value=1", "class Unused:\n value=2"),
        ("def unused():\n return 1", "# no definitions\n"),
        ("async def unused():\n return 1", "def unused():\n return 1"),
    ],
)
def test_definition_time_effects_rejected(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ReplayError, match="Definition|Decorated"):
        review(tmp_path, old, new, "from replay import fixture")


def test_reflection_in_used_function_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="Dynamic"):
        review(
            tmp_path,
            "def used():\n return globals()",
            "def used():\n return globals()",
            "from replay.fixture import used\nused()",
        )


def test_duplicate_definition_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="duplicate"):
        review(
            tmp_path,
            "def used():\n return 1\ndef used():\n return 2",
            "# valid\n",
            "from replay import fixture",
        )


def test_invalid_python_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="valid Python"):
        review(tmp_path, "def ???", "# valid\n", "from replay import fixture")


def test_invalid_digest_and_historical_identity(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="exact source"):
        proof.SourceBytes(tmp_path / "not-read", "invalid").read()
    historical = source(tmp_path / "old.py", "VALUE=1")
    with pytest.raises(ReplayError, match="historical identity"):
        proof.review_source_transition(
            module="replay.fixture",
            historical=historical,
            current=historical,
            authenticated_historical_sha256="0" * 64,
            consumers=(),
        )


def test_actual_old_bytes_missing_or_tampered_rejected(tmp_path: Path) -> None:
    old = source(tmp_path / "old.py", "VALUE=1")
    old.path.write_text("VALUE=2", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        old.read()
    old.path.unlink()
    with pytest.raises(FileNotFoundError):
        old.read()


def test_unchanged_global_reference_and_recursive_function(tmp_path: Path) -> None:
    result = review(
        tmp_path,
        "VALUE=1\ndef used():\n return used() or VALUE",
        "VALUE=1\ndef used():\n return used() or VALUE",
        "from replay.fixture import VALUE, used\nused() or VALUE",
    )
    assert result.used_symbols == ("VALUE", "used")
    assert result.changed_unused_symbols == ()


def test_true_historical_completed_fingerprint(tmp_path: Path) -> None:
    input_path = tmp_path / "reader.py"
    output = tmp_path / "receipt.json"
    input_path.write_text("old reader", encoding="utf-8")
    output.write_text("real completed receipt", encoding="utf-8")
    stage = Stage(
        name="adopt",
        command=("python", "normal"),
        verify_command=("python", "verify"),
        inputs=(input_path,),
        outputs=(output,),
        timeout_seconds=30,
    )
    historical = {str(input_path): file_digest(input_path)}
    fingerprint = _fingerprint(stage)
    input_path.write_text("new separately reviewed reader", encoding="utf-8")
    assert proof.historical_stage_fingerprint(stage, historical) == fingerprint
    assert _fingerprint(stage) != fingerprint
    proof.verify_completed_stage(
        stage,
        checkpoint={
            "status": "complete",
            "fingerprint": fingerprint,
            "outputs": {str(output): file_digest(output)},
        },
        authenticated_inputs=historical,
    )


@pytest.mark.parametrize("status", ["running", "failed"])
def test_running_or_failed_never_becomes_success(tmp_path: Path, status: str) -> None:
    input_path = tmp_path / "input"
    output = tmp_path / "output"
    input_path.write_bytes(b"input")
    output.write_bytes(b"output")
    stage = Stage(
        name="adopt",
        command=("normal",),
        verify_command=("verify",),
        inputs=(input_path,),
        outputs=(output,),
        timeout_seconds=30,
    )
    with pytest.raises(ReplayError, match="Genuine completed"):
        proof.verify_completed_stage(
            stage,
            checkpoint={
                "status": status,
                "fingerprint": _fingerprint(stage),
                "outputs": {str(output): file_digest(output)},
            },
            authenticated_inputs={str(input_path): file_digest(input_path)},
        )


def test_input_map_and_postcommand_are_mandatory(tmp_path: Path) -> None:
    stage = Stage(
        name="adopt",
        command=("normal",),
        inputs=(tmp_path / "in",),
        outputs=(tmp_path / "out",),
        timeout_seconds=30,
    )
    with pytest.raises(ReplayError, match="incomplete"):
        proof.historical_stage_fingerprint(stage, {})
    with pytest.raises(ReplayError, match="incomplete"):
        proof.historical_stage_fingerprint(stage, {str(tmp_path / "in"): "not a hash"})
    with pytest.raises(ReplayError, match="post-command"):
        proof.verify_completed_stage(stage, checkpoint={}, authenticated_inputs={})


def test_completed_output_changed_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output = tmp_path / "output"
    input_path.write_bytes(b"input")
    output.write_bytes(b"output")
    stage = Stage(
        name="adopt",
        command=("normal",),
        verify_command=("verify",),
        inputs=(input_path,),
        outputs=(output,),
        timeout_seconds=30,
    )
    checkpoint = {
        "status": "complete",
        "fingerprint": _fingerprint(stage),
        "outputs": {str(output): file_digest(output)},
    }
    output.write_bytes(b"tampered")
    with pytest.raises(ReplayError, match="output bytes changed"):
        proof.verify_completed_stage(
            stage,
            checkpoint=checkpoint,
            authenticated_inputs={str(input_path): file_digest(input_path)},
        )


def test_full_inventory_consumers_cannot_be_omitted(tmp_path: Path) -> None:
    old = source(tmp_path / "old.py", "def used():\n return 1\ndef unused():\n return 2")
    current = source(tmp_path / "fixture.py", "def used():\n return 1\ndef unused():\n return 3")
    caller = source(tmp_path / "caller.py", "from replay.fixture import unused\nunused()")
    with pytest.raises(ReplayError, match="actually referenced"):
        proof.review_reader_generation(
            historical_pins={current.path: old.sha256, caller.path: caller.sha256},
            current_pins={current.path: current.sha256, caller.path: caller.sha256},
            historical_bytes={current.path: old},
        )


def test_full_inventory_unused_review_and_unchanged_generation(tmp_path: Path) -> None:
    old = source(tmp_path / "old.py", "def used():\n return 1\ndef unused():\n return 2")
    current = source(tmp_path / "fixture.py", "def used():\n return 1\ndef unused():\n return 3")
    caller = source(tmp_path / "caller.py", "from replay.fixture import used\nused()")
    reviewed = proof.review_reader_generation(
        historical_pins={current.path: old.sha256, caller.path: caller.sha256},
        current_pins={current.path: current.sha256, caller.path: caller.sha256},
        historical_bytes={current.path: old},
    )
    assert len(reviewed) == 1
    assert reviewed[0].used_symbols == ("used",)
    assert reviewed[0].changed_unused_symbols == ("unused",)
    assert (
        proof.review_reader_generation(
            historical_pins={caller.path: caller.sha256},
            current_pins={caller.path: caller.sha256},
            historical_bytes={},
        )
        == ()
    )


def test_added_dependency_is_not_implicitly_unused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="dependency inventories differ"):
        proof.review_reader_generation(
            historical_pins={tmp_path / "old.py": "0" * 64},
            current_pins={tmp_path / "old.py": "0" * 64, tmp_path / "added.py": "1" * 64},
            historical_bytes={},
        )


def test_exact_changed_old_bytes_required(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="exact changed source set"):
        proof.review_reader_generation(
            historical_pins={tmp_path / "old.py": "0" * 64},
            current_pins={tmp_path / "old.py": "1" * 64},
            historical_bytes={},
        )
    with pytest.raises(ReplayError, match="different source generation"):
        proof.review_reader_generation(
            historical_pins={tmp_path / "old.py": "0" * 64},
            current_pins={tmp_path / "old.py": "1" * 64},
            historical_bytes={
                tmp_path / "old.py": proof.SourceBytes(tmp_path / "evidence", "2" * 64)
            },
        )


def test_dependency_inventory_bounds(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="dependency inventories differ"):
        proof.review_reader_generation(historical_pins={}, current_pins={}, historical_bytes={})
    pins = {tmp_path / f"module-{index}.py": "0" * 64 for index in range(257)}
    with pytest.raises(ReplayError, match="bounded review budget"):
        proof.review_reader_generation(historical_pins=pins, current_pins=pins, historical_bytes={})


@pytest.mark.parametrize(
    "suffix",
    [
        "\nif __name__ == '__main__':\n used()",
        "\nALIAS = used",
        "\ndef unused(arg=used()) -> used:\n return 1",
        "\nclass Unused:\n method = used",
    ],
)
def test_definition_time_reference_cannot_be_marked_unused(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ReplayError, match="actually referenced"):
        review(
            tmp_path,
            "def used():\n return 1" + suffix,
            "def used():\n return 2" + suffix,
            "from replay import fixture",
        )


def test_reflective_module_and_function_attributes_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="Reflective module"):
        review(
            tmp_path,
            "def unused():\n return 1\nnamespace=globals()",
            "def unused():\n return 2\nnamespace=globals()",
            "from replay import fixture",
        )
    with pytest.raises(ReplayError, match="Reflective consumer"):
        review(
            tmp_path,
            "def used():\n return 1",
            "def used():\n return 1",
            "from replay.fixture import used\nused.__globals__['unknown']()",
        )
    with pytest.raises(ReplayError, match="actually referenced"):
        review(
            tmp_path,
            "def __getattr__(name):\n return 1",
            "def __getattr__(name):\n return 2",
            "from replay import fixture",
        )
