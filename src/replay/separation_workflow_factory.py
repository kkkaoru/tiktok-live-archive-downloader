"""Prepare trusted separation workflows, without extraction or inference.

Graph creation checks metadata and source-code pins only. Actual source/model
hashes, PCM content, and completed-chunk domains remain the original producers'
responsibility. Adoption always runs their read-only verifier in a new Stage;
it never changes an old checkpoint or retries an uncertain extraction.
"""

from __future__ import annotations

import argparse
import stat
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Literal

from replay.bgm_separation import (
    MODEL_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    MODEL_SHA256,
    _chunk_bounds,
)
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.native_invocation import canonical, durable, identity, pin, read, require, rows
from replay.recordings import record
from replay.separation_pipeline_prepare import (
    Preparation,
    _clock,
    _completed,
    load_config,
)
from replay.separation_window_budget import (
    WindowBudget,
    _integer,
    _references,
    _required_provider_paths,
)
from replay.separation_window_pipeline import (
    CHUNK,
    OVERLAP,
    PipelineBudget,
    code_inputs,
    pcm_info,
)
from replay.stages import StageError
from replay.workflow import parse_stages, stage_graph

Mode = Literal["extract", "adopt-existing"]
ENVIRONMENT_PATHS = frozenset(
    {"TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE"}
)
REQUEST_FIELDS = frozenset(
    {
        "version",
        "kind",
        "config",
        "template",
        "mode",
        "python",
        "runtimeInputs",
        "environment",
        "phases",
        "planOutput",
    }
)
TEMPLATE_FIELDS = frozenset(
    {
        "version",
        "kind",
        "source",
        "outputDirectory",
        "budget",
        "codeInputs",
        "providerInputs",
        "modelInputs",
        "chunkSamples",
        "overlapSamples",
        "sampleRate",
        "qualityApproved",
        "productionBackendAdopted",
    }
)
ENV_EXECUTABLE = Path("/usr/bin/env")
EXTRACTION_TIMEOUT = 7200
PIPELINE_TIMEOUT = 86400


@dataclass(frozen=True)
class FactoryRequest:
    path: Path
    config: Preparation
    template: Path
    definition: dict[str, object]
    mode: Mode
    python: Path
    runtime: tuple[Path, ...]
    environment: dict[str, str]
    phases: Path
    plan: Path
    output_directory: Path
    chunks: int


class Arguments(argparse.Namespace):
    request: Path
    output: Path
    verify: bool


def _local(path: Path) -> Path:
    require(
        path.is_relative_to(Path.cwd().resolve() / "edits/work"), "Factory DATA must stay local"
    )
    return path


def _reference(value: object) -> Path:
    row = record(value)
    path = canonical(row.get("path"))
    digest = row.get("sha256")
    require(
        set(row) == {"path", "sha256"}
        and isinstance(digest, str)
        and len(digest) == 64
        and all(letter in "0123456789abcdef" for letter in digest),
        "Expected an exact factory reference",
    )
    require(identity(path)["bytes"] > 0, "Empty factory input")
    return path


def _metadata_reference(value: object) -> Path:
    path = _local(_reference(value))
    require(
        stat.S_IMODE(path.stat().st_mode) == 0o600,
        "Factory DATA mode must be finalized before pinning",
    )
    require(pin(path)["sha256"] == record(value)["sha256"], "Factory metadata SHA256 changed")
    return path


def _environment(value: object) -> dict[str, str]:
    data = record(value)
    require(set(data) == ENVIRONMENT_PATHS, "Expected explicit local scratch/model environment")
    result: dict[str, str] = {}
    for name, value in data.items():
        path = canonical(value)
        require(
            path.is_relative_to(Path.cwd().resolve()) and path.is_dir(),
            "Environment escapes checkout",
        )
        result[name] = str(path)
    return result


def _template(path: Path, config: Preparation) -> tuple[dict[str, object], Path, PipelineBudget]:
    data = read(path)
    require(
        set(data) == TEMPLATE_FIELDS
        and type(data.get("version")) is int
        and data.get("version") == 1
        and data.get("kind") == "separation-window-pipeline"
        and data.get("source") == {"path": str(config.pcm), "sha256": None}
        and type(data.get("chunkSamples")) is int
        and data.get("chunkSamples") == CHUNK
        and type(data.get("overlapSamples")) is int
        and data.get("overlapSamples") == OVERLAP
        and type(data.get("sampleRate")) is int
        and data.get("sampleRate") == MODEL_SAMPLE_RATE
        and data.get("qualityApproved") is False
        and data.get("productionBackendAdopted") is False,
        "Unsupported original pipeline template",
    )
    require(_references(data.get("codeInputs")) == code_inputs(), "Pipeline AST closure changed")
    _references(data.get("providerInputs"))
    _references(data.get("modelInputs"))
    output = _local(canonical(data.get("outputDirectory")))
    require(output.parent == config.directory, "Pipeline output must share the preparation parent")
    raw = record(data.get("budget"))
    window = record(raw.get("window"))
    require(
        set(raw)
        == {"window", "max_samples", "max_saved_bytes", "max_parent_rss_bytes", "timeout_seconds"},
        "Invalid pipeline budget fields",
    )
    require(
        set(window)
        == {"workers", "max_samples", "timeout_seconds", "max_worker_rss_bytes", "max_saved_bytes"},
        "Invalid window budget fields",
    )
    selected = WindowBudget(
        workers=_integer(window["workers"]),
        max_samples=_integer(window["max_samples"]),
        timeout_seconds=_integer(window["timeout_seconds"]),
        max_worker_rss_bytes=_integer(window["max_worker_rss_bytes"]),
        max_saved_bytes=_integer(window["max_saved_bytes"]),
    )
    budget = PipelineBudget(
        selected,
        _integer(raw["max_samples"]),
        _integer(raw["max_saved_bytes"]),
        _integer(raw["max_parent_rss_bytes"]),
        _integer(raw["timeout_seconds"]),
    )
    return data, output, budget


def _bindings(definition: dict[str, object], environment: dict[str, str]) -> tuple[Path, ...]:
    code = _references(definition["codeInputs"])
    providers = _references(definition["providerInputs"])
    model = _references(definition["modelInputs"])
    snapshot = (
        Path(environment["HF_HUB_CACHE"])
        / ("models--" + MODEL_NAME.replace("/", "--"))
        / "snapshots"
        / MODEL_REVISION
    )
    require(
        model
        == {
            str(snapshot / "model.safetensors"): MODEL_SHA256,
            str(snapshot / "config.json"): MODEL_CONFIG_SHA256,
        },
        "Original model/cache pins changed",
    )
    require(_required_provider_paths() <= set(providers), "Missing installed provider code/library")
    bindings = {**code, **providers, **model}
    require(
        len(bindings) == len(code) + len(providers) + len(model),
        "Overlapping code/provider/model roles",
    )
    return tuple(Path(path) for path in bindings)


def _prepared_chunks(config: Preparation, mode: Mode, budget: PipelineBudget) -> int:
    if mode == "adopt-existing":
        samples = pcm_info(config.pcm)
        receipt = read(config.receipt)
        require(
            type(receipt.get("version")) is int
            and receipt.get("version") == 1
            and receipt.get("kind") == "verified-separation-pcm-preparation"
            and receipt.get("inputSampleCount") == samples
            and receipt.get("outputSampleCount") == samples
            and receipt.get("currentSourceContentEquivalent") is True
            and receipt.get("qualityApproved") is False
            and receipt.get("productionBackendAdopted") is False
            and record(receipt.get("output")).get("path") == str(config.pcm)
            and record(receipt.get("output")).get("stat") == identity(config.pcm),
            "No complete prepared PCM receipt; preserve uncertain outputs",
        )
        _completed(config)
        require(samples <= budget.max_samples, "Pipeline sample budget exceeded")
        return len(_chunk_bounds(samples, CHUNK, OVERLAP))
    clock = _clock(config)
    numerator = clock.sample_ends[-1] * MODEL_SAMPLE_RATE
    lower = max(1, (numerator - 1) // clock.sample_rate)
    upper = (numerator + clock.sample_rate) // clock.sample_rate
    require(upper <= budget.max_samples, "Template cannot cover the original one-sample tolerance")
    counts = {len(_chunk_bounds(samples, CHUNK, OVERLAP)) for samples in range(lower, upper + 1)}
    require(len(counts) == 1, "Ambiguous chunk count; prepare PCM first, then explicitly adopt it")
    return counts.pop()


def _extraction_paths(config: Preparation) -> tuple[Path, ...]:
    proofs: tuple[Path, ...] = (config.pcm, config.receipt)
    if config.completion is None:
        proofs += (
            config.directory / "extraction-attempt.json",
            config.directory / "extraction-complete.json",
        )
    return proofs


def load_request(path: Path, *, existing_graph: bool = False) -> FactoryRequest:
    path = _local(canonical(str(path.absolute())))
    data = read(path)
    require(
        set(data) == REQUEST_FIELDS
        and type(data.get("version")) is int
        and data.get("version") == 1
        and data.get("kind") == "separation-workflow-request",
        "Unsupported factory request",
    )
    value = data.get("mode")
    if value == "extract":
        mode: Mode = "extract"
    elif value == "adopt-existing":
        mode = "adopt-existing"
    else:
        raise ReplayError("Explicit extraction or completed-PCM adoption required")
    config = load_config(_metadata_reference(data.get("config")))
    template = _metadata_reference(data.get("template"))
    definition, directory, budget = _template(template, config)
    environment = _environment(data.get("environment"))
    _bindings(definition, environment)
    python_value = data.get("python")
    require(isinstance(python_value, str) and bool(python_value), "Missing lexical CPU Python")
    python = Path(str(python_value))
    require(
        python.is_absolute() and python.is_relative_to(Path.cwd().resolve()),
        "Python alias must stay local",
    )
    runtime = tuple(_reference(value) for value in rows(data.get("runtimeInputs")))
    require(
        len(set(runtime)) == len(runtime)
        and {python.resolve(strict=True), python.parent.parent / "pyvenv.cfg", ENV_EXECUTABLE}
        <= set(runtime),
        "Missing canonical Python/env runtime pins",
    )
    phases = _local(canonical(data.get("phases")))
    plan = _local(canonical(data.get("planOutput")))
    require(
        phases.parent == config.directory
        and plan.parent == config.directory
        and plan not in (template, config.path, config.pcm, config.receipt),
        "Factory output paths overlap or escape",
    )
    require(
        phases not in (plan, path, template, config.path, *_extraction_paths(config))
        and phases != directory,
        "Phase log must not mutate a declared artifact",
    )
    if not existing_graph:
        require(
            not any(item.exists() or item.is_symlink() for item in (plan, phases, directory)),
            "Future outputs already occupied; preserve the existing workflow",
        )
        proofs = _extraction_paths(config)
        if mode == "extract":
            require(
                config.completion is None
                and not any(
                    item.exists() or item.is_symlink()
                    for item in (*proofs, config.directory / "extract.stderr")
                ),
                "Extraction already attempted; reconcile instead of retrying",
            )
        else:
            require(
                all(item.exists() and identity(item)["bytes"] > 0 for item in proofs),
                "Incomplete/empty prepared PCM proof; preserve uncertain outputs",
            )
    chunks = _prepared_chunks(config, mode, budget)
    return FactoryRequest(
        path,
        config,
        template,
        definition,
        mode,
        python,
        runtime,
        environment,
        phases,
        plan,
        directory,
        chunks,
    )


def _prefix(request: FactoryRequest) -> list[str]:
    env = {
        **request.environment,
        "PYTHONPATH": str(Path.cwd().resolve() / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return [
        str(ENV_EXECUTABLE),
        *(f"{name}={value}" for name, value in sorted(env.items())),
        str(request.python),
        "-B",
        "-m",
    ]


def _timed(request: FactoryRequest, name: str, command: list[str]) -> list[str]:
    return [*_prefix(request), "replay.phase_timer", str(request.phases), name, "--", *command]


def _stage(
    name: str,
    command: list[str],
    inputs: tuple[Path, ...],
    outputs: tuple[Path, ...],
    verifier: list[str],
    *,
    adopt: bool = False,
    timeout: int = EXTRACTION_TIMEOUT,
) -> dict[str, object]:
    return {
        "name": name,
        "command": command,
        "inputs": [str(path) for path in dict.fromkeys(inputs)],
        "outputs": [str(path) for path in outputs],
        "timeout_seconds": timeout,
        "verify_command": verifier,
        "adopt_existing": adopt,
    }


def _chunk_outputs(directory: Path, index: int) -> tuple[Path, ...]:
    chunk = directory / "chunks" / f"chunk-{index:06d}"
    return tuple(chunk / leaf for leaf in ("input.json", "complete.json", "vocals.npy"))


def build(request: FactoryRequest) -> dict[str, object]:
    config = request.config
    code = tuple(
        item.path for item in capture(("separation_workflow_factory", "phase_timer", "workflow"))
    )
    common = (
        request.path,
        config.path,
        *tuple(pair[0] for pair in config.references.values()),
        *tuple(config.code),
        *request.runtime,
        *code,
    )
    bridge = [*_prefix(request), "replay.separation_pipeline_prepare"]
    verify = [*bridge, "verify", str(config.path)]
    adopt = request.mode == "adopt-existing"
    name = "verify-and-adopt-model-pcm" if adopt else "extract-and-verify-model-pcm"
    action = verify if adopt else [*bridge, "extract", str(config.path)]
    proofs = _extraction_paths(config)
    first = _stage(
        name,
        _timed(request, name, action),
        common,
        proofs,
        _timed(request, name + "-verification", verify),
        adopt=adopt,
    )
    bindings = _bindings(request.definition, request.environment)
    prepared_inputs = (*common, request.template, *proofs, *bindings)
    prepare = [*bridge, "prepare-plan", str(config.path), str(request.template), str(request.plan)]
    second = _stage(
        "prepare-two-worker-plan",
        _timed(request, "prepare-two-worker-plan", prepare),
        prepared_inputs,
        (request.plan,),
        [*prepare, "--verify"],
    )
    outputs: tuple[Path, ...] = (
        request.output_directory / "input.json",
        request.output_directory / "result.json",
        request.output_directory / "vocals.wav",
    )
    outputs += tuple(
        chain.from_iterable(
            _chunk_outputs(request.output_directory, index) for index in range(request.chunks)
        )
    )
    pipeline = [*_prefix(request), "replay.separation_window_pipeline", str(request.plan)]
    third = _stage(
        "two-worker-original-window-pipeline",
        _timed(request, "two-worker-original-window-pipeline", pipeline),
        (*prepared_inputs, request.plan),
        outputs,
        [*pipeline, "--verify"],
        timeout=PIPELINE_TIMEOUT,
    )
    graph: dict[str, object] = {"version": 1, "stages": [first, second, third]}
    stage_graph(parse_stages(graph))
    return graph


def prepare(request: Path, output: Path, *, verify_only: bool = False) -> Path:
    """Write one exclusive private graph, or compare it without execution/writes."""
    request = canonical(str(request.absolute()))
    output = _local(canonical(str(output.absolute())))
    require(output.parent.is_dir(), "Factory output parent must already exist")
    require(
        stat.S_IMODE(request.stat().st_mode) == 0o600,
        "Request mode must be finalized before pinning",
    )
    before = pin(request)
    parsed = load_request(request, existing_graph=verify_only)
    require(
        output not in (parsed.path, parsed.config.path, parsed.template, parsed.plan, parsed.phases)
        and output.parent == parsed.config.directory,
        "Workflow path must be a distinct preparation child",
    )
    config_pin, template_pin = pin(parsed.config.path), pin(parsed.template)
    graph = build(parsed)
    require(
        pin(request) == before
        and pin(parsed.config.path) == config_pin
        and pin(parsed.template) == template_pin,
        "Factory inputs changed during generation",
    )
    if verify_only:
        require(
            read(output) == graph and stat.S_IMODE(output.stat().st_mode) == 0o600,
            "Generated workflow changed",
        )
    else:
        require(
            not output.exists() and not output.is_symlink(),
            "Workflow already exists; never overwrite it",
        )
        durable(output, graph)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv, namespace=Arguments())
    try:
        print(prepare(arguments.request, arguments.output, verify_only=arguments.verify))
    except (OSError, ValueError, ReplayError, StageError) as error:
        print(f"SEPARATION_WORKFLOW_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
