"""One sandboxed diagnostic, never production adoption or service-write proof."""

import argparse
import importlib
import os
import signal
import subprocess
import sys
import wave
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay.audio_cpu_probe import audit_descriptors
from replay.core import ReplayError
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256, fingerprint
from replay.native_invocation import canonical, durable, read, require
from replay.recordings import record

RAM_RESERVE_BYTES = 6 * 1024**3
MAX_SAMPLES = 60 * 16000


@dataclass(frozen=True)
class ProviderContract:
    """Unchanged WhisperRuntime options; device is explicitly future Metal."""

    device: str = "metal"
    language: str = "ja"
    verbose: bool = False
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    temperature: float = 0.0
    dtype_policy: str = "provider-default-fp16-unchanged"
    clock: str = "continuous-input-pcm-16000"


@dataclass(frozen=True)
class PinnedInput:
    path: Path
    sha256: str

    def validate(self, root: Path) -> None:
        if (
            not self.path.is_absolute()
            or self.path.is_symlink()
            or not self.path.resolve().is_relative_to(root.resolve())
            or len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ReplayError("Expected canonical repository-local pinned input")


@dataclass(frozen=True)
class ProbeRequest:
    pcm: PinnedInput
    cpu_raw: PinnedInput
    cpu_receipt: PinnedInput
    weights: PinnedInput
    config: PinnedInput
    start_sample: int
    sample_count: int
    timeout_seconds: int

    def validate(self, root: Path) -> None:
        for value in (self.pcm, self.cpu_raw, self.cpu_receipt, self.weights, self.config):
            value.validate(root)
        if (
            self.weights.sha256 != DEFAULT_WEIGHTS_SHA256
            or self.config.sha256 != DEFAULT_CONFIG_SHA256
        ):
            raise ReplayError("Only the existing pinned non-turbo large-v3 model is supported")
        if (
            type(self.start_sample) is not int
            or self.start_sample < 0
            or type(self.sample_count) is not int
            or not 0 < self.sample_count <= MAX_SAMPLES
            or type(self.timeout_seconds) is not int
            or not 1 <= self.timeout_seconds <= 3600
        ):
            raise ReplayError("Invalid bounded sample interval or deadline")


def sandbox_profile(root: Path) -> str:
    """Existing repository-write-only policy, never an assertion about services."""
    resolved = root.resolve()
    if not root.is_absolute() or root != resolved or any(c in str(root) for c in '\\"\n\r'):
        raise ReplayError("Sandbox root must be canonical and safely representable")
    return (
        "(version 1)\n(allow default)\n(deny file-write*)\n"
        f'(allow file-write* (subpath "{root}") (literal "/dev/null"))\n'
    )


def planned_environment(work: Path, root: Path) -> dict[str, str]:
    if not work.is_absolute() or not work.resolve().is_relative_to(root.resolve() / "edits/work"):
        raise ReplayError("Probe scratch must stay within repository edits/work")
    return {
        "TMPDIR": str(work / "scratch"),
        "TMP": str(work / "scratch"),
        "TEMP": str(work / "scratch"),
        "XDG_CACHE_HOME": str(work / "cache"),
        "NUMBA_CACHE_DIR": str(work / "numba"),
        "HF_HOME": str(work / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def require_dispatch_supported(request: ProbeRequest, root: Path) -> None:
    """Validate diagnostic admission; independent service coverage stays unknown."""
    request.validate(root)


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Recognition:
    text: str
    tokens: tuple[int, ...]
    words: tuple[TimedWord, ...]


@dataclass(frozen=True)
class Comparison:
    text_equal: bool
    tokens_equal: bool
    words_equal: bool
    cpu_words: tuple[TimedWord, ...]
    metal_words: tuple[TimedWord, ...]
    approved: bool = False


def compare(cpu: Recognition, metal: Recognition) -> Comparison:
    """Report exact differences without normalizing text or inventing tolerances."""
    return Comparison(
        cpu.text == metal.text,
        cpu.tokens == metal.tokens,
        cpu.words == metal.words,
        cpu.words,
        metal.words,
    )


def check_observations(
    *,
    before: tuple[PinnedInput, ...],
    after: tuple[PinnedInput, ...],
    external_writable: tuple[str, ...],
    sandbox_denials: tuple[str, ...],
) -> None:
    """Observations are necessary, never sufficient evidence for confinement.

    Caller retains the measured snapshots and denial records even on failure.
    Empty snapshots do not prove absence of short-lived file writes.
    """
    if before != after:
        raise ReplayError("Probe inputs changed")
    if external_writable or sandbox_denials:
        raise ReplayError("Confinement failed; retain descriptor and sandbox denial evidence")


def load_request(path: Path, *, verify: bool) -> tuple[ProbeRequest, dict[str, object]]:
    data = read(path)
    pins = record(data.get("inputs"))
    parsed: list[PinnedInput] = []
    for name in ("pcm", "cpuRaw", "cpuClock", "weights", "config"):
        item = record(pins.get(name))
        digest = item.get("sha256")
        require(isinstance(digest, str), "Missing input digest")
        parsed.append(PinnedInput(canonical(item.get("path")), str(digest)))
    integers: list[int] = []
    for key in ("startSample", "sampleCount", "timeoutSeconds"):
        value = data.get(key)
        require(type(value) is int, "Expected integer probe bounds")
        integers.append(int(str(value)))
    request = ProbeRequest(
        parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], integers[0], integers[1], integers[2]
    )
    request.validate(Path.cwd())
    require(data.get("fp16") is True, "Pinned provider uses fp16=True; no dtype change")
    baseline_pin = record(data.get("cpuInputs"))
    baseline_path = canonical(baseline_pin.get("path"))
    require(baseline_path.is_relative_to(Path.cwd()), "External CPU manifest")
    code = record(data.get("codeInputs"))
    required = {
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("asr_runtime.py").resolve()),
    }
    provider_root = (
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/mlx_whisper"
    )
    required.update(
        str(provider_root / name) for name in ("transcribe.py", "decoding.py", "load_models.py")
    )
    require(required <= code.keys() and len(code) >= 4, "Missing producer/provider source pins")
    if verify:
        require(
            request.weights.path.parent == request.config.path.parent,
            "Model files must share directory",
        )
        for pinned in parsed:
            require(fingerprint(pinned.path) == pinned.sha256, "Probe input changed")
        for name, digest in code.items():
            source = canonical(name)
            require(source.is_relative_to(Path.cwd()), "External code input")
            require(fingerprint(source) == digest, "Probe code changed")
        require(fingerprint(baseline_path) == baseline_pin.get("sha256"), "CPU manifest changed")
        baseline = read(baseline_path)
        baseline_inputs = record(baseline.get("inputs"))
        require(
            baseline.get("device") == "cpu"
            and baseline.get("dtypeContract") == "provider-default-fp16-unchanged"
            and baseline.get("clock") == "input-pcm-not-projected"
            and all(
                baseline_inputs.get(str(p.path)) == p.sha256
                for p in (request.pcm, request.weights, request.config)
            ),
            "CPU manifest source/model/clock mismatch",
        )
        clock = read(request.cpu_receipt.path)
        require(
            clock.get("startSample") == request.start_sample
            and clock.get("endSample") == request.start_sample + request.sample_count
            and clock.get("sampleRate") == 16000
            and clock.get("rawSHA256") == request.cpu_raw.sha256,
            "CPU baseline clock/raw mismatch",
        )
    return request, data


def worker(request_path: Path, output: Path) -> None:
    request, _ = load_request(request_path, verify=True)
    require(output.resolve().is_relative_to(Path.cwd() / "edits/work"), "External evidence")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "Offline runtime required")
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "HF_HOME"):
        local = canonical(os.environ.get(key))
        require(local.is_relative_to(Path.cwd()) and local.is_dir(), "External scratch")
    with wave.open(str(request.pcm.path), "rb") as audio:
        require(
            (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000),
            "Invalid PCM",
        )
        audio.setpos(request.start_sample)
        payload = audio.readframes(request.sample_count)
    require(len(payload) == request.sample_count * 2, "Truncated PCM")
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768)
    audit_descriptors(output, "before-core-import")
    mx = importlib.import_module("mlx.core")
    audit_descriptors(output, "after-core-import")
    stream = mx.stream
    synchronize = mx.synchronize
    context: object = stream(mx.gpu)
    if not isinstance(context, AbstractContextManager):
        raise ReplayError("Missing Metal stream")
    with context:
        provider = importlib.import_module("mlx_whisper")
        audit_descriptors(output, "after-whisper-import")
        raw: object = provider.transcribe(
            samples,
            path_or_hf_repo=str(request.weights.path.parent),
            language="ja",
            verbose=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            temperature=0.0,
            fp16=True,
        )
        synchronize(mx.gpu)
        audit_descriptors(output, "after-inference-eval")
    durable(output / "raw.json", record(raw))
    load_request(request_path, verify=True)


def verify_result(request_path: Path, output: Path) -> dict[str, object]:
    request, _ = load_request(request_path, verify=True)
    cpu, metal = read(request.cpu_raw.path), read(output / "raw.json")
    stderr = output / "worker.log"
    require(stderr.stat().st_size <= 16 * 1024**2, "Oversized worker log")
    log = stderr.read_text(errors="replace")
    require(
        not any(
            word in log.lower()
            for word in ("deny file-write", "sandbox: deny", "operation not permitted")
        ),
        "Sandbox denial; retain worker.log",
    )
    for phase in (
        "before-core-import",
        "after-core-import",
        "after-whisper-import",
        "after-inference-eval",
    ):
        snapshot = read(output / f"{phase}-descriptors.json")
        require(snapshot.get("externalWritableRegular") == [], "External writable descriptor")
    return {
        "textEqual": cpu.get("text") == metal.get("text"),
        "segmentsEqual": cpu.get("segments") == metal.get("segments"),
        "cpuSegments": cpu.get("segments"),
        "metalSegments": metal.get("segments"),
        "device": "metal",
        "fp16": True,
        "productionApproved": False,
        "independentCompilerServiceCoverage": "unknown",
        "transientWriteCoverage": "unknown",
        "rawSHA256": fingerprint(output / "raw.json"),
    }


def run(request_path: Path, output: Path) -> Path:
    request_digest = fingerprint(request_path)
    request, _ = load_request(request_path, verify=True)
    environment = planned_environment(output, Path.cwd())
    output.mkdir(mode=0o700)
    for key in ("TMPDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "HF_HOME"):
        Path(environment[key]).mkdir()
    profile = output / "sandbox.sb"
    profile.write_text(sandbox_profile(Path.cwd()))
    durable(
        output / "admission.json",
        {
            "requestSHA256": request_digest,
            "sandboxSHA256": fingerprint(profile),
            "planningRAMReserveBytes": RAM_RESERVE_BYTES,
            "hardMemoryLimitEnforced": False,
            "independentCompilerServiceCoverage": "unknown",
            "productionApproved": False,
        },
    )
    command = (
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile),
        sys.executable,
        "-B",
        "-m",
        "replay.isolated_metal_asr_probe",
        str(request_path),
        str(output),
        "--worker",
    )
    with (
        (output / "worker.log").open("xb") as log,
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ | environment,
            start_new_session=True,
        ) as child,
    ):
        try:
            code = child.wait(timeout=request.timeout_seconds)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        require(code == 0, "Metal diagnostic failed; preserve worker.log")
    require(fingerprint(request_path) == request_digest, "Probe request changed")
    result = verify_result(request_path, output)
    durable(output / "comparison.json", result)
    return output / "comparison.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        load_request(args.request, verify=False)
        print("Metadata valid; 6GiB planning reserve, admission and actual confinement unverified")
    elif args.verify:
        print(verify_result(args.request, args.output))
    elif args.worker:
        worker(args.request, args.output)
    else:
        print(run(args.request, args.output))


if __name__ == "__main__":
    main()
