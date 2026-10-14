.PHONY: check public-check swift-check swift-tsan
COVERAGE_JSON ?= coverage.json
PYTEST_COMMAND ?= uv run pytest -p replay.pytest_scratch
check:
	bash -n scripts/bootstrap.sh
	bash -n scripts/replay-authorized.sh
	bash -n scripts/check-public.sh
	bash -n scripts/check-swift-coverage.sh
	bash -n scripts/native-swift.sh
	bash -n scripts/tmux-job.sh
	bash -n scripts/verify-frame-count.sh
	bash -n scripts/expand-vocals.sh
	bash -n scripts/build-expander-revision.sh
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests
	# Synthetic public-mode fixtures must not inherit the private runner's umask.
	umask 022; $(PYTEST_COMMAND) "--cov-report=json:$(COVERAGE_JSON)"
	uv run python -c 'import json, sys; from pathlib import Path; d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); bad=[p for p,v in d["files"].items() if v["summary"]["percent_covered"] < 90]; print("Per-file coverage:", "FAIL " + str(bad) if bad else "PASS (>=90%)"); raise SystemExit(bool(bad))' "$(COVERAGE_JSON)"

swift-check:
	cd TranscriptViewer && swift format lint --strict --recursive Sources Tests Package.swift
	cd TranscriptViewer && swift test --enable-code-coverage -Xswiftc -warnings-as-errors
	cd TranscriptViewer && xcrun llvm-cov report .build/debug/TranscriptViewerPackageTests.xctest/Contents/MacOS/TranscriptViewerPackageTests -object=.build/debug/transcript-summary -object=.build/debug/transcript-check -instr-profile=.build/debug/codecov/default.profdata > .build/coverage.txt
	bash scripts/check-swift-coverage.sh TranscriptViewer/.build/coverage.txt

swift-tsan:
	cd TranscriptViewer && swift test --sanitize=thread -Xswiftc -warnings-as-errors

public-check:
	bash scripts/check-public.sh
