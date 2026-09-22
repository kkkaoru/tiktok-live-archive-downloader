.PHONY: check public-check
check:
	bash -n scripts/bootstrap.sh
	bash -n scripts/check-public.sh
	bash -n scripts/verify-frame-count.sh
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests
	uv run pytest --cov-report=json
	uv run python -c 'import json; from pathlib import Path; d=json.loads(Path("coverage.json").read_text(encoding="utf-8")); bad=[p for p,v in d["files"].items() if v["summary"]["percent_covered"] < 90]; print("Per-file coverage:", "FAIL " + str(bad) if bad else "PASS (>=90%)"); raise SystemExit(bool(bad))'

public-check:
	bash scripts/check-public.sh
