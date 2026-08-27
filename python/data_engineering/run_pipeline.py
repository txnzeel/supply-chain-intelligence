"""Configuration-driven local pipeline orchestrator with auditable run metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.toml"
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


def is_materialized(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_SIGNATURE)) != LFS_SIGNATURE


def fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_steps() -> tuple[dict, list[dict]]:
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    return config, config["pipeline"]["steps"]


def select_steps(steps: list[dict], start_at: str | None, end_at: str | None) -> list[dict]:
    names = [step["name"] for step in steps]
    start = names.index(start_at) if start_at else 0
    end = names.index(end_at) + 1 if end_at else len(steps)
    if start >= end:
        raise ValueError("--from must occur before or equal to --to")
    return steps[start:end]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the supply-chain pipeline.")
    parser.add_argument("--from", dest="start_at", help="First step to run")
    parser.add_argument("--to", dest="end_at", help="Last step to run")
    parser.add_argument("--force", action="store_true", help="Run even when outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="Validate and display the execution plan")
    args = parser.parse_args()

    config, all_steps = load_steps()
    try:
        steps = select_steps(all_steps, args.start_at, args.end_at)
    except ValueError as exc:
        parser.error(str(exc))

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "steps": [],
    }

    log_dir = PROJECT_ROOT / config["project"]["run_log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.json"

    planned_outputs: set[Path] = set()
    for step in steps:
        inputs = [PROJECT_ROOT / item for item in step.get("inputs", [])]
        outputs = [PROJECT_ROOT / item for item in step.get("outputs", [])]
        missing = [
            str(path.relative_to(PROJECT_ROOT))
            for path in inputs
            if not is_materialized(path) and path not in planned_outputs
        ]
        upstream_changed = any(path in planned_outputs for path in inputs)
        already_complete = (
            outputs
            and all(is_materialized(path) for path in outputs)
            and not upstream_changed
        )

        if missing:
            status = "blocked"
            message = f"Missing or unmaterialized inputs: {', '.join(missing)}"
        elif already_complete and not args.force:
            status = "skipped"
            message = "All declared outputs already exist; use --force to rebuild."
        else:
            status = "planned" if args.dry_run else "running"
            message = ""

        record = {"name": step["name"], "status": status, "message": message}
        run["steps"].append(record)
        print(f"[{status.upper():8}] {step['name']} {message}")

        if status == "blocked":
            run["status"] = "failed"
            break
        if status in {"skipped", "planned"}:
            if status == "planned":
                planned_outputs.update(outputs)
            continue

        started = time.perf_counter()
        child_environment = os.environ.copy()
        child_environment["PYTHONUTF8"] = "1"
        child_environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / step["script"])],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=child_environment,
        )
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        record["return_code"] = result.returncode
        record["stdout_tail"] = result.stdout[-4000:]
        record["stderr_tail"] = result.stderr[-4000:]

        if result.returncode != 0:
            record["status"] = "failed"
            run["status"] = "failed"
            print(result.stderr[-2000:], file=sys.stderr)
            break

        absent_outputs = [path for path in outputs if not is_materialized(path)]
        if absent_outputs:
            record["status"] = "failed"
            record["message"] = "Step completed but did not materialize all declared outputs."
            run["status"] = "failed"
            break

        record["status"] = "success"
        record["outputs"] = [fingerprint(path) for path in outputs]
        planned_outputs.update(outputs)

    if args.dry_run and run["status"] == "running":
        run["status"] = "dry_run"
    elif run["status"] == "running":
        run["status"] = "success"
    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    log_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nRun status: {run['status']} | metadata: {log_path.relative_to(PROJECT_ROOT)}")
    return 0 if run["status"] in {"success", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
