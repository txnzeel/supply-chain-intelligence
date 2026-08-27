"""Detect missing, empty, and Git LFS placeholder datasets before execution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


def inspect(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path.relative_to(PROJECT_ROOT))}
    if not path.exists():
        return result | {"status": "missing"}
    result["bytes"] = path.stat().st_size
    with path.open("rb") as handle:
        if handle.read(len(LFS_SIGNATURE)) == LFS_SIGNATURE:
            return result | {"status": "lfs_pointer"}
    if path.stat().st_size == 0:
        return result | {"status": "empty"}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        return result | {"status": "ready", "columns": len(header)}
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        return result | {"status": "invalid", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args()
    paths = sorted((PROJECT_ROOT / "data").rglob("*.csv"))
    report = [inspect(path) for path in paths]
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in report:
            print(f"[{item['status'].upper():11}] {item['path']}")
        counts = {status: sum(item["status"] == status for item in report) for status in sorted({item["status"] for item in report})}
        print(f"\nSummary: {counts}")
    return 1 if any(item["status"] in {"missing", "empty", "invalid", "lfs_pointer"} for item in report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
