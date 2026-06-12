"""
manifest.py — run identity + reproducibility record.

Each run gets a directory runs/<run-id>/ with a manifest.json capturing what was
run, which collectors fired, durations, and artifact paths/checksums.
"""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def make_run_id(mode: str, workload: str, ts: datetime | None = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    salt = hashlib.sha1(os.urandom(8)).hexdigest()[:6]
    safe_wl = "".join(c if c.isalnum() else "-" for c in workload)[:24]
    return f"{stamp}_{mode}_{safe_wl}_{salt}"


def new_run_ctx(out_dir: Path, mode: str, workload: str, ts: datetime | None = None) -> dict:
    ts = ts or datetime.now(timezone.utc)
    run_id = make_run_id(mode, workload, ts)
    run_dir = Path(out_dir) / run_id
    raw_dir = run_dir / "raw"
    inter_dir = run_dir / "intermediate"
    for d in (raw_dir, inter_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "mode": mode,
        "workload": workload,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "run_dir": run_dir,
        "raw_dir": raw_dir,
        "inter_dir": inter_dir,
    }


def _sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return "n/a"


def write_manifest(run_ctx: dict, collectors_status: dict, extra: dict | None = None) -> Path:
    run_dir: Path = run_ctx["run_dir"]
    artifacts = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            artifacts.append({"path": str(p.relative_to(run_dir)), "sha256": _sha256(p)})
    manifest = {
        "run_id": run_ctx["run_id"],
        "mode": run_ctx["mode"],
        "workload": run_ctx["workload"],
        "timestamp": run_ctx["timestamp"],
        "collectors": collectors_status,
        "artifacts": artifacts,
        **(extra or {}),
    }
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out
