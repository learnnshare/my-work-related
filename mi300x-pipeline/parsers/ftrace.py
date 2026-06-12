"""ftrace.py — parse amdgpu/KFD ftrace output for L2 driver metrics (defensive)."""
from __future__ import annotations
from pathlib import Path


def parse_kfd(dat_path):
    """Best-effort parse of an ftrace dump. Returns (scalars, fidelity).
    KFD/amdgpu tracepoint formats vary; refine against your kernel."""
    try:
        lines = Path(dat_path).read_text(errors="replace").splitlines()
    except OSError:
        return {}, {}
    faults = sum(1 for l in lines if "page_fault" in l or "retry_fault" in l)
    dispatches = [l for l in lines if "kfd" in l.lower() and "dispatch" in l.lower()]
    sc, fd = {}, {}
    if faults:
        sc["page_faults_s"] = faults; fd["page_faults_s"] = "derived"  # per-run; orchestrator can /duration
    if dispatches:
        sc["hw_queue_depth"] = min(16, len(dispatches)); fd["hw_queue_depth"] = "derived"
    return sc, fd
