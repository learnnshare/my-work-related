"""kineto.py — parse a PyTorch/Kineto chrome trace for L5 framework metrics."""
from __future__ import annotations
import json
from pathlib import Path


def parse_kineto(trace_path):
    try:
        data = json.loads(Path(trace_path).read_text())
    except (OSError, ValueError):
        return {}, {}
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    gpu_us = host_us = launch_us = 0.0
    wall0, wall1 = None, None
    graph = False
    for e in events:
        dur = e.get("dur", 0) or 0
        cat = (e.get("cat") or "").lower()
        name = (e.get("name") or "").lower()
        ts = e.get("ts")
        if ts is not None:
            wall0 = ts if wall0 is None else min(wall0, ts)
            wall1 = ts + dur if wall1 is None else max(wall1, ts + dur)
        if "kernel" in cat or "gpu" in cat or "hip" in cat:
            gpu_us += dur
        elif "cpu" in cat or "runtime" in cat:
            host_us += dur
        if "launch" in name:
            launch_us += dur
        if "graph" in name and ("capture" in name or "replay" in name):
            graph = True
    wall_us = (wall1 - wall0) if (wall0 is not None and wall1 is not None) else (gpu_us + host_us)
    sc, fd = {}, {}
    if gpu_us:
        sc["gpu_compute_ms"] = round(gpu_us / 1000.0, 4); fd["gpu_compute_ms"] = "measured"
    if wall_us:
        sc["host_overhead_pct"] = round(min(100, 100 * host_us / wall_us), 1); fd["host_overhead_pct"] = "derived"
        sc["launch_overhead_pct"] = round(min(100, 100 * launch_us / wall_us), 1); fd["launch_overhead_pct"] = "derived"
    sc["hip_graph"] = "on" if graph else "off"; fd["hip_graph"] = "measured"
    return sc, fd
