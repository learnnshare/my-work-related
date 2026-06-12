"""
stats_parser.py — parse gem5 m5out/stats.txt into dotted-key dicts.

stats.txt format:
    ---------- Begin Simulation Statistics ----------
    system.cpu.numCycles            123456            # comment
    system.mem_ctrls.bytesRead::total  789
    ...
    ---------- End Simulation Statistics ----------

A run may dump multiple regions (one per m5 workbegin/workend). We return a list
of regions; callers pick the kernel-of-interest region (default: last).
Pure stdlib, no deps — works anywhere.
"""
from __future__ import annotations
import re
from pathlib import Path

BEGIN = "Begin Simulation Statistics"
END = "End Simulation Statistics"


def parse_stats(path) -> list[dict]:
    """Return a list of region dicts {dotted_key: float}."""
    text = Path(path).read_text(errors="replace")
    regions = []
    cur = None
    for line in text.splitlines():
        if BEGIN in line:
            cur = {}
            continue
        if END in line:
            if cur is not None:
                regions.append(cur)
            cur = None
            continue
        if cur is None:
            continue
        # strip comment
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key, val = parts[0], parts[1]
        try:
            cur[key] = float(val)
        except ValueError:
            # non-numeric (e.g. "nan", strings) -> skip but keep raw under _str
            if val.lower() in ("nan", "inf", "-inf"):
                cur[key] = float("nan")
    if cur:  # file without explicit END
        regions.append(cur)
    return regions


class StatsRegion:
    """Glob/regex helpers over one region dict."""

    def __init__(self, d: dict):
        self.d = d

    def get(self, key, default=None):
        return self.d.get(key, default)

    def find(self, pattern):
        """Return {key: value} for keys matching a regex."""
        rx = re.compile(pattern)
        return {k: v for k, v in self.d.items() if rx.search(k)}

    def sum(self, pattern):
        vals = [v for v in self.find(pattern).values() if v == v]  # drop nan
        return sum(vals) if vals else None

    def first(self, *patterns):
        for p in patterns:
            m = self.find(p)
            if m:
                # exact key wins, else first match
                if p in self.d:
                    return self.d[p]
                return next(iter(m.values()))
        return None


def load_region(path, index=-1) -> StatsRegion:
    regions = parse_stats(path)
    if not regions:
        return StatsRegion({})
    return StatsRegion(regions[index])
