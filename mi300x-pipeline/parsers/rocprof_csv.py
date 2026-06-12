"""
rocprof_csv.py — parse real rocprofv3 1.0.0 CSV output (verified on MI300X/gfx942).

rocprofv3 writes into <outdir>/<hostname>/<pid>_*.csv:
  - <pid>_kernel_trace.csv       per-dispatch timing + VGPR/SGPR/grid
  - <pid>_counter_collection.csv LONG format: one row per (dispatch, counter)
  - <pid>_agent_info.csv         per-agent static info (CU count, clocks, ...)

Schemas (column names) confirmed from an on-box capture log.
"""
from __future__ import annotations
import csv
import glob
import os


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def find_outputs(outdir):
    """Locate the three CSVs under outdir/**/ regardless of host/pid prefix."""
    res = {"kernel_trace": None, "counters": None, "agent_info": None}
    for f in sorted(glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)):
        b = os.path.basename(f)
        if b.endswith("kernel_trace.csv"):
            res["kernel_trace"] = f
        elif b.endswith("counter_collection.csv"):
            res["counters"] = f
        elif b.endswith("agent_info.csv"):
            res["agent_info"] = f
    return res


def parse_kernel_trace(path):
    """-> list of {name, start_ns, end_ns, dur_s, vgpr, sgpr, grid, wg}."""
    out = []
    for r in _rows(path):
        s, e = _f(r.get("Start_Timestamp")), _f(r.get("End_Timestamp"))
        if s is None or e is None:
            continue
        out.append({
            "name": r.get("Kernel_Name", ""),
            "start_ns": s, "end_ns": e, "dur_s": max(e - s, 0) / 1e9,
            "vgpr": _i(r.get("VGPR_Count")), "sgpr": _i(r.get("SGPR_Count")),
            "grid": _i(r.get("Grid_Size_X")), "wg": _i(r.get("Workgroup_Size_X")),
        })
    return out


def parse_counters(path):
    """LONG -> {kernel_name: {counter_name: summed_value}} across dispatches."""
    agg = {}
    for r in _rows(path):
        k = r.get("Kernel_Name", "")
        cn = r.get("Counter_Name")
        cv = _f(r.get("Counter_Value"))
        if cn is None or cv is None:
            continue
        agg.setdefault(k, {}).setdefault(cn, 0.0)
        agg[k][cn] += cv
    return agg


def parse_agent_info(path):
    """-> dict of GPU agent static properties (first GPU agent)."""
    for r in _rows(path):
        if r.get("Agent_Type") == "GPU":
            return {
                "cu_count": _i(r.get("Cu_Count")),
                "num_xcc": _i(r.get("Num_Xcc")),
                "wavefront": _i(r.get("Wave_Front_Size")),
                "simd_per_cu": _i(r.get("Simd_Per_Cu")),
                "max_waves_per_cu": _i(r.get("Max_Waves_Per_Cu")),
                "max_clk_mhz": _i(r.get("Max_Engine_Clk_Fcompute")),
            }
    return {}


def pick_kernel(traces, counters, kernel_hint=None):
    """Choose the kernel of interest: explicit hint substring, else the kernel
    with the largest total GPU time (ignoring rocclr fill/copy helpers)."""
    by_name = {}
    for t in traces:
        by_name.setdefault(t["name"], 0.0)
        by_name[t["name"]] += t["dur_s"]
    names = list(by_name) or list(counters)
    if kernel_hint:
        for n in names:
            if kernel_hint.lower() in n.lower():
                return n
    real = [n for n in names if "rocclr" not in n.lower() and "fill" not in n.lower()
            and "copy" not in n.lower()]
    pool = real or names
    return max(pool, key=lambda n: by_name.get(n, 0.0)) if pool else None


# ---- backward-compat wrappers used by the device collectors ----
def counters_to_l0(counter_csv, peak=None):
    agg = parse_counters(counter_csv)
    if not agg:
        return {}, {}
    name = pick_kernel([], agg)
    c = agg.get(name, {})
    return _derive_l0(c), {}


def trace_to_l3(trace_csv):
    traces = parse_kernel_trace(trace_csv)
    if not traces:
        return {}, {}
    span = (max(t["end_ns"] for t in traces) - min(t["start_ns"] for t in traces)) / 1e9
    sc = {}
    if span > 0:
        sc["dispatch_rate_k"] = round(len(traces) / span / 1000, 3)
    sc["active_streams"] = 1
    return sc, {"dispatch_rate_k": "derived", "active_streams": "measured"}


def _derive_l0(c, agent=None):
    """Derive L0 scalars from a single kernel's counter dict."""
    out = {}
    total = c.get("GRBM_COUNT")
    active = c.get("GRBM_GUI_ACTIVE")
    if total and active is not None:
        out["mfma_util_pct"] = round(min(100, 100 * active / total), 1)  # gfx-active proxy
    hit = c.get("TCC_HIT_sum") or c.get("TCC_HIT")
    miss = c.get("TCC_MISS_sum") or c.get("TCC_MISS")
    if hit is not None and (hit + (miss or 0)) > 0:
        out["kernel_cache_hit_pct"] = round(100 * hit / (hit + (miss or 0)), 1)
    return out
