"""
reduce.py — reconcile scalar / time-series / per-kernel data into single values.

The dashboard wants one number per metric. Collectors hand us three shapes:
  - scalars      -> taken as-is (last write wins)
  - time-series  -> reduced per metric (mean / p50 / p99 / max / time-weighted)
  - per-kernel   -> aggregated (p50/p99 of latency; total flops / total time)

Headline reducers follow the dashboard's own definitions in sim.js.
"""
from __future__ import annotations
from statistics import mean


def _vals(series):
    return [s.value for s in series if isinstance(s.value, (int, float))]


def p(series_or_list, q):
    xs = sorted(series_or_list if isinstance(series_or_list, list) and (not series_or_list or isinstance(series_or_list[0], (int, float))) else _vals(series_or_list))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * (q / 100.0)
    lo, hi = int(idx), min(int(idx) + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def reduce_series(series, how="mean"):
    xs = _vals(series)
    if not xs:
        return None
    if how == "mean":
        return mean(xs)
    if how == "max":
        return max(xs)
    if how == "min":
        return min(xs)
    if how == "p50":
        return p(xs, 50)
    if how == "p99":
        return p(xs, 99)
    if how == "time_weighted":
        # weight each sample by the gap to the next; falls back to mean
        if len(series) < 2:
            return xs[0]
        num = den = 0.0
        for a, b in zip(series, series[1:]):
            if isinstance(a.value, (int, float)):
                w = max(b.t_ns - a.t_ns, 1)
                num += a.value * w
                den += w
        return num / den if den else mean(xs)
    return mean(xs)


# Default reducer per canonical scalar name (sim.js headline conventions).
DEFAULT_REDUCERS = {
    "power_w": "mean",
    "temp_c": "max",
    "clock_mhz": "mean",
    "compute_util": "time_weighted",
    "mem_util": "time_weighted",
    "busy": "time_weighted",
    "hbm_util_pct": "time_weighted",
    "mfma_util_pct": "time_weighted",
    "latency_p50": "p50",
    "latency_p99": "p99",
    "launch_latency_us": "p50",
    "kfd_dispatch_us": "p50",
    "page_faults_s": "mean",
    "irqs_s": "mean",
    "dma_gbs": "mean",
}


def reduce_all(series_by_name, overrides=None):
    """Reduce a {name: [Sample,...]} dict to {name: scalar}."""
    overrides = overrides or {}
    out = {}
    for name, series in series_by_name.items():
        how = overrides.get(name, DEFAULT_REDUCERS.get(name, "mean"))
        out[name] = reduce_series(series, how)
    return out


def aggregate_perkernel(rows):
    """rows: list of {name, dur_s, flops?, bytes?}. Returns aggregates used to
    derive achievedTflops/launch latency etc."""
    if not rows:
        return {}
    total_time = sum(r.get("dur_s", 0) for r in rows) or 1e-12
    total_flops = sum(r.get("flops", 0) for r in rows)
    total_bytes = sum(r.get("bytes", 0) for r in rows)
    lat = [r["launch_us"] for r in rows if "launch_us" in r]
    return {
        "n_kernels": len(rows),
        "kernel_time_s": total_time,
        "achieved_tflops": total_flops / total_time / 1e12 if total_flops else None,
        "achieved_bw_tbs": total_bytes / total_time / 1e12 if total_bytes else None,
        "launch_latency_us_p50": p(lat, 50) if lat else None,
        "launch_latency_us_p99": p(lat, 99) if lat else None,
    }
