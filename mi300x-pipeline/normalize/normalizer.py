"""
normalizer.py — raw CollectorResults (either source) -> standardized record.

Steps:
  1. merge all collectors' scalars + reduced time-series + per-kernel aggregates
     into one canonical scalar dict.
  2. derive the flat top-level metrics (FLAT_KEYS) from canonical scalars.
  3. build layers[0..7] via layer_map.build_layers (exact dashboard keys).
  4. assemble {meta, metrics, layers} and validate against the contract.
"""
from __future__ import annotations

from . import schema, reduce as R
from .layer_map import build_layers, FLAT_KEYS

# Map flat dashboard key -> ordered candidate canonical scalar names.
FLAT_SOURCES = {
    "e2eMs": ["e2e_ms"],
    "latencyP50": ["latency_p50", "e2e_ms"],
    "latencyP99": ["latency_p99"],
    "throughput": ["throughput"],
    "throughputUnit": ["throughput_unit"],
    "achievedTflops": ["achieved_tflops", "gemm_tflops"],
    "peakTflops": ["peak_tflops"],
    "achievedBwTBs": ["achieved_bw_tbs"],
    "peakBwTBs": ["peak_bw_tbs"],
    "computeUtil": ["compute_util"],
    "memUtil": ["mem_util"],
    "busy": ["busy"],
    "arithIntensity": ["arith_intensity"],
    "powerW": ["power_w"],
    "tempC": ["temp_c"],
    "clockMHz": ["clock_mhz"],
    "memUsedGB": ["vram_gb"],
    "memTotalGB": ["mem_total_gb"],
    "scaleEff": ["scale_eff"],
    "numGPUs": ["num_gpus"],
    "boundBy": ["bound_by"],
}


def _first(scalars, names):
    for n in names:
        if scalars.get(n) is not None:
            return scalars[n]
    return None


def merge_results(results, reducer_overrides=None):
    """Collapse a list of CollectorResult into one canonical scalar dict +
    a fidelity dict. Time-series reduced; per-kernel aggregated."""
    scalars = {}
    fidelity = {}
    series_by_name = {}
    perkernel = []
    for r in results:
        scalars.update({k: v for k, v in r.scalars.items() if v is not None})
        # keep explicit Nones too (so a source can declare "unobservable")
        for k, v in r.scalars.items():
            scalars.setdefault(k, v)
        fidelity.update(r.fidelity)
        for name, samples in r.series.items():
            series_by_name.setdefault(name, []).extend(samples)
        perkernel.extend(r.perkernel)

    scalars.update({k: v for k, v in R.reduce_all(series_by_name, reducer_overrides).items() if v is not None})

    agg = R.aggregate_perkernel(perkernel)
    if agg.get("achieved_tflops") is not None:
        scalars.setdefault("achieved_tflops", agg["achieved_tflops"])
        scalars.setdefault("gemm_tflops", agg["achieved_tflops"])
    if agg.get("achieved_bw_tbs") is not None:
        scalars.setdefault("achieved_bw_tbs", agg["achieved_bw_tbs"])
    if agg.get("launch_latency_us_p50") is not None:
        scalars.setdefault("launch_latency_us", agg["launch_latency_us_p50"])
    return scalars, fidelity


def derive_flat(scalars):
    """Compute the flat top-level metrics dict from canonical scalars."""
    flat = {}
    for k, names in FLAT_SOURCES.items():
        flat[k] = _first(scalars, names)
    # derive busy if absent
    if flat["busy"] is None:
        cu, mu = flat["computeUtil"], flat["memUtil"]
        flat["busy"] = max([x for x in (cu, mu) if x is not None], default=None)
    # derive boundBy if absent
    if flat["boundBy"] is None and flat["computeUtil"] is not None and flat["memUtil"] is not None:
        flat["boundBy"] = "compute" if flat["computeUtil"] >= flat["memUtil"] else "memory"
    return flat


def normalize(results, *, source, run_id, timestamp, run_config, workload,
              peak_tflops=None, reducer_overrides=None):
    """Produce a validated standardized record from collector results."""
    scalars, fidelity = merge_results(results, reducer_overrides)

    # ensure throughput unit + a few config-derived scalars are present
    scalars.setdefault("throughput_unit", (workload.get("short", "") + "/s") if workload.get("short") else workload.get("unit", ""))
    scalars.setdefault("num_gpus", run_config.get("numGPUs", workload.get("batch") and 1 or 1))
    scalars.setdefault("mem_total_gb", run_config.get("hbmGB", 192))
    scalars.setdefault("peak_bw_tbs", round(run_config.get("bwGBs", 5300) / 1000.0, 3))
    scalars.setdefault("batch", workload.get("batch"))
    pk = peak_tflops or run_config.get("peakTflops")
    if pk is not None:
        scalars.setdefault("peak_tflops", pk)

    flat = derive_flat(scalars)
    layers = build_layers(scalars, workload_name=workload.get("name"),
                          peak_tflops=flat.get("peakTflops"), fidelity=fidelity)

    record = {
        "meta": schema.new_meta(source, run_id, timestamp, run_config, workload, fidelity),
        "metrics": {k: flat.get(k) for k in FLAT_KEYS},
        "layers": layers,
    }
    schema.validate_or_raise(record)
    return record
