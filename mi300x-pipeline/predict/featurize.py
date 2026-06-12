"""
featurize.py — turn standardized records into model features + labels.

Features: flatten layers[*].metrics numeric values into "L{id}.{k}" plus the flat
metrics. Labels: the DEVICE record's measured e2eMs / latencyP99 / throughput /
achievedTflops / achievedBwTBs / powerW. Training pairs sim-or-gem5 features ->
device labels on the same (workload, config) key.
"""
from __future__ import annotations

# the metrics the predictor estimates (these drive the dashboard parity plot)
TARGETS = ["e2eMs", "throughput", "achievedTflops", "achievedBwTBs", "powerW"]


def config_key(record):
    wl = record["meta"]["workload"]
    rc = record["meta"].get("run_config", {})
    return (wl.get("id"), wl.get("precision") or wl.get("pref"),
            wl.get("batch"), rc.get("numGPUs", 1))


def featurize(record):
    """record -> {feature_name: float} (numeric only)."""
    feats = {}
    for layer in record.get("layers", []):
        lid = layer["id"]
        for m in layer["metrics"]:
            v = m.get("v")
            if isinstance(v, (int, float)) and v == v:  # drop None/NaN
                feats[f"L{lid}.{m['k']}"] = float(v)
    for k, v in record.get("metrics", {}).items():
        if isinstance(v, (int, float)) and v == v:
            feats[f"flat.{k}"] = float(v)
    return feats


def labels(record):
    m = record.get("metrics", {})
    return {t: m.get(t) for t in TARGETS if isinstance(m.get(t), (int, float))}


def make_dataset(feature_records, label_records):
    """Join feature records (gem5/sim) to label records (device) by config_key.
    Returns list of {key, features, labels}."""
    labels_by_key = {config_key(r): labels(r) for r in label_records}
    out = []
    for fr in feature_records:
        k = config_key(fr)
        if k in labels_by_key and labels_by_key[k]:
            out.append({"key": k, "features": featurize(fr), "labels": labels_by_key[k]})
    return out
