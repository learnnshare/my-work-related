"""
schema.py — the standardized record schema + validators.

One schema, two producers (device, gem5). A record:

    {
      "meta":   {schema_version, source, run_id, timestamp, gpu, model,
                 run_config, workload, fidelity},
      "metrics":{...FLAT_KEYS...},     # what computeMetrics() returns flat
      "layers": [ {id,name,sub,metrics:[{k,v,unit?,max?,fmt?,_f?}]} x8 ]
    }

The key-set guard (`assert_contract`) ensures emitted keys match the dashboard
contract (FLAT_KEYS + the per-layer metric keys in layer_map) before publish, so
the dashboard never sees a renamed/missing field.
"""
from __future__ import annotations

from .layer_map import FLAT_KEYS, LAYER_METRICS, LAYER_META

SCHEMA_VERSION = "1.0.0"
VALID_SOURCES = {"device", "gem5", "prediction", "synthetic"}


def new_meta(source, run_id, timestamp, run_config, workload, fidelity=None,
             gpu="AMD Instinct MI300X", model="gfx942"):
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "run_id": run_id,
        "timestamp": timestamp,
        "gpu": gpu,
        "model": model,
        "run_config": run_config,
        "workload": workload,
        "fidelity": fidelity or {},
    }


def expected_layer_keys():
    """{layer_id: [metric k, ...]} the contract requires."""
    return {lid: [m["k"] for m in slots] for lid, slots in LAYER_METRICS.items()}


def assert_contract(record: dict) -> list[str]:
    """Validate a record against the dashboard contract.
    Returns a list of problems (empty == valid)."""
    problems = []
    # meta
    meta = record.get("meta", {})
    if meta.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"meta.schema_version != {SCHEMA_VERSION}")
    if meta.get("source") not in VALID_SOURCES:
        problems.append(f"meta.source invalid: {meta.get('source')}")

    # flat metrics: every FLAT_KEY must be present (value may be null)
    metrics = record.get("metrics", {})
    for k in FLAT_KEYS:
        if k not in metrics:
            problems.append(f"metrics missing flat key: {k}")
    extra = set(metrics) - set(FLAT_KEYS) - {"cfg", "wl", "prec"}
    if extra:
        problems.append(f"metrics has unexpected keys: {sorted(extra)}")

    # layers: ids 0..7, names/subs/keys match
    layers = record.get("layers", [])
    if [l.get("id") for l in layers] != list(range(8)):
        problems.append(f"layers ids must be 0..7, got {[l.get('id') for l in layers]}")
    exp = expected_layer_keys()
    for layer in layers:
        lid = layer.get("id")
        if lid not in LAYER_META:
            continue
        if layer.get("name") != LAYER_META[lid][0]:
            problems.append(f"L{lid} name mismatch: {layer.get('name')!r}")
        got = [m.get("k") for m in layer.get("metrics", [])]
        if got != exp[lid]:
            problems.append(f"L{lid} metric keys mismatch:\n  expected {exp[lid]}\n  got      {got}")
    return problems


def validate_or_raise(record: dict) -> None:
    problems = assert_contract(record)
    if problems:
        raise AssertionError("record violates dashboard contract:\n - " + "\n - ".join(problems))
