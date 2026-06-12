"""
sinks.py — predictor output sinks.

Two sinks per the design:
  - file sink: predictions/{run_id}.json (archivable, diffable)
  - dashboard sink: predictionSet-shaped {workload}.json that physical-ai.html
    consumes verbatim (5 pairs, within = errPct <= 20, targetPct = 20)

predictionSet pairs match sim.js: E2E latency, Throughput, Achieved TFLOPS,
HBM bandwidth, Board power.
"""
from __future__ import annotations
import json
from pathlib import Path

from .featurize import featurize, TARGETS

PAIR_DEFS = [
    ("E2E latency", "ms", "e2eMs", True),
    ("Throughput", None, "throughput", False),
    ("Achieved TFLOPS", "TFLOPS", "achievedTflops", False),
    ("HBM bandwidth", "TB/s", "achievedBwTBs", False),
    ("Board power", "W", "powerW", False),
]


def build_prediction_set(device_record, feature_record, predictor, target_pct=20):
    """Compare predictor(feature_record) vs measured(device_record)."""
    measured = device_record["metrics"]
    predicted = predictor.predict(featurize(feature_record))
    unit_thru = measured.get("throughputUnit", "")
    pairs = []
    for k, unit, target, lower_better in PAIR_DEFS:
        meas = measured.get(target)
        pred = predicted.get(target)
        if meas is None or pred is None:
            continue
        err = abs(pred - meas) / (abs(meas) or 1) * 100
        pair = {
            "k": k, "unit": unit if unit else unit_thru,
            "measured": round(meas, 4), "predicted": round(pred, 4),
            "errPct": round(err, 2), "ratio": round(pred / (meas or 1), 4),
            "within": err <= target_pct,
        }
        if lower_better:
            pair["lowerBetter"] = True
        pairs.append(pair)
    within_pct = round(100 * sum(1 for p in pairs if p["within"]) / len(pairs), 1) if pairs else 0
    mean_err = round(sum(p["errPct"] for p in pairs) / len(pairs), 2) if pairs else 0
    return {
        "real": {**measured, "layers": device_record.get("layers", [])},
        "pairs": pairs, "withinPct": within_pct, "meanErrPct": mean_err, "targetPct": target_pct,
    }


def file_sink(prediction_set, out_dir, run_id):
    out = Path(out_dir) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.json"
    path.write_text(json.dumps(prediction_set, indent=2))
    return path


def dashboard_sink(prediction_set, dashboard_data_dir, workload_id):
    out = Path(dashboard_data_dir) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{workload_id}.json"
    path.write_text(json.dumps(prediction_set, indent=2))
    return path
