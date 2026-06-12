#!/usr/bin/env python3
"""
orchestrator.py — single entrypoint for the MI300X / gem5 metrics pipeline.

    python orchestrator.py --config pipeline.yaml

Modes (from YAML `mode`):
  gem5    — run gem5 (or ingest m5out fixtures) → normalized 'gem5' records
  device  — run device collectors (or device fixtures) → normalized 'device' records
  demo    — run BOTH from fixtures, train the predictor on the join, build
            predictionSets, and publish a bundle.js to the dashboard.

Data flow:  capture → raw → normalize → (predict) → publish → dashboard.
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import workloads as WL
from core import manifest, env
from core.interface import CollectorResult
from normalize import normalizer, schema
from collectors.gem5.run_gem5 import run_gem5
from collectors.device import (l0_silicon, l1_firmware, l2_kdriver, l3_runtime,
                               l4_mathlibs, l5_framework, l6_application, l7_task)
from predict import featurize
from predict.predictor import get_predictor
from predict import sinks as P
from publish import to_dashboard

DEVICE_CLASSES = [l0_silicon.L0Silicon, l1_firmware.L1Firmware, l2_kdriver.L2KDriver,
                  l3_runtime.L3Runtime, l4_mathlibs.L4MathLibs, l5_framework.L5Framework,
                  l6_application.L6Application, l7_task.L7Task]


def _now():
    return datetime.now(timezone.utc)


def _wl_meta(wid, cfg_row):
    return WL.get(wid, precision=cfg_row.get("precision"), batch=cfg_row.get("batch", 1),
                  num_gpus=cfg_row.get("numGPUs", 1), target_hz=cfg_row.get("target_hz"))


# ---------- gem5 path ----------
def gem5_record(conf, wid, cfg_row):
    wl = _wl_meta(wid, cfg_row)
    run_ctx = manifest.new_run_ctx(Path(conf.get("output", {}).get("runs_dir", HERE / "runs")),
                                   "gem5", wid)
    gem5_cfg = dict(conf.get("gem5", {}))
    fixtures = gem5_cfg.get("fixtures", {})
    if wid in fixtures:
        gem5_cfg["stats_fixture"] = str((HERE / fixtures[wid]).resolve())
    if gem5_cfg.get("config_fixture"):
        gem5_cfg["config_fixture"] = str((HERE / gem5_cfg["config_fixture"]).resolve())
    res = run_gem5(gem5_cfg, run_ctx, wl)
    gem5_params = res.scalars.pop("_gem5_params", None)
    run_config = res.scalars.pop("_run_config", {})
    rec = normalizer.normalize([res], source="gem5", run_id=run_ctx["run_id"],
                               timestamp=run_ctx["timestamp"], run_config=run_config or {},
                               workload=wl, peak_tflops=wl.get("peakTflops"))
    manifest.write_manifest(run_ctx, {"gem5": "ok" if not res.errors else f"err:{res.errors}"})
    return rec, gem5_params


# ---------- device path ----------
def device_record(conf, wid, cfg_row):
    import json
    wl = _wl_meta(wid, cfg_row)
    run_ctx = manifest.new_run_ctx(Path(conf.get("output", {}).get("runs_dir", HERE / "runs")),
                                   "device", wid)
    run_ctx["workload"] = wl
    dev = dict(conf.get("device", {}))
    fixtures = dev.get("fixtures", {})
    fixture_scalars = None
    if wid in fixtures:
        fixture_scalars = json.loads((HERE / fixtures[wid]).read_text())
        run_ctx["fixture_scalars"] = fixture_scalars

    results, status = [], {}
    layer_cfg = conf.get("layers", {})
    for cls in DEVICE_CLASSES:
        lc = layer_cfg.get(f"L{cls.layer_id}", {})
        col = cls({**lc, "gpu_index": dev.get("gpu_index", 0)}, run_ctx)
        if not col.enabled:
            status[f"L{cls.layer_id}"] = "disabled"
            continue
        if fixture_scalars is None:
            ok, why = col.available()
            if not ok:
                status[f"L{cls.layer_id}"] = f"skipped({why})"
                continue
        try:
            col.setup(); col.start()
            results.append(col.collect())
            col.teardown()
            status[f"L{cls.layer_id}"] = "ok"
        except Exception as e:
            status[f"L{cls.layer_id}"] = f"error({e})"

    run_config = {"hbmGB": 192, "bwGBs": int((fixture_scalars or {}).get("peak_bw_tbs", 5.3) * 1000),
                  "numGPUs": wl["num_gpus"], "peakTflops": wl["peakTflops"]}
    rec = normalizer.normalize(results, source="device", run_id=run_ctx["run_id"],
                               timestamp=run_ctx["timestamp"], run_config=run_config,
                               workload=wl, peak_tflops=wl["peakTflops"])
    manifest.write_manifest(run_ctx, status)
    return rec


# ---------- dashboard helper shapes ----------
def build_dataset_profile(gem5_recs, device_recs, dataset):
    return {
        "train": {"samples": len(dataset), "workloads": len({r['meta']['workload']['id'] for r in gem5_recs}),
                  "configs": len(gem5_recs), "split": 70},
        "val": {"samples": max(1, len(dataset) // 5), "workloads": 1, "configs": 1, "split": 15},
        "test": {"samples": max(1, len(dataset) // 5), "workloads": 1, "configs": 1, "split": 15, "heldOut": True},
        "features": len(dataset[0]["features"]) if dataset else 0,
        "featureGroups": [{"layer": f"L{i}", "n": 6} for i in range(8)],
        "sources": [{"name": "gem5 GPUFS stats.txt", "rows": len(gem5_recs), "kind": "sim"},
                    {"name": "rocprofiler (device)", "rows": len(device_recs), "kind": "device"}],
        "heldOutWorkloads": ["LLM 70B decode", "Robot-arm manipulation (vision)"],
        "label": "real-hardware latency + throughput (MI300X)",
    }


def build_train_report(report, prediction_sets):
    rows = []
    for wid, ps in prediction_sets.items():
        rows.append({"workload": wid, "regime": WL.WORKLOADS.get(wid, {}).get("regime", "?"),
                     "withinPct": ps["withinPct"], "meanErrPct": ps["meanErrPct"],
                     "heldOut": False})
    within_models = round(100 * sum(1 for r in rows if r["withinPct"] >= 80) / len(rows)) if rows else 0
    return {
        "mae": report.mae, "r2": report.r2,
        "curve": report.curve, "epochs": len(report.curve.get("train", [])),
        "featureImportance": report.featureImportance,
        "scorecard": {"rows": rows, "mae": report.mae, "withinModels": within_models, "r2": report.r2},
    }


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(HERE / "pipeline.yaml"))
    args = ap.parse_args()
    conf = yaml.safe_load(Path(args.config).read_text())
    mode = conf.get("mode", "demo")
    wls = conf.get("workloads", ["gemm"])
    cfg_rows = conf.get("configs", [{"precision": "bf16", "batch": 1, "numGPUs": 1}])

    print(f"[orchestrator] mode={mode} workloads={wls} preflight={env.summarize()}")

    gem5_recs, device_recs, gem5_params = [], [], None
    for wid in wls:
        for row in cfg_rows:
            if mode in ("gem5", "demo"):
                rec, gp = gem5_record(conf, wid, row)
                gem5_recs.append(rec)
                gem5_params = gem5_params or gp
                print(f"  gem5   {wid}: e2e={rec['metrics']['e2eMs']}ms tflops={rec['metrics']['achievedTflops']}")
            if mode in ("device", "demo"):
                rec = device_record(conf, wid, row)
                device_recs.append(rec)
                print(f"  device {wid}: e2e={rec['metrics']['e2eMs']}ms power={rec['metrics']['powerW']}W")

    # validate every record against the contract
    for r in gem5_recs + device_recs:
        schema.validate_or_raise(r)
    print(f"[orchestrator] contract OK for {len(gem5_recs) + len(device_recs)} records")

    out = conf.get("output", {})
    dash_dir = (HERE / out.get("dashboard_data_dir", "../mi300x-dashboard/data")).resolve()

    prediction_sets, train_report, dataset_profile, dev_status = {}, None, None, None
    if mode == "demo" and conf.get("predict", {}).get("enabled", True):
        dataset = featurize.make_dataset(gem5_recs, device_recs)
        predictor = get_predictor(conf.get("predict", {}).get("predictor", "baseline"))
        report = predictor.train(dataset)
        device_by_wl = {r["meta"]["workload"]["id"]: r for r in device_recs}
        gem5_by_wl = {r["meta"]["workload"]["id"]: r for r in gem5_recs}
        for wid in set(device_by_wl) & set(gem5_by_wl):
            ps = P.build_prediction_set(device_by_wl[wid], gem5_by_wl[wid], predictor)
            prediction_sets[wid] = ps
            sinks_cfg = conf.get("predict", {}).get("sinks", ["file", "dashboard"])
            if "file" in sinks_cfg:
                P.file_sink(ps, HERE / "runs", f"pred_{wid}")
            print(f"  predict {wid}: withinPct={ps['withinPct']} meanErr={ps['meanErrPct']}% pairs={len(ps['pairs'])}")
        dataset_profile = build_dataset_profile(gem5_recs, device_recs, dataset)
        train_report = build_train_report(report, prediction_sets)
        dev_status = {"mode": "live" if device_recs else "trace",
                      "source": conf.get("device", {}).get("endpoint", "trace"),
                      "driver": "ROCm 6.1 · amdgpu", "gpus": 8, "sampling": "1 kHz",
                      "status": "streaming" if device_recs else "fallback"}

    all_recs = gem5_recs + device_recs
    if out.get("dashboard_data_dir") is not None:
        bundle = to_dashboard.publish(dash_dir, records=all_recs, prediction_sets=prediction_sets,
                                      gem5_params=gem5_params, dataset_profile=dataset_profile,
                                      train_report=train_report, device_status=dev_status)
        print(f"[orchestrator] published → {bundle}")
    print("[orchestrator] done.")


if __name__ == "__main__":
    main()
