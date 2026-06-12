#!/usr/bin/env python3
"""
analyze.py — turn captured records into report-ready figures + a summary table.

Reads the normalized records the capture pipeline published
(mi300x-dashboard/data/records/*.json), de-duplicates to the latest run per
(workload, precision, size), and emits:
  - data/analysis/summary.md        markdown table
  - data/analysis/summary.csv       same, machine-readable
  - data/analysis/tflops_vs_size.png   TFLOPS vs GEMM size (per precision)
  - data/analysis/mfma_vs_size.png     MFMA util vs GEMM size
  - data/analysis/roofline.png         achieved TFLOPS vs arithmetic intensity

Pure analysis — runs anywhere the records are (no GPU needed).
Usage:  python3 analyze.py [--data ../mi300x-dashboard/data]
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_records(data_dir):
    """Latest record per (workload_id, precision, batch/size)."""
    recs = {}
    for f in sorted((data_dir / "records").glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        wl = r["meta"]["workload"]
        key = (wl.get("id"), wl.get("precision") or wl.get("pref"), wl.get("batch"))
        # filenames are timestamp-sorted, so later overwrites earlier -> latest wins
        recs[key] = r
    return list(recs.values())


def gemm_size(rec):
    """Pull the cube size from a GEMM record (batch encodes M=N=K, or parse name)."""
    wl = rec["meta"]["workload"]
    if isinstance(wl.get("batch"), int) and wl["batch"] > 1:
        return wl["batch"]
    m = re.search(r"(\d+)\^3", wl.get("name", ""))
    return int(m.group(1)) if m else None


def summarize(records):
    rows = []
    for r in records:
        m = r["metrics"]
        wl = r["meta"]["workload"]
        l0 = {x["k"]: x["v"] for x in r["layers"][0]["metrics"]}
        l4 = {x["k"]: x["v"] for x in r["layers"][4]["metrics"]}
        rows.append({
            "workload": wl.get("name", wl.get("id")),
            "precision": wl.get("precision") or wl.get("pref"),
            "size": gemm_size(r),
            "TFLOPS": m.get("achievedTflops"),
            "bandwidth_TBs": m.get("achievedBwTBs"),
            "MFMA_pct": l0.get("MFMA / matrix-core util"),
            "L2_hit_pct": l4.get("Kernel cache hit"),
            "bound_by": m.get("boundBy"),
            "e2e_ms": m.get("e2eMs"),
            "source": r["meta"]["source"],
        })
    rows.sort(key=lambda x: (str(x["workload"]), x["precision"] or "", x["size"] or 0))
    return rows


def write_tables(rows, outdir):
    cols = ["workload", "precision", "size", "TFLOPS", "bandwidth_TBs",
            "MFMA_pct", "L2_hit_pct", "bound_by", "e2e_ms", "source"]
    with open(outdir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["# MI300X captured results (real hardware)", "",
          "| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (outdir / "summary.md").write_text("\n".join(md) + "\n")


def make_plots(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gemms = [r for r in rows if r["size"] and r["TFLOPS"]]
    precisions = sorted({r["precision"] for r in gemms})

    # 1) TFLOPS vs size
    plt.figure(figsize=(7, 4.5))
    for p in precisions:
        pts = sorted([r for r in gemms if r["precision"] == p], key=lambda x: x["size"])
        if pts:
            plt.plot([r["size"] for r in pts], [r["TFLOPS"] for r in pts],
                     marker="o", label=p)
    plt.axhline(1307.4, ls="--", c="gray", alpha=.6, label="fp16/bf16 peak (1307)")
    plt.xlabel("GEMM size (M=N=K)"); plt.ylabel("achieved TFLOPS")
    plt.title("MI300X GEMM throughput vs size (real capture)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(outdir / "tflops_vs_size.png", dpi=130); plt.close()

    # 2) MFMA util vs size
    plt.figure(figsize=(7, 4.5))
    for p in precisions:
        pts = sorted([r for r in gemms if r["precision"] == p and r["MFMA_pct"] is not None],
                     key=lambda x: x["size"])
        if pts:
            plt.plot([r["size"] for r in pts], [r["MFMA_pct"] for r in pts],
                     marker="s", label=p)
    plt.xlabel("GEMM size (M=N=K)"); plt.ylabel("MFMA / matrix-core util (%)")
    plt.title("MI300X matrix-core utilization vs size")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(outdir / "mfma_vs_size.png", dpi=130); plt.close()

    # 3) roofline-ish: TFLOPS vs arithmetic intensity (all records)
    plt.figure(figsize=(7, 4.5))
    for r in rows:
        if r["TFLOPS"]:
            plt.scatter(r.get("size") or 1, r["TFLOPS"],
                        c=("tab:red" if r["bound_by"] == "memory" else "tab:blue"))
    plt.xscale("log"); plt.xlabel("GEMM size (log)"); plt.ylabel("achieved TFLOPS")
    plt.title("Throughput by size (blue=compute-bound, red=memory-bound)")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(outdir / "throughput_scatter.png", dpi=130); plt.close()
    return ["tflops_vs_size.png", "mfma_vs_size.png", "throughput_scatter.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../mi300x-dashboard/data")
    args = ap.parse_args()
    data_dir = (HERE / args.data).resolve()
    outdir = data_dir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_records(data_dir)
    rows = summarize(records)
    write_tables(rows, outdir)
    print(f"[analyze] {len(rows)} unique records")
    print("  " + " | ".join(["workload", "prec", "size", "TFLOPS", "MFMA%", "L2%", "bound"]))
    for r in rows:
        print(f"  {str(r['workload'])[:22]:22} {str(r['precision']):5} {str(r['size']):>6} "
              f"{str(r['TFLOPS']):>9} {str(r['MFMA_pct']):>6} {str(r['L2_hit_pct']):>5} {r['bound_by']}")
    try:
        imgs = make_plots(rows, outdir)
        print(f"[analyze] wrote figures: {', '.join(imgs)}")
    except Exception as e:
        print(f"[analyze] plotting skipped ({e}); tables still written")
    print(f"[analyze] outputs in {outdir}")


if __name__ == "__main__":
    main()
