#!/usr/bin/env python3
"""
capture_gem5.py — Path 2: run a tiny kernel in gem5 (gfx90a SE proxy) and emit a
`source:gem5` record in the same L0–L7 contract as the device path.

gem5 25.1 SE mode (apu_se.py) models gfx90a (MI200/CDNA2) — we use it as a CDNA
**proxy** for gfx942/MI300 (true gfx942 needs GPUFS+KVM). Records carry a
meta.run_config.proxy = "gfx90a→gfx942" flag and per-metric fidelity tags so the
dashboard shows what's measured vs derived vs unavailable. gem5 is ~1e4–1e5×
slower than hardware → keep kernels tiny.

Run on the gem5 box (after building gem5 + sourcing env.sh):
    python3 capture_gem5.py --workload vectoradd --size 2048 --cus 8
    python3 capture_gem5.py --fixtures      # parse the bundled synthetic stats.txt (no gem5)
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collectors.gem5 import stats_parser, config_extractor, map_layers   # noqa: E402
from core.interface import CollectorResult, Cadence                      # noqa: E402
from normalize import normalizer                                         # noqa: E402
from publish import to_dashboard                                         # noqa: E402
import workloads as WL                                                   # noqa: E402

GEM5_DIR = os.environ.get("GEM5_DIR", "/workspace/shared/gem5")
GEM5_BIN = os.environ.get("GEM5_BIN", f"{GEM5_DIR}/build/VEGA_X86/gem5.opt")
APU_CFG = os.environ.get("APU_CONFIG", f"{GEM5_DIR}/configs/example/apu_se.py")


def sh(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def build_kernel(workload, size, iters):
    prof = WL.WORKLOADS.get(workload, {})
    src = HERE / "bench" / ("vectoradd.cpp" if workload == "vectoradd" else "gemm.cpp")
    binp = HERE / "bench" / (workload + "_gem5")
    if not shutil.which("hipcc"):
        sys.exit("hipcc not found")
    libs = [] if workload == "vectoradd" else ["-lrocblas", "-I/opt/rocm/include", "-L/opt/rocm/lib"]
    sh(["hipcc", str(src), "-o", str(binp), *libs], check=True)
    return binp


def run_gem5(binp, args_str, gfx, cus, outdir, timeout):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [GEM5_BIN, f"--outdir={outdir}", APU_CFG, "--dgpu", "--gfx-version", gfx,
           "-u", str(cus), "-c", str(binp), "-o", args_str]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        print((r.stdout + r.stderr)[-1500:])
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"  gem5 timed out after {timeout}s (kernel too big? shrink --size)")
        return 124


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="vectoradd", choices=["vectoradd", "gemm"])
    ap.add_argument("--size", type=int, default=2048, help="n (vadd) or M=N=K (gemm) — keep TINY")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--gfx", default="gfx90a")
    ap.add_argument("--cus", type=int, default=8)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--fixtures", action="store_true", help="parse bundled synthetic stats.txt")
    ap.add_argument("--dashboard", default="../mi300x-dashboard/data")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ts = datetime.now(timezone.utc)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ") + "_gem5_" + args.workload
    raw = HERE / "runs" / run_id / "m5out"

    if args.fixtures:
        print("[gem5] FIXTURE mode — parsing bundled synthetic stats.txt")
        stats_path = HERE / "fixtures" / "gem5" / "stats.txt"
        cfg_path = HERE / "fixtures" / "gem5" / "config.json"
    else:
        if not Path(GEM5_BIN).exists():
            sys.exit(f"gem5 not found at {GEM5_BIN} (build it; source env.sh)")
        binp = build_kernel(args.workload, args.size, args.iters)
        a = (f"{args.size} {args.iters}" if args.workload == "vectoradd"
             else f"{args.size} {args.size} {args.size} {args.precision} {args.iters}")
        print(f"[gem5] running {args.workload} size={args.size} on gfx={args.gfx} CUs={args.cus} (proxy for gfx942)")
        rc = run_gem5(binp, a, args.gfx, args.cus, raw, args.timeout)
        stats_path = raw / "stats.txt"
        cfg_path = raw / "config.json"
        if rc != 0 or not stats_path.exists():
            sys.exit(f"[gem5] run failed (rc={rc}); no stats.txt. See output above.")

    region = stats_parser.load_region(stats_path, index=-1)
    if not region.d:
        sys.exit("[gem5] empty stats.txt — kernel didn't simulate")

    wl = WL.get(args.workload, precision=args.precision, batch=1)
    # map_layers computes flops = flopsPerItem × batch and bytes = actBytesPerItem × batch.
    # GEMM: keep batch=size (so size shows in the record key) ⇒ flopsPerItem = 2·size²
    #   (× size = 2·size³ total) and actBytesPerItem = 6·size (× size = 6·size², 3 fp16 mats).
    # vadd: batch=1, flops = 128·size, bytes = 12·size (a,b,c float32).
    if args.workload == "gemm":
        wl["batch"] = args.size
        wl["flopsPerItem"] = 2 * args.size ** 2
        wl["actBytesPerItem"] = 6 * args.size
    else:
        wl["batch"] = 1
        wl["flopsPerItem"] = 128 * args.size
        wl["actBytesPerItem"] = 12 * args.size
    wl["numKernels"] = args.iters
    # gfx90a proxy run-config, tuned toward MI300 peaks for comparable ratios
    run_config = {"isa": args.gfx, "cus": args.cus, "xcds": 8, "clockMHz": 2100,
                  "hbmGB": 192, "bwGBs": 5300, "peakTflops": wl["peakTflops"],
                  "partition": "SE", "numGPUs": 1, "proxy": "gfx90a→gfx942",
                  "fidelity_note": "gem5 SE models MI200/CDNA2; CDNA proxy for MI300"}
    scalars, fidelity = map_layers.gem5_to_scalars(region, run_config, wl, mode="GPUSE")

    res = CollectorResult(layer_id=-1, cadence=Cadence.PERKERNEL)
    res.scalars.update(scalars)
    res.fidelity.update(fidelity)
    rec = normalizer.normalize([res], source="gem5", run_id=run_id,
                               timestamp=ts.isoformat().replace("+00:00", "Z"),
                               run_config=run_config, workload=wl,
                               peak_tflops=wl["peakTflops"])

    print(f"\n[gem5] record: {wl['name']}  (proxy gfx90a→gfx942)")
    for k in ("e2eMs", "achievedTflops", "achievedBwTBs", "computeUtil", "memUtil", "boundBy"):
        print(f"  {k:16} = {rec['metrics'].get(k)}")
    print("  L4 cache hit (gem5 strength):",
          next((m["v"] for m in rec["layers"][4]["metrics"] if m["k"] == "Kernel cache hit"), None))

    dash = (HERE / args.dashboard).resolve()
    records = [rec]
    if args.keep:
        import json
        for f in sorted((dash / "records").glob("*.json")):
            try:
                prev = json.loads(f.read_text())
                if prev["meta"]["run_id"] != run_id:
                    records.append(prev)
            except Exception:
                pass
    to_dashboard.publish(dash, records=records,
                         device_status={"mode": "live", "source": "gem5 gfx90a (proxy)",
                                        "driver": "gem5 25.1 VEGA_X86", "gpus": 1,
                                        "sampling": "kernel-region", "status": "simulated"})
    print(f"\n[gem5] published {len(records)} record(s) → {dash}/bundle.js")


if __name__ == "__main__":
    main()
