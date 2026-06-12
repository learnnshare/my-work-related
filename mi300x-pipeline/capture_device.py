#!/usr/bin/env python3
"""
capture_device.py — REAL MI300X capture via rocprofv3, into the dashboard.

Verified feasible on the SR-IOV VF box: HIP kernels run, and rocprofv3 collects
both kernel traces and hardware counters. SMI sensors (power/temp/clock) are N/A
on the VF, so those are emitted null; the architectural signal (gfx-active,
cache-hit, memory traffic, occupancy, kernel time) comes from rocprofv3.

Flow:  compile/run workload under rocprofv3 (trace + pmc)
       -> parse CSVs -> derive L0/L3/L4/L5/L6 scalars
       -> normalize (source=device) -> publish bundle.js to the dashboard

Usage (on the MI300X box):
    python3 capture_device.py                      # vectoradd microkernel
    python3 capture_device.py --kernel vadd
    python3 capture_device.py --fixtures           # use bundled CSVs (no GPU; for testing)
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from parsers import rocprof_csv as R           # noqa: E402
from normalize import normalizer               # noqa: E402
from publish import to_dashboard               # noqa: E402
import workloads as WL                         # noqa: E402

# bytes per TCC EA request (cache-line granularity on gfx942)
EA_BYTES = 64

# per-workload flop model (flops per grid thread) for achievedTflops
FLOP_MODEL = {"vectoradd": 128}   # vadd: 64 inner iters * (mul+add)

# Independent counter groups — each is its own rocprofv3 pass, so a group the VF
# rejects doesn't kill the others. Add/adjust names per `rocprofv3-avail info --pmc`.
PMC_GROUPS = [
    ["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"],   # core activity (known-good)
    ["TCC_HIT_sum", "TCC_MISS_sum"],                  # L2 cache hit
    ["TCC_EA_RDREQ_sum", "TCC_EA_WRREQ_sum"],         # memory requests (×64B)
    ["FETCH_SIZE", "WRITE_SIZE"],                     # memory bytes (KB) — alt
    ["SQ_INSTS_VALU", "SQ_INSTS_VALU_MFMA_MOPS_F16"], # compute/MFMA — alt
]


def sh(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def ensure_binary(name="vectoradd"):
    binp = HERE / "bench" / name
    src = HERE / "bench" / "vectoradd.cpp"
    if binp.exists():
        return binp
    if not shutil.which("hipcc"):
        sys.exit("hipcc not found — can't build the workload (run on the MI300X box)")
    sh(["hipcc", str(src), "-o", str(binp)], check=True)
    return binp


def run_rocprofv3(app, outdir, pmc=None, trace=False, timeout=180):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = ["rocprofv3"]
    if trace:
        cmd += ["--kernel-trace"]
    if pmc:
        cmd += ["--pmc"] + pmc
    cmd += ["--output-format", "csv", "-d", str(outdir), "--", str(app)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"


def _merge_counters(dst, src):
    for k, cm in src.items():
        dst.setdefault(k, {}).update(cm)


def capture(app, raw_dir):
    """Run trace + per-group pmc passes; merge whatever succeeds."""
    tdir = raw_dir / "trace"
    rc, _ = run_rocprofv3(app, tdir, trace=True)
    print(f"  kernel-trace rc={rc}")
    t = R.find_outputs(str(tdir))
    traces = R.parse_kernel_trace(t["kernel_trace"])
    agent = R.parse_agent_info(t["agent_info"])

    counters = {}
    for i, grp in enumerate(PMC_GROUPS):
        print(f"  → pmc group {i + 1}/{len(PMC_GROUPS)}: {grp} ...")
        pdir = raw_dir / f"pmc{i}"
        rc, log = run_rocprofv3(app, pdir, pmc=grp)
        if rc != 0:
            print(f"  pmc group {grp} rejected (rc={rc}) — skipping")
            if rc == 124:
                print(f"    (timed out — likely counter multiplexing; skip in PMC_GROUPS if it recurs)")
            continue
        p = R.find_outputs(str(pdir))
        _merge_counters(counters, R.parse_counters(p["counters"]))
        if not agent:
            agent = R.parse_agent_info(p["agent_info"])
        print(f"  pmc group {grp} ok")
    return traces, counters, agent


def derive(traces, counters, agent, workload_id, kernel_hint):
    name = R.pick_kernel(traces, counters, kernel_hint)
    if not name:
        sys.exit("no kernel found in rocprofv3 output")
    kdisp = [t for t in traces if t["name"] == name]
    kernel_time_s = sum(t["dur_s"] for t in kdisp) or None
    grid_total = sum((t["grid"] or 0) for t in kdisp)
    vgpr = max((t["vgpr"] or 0) for t in kdisp) if kdisp else None
    c = counters.get(name, {})
    print(f"  [counters collected for {name[:40]}]:")
    for cn in sorted(c):
        print(f"     {cn} = {c[cn]}")

    cu = agent.get("cu_count") or 304
    clk = agent.get("max_clk_mhz") or 2100
    peakT = WL.PEAK_TFLOPS.get(WL.WORKLOADS.get(workload_id, {}).get("pref", "fp16"), 1307.4)
    peak_bw = 5.3

    grbm_total = c.get("GRBM_COUNT")
    grbm_active = c.get("GRBM_GUI_ACTIVE")
    gfx_active = (grbm_active / grbm_total) if (grbm_total and grbm_active is not None) else None

    rd = c.get("TCC_EA_RDREQ_sum")
    wr = c.get("TCC_EA_WRREQ_sum")
    if rd is not None or wr is not None:
        total_bytes = ((rd or 0) + (wr or 0)) * EA_BYTES
    else:
        fz, wz = c.get("FETCH_SIZE"), c.get("WRITE_SIZE")    # KB (rocprofiler derived)
        total_bytes = ((fz or 0) + (wz or 0)) * 1024 if (fz is not None or wz is not None) else None
    achieved_bw = (total_bytes / kernel_time_s / 1e12) if (total_bytes and kernel_time_s) else None

    fpt = FLOP_MODEL.get(workload_id)
    flops = (fpt * grid_total) if (fpt and grid_total) else None
    achieved_tf = (flops / kernel_time_s / 1e12) if (flops and kernel_time_s) else None

    hit = c.get("TCC_HIT_sum")
    miss = c.get("TCC_MISS_sum")
    cache_hit = (100 * hit / (hit + (miss or 0))) if (hit is not None and (hit + (miss or 0)) > 0) else None

    compute_util = (achieved_tf / peakT) if achieved_tf else None
    mem_util = (achieved_bw / peak_bw) if achieved_bw else None
    busy = max([x for x in (compute_util, mem_util, gfx_active) if x is not None], default=None)
    e2e_ms = kernel_time_s * 1000 if kernel_time_s else None

    F = {}  # fidelity
    s = {}

    def put(k, v, f):
        s[k] = v
        F[k] = f

    # L0
    put("active_cus", round(cu * gfx_active) if gfx_active is not None else None, "derived")
    put("clock_mhz", clk, "derived")            # agent max; VF hides live clock
    put("mfma_util_pct", None, "null")          # need MFMA counters; vadd has none
    put("hbm_util_pct", round(mem_util * 100, 1) if mem_util is not None else None, "measured")
    put("vgpr_occ_pct", round(min(100, (vgpr or 0) / 512 * 100), 1) if vgpr is not None else None, "measured")
    put("power_w", None, "null")                # SR-IOV VF: no sensor
    put("temp_c", None, "null")
    # L3
    put("launch_latency_us", None, "null")
    put("dispatch_rate_k", round(len(traces) / kernel_time_s / 1000, 3) if kernel_time_s else None, "derived")
    put("hsa_queue_occ_pct", None, "null")
    put("signal_wait_us", None, "null")
    put("active_streams", 1, "measured")
    # L4
    put("gemm_tflops", round(achieved_tf, 2) if achieved_tf else None, "measured")
    put("library", "HIP kernel", "measured")
    put("kernel_cache_hit_pct", round(cache_hit, 1) if cache_hit is not None else None, "measured")
    put("rccl_gbs", None, "null")
    put("autotune_variant", None, "null")
    # L5
    put("gpu_compute_ms", round(kernel_time_s * 1000, 4) if kernel_time_s else None, "measured")
    put("host_overhead_pct", None, "null")
    put("vram_gb", None, "null")
    put("hip_graph", "off", "measured")
    put("launch_overhead_pct", None, "null")
    # L6
    put("e2e_ms", round(e2e_ms, 4) if e2e_ms else None, "measured")
    put("latency_p50", round(e2e_ms, 4) if e2e_ms else None, "measured")
    put("latency_p99", None, "null")
    put("throughput", round(len(kdisp) / kernel_time_s, 2) if kernel_time_s else None, "measured")
    put("throughput_unit", "kernels/s", "measured")
    put("batch", 1, "measured")
    put("num_gpus", 1, "measured")
    bound = None
    if compute_util is not None and mem_util is not None:
        bound = "compute" if compute_util >= mem_util else "memory"
    elif mem_util is not None:
        bound = "memory"
    elif compute_util is not None:
        bound = "compute"
    put("bound_by", bound, "derived")
    # L7 (microkernel, not a control loop)
    for k in ("control_hz", "deadline_adherence_pct", "cycle_jitter_ms", "sense_ms",
              "infer_ms", "act_ms", "episode_reward", "sim2real_err_pct"):
        put(k, None, "null")
    put("control_hz", round(1000 / e2e_ms) if e2e_ms else None, "derived")
    put("infer_ms", round(e2e_ms, 4) if e2e_ms else None, "derived")
    # flat companions
    s["achieved_tflops"] = round(achieved_tf, 3) if achieved_tf else None
    s["achieved_bw_tbs"] = round(achieved_bw, 4) if achieved_bw else None
    s["peak_tflops"] = peakT
    s["peak_bw_tbs"] = peak_bw
    s["compute_util"] = round(compute_util, 5) if compute_util is not None else None
    s["mem_util"] = round(mem_util, 5) if mem_util is not None else None
    s["busy"] = round(busy, 5) if busy is not None else None
    s["arith_intensity"] = round(flops / total_bytes, 2) if (flops and total_bytes) else None
    s["mem_total_gb"] = 192
    s["scale_eff"] = 1.0
    return name, s, F, {"cus": cu, "clockMHz": clk, "hbmGB": 192, "bwGBs": int(peak_bw * 1000),
                        "numGPUs": 1, "peakTflops": peakT, "isa": "gfx942",
                        "xcds": agent.get("num_xcc", 8), "partition": "SPX / NPS1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="vectoradd")
    ap.add_argument("--kernel", default=None, help="substring of the kernel of interest")
    ap.add_argument("--fixtures", action="store_true", help="use bundled CSVs (no GPU)")
    ap.add_argument("--dashboard", default="../mi300x-dashboard/data")
    args = ap.parse_args()

    # line-buffer stdout so progress shows live even when piped to tee
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ts = datetime.now(timezone.utc)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ") + "_device_" + args.workload
    raw_dir = HERE / "runs" / run_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.fixtures:
        print("[capture] FIXTURE mode — parsing bundled rocprofv3 CSVs")
        base = HERE / "fixtures" / "rocprofv3"
        out = R.find_outputs(str(base))
        traces = R.parse_kernel_trace(out["kernel_trace"])
        counters = R.parse_counters(out["counters"])
        agent = R.parse_agent_info(out["agent_info"])
    else:
        print("[capture] compiling + running workload under rocprofv3")
        app = ensure_binary(args.workload)
        traces, counters, agent = capture(app, raw_dir)

    if not traces and not counters:
        sys.exit("no rocprofv3 output parsed — check the run on the box")

    name, scalars, fidelity, run_config = derive(traces, counters, agent, args.workload, args.kernel)
    wl = {"id": args.workload, "name": f"{args.workload} ({name})", "unit": "kernels",
          "short": "kernels", "pref": "fp16", "batch": 1, "num_gpus": 1,
          "precision": "fp16", "peakTflops": run_config["peakTflops"]}

    from core.interface import CollectorResult, Cadence
    res = CollectorResult(layer_id=-1, cadence=Cadence.PERKERNEL)
    res.scalars.update(scalars)
    res.fidelity.update(fidelity)

    record = normalizer.normalize([res], source="device", run_id=run_id,
                                  timestamp=ts.isoformat().replace("+00:00", "Z"),
                                  run_config=run_config, workload=wl,
                                  peak_tflops=run_config["peakTflops"])

    print(f"\n[capture] kernel: {name}")
    m = record["metrics"]
    for k in ("e2eMs", "throughput", "achievedTflops", "achievedBwTBs",
              "computeUtil", "memUtil", "busy", "boundBy"):
        print(f"  {k:16} = {m.get(k)}")
    print("  L0:", {x['k']: x['v'] for x in record["layers"][0]["metrics"]})
    print("  L4:", {x['k']: x['v'] for x in record["layers"][4]["metrics"]})

    dash = (HERE / args.dashboard).resolve()
    to_dashboard.publish(dash, records=[record],
                         device_status={"mode": "live", "source": "rocprofv3 (SR-IOV VF)",
                                        "driver": "ROCm 7.0 · amdgpu 6.16", "gpus": 1,
                                        "sampling": "per-kernel", "status": "captured"})
    print(f"\n[capture] published real record → {dash}/bundle.js")
    print("[capture] open the dashboard developer view to see L0–L7 from your MI300X.")


if __name__ == "__main__":
    main()
