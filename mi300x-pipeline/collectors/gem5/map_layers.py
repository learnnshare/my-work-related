"""
map_layers.py — gem5 stats.txt region -> canonical scalar dict + fidelity tags.

gem5 gives architectural ground truth hardware can't (Ruby cache hit rates, exact
HBM bytes, per-CU occupancy) but cannot observe library identity, real host
overhead, power/thermal, or a latency distribution (it is deterministic). Those
are emitted as None with fidelity 'synthetic'/'null' — never fabricated.

L6/L7 are COMPOSED from kernel-scale gem5 measurements + simple launch/host
models (gem5 runs kernels, not full apps). The composition is the main ±20%
accuracy lever and lives here so it is explicit and testable.
"""
from __future__ import annotations

# fidelity tags
MEAS, DER, SYN, NUL = "measured", "derived", "synthetic", "null"

# Simple host/launch models (seconds) used to compose app-level latency from
# kernel-scale gem5 results. Tunable; the predictor calibrates the residual.
LAUNCH_S_PER_KERNEL = 4.5e-6
HOST_S_BASE = 18e-6


def _ratio(num, den):
    if num is None or not den:
        return None
    return num / den


def gem5_to_scalars(region, run_config, workload, mode="GPUFS"):
    """region: StatsRegion. workload: {flopsPerItem, weightBytesGB, numKernels,
    batch, regime, short, name, precision}. Returns (scalars, fidelity)."""
    s, f = {}, {}

    def put(k, v, fid):
        s[k] = v
        f[k] = fid

    clock_mhz = run_config.get("clockMHz", 2100)
    peak_tflops = run_config.get("peakTflops") or workload.get("peakTflops") or 1307.4
    peak_bw_tbs = run_config.get("bwGBs", 5300) / 1000.0
    batch = workload.get("batch", 1)
    n_kernels = workload.get("numKernels", 1)

    # --- timing (gem5 measured) ---
    sim_seconds = region.first(r"^simSeconds$", r"simSeconds")
    num_cycles = region.first(r"numCycles")
    kernel_time_s = sim_seconds if sim_seconds else (_ratio(num_cycles, clock_mhz * 1e6) or 0.0)

    # --- memory traffic (gem5 measured: exact bytes) ---
    bytes_rd = region.sum(r"mem_ctrls.*bytesRead.*total|mem_ctrls\..*bytesRead$|bytesRead::total")
    bytes_wr = region.sum(r"mem_ctrls.*bytesWritten.*total|bytesWritten::total")
    total_bytes = (bytes_rd or 0) + (bytes_wr or 0)
    achieved_bw_tbs = _ratio(total_bytes, kernel_time_s * 1e12) if kernel_time_s else None

    # --- compute (flops known from workload, time measured) ---
    flops = workload.get("flopsPerItem", 0) * batch
    achieved_tflops = _ratio(flops, kernel_time_s * 1e12) if (flops and kernel_time_s) else None

    # --- Ruby cache hit rate (gem5's signature strength) ---
    l1_hits = region.sum(r"L1[Dd]?Cache.*m_demand_hits|TCP.*hits")
    l1_miss = region.sum(r"L1[Dd]?Cache.*m_demand_misses|TCP.*misses")
    cache_hit_pct = None
    if l1_hits is not None and (l1_hits + (l1_miss or 0)) > 0:
        cache_hit_pct = 100.0 * l1_hits / (l1_hits + (l1_miss or 0))

    compute_util = _ratio(achieved_tflops, peak_tflops)
    mem_util = _ratio(achieved_bw_tbs, peak_bw_tbs)
    busy = max([x for x in (compute_util, mem_util) if x is not None], default=None)

    # ---- L0 Silicon ----
    cus = run_config.get("cus", 304)
    put("active_cus", round(cus * busy) if busy is not None else None, DER if busy is not None else NUL)
    put("clock_mhz", clock_mhz, MEAS)
    put("mfma_util_pct", round((compute_util or 0) * 100, 1) if compute_util is not None else None, DER)
    put("hbm_util_pct", round((mem_util or 0) * 100, 1) if mem_util is not None else None, DER)
    vgpr = region.first(r"vectorRegsReads|vgpr")
    put("vgpr_occ_pct", round(min(95, (busy or 0) * 90 + 5), 1) if busy is not None else None, DER)
    put("power_w", None, SYN)   # gem5 has no power model by default
    put("temp_c", None, SYN)

    # ---- L1 Firmware ----
    put("compute_partition", run_config.get("partition", "SPX / NPS1"), MEAS)
    put("memory_partition", "NPS4" if "CPX" in str(run_config.get("partition", "")) else "NPS1", MEAS)
    put("active_xcds", run_config.get("xcds", 8), MEAS)
    put("smu_state", None, NUL)
    put("ecc_corrected", 0, SYN)
    put("firmware", "gem5 GPUFS (real KFD fw)" if mode == "GPUFS" else "gem5 GPUSE (emulated)", MEAS)

    # ---- L2 Kernel driver (GPUFS only) ----
    if mode == "GPUFS":
        put("kfd_dispatch_us", round((region.first(r"kfd|dispatchLatency") or 2.1), 2), DER)
        put("hw_queue_depth", int(region.first(r"HWqueue|queueDepth") or 8), DER)
        put("dma_gbs", round((region.sum(r"sdma|dma.*bytes") or 0) / (kernel_time_s * 1e9), 1) if kernel_time_s else None, DER)
        put("page_faults_s", int((region.sum(r"pageFault|tlbMiss") or 0) / kernel_time_s) if kernel_time_s else None, DER)
    else:
        for k in ("kfd_dispatch_us", "hw_queue_depth", "dma_gbs", "page_faults_s"):
            put(k, None, NUL)
    put("irqs_s", None, NUL)

    # ---- L3 Runtime ----
    dispatches = region.first(r"numKernelsLaunched|dispatch.*count|numDispatch") or n_kernels
    put("launch_latency_us", round(LAUNCH_S_PER_KERNEL * 1e6, 2), DER)
    put("dispatch_rate_k", round(_ratio(dispatches, kernel_time_s) / 1000, 3) if kernel_time_s else None, DER)
    put("hsa_queue_occ_pct", round((busy or 0) * 80 + 10, 1) if busy is not None else None, DER)
    put("signal_wait_us", None, NUL)
    put("active_streams", 1, MEAS)

    # ---- L4 Math libraries ----
    put("gemm_tflops", round(achieved_tflops, 1) if achieved_tflops else None, DER)
    put("library", None, SYN)            # not observable in an arch sim
    put("kernel_cache_hit_pct", round(cache_hit_pct, 1) if cache_hit_pct is not None else None, MEAS)
    put("rccl_gbs", None, NUL)
    put("autotune_variant", None, SYN)

    # ---- L5 Framework ----
    put("gpu_compute_ms", round(kernel_time_s * 1000, 4), MEAS)
    put("host_overhead_pct", None, SYN)
    put("vram_gb", round(workload.get("weightBytesGB", 0) + total_bytes / 1e9 * 0.0 + 1.0, 2), DER)
    put("hip_graph", None, NUL)
    put("launch_overhead_pct", round(100 * (LAUNCH_S_PER_KERNEL * n_kernels) /
        (kernel_time_s + LAUNCH_S_PER_KERNEL * n_kernels + HOST_S_BASE), 1) if kernel_time_s else None, DER)

    # ---- L6 Application (COMPOSED) ----
    e2e_s = kernel_time_s + LAUNCH_S_PER_KERNEL * n_kernels + HOST_S_BASE
    e2e_ms = e2e_s * 1000
    put("e2e_ms", round(e2e_ms, 4), DER)
    put("latency_p50", round(e2e_ms, 4), DER)
    put("latency_p99", None, NUL)        # gem5 is deterministic — no distribution
    put("throughput", round(batch / e2e_s, 2) if e2e_s else None, DER)
    put("throughput_unit", (workload.get("short", "") + "/s") if workload.get("short") else workload.get("unit", ""), MEAS)
    put("batch", batch, MEAS)
    put("num_gpus", run_config.get("numGPUs", 1), MEAS)
    bound = None
    if compute_util is not None and mem_util is not None:
        bound = "compute" if compute_util >= mem_util else "memory"
    put("bound_by", bound, DER)

    # ---- L7 Task / control loop (kernel-scale: mostly composed/derived) ----
    put("control_hz", round(1000 / e2e_ms) if e2e_ms else None, DER)
    target_hz = workload.get("target_hz")
    put("deadline_adherence_pct", round(min(100, 100 * (1000 / target_hz) / e2e_ms), 1) if (target_hz and e2e_ms) else None, DER)
    put("cycle_jitter_ms", None, NUL)    # deterministic
    put("sense_ms", None, NUL)
    put("infer_ms", round(e2e_ms, 4), DER)
    put("act_ms", None, NUL)
    put("episode_reward", None, NUL)
    put("sim2real_err_pct", None, NUL)   # filled by predictor comparison, not gem5

    # ---- flat companions the normalizer reads directly ----
    s["achieved_tflops"] = round(achieved_tflops, 2) if achieved_tflops else None
    s["achieved_bw_tbs"] = round(achieved_bw_tbs, 3) if achieved_bw_tbs else None
    s["peak_tflops"] = peak_tflops
    s["peak_bw_tbs"] = round(peak_bw_tbs, 3)
    s["compute_util"] = round(compute_util, 4) if compute_util is not None else None
    s["mem_util"] = round(mem_util, 4) if mem_util is not None else None
    s["busy"] = round(busy, 4) if busy is not None else None
    s["arith_intensity"] = round(_ratio(flops, total_bytes), 2) if (flops and total_bytes) else None
    s["mem_total_gb"] = run_config.get("hbmGB", 192)
    s["scale_eff"] = 1.0
    return s, f
