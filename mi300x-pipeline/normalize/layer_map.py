"""
layer_map.py — the SINGLE source of truth for the dashboard data contract.

Every metric key, unit, max, and fmt here must match the dashboard's
assets/sim.js buildLayers() output byte-for-byte (plus the new L7). Both the
device and gem5 normalizers map their raw values onto these slots, so the
dashboard renders real data with zero chart/DOM changes.

A metric slot:
    {"k": <exact dashboard key>, "unit"?: str, "max"?: float, "fmt"?: str,
     "src": <internal field name the normalizer fills>}

`fmt` mirrors sim.js: 'count' | 'text' | 'k' | 'big' | None.
`src` is the internal canonical name produced by collectors/parsers; the
normalizer copies record[src] -> metric "v".
"""

# Static descriptors (match sim.js names/subs exactly for L0..L6).
LAYER_META = {
    0: ("L0 · Silicon / Microarchitecture", "CDNA 3 · 304 CUs · 8 XCDs · HBM3"),
    1: ("L1 · Firmware / HW Abstraction", "SMU · partitioning · ECC"),
    2: ("L2 · Kernel Driver (amdgpu / KFD)", "queues · DMA · page faults"),
    3: ("L3 · Runtime (ROCr / HSA)", "kernel dispatch · signals · queues"),
    4: ("L4 · Math Libraries", "rocBLAS · hipBLASLt · MIOpen · RCCL"),
    5: ("L5 · Framework (PyTorch + HIP)", "ops · host overhead · memory"),
    6: ("L6 · Application / Workload", "end-to-end"),
    7: ("L7 · Task / Control Loop", "Physical AI · real-time control"),
}

# Per-layer metric slots. `src` keys are the canonical names collectors emit.
LAYER_METRICS = {
    0: [
        {"k": "Active CUs", "fmt": "count", "max": 304, "src": "active_cus"},
        {"k": "Engine clock", "unit": "MHz", "src": "clock_mhz"},
        {"k": "MFMA / matrix-core util", "unit": "%", "src": "mfma_util_pct"},
        {"k": "HBM3 bandwidth util", "unit": "%", "src": "hbm_util_pct"},
        {"k": "VGPR occupancy", "unit": "%", "src": "vgpr_occ_pct"},
        {"k": "Board power", "unit": "W", "max": 750, "src": "power_w"},
        {"k": "Junction temp", "unit": "°C", "max": 95, "src": "temp_c"},
    ],
    1: [
        {"k": "Compute partition", "fmt": "text", "src": "compute_partition"},
        {"k": "Memory partition", "fmt": "text", "src": "memory_partition"},
        {"k": "Active XCDs", "fmt": "count", "max": 8, "src": "active_xcds"},
        {"k": "SMU power state", "fmt": "text", "src": "smu_state"},
        {"k": "ECC corrected errors", "fmt": "count", "src": "ecc_corrected"},
        {"k": "Firmware", "fmt": "text", "src": "firmware"},
    ],
    2: [
        {"k": "KFD dispatch latency", "unit": "µs", "src": "kfd_dispatch_us"},
        {"k": "HW queue depth", "fmt": "count", "max": 16, "src": "hw_queue_depth"},
        {"k": "DMA H2D/D2H", "unit": "GB/s", "src": "dma_gbs"},
        {"k": "Page faults / s", "fmt": "count", "src": "page_faults_s"},
        {"k": "IRQs / s", "fmt": "count", "src": "irqs_s"},
    ],
    3: [
        {"k": "Kernel-launch latency", "unit": "µs", "src": "launch_latency_us"},
        {"k": "Dispatch rate", "fmt": "k", "src": "dispatch_rate_k"},
        {"k": "HSA queue occupancy", "unit": "%", "src": "hsa_queue_occ_pct"},
        {"k": "Signal wait", "unit": "µs", "src": "signal_wait_us"},
        {"k": "Active streams", "fmt": "count", "src": "active_streams"},
    ],
    4: [
        {"k": "GEMM achieved", "unit": "TFLOPS", "src": "gemm_tflops"},
        {"k": "Library", "fmt": "text", "src": "library"},
        {"k": "Kernel cache hit", "unit": "%", "src": "kernel_cache_hit_pct"},
        {"k": "RCCL bus BW", "unit": "GB/s", "src": "rccl_gbs"},
        {"k": "Autotune variant", "fmt": "text", "src": "autotune_variant"},
    ],
    5: [
        {"k": "GPU compute time", "unit": "ms", "src": "gpu_compute_ms"},
        {"k": "Host overhead", "unit": "%", "src": "host_overhead_pct"},
        {"k": "VRAM allocated", "unit": "GB", "max": 192, "src": "vram_gb"},
        {"k": "HIP graph capture", "fmt": "text", "src": "hip_graph"},
        {"k": "Launch overhead", "unit": "%", "src": "launch_overhead_pct"},
    ],
    6: [
        {"k": "End-to-end latency", "unit": "ms", "src": "e2e_ms"},
        {"k": "Throughput", "fmt": "big", "src": "throughput"},
        {"k": "Batch size", "fmt": "count", "src": "batch"},
        {"k": "GPUs", "fmt": "count", "src": "num_gpus"},
        {"k": "Bound by", "fmt": "text", "src": "bound_by"},
    ],
    7: [
        {"k": "Control rate achieved", "unit": "Hz", "src": "control_hz"},
        {"k": "Deadline adherence", "unit": "%", "max": 100, "src": "deadline_adherence_pct"},
        {"k": "Cycle jitter", "unit": "ms", "src": "cycle_jitter_ms"},
        {"k": "Sense", "unit": "ms", "src": "sense_ms"},
        {"k": "Infer (policy)", "unit": "ms", "src": "infer_ms"},
        {"k": "Act", "unit": "ms", "src": "act_ms"},
        {"k": "Episode reward", "fmt": "big", "src": "episode_reward"},
        {"k": "Sim→real error", "unit": "%", "max": 100, "src": "sim2real_err_pct"},
    ],
}

# Flat top-level metric keys the dashboard reads from computeMetrics().
FLAT_KEYS = [
    "e2eMs", "latencyP50", "latencyP99", "throughput", "throughputUnit",
    "achievedTflops", "peakTflops", "achievedBwTBs", "peakBwTBs",
    "computeUtil", "memUtil", "busy", "arithIntensity",
    "powerW", "tempC", "clockMHz", "memUsedGB", "memTotalGB",
    "scaleEff", "numGPUs", "boundBy",
]

# Sub-line for L6 is the workload name at runtime; default kept for fallback.
def layer_sub(layer_id, workload_name=None):
    if layer_id == 6 and workload_name:
        return workload_name
    return LAYER_META[layer_id][1]


def build_layers(scalars, workload_name=None, peak_tflops=None, fidelity=None):
    """Assemble the dashboard `layers[]` array from a flat `scalars` dict.

    scalars: {src_name: value} produced by the normalizer (numbers or strings,
             or None where a source genuinely cannot observe the metric).
    fidelity: optional {src_name: 'measured'|'derived'|'synthetic'|'null'} ->
              attached as `_f` (ignored by current renderer, used by future badges).
    """
    fidelity = fidelity or {}
    layers = []
    for lid in range(8):
        name, _ = LAYER_META[lid]
        metrics = []
        for slot in LAYER_METRICS[lid]:
            m = {"k": slot["k"]}
            v = scalars.get(slot["src"], None)
            m["v"] = v
            if "unit" in slot:
                m["unit"] = slot["unit"]
            if "fmt" in slot:
                m["fmt"] = slot["fmt"]
            if "max" in slot:
                # L4 GEMM achieved uses peak TFLOPS as its bar max
                m["max"] = peak_tflops if (slot["src"] == "gemm_tflops" and peak_tflops) else slot["max"]
            if slot["src"] in fidelity:
                m["_f"] = fidelity[slot["src"]]
            metrics.append(m)
        layers.append({"id": lid, "name": name, "sub": layer_sub(lid, workload_name), "metrics": metrics})
    return layers
