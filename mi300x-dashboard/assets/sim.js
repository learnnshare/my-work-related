/* =====================================================================
   MI300X Dashboard — Simulation Engine (shared)
   ---------------------------------------------------------------------
   Generates plausible-but-fake metrics from the UI knobs. Numbers are
   illustrative ONLY; they encode the SHAPE of real behaviour (roofline
   limits, launch-bound batch-1 latency, multi-GPU scaling losses) so the
   mockup feels real. Do not cite these as benchmarks.
   ===================================================================== */

const MI300X = {
  name: 'AMD Instinct MI300X',
  arch: 'CDNA 3 (gfx942)',
  cus: 304,
  xcds: 8,
  memGB: 192,
  hbmTBs: 5.3,          // HBM3 peak bandwidth, TB/s
  baseClockMHz: 2100,
  boardWatts: 750,
  // Peak compute by precision, TFLOPS (spec-sheet ceilings)
  peak: { fp32: 163.4, tf32: 653.7, fp16: 1307, bf16: 1307, fp8: 2614.9 },
};

/* ---- Workload library -------------------------------------------------
   Each preset captures the rough computational "personality" of a job:
   flops/byte per item, fixed model footprint, kernel count, host overhead.
*/
const WORKLOADS = {
  ppo_infer: {
    name: 'PPO Policy Inference (batch-1 robot control)',
    unit: 'inferences', short: 'inf',
    flopsPerItem: 1.5e6, actBytesPerItem: 2.0e5, weightBytesGB: 0.004,
    numKernels: 6, launchUs: 4.5, cpuUs: 18, pref: 'fp16',
    regime: 'launch-bound', cpuBound: 0.05,
    note: 'Tiny MLP. Latency is dominated by kernel-launch + host dispatch, not FLOPs.',
  },
  ppo_train: {
    name: 'PPO Training (MuJoCo Walker2d)',
    unit: 'env-steps', short: 'steps',
    flopsPerItem: 6.0e6, actBytesPerItem: 1.2e6, weightBytesGB: 0.01,
    numKernels: 22, launchUs: 4.0, cpuUs: 140, pref: 'bf16',
    regime: 'cpu-bound', cpuBound: 0.7,
    note: 'Physics runs on CPU. The GPU is rarely the bottleneck — end-to-end speedup is gated by the host.',
  },
  llm7b: {
    name: 'LLM Inference — 7B (decode)',
    unit: 'tokens', short: 'tok',
    flopsPerItem: 1.4e10, actBytesPerItem: 1.4e7, weightBytesGB: 14,
    numKernels: 120, launchUs: 3.2, cpuUs: 35, pref: 'fp16',
    regime: 'memory-bound', cpuBound: 0.08,
    note: 'Decode is HBM-bandwidth bound: weights are re-streamed every token. MI300X bandwidth shines here.',
  },
  llm70b: {
    name: 'LLM Inference — 70B (decode)',
    unit: 'tokens', short: 'tok',
    flopsPerItem: 1.4e11, actBytesPerItem: 6.0e7, weightBytesGB: 140,
    numKernels: 560, launchUs: 3.0, cpuUs: 40, pref: 'fp8',
    regime: 'memory-bound', cpuBound: 0.05,
    note: 'Fits in 192 GB where most GPUs cannot. Bandwidth- and capacity-bound.',
  },
  gemm: {
    name: 'rocBLAS GEMM (8192³)',
    unit: 'GEMMs', short: 'gemm',
    flopsPerItem: 2 * Math.pow(8192, 3), actBytesPerItem: 8192 * 8192 * 2 * 3, weightBytesGB: 0.4,
    numKernels: 1, launchUs: 6, cpuUs: 8, pref: 'bf16',
    regime: 'compute-bound', cpuBound: 0.02,
    note: 'Large dense GEMM — the closest thing to peak FLOPs. Scales hard with precision.',
  },
  babelstream: {
    name: 'BabelStream (Triad, bandwidth)',
    unit: 'iterations', short: 'iter',
    flopsPerItem: 2 * 256e6, actBytesPerItem: 256e6 * 8 * 3, weightBytesGB: 6,
    numKernels: 4, launchUs: 5, cpuUs: 10, pref: 'fp32',
    regime: 'memory-bound', cpuBound: 0.02,
    note: 'Pure memory test — measures how close you get to 5.3 TB/s HBM3.',
  },
  resnet: {
    name: 'ResNet-50 Inference (vision)',
    unit: 'images', short: 'img',
    flopsPerItem: 8.2e9, actBytesPerItem: 6.0e6, weightBytesGB: 0.1,
    numKernels: 54, launchUs: 4.0, cpuUs: 25, pref: 'fp16',
    regime: 'balanced', cpuBound: 0.12,
    note: 'Balanced conv workload — a good middle-of-the-road benchmark.',
  },
};

const PRECISIONS = ['fp32', 'tf32', 'fp16', 'bf16', 'fp8'];

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* Deterministic pseudo-jitter (so "live" mode wiggles without Math.random
   making layouts jump unpredictably). */
function jitter(seed, amp) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 2 * amp;
}

/* ---- Core model ------------------------------------------------------- */
function computeMetrics(cfg) {
  const wl = WORKLOADS[cfg.workload] || WORKLOADS.ppo_infer;
  const prec = cfg.precision || wl.pref;
  const batch = clamp(cfg.batch || 1, 1, 1_000_000);
  const numGPUs = clamp(cfg.numGPUs || 1, 1, 8);
  const powerFrac = clamp(cfg.powerFrac ?? 1, 0.4, 1);          // power-cap slider
  const tick = cfg.tick || 0;                                    // for live jitter

  // Power cap throttles clocks (roughly sqrt relationship near the top).
  const clockScale = clamp(0.6 + 0.4 * Math.sqrt(powerFrac), 0.6, 1);
  const peakT = MI300X.peak[prec] * clockScale;                  // effective TFLOPS
  const bwTBs = MI300X.hbmTBs * clamp(0.8 + 0.2 * clockScale, 0.8, 1);

  // Partition mode: CPX splits the GPU into independent slices (more
  // parallel instances, slightly less peak each).
  const cpx = cfg.partition === 'cpx';
  const partitionEff = cpx ? 0.92 : 1.0;

  // Utilization efficiency rises with batch (amortizes launch + fills CUs).
  const fillK = { 'launch-bound': 4000, 'cpu-bound': 2000, 'memory-bound': 64,
                  'compute-bound': 1, 'balanced': 128 }[wl.regime] || 128;
  const matEff = clamp(0.18 + 0.62 * (batch / (batch + fillK)), 0.05, 0.82) * partitionEff;

  // Roofline: kernel time = max(compute-limited, bandwidth-limited).
  const flopsTotal = wl.flopsPerItem * batch;
  const computeTimeS = flopsTotal / (peakT * 1e12 * matEff);
  const memBytes = wl.weightBytesGB * 1e9 + wl.actBytesPerItem * batch;
  const memTimeS = memBytes / (bwTBs * 1e12 * clamp(matEff + 0.25, 0.2, 0.95));
  const kernelTimeS = Math.max(computeTimeS, memTimeS);
  const boundBy = computeTimeS >= memTimeS ? 'compute' : 'memory';

  // End-to-end latency (per batch) incl. launch + host overhead.
  const launchS = wl.numKernels * wl.launchUs * 1e-6;
  const cpuS = wl.cpuUs * 1e-6 * (1 + wl.cpuBound * 3);
  const e2eS = launchS + kernelTimeS + cpuS;
  const e2eMs = e2eS * 1000 * (1 + jitter(tick + batch, 0.03));

  // Throughput. Multi-GPU scaling has realistic losses.
  const scaleEff = Math.pow(0.93, numGPUs - 1);
  const itemsPerSecOne = batch / e2eS;
  const itemsPerSec = itemsPerSecOne * numGPUs * scaleEff;

  // Achieved compute / bandwidth.
  const achievedTflops = flopsTotal / kernelTimeS / 1e12;
  const achievedBwTBs = memBytes / kernelTimeS / 1e12;
  const computeUtil = clamp(achievedTflops / MI300X.peak[prec], 0, 1);
  const memUtil = clamp(achievedBwTBs / MI300X.hbmTBs, 0, 1);
  const arithIntensity = flopsTotal / memBytes;                  // FLOP/byte

  // Power & thermals follow the busier of the two utilizations.
  const busy = Math.max(computeUtil, memUtil);
  const powerW = (120 + (MI300X.boardWatts * powerFrac - 120) * (0.25 + 0.75 * busy))
                 * (1 + jitter(tick, 0.02));
  const tempC = 38 + 0.052 * powerW + jitter(tick + 7, 1.2);

  // Memory footprint.
  const memUsedGB = clamp(wl.weightBytesGB + wl.actBytesPerItem * batch / 1e9 + 1.2, 0.3, MI300X.memGB);

  /* ---------- Per-layer metrics (developer view, L0..L6) ----------- */
  const layers = buildLayers({
    wl, prec, batch, numGPUs, cpx, clockScale, peakT, bwTBs,
    computeUtil, memUtil, busy, powerW, tempC, achievedTflops, achievedBwTBs,
    kernelTimeS, e2eMs, launchS, cpuS, memUsedGB, boundBy, itemsPerSec, scaleEff, tick,
  });

  return {
    cfg: { ...cfg, precision: prec, batch, numGPUs },
    wl, prec, boundBy,
    e2eMs,
    latencyP50: e2eMs,
    latencyP99: e2eMs * (1.35 + jitter(tick + 3, 0.05)),
    throughput: itemsPerSec,
    throughputUnit: wl.unit,
    achievedTflops, peakTflops: MI300X.peak[prec],
    achievedBwTBs, peakBwTBs: MI300X.hbmTBs,
    computeUtil, memUtil, busy,
    arithIntensity,
    powerW, tempC, clockMHz: MI300X.baseClockMHz * clockScale,
    memUsedGB, memTotalGB: MI300X.memGB,
    scaleEff, numGPUs,
    layers,
  };
}

function buildLayers(s) {
  const pct = v => clamp(v * 100, 0, 100);
  return [
    {
      id: 0, name: 'L0 · Silicon / Microarchitecture',
      sub: 'CDNA 3 · 304 CUs · 8 XCDs · HBM3',
      metrics: [
        { k: 'Active CUs', v: Math.round(MI300X.cus * (0.3 + 0.7 * s.busy)), max: MI300X.cus, fmt: 'count' },
        { k: 'Engine clock', v: Math.round(s.clockScale * MI300X.baseClockMHz), unit: 'MHz' },
        { k: 'MFMA / matrix-core util', v: pct(s.computeUtil), unit: '%' },
        { k: 'HBM3 bandwidth util', v: pct(s.memUtil), unit: '%' },
        { k: 'VGPR occupancy', v: pct(clamp(0.4 + 0.5 * s.busy, 0, 1)), unit: '%' },
        { k: 'Board power', v: Math.round(s.powerW), unit: 'W', max: MI300X.boardWatts },
        { k: 'Junction temp', v: Math.round(s.tempC), unit: '°C', max: 95 },
      ],
    },
    {
      id: 1, name: 'L1 · Firmware / HW Abstraction',
      sub: 'SMU · partitioning · ECC',
      metrics: [
        { k: 'Compute partition', v: s.cpx ? 'CPX (8× slice)' : 'SPX (single)', fmt: 'text' },
        { k: 'Memory partition', v: s.cpx ? 'NPS4' : 'NPS1', fmt: 'text' },
        { k: 'Active XCDs', v: s.cpx ? 8 : Math.max(1, Math.round(MI300X.xcds * s.busy)), max: 8, fmt: 'count' },
        { k: 'SMU power state', v: s.busy > 0.5 ? 'P-high' : s.busy > 0.1 ? 'P-mid' : 'P-idle', fmt: 'text' },
        { k: 'ECC corrected errors', v: 0, fmt: 'count' },
        { k: 'Firmware', v: 'real (MI300X)', fmt: 'text' },
      ],
    },
    {
      id: 2, name: 'L2 · Kernel Driver (amdgpu / KFD)',
      sub: 'queues · DMA · page faults',
      metrics: [
        { k: 'KFD dispatch latency', v: +(2.1 + jitter(s.tick, 0.3)).toFixed(2), unit: 'µs' },
        { k: 'HW queue depth', v: Math.round(2 + 14 * s.busy), max: 16, fmt: 'count' },
        { k: 'DMA H2D/D2H', v: +(28 + 200 * s.busy * (1 - s.cpx * 0.1)).toFixed(0), unit: 'GB/s' },
        { k: 'Page faults / s', v: Math.round(clamp(40 * (1 - s.busy), 0, 60)), fmt: 'count' },
        { k: 'IRQs / s', v: Math.round(1200 + 9000 * s.busy), fmt: 'count' },
      ],
    },
    {
      id: 3, name: 'L3 · Runtime (ROCr / HSA)',
      sub: 'kernel dispatch · signals · queues',
      metrics: [
        { k: 'Kernel-launch latency', v: +(s.wl.launchUs + jitter(s.tick + 1, 0.4)).toFixed(2), unit: 'µs' },
        { k: 'Dispatch rate', v: Math.round(s.wl.numKernels / Math.max(s.e2eMs / 1000, 1e-4)), unit: 'k/s', fmt: 'k' },
        { k: 'HSA queue occupancy', v: pct(clamp(0.3 + 0.6 * s.busy, 0, 1)), unit: '%' },
        { k: 'Signal wait', v: +(0.8 + 2 * (1 - s.busy)).toFixed(2), unit: 'µs' },
        { k: 'Active streams', v: s.cpx ? 8 : Math.max(1, Math.round(4 * s.busy)), fmt: 'count' },
      ],
    },
    {
      id: 4, name: 'L4 · Math Libraries',
      sub: 'rocBLAS · hipBLASLt · MIOpen · RCCL',
      metrics: [
        { k: 'GEMM achieved', v: +s.achievedTflops.toFixed(1), unit: 'TFLOPS', max: s.peakT },
        { k: 'Library', v: s.boundBy === 'compute' ? 'hipBLASLt' : 'rocBLAS', fmt: 'text' },
        { k: 'Kernel cache hit', v: pct(clamp(0.7 + 0.25 * s.busy, 0, 1)), unit: '%' },
        { k: 'RCCL bus BW', v: s.numGPUs > 1 ? +(45 * s.scaleEff).toFixed(0) : 0, unit: 'GB/s' },
        { k: 'Autotune variant', v: s.boundBy === 'compute' ? 'MFMA 32×32' : 'split-K', fmt: 'text' },
      ],
    },
    {
      id: 5, name: 'L5 · Framework (PyTorch + HIP)',
      sub: 'ops · host overhead · memory',
      metrics: [
        { k: 'GPU compute time', v: +(s.kernelTimeS * 1000).toFixed(3), unit: 'ms' },
        { k: 'Host overhead', v: pct(clamp(s.cpuS / (s.kernelTimeS + s.cpuS + s.launchS), 0, 1)), unit: '%' },
        { k: 'VRAM allocated', v: +s.memUsedGB.toFixed(1), unit: 'GB', max: MI300X.memGB },
        { k: 'HIP graph capture', v: s.batch >= 64 ? 'on' : 'off', fmt: 'text' },
        { k: 'Launch overhead', v: pct(clamp(s.launchS / (s.kernelTimeS + s.cpuS + s.launchS), 0, 1)), unit: '%' },
      ],
    },
    {
      id: 6, name: 'L6 · Application / Workload',
      sub: s.wl.name,
      metrics: [
        { k: 'End-to-end latency', v: +s.e2eMs.toFixed(3), unit: 'ms' },
        { k: 'Throughput', v: Math.round(s.itemsPerSec), unit: s.wl.short + '/s', fmt: 'big' },
        { k: 'Batch size', v: s.batch, fmt: 'count' },
        { k: 'GPUs', v: s.numGPUs, fmt: 'count' },
        { k: 'Bound by', v: s.boundBy + (s.wl.cpuBound > 0.4 ? ' + CPU' : ''), fmt: 'text' },
      ],
    },
  ];
}

/* ---- Cost helpers (executive view) ----------------------------------- */
function costModel(metrics, dollarsPerGpuHr) {
  const rate = dollarsPerGpuHr || 4.89;       // illustrative cloud MI300X $/GPU-hr
  const hourlyCost = rate * metrics.numGPUs;
  const itemsPerHour = metrics.throughput * 3600;
  const costPerMillion = itemsPerHour > 0 ? (hourlyCost / itemsPerHour) * 1e6 : 0;
  const perfPerDollar = hourlyCost > 0 ? metrics.throughput / hourlyCost : 0;
  const perfPerWatt = metrics.powerW > 0 ? metrics.throughput / metrics.powerW : 0;
  const monthly = hourlyCost * 24 * 30;
  return { rate, hourlyCost, itemsPerHour, costPerMillion, perfPerDollar, perfPerWatt, monthly };
}

/* ---- Sim-to-Real prediction (the project's core thesis) --------------
   The objective: estimate REAL MI300X performance from SIMULATION-derived
   metrics (Path 1 agnostic sim + Path 2 gem5 arch sim) within ±20%, without
   touching the hardware until validation. Here we model that prediction:
   - "measured" = computeMetrics() (stands in for real-hardware ground truth)
   - "predicted" = the sim-based estimate, offset by a realistic error that
     depends on regime (sim predicts compute-bound kernels well; CPU-bound
     and launch-bound workloads are harder, so error is larger).
   Replace this with (gem5 features -> trained model -> estimate) for real. */
const PREDICT_BIAS = {            // baseline relative error by workload regime
  'compute-bound': 0.05,         // sim nails compute-bound kernels
  'memory-bound': 0.10,
  'balanced': 0.11,
  'launch-bound': 0.20,          // launch/host effects are hard to simulate...
  'cpu-bound': 0.26,             // ...CPU-coupled workloads hardest of all
};
const TARGET_ACCURACY = 0.20;     // ±20% success criterion from the objective

function predictionSet(cfg) {
  const real = computeMetrics(cfg);
  const b = PREDICT_BIAS[real.wl.regime] ?? 0.10;
  const t = cfg.tick || 0;
  // Each metric: a signed sim error (systematic component + bounded jitter).
  const err = (seed, dir) => dir * (b * 0.8) + jitter(seed, b * 0.6);
  const pairs = [
    { k: 'E2E latency', unit: 'ms', lowerBetter: true,
      measured: real.e2eMs, predicted: real.e2eMs * (1 + err(t + 11, +1)) },
    { k: 'Throughput', unit: real.wl.short + '/s',
      measured: real.throughput, predicted: real.throughput * (1 - err(t + 12, +1)) },
    { k: 'Achieved TFLOPS', unit: 'TFLOPS',
      measured: real.achievedTflops, predicted: real.achievedTflops * (1 - err(t + 13, +1) * 0.6) },
    { k: 'HBM bandwidth', unit: 'TB/s',
      measured: real.achievedBwTBs, predicted: real.achievedBwTBs * (1 - err(t + 14, +1) * 0.7) },
    { k: 'Board power', unit: 'W',
      measured: real.powerW, predicted: real.powerW * (1 + err(t + 15, +1) * 0.5) },
  ];
  pairs.forEach(p => {
    p.errPct = Math.abs(p.predicted - p.measured) / (Math.abs(p.measured) || 1) * 100;
    p.ratio = p.predicted / (p.measured || 1);
    p.within = p.errPct <= TARGET_ACCURACY * 100;
  });
  const within = pairs.filter(p => p.within).length;
  return {
    real, pairs,
    withinPct: within / pairs.length * 100,
    meanErrPct: pairs.reduce((s, p) => s + p.errPct, 0) / pairs.length,
    targetPct: TARGET_ACCURACY * 100,
  };
}

/* Build a short synthetic time-series for trend charts. */
function timeSeries(cfg, points) {
  const out = [];
  for (let i = 0; i < points; i++) {
    const m = computeMetrics({ ...cfg, tick: (cfg.tick || 0) - (points - i) });
    out.push(m);
  }
  return out;
}
