# MI300X Performance Console — Interactive Mockup

A self-contained, **offline** prototype of the final deliverable: a performance
console for robotics workloads on the **AMD Instinct MI300X**. It shows the
look, feel, and *behaviour* of the product. **All metrics are simulated and
illustrative — not benchmarks.**

## Run it

No build, no server, no internet. Just open the file:

```
mi300x-dashboard/index.html
```

Double-click `index.html` (or open it in any modern browser). From the landing
page choose the **Executive** or **Developer** dashboard.

> Tip: on WSL2 you can run `explorer.exe index.html` from this folder, or open
> the Windows path to the file directly in your browser.

## What's inside

| File | Purpose |
|---|---|
| `index.html` | Landing page — pick a dashboard |
| `executive.html` | **Domain-neutral.** Outcome KPIs: throughput, cost/1M, $/hr, SLA, TCO projection |
| `developer.html` | **Domain-neutral.** L0–L6 AMD/ROCm stack telemetry + roofline, latency, timeline |
| `physical-ai.html` | **Use-case.** Real-time control deadlines + sim-to-real prediction (±20%) |
| `architect.html` | **Use-case / power-user.** gem5 config from spec, datasets, predictor accuracy, device/trace I/O, LLM copilot |
| `CONCEPT.md` | Design brief / brainstorm behind the dashboards |
| `assets/sim.js` | Shared simulation engine — knobs → metrics + sim-to-real prediction |
| `assets/charts.js` | Dependency-free SVG charts (gauge, line, bar, donut, roofline, histogram, timeline, parity, budget) |
| `assets/style.css` | Shared AMD-inspired dark theme |

### Four lenses on the same MI300X telemetry

- **Executive** and **Developer** are *domain-neutral* — useful for any workload.
- **Physical AI** is the *use-case* lens for the Real-to-Sim initiative: it frames
  performance around robot control deadlines and validates predicted (simulation)
  vs measured (real MI300X) metrics against the ±20% success criterion.
- **Architect** is the *power-user workbench* that wires the whole pipeline
  together: design spec → gem5 config, dataset setup, predictor training &
  accuracy, real-device/trace I/O, and an LLM copilot grounded in the config.

See [`CONCEPT.md`](CONCEPT.md) for the full design rationale.

## Interactive knobs

Both dashboards share a live simulation driven by:

- **Workload** — PPO inference, PPO training, 7B/70B LLM decode, rocBLAS GEMM,
  BabelStream, ResNet-50
- **Batch size** — 1 … 262,144 (watch latency-bound flip to throughput-bound)
- **Precision** — FP32 / TF32 / FP16 / BF16 / FP8
- **GPUs** — 1 … 8 (multi-GPU scaling loss is modelled)
- **Power cap** — throttles clocks (developer view)
- **Partition mode** — SPX vs CPX (developer view)
- **Cloud $/GPU-hr** and **latency SLA** (executive view)
- **Live telemetry** toggle — values gently jitter to feel real-time

## Developer view — the L0→L6 stack

Each layer panel maps to a real telemetry source in production:

| Layer | Name | Real source it would map to |
|---|---|---|
| **L0** | Silicon / microarchitecture | hardware counters via rocprofiler |
| **L1** | Firmware / HW abstraction | SMU, partition (SPX/CPX, NPS), ECC |
| **L2** | Kernel driver | amdgpu / KFD queues, DMA, page faults |
| **L3** | Runtime | ROCr / HSA dispatch, signals, queues |
| **L4** | Math libraries | rocBLAS / hipBLASLt / MIOpen / RCCL |
| **L5** | Framework | PyTorch + HIP op breakdown, VRAM |
| **L6** | Application / workload | end-to-end latency, throughput |

## From mockup to product

The architecture is intentionally simple to productionize: replace
`assets/sim.js` with a module that reads **real** telemetry (rocprofiler,
rocm-smi, PyTorch profiler, application instrumentation) exposing the same
shape of object that `computeMetrics()` returns. The dashboards and charts
need no changes.

## Behaviour the simulation deliberately reproduces

- **Batch-1 inference is launch-bound** — latency ≈ tens of µs regardless of
  how big the GPU is.
- **LLM decode is HBM-bandwidth bound** — where the MI300X's 5.3 TB/s shines.
- **GEMM is compute-bound** and scales hard with lower precision (FP8 ≫ FP32).
- **PPO training is partly CPU-bound** — GPU is not the bottleneck.
- **Multi-GPU scaling loses efficiency** (~93% per added GPU here).
- **Roofline** places each workload against the compute ceiling and the
  bandwidth slope, with the ridge point marked.

*Numbers are planning illustrations only. See `../my-notes/` for the project
objective and methodology behind this deliverable.*
