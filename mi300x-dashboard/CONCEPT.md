# Dashboard Concept — MI300X Performance Console

A design brief for the deliverable: a performance console for the **AMD Instinct
MI300X**, serving executives and developers, and showcasing the **Physical AI
simulation-to-real gap** initiative. Data is simulated today; the architecture
is built to swap in real telemetry from servers and cloud later.

Related: [`../my-notes/my-objective-refined.md`](../my-notes/my-objective-refined.md)
(objective + methodology).

---

## 1. The core idea

One product, three lenses on the **same** MI300X telemetry:

| Lens | Audience | Question it answers | Domain |
|---|---|---|---|
| **Executive** | leadership, buyers | "Is this fast/cheap enough, and what will it cost?" | neutral |
| **Developer** | engineers | "Where is my time going across the stack (L0–L6)?" | neutral |
| **Physical AI** | robotics / research | "Will my policy hold its real-time deadline — and can I trust simulation to tell me before I deploy?" | use-case |
| **Architect** | research / power-user | "How do I configure gem5 from a spec, build the datasets, train the predictor, wire in real data, and reason about it all?" | use-case / workbench |

The first two are **domain-neutral**: useful for any workload on MI300X. The
last two are **use-case lenses** for the Real-to-Sim initiative — Physical AI for
consuming the result (deadlines, sim-to-real validation), Architect for building
the pipeline that produces it. Keeping them separate means the neutral views stay
reusable while the domain story gets dedicated homes.

## 2. Why this framing

The project's thesis (see objective doc) is: **estimate real-hardware
performance from simulation within ±20%, before touching the hardware.** A
dashboard that only shows live MI300X metrics misses the point — the *value* is
the prediction. So the Physical AI lens makes the prediction itself the hero:
predicted (from Path 1 agnostic sim + Path 2 gem5 arch sim) vs measured (Path 3
real MI300X), scored against the ±20% band.

For executives that becomes "how much hardware validation can we skip?"; for
developers it becomes "which layer's metrics carry the predictive signal?"; for
the robotics user it becomes "will the control loop meet its deadline?"

## 3. The three dashboards

### 3.1 Executive (domain-neutral)
KPIs: throughput, cost per 1M items, $/hour spend, SLA met/miss. Plus
utilization donut, throughput trend, efficiency (perf/$, perf/W, scaling),
monthly TCO projection across 1→8 GPUs, and a cost-vs-batch chart that makes
the batching argument visually. Knobs: workload, batch, precision, GPU count,
cloud $/GPU-hr, SLA target.

### 3.2 Developer (domain-neutral) — the L0→L6 stack
An expandable stack of the AMD/ROCm layers, each with live metrics and bars,
plus roofline, latency histogram, and kernel-dispatch timeline.

| Layer | Name | Example metrics | Real source (later) |
|---|---|---|---|
| **L0** | Silicon / microarchitecture | active CUs, clock, MFMA util, HBM util, VGPR, power, temp | rocprofiler HW counters |
| **L1** | Firmware / HW abstraction | SPX/CPX partition, NPS, active XCDs, SMU state, ECC | SMI / SMU |
| **L2** | Kernel driver (amdgpu/KFD) | dispatch latency, queue depth, DMA, page faults, IRQs | amdgpu / KFD sysfs |
| **L3** | Runtime (ROCr/HSA) | launch latency, dispatch rate, queue occupancy, signals | ROCr / rocprofiler |
| **L4** | Math libraries | achieved TFLOPS, library/variant, cache hit, RCCL BW | rocBLAS/hipBLASLt/MIOpen/RCCL logs |
| **L5** | Framework (PyTorch+HIP) | compute time, host overhead %, VRAM, graph capture | PyTorch profiler |
| **L6** | Application / workload | end-to-end latency, throughput, batch, bound-by | app instrumentation |

### 3.3 Physical AI (use-case) — the sim-to-real gap
Frames MI300X performance around robot control:
- **Real-time control budget** — per-cycle latency vs the deadline implied by the
  target control frequency (e.g. 200 Hz → 5 ms), with a p99 marker and a clear
  "within budget / DEADLINE MISS" verdict.
- **Control cycle breakdown** — sense → infer (policy on MI300X) → act, so you
  see how much of the budget the GPU policy actually consumes.
- **Sim → Real validation** — the hero panel: predicted vs measured metrics on a
  parity plot with the ±20% band shaded; a confidence gauge (% within ±20%),
  mean error, and the hardest-to-predict metric called out.
- **Deadline adherence, achievable control rate, jitter** as headline KPIs.
- Tasks: bipedal walk, quadruped, humanoid, arm manipulation, VLA policy;
  phase toggle for deploy (inference) vs train (RL).

### 3.4 Architect (use-case workbench) — the pipeline builder
A 6-step stepper that assembles the whole prediction pipeline:
1. **Hardware → gem5** — upload a design spec (PDF/JSON/YAML); parameters are
   extracted (CUs, XCDs, caches, HBM, clock, ISA, Ruby protocol) with a
   per-field confidence, editable, and rendered as a ready-to-run gem5 config.
2. **Workload** — configure the benchmark and preview predicted behaviour.
3. **Datasets** — training/validation/held-out setup, feature catalog by layer,
   data sources (sim now; rocprofiler when a device connects).
4. **Train & Accuracy** — fit the predictor, watch the train/val MAE curve,
   read the scorecard (MAE, R², % within ±20%) and feature importance.
5. **Device / Trace** — connect a real MI300X endpoint for live ground truth,
   or fall back transparently to a recorded trace so panels never go empty.
6. **LLM Copilot** — configure provider/model (default `claude-fable-5`),
   endpoint, key, temperature; chat with an assistant grounded in the current
   workbench state (offline rule-based today; real API later).

## 4. Data roadmap — simulated now, real later

The whole front-end depends on a single function shape: `computeMetrics(cfg)` →
metrics object, and `predictionSet(cfg)` → predicted/measured pairs. Today these
are generated in `assets/sim.js`. Productionizing = replacing that one module:

```
  Phase 1 (now):   knobs → sim.js model → dashboards
  Phase 2:         knobs → real telemetry adapter → dashboards
                          ├─ rocprofiler / rocm-smi (L0–L4)   [cloud MI300X]
                          ├─ PyTorch profiler (L5)
                          ├─ app instrumentation (L6)
                          └─ gem5 + agnostic-sim features → trained
                             predictor → predicted side of the parity plot
```

No dashboard or chart code changes — only the data source. That is the point of
isolating the simulation behind `computeMetrics`/`predictionSet`.

## 5. Visual language

Dark AMD-inspired theme; red = AMD/attention, teal = healthy, amber = caution/
budget, green = within-target. Dependency-free inline SVG charts so the whole
thing runs offline from `file://`. Live toggle adds gentle deterministic jitter
so it feels like streaming telemetry without layouts jumping.

## 6. What the simulation deliberately reproduces

- Batch-1 inference is **launch-bound** (tens of µs regardless of GPU size) — and
  therefore the **hardest to predict** (largest sim-to-real error).
- LLM decode is **HBM-bandwidth bound**; GEMM is **compute-bound** and scales with
  precision; PPO training is partly **CPU-bound**.
- Multi-GPU scaling loses efficiency (~93%/GPU).
- Prediction error grows from compute-bound (~5%) to CPU/launch-bound (~15–19%),
  mirroring how simulation fidelity actually degrades.

*All figures are illustrative planning numbers, not benchmarks.*
