# Estimating Real-World Compute Performance of Robotics Workloads from Simulation

**Status:** Draft v1 · June 2026
**Scope note:** This phase targets the AMD/ROCm stack only, with the AMD
Instinct **MI300X** as the specific ground-truth and simulated target.
NVIDIA/CUDA is deferred to a later phase.

---

## 1. Business Goal

Physical AI products (robots, autonomous systems) succeed or fail on whether
their control software meets real-time performance budgets on the compute
hardware actually deployed in the field. Today, validating this requires
buying and provisioning every candidate hardware platform — slow, expensive,
and late in the development cycle.

**The business goal is to reduce the cost and lead time of hardware
selection and deployment validation for robotics workloads, by making
real-world performance estimable from simulation before hardware is
acquired.** A team that can predict "this control policy will run at X ms
latency on platform Y within ±20%" from simulation alone can shortlist
hardware earlier, negotiate from data, and catch deployment blockers before
integration.

## 2. Project Objective

Address the simulation-to-real gap on its **compute-performance dimension**:
a workload that behaves well in a robotics simulator gives no indication of
how it will perform on real deployment hardware. This project closes that
gap in two steps:

1. **Thoroughly measure fine-grained performance** of a robotics workload
   across the full compute stack — from the application layer down to
   microarchitectural behavior — using simulation environments that do not
   require the target hardware.
2. **Use those fine-grained measurements to estimate real-world metrics**
   (end-to-end latency, throughput) on the physical AMD/ROCm platform, and
   validate the estimates against measured ground truth.

**Success criterion:** predict policy-inference latency (batch size 1) and
training throughput (environment steps/second) on a real AMD **Instinct
MI300X** GPU within **±20%**, validated on workloads held out from model
fitting. Cloud MI300X access is available for this phase, and the MI300X is
also one of gem5's officially supported GPU models (gfx942) — so the same
device family is the ground truth in Path 3 *and* the simulated target in
Path 2, which tightens the simulation-to-real comparison considerably.

## 3. Approach

### 3.1 High level

One shared workload suite is measured along three paths. Two paths produce
*features* without needing the target GPU; the third produces *ground truth*
on real hardware. A prediction model maps features to ground truth and is
validated on held-out workloads.

```mermaid
flowchart LR
    W["Shared workload suite"] --> P1["Path 1<br/>Robotics simulator<br/>stack-level metrics"]
    W --> P2["Path 2<br/>gem5 AMD GPU model<br/>microarchitectural metrics"]
    W --> P3["Path 3<br/>Real AMD MI300X hardware<br/>ground truth"]
    P1 --> M["Prediction model"]
    P2 --> M
    P3 -- "labels + held-out validation" --> M
    M --> O["Estimated real-world performance<br/>within ±20%"]
```

### 3.2 Technical details

**Tools (open source preferred):**

| Component | Tool | License / note |
|---|---|---|
| Robot simulation | MuJoCo + Gymnasium | Apache 2.0; standard walker/humanoid locomotion tasks |
| Learning framework | PyTorch (ROCm build) | BSD; same code path as CUDA, keeping a later NVIDIA phase cheap |
| Architectural simulation | gem5 (GPUFS, VEGA_X86) | BSD; official AMD GPU models incl. **MI300X (gfx942)** running real ROCm 6.x |
| GPU microbenchmarks | BabelStream, custom HIP kernels | Open source; HIP keeps source portable |
| Profiling | rocprofiler, `torch.utils.benchmark`, gem5 stats | Open source / vendor-provided |

**Sample workloads (small → large):**

1. **HIP microkernels** — vector copy, reduction, GEMM at several sizes.
   Small enough to run in gem5; they anchor the microarchitectural features.
2. **PyTorch operator benchmarks** — the linear layers, activations, and
   batched GEMMs that a control policy is actually made of, at the policy's
   real tensor shapes.
3. **PPO locomotion training and inference** — a walker/humanoid agent in
   MuJoCo. Physics runs on CPU; the GPU work is policy training and batch-1
   inference. This is the application-level workload whose real-world
   performance we ultimately estimate.

## 4. Operationalization

1. **Weeks 1–2 — Baseline workload.** Stand up MuJoCo + Gymnasium + PyTorch;
   train PPO on Walker2d; define the exact policy shapes that parameterize
   all other workloads.
2. **Weeks 3–4 — Fine-grained measurement, Path 1.** Instrument the stack
   layers (application timing, framework operator profiles, runtime/driver
   counters via rocprofiler); build the automated measurement harness
   (pinned clocks, warmup, synchronize-before-timing, 30+ runs, median +
   spread).
3. **Weeks 5–7 — Fine-grained measurement, Path 2.** Build gem5 VEGA_X86
   (Docker toolchain), validate in GPUSE mode with HIP samples, then run the
   microkernels and operator-sized kernels in GPUFS on the **MI300X (gfx942)**
   model; extract IPC, cache hit rates, memory traffic, occupancy.
4. **Weeks 8–9 — Ground truth, Path 3.** Run the full suite on a real AMD
   **Instinct MI300X** via cloud access, with the same harness.
5. **Weeks 10–12 — Prediction and validation.** Fit regression models from
   Path 1+2 features to Path 3 labels; compose kernel-level estimates into
   application-level estimates; evaluate on held-out workloads; iterate on
   features where error exceeds ±20%.

Resource envelope: one Linux machine (16 cores, 32–64 GB RAM, ~150 GB free
disk) covers Paths 1–2; cloud MI300X hours cover Path 3.

## 5. Expected Results

1. An **open measurement harness** producing layered, reproducible
   performance profiles of robotics workloads on the ROCm stack.
2. A **dataset** pairing simulation-derived features with real-hardware
   ground truth across the workload suite.
3. A **prediction model** estimating real-world latency/throughput within
   ±20% on held-out workloads — plus an analysis of *which stack layers
   carry the most predictive signal*, which is a useful result even where
   the ±20% target is missed.
4. A documented gem5-based methodology for asking "what if the hardware had
   more compute units / larger caches?" without owning that hardware.

## 6. Benefit

- **Earlier, cheaper hardware decisions:** deployment performance becomes a
  simulation-time question, not a procurement-time discovery.
- **De-risked deployment:** latency-budget violations surface before
  integration on the physical robot.
- **Vendor-extensible foundation:** because the workload code (PyTorch/HIP)
  is portable, adding the NVIDIA/CUDA path later requires only the
  ground-truth and simulator backends, not a new methodology.
- **Skills and artifacts that compound:** the harness, dataset, and gem5
  workflow are each independently reusable for future architecture and
  benchmarking studies.

## 7. Reference: The AMD Instinct MI300 Product Line

This section documents the target hardware family so the proposal is
self-contained. The MI300 series is AMD's data-center accelerator line built
on the **CDNA 3** architecture, using advanced 3D chiplet packaging that
stacks compute dies (XCDs) on top of I/O dies over a silicon interposer,
with HBM3 stacks around the perimeter.

### 7.1 Product line at a glance

| Product | Type | Architecture | Key memory | Role |
|---|---|---|---|---|
| **MI300A** | APU (CPU+GPU) | CDNA 3 + Zen 4 | 128 GB HBM3, unified | Converged HPC+AI (e.g. El Capitan supercomputer) |
| **MI300X** | Discrete GPU | CDNA 3 | 192 GB HBM3 | Generative AI / large-model training & inference |
| **MI325X** | Discrete GPU | CDNA 3 | 256 GB HBM3E | Memory-expanded refresh of MI300X |
| **MI355X** | Discrete GPU | CDNA 4 | 288 GB HBM3E, ~8 TB/s | Next-gen successor (newer architecture) |

The **MI300A** is a hybrid APU: it fuses 24 Zen 4 CPU cores with CDNA 3 GPU
chiplets (228 CUs) sharing a single 128 GB unified HBM3 pool — eliminating
host-device copies for tightly coupled HPC. The **MI300X** replaces the CPU
chiplets with more GPU chiplets and memory, making it a pure accelerator;
it is the variant used in this project.

### 7.2 MI300X technical specifications

| Attribute | MI300X |
|---|---|
| Architecture | AMD CDNA 3 |
| ISA target (ROCm/gem5) | gfx942 |
| Compute Units | 304 |
| Peak engine clock | 2,100 MHz |
| Memory | 192 GB HBM3 |
| Memory bus | 8,192-bit |
| Peak memory bandwidth | ~5.3 TB/s (5.325 TB/s) |
| Form factor | OAM module; 8-GPU platform board |
| Typical board power | ~750 W |

### 7.3 Peak compute throughput (MI300X)

| Precision | Peak (TFLOPS) |
|---|---|
| FP32 (single) | 163.4 |
| TF32 (matrix) | 653.7 |
| FP16 (half) | 1,307.4 |
| FP8 | 2,614.9 |

(Matrix/tensor figures roughly double again with structured sparsity.)

### 7.4 Architectural details relevant to this project

- **Chiplet/XCD layout:** the 304 CUs are spread across multiple Accelerator
  Complex Dies (XCDs). This non-uniformity matters when interpreting gem5
  occupancy and scaling results.
- **Unified Infinity Cache + HBM3:** the large last-level cache and very high
  HBM3 bandwidth mean memory-bound robotics kernels (small-batch inference)
  behave very differently from compute-bound training GEMMs — the prediction
  model must capture both regimes.
- **Partitioning modes:** MI300X supports compute and memory partitioning
  (NPS / compute-partition modes). gem5's MI300X model now uses real firmware
  and can exercise these, which is exactly the kind of "what-if" knob the
  Path 2 architecture study exploits.
- **Batch-1 latency relevance:** with 304 CUs and 192 GB, the MI300X is
  heavily over-provisioned for a single robot policy's inference. The
  interesting research question is precisely how poorly such a large
  accelerator is utilized at robotics batch sizes — a gap the simulation
  metrics should reveal.

*Specifications sourced from AMD's official MI300X data sheet and product
pages; see Sources.*

## Sources

- AMD Instinct MI300X product page: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- AMD Instinct MI300X data sheet (PDF): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
- AMD Instinct MI300 series overview: https://www.amd.com/en/products/accelerators/instinct/mi300.html
- gem5 GPUFS / AMD GPU models: https://www.gem5.org/documentation/general_docs/gpu_models/gpufs
