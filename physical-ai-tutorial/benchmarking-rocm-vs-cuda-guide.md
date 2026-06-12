# Benchmarking Compute Stacks (ROCm vs CUDA) with a Robot-Simulation Workload

Mentor notes — June 2026

## The key correction up front

**ROCm on gem5: yes, fully supported — it's the official path.** gem5's GPU model
*is* an AMD model. Current gem5 (v24/v25+) ships a full-system GPU mode called
**GPUFS** that boots a real Linux kernel with the actual amdgpu driver and runs a
real ROCm userspace (ROCm 6.1 disk images provided) on simulated Vega (gfx900),
MI210/MI250X (gfx90a), and MI300X (gfx942) models. AMD's research team maintains
this; the MI300X model uses real firmware.

**But CUDA does NOT run on gem5.** There is no NVIDIA GPU model in gem5. NVIDIA
simulation lives in a separate simulator — Accel-Sim / GPGPU-Sim 4.0
(trace-driven simulation of NVIDIA SASS). So "benchmark ROCm vs CUDA on gem5"
is not possible. Two valid options:

1. **Compare stacks on real hardware** (an AMD GPU and an NVIDIA GPU, or cloud
   instances). This is the methodologically sound way to compare performance
   and latency across stacks.
2. **Use simulators for microarchitecture studies** — gem5 for AMD, Accel-Sim
   for NVIDIA — but treat them as separate research tools. Comparing absolute
   numbers across two different simulators is not a valid comparison.

Expectation check: gem5's GPU model runs ~10,000–100,000× slower than real
hardware. You simulate kernels that take milliseconds on hardware, not
applications. A robot simulator will never run inside gem5 — and it doesn't
need to.

## 1. Open-source robot simulation tools

- **MuJoCo** (Apache 2.0, Google DeepMind) — the standard for locomotion/control
  research. Ready-made walker, humanoid, quadruped tasks via Gymnasium and
  `dm_control`. Physics runs on CPU (a feature for this project). **MJX**
  (MuJoCo-on-JAX) exists if you later want GPU-accelerated physics.
- **Gazebo + ROS 2** — standard for robotics-engineering workflows (sensors,
  URDF robots, controllers). Heavier setup; better when the robotics stack
  itself is the subject.
- **PyBullet** — easiest entry point, slightly dated but fine for learning.
- **Avoid NVIDIA Isaac Sim/Isaac Lab here** — free but not open source, and
  CUDA-only, so useless as a fair ROCm-vs-CUDA workload.

**Recommendation: MuJoCo + Gymnasium + PyTorch**, training a PPO agent on
`Walker2d` or `Humanoid`. Physics runs on CPU (identical on both systems);
the neural-network training runs on the GPU through PyTorch — and PyTorch
supports both CUDA and ROCm with the same Python code (`torch.device("cuda")`
works on ROCm too). A controlled experiment where only the GPU stack changes.

## 2. gem5 with AMD's architecture models

Practical path:

1. Build gem5 with the **VEGA_X86** target. Use the Docker images from
   gem5-resources (https://resources.gem5.org/) — the GPU toolchain (specific
   ROCm + kernel versions) is fussy; the containers solve that.
2. Start in **GPUSE (syscall emulation) mode** with the bundled `square` or
   HIP-samples benchmarks to learn the flow — faster and simpler.
3. Graduate to **GPUFS** with the MI200/MI300X configs from the standard
   library when you want full-driver fidelity.
4. Workloads: small HIP kernels, modest rocBLAS GEMMs, single inference
   layers. The ISCA 2024 gem5 tutorial shows small PyTorch workloads; the
   HPCA 2023 GPU tutorial is the best end-to-end walkthrough.

What gem5 buys you that real hardware can't: change the architecture — CU
count, cache sizes, memory bandwidth — and rerun the same kernel. Example
study: "how does GEMM latency scale with CU count on a simulated MI210?"

## 3. Benchmarking workflow (three layers, on real hardware)

**Layer 1 — Microbenchmarks (stack primitives).**
- BabelStream (https://github.com/UoB-HPC/BabelStream) — open source, CUDA and
  HIP backends from the same source — for memory bandwidth.
- mixbench or GEMM sweeps via rocBLAS vs cuBLAS for compute throughput.
- Kernel-launch latency and host↔device copy latency with a tiny HIP/CUDA
  program — HIP compiles for both vendors via `hipify`, keeping source identical.

**Layer 2 — Framework level.**
- PyTorch operator benchmarks (`torch.utils.benchmark`): conv, attention, GEMM
  at the sizes the robot policy actually uses, in FP32 and FP16/BF16.

**Layer 3 — Application level.**
- MuJoCo + PPO walker training: wall-clock time per training iteration,
  environment steps/second, and policy-inference latency at batch size 1
  (the robotics-relevant latency number).

### Methodology rules

- Same PyTorch version on both machines (official ROCm and CUDA wheels), same
  Python env, same seeds, same model and batch sizes.
- Warm up first (JIT/kernel-autotuning overhead), then measure 30+ runs;
  report median + spread, never single numbers.
- Synchronize before timing (`torch.cuda.synchronize()` — works on ROCm too);
  GPU launches are async and naive timing measures nothing.
- Pin clocks if possible (`nvidia-smi -lgc`, `rocm-smi --setperflevel`), log
  temperatures, and record per-kernel profiles with rocprofiler vs Nsight
  Compute.
- Acknowledge the hardware confound honestly: you compare *stack + silicon*,
  not stacks in isolation. Normalize by each card's theoretical peak FLOPS /
  bandwidth to make the comparison meaningful.

## Suggested order of attack

1. Week 1: get MuJoCo + PPO training running on whatever machine you have
   (no GPU comparison yet).
2. Build the layer-1 microbenchmarks in HIP; verify they compile for both
   vendors.
3. Run the three layers on both stacks; collect and plot.
4. Separately, do the gem5 ROCm track as an AMD-only architecture-exploration
   study — don't force it into the CUDA comparison.

Three deliverables: a cross-stack benchmark suite, an RL locomotion workload,
and a gem5 architectural study. That's more than one student project — treat
the gem5 part as phase 2 if needed.

## Resource estimates (approximate)

### Track A — Real-hardware benchmarking
- Disk per machine: **~50 GB** (ROCm 6.x ~25–30 GB; CUDA toolkit ~6–10 GB;
  PyTorch wheel ~3–4 GB per stack; MuJoCo/Gymnasium env ~1–2 GB; logs ~5 GB).
- GPU: 8 GB VRAM is plenty (4 GB works for Walker2d). ROCm: pick an officially
  supported card — RX 7900 XT/XTX (gfx1100) is the safest consumer choice.
  NVIDIA: any RTX 3060+.
- CPU: 8+ cores (MuJoCo physics is CPU-bound with vectorized envs). RAM: 16–32 GB.
- Runtime: Walker2d PPO ~1–3 h; Humanoid ~0.5–1 day; microbenchmark sweeps
  minutes–1 h.
- Cloud fallback: AMD Developer Cloud / DigitalOcean MI300X by the hour;
  RunPod/Lambda for NVIDIA (~$0.20–0.50/hr small cards; ~$20–50 total campaign).

### Track B — gem5 + ROCm
- **No GPU needed** — gem5 simulates the GPU on the CPU.
- Disk: **budget 100–150 GB** (gem5 source + VEGA_X86 build ~15–25 GB; GCN/GPU
  Docker image ~10–15 GB; GPUFS ROCm 6.1 disk images ~20–30 GB each, the
  PyTorch ML image ~50+ GB; outputs ~10 GB).
- CPU: build is parallel (~30–45 min on 16 cores, ~2 h on 8); simulation is
  essentially single-threaded — single-core speed matters most.
- RAM: 32 GB recommended, 16 GB floor.
- Hard requirement for GPUFS: x86 Linux host with **KVM access** (boot
  fast-forward). Bare-metal Linux is safest; most VMs/default WSL2 won't work.
  SE-mode (GPUSE) doesn't need KVM — another reason to start there.
- Runtime: small HIP kernel / 512×512 GEMM = minutes–hours of sim time; a
  single PyTorch inference layer = many hours–days. Simulate kernels, not
  training runs.

### Track C — Accel-Sim (optional)
- Disk: framework + build ~5 GB; SASS traces are the real cost (MBs for small
  kernels, tens of GB for real apps).
- Needs a real NVIDIA GPU once to generate traces (NVBit); simulation after
  that is CPU-only, single-threaded, hours per workload.

### One machine that covers everything
Linux, **16-core CPU, 32–64 GB RAM, 500 GB free NVMe**. Add one mid-range AMD
GPU and one NVIDIA GPU (or rent cloud hours) for Track A. Priorities: disk and
RAM block gem5 work; GPU horsepower is almost irrelevant at student scale.

## Sources

- gem5 GPUFS docs: https://www.gem5.org/documentation/general_docs/gpu_models/gpufs
- gem5 Vega SE-mode docs: https://www.gem5.org/documentation/general_docs/gpu_models/vega
- gem5 release notes: https://github.com/gem5/gem5/blob/stable/RELEASE-NOTES.md
- ISCA 2024 gem5 GPU/ML tutorial: https://www.gem5.org/assets/files/isca2024-tutorial/05-gpu.pdf
- HPCA 2023 gem5 GPU tutorial: https://www.gem5.org/assets/files/hpca2023-tutorial/gem5-tutorial-hpca23-gpu.pdf
- Accel-Sim: https://accel-sim.github.io/
- GPGPU-Sim 4.0: https://github.com/accel-sim/gpgpu-sim_distribution
- gem5 resources: https://resources.gem5.org/
- BabelStream: https://github.com/UoB-HPC/BabelStream
