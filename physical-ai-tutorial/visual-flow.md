# Visual Guide — How Everything Fits Together

Companion to `benchmarking-rocm-vs-cuda-guide.md`. View in VS Code markdown
preview or on GitHub to see the rendered diagrams.

## 1. The big picture — three tracks, one research question

The key insight: Track A is the *only* place ROCm and CUDA are directly
compared. Tracks B and C are separate, single-vendor microarchitecture
studies — never compare absolute numbers between two different simulators.

```mermaid
flowchart TD
    Q["Research question:<br/>How do ROCm and CUDA compare<br/>on performance and latency?"]

    Q --> TA
    Q --> TB
    Q --> TC

    subgraph TA["Track A · Real hardware — THE comparison"]
        direction TB
        A1["Layer 1 · Microbenchmarks<br/>BabelStream · GEMM · launch latency"]
        A2["Layer 2 · PyTorch operators<br/>conv · attention · GEMM"]
        A3["Layer 3 · Application<br/>MuJoCo + PPO walker training"]
        A1 --> A2 --> A3
    end

    subgraph TB["Track B · gem5 + ROCm — AMD-only"]
        direction TB
        B1["Build gem5 VEGA_X86"]
        B2["GPUSE mode · HIP samples"]
        B3["GPUFS mode · MI210 / MI300X"]
        B4["Sweep CU count, caches, bandwidth"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph TC["Track C · Accel-Sim — NVIDIA-only, optional"]
        direction TB
        C1["Trace kernels with NVBit<br/>(needs a real NVIDIA GPU once)"]
        C2["Simulate SASS traces on CPU"]
        C1 --> C2
    end

    TA --> R1["Deliverable 1:<br/>cross-stack comparison, normalized"]
    TB --> R2["Deliverable 2:<br/>AMD microarchitecture insights"]
    TC --> R3["Deliverable 3:<br/>NVIDIA microarchitecture insights"]

    TB -. "do NOT compare absolute numbers<br/>across different simulators" .- TC
```

## 2. The controlled experiment — same code, two stacks

Why MuJoCo + PyTorch makes a fair benchmark: physics runs on the CPU
(identical on both machines), and the same Python code drives either GPU
stack. The only thing that changes between machines is everything below the
PyTorch wheel.

```mermaid
flowchart TB
    subgraph WL["Identical workload code on both machines"]
        direction LR
        MJ["MuJoCo physics<br/>(CPU)"] --- GYM["Gymnasium<br/>environment"] --- PPO["PPO agent · PyTorch<br/>torch.device('cuda') works on BOTH"]
    end

    PPO --> AMDW
    PPO --> NVW

    subgraph AMD["AMD machine — ROCm stack"]
        direction TB
        AMDW["PyTorch ROCm wheel"] --> HIPL["HIP · rocBLAS · MIOpen"]
        HIPL --> ROCR["ROCr runtime"]
        ROCR --> ADRV["amdgpu kernel driver"]
        ADRV --> AGPU["AMD GPU<br/>e.g. RX 7900 XTX"]
    end

    subgraph NV["NVIDIA machine — CUDA stack"]
        direction TB
        NVW["PyTorch CUDA wheel"] --> CUL["cuBLAS · cuDNN"]
        CUL --> CURT["CUDA runtime"]
        CURT --> NDRV["NVIDIA kernel driver"]
        NDRV --> NGPU["NVIDIA GPU<br/>e.g. RTX 4070"]
    end
```

## 3. The measurement loop — what makes it "controlled"

Every number you report should have gone through this loop. The synchronize
step is the one beginners miss: GPU launches are asynchronous, so timing
without it measures nothing.

```mermaid
flowchart TD
    S1["Set up identical environments<br/>same PyTorch version, seeds, model, batch size"]
    S1 --> S2["Pin clocks, log temperature<br/>nvidia-smi -lgc / rocm-smi --setperflevel"]
    S2 --> S3["Warm up — discard first iterations<br/>(JIT compile, kernel autotuning)"]
    S3 --> S4["Run one measurement"]
    S4 --> S5["torch.cuda.synchronize()<br/>THEN stop the timer"]
    S5 --> S6{"30+ runs<br/>collected?"}
    S6 -- "no" --> S4
    S6 -- "yes" --> S7["Report median + spread,<br/>never a single number"]
    S7 --> S8["Per-kernel profile<br/>rocprofiler / Nsight Compute"]
    S8 --> S9["Normalize by each card's peak<br/>FLOPS and memory bandwidth"]
    S9 --> S10["Fair stack comparison"]
```

## 4. The gem5 experiment loop — architecture exploration

This is what gem5 buys you that real hardware can't: change the architecture,
rerun the same kernel. Note the two modes — start in GPUSE (simpler, no KVM),
graduate to GPUFS (real driver, needs KVM on bare-metal Linux).

```mermaid
flowchart TD
    G1["Pull gem5 GPU Docker image<br/>~10–15 GB"]
    G1 --> G2["Build gem5 VEGA_X86<br/>~30–45 min on 16 cores"]
    G2 --> G3["GPUSE syscall-emulation mode<br/>run the 'square' HIP sample"]
    G3 --> G4{"Works?"}
    G4 -- "no — fix toolchain<br/>(use the Docker image!)" --> G3
    G4 -- "yes" --> G5["GPUFS full-system mode<br/>needs KVM + ROCm 6.1 disk image (~20–30 GB)"]
    G5 --> G6["Run a small kernel<br/>e.g. 512x512 GEMM via rocBLAS"]
    G6 --> G7["Collect stats.txt<br/>cycles, cache hit rates, memory traffic"]
    G7 --> G8["Change ONE parameter<br/>CU count / cache size / bandwidth"]
    G8 --> G6
    G7 --> G9["Architecture study:<br/>e.g. GEMM latency vs CU count on MI210"]
```

## 5. Suggested roadmap

```mermaid
gantt
    title Suggested roadmap (Track B is phase 2)
    dateFormat YYYY-MM-DD
    section Track A
    MuJoCo + PPO running locally          :a1, 2026-06-15, 7d
    HIP microbenchmarks, Layer 1          :a2, after a1, 7d
    Run all 3 layers on both stacks       :a3, after a2, 14d
    Analyze, normalize, write up          :a4, after a3, 7d
    section Track B, phase 2
    gem5 build + GPUSE square             :b1, after a2, 14d
    GPUFS + GEMM parameter sweeps         :b2, after b1, 14d
    Architecture study write-up           :b3, after b2, 7d
```

## How to read these together

1. Diagram 1 is the map — three independent tracks, three deliverables.
2. Diagram 2 explains *why* Track A is a fair experiment.
3. Diagram 3 is the procedure you repeat for every measurement in Track A.
4. Diagram 4 is the procedure for Track B.
5. Diagram 5 is when to do what.
