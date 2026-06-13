# MI300X / ROCm grounding knowledge (for the agent)

Concise, factual grounding the agent uses to turn raw L0–L7 counters into
hardware-specific advice. Keep terse and true; the agent cites the *record's*
measured numbers, this file supplies the architectural context.

## Device (AMD Instinct MI300X, CDNA 3, gfx942)
- 304 compute units across 8 XCD chiplets; wavefront 64; 4 SIMDs/CU.
- Memory: 192 GB HBM3, ~5.3 TB/s peak bandwidth, 8192-bit bus.
- Caches: L1 vector 32 KB/CU, L2 (TCC) 4 MB, Infinity Cache (LLC) 256 MB.
- Peak compute (TFLOPS): fp32 163, tf32 654, fp16/bf16 1307, fp8 2615.
- Engine clock ~2100 MHz; board power ~750 W.

## Precision / matrix cores (MFMA)
- The matrix cores (MFMA) are used for fp16/bf16/fp8 GEMM **only with fp32
  accumulate** (HHS/BBS path via `gemm_ex` / hipBLASLt). Pure fp16-accumulate
  `hgemm` runs on the VALU and does NOT use MFMA → far lower TFLOPS.
- fp8 GEMM lives in **hipBLASLt** on ROCm 7 (rocBLAS dropped the fp8 enum);
  fp8 ≈ 1.7–2× fp16 throughput when the matrix path engages.
- Counter mapping: fp16 MFMA → `SQ_INSTS_VALU_MFMA_MOPS_F16`, bf16 →
  `..._BF16`, fp8 → `..._F8`. MFMA util 0% with high `SQ_INSTS_VALU` ⇒ matrix
  cores idle (likely wrong precision/accumulate path).

## Roofline intuition
- Memory-bound (memUtil ≫ computeUtil): batch-1 inference, small/streaming
  kernels, decode. Fixes: raise arithmetic intensity, tile/block for L2 reuse,
  fuse ops, batch more work.
- Compute-bound (computeUtil ≥ memUtil): large GEMM/training. Fixes: lower
  precision (fp16→fp8), bigger tiles, ensure MFMA path.
- Launch/host-bound (high host or launch overhead %): tiny kernels, many
  dispatches, CPU-coupled (e.g. MuJoCo physics). Fixes: HIP graphs, kernel
  fusion, fewer/bigger dispatches.

## Robotics / batch-1 over-provisioning
- A 304-CU / 192 GB MI300X is hugely over-provisioned for a single robot
  policy's batch-1 inference → very low occupancy/util. The right question is
  not "make it faster" but "is this the right (smaller/partitioned) device?"
  Consider CPX/NPS partitioning to slice the GPU.

## SR-IOV VF measurement limits (this box)
- The virtual function does NOT expose power/temp/clock sensors → those are
  reported `null` (honest), not estimated.
- Memory bandwidth on the VF is derived from `TCC_MISS × 64 B` (L2 misses →
  HBM traffic) as a proxy; treat bandwidth as approximate, TFLOPS/MFMA/cache-hit
  as direct.

## gem5 (Path 2) fidelity
- gem5 SE mode models **gfx90a (MI200, CDNA2)** as a proxy for gfx942 here;
  cache sizes/ISA differ → architectural trends transfer, absolute values do
  not. True gfx942 needs GPUFS+KVM (future).
