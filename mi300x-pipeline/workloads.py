"""
workloads.py — workload presets (mirror of the dashboard's WORKLOADS in sim.js).

Only the fields the pipeline needs: identity + the computational "personality"
used by the gem5 layer mapper (flopsPerItem, weightBytesGB, numKernels) and the
dashboard-facing unit/short/pref/regime. peakTflops is per-precision.
"""
from __future__ import annotations

PEAK_TFLOPS = {"fp32": 163.4, "tf32": 653.7, "fp16": 1307.4, "bf16": 1307.4, "fp8": 2614.9}

WORKLOADS = {
    "gemm": dict(id="gemm", name="rocBLAS GEMM (8192³)", unit="GEMMs", short="gemm",
                 pref="bf16", regime="compute-bound", numKernels=1,
                 flopsPerItem=2 * 8192 ** 3, weightBytesGB=0.4, actBytesPerItem=8192 * 8192 * 2 * 3),
    "ppo_infer": dict(id="ppo_infer", name="PPO Policy Inference (batch-1 robot control)",
                      unit="inferences", short="inf", pref="fp16", regime="launch-bound",
                      numKernels=6, flopsPerItem=1.5e6, weightBytesGB=0.004, actBytesPerItem=2.0e5,
                      target_hz=200),
    "llm7b": dict(id="llm7b", name="LLM Inference — 7B (decode)", unit="tokens", short="tok",
                  pref="fp16", regime="memory-bound", numKernels=120,
                  flopsPerItem=1.4e10, weightBytesGB=14, actBytesPerItem=1.4e7),
    "babelstream": dict(id="babelstream", name="BabelStream (Triad, bandwidth)", unit="iterations",
                        short="iter", pref="fp32", regime="memory-bound", numKernels=4,
                        flopsPerItem=2 * 256e6, weightBytesGB=6, actBytesPerItem=256e6 * 8 * 3),
    "resnet": dict(id="resnet", name="ResNet-50 Inference (vision)", unit="images", short="img",
                   pref="fp16", regime="balanced", numKernels=54,
                   flopsPerItem=8.2e9, weightBytesGB=0.1, actBytesPerItem=6.0e6),
}


def get(workload_id, precision=None, batch=1, num_gpus=1, target_hz=None):
    w = dict(WORKLOADS[workload_id])
    w["precision"] = precision or w["pref"]
    w["batch"] = batch
    w["num_gpus"] = num_gpus
    w["peakTflops"] = PEAK_TFLOPS.get(w["precision"], 1307.4)
    if target_hz:
        w["target_hz"] = target_hz
    return w
