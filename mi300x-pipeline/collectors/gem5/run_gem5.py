"""
run_gem5.py — configure + launch a gem5 GPU run, or ingest an existing m5out/.

On a machine with gem5 built, this invokes:
    <gem5.opt> <gpu config .py> --workload ... --num-compute-units N ...
bracketing the kernel of interest with m5 workbegin/workend so stats.txt isolates
the kernel region. With no gem5 available (e.g. dev box), point `stats_fixture`
at a recorded stats.txt + config.json to exercise the full parse/map/normalize
path locally.

Returns a single CollectorResult carrying all gem5-derived scalars + fidelity.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

from core.interface import CollectorResult, Cadence
from . import stats_parser, config_extractor, map_layers


def _resolve_m5out(cfg, run_ctx, workload):
    """Run gem5 (if available) and return path to m5out, else use fixture."""
    fixture = cfg.get("stats_fixture")
    binary = cfg.get("binary")
    gem5_mode = cfg.get("mode", "GPUFS")

    if binary and shutil.which(binary) or (binary and Path(binary).exists()):
        m5out = run_ctx["raw_dir"] / "gem5" / "m5out"
        m5out.mkdir(parents=True, exist_ok=True)
        cmd = _build_cmd(cfg, workload, m5out)
        try:
            subprocess.run(cmd, check=True, cwd=str(m5out.parent),
                           timeout=cfg.get("timeout_s", 7200))
            return m5out, gem5_mode, None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            return None, gem5_mode, f"gem5 launch failed: {e}"

    if fixture:
        return Path(fixture).parent, gem5_mode, None

    return None, gem5_mode, "no gem5 binary and no stats_fixture provided"


def _build_cmd(cfg, workload, m5out):
    knobs = cfg.get("knobs", {})
    config_py = cfg.get("config", str(Path(__file__).parent / "configs" / "gpu_mi300x.py"))
    cmd = [cfg["binary"], f"--outdir={m5out}", config_py]
    cmd += ["--mode", cfg.get("mode", "GPUFS")]
    if knobs.get("cus"):
        cmd += ["--num-compute-units", str(knobs["cus"])]
    if cfg.get("disk_image"):
        cmd += ["--disk-image", cfg["disk_image"]]
    cmd += ["--workload", workload.get("id", "gemm")]
    cmd += ["--batch", str(workload.get("batch", 1))]
    cmd += ["--precision", str(workload.get("precision", workload.get("pref", "bf16")))]
    return cmd


def run_gem5(cfg, run_ctx, workload) -> CollectorResult:
    res = CollectorResult(layer_id=-1, cadence=Cadence.PERKERNEL)
    base, gem5_mode, err = _resolve_m5out(cfg, run_ctx, workload)
    if err and base is None:
        res.errors.append(err)
        return res

    stats_path = cfg.get("stats_fixture") or (base / "stats.txt")
    config_path = cfg.get("config_fixture") or (base / "config.json")
    if not Path(stats_path).exists():
        res.errors.append(f"stats.txt not found at {stats_path}")
        return res

    region = stats_parser.load_region(stats_path, index=cfg.get("region_index", -1))
    knobs = cfg.get("knobs", {})
    # carry the extracted realized config into run_config for normalization
    params = config_extractor.extract_params(config_path, knobs_fallback=knobs)
    run_config = dict(params["raw"])
    run_config["numGPUs"] = workload.get("num_gpus", 1)
    if "peakTflops" in workload:
        run_config["peakTflops"] = workload["peakTflops"]

    scalars, fidelity = map_layers.gem5_to_scalars(region, run_config, workload, mode=gem5_mode)
    res.scalars.update(scalars)
    res.fidelity.update(fidelity)
    res.raw_artifacts.append(str(stats_path))
    # stash the extracted gem5 params for the architect view
    res.scalars["_gem5_params"] = params
    res.scalars["_run_config"] = run_config
    return res
