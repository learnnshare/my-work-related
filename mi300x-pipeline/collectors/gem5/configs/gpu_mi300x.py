"""
gpu_mi300x.py — gem5 GPU config TEMPLATE for an MI300X-like (gfx942) model.

This is a STARTING POINT to be run by gem5 itself (`gem5.opt gpu_mi300x.py ...`),
not by the host Python. It imports m5/gem5 objects that only exist inside the
gem5 binary. The runner (run_gem5.py) passes knobs as CLI args.

In practice you would base this on gem5's bundled GPUFS config
(configs/example/gpufs/mi300x.py in recent gem5) and override the knobs below.
Kept minimal here as the integration point; see RUNBOOK.md for build/run.
"""
import argparse

# These imports resolve only inside gem5:
try:
    import m5
    from m5.objects import *  # noqa: F401,F403
    INSIDE_GEM5 = True
except Exception:  # imported by host tooling for linting — do nothing
    INSIDE_GEM5 = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="GPUFS", choices=["GPUFS", "GPUSE"])
    ap.add_argument("--num-compute-units", type=int, default=304)
    ap.add_argument("--num-xcds", type=int, default=8)
    ap.add_argument("--clock", default="2100MHz")
    ap.add_argument("--l1-kb", type=int, default=32)
    ap.add_argument("--l2-mb", type=int, default=4)
    ap.add_argument("--llc-mb", type=int, default=256)
    ap.add_argument("--hbm-gb", type=int, default=192)
    ap.add_argument("--mem-type", default="HBM_2000_4H_1x64")
    ap.add_argument("--disk-image", default=None)
    ap.add_argument("--workload", default="gemm")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--precision", default="bf16")
    return ap.parse_args()


def build_system(args):
    """Construct the gem5 System with a GPU_VIPER Ruby protocol + HBM model.
    Skeleton — wire to gem5's GPUFS/GPUSE builders for your gem5 version."""
    raise NotImplementedError(
        "Wire this to your gem5 tree's GPUFS/GPUSE builders. Use --num-compute-units, "
        "cache sizes, and --mem-type to realize the MI300X knobs. Bracket the kernel of "
        "interest with m5.workbegin()/m5.workend() so stats.txt isolates the kernel region."
    )


if __name__ == "__m5_main__" and INSIDE_GEM5:
    args = parse_args()
    system = build_system(args)
    root = Root(full_system=(args.mode == "GPUFS"), system=system)  # noqa: F405
    m5.instantiate()
    print(f"[gpu_mi300x] running {args.workload} batch={args.batch} prec={args.precision} "
          f"CUs={args.num_compute_units} mode={args.mode}")
    exit_event = m5.simulate()
    print(f"[gpu_mi300x] exiting @ {m5.curTick()} because {exit_event.getCause()}")
