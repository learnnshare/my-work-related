#!/usr/bin/env bash
# 07_gem5_apptainer.sh — run gem5 GPU sim via Apptainer (rootless) on a SHARED
# university server, with EVERYTHING under /home (no sudo, no daemon, no /tmp on
# the shared fs, no impact on other users).
#
# Why apptainer + this image: the gem5 'gcn-gpu' image bundles the OLDER ROCm
# userspace that gem5's GPU model emulates — so GPU kernels actually DISPATCH
# (the thing that failed on the ROCm-7 cloud VF, where CU counters were 0).
#
# SE mode needs NO /dev/kvm (you're not in the kvm group — that's fine).
# Resumable: skips the pull/build if already done.
#
# Usage:  bash scripts/07_gem5_apptainer.sh            # full: pull → build → run
#         STEP=pull  bash scripts/07_gem5_apptainer.sh # just pull the image
#         STEP=build bash scripts/07_gem5_apptainer.sh # just build gem5 inside
#         STEP=run   bash scripts/07_gem5_apptainer.sh # just run the GPU smoke
set +e

ROOT="${GEM5_HOME:-/home/amarnath/gem5-gpu}"          # ALL artifacts live here
SIF="$ROOT/gcn-gpu.sif"
GEM5="$ROOT/gem5"
IMAGE="${IMAGE:-docker://ghcr.io/gem5/gcn-gpu:latest}"
JOBS="${JOBS:-8}"   # modest default — be a good neighbor on the shared server (set JOBS=N to change)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STEP="${STEP:-all}"

# keep apptainer cache + build scratch OFF the shared /tmp — in /home only
export APPTAINER_CACHEDIR="$ROOT/.apptainer/cache"
export APPTAINER_TMPDIR="$ROOT/.apptainer/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$ROOT"

say(){ printf '\n=== %s ===\n' "$*"; }
command -v apptainer >/dev/null || { echo "apptainer not found"; exit 1; }
say "Apptainer $(apptainer --version) · all under $ROOT · rootless (no sudo)"

# 1. pull the gem5 GPU image → a single .sif in /home
if [ "$STEP" = "all" ] || [ "$STEP" = "pull" ]; then
  if [ -f "$SIF" ]; then echo "  SIF exists: $SIF (skip pull)"
  else
    say "Pulling $IMAGE → $SIF  (several GB; minutes)"
    apptainer pull --force "$SIF" "$IMAGE" || { echo "pull failed"; exit 1; }
  fi
  [ "$STEP" = "pull" ] && { echo "pull done"; exit 0; }
fi
[ -f "$SIF" ] || { echo "no SIF — run STEP=pull first"; exit 1; }

# 2. clone + build gem5 INSIDE the container (matched ROCm/toolchain), into /home
if [ "$STEP" = "all" ] || [ "$STEP" = "build" ]; then
  [ -d "$GEM5/.git" ] || git clone --depth 1 --branch stable https://github.com/gem5/gem5.git "$GEM5"
  if [ -x "$GEM5/build/VEGA_X86/gem5.opt" ]; then echo "  gem5.opt exists (skip build)"
  else
    say "Building gem5 VEGA_X86 inside the container ($JOBS jobs; 45+ min)"
    apptainer exec --bind /home/amarnath "$SIF" bash -lc \
      "cd '$GEM5' && scons build/VEGA_X86/gem5.opt -j$JOBS --ignore-style" \
      || { echo "build failed"; exit 1; }
  fi
  [ "$STEP" = "build" ] && { echo "build done"; exit 0; }
fi
[ -x "$GEM5/build/VEGA_X86/gem5.opt" ] || { echo "no gem5.opt — run STEP=build"; exit 1; }

# 3. compile a tiny kernel FOR gfx900 INSIDE the container (matched ROCm), run it
if [ "$STEP" = "all" ] || [ "$STEP" = "run" ]; then
  OUT="$ROOT/m5out_smoke"; mkdir -p "$OUT"
  VADD="$ROOT/vectoradd_gfx900"
  say "Compiling vadd for gfx900 + running gem5 SE inside the container"
  apptainer exec --bind /home/amarnath "$SIF" bash -lc "
    set -e
    hipcc --offload-arch=gfx900 '$REPO/bench/vectoradd.cpp' -o '$VADD'
    cd '$GEM5'
    build/VEGA_X86/gem5.opt --outdir='$OUT' configs/example/apu_se.py \
        --dgpu --gfx-version gfx900 -u 4 -c '$VADD' -o '2048 1'
  " 2>&1 | tail -25
  say "Result"
  VALU=$(grep -m1 'CUs0.vALUInsts' "$OUT/stats.txt" 2>/dev/null | awk '{print $2}')
  if [ -n "$VALU" ] && [ "$VALU" != "0" ]; then
    echo "  ✅ REAL GPU WORK: vALUInsts=$VALU (kernel dispatched!)"
    grep -iE "simSeconds|vALUInsts|vectorMem|TCC|numKernel" "$OUT/stats.txt" | head -15 | sed 's/^/    /'
    echo "  → m5out at $OUT ; turn it into a record:"
    echo "     python3 $REPO/capture_gem5.py --m5out $OUT --keep    # (after wiring --m5out)"
  else
    echo "  GPU still idle (vALUInsts=0). The image's ROCm may not match gfx900 dispatch;"
    echo "  try the gem5-bundled 'square' sample inside the image, or report output."
  fi
fi
