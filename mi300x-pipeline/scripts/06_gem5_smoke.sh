#!/usr/bin/env bash
# 06_gem5_smoke.sh — smoke-test the gem5 GPU model: run a TINY HIP kernel in
# gem5 GPUSE (apu_se.py) on the gfx942 model and check it produces stats.txt.
# This is the gating test for Path 2 (does gem5 simulate a kernel on this box?).
#
# gem5 is ~1e4-1e5x slower than hardware, so we run a tiny problem (n=2048, 1
# launch) on a few CUs — seconds-to-minutes of sim, not the real-capture size.
#
# Usage:   bash scripts/06_gem5_smoke.sh
#          CUS=8 N=4096 ITERS=1 bash scripts/06_gem5_smoke.sh
set +e

GEM5_DIR="${GEM5_DIR:-/workspace/shared/gem5}"
ENVSH="${ENVSH:-/workspace/shared/gem5-tools/env.sh}"
B="$GEM5_DIR/build/VEGA_X86/gem5.opt"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="$REPO/bench/vectoradd"
OUT="$REPO/runs/gem5_smoke/m5out"
CUS="${CUS:-4}"; N="${N:-2048}"; ITERS="${ITERS:-1}"; GFX="${GFX:-gfx942}"
# MI300X is a DISCRETE gpu -> apu_se.py needs --dgpu for gfx942/gfx90a/gfx908
DGPU="${DGPU:---dgpu}"

echo "== gem5 GPUSE smoke test =="
[ -f "$ENVSH" ] && { . "$ENVSH"; echo "sourced $ENVSH"; }
[ -x "$B" ] || { echo "gem5.opt not found at $B — build it with scripts/05_build_gem5_shared.sh"; exit 1; }
command -v hipcc >/dev/null || { echo "hipcc not found"; exit 1; }

echo "compiling tiny-capable vectoradd ..."
hipcc "$REPO/bench/vectoradd.cpp" -o "$APP" 2>&1 | grep -i error
[ -x "$APP" ] || { echo "compile failed"; exit 1; }

mkdir -p "$OUT"
echo "running: gem5 apu_se.py $DGPU  gfx=$GFX  CUs=$CUS  vectoradd n=$N iters=$ITERS"
echo "(this is gem5 — be patient; a tiny run can still take a few minutes)"
cd "$GEM5_DIR" || exit 1
"$B" --outdir="$OUT" configs/example/apu_se.py \
    $DGPU --gfx-version "$GFX" -u "$CUS" \
    -c "$APP" -o "$N $ITERS" 2>&1 | tail -40

echo
echo "== result =="
if [ -f "$OUT/stats.txt" ]; then
    echo "stats.txt produced — key GPU stats:"
    grep -iE "simSeconds|system.cpu.*numCycles|gpu.*numCycles|TCC|TCP|numKernels|CUs.*Inst|mem_ctrls.*bytes" \
        "$OUT/stats.txt" | head -25 | sed 's/^/  /'
    echo
    echo "FULL stats at: $OUT/stats.txt"
    echo "config at:     $OUT/config.json"
    echo
    echo "If this looks sane, commit it so I can wire the gem5->record path:"
    echo "  cp -r $OUT $REPO/../testfolder/gem5_smoke_m5out"
    echo "  git add ../testfolder/gem5_smoke_m5out && git commit -m 'gem5 smoke m5out' && git push"
else
    echo "NO stats.txt — the run didn't complete. Paste the output above; the"
    echo "common GPUSE issues are ROCm-version/driver mismatch or kernel dispatch."
fi
