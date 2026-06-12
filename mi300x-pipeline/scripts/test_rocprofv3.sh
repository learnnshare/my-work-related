#!/usr/bin/env bash
# test_rocprofv3.sh — probe whether this box (a SR-IOV VF) can:
#   1) run a HIP kernel at all,
#   2) be traced by rocprofv3 (--kernel-trace),
#   3) expose hardware performance counters (--pmc).
#
# This decides whether REAL L0/L3/L4 capture is possible here, or whether the VF
# blocks counters (in which case capture must happen on the host/PF).
#
# Run on the box:
#   cd mi300x-pipeline
#   bash scripts/test_rocprofv3.sh | tee testfolder/test-rocprofv3.log
#   git add testfolder/test-rocprofv3.log && git commit -m "rocprofv3 probe" && git push
set +e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/testfolder/rocprofv3-test"; mkdir -p "$OUT"
APP="$OUT/vectoradd"

hr(){ printf '\n========== %s ==========\n' "$*"; }

hr "0. Tools"
echo "hipcc:     $(command -v hipcc || echo MISSING)"
echo "rocprofv3: $(command -v rocprofv3 || echo MISSING)"
rocprofv3 --version 2>&1 | head -3

hr "1. Compile HIP microkernel (hipcc)"
echo "\$ hipcc $ROOT/bench/vectoradd.cpp -o $APP"
hipcc "$ROOT/bench/vectoradd.cpp" -o "$APP" 2>&1
[ -x "$APP" ] && echo "  compiled OK" || { echo "  COMPILE FAILED — stop here, paste this log"; exit 1; }

hr "2. Run bare (does a HIP kernel run on this VF?)"
echo "\$ $APP"
timeout 60 "$APP" 2>&1
echo "  exit=$?"

hr "3. rocprofv3 kernel trace (--kernel-trace)"
echo "\$ rocprofv3 --kernel-trace --output-format csv -d $OUT/trace -- $APP"
timeout 120 rocprofv3 --kernel-trace --output-format csv -d "$OUT/trace" -- "$APP" 2>&1 | tail -10
echo "  --- trace CSV files ---"
find "$OUT/trace" -name '*.csv' 2>/dev/null | while read -r f; do echo "  $f"; head -5 "$f" | sed 's/^/    /'; done

hr "4. rocprofv3 hardware counters (--pmc)  [the key test]"
# small, broadly-available gfx942 counter set; if it errors we list what's allowed
echo "\$ rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE GRBM_COUNT --output-format csv -d $OUT/pmc -- $APP"
timeout 120 rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE GRBM_COUNT \
    --output-format csv -d "$OUT/pmc" -- "$APP" 2>&1 | tail -20
echo "  --- counter CSV files ---"
find "$OUT/pmc" -name '*.csv' 2>/dev/null | while read -r f; do echo "  $f"; head -5 "$f" | sed 's/^/    /'; done

hr "5. If step 4 failed: list available counters"
echo "\$ rocprofv3 --list-avail (head)"
rocprofv3 --list-avail 2>&1 | head -40

hr "DONE"
echo "Verdict to look for:"
echo "  - step 2 prints 'OK ran 50 vadd launches'  -> kernels run on the VF"
echo "  - step 3 trace CSV has rows                 -> rocprofv3 tracing works"
echo "  - step 4 counter CSV has SQ_WAVES values    -> HW counters work (REAL L0 capture possible!)"
echo "  - step 4 error / empty                      -> VF blocks counters (capture needs host/PF)"
echo "Paste testfolder/test-rocprofv3.log back."
