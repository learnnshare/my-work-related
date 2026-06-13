#!/usr/bin/env bash
# run_all_demo.sh — end-to-end completeness demo:
#   Path 3 (device) + Path 2 (gem5 gfx90a proxy) → predictor → grounded agent → dashboard.
#
# Degrades gracefully: steps that need the GPU/gem5 are skipped if absent, falling
# back to the already-captured records so the predictor/agent/dashboard always run.
# Set ANTHROPIC_API_KEY to use the live LLM agent (else rule-based fallback).
#
# Usage:  bash scripts/run_all_demo.sh
set +e
PIPE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PIPE" || exit 1
GEM5_BIN="${GEM5_BIN:-/workspace/shared/gem5/build/VEGA_X86/gem5.opt}"
LLM=""; [ -n "$ANTHROPIC_API_KEY" ] && LLM="--llm"
hr(){ printf '\n=== %s ===\n' "$*"; }

hr "1) Path 3 — real MI300X capture (skip if no rocprofv3)"
if command -v rocprofv3 >/dev/null 2>&1; then
  python3 -u capture_device.py --sweep --precisions fp16,bf16 2>&1 | tail -6
else
  echo "  no rocprofv3 — using existing device records in data/"
fi

hr "2) Path 2 — gem5 sim (gfx90a proxy; skip if gem5 not built)"
if [ -x "$GEM5_BIN" ]; then
  source /workspace/shared/gem5-tools/env.sh 2>/dev/null
  python3 -u capture_gem5.py --workload gemm --size 2048 --cus 8 --keep 2>&1 | tail -8
  python3 -u capture_gem5.py --workload vectoradd --size 2048 --keep 2>&1 | tail -6
else
  echo "  gem5 not built at $GEM5_BIN — using existing gem5 records (if any)"
fi

hr "3) Predictor — sim→real + real→real held-out (measured error)"
python3 -u predict_sim2real.py 2>&1 | tail -12

hr "4) Grounded agent — per-layer bottleneck reports for every record"
python3 -u agent/agent.py --all $LLM 2>&1 | tail -6

hr "5) Refresh dashboard bundle (records + agent + predictions + scorecard)"
python3 -u refresh_bundle.py $LLM 2>&1 | tail -6

hr "6) Analysis figures"
python3 -u analyze.py 2>&1 | tail -4

hr "DONE — open the dashboard"
echo "  cd ../mi300x-dashboard && python3 -m http.server 8000   # http://<host>:8000/index.html"
echo "  developer.html → grounded agent panel · physical-ai.html → predicted-vs-measured · architect.html → scorecard"
