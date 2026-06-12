#!/usr/bin/env bash
# preflight.sh — quick environment check on whichever box you run it.
# Tells you what's present and which collectors will work vs be skipped.
#
# Usage:  ./preflight.sh
source "$(dirname "$0")/lib.sh"

banner "Preflight — $(hostname)"
is_wsl && echo "  env: WSL ($(uname -r))" || echo "  env: $(uname -sr)"

check() { if have "$1"; then ok "$1 — $(command -v "$1")"; else warn "$1 missing"; fi; }
step "Core"
check python3; check pip3; check git; check ssh; check rsync
step "ROCm / profiling (cloud MI300X only)"
check rocminfo; check amd-smi; check rocm-smi; check rocprofv3; check rocprof
step "gem5 / containers (optional)"
check docker; check scons
[ -x "$HOME/gem5/build/VEGA_X86/gem5.opt" ] && ok "gem5.opt built" || warn "gem5.opt not built"

step "Python deps"
python3 - <<'PY' 2>/dev/null || warn "python deps check failed"
for m in ("yaml","sklearn","numpy"):
    try:
        __import__(m); print(f"  ✓ {m}")
    except Exception:
        print(f"  ! {m} missing")
try:
    import torch; print(f"  ✓ torch {torch.__version__} (GPU: {torch.cuda.is_available()})")
except Exception:
    print("  ! torch missing (needed on the MI300X box, ROCm build)")
PY

step "Pipeline self-check (env.summarize)"
( cd "$REPO_DIR" && python3 - <<'PY' 2>/dev/null || warn "run scripts/01 or 02 first"
import sys, json; sys.path.insert(0, ".")
from core import env
print(json.dumps(env.summarize(), indent=2))
PY
)
echo
ok "preflight done"
