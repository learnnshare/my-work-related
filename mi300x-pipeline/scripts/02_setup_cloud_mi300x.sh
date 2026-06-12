#!/usr/bin/env bash
# 02_setup_cloud_mi300x.sh — set up the CLOUD Ubuntu 22.04 box with the MI300X.
#
# Verifies ROCm + profiling tools, installs Python deps and PyTorch-ROCm, adds
# you to the render/video groups, and runs the device preflight. ROCm is usually
# preinstalled on MI300X cloud images; pass --install-rocm to install it.
#
# Run this ON THE CLOUD BOX (after: ssh mi300x).
#
# Usage:
#   ./02_setup_cloud_mi300x.sh                 # verify + deps (assumes ROCm present)
#   ./02_setup_cloud_mi300x.sh --install-rocm  # also install ROCm 6.1 (needs sudo)
#   ./02_setup_cloud_mi300x.sh --rocm-ver 6.1.3
source "$(dirname "$0")/lib.sh"

INSTALL_ROCM=0; ROCM_VER="6.1.3"
while [ $# -gt 0 ]; do
  case "$1" in
    --install-rocm) INSTALL_ROCM=1; shift;;
    --rocm-ver) ROCM_VER="$2"; shift 2;;
    *) die "unknown arg: $1";;
  esac
done

banner "Cloud MI300X setup (Ubuntu 22.04 / jammy)"
is_wsl && die "this is the CLOUD script — run it on the MI300X box, not WSL"

# 1. base packages
step "Installing base packages"
SUDO apt-get update -y
SUDO apt-get install -y python3 python3-venv python3-pip git rsync build-essential
ok "base packages installed"

# 2. ROCm (optional install, else verify)
if [ "$INSTALL_ROCM" -eq 1 ]; then
  step "Installing ROCm $ROCM_VER (amdgpu-install)"
  TMP="$(mktemp -d)"
  URL="https://repo.radeon.com/amdgpu-install/${ROCM_VER}/ubuntu/jammy/amdgpu-install_${ROCM_VER%.*}.${ROCM_VER##*.}0-1_all.deb"
  warn "If the URL 404s, get the correct .deb from https://repo.radeon.com/amdgpu-install/${ROCM_VER}/ubuntu/jammy/"
  ( cd "$TMP" && curl -fLO "$URL" && SUDO apt-get install -y ./amdgpu-install_*.deb )
  SUDO amdgpu-install -y --usecase=rocm
  ok "ROCm install attempted — a reboot may be required"
fi

step "Verifying ROCm tooling"
have rocminfo  && ok "rocminfo present" || warn "rocminfo missing (run with --install-rocm or load the ROCm module)"
have rocm-smi  && ok "rocm-smi present"  || warn "rocm-smi missing"
if have rocprofv3; then ok "rocprofv3 present (rocprofiler-sdk)"
elif have rocprof; then warn "only legacy rocprof present — L0/L3 use rocprofv3; consider ROCm >= 6.2"
else warn "no rocprofiler — L0 counters / L3 trace will be skipped"; fi
have rocm-smi && { echo; rocm-smi --showproductname 2>/dev/null || true; echo; }

# 3. group membership for non-root counter/SMI access
step "Adding $(whoami) to render,video groups"
SUDO usermod -aG render,video "$(whoami)" || warn "could not modify groups (need sudo)"
ok "group change takes effect after re-login (exit and ssh back in)"

# 4. python venv + deps
VENV="$REPO_DIR/.venv"
step "Creating virtualenv + installing pipeline deps"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
pip install -r "$REPO_DIR/requirements.txt"
ok "pipeline deps installed"

# 5. PyTorch for ROCm (not the default CUDA wheel!)
step "Installing PyTorch (ROCm build)"
pip install --index-url "https://download.pytorch.org/whl/rocm6.1" torch \
  && ok "torch (ROCm) installed" \
  || warn "torch ROCm install failed — pick the wheel matching your ROCm at pytorch.org"
python - <<'PY' || true
try:
    import torch; print("  torch", torch.__version__, "| GPU visible:", torch.cuda.is_available())
except Exception as e:
    print("  torch not importable yet:", e)
PY

# 6. preflight
step "Running device preflight"
( cd "$REPO_DIR" && python - <<'PY'
import sys; sys.path.insert(0, ".")
from core import env
import json; print(json.dumps(env.summarize(), indent=2))
PY
)

banner "Next steps"
echo "1. Re-login so group changes apply:   ${BLD}exit${RST} then ${BLD}ssh mi300x${RST}"
echo "2. Run a real capture:                ${BLD}python orchestrator.py --config pipeline.device.yaml${RST}"
echo "   (set mode: device; privileged L0 counters / L2 ftrace need sudo)"
echo "3. Copy results back to WSL:           ${BLD}rsync -av mi300x:~/mi300x-pipeline/runs/ ./runs/${RST}"
