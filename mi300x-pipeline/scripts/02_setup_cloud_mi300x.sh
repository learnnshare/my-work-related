#!/usr/bin/env bash
# 02_setup_cloud_mi300x.sh — set up the CLOUD Ubuntu 22.04 box with the MI300X.
#
# Verifies ROCm + profiling tools, installs Python deps and PyTorch-ROCm, adds
# you to the render/video groups, and runs the device preflight. ROCm is usually
# preinstalled on MI300X cloud images; pass --install-rocm to install it.
#
# Run this ON THE CLOUD BOX (after: ssh mi300x, or in a Jupyter terminal).
# Get the repo onto the box first, e.g.:
#   git clone <your-repo-url> ~/mi300x-pipeline   (Jupyter/cloud)
#   # or from WSL:  rsync -av --exclude runs/ <repo> mi300x:~/mi300x-pipeline/
#
# Tested target: ROCm 7.0 + amd-smi on an MI300X (baremetal, often root).
#
# Usage:
#   ./02_setup_cloud_mi300x.sh                 # verify + deps (assumes ROCm present)
#   ./02_setup_cloud_mi300x.sh --install-rocm  # also install ROCm (needs sudo; rarely needed)
#   ./02_setup_cloud_mi300x.sh --rocm-ver 6.4.1
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

# 1. base packages (non-fatal: containers may restrict apt; tools are often preinstalled)
step "Installing base packages"
if have apt-get; then
  SUDO apt-get update -y || warn "apt-get update failed (restricted container?) — continuing"
  SUDO apt-get install -y python3 python3-venv python3-pip git rsync build-essential \
    || warn "apt install failed — continuing; ensure python3/pip/git exist"
else
  warn "no apt-get — assuming python3/pip/git are already present"
fi
have python3 || die "python3 missing and can't install it — provision it manually"
ok "base packages ready"

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
if have amd-smi; then ok "amd-smi present (ROCm 7+ tool — preferred)"
elif have rocm-smi; then ok "rocm-smi present (legacy tool)"
else warn "neither amd-smi nor rocm-smi found"; fi
if have rocprofv3; then ok "rocprofv3 present (rocprofiler-sdk)"
elif have rocprof; then warn "only legacy rocprof present — L0/L3 prefer rocprofv3"
else warn "no rocprofiler — L0 counters / L3 trace will be skipped"; fi
if have amd-smi; then echo; amd-smi static -g 0 2>/dev/null | head -20 || amd-smi list 2>/dev/null || true; echo
elif have rocm-smi; then echo; rocm-smi --showproductname 2>/dev/null || true; echo; fi

# 3. group membership for non-root counter/SMI access (skip if root)
if is_root; then
  ok "running as root — no group changes needed; privileged collectors (L0 counters, L2 ftrace) work directly"
else
  step "Adding $(whoami) to render,video groups"
  SUDO usermod -aG render,video "$(whoami)" || warn "could not modify groups (need sudo)"
  ok "group change takes effect after re-login (exit and ssh back in)"
fi

# 4. python deps (venv if possible; else --user, common in container images)
VENV="$REPO_DIR/.venv"
step "Installing pipeline deps"
if python3 -m venv "$VENV" 2>/dev/null; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip >/dev/null 2>&1 || true
  pip install -r "$REPO_DIR/requirements.txt" && ok "deps installed in venv ($VENV)"
else
  warn "venv unavailable — installing with pip --user into the base env"
  python3 -m pip install --user -r "$REPO_DIR/requirements.txt" && ok "deps installed (--user)"
fi

# 5. PyTorch for ROCm. A CUDA/CPU wheel (e.g. 2.10.0+cu128) CANNOT use the
# MI300X — detect that and replace it with a ROCm build.
step "Checking PyTorch build"
TORCH_STATE="$(python - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print("ok")
    elif getattr(torch.version, "hip", None) is None:
        print("wrongbuild")   # CUDA/CPU wheel on an AMD box
    else:
        print("rocm_no_gpu")
except Exception:
    print("missing")
PY
)"
if [ "$TORCH_STATE" = "ok" ]; then
  ok "torch is a ROCm build and sees the GPU — leaving it alone"
else
  if [ "$TORCH_STATE" = "wrongbuild" ]; then
    warn "installed torch is a CUDA/CPU build (no HIP) — it can't use the MI300X; replacing"
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  elif [ "$TORCH_STATE" = "missing" ]; then
    warn "torch missing — installing a ROCm build"
  else
    warn "torch is ROCm but GPU not visible — check drivers; will try a reinstall"
  fi
  # auto-pick the best available ROCm wheel index (your box reaches pytorch.org)
  IDX=""
  for cand in ${TORCH_ROCM_INDEX:-} rocm7.0 rocm6.4 rocm6.3; do
    url="$cand"; case "$cand" in http*) ;; *) url="https://download.pytorch.org/whl/$cand";; esac
    if curl -sf -o /dev/null "$url/"; then IDX="$url"; break; fi
  done
  [ -n "$IDX" ] || IDX="https://download.pytorch.org/whl/rocm6.4"
  step "Installing torch from $IDX"
  pip install --index-url "$IDX" torch || warn "torch ROCm install failed — see pytorch.org/get-started/locally"
fi
python - <<'PY' || true
try:
    import torch
    print("  torch", torch.__version__, "| hip", getattr(torch.version,"hip",None),
          "| GPU visible:", torch.cuda.is_available())
except Exception as e:
    print("  torch not importable:", e)
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
