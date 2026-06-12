#!/usr/bin/env bash
# 03_setup_gem5.sh — build gem5 with the AMD GPU model (VEGA_X86), or pull the
# gem5 GPU docker image. gem5 can run on the cloud box OR locally in WSL (CPU
# only; it is slow but needs no GPU). GPUFS full-system mode needs KVM.
#
# Usage:
#   ./03_setup_gem5.sh --docker          # pull the prebuilt GPU image (easiest)
#   ./03_setup_gem5.sh --build           # clone + scons build (45+ min)
#   ./03_setup_gem5.sh --build --jobs 16
source "$(dirname "$0")/lib.sh"

MODE=""; JOBS="$(nproc 2>/dev/null || echo 8)"; GEM5_DIR="$HOME/gem5"
while [ $# -gt 0 ]; do
  case "$1" in
    --docker) MODE="docker"; shift;;
    --build)  MODE="build"; shift;;
    --jobs)   JOBS="$2"; shift 2;;
    --dir)    GEM5_DIR="$2"; shift 2;;
    *) die "unknown arg: $1";;
  esac
done
[ -n "$MODE" ] || die "choose a mode: --docker (recommended) or --build"

banner "gem5 setup ($MODE)"
if is_wsl; then
  warn "WSL: gem5 builds/runs on CPU. GPUFS needs KVM — check 'kvm-ok'. SE mode is fine without KVM."
fi

if [ "$MODE" = "docker" ]; then
  have docker || die "docker not installed. Install Docker first (https://docs.docker.com/engine/install/ubuntu/)"
  step "Pulling gem5 GPU docker image"
  docker pull ghcr.io/gem5/gpu-fs:latest \
    && ok "pulled ghcr.io/gem5/gpu-fs:latest" \
    || warn "pull failed — see https://resources.gem5.org for the current GPU image tag"
  echo
  echo "Run gem5 in the container, mounting this repo:"
  echo "  ${BLD}docker run --rm -it -v $REPO_DIR:/work ghcr.io/gem5/gpu-fs:latest bash${RST}"
  exit 0
fi

# --build
step "Installing gem5 build dependencies"
SUDO apt-get update -y
SUDO apt-get install -y build-essential git m4 scons zlib1g-dev libprotobuf-dev \
  protobuf-compiler libprotoc-dev libgoogle-perftools-dev python3-dev libboost-all-dev \
  pkg-config python3-pip
pip3 install --user scons mypy pre-commit 2>/dev/null || true
ok "build deps installed"

step "Cloning gem5 → $GEM5_DIR"
if [ -d "$GEM5_DIR/.git" ]; then ok "gem5 already cloned"; else git clone https://github.com/gem5/gem5.git "$GEM5_DIR"; fi

step "Building VEGA_X86/gem5.opt with $JOBS jobs (this takes a while)"
( cd "$GEM5_DIR" && scons build/VEGA_X86/gem5.opt -j "$JOBS" ) \
  && ok "built $GEM5_DIR/build/VEGA_X86/gem5.opt" \
  || die "build failed — see gem5 docs; the GPU toolchain is fussy, prefer --docker"

banner "Next steps"
echo "Point the pipeline at your binary in pipeline.yaml:"
echo "   ${BLD}gem5.binary: $GEM5_DIR/build/VEGA_X86/gem5.opt${RST}"
echo "Wire collectors/gem5/configs/gpu_mi300x.py to your gem5 GPUFS/GPUSE builders (see RUNBOOK.md),"
echo "then:   ${BLD}python orchestrator.py --config pipeline.yaml${RST}   # mode: gem5"
