#!/usr/bin/env bash
# setup_persistent_env.sh — install Python packages INTO /workspace/shared so
# they survive the box's 4-hour reset (only /workspace/shared persists).
#
# Strategy: `pip install --target=<shared>/site-packages` + a PYTHONPATH export.
# This is resume-proof — it doesn't rely on venv symlinks into the base image,
# and PYTHONPATH is searched BEFORE system site-packages, so a ROCm torch here
# shadows the image's CUDA torch.
#
# Usage:
#   bash scripts/setup_persistent_env.sh                 # core deps (pyyaml, sklearn, numpy)
#   bash scripts/setup_persistent_env.sh --with-torch    # also persist ROCm PyTorch (big)
#   PREFIX=/workspace/shared/myenv bash scripts/setup_persistent_env.sh
#
# AFTER it finishes (and on every resume):  source <PREFIX>/activate.sh
source "$(dirname "$0")/lib.sh"

PREFIX="${PREFIX:-/workspace/shared/mi300x-env}"
SITE="$PREFIX/site-packages"
WITH_TORCH=0
[ "${1:-}" = "--with-torch" ] && WITH_TORCH=1

banner "Persistent env → $PREFIX"
case "$PREFIX" in
  /workspace/shared/*) ok "target is under /workspace/shared (persists across resets)";;
  *) warn "PREFIX is NOT under /workspace/shared — it will be lost on reset!";;
esac
mkdir -p "$SITE"

# free space check (this mount is small — ~28 GB on the hack box)
step "Disk on the target mount"
df -h "$PREFIX" | sed 's/^/  /'
[ "$WITH_TORCH" = "1" ] && warn "ROCm torch is several GB — make sure the mount has room"

# 1. core deps into the shared dir
step "Installing core requirements into $SITE"
python3 -m pip install --target="$SITE" --upgrade -r "$REPO_DIR/requirements.txt"
ok "core deps installed"

# 2. optional: persist a ROCm PyTorch (shadows the image's CUDA build)
if [ "$WITH_TORCH" = "1" ]; then
  step "Selecting a ROCm torch wheel index"
  IDX=""
  for cand in ${TORCH_ROCM_INDEX:-} rocm7.0 rocm6.4 rocm6.3; do
    url="$cand"; case "$cand" in http*) ;; *) url="https://download.pytorch.org/whl/$cand";; esac
    if have curl && curl -sf -o /dev/null "$url/"; then IDX="$url"; break; fi
  done
  [ -n "$IDX" ] || IDX="https://download.pytorch.org/whl/rocm6.4"
  step "Installing torch (ROCm) from $IDX into $SITE"
  python3 -m pip install --target="$SITE" --upgrade --index-url "$IDX" torch \
    && ok "ROCm torch persisted" || warn "torch install failed — check space / index"
fi

# 3. write the activate file you source on every resume
ACT="$PREFIX/activate.sh"
cat > "$ACT" <<EOF
# source this on every resume:  source $ACT
export PYTHONPATH="$SITE:\${PYTHONPATH:-}"
export PATH="$SITE/bin:\$PATH"
echo "[mi300x-env] PYTHONPATH -> $SITE"
EOF
ok "wrote $ACT"

banner "Use it"
echo "Now and on every resume, run:"
echo "   ${BLD}source $ACT${RST}"
echo "Verify:"
echo "   ${BLD}python3 -c 'import yaml,sklearn,numpy; print(\"deps OK\")'${RST}"
[ "$WITH_TORCH" = "1" ] && echo "   ${BLD}python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'${RST}"
echo
echo "Tip: add the source line to your shell startup so it's automatic:"
echo "   ${BLD}echo 'source $ACT' >> ~/.bashrc${RST}   # if ~ persists; else keep it in /workspace/shared"
