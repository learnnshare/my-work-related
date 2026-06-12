#!/usr/bin/env bash
# 05_build_gem5_shared.sh — build gem5 (AMD GPU model) ENTIRELY under
# /workspace/shared so both the binary AND its dependencies survive the box reset.
#
# This box HAS sudo+apt, but apt installs to system dirs that are wiped on reset.
# So instead of installing system-wide, we DOWNLOAD the .deb build deps and
# EXTRACT them into /workspace/shared/gem5-tools/aptroot, then point the compiler
# and linker there with env vars. scons is pip-installed into the same shared
# tree. gem5 source + binary go to /workspace/shared/gem5.
#
# After it finishes, and on EVERY resume, just:   source <env file printed below>
#
# Usage:   bash scripts/05_build_gem5_shared.sh
set +e

SHARED="${SHARED:-/workspace/shared}"
GEM5_DIR="$SHARED/gem5"
TOOLS="$SHARED/gem5-tools"
APTROOT="$TOOLS/aptroot"
DEBS="$TOOLS/debs"
PYSITE="$TOOLS/pysite"
ENVSH="$TOOLS/env.sh"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
GEM5_BRANCH="${GEM5_BRANCH:-stable}"
TARGET="${BUILD_TARGET:-VEGA_X86}"
ARCH="x86_64-linux-gnu"

# gem5 build/runtime deps to pull into the shared aptroot (gcc itself is in the
# base image already). python3-dev gives Python.h; protobuf/zlib/boost are gem5 deps.
PKGS="scons python3-dev libpython3-dev m4 zlib1g-dev libprotobuf-dev protobuf-compiler libgoogle-perftools-dev libboost-dev pkg-config"

if [ -t 1 ]; then GRN=$'\033[0;32m';YLW=$'\033[0;33m';RED=$'\033[0;31m';BLU=$'\033[0;34m';B=$'\033[1m';R=$'\033[0m'; else GRN=;YLW=;RED=;BLU=;B=;R=; fi
step(){ echo "${BLU}${B}==>${R} ${B}$*${R}"; }
ok(){ echo "  ${GRN}✓${R} $*"; }
warn(){ echo "  ${YLW}!${R} $*"; }
err(){ echo "  ${RED}✗${R} $*" >&2; }
banner(){ echo;echo "${B}──────────────────────────────────────────────${R}";echo "${B} $* ${R}";echo "${B}──────────────────────────────────────────────${R}"; }
have(){ command -v "$1" >/dev/null 2>&1; }

banner "gem5 → $GEM5_DIR  (deps + binary all under /workspace/shared)"
case "$SHARED" in /workspace/shared*) ok "target under /workspace/shared (persists)";; *) warn "SHARED=$SHARED not under /workspace/shared — may be wiped";; esac
mkdir -p "$APTROOT" "$DEBS" "$PYSITE"
df -h "$SHARED" | sed 's/^/  /'
warn "gem5 build is ~10-15 GB; this mount is ~28 GB — watch space. NFS build is slow (45-90+ min)."

# 1. scons + python build helpers into the shared pysite
step "Installing scons into $PYSITE (persistent)"
python3 -m pip install --target="$PYSITE" --upgrade scons six pyparsing >/dev/null 2>&1 \
  && ok "scons + helpers installed" || warn "pip install reported an issue (continuing)"

# 2. download apt build-deps as .debs, extract into the shared aptroot
step "Downloading build-dep .debs (sudo apt, cached in $DEBS)"
if have apt-get; then
  sudo apt-get update -y >/dev/null 2>&1
  # --download-only fetches the packages (+ any not-installed deps) without installing
  sudo apt-get install -y --download-only -o Dir::Cache::archives="$DEBS" $PKGS 2>&1 | tail -2
  # if already-installed packages weren't re-fetched, force a download so they persist
  if [ -z "$(ls "$DEBS"/*.deb 2>/dev/null)" ]; then
    sudo apt-get install -y --reinstall --download-only -o Dir::Cache::archives="$DEBS" $PKGS 2>&1 | tail -2
  fi
  ndeb=$(ls "$DEBS"/*.deb 2>/dev/null | wc -l)
  ok "downloaded $ndeb .deb files"
else
  err "no apt-get — can't fetch deps"; exit 1
fi

step "Extracting .debs into $APTROOT (no system install)"
n=0; for d in "$DEBS"/*.deb; do dpkg -x "$d" "$APTROOT" 2>/dev/null && n=$((n+1)); done
ok "extracted $n packages into the shared aptroot"

# 3. write the env file you source now and on every resume
step "Writing $ENVSH"
cat > "$ENVSH" <<EOF
# source this before building/using gem5 (now and on every resume)
export PATH="$APTROOT/usr/bin:$PYSITE/bin:\$PATH"
export CPATH="$APTROOT/usr/include:$APTROOT/usr/include/$ARCH:$APTROOT/usr/include/python3.12:\${CPATH:-}"
export CPLUS_INCLUDE_PATH="\$CPATH"
export LIBRARY_PATH="$APTROOT/usr/lib/$ARCH:$APTROOT/usr/lib:\${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$APTROOT/usr/lib/$ARCH:$APTROOT/usr/lib:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$APTROOT/usr/lib/$ARCH/pkgconfig:\${PKG_CONFIG_PATH:-}"
export PYTHONPATH="$PYSITE:\${PYTHONPATH:-}"
EOF
ok "wrote $ENVSH"
# shellcheck disable=SC1090
source "$ENVSH"
SCONS="$PYSITE/bin/scons"; [ -x "$SCONS" ] || SCONS="python3 -m SCons"

# 4. GO/NO-GO probe (now with the shared aptroot on the paths)
banner "Dependency check"
NOGO=0
have gcc && ok "gcc $(gcc -dumpversion)" || { err "gcc missing"; NOGO=1; }
PYH=""; for d in "$APTROOT/usr/include/python3.12" /usr/include/python3.12; do [ -f "$d/Python.h" ] && PYH="$d"; done
[ -n "$PYH" ] && ok "Python.h at $PYH" || { err "Python.h not found (python3-dev extract failed)"; NOGO=1; }
$SCONS --version >/dev/null 2>&1 && ok "scons runnable" || { err "scons not runnable"; NOGO=1; }
have protoc && ok "protoc $(protoc --version 2>/dev/null)" || warn "protoc absent (gem5 builds with fewer features)"
[ "$NOGO" = "1" ] && { err "missing a hard dep — fix above before building"; exit 1; }

# 5. clone gem5 into shared
step "Cloning gem5 ($GEM5_BRANCH) → $GEM5_DIR"
if [ -d "$GEM5_DIR/.git" ]; then ok "already cloned"
else git clone --depth 1 --branch "$GEM5_BRANCH" https://github.com/gem5/gem5.git "$GEM5_DIR" \
       || git clone --depth 1 https://github.com/gem5/gem5.git "$GEM5_DIR" || { err "clone failed"; exit 1; }
     ok "cloned"; fi

# 6. build
banner "Building $TARGET/gem5.opt — $JOBS jobs (long; leave it running)"
cd "$GEM5_DIR" || { err "cd failed"; exit 1; }
$SCONS "build/$TARGET/gem5.opt" -j"$JOBS" --ignore-style
if [ -x "$GEM5_DIR/build/$TARGET/gem5.opt" ]; then
  banner "SUCCESS — gem5 built in /workspace/shared"
  echo "Binary: ${B}$GEM5_DIR/build/$TARGET/gem5.opt${R}"
  echo "Verify runtime libs resolve:  ${B}source $ENVSH && ldd $GEM5_DIR/build/$TARGET/gem5.opt | grep -i 'not found' || echo OK${R}"
  echo
  echo "On EVERY resume (deps live in shared, just re-export):"
  echo "  ${B}source $ENVSH${R}"
  echo "Point the pipeline at it:  ${B}gem5.binary: $GEM5_DIR/build/$TARGET/gem5.opt${R}"
else
  err "build did not produce gem5.opt — check scons output above"; exit 1
fi
