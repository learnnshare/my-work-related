#!/usr/bin/env bash
# lib.sh — shared helpers for the setup scripts. Source this; don't run it.
set -euo pipefail

# colors (disabled if not a tty)
if [ -t 1 ]; then
  RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[0;33m'; BLU=$'\033[0;34m'; BLD=$'\033[1m'; RST=$'\033[0m'
else
  RED=; GRN=; YLW=; BLU=; BLD=; RST=
fi

step() { echo "${BLU}${BLD}==>${RST} ${BLD}$*${RST}"; }
ok()   { echo "  ${GRN}✓${RST} $*"; }
warn() { echo "  ${YLW}!${RST} $*"; }
err()  { echo "  ${RED}✗${RST} $*" >&2; }
die()  { err "$*"; exit 1; }

have()        { command -v "$1" >/dev/null 2>&1; }
is_wsl()      { grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; }
is_root()     { [ "$(id -u)" -eq 0 ]; }
SUDO()        { if is_root; then "$@"; else sudo "$@"; fi; }

# pretty section header
banner() {
  echo
  echo "${BLD}────────────────────────────────────────────────────────${RST}"
  echo "${BLD} $* ${RST}"
  echo "${BLD}────────────────────────────────────────────────────────${RST}"
}

# repo root = parent of scripts/
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
