#!/usr/bin/env bash
# probe_env.sh — READ-ONLY diagnostic for the isolated MI300X box.
# Collects GPU / ROCm / amd-smi / rocprofiler / python / torch / network /
# resource info into a report file you can paste back. Changes nothing.
#
# Usage:   bash scripts/probe_env.sh
# Output:  ./env_report_<host>_<date>.txt  (also printed to screen)
#
# Safe by design: never installs, never writes outside the report file, and keeps
# going past any failing command.
set +e

HOST="$(hostname 2>/dev/null || echo unknown)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || echo nodate)"
REPORT="./env_report_${HOST}_${STAMP}.txt"

# log to both screen and file
exec > >(tee "$REPORT") 2>&1

hr() { printf '\n========== %s ==========\n' "$*"; }
cap() { # cap "Label" cmd args...
  local label="$1"; shift
  printf '\n--- %s ---\n$ %s\n' "$label" "$*"
  if command -v "${1%% *}" >/dev/null 2>&1 || command -v "$1" >/dev/null 2>&1; then
    timeout 25 "$@" 2>&1 | sed 's/^/  /' || echo "  (command failed or timed out)"
  else
    echo "  (not installed: $1)"
  fi
}
net() { # net "Label" url   — reachability test
  local label="$1" url="$2"
  printf '\n--- net: %s ---\n' "$label"
  if command -v curl >/dev/null 2>&1; then
    timeout 10 curl -sS -o /dev/null -w "  curl %{http_code} in %{time_total}s -> $url\n" "$url" 2>&1 \
      || echo "  curl FAILED -> $url"
  elif command -v wget >/dev/null 2>&1; then
    timeout 10 wget -q --spider "$url" && echo "  wget OK -> $url" || echo "  wget FAILED -> $url"
  else
    echo "  (no curl/wget)"
  fi
}

hr "MI300X ENV REPORT"
echo "host: $HOST   utc: $STAMP   report: $REPORT"

hr "IDENTITY / OS"
cap "whoami" whoami
echo "  uid=$(id -u) (root=$([ "$(id -u)" -eq 0 ] && echo yes || echo no))"
cap "os-release" cat /etc/os-release
cap "kernel" uname -a
echo "  WSL: $(grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null && echo yes || echo no)"
echo "  container hints: $(ls -d /.dockerenv 2>/dev/null; grep -qa docker /proc/1/cgroup 2>/dev/null && echo cgroup-docker)"

hr "GPU (amd-smi preferred)"
cap "amd-smi version" amd-smi version
cap "amd-smi list" amd-smi list
cap "amd-smi static -g 0" amd-smi static -g 0
cap "amd-smi monitor (1 shot)" amd-smi monitor -g 0 -i 1
cap "amd-smi partition" amd-smi partition
cap "rocm-smi (fallback)" rocm-smi --showproductname --showpartition

hr "ROCm STACK"
cap "rocminfo (head)" bash -c "rocminfo 2>/dev/null | sed -n '1,60p'"
echo "  gfx target: $(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')"
cap "rocm dirs" bash -c "ls -d /opt/rocm* 2>/dev/null"
cap "rocm version file" bash -c "cat /opt/rocm/.info/version* 2>/dev/null"
cap "hipcc" hipcc --version

hr "PROFILING TOOLS"
cap "rocprofv3 which/ver" bash -c "which rocprofv3 && rocprofv3 --version 2>&1 | head -3"
cap "rocprof (legacy)" bash -c "which rocprof && rocprof --version 2>&1 | head -3"
echo "  ftrace writable: $([ -w /sys/kernel/tracing ] && echo yes || echo no)"
echo "  /proc/interrupts readable: $([ -r /proc/interrupts ] && echo yes || echo no)"
echo "  perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"

hr "PYTHON / TORCH"
cap "python3" python3 --version
cap "pip" bash -c "python3 -m pip --version"
cap "key python deps" python3 - <<'PY'
for m in ("torch","yaml","numpy","sklearn"):
    try:
        mod=__import__(m); print(f"  {m}: {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m}: MISSING ({e.__class__.__name__})")
try:
    import torch
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.version.hip: {getattr(torch.version,'hip',None)}")
    if torch.cuda.is_available():
        print(f"  device 0: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  torch GPU check skipped: {e}")
PY

hr "NETWORK (you said: wget/pip/git allowed)"
net "github (git)" "https://github.com"
net "pypi (pip)" "https://pypi.org/simple/"
net "pytorch wheels" "https://download.pytorch.org/whl/rocm6.4/"
net "rocm repo" "https://repo.radeon.com"
cap "git reachability" bash -c "timeout 12 git ls-remote https://github.com/gem5/gem5.git HEAD 2>&1 | head -1"

hr "RESOURCES"
cap "cpu count" nproc
cap "memory" free -h
cap "disk" df -h
cap "cwd" pwd
echo "  writable cwd: $([ -w . ] && echo yes || echo no)"

hr "DONE"
echo "Report written to: $REPORT"
echo "Paste this file back (or: cat $REPORT) and I'll tune the collectors + setup."
