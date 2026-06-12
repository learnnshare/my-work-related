"""
env.py — preflight checks: tool availability and privilege.

Used by the orchestrator so a missing tool degrades gracefully (collector
skipped, recorded in the manifest) instead of failing the whole run.
"""
from __future__ import annotations
import os
import shutil


def has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def can_read(path: str) -> bool:
    return os.access(path, os.R_OK)


def can_write(path: str) -> bool:
    return os.access(path, os.W_OK)


def in_group(*names) -> bool:
    try:
        import grp
        gids = set(os.getgroups())
        for n in names:
            try:
                if grp.getgrnam(n).gr_gid in gids:
                    return True
            except KeyError:
                continue
    except Exception:
        pass
    return False


def ftrace_writable() -> bool:
    return can_write("/sys/kernel/tracing") or can_write("/sys/kernel/debug/tracing")


def summarize() -> dict:
    """A snapshot used in preflight + manifest."""
    return {
        "root": is_root(),
        "groups_render_video": in_group("render", "video"),
        "amd_smi": has_binary("amd-smi"),
        "rocm_smi": has_binary("rocm-smi"),
        "rocprofv3": has_binary("rocprofv3") or has_binary("rocprof"),
        "ftrace_writable": ftrace_writable(),
        "proc_interrupts": can_read("/proc/interrupts"),
        "perf_event_paranoid": _read_int("/proc/sys/kernel/perf_event_paranoid"),
    }


def _read_int(path):
    try:
        return int(open(path).read().strip())
    except Exception:
        return None
