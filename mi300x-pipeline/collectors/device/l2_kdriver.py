"""L2 Kernel driver (amdgpu/KFD) — /proc/interrupts diff + ftrace (privileged)."""
from __future__ import annotations
from core.interface import Cadence, Sample
from core.env import ftrace_writable, can_read
from .common import DeviceCollector, read_file


class L2KDriver(DeviceCollector):
    layer_id = 2
    name = "l2_kdriver"

    def available(self):
        if not can_read("/proc/interrupts"):
            return False, "/proc/interrupts unreadable"
        return True, "ok (ftrace optional)" if ftrace_writable() else "ok (no ftrace: faults/queue limited)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._irq0 = None
        self.series = {}

    def _amdgpu_irqs(self):
        txt = read_file("/proc/interrupts") or ""
        total = 0
        for line in txt.splitlines():
            if "amdgpu" in line:
                for tok in line.split()[1:]:
                    if tok.isdigit():
                        total += int(tok)
        return total

    def start(self, workload_pid=None):
        self._irq0 = self._amdgpu_irqs()
        self._t0 = None

    def sample(self, t_ns):
        cur = self._amdgpu_irqs()
        if self._irq0 is not None:
            self.series.setdefault("_irq_cum", []).append(Sample(t_ns, "_irq_cum", cur))

    def collect_real(self, res):
        res.cadence = Cadence.TIMESERIES
        # IRQs/s from first/last cumulative samples
        irqs = self.series.get("_irq_cum", [])
        irqs_s = None
        if len(irqs) >= 2:
            dt = (irqs[-1].t_ns - irqs[0].t_ns) / 1e9
            if dt > 0:
                irqs_s = (irqs[-1].value - irqs[0].value) / dt
        res.scalars.update({
            "kfd_dispatch_us": None,      # requires ftrace KFD tracepoint parsing
            "hw_queue_depth": None,
            "dma_gbs": None,
            "page_faults_s": None,
            "irqs_s": int(irqs_s) if irqs_s is not None else None,
        })
        res.fidelity.update({"irqs_s": "measured" if irqs_s is not None else "null",
                             "kfd_dispatch_us": "null", "hw_queue_depth": "null",
                             "dma_gbs": "null", "page_faults_s": "null"})
        # ftrace artifacts (if enabled) parsed by parsers/ftrace.py
        ft = self.run_ctx.get("ftrace_dat")
        if ft:
            from parsers.ftrace import parse_kfd
            sc, fd = parse_kfd(ft)
            res.scalars.update(sc)
            res.fidelity.update(fd)
        return res
