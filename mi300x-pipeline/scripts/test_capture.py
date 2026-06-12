#!/usr/bin/env python3
"""
test_capture.py — quick on-box test of the amd-smi device collectors (L0/L1).

Proves the SMI path reads real MI300X values, and DUMPS the raw amd-smi JSON so
the parsers can be locked to this box's exact field names. Read-only.

Run on the MI300X box:
    cd mi300x-pipeline
    python3 scripts/test_capture.py | tee testfolder/test-capture.log
Then commit testfolder/test-capture.log and push; paste it back if you like.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from collectors.device.common import amd_smi_json          # noqa: E402
from collectors.device.l0_silicon import L0Silicon         # noqa: E402
from collectors.device.l1_firmware import L1Firmware       # noqa: E402
from core.sampling import now_ns                           # noqa: E402

GPU = 0
run_ctx = {"raw_dir": Path("/tmp/mi300x_testcap")}
run_ctx["raw_dir"].mkdir(parents=True, exist_ok=True)


def section(t):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)


section("RAW amd-smi JSON (for parser tuning — paste this back)")
for sub in ("metric", "static", "partition"):
    print(f"\n--- amd-smi {sub} -g {GPU} --json ---")
    d = amd_smi_json(sub, gpu=GPU)
    if d is None:
        print("  (amd-smi not available or returned no JSON)")
        continue
    # print the top-level structure + a trimmed dump
    if isinstance(d, dict):
        print("  top-level keys:", list(d.keys()))
    txt = json.dumps(d, indent=2)
    print("\n".join("  " + l for l in txt.splitlines()[:80]))
    if len(txt.splitlines()) > 80:
        print("  ... (truncated)")


section("L0 Silicon collector (samples amd-smi 3×)")
l0 = L0Silicon({"gpu_index": GPU}, run_ctx)
ok, why = l0.available()
print(f"available: {ok} ({why})")
if ok:
    l0.setup(); l0.start()
    for _ in range(3):
        l0.sample(now_ns()); time.sleep(0.3)
    res = l0.collect()
    print("scalars:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in res.scalars.items()}, indent=2))
    print("series keys:", list(res.series.keys()))
    print("fidelity:", res.fidelity)
    print("errors:", res.errors)


section("L1 Firmware collector")
l1 = L1Firmware({"gpu_index": GPU}, run_ctx)
ok, why = l1.available()
print(f"available: {ok} ({why})")
if ok:
    l1.setup()
    res = l1.collect()
    print("scalars:", json.dumps(res.scalars, indent=2, default=str))
    print("fidelity:", res.fidelity)

section("DONE")
print("If L0 'series'/scalars or L1 partition came back empty/None, the raw JSON")
print("above tells me which field names to map. Paste this log back.")
