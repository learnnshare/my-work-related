"""rocm_smi_json.py — helpers to pull numeric fields out of `rocm-smi --json`."""
from __future__ import annotations


def first_card(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) and (k.lower().startswith("card") or "gpu" in k.lower()):
                return v
        # some versions nest under the first key
        return next(iter(d.values())) if d else {}
    return {}


def num(card, *substrings):
    for sub in substrings:
        for k, v in card.items():
            if sub.lower() in k.lower():
                try:
                    return float(str(v).split()[0])
                except (ValueError, IndexError):
                    continue
    return None
