"""Per-stage design-count log: the funnel record of a run.

`funnel_summary.csv` is written incrementally, every `log()` call rewriting it,
so that a run in progress can be monitored and a completed run compared against a
reference. The `step` and `n_designs` columns constitute the behavioural contract
of the pipeline: any change that alters them alters the results.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd


class FunnelTracker:
    """Logs per-step design counts to a CSV for post-run analysis."""
    def __init__(self, out_dir: str):
        self.rows = []
        self.out_path = os.path.join(out_dir, "funnel_summary.csv")

    def log(self, step: str, n_designs: int, note: str = ""):
        self.rows.append({
            "step": step,
            "n_designs": n_designs,
            "timestamp": datetime.now().isoformat(),
            "note": note,
        })
        print(f"  [{step}] {n_designs} designs  {note}")
        self.save()

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.out_path, index=False)

