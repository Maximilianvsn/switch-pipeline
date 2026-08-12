"""Compare a refactored run's funnel against the pre-refactor golden run.

The refactor of 2026-07-27 is behaviour-preserving by intent; this script
verifies that claim against a real execution:

    python src/verify_against_golden.py outputs/smoke_golden_baseline \
                                            outputs/smoke_final_verify

`funnel_summary.csv` is the fingerprint. Its `step` column records which stages
ran and in what order; `n_designs` records how many designs survived each filter.
If both match, the refactored pipeline took the same path through the same gates
with the same thresholds.

## Interpretation of a mismatch

  step missing or extra   a stage was dropped, renamed or newly gated out
  n_designs differs       a filter changed behaviour, or the run is stochastic

The second case is why the files are not compared directly. RFdiffusion3
diffusion and AF2 are stochastic, so counts downstream of a threshold may differ
by a design or two between identical runs, whereas counts fixed by configuration,
such as backbones generated and sequences sampled, must match exactly.

Steps are classified accordingly:

  EXACT      the count is determined by configuration; any difference is a
             regression
  TOLERANT   the count lies downstream of a scoring threshold; small drift is
             expected

A differing TOLERANT step is not automatically a pass and warrants inspection.
Pinning `s1_rfd3_holo_seed` and rerunning removes the stochasticity.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

# Counts fixed by configuration, not by any score threshold. These are the real
# regression detectors: holo_batch backbones, nseq sequences per design, and the
# scramble null which is 1:1 with the designs it mirrors.
EXACT_STEPS = {
    "s1_rfd3_holo",
    "s2_rfd3_apo",
    "s3_ligandmpnn",
    "s3_5a_apo_ligandmpnn",
    "s5_dynamicmpnn",
    "s5b_mpnn_msd",
    "s5_5_af2_gate",
    "s7_5_msd_af2_gate",
    "scramble_null",
}


def load_funnel(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "funnel_summary.csv")
    if not os.path.isfile(path):
        sys.exit(f"no funnel_summary.csv in {run_dir}")
    df = pd.read_csv(path)
    # A resumed or re-run step appends a second row; the last one is the truth.
    return df.drop_duplicates(subset="step", keep="last")[["step", "n_designs"]]


def compare(golden_dir: str, candidate_dir: str) -> int:
    g = load_funnel(golden_dir).set_index("step")["n_designs"]
    c = load_funnel(candidate_dir).set_index("step")["n_designs"]

    missing = [s for s in g.index if s not in c.index]
    extra = [s for s in c.index if s not in g.index]
    shared = [s for s in g.index if s in c.index]

    print(f"golden    : {golden_dir}  ({len(g)} steps)")
    print(f"candidate : {candidate_dir}  ({len(c)} steps)\n")

    fail = 0
    rows = []
    for step in shared:
        gv, cv = int(g[step]), int(c[step])
        if gv == cv:
            verdict = "match"
        elif step in EXACT_STEPS:
            verdict = "REGRESSION"
            fail += 1
        else:
            verdict = "drift (tolerant)"
        rows.append((step, gv, cv, verdict))

    width = max(len(r[0]) for r in rows) if rows else 20
    print(f"  {'step':<{width}} {'golden':>8} {'cand':>8}  verdict")
    for step, gv, cv, verdict in rows:
        flag = "" if verdict == "match" else "  <--"
        print(f"  {step:<{width}} {gv:>8} {cv:>8}  {verdict}{flag}")

    if missing:
        print(f"\n  MISSING from candidate ({len(missing)}): {', '.join(missing)}")
        fail += len(missing)
    if extra:
        print(f"\n  EXTRA in candidate ({len(extra)}): {', '.join(extra)}")

    drift = [r for r in rows if r[3].startswith("drift")]
    print()
    if fail:
        print(f"FAIL — {fail} regression(s) on config-determined steps or missing steps.")
    elif drift:
        print(f"PASS with {len(drift)} tolerant drift(s) — inspect the list above; "
              "these sit downstream of scoring thresholds and are stochastic.")
    else:
        print("PASS — every step present and every count identical.")
    return 1 if fail else 0


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    raise SystemExit(compare(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
