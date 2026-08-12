"""Find designs matching arbitrary metric criteria in a completed run.

    python src/query_designs.py outputs/<run> \
        --min af2_switch_plddt=0.6 --max af2_apo_i_pae=0.40 --max af2_holo_i_pae=0.34 \
        --sort af2_switch_plddt --top 10 [--out hits.csv] [--msd]

--min COL=VAL keeps rows with COL >= VAL; --max COL=VAL keeps COL <= VAL (both
repeatable). Defaults: rank by af2_switch_plddt, no filters (shows the top).
Prints the interesting columns; --list-cols shows everything available.
"""
import argparse
import os
import sys
import pandas as pd

KEY_COLS = ["poses_description", "af2_tier", "af2_switch_plddt",
            "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
            "af2_holo_i_ptm", "af2_apo_i_ptm", "binder_ca_rmsd",
            "s6a_boltz_holo_iptm", "s6b_boltz_apo_iptm"]


def _pairs(items):
    out = []
    for it in items or []:
        if "=" not in it:
            sys.exit(f"bad filter '{it}', expected COL=VALUE")
        c, v = it.split("=", 1)
        out.append((c.strip(), float(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run output dir (contains final_all_ranked.csv)")
    ap.add_argument("--min", action="append", default=[], help="COL=VAL, keep COL >= VAL")
    ap.add_argument("--max", action="append", default=[], help="COL=VAL, keep COL <= VAL")
    ap.add_argument("--sort", default="af2_switch_plddt")
    ap.add_argument("--asc", action="store_true", help="sort ascending (default descending)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None, help="also write matches to this CSV")
    ap.add_argument("--msd", action="store_true", help="query the ProteinMPNN-MSD arm instead")
    ap.add_argument("--list-cols", action="store_true")
    a = ap.parse_args()

    fname = "msd_final_all_ranked.csv" if a.msd else "final_all_ranked.csv"
    path = os.path.join(a.run, fname)
    if not os.path.isfile(path):
        sys.exit(f"no {fname} in {a.run}")
    df = pd.read_csv(path)
    if a.list_cols:
        print("\n".join(sorted(df.columns))); return

    mask = pd.Series(True, index=df.index)
    for c, v in _pairs(a.min):
        mask &= pd.to_numeric(df[c], errors="coerce") >= v
    for c, v in _pairs(a.max):
        mask &= pd.to_numeric(df[c], errors="coerce") <= v
    hits = df[mask].copy()
    if a.sort in hits.columns:
        hits = hits.sort_values(a.sort, ascending=a.asc)
    hits = hits.head(a.top)

    show = [c for c in KEY_COLS if c in hits.columns]
    filt = "  ".join([f"{c}>={v}" for c, v in _pairs(a.min)] + [f"{c}<={v}" for c, v in _pairs(a.max)])
    print(f"\n{int(mask.sum())} / {len(df)} designs match{('  [' + filt + ']') if filt else ''}"
          f"   (showing top {len(hits)} by {a.sort})\n")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:.3f}"):
        print(hits[show].to_string(index=False))
    if a.out:
        hits.to_csv(a.out, index=False)
        print(f"\nwrote {len(hits)} rows -> {a.out}")


if __name__ == "__main__":
    main()
