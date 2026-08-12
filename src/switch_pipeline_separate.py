"""Independent-backbone variant: generate each state against its own target.

The canonical pipeline derives state 2 by partial diffusion of the state-1
binder, so the two states share a lineage: state 2 is a perturbation of state 1.
That coupling is what makes a shared sequence plausible and is what
`apo_target.partial_t` controls.

This variant removes the coupling:

    RFD3(binder vs target 1)  ->  state-1 backbones   \\
                                                       >-- paired -> DynamicMPNN
    RFD3(binder vs target 2)  ->  state-2 backbones   /

The two backbones are generated independently and share only their length, so a
single sequence must be compatible with two unrelated folds.

## Purpose

This is the control for the partial-diffusion design. A comparable success rate
would indicate that the lineage coupling is not responsible for the result;
failure is direct evidence that the coupling matters, which cannot be established
within the canonical pipeline, where the coupling is never absent.

## Scope of the metrics

The canonical geometry gate enforces the switch premise: the two states must
reuse a common binder surface while rendering the targets mutually exclusive.
Those metrics (`binder_ca_rmsd`, `interface_jaccard`,
`target_target_clash_pairs`) are computed by superposing two versions of one
backbone. Here the backbones are unrelated, so no common frame exists and the
quantities are not interpretable; they are reported as diagnostics and never
applied as gates.

This variant accordingly designs one sequence tolerated by two distinct folds,
rather than a conformational switch, and its output requires additional evidence
before being described as one.

## Shared components

Everything after backbone generation is shared: designability gating,
DynamicMPNN, the AF2 two-state gate, the paired nulls, Boltz-2 scoring, ranking
and reporting all come from the same modules, making the two pipelines directly
comparable.

    python src/switch_pipeline_separate.py --config <cfg> --run-name <name> [--smoke]
"""
from __future__ import annotations

import os
import re

import pandas as pd
from protflow.poses import Poses
from protflow.tools.rfdiffusion3 import RFD3Params

import pipeline_context
import switch_pipeline


_TARGET_SEG = re.compile(r"^([A-Za-z])(\d+)-(\d+)$")


def _binder_length(contig: str) -> int:
    """Residue count of the de novo binder segment.

    Keyed on segment SHAPE, not position: the binder is the one bare integer
    field, as opposed to "<chain><lo>-<hi>" target spans or the "/0" chain break.
    Position would not work, because the two states deliberately order their
    contigs differently -- holo puts the binder last ("A19-115,/0,80"), apo puts
    it first ("80,/0,A1-106,..."). See build of `apo_contig` for why.
    """
    lens = [f.strip() for f in str(contig).split(",") if f.strip().isdigit()]
    if len(lens) != 1:
        raise ValueError(
            f"expected exactly one de novo binder segment in contig {contig!r}, found {lens}")
    return int(lens[0])


def target_contig_from_pdb(pdb: str, chain: str = "A") -> str:
    """Contig segments covering the residues actually present in a target chain.

    Crystal structures have gaps. RFD3 validates every indexed contig residue
    against the atom array and aborts if one is missing, so a naive "A1-255" on
    a chain that skips 107-108 and 187-190 fails at input parsing -- which is
    exactly how the first run of this variant died. Emitting one segment per
    contiguous run ("A1-106,A109-186,A191-255") is what the canonical pipeline
    does in write_apo_inputs.py, and it is target-agnostic.
    """
    present = sorted({
        int(line[22:26]) for line in open(pdb)
        if line.startswith("ATOM") and line[21] == chain
    })
    if not present:
        raise ValueError(f"no chain {chain} residues in {pdb}")
    segments, start, prev = [], present[0], present[0]
    for rid in present[1:]:
        if rid != prev + 1:
            segments.append(f"{chain}{start}-{prev}")
            start = rid
        prev = rid
    segments.append(f"{chain}{start}-{prev}")
    return ",".join(segments)


def check_contig_against_pdb(contig: str, pdb: str) -> None:
    """Fail fast if a configured contig names residues the structure lacks."""
    present = {
        (line[21], int(line[22:26])) for line in open(pdb)
        if line.startswith("ATOM")
    }
    for seg in str(contig).split(","):
        m = _TARGET_SEG.match(seg.strip())
        if not m:  # a bare binder length or the "/0" chain break, not a target span
            continue
        chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        missing = [r for r in range(lo, hi + 1) if (chain, r) not in present]
        if missing:
            raise ValueError(
                f"contig {contig!r} names residues absent from {os.path.basename(pdb)}: "
                f"{missing[:6]}{'...' if len(missing) > 6 else ''}. "
                f"Chain {chain} actually covers {target_contig_from_pdb(pdb, chain)}")


def generate_state(ctx, target: dict, contig: str, prefix: str, n_batches: int):
    """One independent RFD3 run: binder against a single target."""
    poses = Poses(poses=[target["pdb"]], work_dir=ctx.OUTPUTS, jobstarter=ctx.gpu_jst)
    params = RFD3Params(poses=poses)
    params.set_input_specs(contig=contig, select_hotspots=target.get("hotspots", ""))
    poses = ctx.rfd3.run(
        poses=poses, prefix=prefix, params=params,
        n_batches=n_batches, diffusion_batch_size=ctx.DIFFUSION_BATCH_SIZE,
        options="skip_existing=True",
    )
    ctx.funnel.log(prefix, len(poses.df),
                   f"independent backbones vs {os.path.basename(target['pdb'])}")
    return poses


def pair_by_rank(state1: pd.DataFrame, state2: pd.DataFrame,
                 s1_loc: str, s2_loc: str) -> pd.DataFrame:
    """Pair the i-th state-1 backbone with the i-th state-2 backbone.

    Rank pairing rather than the N x M Cartesian product: the product explodes
    (400 x 400) and, because the backbones are independent, carries no more
    information per pair than a 1:1 matching does. Truncates to the shorter list,
    so the pair count is min(n1, n2) and each backbone is used at most once.
    """
    n = min(len(state1), len(state2))
    if n == 0:
        raise RuntimeError("no backbones survived on one of the two states")
    a = state1.head(n).reset_index(drop=True)
    b = state2.head(n).reset_index(drop=True)
    paired = a.copy()
    paired["state2_pdb"] = b[s2_loc].to_numpy()
    paired["state2_description"] = b["poses_description"].to_numpy()
    paired["pair_index"] = range(n)
    return paired


def main() -> None:
    ctx = pipeline_context.build_context()
    cfg, funnel = ctx.cfg, ctx.funnel
    holo, apo = ctx.holo, ctx.apo

    holo_contig = holo["contig"]
    binder_len = _binder_length(holo_contig)
    check_contig_against_pdb(holo_contig, holo["pdb"])
    # State 2 needs its own contig. Only the binder LENGTH is inherited from the
    # holo contig -- a shared sequence is not threadable otherwise -- while the
    # target segments are read off the apo structure, so residue gaps cannot
    # produce an unsatisfiable contig.
    # BINDER SEGMENT FIRST, and this is load-bearing. RFD3 reletters output chains
    # sequentially in contig order, and the shared tail hardcodes the resulting
    # apo layout in two places: switch_pipeline.py sets apo_binder_chain = "A",
    # and af2_gate.build_state_requests passes "B" as the apo TARGET chain. So an
    # apo backbone must come out as binder=A, target=B.
    #
    # Putting the target first (the natural reading order, and what the first
    # version of this file did) inverts that: DynamicMPNN was handed the
    # 80-residue state-1 binder against 249-residue PCNA and died on
    # "Explicit alignment indices are required when the selected chains do not
    # have the same length". Ordering it here is what write_apo_inputs.py already
    # does for the canonical pipeline, and keeps the shared tail untouched.
    apo_contig = apo.get("contig")
    if apo_contig:
        check_contig_against_pdb(apo_contig, apo["pdb"])
    else:
        apo_contig = f"{binder_len},/0,{target_contig_from_pdb(apo['pdb'])}"
    if _binder_length(apo_contig) != binder_len:
        raise ValueError(
            f"binder length must match across states: holo {binder_len} "
            f"vs apo {_binder_length(apo_contig)} (contigs {holo_contig!r}, {apo_contig!r})")

    print("=" * 60)
    print("SEPARATE-BACKBONE VARIANT: independent generation per target")
    print(f"  state 1 contig: {holo_contig}")
    print(f"  state 2 contig: {apo_contig}   (binder length {binder_len}, matched)")
    print("        state 2 is binder-FIRST so RFD3 emits binder=A, target=B,")
    print("        which is the layout the shared tail assumes.")
    print("  NOTE: the two backbones share no lineage, so binder_ca_rmsd /")
    print("        interface_jaccard are neither computed nor gated here --")
    print("        s2_state_pair_geometry.csv is never written, and the run")
    print("        audit therefore always reports evidence_ready = False.")
    print("=" * 60)

    n_batches = ctx.HOLO_N_BATCHES
    s1 = generate_state(ctx, holo, holo_contig, "s1_rfd3_holo", n_batches)
    s2 = generate_state(ctx, apo, apo_contig, "s2_rfd3_apo_independent", n_batches)

    # No per-state designability pre-filter in this variant. In the canonical
    # pipeline that filter exists to avoid paying for state-2 partial diffusion on
    # undesignable state-1 backbones; here the two states are generated
    # independently and in parallel, so there is nothing downstream of state 1 to
    # protect. The AF2 two-state gate does the real filtering either way.
    paired = pair_by_rank(s1.df, s2.df, "s1_rfd3_holo_location",
                          "s2_rfd3_apo_independent_location")
    funnel.log("paired_independent_states", len(paired),
               f"rank-paired 1:1 (min of {len(s1.df)} state-1, {len(s2.df)} state-2)")
    paired.to_csv(os.path.join(ctx.OUTPUTS, "paired_independent_states.csv"), index=False)

    poses = Poses(poses=list(paired["s1_rfd3_holo_location"]),
                  work_dir=ctx.OUTPUTS, jobstarter=ctx.gpu_jst)
    for col in paired.columns:
        if col not in poses.df.columns:
            poses.df[col] = paired[col].to_numpy()

    # The shared tail needs exactly three upstream columns (verified by scanning
    # its dataframe accesses): the state-1 description and location, and the
    # state-2 PDB path. Assert rather than discover a KeyError mid-run.
    required = ["s1_rfd3_holo_description", "s1_rfd3_holo_location", "state2_pdb"]
    missing = [c for c in required if c not in poses.df.columns]
    if missing:
        raise KeyError(f"paired frame missing columns the shared tail needs: {missing}")

    switch_pipeline.run_shared_tail(ctx, poses)


if __name__ == "__main__":
    main()
