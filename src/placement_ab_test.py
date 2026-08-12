"""Stage-0 comparison of state-2 binder placement strategies (CPU only, no GPU).

State 2 is seeded by rigidly placing the state-1 binder against target B and then
partial-diffusing it. Two independent choices are baked into that seed:

  * which binder atoms are aimed at the epitope — the state-1 interface patch
    (biases toward surface reuse) or every CA (any face may present);
  * what the placement optimises — the single shortest patch-to-hotspot distance
    (current) or how much of the binder is seated on the epitope.

Production couples them: `interface_seeding.combine_binder_target_same_interface`
uses the patch with the min-distance objective. This script separates them so a
regression in state-2 interface size can be attributed to one or the other.

    A  patch  + min-distance   production baseline
    B  all-CA + min-distance   isolates the surface-reuse bias
    C  patch  + contacts       isolates the placement objective
    D  all-CA + contacts       both changed (reported, not required)

Everything measured here is a property of the *staged* complex before any
diffusion, so an arm that does not increase seated contact cannot help
downstream and can be dropped before spending GPU time.

Usage:
    python placement_ab_test.py \
        --run-dir /path/to/outputs/production_report_20260803 \
        --target-pdb /path/to/pcna_A.pdb \
        --target-hotspots A251,A252,A253,A255 \
        --n 30 --out placement_ab_test.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from docking_utils import search_hotspot_placement, search_hotspot_placement_contacts
from interface_seeding import _load, _parse_hotspots

CONTACT_CUTOFF = 5.0
ARMS = ("A_patch_mindist", "B_allca_mindist", "C_patch_contacts", "D_allca_contacts")


def _heavy(atoms):
    return [atom for atom in atoms if (atom.element or "").upper() != "H"]


def load_state1(holo_path: str, binder_chain: str):
    """Return binder residues/atoms plus the state-1 interface patch and its residue ids."""
    model = _load(holo_path, "state1")[0]
    if binder_chain not in model:
        raise ValueError(f"binder chain {binder_chain} absent from {holo_path}")
    target_coords = np.asarray([
        atom.coord for chain in model if chain.id != binder_chain
        for residue in chain if residue.id[0] == " " for atom in _heavy(residue)
    ])
    residues, patch_coords, patch_ids = [], [], set()
    for residue in model[binder_chain]:
        if residue.id[0] != " ":
            continue
        atoms = _heavy(residue)
        if not atoms:
            continue
        residues.append((residue.id[1], atoms))
        coords = np.asarray([atom.coord for atom in atoms])
        near = cKDTree(target_coords).query(coords, k=1)[0].min()
        if near <= CONTACT_CUTOFF:
            patch_ids.add(residue.id[1])
            patch_coords.extend(coords)
    if len(patch_coords) < 3:
        raise RuntimeError(f"no state-1 interface patch within {CONTACT_CUTOFF} A: {holo_path}")
    return residues, np.asarray(patch_coords), patch_ids


def load_target(target_pdb: str, hotspots: str):
    model = _load(target_pdb, "targetB")[0]
    wanted = _parse_hotspots(hotspots)
    coords = np.asarray([
        atom.coord for chain in model for residue in chain
        if residue.id[0] == " " for atom in _heavy(residue)
    ])
    hotspot_coords = np.asarray([
        atom.coord for chain in model for residue in chain
        if residue.id[0] == " " and (chain.id, residue.id[1]) in wanted
        for atom in _heavy(residue)
    ])
    if not len(hotspot_coords):
        raise RuntimeError(f"no hotspot atoms matched {hotspots} in {target_pdb}")
    return coords, hotspot_coords


def measure(residues, transform, target_coords, hotspot_coords, patch_ids) -> dict:
    """Contact profile of the placed binder against target B, per residue."""
    tree_target, tree_hotspot = cKDTree(target_coords), cKDTree(hotspot_coords)
    contact_residues, hotspot_residues, n_contact_atoms = set(), set(), 0
    min_target, min_hotspot = np.inf, np.inf
    for res_id, atoms in residues:
        placed = transform(np.asarray([atom.coord for atom in atoms]))
        target_distance = tree_target.query(placed, k=1)[0]
        hotspot_distance = tree_hotspot.query(placed, k=1)[0]
        min_target = min(min_target, float(target_distance.min()))
        min_hotspot = min(min_hotspot, float(hotspot_distance.min()))
        n_contact_atoms += int((target_distance <= CONTACT_CUTOFF).sum())
        if target_distance.min() <= CONTACT_CUTOFF:
            contact_residues.add(res_id)
        if hotspot_distance.min() <= CONTACT_CUTOFF:
            hotspot_residues.add(res_id)
    reused = contact_residues & patch_ids
    return {
        "n_contact_residues": len(contact_residues),
        "n_contact_atoms": n_contact_atoms,
        "n_hotspot_contact_residues": len(hotspot_residues),
        "min_target_dist": min_target,
        "min_hotspot_dist": min_hotspot,
        # seed-stage analogue of interface_reuse_fraction / jaccard: how much of
        # the new contact surface was already the state-1 interface.
        "reuse_fraction_seed": len(reused) / len(contact_residues) if contact_residues else 0.0,
        "jaccard_seed": (
            len(reused) / len(contact_residues | patch_ids) if (contact_residues | patch_ids) else 0.0
        ),
    }


def place(arm: str, patch, binder_ca, clash_coords, target_coords, hotspot_coords, seed: int):
    """Run one arm; return (transform, extra-diagnostics).

    `clash_coords` is what the search is forbidden to push into the target.
    Production passes CA atoms only, which lets side chains overlap the target
    (observed down to ~1.2 A); pass binder heavy atoms for a correct check.
    """
    mobile = patch if arm.startswith(("A", "C")) else binder_ca
    if "mindist" in arm:
        min_hs, min_prot, offset, rotation, centroid, center = search_hotspot_placement(
            mobile_coords=mobile, protein_coords=target_coords,
            hotspot_atom_coords=hotspot_coords, n_orientations=300, clash_cutoff=2.5,
            offsets=np.arange(4.0, 24.0, 1.0), hotspot_warn_cutoff=5.0,
            clash_mobile_coords=clash_coords, seed=seed,
        )
        extra = {"placement_score": np.nan, "placement_offset": offset}
    else:
        result = search_hotspot_placement_contacts(
            mobile_coords=mobile, protein_coords=target_coords,
            hotspot_atom_coords=hotspot_coords, n_orientations=3000, clash_cutoff=2.5,
            contact_cutoff=CONTACT_CUTOFF, offsets=np.arange(4.0, 24.0, 1.0),
            clash_mobile_coords=clash_coords, seed=seed,
        )
        rotation, centroid, center = result.rotation, result.mobile_centroid, result.target_center
        extra = {"placement_score": result.score, "placement_offset": result.offset}
    return (lambda coords: rotation.apply(coords - centroid) + center), extra


def collect_backbones(run_dir: str, n: int, column: str) -> list[str]:
    """Distinct state-1 complexes from a finished run, in a stable order."""
    for name in ("s2_5_state2_designability.csv", "s5_5_af2_gate_all.csv"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            frame = pd.read_csv(path)
            if column in frame.columns:
                paths = [p for p in frame[column].dropna().astype(str).unique() if os.path.isfile(p)]
                if paths:
                    return sorted(paths)[:n]
    raise FileNotFoundError(f"no usable state-1 backbone column {column!r} under {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="finished run to source state-1 backbones from")
    parser.add_argument("--target-pdb", required=True, help="target B structure")
    parser.add_argument("--target-hotspots", required=True, help="e.g. A251,A252,A253,A255")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--binder-chain", default="B", help="binder chain in the state-1 complex")
    parser.add_argument("--backbone-col", default="s1_rfd3_holo_location")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--clash-atoms", choices=("heavy", "ca"), default="heavy",
                        help="atoms the search may not push into the target. "
                             "'ca' reproduces production, which permits side-chain clashes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="placement_ab_test.csv")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {list(ARMS)}")

    backbones = collect_backbones(args.run_dir, args.n, args.backbone_col)
    target_coords, hotspot_coords = load_target(args.target_pdb, args.target_hotspots)
    print(f"{len(backbones)} state-1 backbones x {len(arms)} arms "
          f"({len(target_coords)} target atoms, {len(hotspot_coords)} hotspot atoms)")

    rows = []
    for index, holo in enumerate(backbones):
        try:
            residues, patch, patch_ids = load_state1(holo, args.binder_chain)
        except Exception as exc:
            print(f"  [{index}] skip {os.path.basename(holo)}: {type(exc).__name__}: {exc}")
            continue
        binder_ca = np.asarray([
            atom.coord for _, atoms in residues for atom in atoms if atom.get_name() == "CA"
        ])
        binder_heavy = np.asarray([atom.coord for _, atoms in residues for atom in atoms])
        clash_coords = binder_ca if args.clash_atoms == "ca" else binder_heavy
        for arm in arms:
            try:
                transform, extra = place(arm, patch, binder_ca, clash_coords,
                                         target_coords, hotspot_coords, args.seed + index)
                metrics = measure(residues, transform, target_coords, hotspot_coords, patch_ids)
            except Exception as exc:
                print(f"  [{index}] {arm}: FAILED {type(exc).__name__}: {exc}")
                continue
            rows.append({"backbone": os.path.basename(holo), "arm": arm,
                         "clash_atoms": args.clash_atoms,
                         "state1_patch_residues": len(patch_ids), **metrics, **extra})
        if (index + 1) % 5 == 0:
            print(f"  ...{index + 1}/{len(backbones)} backbones")

    if not rows:
        raise SystemExit("no placements succeeded")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)

    report = ["n_contact_residues", "n_hotspot_contact_residues", "n_contact_atoms",
              "reuse_fraction_seed", "jaccard_seed", "min_hotspot_dist", "min_target_dist"]
    print(f"\nwrote {args.out}  ({len(frame)} rows)\n")
    print("=== median per arm ===")
    print(frame.groupby("arm")[report].median().round(3).to_string())
    print("\n=== paired vs baseline A (same backbone) ===")
    wide = frame.pivot_table(index="backbone", columns="arm", values="n_contact_residues")
    if "A_patch_mindist" in wide.columns:
        for arm in [c for c in wide.columns if c != "A_patch_mindist"]:
            both = wide[["A_patch_mindist", arm]].dropna()
            if both.empty:
                continue
            delta = both[arm] - both["A_patch_mindist"]
            print(f"  {arm:18s} contact residues: median delta {delta.median():+.1f}  "
                  f"better on {int((delta > 0).sum())}/{len(delta)} backbones")


if __name__ == "__main__":
    main()
