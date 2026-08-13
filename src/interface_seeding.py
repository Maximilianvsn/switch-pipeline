"""Seed target-B binding through the binder surface used in state 1.

Placement objective (`objective=`):

* ``"contacts"`` (default) maximises how much of the state-1 patch is seated on
  the target-B epitope.
* ``"min_distance"`` is the original behaviour: minimise the single shortest
  patch-to-hotspot distance, stopping at the first offset within 5 A.

Measured over 30 backbones against PCNA with correct heavy-atom sterics, the
contact objective raised hotspot-contact residues from a median of 1 to 7 and
won on 30/30 backbones (Wilcoxon p = 1.6e-6). The min-distance objective
satisfies its own criterion with a single atom near a single hotspot while the
rest of the binder faces solvent, which is how state-2 seeds ended up with about
half the interface residues of state 1.

Aiming the state-1 patch rather than every CA is kept: dropping it changed
nothing once the objective was fixed (arms C and D were indistinguishable) and
it is what biases the seed toward shared-surface reuse.
"""
from __future__ import annotations

import gzip

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser

from docking_utils import search_hotspot_placement, search_hotspot_placement_contacts

# Coarse rotations tried per offset before local refinement. Tuned for cost:
# placement runs once per designable state-1 backbone (1014 of them in the last
# production run), serially in the orchestrator, so this sits on the critical path.
# Quality saturates by ~1200 orientations (22 vs 24 seated atoms); 300 keeps 92%
# of that for 2.8 s/backbone (~47 min at production scale) instead of 26 s (7.3 h).
PLACEMENT_ORIENTATIONS = 300


def _parse_hotspots(hotspots: str):
    return {(token.strip()[0], int(token.strip()[1:])) for token in hotspots.split(",") if token.strip()}


def _load(path: str, name: str):
    if path.endswith(".cif.gz"):
        with gzip.open(path, "rt") as handle:
            return MMCIFParser(QUIET=True).get_structure(name, handle)
    parser = MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)
    return parser.get_structure(name, path)


def _state1_interface_coords(holo_path: str, binder_chain: str, cutoff: float = 5.0):
    model = _load(holo_path, "state1")[0]
    if binder_chain not in model:
        raise ValueError(f"Binder chain {binder_chain} absent from {holo_path}")
    target = np.asarray([
        atom.coord for chain in model if chain.id != binder_chain
        for residue in chain if residue.id[0] == " " for atom in residue
        if (atom.element or "").upper() != "H"
    ])
    selected = []
    cutoff2 = cutoff * cutoff
    for residue in model[binder_chain]:
        if residue.id[0] != " ":
            continue
        atoms = [atom for atom in residue if (atom.element or "").upper() != "H"]
        coords = np.asarray([atom.coord for atom in atoms])
        if len(coords) and np.any(np.sum((coords[:, None, :] - target[None, :, :]) ** 2, axis=2) <= cutoff2):
            selected.extend(atom.coord for atom in atoms)
    if len(selected) < 3:
        raise RuntimeError(f"State-1 binder has no usable interface patch within {cutoff} A: {holo_path}")
    return np.asarray(selected)


def combine_binder_target_same_interface(
    holo_path: str,
    binder_pdb: str,
    target_pdb: str,
    out_pdb: str,
    target_hotspots: str,
    binder_chain: str,
    interface_cutoff: float = 5.0,
    objective: str = "contacts",
    n_orientations: int | None = None,
):
    """Rigidly aim state 1's binder interface patch at target-B hotspots."""
    binder = _load(binder_pdb, "binder")
    target = _load(target_pdb, "target")
    preferred = _state1_interface_coords(holo_path, binder_chain, interface_cutoff)
    hotspot_set = _parse_hotspots(target_hotspots)
    hotspot_coords = np.asarray([
        atom.coord for chain in target[0] for residue in chain
        if residue.id[0] == " " and (chain.id, residue.id[1]) in hotspot_set
        for atom in residue
    ])
    if not len(hotspot_coords):
        raise RuntimeError(f"No target-B hotspot atoms found for {target_hotspots}")
    target_coords = np.asarray([
        atom.coord for chain in target[0] for residue in chain if residue.id[0] == " "
        for atom in residue if (atom.element or "").upper() != "H"
    ])
    binder_atoms = [atom for chain in binder[0] for residue in chain for atom in residue]
    binder_coords = np.asarray([atom.coord for atom in binder_atoms])
    # Clash-check every heavy atom, not just the CA trace. Checking CAs alone
    # permitted side-chain overlaps with the target down to ~1.2 A, so RFD3 was
    # being handed sterically impossible seeds to resolve.
    binder_heavy = np.asarray([
        atom.coord for atom in binder_atoms if (atom.element or "").upper() != "H"
    ])
    seated = None
    if objective == "contacts":
        result = search_hotspot_placement_contacts(
            mobile_coords=preferred,
            protein_coords=target_coords,
            hotspot_atom_coords=hotspot_coords,
            n_orientations=n_orientations or PLACEMENT_ORIENTATIONS,
            clash_cutoff=2.5,
            contact_cutoff=interface_cutoff,
            offsets=np.arange(4.0, 24.0, 1.0),
            clash_mobile_coords=binder_heavy,
        )
        min_hs, min_prot, offset = result.min_hotspot_dist, result.min_protein_dist, result.offset
        rotation, mobile_center, target_center = (
            result.rotation, result.mobile_centroid, result.target_center
        )
        seated = result.n_hotspot_contacts
    elif objective == "min_distance":
        min_hs, min_prot, offset, rotation, mobile_center, target_center = search_hotspot_placement(
            mobile_coords=preferred,
            protein_coords=target_coords,
            hotspot_atom_coords=hotspot_coords,
            n_orientations=n_orientations or 300,
            clash_cutoff=2.5,
            offsets=np.arange(4.0, 24.0, 1.0),
            hotspot_warn_cutoff=5.0,
            clash_mobile_coords=binder_heavy,
        )
    else:
        raise ValueError(
            f"objective must be 'contacts' or 'min_distance', got {objective!r}"
        )
    for atom, coord in zip(binder_atoms, binder_coords):
        atom.set_coord(rotation.apply(coord - mobile_center) + target_center)

    combined = binder.copy()
    model = combined[0]
    target_chain = list(target[0].get_chains())[0].copy()
    occupied = {chain.id for chain in model}
    if target_chain.id in occupied:
        target_chain.id = "A" if "A" not in occupied else "C"
    model.add(target_chain)
    writer = PDBIO()
    writer.set_structure(combined)
    writer.save(out_pdb)
    print(
        f"  Same-interface seed -> target B [{objective}]: patch-hotspot {min_hs:.1f}A, "
        f"binder-target {min_prot:.1f}A (offset {offset:.0f}A)"
        + (f", {seated} patch atoms on the epitope" if seated is not None else "")
    )
    return out_pdb
