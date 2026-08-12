"""Seed target-B binding through the binder surface used in state 1."""
from __future__ import annotations

import gzip

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser

from docking_utils import search_hotspot_placement


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
    min_hs, min_prot, offset, rotation, mobile_center, target_center = search_hotspot_placement(
        mobile_coords=preferred,
        protein_coords=target_coords,
        hotspot_atom_coords=hotspot_coords,
        n_orientations=300,
        clash_cutoff=2.5,
        offsets=np.arange(4.0, 24.0, 1.0),
        hotspot_warn_cutoff=5.0,
        clash_mobile_coords=binder_heavy,
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
        f"  Same-interface seed -> target B: patch-hotspot {min_hs:.1f}A, "
        f"binder-target {min_prot:.1f}A (offset {offset:.0f}A)"
    )
    return out_pdb
