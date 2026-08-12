"""Geometry and lineage checks for paired protein-bound switch states.

The designed chain may carry different chain identifiers in the two
RFdiffusion3 outputs, so every comparison is made by residue order after
extracting C-alpha coordinates. These functions use geometry alone, which is
inexpensive enough to evaluate before sequence design.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(path: str):
    plain = str(path)
    if plain.endswith(".cif.gz"):
        with gzip.open(plain, "rt") as handle:
            return MMCIFParser(QUIET=True).get_structure(Path(plain).stem, handle)[0]
    parser = MMCIFParser(QUIET=True) if plain.endswith(".cif") else PDBParser(QUIET=True)
    return parser.get_structure(Path(plain).stem, plain)[0]


def _protein_residues(model, chain_id: str):
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id!r} not found; present chains: {[c.id for c in model]}")
    return [res for res in model[chain_id] if res.id[0] == " " and "CA" in res]


def _ca_coords(model, chain_id: str) -> np.ndarray:
    return np.asarray([res["CA"].coord for res in _protein_residues(model, chain_id)], dtype=float)


def _superpose(mobile: np.ndarray, reference: np.ndarray):
    if len(mobile) != len(reference) or len(mobile) < 3:
        raise ValueError(
            f"Cannot align binder chains with {len(mobile)} and {len(reference)} C-alpha atoms"
        )
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    a = mobile - mobile_center
    b = reference - reference_center
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    fitted = a @ rotation + reference_center
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - reference) ** 2, axis=1))))
    translation = reference_center - mobile_center @ rotation
    return rmsd, rotation, translation


def aligned_binder_rmsd(
    state1_pdb: str, state2_pdb: str, state1_binder_chain: str, state2_binder_chain: str
) -> float:
    state1 = _model(state1_pdb)
    state2 = _model(state2_pdb)
    rmsd, _, _ = _superpose(
        _ca_coords(state2, state2_binder_chain),
        _ca_coords(state1, state1_binder_chain),
    )
    return rmsd


def interface_positions(path: str, binder_chain: str, cutoff: float = 5.0) -> set[int]:
    """Return 1-based binder positions contacting any other protein chain."""
    model = _model(path)
    binder_residues = _protein_residues(model, binder_chain)
    target_atoms = np.asarray([
        atom.coord
        for chain in model
        if chain.id != binder_chain
        for residue in chain
        if residue.id[0] == " "
        for atom in residue
        if (atom.element or "").upper() != "H"
    ], dtype=float)
    if not len(target_atoms):
        return set()

    contacts = set()
    cutoff2 = cutoff * cutoff
    for position, residue in enumerate(binder_residues, start=1):
        coords = np.asarray([
            atom.coord for atom in residue if (atom.element or "").upper() != "H"
        ], dtype=float)
        if len(coords) and np.any(np.sum((coords[:, None, :] - target_atoms[None, :, :]) ** 2, axis=2) <= cutoff2):
            contacts.add(position)
    return contacts


def _target_heavy_atoms(model, binder_chain: str) -> np.ndarray:
    return np.asarray([
        atom.coord
        for chain in model
        if chain.id != binder_chain
        for residue in chain
        if residue.id[0] == " "
        for atom in residue
        if (atom.element or "").upper() != "H"
    ], dtype=float)


def _clash_pairs(a: np.ndarray, b: np.ndarray, cutoff: float) -> int:
    cutoff2 = cutoff * cutoff
    count = 0
    for start in range(0, len(a), 256):
        distances2 = np.sum((a[start:start + 256, None, :] - b[None, :, :]) ** 2, axis=2)
        count += int(np.count_nonzero(distances2 < cutoff2))
    return count


def paired_state_geometry(
    state1_pdb: str,
    state2_pdb: str,
    state1_binder_chain: str,
    state2_binder_chain: str,
    interface_cutoff: float = 5.0,
    clash_cutoff: float = 2.5,
) -> dict[str, float | int]:
    """Measure conformational change, interface reuse, and target incompatibility.

    State 2 is aligned onto state 1 through the designed chain. Target-target
    clashes after that alignment are evidence that both partners cannot occupy
    their modeled poses simultaneously. A clash count is a geometry diagnostic,
    not an affinity or free-energy estimate.
    """
    model1 = _model(state1_pdb)
    model2 = _model(state2_pdb)
    rmsd, rotation, translation = _superpose(
        _ca_coords(model2, state2_binder_chain),
        _ca_coords(model1, state1_binder_chain),
    )
    interface1 = interface_positions(state1_pdb, state1_binder_chain, interface_cutoff)
    interface2 = interface_positions(state2_pdb, state2_binder_chain, interface_cutoff)
    union = interface1 | interface2
    overlap = interface1 & interface2
    jaccard = len(overlap) / len(union) if union else 0.0
    reuse_fraction = len(overlap) / min(len(interface1), len(interface2)) if interface1 and interface2 else 0.0

    target1 = _target_heavy_atoms(model1, state1_binder_chain)
    target2 = _target_heavy_atoms(model2, state2_binder_chain)
    target2_fitted = target2 @ rotation + translation
    clashes = _clash_pairs(target1, target2_fitted, clash_cutoff) if len(target1) and len(target2) else 0
    return {
        "binder_ca_rmsd": rmsd,
        "state1_interface_n": len(interface1),
        "state2_interface_n": len(interface2),
        "interface_overlap_n": len(overlap),
        "interface_jaccard": float(jaccard),
        "interface_reuse_fraction": float(reuse_fraction),
        "target_target_clash_pairs": clashes,
    }
