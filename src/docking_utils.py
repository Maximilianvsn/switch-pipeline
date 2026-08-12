"""
Shared geometry helper: place a disconnected mobile component, a small-molecule
ligand or an entire binder chain, in contact with a target protein's hotspot
patch, as a seed for RFdiffusion3.

The placement performs a rotation search with an outward offset. It is used on
the apo side to seat the binder against the target before partial diffusion, in
place of raw-coordinate concatenation, which left the binder 9-12 A from the
target with no interface contacts.

This is a geometric placement rather than a docking calculation: no energetics,
hydrogen bonding or side-chain packing are modelled. It produces a clash-free
rigid pose whose mobile atoms approach the hotspot atoms as closely as possible,
so that the downstream generative model starts at the intended interface.
"""
from collections import namedtuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def search_hotspot_placement(
    mobile_coords: np.ndarray,
    protein_coords: np.ndarray,
    hotspot_atom_coords: np.ndarray,
    n_orientations: int = 2000,
    clash_cutoff: float = 2.5,
    offsets: np.ndarray | None = None,
    hotspot_warn_cutoff: float = 8.0,
    seed: int = 42,
    clash_mobile_coords: np.ndarray | None = None,
):
    """Search rigid placements of a mobile component near a hotspot patch.

    Parameters
    ----------
    mobile_coords : (N,3) coordinates used to score placements (the ligand's
        heavy atoms, or the binder's CA atoms — a small point set for speed).
    protein_coords : (M,3) target atoms used for clash detection.
    hotspot_atom_coords : (K,3) atoms of the hotspot residues; placements are
        scored by how close the mobile component gets to these.
    n_orientations : number of random rigid rotations tried per offset.
    clash_cutoff : reject any placement whose nearest mobile-to-protein atom
        distance is below this (Angstroms).
    offsets : outward offsets (A) of the mobile centroid from the hotspot
        centroid to try, in increasing order. For a small ligand ~3-18 A; for a
        globular binder the centroid must sit ~one binder-radius out, so pass a
        larger range (e.g. np.arange(10, 34, 2)).
    hotspot_warn_cutoff : once a placement gets the mobile component within this
        distance of the hotspots, stop searching further (outer) offsets.
    seed : RNG seed for the rotation set (deterministic placement).

    Returns
    -------
    (min_hotspot_dist, min_protein_dist, offset, rot, mobile_centroid,
     target_center) — apply the winning rigid transform to ANY atom set X of the
    mobile component with:  rot.apply(X - mobile_centroid) + target_center.

    Raises
    ------
    RuntimeError if no clash-free placement is found across all offsets ×
    rotations (patch too crowded, or the mobile component too large).
    """
    if offsets is None:
        offsets = np.arange(3.0, 18.0, 1.0)

    mobile_centroid = mobile_coords.mean(axis=0)
    mobile_centered = mobile_coords - mobile_centroid
    clash_centered = (
        np.asarray(clash_mobile_coords) - mobile_centroid
        if clash_mobile_coords is not None else mobile_centered
    )

    hotspot_centroid = hotspot_atom_coords.mean(axis=0)
    # local surface-normal estimate: from the target atoms *near* the hotspot
    # patch out to the hotspot centroid (using the whole-protein centroid would
    # be skewed by the protein's overall shape).
    local_mask = np.linalg.norm(protein_coords - hotspot_centroid, axis=1) < 15.0
    local_centroid = protein_coords[local_mask].mean(axis=0) if local_mask.any() else protein_coords.mean(axis=0)
    outward = hotspot_centroid - local_centroid
    norm = np.linalg.norm(outward)
    outward = outward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    # KD-trees give the same nearest-neighbour distances as the dense pairwise
    # matrices this used to build, but stay cheap when the clash set is every
    # binder heavy atom rather than just its CAs.
    tree_protein, tree_hotspot = cKDTree(protein_coords), cKDTree(hotspot_atom_coords)
    rots = Rotation.random(n_orientations, random_state=np.random.RandomState(seed))
    best = None  # (min_hs, offset, min_prot, rot, target_center)
    for offset in offsets:
        target_center = hotspot_centroid + outward * offset
        for rot in rots:
            clash_candidate = rot.apply(clash_centered) + target_center
            min_prot = float(tree_protein.query(clash_candidate, k=1)[0].min())
            if min_prot < clash_cutoff:
                continue
            candidate = rot.apply(mobile_centered) + target_center
            min_hs = float(tree_hotspot.query(candidate, k=1)[0].min())
            if best is None or min_hs < best[0]:
                best = (min_hs, offset, min_prot, rot, target_center)
        if best is not None and best[0] <= hotspot_warn_cutoff:
            break

    if best is None:
        raise RuntimeError(
            f"Could not find any clash-free placement near the hotspot region "
            f"(clash_cutoff={clash_cutoff}A) after searching {len(offsets)} offsets x "
            f"{n_orientations} rotations. The hotspot patch may be too sterically "
            f"crowded for this component's size, or the hotspots may be buried."
        )
    min_hs, offset, min_prot, rot, target_center = best
    return min_hs, min_prot, offset, rot, mobile_centroid, target_center


# ── contact-maximising placement ──────────────────────────────────────────────
#
# `search_hotspot_placement` scores a pose by the single shortest distance from
# the mobile patch to the hotspot atoms and stops at the first offset that gets
# within `hotspot_warn_cutoff`. A pose whose edge grazes one hotspot atom scores
# perfectly while the rest of the face points into solvent, which is how state-2
# seeds end up with ~half the interface residues of state 1.
#
# This variant instead maximises how much of the binder is actually seated on
# the epitope, scans every offset, and refines locally around the best coarse
# poses. It is additive: the original function is untouched.

PlacementResult = namedtuple(
    "PlacementResult",
    "min_hotspot_dist min_protein_dist offset rotation mobile_centroid target_center "
    "n_contacts n_hotspot_contacts score",
)


def _outward_normal(protein_coords: np.ndarray, hotspot_centroid: np.ndarray) -> np.ndarray:
    """Local surface normal at the hotspot patch (same estimate as the original)."""
    local = protein_coords[np.linalg.norm(protein_coords - hotspot_centroid, axis=1) < 15.0]
    local_centroid = local.mean(axis=0) if len(local) else protein_coords.mean(axis=0)
    outward = hotspot_centroid - local_centroid
    norm = np.linalg.norm(outward)
    return outward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def search_hotspot_placement_contacts(
    mobile_coords: np.ndarray,
    protein_coords: np.ndarray,
    hotspot_atom_coords: np.ndarray,
    n_orientations: int = 3000,
    clash_cutoff: float = 2.5,
    contact_cutoff: float = 5.0,
    offsets: np.ndarray | None = None,
    seed: int = 42,
    clash_mobile_coords: np.ndarray | None = None,
    hotspot_weight: float = 1.0,
    contact_weight: float = 0.25,
    refine_top_k: int = 5,
    n_refine: int = 60,
    refine_angle_deg: float = 12.0,
    refine_offset: float = 1.0,
) -> PlacementResult:
    """Rigid placement maximising seated contact at the hotspot patch.

    Score is ``hotspot_weight * (mobile atoms contacting a hotspot atom)
    + contact_weight * (mobile atoms contacting any target atom)``, so the pose
    is driven onto the epitope while overall seating breaks ties. Any pose with
    a mobile-to-target distance below ``clash_cutoff`` is rejected outright.

    Unlike the min-distance search there is no early exit: every offset is
    scanned, then the best ``refine_top_k`` poses get a local rotation/offset
    refinement. Neighbour queries use a KD-tree, so the larger orientation count
    costs less than the original's dense pairwise distance matrices.
    """
    if offsets is None:
        offsets = np.arange(4.0, 24.0, 1.0)

    mobile_centroid = mobile_coords.mean(axis=0)
    mobile_centered = mobile_coords - mobile_centroid
    clash_centered = (
        np.asarray(clash_mobile_coords) - mobile_centroid
        if clash_mobile_coords is not None else mobile_centered
    )
    tree_protein = cKDTree(protein_coords)
    tree_hotspot = cKDTree(hotspot_atom_coords)
    hotspot_centroid = hotspot_atom_coords.mean(axis=0)
    outward = _outward_normal(protein_coords, hotspot_centroid)

    def evaluate(rotation, offset):
        """Return (score, n_contacts, n_hotspot, min_hotspot, min_protein) or None if clashing."""
        center = hotspot_centroid + outward * float(offset)
        clash_distance, _ = tree_protein.query(rotation.apply(clash_centered) + center, k=1)
        min_protein = float(clash_distance.min())
        if min_protein < clash_cutoff:
            return None
        placed = rotation.apply(mobile_centered) + center
        contact_distance, _ = tree_protein.query(placed, k=1)
        hotspot_distance, _ = tree_hotspot.query(placed, k=1)
        n_contacts = int((contact_distance <= contact_cutoff).sum())
        n_hotspot = int((hotspot_distance <= contact_cutoff).sum())
        score = hotspot_weight * n_hotspot + contact_weight * n_contacts
        return score, n_contacts, n_hotspot, float(hotspot_distance.min()), min_protein

    rng = np.random.RandomState(seed)
    coarse = []
    for rotation in Rotation.random(n_orientations, random_state=rng):
        for offset in offsets:
            result = evaluate(rotation, offset)
            if result is not None:
                coarse.append((result[0], rotation, float(offset), result))
    if not coarse:
        raise RuntimeError(
            f"No clash-free placement found (clash_cutoff={clash_cutoff} A) over "
            f"{len(offsets)} offsets x {n_orientations} rotations. The hotspot patch "
            f"may be too crowded, or the mobile component too large."
        )

    coarse.sort(key=lambda item: -item[0])
    best = coarse[0]
    for _, rotation, offset, _ in coarse[:refine_top_k]:
        for _ in range(n_refine):
            perturbed = Rotation.from_rotvec(
                np.deg2rad(refine_angle_deg) * rng.normal(size=3)
            ) * rotation
            nudged = float(np.clip(offset + rng.uniform(-refine_offset, refine_offset),
                                   float(offsets[0]), float(offsets[-1])))
            result = evaluate(perturbed, nudged)
            if result is not None and result[0] > best[0]:
                best = (result[0], perturbed, nudged, result)

    score, rotation, offset, (_, n_contacts, n_hotspot, min_hotspot, min_protein) = best
    return PlacementResult(
        min_hotspot_dist=min_hotspot, min_protein_dist=min_protein, offset=offset,
        rotation=rotation, mobile_centroid=mobile_centroid,
        target_center=hotspot_centroid + outward * offset,
        n_contacts=n_contacts, n_hotspot_contacts=n_hotspot, score=score,
    )
