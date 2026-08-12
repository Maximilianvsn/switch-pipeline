"""Structure and confidence-file I/O: CIF/PDB chain surgery, pLDDT/PAE readers.

Every function here is pure, taking filesystem input and returning a value or
writing a file, and captured no enclosing state, so the extraction is
behaviour-preserving by construction.

`make_backbone_rmsd` is the exception: it works around an upstream ProtFlow
defect, documented in its own docstring.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from protflow.poses import Poses
from protflow.metrics.rmsd import BackboneRMSD
from protflow import load_config_path, require_config


def collapse_to_best_model(poses: Poses, metric_col: str, group_col: str = "_pre_boltz_id") -> Poses:
    """
    Collapse multiple Boltz models/diffusion-samples per design down to
    the single best-scoring row, keyed on group_col.

    Boltz with --diffusion_samples N produces N rows per input pose. If
    poses then feeds into another Boltz step without collapsing first,
    row counts compound multiplicatively across sequential Boltz calls
    (e.g. 4 steps x 3 samples = 81x blowup). Call this immediately after
    every boltz.run() in a multi-step chain to keep row counts stable.

    Requires group_col to already exist in poses.df (set it to the
    poses_description value *before* calling boltz.run()).
    """
    df = poses.df
    if group_col not in df.columns:
        raise KeyError(f"{group_col} not found — set poses.df['{group_col}'] before calling boltz.run()")
    best_idx = df.groupby(group_col)[metric_col].idxmax()
    poses.df = df.loc[best_idx].reset_index(drop=True)
    return poses


def load_plddt_mean(npz_path: str) -> float:
    """Read a Boltz pLDDT .npz and return the mean over all residues."""
    if not os.path.isfile(str(npz_path)):
        return np.nan
    data = np.load(npz_path)
    arr = data[list(data.keys())[0]]
    return float(np.mean(arr))


def get_chain_token_boundaries(cif_path: str) -> dict:
    """
    Map each chain id to its [start, end) index range in Boltz's flat
    per-token pLDDT array. Standard polymer residues are one token each;
    ligands/hetero groups are tokenized per heavy atom (Boltz follows the
    AF3-style atomized tokenization for non-polymer entities), so a
    naive "one token per residue" assumption undercounts ligand chains.
    """
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("x", cif_path)

    boundaries = {}
    idx = 0
    for chain in struct[0].get_chains():
        residues = list(chain.get_residues())
        is_ligand = any(r.id[0] != " " for r in residues)
        if is_ligand:
            n_tokens = sum(len([a for a in r.get_atoms() if a.element != "H"]) for r in residues)
        else:
            n_tokens = len(residues)
        boundaries[chain.get_id()] = (idx, idx + n_tokens)
        idx += n_tokens
    return boundaries


def get_binder_plddt(cif_path: str, plddt_npz_path: str, binder_chain: str) -> float:
    """
    Mean pLDDT over only the binder chain's tokens (not the whole
    complex). A high complex-wide pLDDT can hide a poorly-folded binder
    if the target chain dominates the average — this isolates the
    binder's own structural confidence.
    """
    if not os.path.isfile(str(cif_path)) or not os.path.isfile(str(plddt_npz_path)):
        return np.nan
    try:
        boundaries = get_chain_token_boundaries(cif_path)
        if binder_chain not in boundaries:
            return np.nan
        start, end = boundaries[binder_chain]
        data = np.load(plddt_npz_path)
        arr = data[list(data.keys())[0]]
        if end > len(arr):
            return np.nan
        return float(np.mean(arr[start:end]))
    except Exception:
        return np.nan


def get_interface_pae(cif_path: str, pae_npz_path: str, c1: str = "A", c2: str = "B") -> float:
    """Mean predicted aligned error over the binder<->target token block
    (both off-diagonal quadrants) — the field-standard de novo binder metric
    (Bennett 2023: pae_interaction < ~10 A marks a confident designed
    interface). Lower is better. Unlike ipTM this is a direct, physically
    interpretable measure of how confidently the model places the two chains
    *relative to each other*, so it is far less prone to the self-consistency
    circularity that inflates ipTM for designed sequences.

    Reuses get_chain_token_boundaries so the token indices line up with
    Boltz's atomized-ligand tokenization. Returns NaN on any parse issue
    (missing file, chain absent, shape mismatch) so it can never hard-fail a
    scoring job — a missing PAE just drops out of the gate.
    """
    if not (os.path.isfile(str(cif_path)) and os.path.isfile(str(pae_npz_path))):
        return np.nan
    try:
        bounds = get_chain_token_boundaries(cif_path)
        if c1 not in bounds or c2 not in bounds:
            return np.nan
        data = np.load(pae_npz_path)
        P = data[list(data.keys())[0]]
        a0, a1 = bounds[c1]
        b0, b1 = bounds[c2]
        if max(a1, b1) > P.shape[0]:
            return np.nan
        block = np.concatenate([P[a0:a1, b0:b1].ravel(), P[b0:b1, a0:a1].ravel()])
        return float(block.mean()) if block.size else np.nan
    except Exception:
        return np.nan


def _extract_binder_to_chain(pdb_path: str, src_chain: str, dst_chain: str, out_path: str):
    """Extract a single chain from a PDB/CIF, renaming it to dst_chain.

    Renaming is done in two passes: first the target chain is extracted
    into its own single-chain structure (no id collision possible there),
    then it is renamed. Renaming in place on the original multi-chain
    structure is unsafe when dst_chain already exists as a sibling —
    BioPython's child_dict does not update atomically on `.id =`
    assignment, so the original sibling gets written out instead of the
    renamed chain (silent wrong-chain bug).
    """
    import gzip, shutil
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

    path = pdb_path
    if path.endswith(".gz"):
        decompressed = path.replace(".gz", "")
        with gzip.open(path, "rb") as fi, open(decompressed, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        path = decompressed

    parser = MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)
    struct = parser.get_structure("s", path)

    chain_ids = [c.get_id() for c in struct[0].get_chains()]
    if src_chain not in chain_ids:
        # Only currently reachable if a structure is missing the expected
        # chain entirely (verified not to happen for real holo/apo output
        # in this pipeline) — fall back to "the other" chain only when
        # which is unambiguous; a silently wrong choice among several candidates
        # is worse than a clear error.
        fallback_candidates = [c for c in chain_ids if c != "A"]
        if len(fallback_candidates) != 1:
            raise ValueError(
                f"_extract_binder_to_chain: requested chain '{src_chain}' not found in "
                f"{pdb_path} (chains present: {chain_ids}); fallback is ambiguous "
                f"({len(fallback_candidates)} non-'A' candidates: {fallback_candidates}). Refusing to guess."
            )
        print(f"WARNING: _extract_binder_to_chain: chain '{src_chain}' not found in {pdb_path} "
              f"(chains present: {chain_ids}) — falling back to chain '{fallback_candidates[0]}'.")
        src_chain = fallback_candidates[0]

    class _Sel(Select):
        def accept_chain(self, chain):
            return chain.get_id() == src_chain

    io = PDBIO()
    io.set_structure(struct)
    io.save(out_path, _Sel())

    if src_chain != dst_chain:
        single_struct = PDBParser(QUIET=True).get_structure("x", out_path)
        single_struct[0][src_chain].id = dst_chain
        io2 = PDBIO()
        io2.set_structure(single_struct)
        io2.save(out_path)


def _extract_chain_by_length(pdb_path: str, target_len: int, dst_chain: str, out_path: str,
                             tolerance: int = 2) -> str:
    """Extract whichever chain's residue count matches `target_len` (the
    binder's own sequence length), renaming it to dst_chain. Used for AF2's
    saved structures (consensus tier) instead of a hardcoded chain label:
    colabdesign's internal chain-labeling convention for its OWN output was
    not assumed here — matching by length is correct regardless of whatever
    that convention turns out to be, and fails loudly (not silently) if no
    chain is a plausible match.
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    struct = PDBParser(QUIET=True).get_structure("s", pdb_path)
    counts = {c.get_id(): len([r for r in c if r.id[0] == " "]) for c in struct[0]}
    matches = [cid for cid, n in counts.items() if abs(n - target_len) <= tolerance]
    if len(matches) != 1:
        raise ValueError(
            f"_extract_chain_by_length: no unambiguous chain of length ~{target_len} "
            f"(+/-{tolerance}) in {pdb_path} (chain:residue-count = {counts}). "
            f"{'No match' if not matches else f'{len(matches)} ambiguous matches: {matches}'} -- refusing to guess."
        )
    src_chain = matches[0]

    class _Sel(Select):
        def accept_chain(self, chain):
            return chain.get_id() == src_chain

    io = PDBIO()
    io.set_structure(struct)
    io.save(out_path, _Sel())

    if src_chain != dst_chain:
        single_struct = PDBParser(QUIET=True).get_structure("x", out_path)
        single_struct[0][src_chain].id = dst_chain
        io2 = PDBIO()
        io2.set_structure(single_struct)
        io2.save(out_path)
    return out_path


def make_backbone_rmsd(**kwargs) -> BackboneRMSD:
    """
    BackboneRMSD (and most other PROTFLOW_ENV-consuming classes across
    ProtFlow) build their python interpreter path as
    os.path.join(PROTFLOW_ENV, "python") — but this cluster's config
    (matching ProtFlow's own config_template.py, which documents
    PROTFLOW_ENV as "/path/to/.../bin/python3") sets PROTFLOW_ENV to the
    full python executable path already, not a directory. That join
    produces a nonsense path ("<python3 path>/python") and every job
    fails with "Not a directory". This is a pre-existing bug in ProtFlow
    itself (confirmed present in ~10 other classes, not just this one) —
    worked around locally here rather than patching the shared dependency,
    since nothing else in this project exercises BackboneRMSD to have
    surfaced it before.
    """
    instance = BackboneRMSD(**kwargs)
    instance.python = load_config_path(require_config(), "PROTFLOW_ENV")
    return instance


def _cif_locations_to_pdb(cif_paths: list[str], out_dir: str, chain: str = "A") -> list[str]:
    """
    BackboneRMSD's calc_rmsd.py backend only accepts .pdb, but Boltz
    always outputs .cif — convert (and extract the single binder chain,
    "A" in Boltz's single-sequence monomer/complex output convention,
    verified empirically against real completed-run output, not assumed).
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_paths = []
    for i, cif_path in enumerate(cif_paths):
        # "pose_" prefix is load-bearing, not cosmetic: calc_rmsd.py derives
        # description from this filename's stem, and BackboneRMSD.run()
        # reads its scores back via pd.read_json(), which silently coerces
        # an all-numeric "description" column (e.g. "0", "1", ...) to
        # integers while the location-derived comparison column stays
        # string — failing RunnerOutput's description==location check on
        # every row. A non-numeric stem sidesteps the coercion entirely.
        pdb_path = os.path.join(out_dir, f"pose_{i}.pdb")
        _extract_binder_to_chain(cif_path, chain, chain, pdb_path)
        pdb_paths.append(pdb_path)
    return pdb_paths

