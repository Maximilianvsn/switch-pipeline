"""
Generate per-design RFD3 apo partial-diffusion input JSONs.

Run after Step 1 (holo diffusion) has completed.
Takes each state-1 RFD3 output, extracts the binder chain, combines it
with target B, and writes a JSON spec for RFD3 partial diffusion.

Usage (standalone):
    python write_apo_inputs.py \
        --holo_dir  $WS/outputs/s1_rfd3_holo \
        --pcna_pdb  $WS/inputs/pcna_A.pdb \
        --out_dir   $WS/inputs/rfd3_apo_inputs

Usage (from pipeline):
    Called as a function via generate_apo_inputs().
"""
import argparse
import json
import os
import sys
import gzip
from glob import glob
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docking_utils import search_hotspot_placement
from switch_geometry import sha256_file
from interface_seeding import combine_binder_target_same_interface


class ChainSelect(Select):
    def __init__(self, chain_id, new_chain_id=None):
        self.chain_id = chain_id
        self.new_chain_id = new_chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id


def extract_binder_chain(cif_path: str, out_pdb: str, binder_chain: str = "B"):
    if cif_path.endswith(".gz"):
        import shutil
        decompressed = cif_path.replace(".gz", "")
        with gzip.open(cif_path, "rb") as f_in, open(decompressed, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        cif_path = decompressed

    parser = MMCIFParser(QUIET=True) if cif_path.endswith(".cif") else PDBParser(QUIET=True)
    structure = parser.get_structure("holo", cif_path)

    chain_ids = [c.get_id() for c in structure[0].get_chains()]
    if binder_chain not in chain_ids:
        for cid in chain_ids:
            if cid not in ("A",):
                binder_chain = cid
                break

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb, ChainSelect(binder_chain))
    return out_pdb, binder_chain


def _parse_hotspot_str(hotspots: str):
    """Parse "A251,A252" -> [("A", 251), ("A", 252)]."""
    out = []
    for tok in hotspots.split(","):
        tok = tok.strip()
        if tok:
            out.append((tok[0], int(tok[1:])))
    return out


def _dock_binder_to_hotspots(binder_struct, pcna_struct, pcna_hotspots: str):
    """Rigidly move the binder so it sits in contact with PCNA's hotspot patch.

    The binder comes from the holo RFD3 output in PD-L1's coordinate frame while
    PCNA is in its own frame, so naive concatenation left them ~9-12 A apart
    with ZERO interface contacts — RFD3 partial diffusion then had no real apo
    interface to refine, and the hotspot-directed PCNA design was silently
    unenforced. This reuses the same hotspot-placement search originally used to
    seat a cofactor at the PD-L1 interface. It is a seed, not a docking result: partial
    diffusion forms the actual interface. Mutates binder_struct atom coords in
    place. Non-fatal on failure (leaves the binder where it was, with a warning).
    """
    hotspot_set = set(_parse_hotspot_str(pcna_hotspots))
    hotspot_coords = np.array([
        a.coord for chain in pcna_struct[0] for res in chain
        if res.id[0] == " " and (chain.id, res.id[1]) in hotspot_set for a in res
    ])
    if len(hotspot_coords) == 0:
        print(f"  WARNING: none of PCNA hotspots '{pcna_hotspots}' found in PCNA structure — "
              f"skipping binder docking (apo binder may end up off-interface).")
        return

    pcna_coords = np.array([
        a.coord for chain in pcna_struct[0] for res in chain if res.id[0] == " " for a in res
    ])
    binder_atoms = [a for chain in binder_struct[0] for res in chain for a in res]
    binder_all = np.array([a.coord for a in binder_atoms])
    binder_ca = np.array([a.coord for a in binder_atoms if a.get_name() == "CA"])
    mobile = binder_ca if len(binder_ca) >= 3 else binder_all

    try:
        # A globular ~80-residue binder needs its centroid ~one binder-radius
        # out from the epitope for surface contact, so the offset range is
        # larger than for a small ligand.
        min_hs, min_prot, offset, rot, mob_cen, tgt_cen = search_hotspot_placement(
            mobile_coords=mobile, protein_coords=pcna_coords, hotspot_atom_coords=hotspot_coords,
            n_orientations=150, clash_cutoff=2.5, offsets=np.arange(8.0, 32.0, 2.0),
            hotspot_warn_cutoff=8.0,
        )
    except RuntimeError as e:
        print(f"  WARNING: binder->PCNA docking found no clash-free placement ({e}); "
              f"leaving binder in place.")
        return

    for atom, coord in zip(binder_atoms, binder_all):
        atom.set_coord(rot.apply(coord - mob_cen) + tgt_cen)
    print(f"  Docked binder onto PCNA hotspots {pcna_hotspots}: binder-to-hotspot min "
          f"{min_hs:.1f}A, min binder-to-PCNA {min_prot:.1f}A (offset {offset:.0f}A)")
    if min_hs > 8.0:
        print(f"  WARNING: docked binder is {min_hs:.1f}A from PCNA hotspots (>8A) — "
              f"partial diffusion may not form a proper interface.")


def generate_apo_inputs(
    holo_dir: str,
    pcna_pdb: str,
    out_dir: str,
    pcna_hotspots: str = "A251,A252,A253,A255",
    partial_t: float = 2.0,
    binder_chain: str = "B",
    include_stems: set[str] | None = None,
) -> list[str]:
    """include_stems: if given, only process holo backbones whose file stem
    (matching s1_rfd3_holo_description) is in this set — lets a designability
    pre-filter cull undesignable backbones BEFORE the expensive apo partial
    diffusion (and the per-backbone docking search here) runs on them.
    None (default) processes every backbone found in holo_dir, unchanged."""
    os.makedirs(out_dir, exist_ok=True)
    staging_dir = os.path.join(out_dir, "combined_pdbs")
    os.makedirs(staging_dir, exist_ok=True)

    pcna_pdb = os.path.abspath(pcna_pdb)
    json_paths = []
    lineage = []

    input_files = sorted(
        glob(os.path.join(holo_dir, "**/*.cif.gz"), recursive=True)
        + glob(os.path.join(holo_dir, "**/*.cif"), recursive=True)
        + glob(os.path.join(holo_dir, "**/*.pdb"), recursive=True)
    )

    if not input_files:
        raise FileNotFoundError(f"No structure files (CIF/PDB) found in {holo_dir}")

    if include_stems is not None:
        before = len(input_files)
        input_files = [f for f in input_files if Path(f).stem.replace(".cif", "") in include_stems]
        print(f"  generate_apo_inputs: designability pre-filter kept {len(input_files)}/{before} backbones")
        if not input_files:
            raise FileNotFoundError(
                "generate_apo_inputs: include_stems filtered out ALL backbones — "
                "designability pre-filter threshold may be too strict.")

    for cif in input_files:
        name = Path(cif).stem.replace(".cif", "")

        binder_pdb = os.path.join(staging_dir, f"{name}_binder.pdb")
        _, actual_binder_chain = extract_binder_chain(cif, binder_pdb, binder_chain)

        combined_pdb = os.path.join(staging_dir, f"{name}_binder_pcna.pdb")
        combine_binder_target_same_interface(
            holo_path=cif, binder_pdb=binder_pdb, target_pdb=pcna_pdb,
            out_pdb=combined_pdb, target_hotspots=pcna_hotspots,
            binder_chain=actual_binder_chain,
        )

        binder_parser = PDBParser(QUIET=True)
        binder_struct = binder_parser.get_structure("b", binder_pdb)
        binder_cid = list(binder_struct[0].get_chains())[0].get_id()
        binder_residues = list(binder_struct[0][binder_cid].get_residues())
        n_res = len(binder_residues)

        # Build PCNA contig from actual residue numbers to handle gaps in structure
        combined_struct = binder_parser.get_structure("c", combined_pdb)
        pcna_res_ids = sorted([r.get_id()[1] for r in combined_struct[0]["A"].get_residues()])
        pcna_segments = []
        seg_start = seg_prev = pcna_res_ids[0]
        for rid in pcna_res_ids[1:]:
            if rid == seg_prev + 1:
                seg_prev = rid
            else:
                pcna_segments.append(f"A{seg_start}-{seg_prev}")
                seg_start = seg_prev = rid
        pcna_segments.append(f"A{seg_start}-{seg_prev}")
        pcna_contig = ",".join(pcna_segments)

        spec_key = f"{name}_binder_pcna"
        spec = {
            spec_key: {
                "input": os.path.abspath(combined_pdb),
                "contig": f"{binder_cid}1-{n_res},{pcna_contig}",
                "partial_t": partial_t,
                # Indexed contig residues are coordinate-fixed by default in
                # RFD3. Override that default explicitly: PCNA is the fixed
                # target motif, while the binder is the partially diffused
                # region whose conformation must be allowed to change.
                "select_fixed_atoms": pcna_contig,
                "select_hotspots": pcna_hotspots,
            }
        }

        json_path = os.path.join(out_dir, f"{name}_apo.json")
        with open(json_path, "w") as f:
            json.dump(spec, f, indent=2)
        json_paths.append(json_path)
        lineage.append({
            "source_id": name,
            "source_structure": os.path.abspath(cif),
            "source_sha256": sha256_file(cif),
            "staged_binder": os.path.abspath(binder_pdb),
            "staged_binder_sha256": sha256_file(binder_pdb),
            "staged_complex": os.path.abspath(combined_pdb),
            "staged_complex_sha256": sha256_file(combined_pdb),
            "spec_json": os.path.abspath(json_path),
            "spec_sha256": sha256_file(json_path),
            "partial_t_angstrom": float(partial_t),
            "fixed_target_selection": pcna_contig,
        })

    with open(os.path.join(out_dir, "lineage_manifest.json"), "w") as handle:
        json.dump(lineage, handle, indent=2)

    return json_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RFD3 apo input JSONs")
    parser.add_argument("--holo_dir", required=True, help="Directory with holo CIF outputs")
    parser.add_argument("--pcna_pdb", required=True, help="Path to pcna_A.pdb")
    parser.add_argument("--out_dir", required=True, help="Output directory for apo JSONs")
    parser.add_argument("--partial_t", type=float, default=2.0, help="RFD3 coordinate noise in Angstroms")
    parser.add_argument("--binder_chain", default="B")
    args = parser.parse_args()

    paths = generate_apo_inputs(
        holo_dir=args.holo_dir,
        pcna_pdb=args.pcna_pdb,
        out_dir=args.out_dir,
        partial_t=args.partial_t,
        binder_chain=args.binder_chain,
    )
    print(f"Generated {len(paths)} apo input JSONs in {args.out_dir}")
