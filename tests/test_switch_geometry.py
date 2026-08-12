"""Unit tests for the two-state geometry metrics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from switch_geometry import aligned_binder_rmsd, interface_positions, paired_state_geometry


def _write_pdb(path: Path, chains):
    serial = 1
    lines = []
    for chain_id, coords in chains:
        for residue_number, (x, y, z) in enumerate(coords, start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{residue_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
            )
            serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def test_geometry_is_chain_id_agnostic_and_measures_interface_reuse():
    binder1 = [(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)]
    target1 = [(0, 3, 0), (3, 3, 0), (6, 3, 0), (9, 3, 0)]
    binder2 = [(10, 5, 0), (13, 5, 1.5), (16, 5, 0), (19, 5, 0)]
    target2 = [(10, 8, 0), (13, 8, 1.5), (16, 8, 0), (19, 8, 0)]
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "state1.pdb"
        second = Path(directory) / "state2.pdb"
        _write_pdb(first, [("A", target1), ("B", binder1)])
        _write_pdb(second, [("A", binder2), ("B", target2)])

        rmsd = aligned_binder_rmsd(str(first), str(second), "B", "A")
        assert 0.4 < rmsd < 1.0
        assert interface_positions(str(first), "B", cutoff=4.0) == {1, 2, 3, 4}
        metrics = paired_state_geometry(str(first), str(second), "B", "A", interface_cutoff=4.0)
        assert metrics["interface_jaccard"] == 1.0
        assert metrics["target_target_clash_pairs"] > 0


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        unittest.FunctionTestCase(test_geometry_is_chain_id_agnostic_and_measures_interface_reuse)
    ])
