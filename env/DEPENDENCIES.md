# External dependencies

This pipeline orchestrates several third-party tools. They are **not** vendored
into this repository — `scripts/setup.sh` clones each at the pinned commit below
and applies the patches under `patches/` where we carry local changes.

## Tool repositories

| Tool | Upstream | Pinned commit | Local changes |
|------|----------|---------------|---------------|
| ProteinMPNN | https://github.com/dauparas/ProteinMPNN | `8907e66` | none (clean) |
| boltz | https://github.com/jwohlwend/boltz | `b1ebfc4` | none (clean) |
| DynamicMPNN | https://github.com/Alex-Abrudan/DynamicMPNN | `1f3e326` | `patches/DynamicMPNN.patch` (3 files: foldseek scripts, `train.py`) |
| LigandMPNN | https://github.com/dauparas/LigandMPNN | `26ec57a` | `patches/LigandMPNN.patch` (5 `openfold/` files) |
| ProtFlow | https://github.com/mabr3112/ProtFlow | `43a4381` | `patches/ProtFlow.patch` + `patches/protflow_dynamicmpnn.py.new` |

### ProtFlow — recommended: use a fork

ProtFlow is adapted the most (primarily the tool **runner classes**, plus a new
`protflow/tools/dynamicmpnn.py` runner that the pipeline imports). The patch
reproduces the validated working tree, but a 700-line patch is fragile to
maintain across upstream updates. The cleaner long-term setup is to **push these
changes to your own fork** of `mabr3112/ProtFlow` and have `setup.sh` clone that
fork instead. Until then, the patch is the source of truth.

## Model weights & large installs (not fetched by setup.sh)

These are multi-GB and cluster-specific — obtain them separately and place them
at the paths the configs expect (relative to the repo root):

- `DynamicMPNN/checkpoints/multi_chain_reload.ckpt` — DynamicMPNN weights.
- `ProteinMPNN/vanilla_model_weights/` — ships with the ProteinMPNN repo.
- `LigandMPNN/model_params/` — download per LigandMPNN's `get_model_params.sh`.
- `tools_af2/localcolabfold/` — [localcolabfold](https://github.com/YoshitakaMo/localcolabfold)
  install; the AF2 initial-guess gate reads `colabfold/params/` from here.
- Boltz-2 weights — auto-downloaded by boltz on first run.
