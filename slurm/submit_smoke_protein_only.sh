#!/bin/bash
#SBATCH --job-name=smoke_po
#SBATCH --partition=cpu-single
#SBATCH --mem=8G
#SBATCH --time=24:00:00   # measured canonical smokes: 1h13m-11h39m. The
#                          # spread is queue latency (~71% of wall time), not
#                          # compute, so 12h once left only 21 min of headroom.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Smoke test (integration check, minimal counts). Orchestrator only -- never
# touches a GPU directly; it submits Slurm array jobs per step.
#
#   sbatch slurm/submit_smoke_protein_only.sh [run_name]
#
# WS resolves to the repository root: SLURM_SUBMIT_DIR when submitted with
# `sbatch slurm/...` from the root, otherwise this script's own parent.
set -euo pipefail

WS="${WS:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONDA_SH="${CONDA_SH:-/gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh}"
RUN_NAME="${1:-smoke_po_$(date +%Y%m%d_%H%M%S)}"

export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.conda/envs/' | paste -sd ':' -)"
source "$CONDA_SH"
conda activate protflow
cd "$WS"
mkdir -p logs

echo "=== smoke '$RUN_NAME' started $(date) ==="
python -u src/switch_pipeline.py \
    --config configs/pdl1_pcna_protein_only_nogate.yaml \
    --run-name "$RUN_NAME" \
    --smoke \
    2>&1 | tee "logs/${RUN_NAME}.log"
echo "=== smoke '$RUN_NAME' finished $(date) ==="
