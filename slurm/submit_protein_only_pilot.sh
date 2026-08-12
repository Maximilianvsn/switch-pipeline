#!/bin/bash
#SBATCH --job-name=proteins_pilot
#SBATCH --partition=cpu-single
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Reduced-scale (holo_batch=150) full-path preview of the production protein-only
# pipeline: runs generation -> AF2 paired-null gate -> Boltz -> protein_only
# evaluation, so the 2026-07-19 changes (reuse/jaccard/nseq/apo_batch/temperature/
# gate) can be validated end-to-end before a full production run. Resume by
# re-submitting with the same --run-name.

set -euo pipefail

# Hardcoded to match the other live submit scripts. BASH_SOURCE-based derivation
# does NOT work under sbatch (SLURM spools the script to a temp dir), which sends
# $WS/outputs to a nonexistent path and hangs the run-dir mkdir loop.
WS="${WS:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONDA_SH="${CONDA_SH:-/gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh}"
RUN_NAME="${1:-}"

# Pick the next unused pilot version if no name given; pre-create the directory so
# the run name is reserved atomically (also lets a custom name run as a resume).
if [[ -z "$RUN_NAME" ]]; then
  DATE_TAG="$(date +%Y%m%d)"
  RUN_PREFIX="pilot_proteins_pdl1_pcna_${DATE_TAG}"
  VERSION=1
  while ! mkdir "$WS/outputs/${RUN_PREFIX}_v${VERSION}" 2>/dev/null; do
    VERSION=$((VERSION + 1))
  done
  RUN_NAME="${RUN_PREFIX}_v${VERSION}"
elif [[ ! -d "$WS/outputs/$RUN_NAME" ]]; then
  mkdir -p "$WS/outputs/$RUN_NAME"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$RUN_NAME" \
    || echo "WARNING: could not update Slurm job name to '$RUN_NAME'" >&2
fi

export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.conda/envs/' | paste -sd ':' -)"
source "$CONDA_SH"
export BASH_ENV="$CONDA_SH"
conda activate protflow

cd "$WS"
echo "=== pilot run '$RUN_NAME' started at $(date) ==="
python -u src/switch_pipeline.py \
  --config configs/pdl1_pcna_protein_only_pilot.yaml \
  --run-name "$RUN_NAME" \
  2>&1 | tee "logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "=== pilot run '$RUN_NAME' finished at $(date) ==="
