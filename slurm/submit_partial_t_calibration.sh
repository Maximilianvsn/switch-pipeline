#!/bin/bash
#SBATCH --job-name=partial_t_calibration
#SBATCH --partition=cpu-single
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WS="${WS:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONDA_SH="${CONDA_SH:-/gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh}"
RUN_NAME="${1:-}"

if [[ -z "$RUN_NAME" ]]; then
  DATE_TAG="$(date +%Y%m%d)"
  RUN_PREFIX="calibration_proteins_partial_t_${DATE_TAG}"
  VERSION=1
  while ! mkdir "$WS/outputs/${RUN_PREFIX}_v${VERSION}" 2>/dev/null; do
    VERSION=$((VERSION + 1))
  done
  RUN_NAME="${RUN_PREFIX}_v${VERSION}"
fi

NAME_PATTERN='^calibration_proteins_partial_t_[0-9]{8}_v[0-9]+$'
if [[ ! "$RUN_NAME" =~ $NAME_PATTERN && ! -d "$WS/outputs/$RUN_NAME" ]]; then
  echo "Invalid new calibration run name: $RUN_NAME" >&2
  echo "Expected: calibration_proteins_partial_t_YYYYMMDD_vN" >&2
  exit 2
fi
if [[ ! -d "$WS/outputs/$RUN_NAME" ]]; then
  if ! mkdir "$WS/outputs/$RUN_NAME"; then
    echo "Could not reserve run directory: $WS/outputs/$RUN_NAME" >&2
    exit 2
  fi
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if ! scontrol update JobId="$SLURM_JOB_ID" JobName="$RUN_NAME"; then
    echo "WARNING: could not update Slurm job name to '$RUN_NAME'" >&2
  fi
fi

export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.conda/envs/' | paste -sd ':' -)"
source "$CONDA_SH"
export BASH_ENV="$CONDA_SH"
conda activate protflow

cd "$WS"
echo "=== partial_t calibration '$RUN_NAME' started at $(date) ==="
python -u src/switch_pipeline.py \
  --config configs/pdl1_pcna_protein_only.yaml \
  --run-name "$RUN_NAME" \
  --geometry-calibration \
  --partial-t-values 2 5 10 15 \
  2>&1 | tee "logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "=== partial_t calibration '$RUN_NAME' finished at $(date) ==="
