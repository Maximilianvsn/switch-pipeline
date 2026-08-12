#!/bin/bash
#SBATCH --job-name=proteins_nogate
#SBATCH --partition=cpu-single
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Full production run with the AF2 paired-null HARD STOP disabled (nogate config),
# so it proceeds to four-state Boltz + evaluation and yields scored designs to
# write about. The null is still computed/reported. Resume with the same name.

set -euo pipefail

WS="${WS:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONDA_SH="${CONDA_SH:-/gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh}"
RUN_NAME="${1:-}"

if [[ -z "$RUN_NAME" ]]; then
  DATE_TAG="$(date +%Y%m%d)"
  RUN_PREFIX="production_proteins_pdl1_pcna_nogate_${DATE_TAG}"
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
echo "=== nogate production run '$RUN_NAME' started at $(date) ==="
python -u src/switch_pipeline.py \
  --config configs/pdl1_pcna_protein_only_nogate.yaml \
  --run-name "$RUN_NAME" \
  2>&1 | tee "logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "=== nogate production run '$RUN_NAME' finished at $(date) ==="
