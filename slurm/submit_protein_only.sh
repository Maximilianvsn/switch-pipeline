#!/bin/bash
#SBATCH --job-name=proteins_pipeline
#SBATCH --partition=cpu-single
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WS="${WS:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
RUN_NAME="${1:-}"
MODE="${2:-production}"

if [[ "$MODE" != "production" && "$MODE" != "smoke" ]]; then
  echo "Mode must be 'production' or 'smoke', got: $MODE" >&2
  exit 2
fi

# Keep every new run on one searchable convention. When no name is supplied,
# choose the next unused version; pass an existing name explicitly to resume it.
if [[ -z "$RUN_NAME" ]]; then
  DATE_TAG="$(date +%Y%m%d)"
  if [[ "$MODE" == "smoke" ]]; then
    RUN_PREFIX="smoke_proteins_fullpath_${DATE_TAG}"
  else
    RUN_PREFIX="production_proteins_pdl1_pcna_${DATE_TAG}"
  fi
  VERSION=1
  # Reserve the run directory atomically. A check-then-create loop lets two
  # jobs that start together choose the same version and silently share caches.
  while ! mkdir "$WS/outputs/${RUN_PREFIX}_v${VERSION}" 2>/dev/null; do
    VERSION=$((VERSION + 1))
  done
  RUN_NAME="${RUN_PREFIX}_v${VERSION}"
fi

if [[ "$MODE" == "smoke" ]]; then
  NAME_PATTERN='^smoke_proteins_[a-z0-9][a-z0-9_]*_[0-9]{8}_v[0-9]+$'
else
  NAME_PATTERN='^production_proteins_[a-z0-9][a-z0-9_]*_[0-9]{8}_v[0-9]+$'
fi

# Existing nonconforming names remain resumable so old output provenance is not
# broken, but all fresh runs must use the convention above.
if [[ ! "$RUN_NAME" =~ $NAME_PATTERN && ! -d "$WS/outputs/$RUN_NAME" ]]; then
  echo "Invalid new $MODE run name: $RUN_NAME" >&2
  if [[ "$MODE" == "smoke" ]]; then
    echo "Expected: smoke_proteins_<purpose>_YYYYMMDD_vN" >&2
  else
    echo "Expected: production_proteins_<target>_YYYYMMDD_vN" >&2
  fi
  exit 2
fi

# Explicit fresh names need the same atomic claim as automatically generated
# names. Existing directories are intentional resumes and remain allowed.
if [[ ! -d "$WS/outputs/$RUN_NAME" ]]; then
  if ! mkdir "$WS/outputs/$RUN_NAME"; then
    echo "Could not reserve run directory: $WS/outputs/$RUN_NAME" >&2
    exit 2
  fi
fi

# Make squeue/sacct show the exact run name instead of a generic wrapper name.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if ! scontrol update JobId="$SLURM_JOB_ID" JobName="$RUN_NAME"; then
    echo "WARNING: could not update Slurm job name to '$RUN_NAME'" >&2
  fi
fi

export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.conda/envs/' | paste -sd ':' -)"
source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
# ProtFlow-generated batch commands call `conda activate` from non-interactive
# shells. BASH_ENV makes the conda shell function available there as well.
export BASH_ENV=/gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate protflow

cd "$WS"
ARGS=(
  --config configs/pdl1_pcna_protein_only.yaml
  --run-name "$RUN_NAME"
)
if [[ "$MODE" == "smoke" ]]; then
  ARGS+=(--smoke)
fi

echo "=== $MODE run '$RUN_NAME' started at $(date) ==="
python -u src/switch_pipeline.py "${ARGS[@]}" \
  2>&1 | tee "logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "=== $MODE run '$RUN_NAME' finished at $(date) ==="
