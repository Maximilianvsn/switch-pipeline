#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# setup.sh — fetch the third-party tools this pipeline orchestrates.
#
# Clones each dependency at the exact commit the pipeline was validated
# against, then applies the local patches under patches/ where we carry
# modifications. Run once from the repo root:  bash scripts/setup.sh
#
# What this does NOT do: create conda envs or download model weights.
# See README.md ("Setup") and env/DEPENDENCIES.md for those — they are
# large and cluster-specific.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root
ROOT="$(pwd)"

clone_at() {  # name  url  commit
  local name="$1" url="$2" commit="$3"
  if [[ -d "$name/.git" ]]; then
    echo "[$name] already present — skipping clone"
  else
    echo "[$name] cloning $url"
    git clone "$url" "$name"
  fi
  git -C "$name" fetch --quiet origin "$commit" 2>/dev/null || true
  git -C "$name" checkout --quiet "$commit"
  echo "[$name] at $(git -C "$name" rev-parse --short HEAD)"
}

apply_patch() {  # name  patchfile
  local name="$1" patch="$2"
  [[ -f "$patch" ]] || { echo "[$name] no patch"; return; }
  if git -C "$name" apply --check "$ROOT/$patch" 2>/dev/null; then
    git -C "$name" apply "$ROOT/$patch"
    echo "[$name] applied $(basename "$patch")"
  else
    echo "[$name] WARNING: $(basename "$patch") does not apply cleanly — apply manually" >&2
  fi
}

# ── clean upstream clones ──
clone_at ProteinMPNN https://github.com/dauparas/ProteinMPNN.git   8907e6671bfbfc92303b5f79c4b5e6ce47cdef57
clone_at boltz        https://github.com/jwohlwend/boltz.git       b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc

# ── clones carrying local patches ──
clone_at DynamicMPNN  https://github.com/Alex-Abrudan/DynamicMPNN.git 1f3e326c0f4d275ee8b3918e4726e19d3eef6c3f
apply_patch DynamicMPNN patches/DynamicMPNN.patch

clone_at LigandMPNN   https://github.com/dauparas/LigandMPNN.git   26ec57ac976ade5379920dbd43c7f97a91cf82de
apply_patch LigandMPNN patches/LigandMPNN.patch

# ── ProtFlow: heavily adapted (runner classes). Prefer your own fork; the
#    patch + the new dynamicmpnn.py runner reproduce the working tree. ──
clone_at ProtFlow     https://github.com/mabr3112/ProtFlow.git     43a4381821ac85a44e2ec5ec16abb3ca44d12d9f
apply_patch ProtFlow patches/ProtFlow.patch
cp -n patches/protflow_dynamicmpnn.py.new ProtFlow/protflow/tools/dynamicmpnn.py \
  && echo "[ProtFlow] installed new runner protflow/tools/dynamicmpnn.py"

echo
echo "Done. Next:"
echo "  1. Create conda envs (protflow / ligandmpnn_env / BindCraft) — see README.md."
echo "  2. Download model weights (checkpoints/, tools_af2/) — see DEPENDENCIES.md."
echo "  3. Point configs/ at your tool envs (see README 'Environments')."
