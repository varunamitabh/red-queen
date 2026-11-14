#!/usr/bin/env bash
# Day-1 smoke test: rendering, GPU, and one SmolVLA episode in LIBERO-plus, fully offline.
# Submit from the repo root, adding your own account/partition/GPU flags, e.g.
#   sbatch --account=<acct> --partition=<gpu partition> --gpus=1 --export=ALL,RQ_ROOT=$RQ_ROOT scripts/hpc/smoke.sbatch
#SBATCH --job-name=rq-smoke
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
export RQ_OFFLINE=1
source scripts/hpc/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/hpc/render_check.py || {
  echo "EGL failed; retrying with CPU rendering"; export MUJOCO_GL=osmesa; python scripts/hpc/render_check.py; }

# Fail fast, with a clear message, if the asset pack is not where the simulator looks.
python - <<'PY'
import os, sys
from pathlib import Path
from redqueen.assets import SENTINEL
p = Path(os.environ["RQ_SRC"]) / "LIBERO-plus/libero/libero/assets" / SENTINEL
sys.exit(0 if p.exists() else f"LIBERO-plus assets missing: {p}\nRun `bash scripts/hpc/download.sh` on the login node.")
PY

OUT="$RQ_OUT/smoke-$SLURM_JOB_ID"
SECONDS=0
lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero_plus \
  --env.task=libero_10 \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir="$OUT"
echo "one episode end to end (model load included): ${SECONDS}s, MUJOCO_GL=$MUJOCO_GL"
find "$OUT" -maxdepth 3 -type f | head -20
