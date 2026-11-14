#!/usr/bin/env bash
# Speed and baseline calibration: the same 20 loop-pool variants, run once per value of
# n_action_steps (how many actions of each 50-action chunk are executed before re-planning).
# Answers: how many seconds per episode, and how much success changes with re-planning rate.
#   sbatch <your account/partition/GPU flags> --export=ALL,RQ_ROOT=$RQ_ROOT scripts/hpc/calibrate.sbatch
# Optional: N_VARIANTS=20 NAS_LIST="10 50 1" (slowest last, so a timeout still leaves results).
#SBATCH --job-name=rq-calib
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
export RQ_OFFLINE=1
source scripts/hpc/env.sh

IDS=$(python scripts/sample_ids.py --n "${N_VARIANTS:-20}" --seed 0)
[ -z "$IDS" ] && echo "ERROR: sample_ids.py produced no output — is redqueen installed? (pip install -e .)" >&2 && exit 1
python scripts/sample_ids.py --n "${N_VARIANTS:-20}" --seed 0 --table
OUT="$RQ_OUT/calib-$SLURM_JOB_ID"
mkdir -p "$RQ_OUT"
for NAS in ${NAS_LIST:-10 50 1}; do
  SECONDS=0
  lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps="$NAS" \
    --env.type=libero_plus \
    --env.task=libero_10 \
    --env.task_ids="$IDS" \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --output_dir="$OUT/nas$NAS" > "$OUT.nas$NAS.log" 2>&1 || { tail -40 "$OUT.nas$NAS.log"; exit 1; }
  echo "n_action_steps=$NAS wall=${SECONDS}s"
  python scripts/hpc/summarize_eval.py "$OUT/nas$NAS/eval_info.json" --label "nas$NAS"
done