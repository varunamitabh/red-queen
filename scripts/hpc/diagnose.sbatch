#!/usr/bin/env bash
# Why did calibration score 0/20? Three arms, 20 episodes each, n_action_steps=10:
#   long_clean : the same 20 LIBERO-10 loop-pool variants, instruction suffix removed
#   obj_raw    : 20 of the easiest (level 1) LIBERO-Object variants, instruction as shipped
#   obj_clean  : the same LIBERO-Object variants, instruction suffix removed
# LIBERO-Object is the suite this checkpoint is strongest on, so near-zero there means a setup bug.
#   sbatch <your account/partition/GPU flags> --export=ALL,RQ_ROOT=$RQ_ROOT scripts/hpc/diagnose.sbatch
#SBATCH --job-name=rq-diag
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
export RQ_OFFLINE=1
source scripts/hpc/env.sh
mkdir -p "$RQ_OUT"
OUT="$RQ_OUT/diag-$SLURM_JOB_ID"
OBJ_ARGS="--suite libero_object --levels 1 --families 'Robot Initial States' 'Light Conditions' 'Objects Layout' --n 20 --seed 0"
LONG=$(python scripts/sample_ids.py --n 20 --seed 0)
OBJ=$(eval python scripts/sample_ids.py $OBJ_ARGS)
eval python scripts/sample_ids.py $OBJ_ARGS --table

run() {  # label suite ids [--clean-language]
  local label=$1 suite=$2 ids=$3; shift 3
  SECONDS=0
  python scripts/hpc/run_eval.py "$@" \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=10 \
    --env.type=libero_plus \
    --env.task="$suite" \
    --env.task_ids="$ids" \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --output_dir="$OUT/$label" > "$OUT.$label.log" 2>&1 || { tail -40 "$OUT.$label.log"; exit 1; }
  echo "$label wall=${SECONDS}s"
  grep -m1 "clean-language" "$OUT.$label.log" || true
  python scripts/hpc/summarize_eval.py "$OUT/$label/eval_info.json" --label "$label"
  echo "INSTRUCTIONS $label: $(head -c 600 "$OUT/$label/task_descriptions.json" | tr '\n' ' ')"
}
run obj_raw    libero_object "$OBJ"
run obj_clean  libero_object "$OBJ" --clean-language
run long_clean libero_10     "$LONG" --clean-language
echo "videos: $OUT/*/videos"