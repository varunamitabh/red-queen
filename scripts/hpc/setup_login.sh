#!/usr/bin/env bash
# One-time setup. Run on the LOGIN node (it has internet), from the repo root:
#   export RQ_ROOT=/path/outside/home/redqueen && bash scripts/hpc/setup_login.sh
set -euo pipefail
source scripts/hpc/env.sh
mkdir -p "$RQ_SRC" "$RQ_OUT" "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$LIBERO_CONFIG_PATH"

LIBERO_PLUS_REV=4976dc30028e805ff8094b55501d532c48fec182

if [ ! -d "$RQ_ENV" ]; then
  # ffmpeg for video decoding; imagemagick/fontconfig/expat replace the apt packages
  # LIBERO-plus asks for, since there is no sudo here.
  conda create -y -p "$RQ_ENV" -c conda-forge python=3.12 ffmpeg imagemagick fontconfig expat
fi
eval "$(conda shell.bash hook)" && conda activate "$RQ_ENV"

[ -d "$RQ_SRC/lerobot" ] || git clone https://github.com/huggingface/lerobot.git "$RQ_SRC/lerobot"
[ -d "$RQ_SRC/LIBERO-plus" ] || git clone https://github.com/sylvestf/LIBERO-plus.git "$RQ_SRC/LIBERO-plus"
git -C "$RQ_SRC/LIBERO-plus" checkout -q "$LIBERO_PLUS_REV"

# Install order follows the LeRobot LIBERO-plus docs.
pip install -e "$RQ_SRC/lerobot[libero,smolvla,training]" "robosuite==1.4.1" bddl easydict mujoco wand scikit-image gym
pip install --no-deps -e "$RQ_SRC/LIBERO-plus"
pip uninstall -y hf-libero || true      # so `import libero` resolves to the fork
pip install -e ".[dev]"                 # this repo

# Write LIBERO's config so its first import never asks a question on a compute node.
B="$RQ_SRC/LIBERO-plus/libero/libero"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $B
bddl_files: $B/bddl_files
init_states: $B/init_files
datasets: $B/../datasets
assets: $B/assets
YAML

{
  echo "date: $(date -u +%FT%TZ)"
  echo "lerobot: $(git -C "$RQ_SRC/lerobot" rev-parse HEAD)"
  echo "libero_plus: $(git -C "$RQ_SRC/LIBERO-plus" rev-parse HEAD)"
  python -c "import torch, mujoco; print('torch:', torch.__version__); print('mujoco:', mujoco.__version__)"
} | tee "$RQ_ROOT/versions.txt"
echo "Setup done. Next: bash scripts/hpc/download.sh"
