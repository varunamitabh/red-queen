# Source this in every shell and job:  source scripts/hpc/env.sh
# RQ_ROOT must be a directory OUTSIDE your home (home has a 40G quota and is full).
: "${RQ_ROOT:?set RQ_ROOT first, e.g. export RQ_ROOT=/path/to/group/storage/$USER/redqueen}"
case "$RQ_ROOT" in "$HOME"*) echo "RQ_ROOT is inside \$HOME, which is full. Pick group or scratch storage." >&2; return 1 2>/dev/null || exit 1;; esac

export RQ_ENV="$RQ_ROOT/env"            # conda environment
export RQ_SRC="$RQ_ROOT/src"            # lerobot and LIBERO-plus checkouts
export RQ_OUT="$RQ_ROOT/outputs"        # rollouts, checkpoints, logs

# Keep every cache off the home directory.
export HF_HOME="$RQ_ROOT/hf"
export PIP_CACHE_DIR="$RQ_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$RQ_ROOT/cache/conda"
export XDG_CACHE_HOME="$RQ_ROOT/cache/xdg"
export TORCH_HOME="$RQ_ROOT/cache/torch"

# LIBERO reads its paths from here instead of ~/.libero (and will not prompt).
export LIBERO_CONFIG_PATH="$RQ_ROOT/libero_config"
export PYTHONPATH="$RQ_SRC/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"    # NVIDIA EGL is present on the GPU nodes
export MAGICK_HOME="$RQ_ENV"            # ImageMagick from conda, used by LIBERO-plus
export TOKENIZERS_PARALLELISM=false

# Compute nodes have no internet. Jobs set RQ_OFFLINE=1 before sourcing this.
if [ "${RQ_OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

if [ -d "$RQ_ENV" ]; then
  eval "$(conda shell.bash hook)" && conda activate "$RQ_ENV"
fi
