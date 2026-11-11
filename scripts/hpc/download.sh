#!/usr/bin/env bash
# Run on the LOGIN node after setup_login.sh.  Pass --with-train-data in week 2.
set -euo pipefail
source scripts/hpc/env.sh
python scripts/hpc/download.py "$@"
du -sh "$HF_HOME" "$RQ_SRC" "$RQ_ENV"
