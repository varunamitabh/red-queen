# Red Queen (working name)

A co-evolving, gaming-resistant evaluation loop for a vision-language-action policy
(SmolVLA on LIBERO / LIBERO-plus). Private work in progress. Nothing here is published.

Design doc: https://claude.ai/artifact/4A39qR7LhubhgySRjw1nwN

## Status

| Part | State |
| --- | --- |
| Core logic (`redqueen/`): stats, monitor scores, conformal alarm, adversary, sealed exam | Written, unit-tested locally |
| HPC scripts (`scripts/hpc/`) | Written, **not yet run on the cluster** |
| Rollout logging wrapper, validity gate, oracle | Not started (needs the simulator working first) |
| Any result | None yet |

## Laptop

```bash
pip install -e ".[dev]"
pytest -q
```

## HPC, first time

Home is nearly full (40G quota), so everything lives under `RQ_ROOT` on group or scratch storage.

```bash
export RQ_ROOT=/path/outside/home/redqueen      # add this line to ~/.bashrc
bash scripts/hpc/setup_login.sh                 # login node: conda env, lerobot, LIBERO-plus
bash scripts/hpc/download.sh                    # login node: checkpoint, backbone, assets
sbatch --account=<acct> --partition=<gpu partition> --gpus=1 \
  --export=ALL,RQ_ROOT=$RQ_ROOT scripts/hpc/smoke.sbatch
```

The smoke job checks headless rendering, the GPU, and one offline SmolVLA episode, and
prints how long it took. Paste its `slurm-rq-smoke-*.out` back before going further.

## Sealed exam

`python scripts/build_exam.py` writes `configs/exam_manifest.json` and its SHA-256.
Run it once, commit both files, and never rebuild after round 0 starts.
With the defaults (LIBERO-10, level 5 held out) it gives tiers A/B/C of 300/247/300
variants, 150 for validation and 665 in the loop pool.

`configs/task_classification.json` is copied unchanged from
sylvestf/LIBERO-plus at commit 4976dc3.
