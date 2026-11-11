"""Fetch everything a compute node will need, since it has no internet.

Run on the login node:  python scripts/hpc/download.py [--with-train-data]
"""
import argparse
import json
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
p.add_argument("--with-train-data", action="store_true", help="also fetch the fine-tuning datasets (week 2, large)")
a = p.parse_args()

policy_dir = snapshot_download(a.policy)
print("policy:", policy_dir)

# The checkpoint names its vision-language backbone; the tokenizer/processor load from it.
cfg = json.loads((Path(policy_dir) / "config.json").read_text())
backbone = cfg.get("vlm_model_name")
if backbone:
    print("backbone:", snapshot_download(backbone))
else:
    print("WARNING: no vlm_model_name in the policy config; check what it loads at start-up.")

assets_root = Path(os.environ["RQ_SRC"]) / "LIBERO-plus" / "libero" / "libero"
if (assets_root / "assets" / "new_objects").exists():
    print("LIBERO-plus assets already extracted")
else:
    z = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset")
    print("extracting", z, "->", assets_root)
    with zipfile.ZipFile(z) as f:
        f.extractall(assets_root)

if a.with_train_data:
    for repo in ("lerobot/libero", "lerobot/libero_plus"):
        print(repo, "->", snapshot_download(repo, repo_type="dataset"))
