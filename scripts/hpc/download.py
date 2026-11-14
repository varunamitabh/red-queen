"""Fetch everything a compute node will need, since it has no internet.

Run on the login node:  python scripts/hpc/download.py [--with-train-data]
"""
import argparse
import json
import os
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

from redqueen.assets import SENTINEL, extract_assets

assets_dir = Path(os.environ["RQ_SRC"]) / "LIBERO-plus" / "libero" / "libero" / "assets"
if (assets_dir / SENTINEL).exists():
    print("LIBERO-plus assets already in place")
else:
    z = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset")
    print("extracting", z, "->", assets_dir, "(this takes a few minutes)")
    print("extracted", extract_assets(z, assets_dir), "files")

if a.with_train_data:
    for repo in ("lerobot/libero", "lerobot/libero_plus"):
        print(repo, "->", snapshot_download(repo, repo_type="dataset"))
