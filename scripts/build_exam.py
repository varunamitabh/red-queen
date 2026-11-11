"""Build the sealed-exam manifest and print its hash. Run once, before round 0.

  python scripts/build_exam.py            # writes configs/exam_manifest.json
"""
import argparse
import json
from pathlib import Path

from redqueen.exam import build_manifest, manifest_hash
from redqueen.scenarios import load_scenarios

LOOP = ("Objects Layout", "Robot Initial States", "Language Instructions", "Light Conditions")
SEALED = ("Camera Viewpoints", "Background Textures")  # Sensor Noise is held as a spare
SOURCE = "sylvestf/LIBERO-plus@4976dc30028e805ff8094b55501d532c48fec182 task_classification.json"

p = argparse.ArgumentParser()
p.add_argument("--classification", default="configs/task_classification.json")
p.add_argument("--suites", nargs="+", default=["libero_10"])
p.add_argument("--heldout-levels", nargs="+", type=int, default=[5])
p.add_argument("--n-per-tier", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="configs/exam_manifest.json")
a = p.parse_args()

out = Path(a.out)
if out.exists():
    raise SystemExit(f"{out} already exists. The exam is built once; delete it by hand only before round 0.")
scenarios, skipped = load_scenarios(a.classification, tuple(a.suites))
m = build_manifest(scenarios, LOOP, SEALED, tuple(a.heldout_levels), a.n_per_tier, seed=a.seed, source=SOURCE)
out.write_text(json.dumps(m, indent=1, sort_keys=True))
Path(str(out) + ".sha256").write_text(manifest_hash(m) + "\n")
print(f"scenarios {len(scenarios)} (skipped {skipped})")
print({k: len(v) for k, v in m["tiers"].items()}, "validation", len(m["validation"]), "loop_pool", len(m["loop_pool"]))
print("sha256", manifest_hash(m))
