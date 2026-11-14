"""One line per lerobot eval_info.json: successes, Wilson interval, seconds per episode."""
import argparse
import json

from redqueen.stats import wilson_interval

p = argparse.ArgumentParser()
p.add_argument("paths", nargs="+")
p.add_argument("--label", default="")
a = p.parse_args()
for path in a.paths:
    info = json.load(open(path))
    o = info["overall"]
    n, k = o["n_episodes"], o.get("n_success", round(o["pc_success"] * o["n_episodes"] / 100))
    lo, hi = wilson_interval(k, n)
    print(f"RESULT {a.label or path}: {k}/{n} success ({100*lo:.0f}%..{100*hi:.0f}%), {o['eval_s']/n:.0f}s per episode")
    failed = [t["task_id"] for t in info["per_task"] if not all(t["metrics"]["successes"])]
    print(f"RESULT {a.label or path}: failed task indices {failed}")