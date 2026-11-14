"""Print 0-based task indices for lerobot's --env.task_ids, drawn only from the loop pool.

  python scripts/sample_ids.py --n 20 --seed 0      ->  [12,480,...]

Sealed tiers and the validation split are never sampled. The draw is spread evenly
over (family, level) cells so a small sample still covers every loop family.
"""
import argparse
import json
import random
from pathlib import Path

from redqueen.scenarios import by_cell, load_scenarios

p = argparse.ArgumentParser()
p.add_argument("--manifest", default="configs/exam_manifest.json")
p.add_argument("--classification", default="configs/task_classification.json")
p.add_argument("--n", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--suite", help="diagnostics only: sample a suite that is NOT in the exam manifest")
p.add_argument("--levels", nargs="+", type=int)
p.add_argument("--families", nargs="+")
p.add_argument("--table", action="store_true", help="print index, family, level, name instead of the id list")
a = p.parse_args()

m = json.loads(Path(a.manifest).read_text())
pool = set(m["loop_pool"])
suites = tuple(sorted({k.split("/")[0] for k in pool}))
if a.suite:
    if a.suite in suites:
        raise SystemExit(f"{a.suite} is covered by the exam manifest; sample it through the loop pool only")
    scenarios, _ = load_scenarios(a.classification, (a.suite,))
    eligible = scenarios
else:
    scenarios, _ = load_scenarios(a.classification, suites)
    eligible = [s for s in scenarios if s.key in pool]
if a.levels:
    eligible = [s for s in eligible if s.level in a.levels]
if a.families:
    eligible = [s for s in eligible if s.family in a.families]
cells = by_cell(eligible)
rng = random.Random(a.seed)
queues = []
for c in sorted(cells):
    q = sorted(cells[c])
    rng.shuffle(q)
    queues.append(q)
picked = []
while len(picked) < a.n and any(queues):
    for q in queues:
        if q and len(picked) < a.n:
            picked.append(q.pop())
picked.sort()
if a.table:
    for s in picked:
        print(s.registry_index, s.family, s.level, s.name, sep="\t")
else:
    print("[" + ",".join(str(s.registry_index) for s in picked) + "]")