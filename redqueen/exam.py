"""The sealed exam: built once, hashed, and never shown to the adversary or trainer.

Tiers, in LIBERO-plus terms:
  A  held-out variants from cells the loop does see (same family and level)
  B  difficulty levels the loop never sees, within in-loop families
  C  perturbation families the loop never sees
`validation` is a separate split for checkpoint selection. `loop_pool` is
everything the adversary and the trainer are allowed to touch.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .scenarios import Scenario, by_cell


def _stratified(scenarios: list[Scenario], n: int, rng: np.random.Generator) -> list[Scenario]:
    """Up to n scenarios, drawn round-robin across cells so no cell dominates."""
    buckets = [list(v) for _, v in sorted(by_cell(scenarios).items())]
    for b in buckets:
        rng.shuffle(b)
    out: list[Scenario] = []
    while len(out) < n and any(buckets):
        for b in buckets:
            if b and len(out) < n:
                out.append(b.pop())
    return out


def build_manifest(
    scenarios: list[Scenario],
    loop_families: tuple[str, ...],
    sealed_families: tuple[str, ...],
    heldout_levels: tuple[int, ...],
    n_per_tier: int = 300,
    n_validation: int = 150,
    seed: int = 0,
    source: str = "",
) -> dict:
    if set(loop_families) & set(sealed_families):
        raise ValueError("a family cannot be both in the loop and sealed")
    rng = np.random.default_rng(seed)
    scenarios = sorted(scenarios)
    in_loop = [s for s in scenarios if s.family in loop_families and s.level not in heldout_levels]
    tier_b_all = [s for s in scenarios if s.family in loop_families and s.level in heldout_levels]
    tier_c_all = [s for s in scenarios if s.family in sealed_families]

    tier_a = _stratified(in_loop, n_per_tier, rng)
    taken = {s.key for s in tier_a}
    validation = _stratified([s for s in in_loop if s.key not in taken], n_validation, rng)
    taken |= {s.key for s in validation}
    loop_pool = [s for s in in_loop if s.key not in taken]

    keys = lambda xs: sorted(s.key for s in xs)  # noqa: E731
    return {
        "version": 1,
        "seed": seed,
        "source": source,
        "loop_families": sorted(loop_families),
        "sealed_families": sorted(sealed_families),
        "heldout_levels": sorted(heldout_levels),
        "tiers": {
            "A": keys(tier_a),
            "B": keys(_stratified(tier_b_all, n_per_tier, rng)),
            "C": keys(_stratified(tier_c_all, n_per_tier, rng)),
        },
        "validation": keys(validation),
        "loop_pool": keys(loop_pool),
    }


def manifest_hash(manifest: dict) -> str:
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def sealed_keys(manifest: dict) -> set[str]:
    return {k for tier in manifest["tiers"].values() for k in tier}


def assert_no_leak(manifest: dict, keys, what: str) -> None:
    """Raise if anything the loop is about to use belongs to the sealed exam."""
    leaked = sealed_keys(manifest) & set(keys)
    if leaked:
        raise RuntimeError(f"{what} touches {len(leaked)} sealed scenarios, e.g. {sorted(leaked)[:3]}")
    stray = set(keys) - set(manifest["loop_pool"]) - set(manifest["validation"])
    if stray:
        raise RuntimeError(f"{what} uses {len(stray)} scenarios outside the loop pool, e.g. {sorted(stray)[:3]}")


class ExamLedger:
    """Append-only log of every time the sealed exam is scored."""

    def __init__(self, path: str | Path, manifest: dict):
        self.path = Path(path)
        self.hash = manifest_hash(manifest)

    def record(self, round_idx: int, policy: str, tier: str, note: str = "") -> int:
        row = {"time": time.time(), "manifest": self.hash, "round": round_idx, "policy": policy, "tier": tier, "note": note}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return self.count()

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text().splitlines() if line.strip())
