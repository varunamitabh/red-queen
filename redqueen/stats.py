"""Statistics used in every results table. Pure numpy, no simulator needed."""
from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate. Returns (low, high)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_discordant(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Counts of scenarios where only policy A succeeded and only policy B succeeded.

    a and b are boolean success arrays over the SAME scenarios in the same order.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays must cover the same scenarios")
    return int(np.sum(a & ~b)), int(np.sum(~a & b))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve; label 1 = failure. Ties get average ranks.

    Returns nan when only one class is present.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_lift_ci(
    fail_adv: np.ndarray, fail_uniform: np.ndarray, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Adversary lift = valid-failure rate of the adversary / that of uniform sampling.

    Inputs are per-rollout booleans (True = valid failure). Returns (lift, low, high)
    with a 95% percentile bootstrap interval. Resamples with zero uniform failures
    are dropped, so the interval is conditional on a defined ratio.
    """
    fa = np.asarray(fail_adv, dtype=float)
    fu = np.asarray(fail_uniform, dtype=float)
    if fu.mean() == 0:
        return (float("inf"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ra = rng.choice(fa, size=(n_boot, len(fa))).mean(axis=1)
    ru = rng.choice(fu, size=(n_boot, len(fu))).mean(axis=1)
    ok = ru > 0
    ratios = ra[ok] / ru[ok]
    low, high = np.percentile(ratios, [2.5, 97.5])
    return (float(fa.mean() / fu.mean()), float(low), float(high))
