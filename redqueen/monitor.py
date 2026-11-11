"""Runtime failure scores computed from the policy's own outputs.

Nothing here is ever used as a training signal for the policy. These scores only
rank scenarios and raise alarms; see the anti-gaming table in the design doc.
"""
from __future__ import annotations

import numpy as np

from .stats import auroc


def chunk_disagreement(chunks: np.ndarray) -> float:
    """Variance across K action chunks sampled for the same observation.

    chunks: (K, H, D) = samples x horizon x action dim. Returns the variance across
    samples, averaged over horizon and action dim. High = the policy is unsure.
    """
    chunks = np.asarray(chunks, dtype=float)
    if chunks.ndim != 3 or chunks.shape[0] < 2:
        raise ValueError("need (K>=2, H, D) sampled chunks")
    return float(chunks.var(axis=0).mean())


def temporal_inconsistency(prev_chunk: np.ndarray, new_chunk: np.ndarray, executed: int) -> float:
    """Mean L2 distance between consecutive chunks on the steps both predicted.

    prev_chunk and new_chunk are (H, D). `executed` steps of prev_chunk were run
    before new_chunk was predicted, so prev[executed:] overlaps new[:H-executed].
    """
    prev_chunk = np.asarray(prev_chunk, dtype=float)
    new_chunk = np.asarray(new_chunk, dtype=float)
    overlap = prev_chunk.shape[0] - executed
    if overlap <= 0:
        raise ValueError("no overlap: executed >= chunk horizon")
    diff = prev_chunk[executed:] - new_chunk[:overlap]
    return float(np.linalg.norm(diff, axis=-1).mean())


def running_max(scores: np.ndarray) -> np.ndarray:
    """Prefix score: the alarm statistic at step t uses only steps <= t."""
    return np.maximum.accumulate(np.asarray(scores, dtype=float))


def timestep_baseline(length: int, horizon: int) -> np.ndarray:
    """The cheat detector: knows only how much time has passed."""
    return np.arange(1, length + 1, dtype=float) / horizon


def episode_auroc(episode_scores: list[np.ndarray], failed: np.ndarray) -> float:
    """Naive metric: AUROC of each episode's final running-max score.

    Inflated by the fixed-horizon leak, because failed episodes run longer.
    """
    final = np.array([running_max(s)[-1] for s in episode_scores])
    return auroc(final, failed)


def time_matched_auroc(episode_scores: list[np.ndarray], failed: np.ndarray, t: int) -> float:
    """Controlled metric: AUROC at step t among episodes still running at step t.

    Every episode compared has seen exactly t+1 steps, so elapsed time carries no
    information and the timestep baseline scores 0.5 by construction.
    """
    failed = np.asarray(failed, dtype=bool)
    alive = [i for i, s in enumerate(episode_scores) if len(s) > t]
    if not alive:
        return float("nan")
    at_t = np.array([running_max(episode_scores[i][: t + 1])[-1] for i in alive])
    return auroc(at_t, failed[alive])
