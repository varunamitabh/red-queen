"""Alarm threshold with a false-alarm guarantee, and lead time."""
from __future__ import annotations

import math

import numpy as np


def conformal_threshold(success_episode_max_scores: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal threshold from calibration episodes that SUCCEEDED.

    Each input is one successful episode's maximum score. An alarm fires when the
    running score exceeds the returned value. For a new successful episode that is
    exchangeable with the calibration set, P(false alarm) <= alpha.
    """
    s = np.sort(np.asarray(success_episode_max_scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("need at least one successful calibration episode")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")  # too few calibration episodes for this alpha
    return float(s[k - 1])


def first_alarm(scores: np.ndarray, threshold: float) -> int | None:
    """First step whose score exceeds the threshold, or None."""
    hits = np.flatnonzero(np.asarray(scores, dtype=float) > threshold)
    return int(hits[0]) if hits.size else None


def lead_time(alarm_step: int | None, irrecoverable_step: int | None) -> int | None:
    """Steps of warning before the failure became irrecoverable.

    Measured against the irrecoverable step from privileged state, never against
    episode end (which is always the time limit and would flatter every detector).
    Negative = the alarm came too late. None = no alarm, or the episode had no
    irrecoverable step.
    """
    if alarm_step is None or irrecoverable_step is None:
        return None
    return irrecoverable_step - alarm_step
