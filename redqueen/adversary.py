"""Scenario samplers: a Thompson-sampling adversary and the uniform baseline.

The adversary is rewarded for VALID failures only. A scenario the validity gate
rejects earns reward 0, exactly like a success, so the adversary cannot raise its
score by drifting toward impossible scenes and stops spending budget on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scenarios import Cell, Scenario, by_cell


@dataclass
class CellStats:
    failures: int = 0
    successes: int = 0
    invalid: int = 0

    @property
    def runs(self) -> int:
        return self.failures + self.successes


@dataclass
class _Sampler:
    pool: list[Scenario]
    seed: int = 0
    stats: dict[Cell, CellStats] = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._remaining = {c: list(v) for c, v in sorted(by_cell(self.pool).items())}
        for v in self._remaining.values():
            self.rng.shuffle(v)
        self.stats = {c: CellStats() for c in self._remaining}
        self._cell_of = {s.key: s.cell for s in self.pool}

    def _open_cells(self) -> list[Cell]:
        return [c for c, v in self._remaining.items() if v]

    def _choose_cell(self, cells: list[Cell]) -> Cell:
        raise NotImplementedError

    def propose(self, n: int) -> list[Scenario]:
        """Next n scenarios, without replacement. Fewer if the pool runs out."""
        out: list[Scenario] = []
        while len(out) < n:
            cells = self._open_cells()
            if not cells:
                break
            out.append(self._remaining[self._choose_cell(cells)].pop())
        return out

    def update(self, scenario_key: str, valid: bool, failed: bool) -> None:
        st = self.stats[self._cell_of[scenario_key]]
        if not valid:
            st.invalid += 1
        elif failed:
            st.failures += 1
        else:
            st.successes += 1

    def valid_failures(self) -> int:
        return sum(s.failures for s in self.stats.values())

    def failing_cells(self) -> int:
        return sum(1 for s in self.stats.values() if s.failures > 0)


class UniformSampler(_Sampler):
    """Baseline: each open cell is equally likely."""

    def _choose_cell(self, cells: list[Cell]) -> Cell:
        return cells[int(self.rng.integers(len(cells)))]


class ThompsonAdversary(_Sampler):
    """Beta-Bernoulli Thompson sampling over cells; reward = valid failure."""

    def _choose_cell(self, cells: list[Cell]) -> Cell:
        draws = [
            self.rng.beta(1 + self.stats[c].failures, 1 + self.stats[c].successes + self.stats[c].invalid)
            for c in cells
        ]
        return cells[int(np.argmax(draws))]
