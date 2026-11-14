"""Scenarios and cells, read from LIBERO-plus's task_classification.json.

In LIBERO-plus every registered task is one perturbed variant, so a scenario is
(suite, task id). A cell groups variants by (suite, perturbation family, level).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FAMILIES = (
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
)


@dataclass(frozen=True, order=True)
class Cell:
    suite: str
    family: str
    level: int


@dataclass(frozen=True, order=True)
class Scenario:
    suite: str
    task_id: int  # the "id" field in task_classification.json (1-based there)
    name: str
    family: str
    level: int

    @property
    def cell(self) -> Cell:
        return Cell(self.suite, self.family, self.level)

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task_id}"

    @property
    def registry_index(self) -> int:
        """0-based index for lerobot's --env.task_ids (checked on the cluster for id 1 -> index 0)."""
        return self.task_id - 1

def load_scenarios(path: str | Path, suites: tuple[str, ...] | None = None) -> tuple[list[Scenario], int]:
    """Returns (scenarios, n_skipped). Entries with no difficulty level are skipped."""
    raw = json.loads(Path(path).read_text())
    out: list[Scenario] = []
    skipped = 0
    for suite, entries in raw.items():
        if suites is not None and suite not in suites:
            continue
        for e in entries:
            if e.get("difficulty_level") is None or e.get("category") not in FAMILIES:
                skipped += 1
                continue
            out.append(Scenario(suite, int(e["id"]), e["name"], e["category"], int(e["difficulty_level"])))
    return sorted(out), skipped


def by_cell(scenarios: list[Scenario]) -> dict[Cell, list[Scenario]]:
    cells: dict[Cell, list[Scenario]] = {}
    for s in scenarios:
        cells.setdefault(s.cell, []).append(s)
    return cells
