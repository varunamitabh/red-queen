"""Instruction text for LIBERO-plus variants.

LIBERO-plus derives a task's instruction from its file name. For every family except
Language Instructions the perturbation suffix is left in, so the policy is told e.g.
"turn on the stove and put the moka pot on it table 1" or "... view 0 0 100 0 0 initstate 441".
`clean_instruction` returns the instruction without that suffix. Language variants take
their text from the BDDL file and are left alone (returns None).
"""
from __future__ import annotations

import re

_SUFFIX = re.compile(
    r"(?:_view_\d+(?:_\d+)*_initstate_\d+(?:_noise_\d+)?|_(?:table|tb|add|light)_\d+|_level\d+_sample\d+)$"
)
_SCENE = re.compile(r"^[A-Z_]+SCENE\d+_")


def clean_instruction(task_name: str) -> str | None:
    if "_language_" in task_name:
        return None
    return " ".join(_SCENE.sub("", _SUFFIX.sub("", task_name)).split("_"))