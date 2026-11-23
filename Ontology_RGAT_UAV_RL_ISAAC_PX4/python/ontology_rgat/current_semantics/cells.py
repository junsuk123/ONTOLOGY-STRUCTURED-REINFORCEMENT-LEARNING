# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""The ten local evaluation cells (paper Table 4) and the plan-file helpers.

Five categories, each with a nominal (``*_id``) and an extended (``*_ood``) condition:
approach direction, platform motion, visibility, image noise and their combination.  Every
cell is flown with the deterministic policy at difficulty 1.0
(``config/current_semantics/evaluation_cells_v1.yaml``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import datetime as dt
import math
from pathlib import Path
from typing import Any

import yaml

from .contracts import PROJECT_ROOT, CurrentSemanticsConfig

DEFAULT_CELLS_PATH = PROJECT_ROOT / "config" / "current_semantics" / "evaluation_cells_v1.yaml"
# The name under which the cells were fixed before any compared model trained; the
# equal-budget plan names it in ``cells_from``.
CELLS_PROTOCOL = "development/1"
CELLS = ("composition_id", "composition_ood", "motion_id", "motion_ood", "dropout_id",
         "dropout_ood", "noise_id", "noise_ood", "combined_id", "combined_ood")


@dataclass(frozen=True)
class EvaluationCell:
    name: str
    split: str
    pad_acceleration_m_s2: tuple[float, float] | None = None
    optical_noise_scale: float = 1.0
    dropout_s: tuple[float, float] | None = None


@dataclass(frozen=True)
class CellPlan:
    protocol: str
    dropout_start_s: tuple[float, float]   # an exogenous dropout starts U(start) s after handover
    cells: tuple[EvaluationCell, ...]


# ------------------------------------------------------------------- loading

def _section(data: Any, keys: tuple[str, ...], where: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown, missing = set(data) - set(keys), set(keys) - set(data)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return data


def _number(value: Any, where: str, *, low: float = -math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value < low:
        raise ValueError(f"{where} must be a finite number >= {low}")
    return float(value)


def _range(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{where} must be [low, high]")
    low, high = (_number(v, where, low=0.0) for v in value)
    if not low < high:
        raise ValueError(f"{where} must have low < high")
    return low, high


def _seeds(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value):
        raise ValueError(f"{where} must be a non-empty list of non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} must be distinct")
    return tuple(value)


def load_cells(path: str | Path, config: CurrentSemanticsConfig) -> CellPlan:
    """Load the cells and check that each one moves exactly the factor it names: the nominal
    cells keep the trained ranges, each extended cell departs from them in one factor, and
    the combined extended cell combines all three."""
    data = _section(yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                    ("protocol", "dropout_start_s", "cells"), "evaluation_cells")
    if data["protocol"] != CELLS_PROTOCOL:
        raise ValueError(f"evaluation_cells.protocol must be {CELLS_PROTOCOL!r}")
    cells_data = _section(data["cells"], CELLS, "evaluation_cells.cells")
    blackout = config.g4.training_randomization.vision_blackout
    cells = []
    for name in CELLS:
        spec = cells_data[name]
        if not isinstance(spec, dict) or "split" not in spec or set(spec) - {
                "split", "pad_acceleration_m_s2", "optical_noise_scale", "dropout_s"}:
            raise ValueError(f"evaluation_cells.cells.{name} is malformed")
        cells.append(EvaluationCell(
            name=name, split=spec["split"],
            pad_acceleration_m_s2=(_range(spec["pad_acceleration_m_s2"], name)
                                   if "pad_acceleration_m_s2" in spec else None),
            optical_noise_scale=_number(spec.get("optical_noise_scale", 1.0), name, low=1.0),
            dropout_s=_range(spec["dropout_s"], name) if "dropout_s" in spec else None))
    cells = tuple(cells)
    by = {c.name: c for c in cells}
    from .scenario import NOMINAL_PAD_ACCELERATION_M_S2
    trained_dropout = (blackout.duration_min_s, blackout.duration_max_s)
    expect_split = {name: ("composition" if name == "composition_ood" else "in_distribution")
                    for name in CELLS}
    for c in cells:
        if c.split != expect_split[c.name]:
            raise ValueError(f"cell {c.name} must use the {expect_split[c.name]} split")
    for name in ("composition_id", "composition_ood", "motion_id", "noise_id"):
        if (by[name].pad_acceleration_m_s2, by[name].optical_noise_scale, by[name].dropout_s) \
                != (None, 1.0, None):
            raise ValueError(f"cell {name} must be the nominal condition")
    if by["dropout_id"].dropout_s != trained_dropout or by["combined_id"] != replace(
            by["dropout_id"], name="combined_id"):
        raise ValueError("dropout_id and combined_id must carry the trained blackout range "
                         f"{trained_dropout} and nothing else")
    motion, drop, noise = (by["motion_ood"].pad_acceleration_m_s2, by["dropout_ood"].dropout_s,
                           by["noise_ood"].optical_noise_scale)
    if (motion is None or motion[0] < NOMINAL_PAD_ACCELERATION_M_S2[1]
            or (by["motion_ood"].optical_noise_scale, by["motion_ood"].dropout_s) != (1.0, None)):
        raise ValueError("motion_ood must move only the pad acceleration, beyond the trained range")
    if (drop is None or drop[0] < trained_dropout[1]
            or (by["dropout_ood"].pad_acceleration_m_s2, by["dropout_ood"].optical_noise_scale)
            != (None, 1.0)):
        raise ValueError("dropout_ood must move only the dropout, beyond the trained range")
    if not noise > 1.0 or (by["noise_ood"].pad_acceleration_m_s2, by["noise_ood"].dropout_s) \
            != (None, None):
        raise ValueError("noise_ood must move only the optical noise")
    if by["combined_ood"] != EvaluationCell("combined_ood", "in_distribution", motion, noise,
                                            drop):
        raise ValueError("combined_ood must combine exactly the three extended settings")
    return CellPlan(protocol=data["protocol"],
                    dropout_start_s=_range(data["dropout_start_s"], "dropout_start_s"),
                    cells=cells)


# ---------------------------------------------------------------- evaluation

def cell_conditions(config: CurrentSemanticsConfig, cell: dict[str, Any],
                    start: tuple[float, float]) -> tuple[Any, Any]:
    """(PlantFactors, evaluation dropout or None) of one cell (``asdict(EvaluationCell)``):
    the one definition the local cells and the Isaac evaluation share."""
    from .scenario import PlantFactors
    blackout = config.g4.training_randomization.vision_blackout
    dropout = (None if cell["dropout_s"] is None else replace(
        blackout, probability=1.0, start_min_s=start[0], start_max_s=start[1],
        duration_min_s=cell["dropout_s"][0], duration_max_s=cell["dropout_s"][1]))
    factors = PlantFactors(
        pad_acceleration_m_s2=(None if cell["pad_acceleration_m_s2"] is None
                               else tuple(cell["pad_acceleration_m_s2"])),
        optical_noise_scale=cell["optical_noise_scale"])
    return factors, dropout


def _cell_work(task: tuple) -> dict[str, Any]:
    """One deterministic evaluation episode in a worker process."""
    from .pilot import _WORKER, run_episode
    arm, state, seed, cell, start, action_seed = task
    config = _WORKER["config"]
    factors, dropout = cell_conditions(config, cell, start)
    episode = run_episode(config, arm, state, seed=seed, difficulty=1.0, deterministic=True,
                          action_seed=action_seed, split=cell["split"], factors=factors,
                          dropout=dropout)
    return {key: episode.get(key) for key in ("seed", "refused", "status", "success",
                                              "decisions", "return", "scenario")}


def unsafe(status: str | None) -> bool:
    from ..spatial.environment import Evaluator
    return status in Evaluator.UNSAFE_STATUSES


def summarize_cell(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    flown = [e for e in episodes if not e.get("refused")]
    statuses: dict[str, int] = {}
    for e in flown:
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    n = len(flown)
    return {"episodes": n, "refused": len(episodes) - n,
            "success": sum(e["status"] == "SUCCESS" for e in flown) / n if n else None,
            "unsafe": sum(unsafe(e["status"]) for e in flown) / n if n else None,
            "timeout": (statuses.get("TASK_TIMEOUT") or 0) / n if n else None,
            "statuses": statuses}


STATUS = "status.json"  # per-model phase: training, evaluating, done


def _status(model_root: Path, phase: str, **detail: Any) -> None:
    from .experiment import write_json
    model_root.mkdir(parents=True, exist_ok=True)
    write_json(model_root / STATUS, {"phase": phase, "utc": dt.datetime.now(
        dt.timezone.utc).isoformat(timespec="seconds"), **detail})
