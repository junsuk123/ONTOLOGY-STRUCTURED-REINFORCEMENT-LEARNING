# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""The omnidirectional approach scenario (paper Sec. 4.1, Fig. 4).

The base ``LocalBackend`` draws the pad's travel heading from U(-0.2, 0.2) rad
and the initial horizontal offset from a 0.4 m square, so every episode is
"pad driving +x".  This module draws both factors over the full circle:

    A  bearing of the PAD seen from the drone at reset (own-minus-pad = -r (cos A, sin A))
    B  heading of the pad's travel

each in ``bins`` equal sectors centred on 0, 360/bins, ...  Training and the
trained-direction evaluation draw (A bin, B bin) uniformly from the
combinations NOT held out; the held-out-direction evaluation draws only from
``held_out``.  The held-out set is a permutation (every A bin and every B bin
is held out exactly once, so both marginals stay uniform in training) that is
not a cyclic shift, so every relative-geometry class (B - A) keeps training
examples.

The plant, the sensor and every other draw are the base ``LocalBackend``'s;
``ScenarioLocalBackend`` only replaces the offset and the heading after its
reset (curriculum shrink and camera-footprint limit applied the same way) and
re-primes the first frame from the new geometry.  The draw uses its own seeded
stream, so the plant's stream is not moved.

Isaac Sim flies the SAME draw: ``isaac_scenario`` encodes it in the scenario
name (``spatial.scenarios.encode_current_omni``), Isaac's pad_motion.py and
landing_world.py apply it, and ``check_isaac_reset`` refuses an episode whose
acknowledged deck heading, CA acceleration, entry offset or entry yaw differs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..spatial.environment import LocalBackend

SCENARIO_VERSION = "spatial-current-scenario/1"
# The paper's results use version /1 (the base contract).  Two variants stay loadable:
# /2 draws the same way but ``held_out`` is one whole relative-geometry class B - A (a
# cyclic shift); /3 keeps the /1-/2 draw and adds the pad's initial speed
# (``CouplingScenario``), and its held-out set may be any balanced permutation.
SCENARIO_VERSIONS = (SCENARIO_VERSION, "spatial-current-scenario/2",
                     "spatial-current-scenario/3")
SPLITS = ("in_distribution", "composition")
_STREAM = 0x5CE0


@dataclass(frozen=True)
class CompositionScenario:
    """task.scenario: the two composition factors and the held-out combinations."""
    bins: int
    offset_radius_m: tuple[float, float]
    held_out: tuple[tuple[int, int], ...]  # (bearing bin, heading bin)
    version: str = SCENARIO_VERSION

    @property
    def allowed(self) -> tuple[tuple[int, int], ...]:
        return tuple((a, b) for a in range(self.bins) for b in range(self.bins)
                     if (a, b) not in self.held_out)

    @property
    def sector_rad(self) -> float:
        return 2.0 * math.pi / self.bins


@dataclass(frozen=True)
class EpisodeScenario:
    """One episode's composition draw (evaluator-only: never part of the observation)."""
    split: str
    bearing_bin: int
    heading_bin: int
    bearing_rad: float
    heading_rad: float
    offset_radius_m: float

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "bearing_bin": self.bearing_bin,
                "heading_bin": self.heading_bin, "bearing_deg": math.degrees(self.bearing_rad),
                "heading_deg": math.degrees(self.heading_rad),
                "offset_radius_m": self.offset_radius_m}


@dataclass(frozen=True)
class CouplingScenario(CompositionScenario):
    """Version /3: the /1-/2 draw plus the pad's initial speed v0 ~ U(pad_speed_m_s) in
    place of the plant's U(0.3, 0.8) m/s (its CA phase a2/t1/t2 is the plant's).  A
    separate class, so the /1-/2 contracts keep their exact field set and task hashes."""
    pad_speed_m_s: tuple[float, float] = (0.3, 0.8)


def pad_speed(spec: Any, seed: int, plant_v0: float, difficulty: float) -> float | None:
    """A ``CouplingScenario``'s initial pad speed for this seed (None for the other
    versions).  Below difficulty 1 it is blended with the plant's own curriculum-eased
    v0, exactly as the plant blends."""
    speeds = getattr(spec, "pad_speed_m_s", None)
    if speeds is None:
        return None
    drawn = float(np.random.default_rng([int(seed), _STREAM, 0x5B]).uniform(*speeds))
    return drawn if difficulty >= 1.0 else (1 - difficulty) * plant_v0 + difficulty * drawn


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sector(angle: float, bins: int) -> int:
    """The bin whose sector (centred on k * 360/bins) holds ``angle``."""
    width = 2.0 * math.pi / bins
    return int(math.floor((angle + width / 2.0) / width)) % bins


def draw(spec: CompositionScenario, seed: int, split: str) -> EpisodeScenario:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    rng = np.random.default_rng([int(seed), _STREAM])
    combos = spec.allowed if split == "in_distribution" else spec.held_out
    a_bin, b_bin = combos[int(rng.integers(len(combos)))]
    width = spec.sector_rad
    bearing = wrap(a_bin * width + rng.uniform(-width / 2.0, width / 2.0))
    heading = wrap(b_bin * width + rng.uniform(-width / 2.0, width / 2.0))
    radius = float(rng.uniform(*spec.offset_radius_m))
    return EpisodeScenario(split=split, bearing_bin=a_bin, heading_bin=b_bin,
                           bearing_rad=bearing, heading_rad=heading, offset_radius_m=radius)


NOMINAL_PAD_ACCELERATION_M_S2 = (0.1, 0.4)   # LocalBackend.reset's a2 draw
NOMINAL_OPTICAL_NOISE_M = 0.02                # LocalBackend.measure's solve noise


@dataclass(frozen=True)
class PlantFactors:
    """Evaluation-only departures of the plant (paper Table 4, moving-platform and
    image-noise rows).
    ``pad_acceleration_m_s2`` redraws the pad's CA-phase acceleration from that range
    (nominal 0.1-0.4); ``optical_noise_scale`` multiplies the 0.02 m solve noise by
    adding an independent N(0, 0.02 sqrt(k^2 - 1)) per frame from its own stream."""
    pad_acceleration_m_s2: tuple[float, float] | None = None
    optical_noise_scale: float = 1.0

    @property
    def nominal(self) -> bool:
        return self.pad_acceleration_m_s2 is None and self.optical_noise_scale == 1.0


def pad_acceleration(seed: int, factors: PlantFactors) -> float | None:
    """The Motion factor's CA-phase acceleration for this seed; None keeps the seed's own."""
    if factors.pad_acceleration_m_s2 is None:
        return None
    return float(np.random.default_rng([int(seed), _STREAM, 0xA2]).uniform(
        *factors.pad_acceleration_m_s2))


# Isaac's entry yaw is the world's 0 plus the profile's seeded misalignment
# (benchmark.initial_conditions.platform_yaw_misalignment_deg, +-5 deg).
ISAAC_ENTRY_YAW_MAX_RAD = math.radians(5.0)


def isaac_scenario(spec: CompositionScenario, seed: int, split: str,
                   factors: PlantFactors = PlantFactors()) -> tuple[EpisodeScenario, str]:
    """This episode's draw and the Isaac scenario name that carries it."""
    from ..spatial.scenarios import encode_current_omni
    if getattr(spec, "pad_speed_m_s", None) is not None:
        raise ValueError("scenario/3's initial pad speed is not carried by the Isaac scenario "
                         "name yet (spatial_current_omni/1); extend pad_motion.py first")
    if factors.optical_noise_scale != 1.0:
        raise ValueError("the optical-noise factor is applied by the local plant only")
    drawn = draw(spec, seed, split)
    return drawn, encode_current_omni(drawn.heading_rad, drawn.bearing_rad,
                                      drawn.offset_radius_m, pad_acceleration(seed, factors))


def check_isaac_reset(detail: Any, seed: int, name: str) -> dict[str, Any]:
    """Isaac's reset acknowledgement against the draw it was sent; raises on any
    difference (the stack then flew something else and is not trusted)."""
    from ..spatial.scenarios import current_omni_motion, parse_current_omni
    expected = parse_current_omni(name)
    if expected is None:
        raise ValueError(f"{name!r} is not a current-omni scenario")
    if not isinstance(detail, dict) or not isinstance(detail.get("pad"), dict):
        raise RuntimeError("Isaac reset acknowledgement carries no deck manifest; the world "
                           "predates the current-omni scenario")
    pad = detail["pad"]
    motion = current_omni_motion(seed, expected)
    offset = np.asarray(detail.get("entry_offset_pad_m", ()), dtype=float)
    want = -expected["radius_m"] * np.array([math.cos(expected["bearing_rad"]),
                                             math.sin(expected["bearing_rad"])])
    yaw = float(detail.get("entry_yaw_enu_rad", math.nan))
    problems = []
    if pad.get("current_omni") != expected or pad.get("benchmark_scenario") != name:
        problems.append(f"deck scenario {pad.get('current_omni')!r}")
    if pad.get("heading_rad") != expected["heading_rad"]:
        problems.append(f"deck heading {pad.get('heading_rad')!r}")
    if (pad.get("spatial_motion") or {}).get("a2") != motion.a2:
        problems.append(f"deck a2 {(pad.get('spatial_motion') or {}).get('a2')!r} != {motion.a2!r}")
    if offset.shape != (3,) or not np.allclose(offset[:2], want, rtol=0.0, atol=1e-9):
        problems.append(f"entry offset {offset.tolist()} != {want.tolist()}")
    if not abs(yaw) <= ISAAC_ENTRY_YAW_MAX_RAD + 1e-12:
        problems.append(f"entry yaw {yaw!r} rad")
    if problems:
        raise RuntimeError("Isaac did not fly the requested current-omni draw: "
                           + "; ".join(problems))
    return {"isaac_scenario": name, "entry_offset_pad_m": offset.tolist(),
            "entry_yaw_enu_rad": yaw, "pad_a2_m_s2": motion.a2}


class ScenarioLocalBackend(LocalBackend):
    """The local plant with this module's offset and heading (and, for evaluation
    cells only, the ``PlantFactors``)."""

    def __init__(self, cfg: Any, *, difficulty: float = 1.0, scenario: CompositionScenario,
                 split: str, factors: PlantFactors = PlantFactors()):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not factors.nominal and difficulty != 1.0:
            raise ValueError("plant factors are evaluation conditions: difficulty must be 1.0")
        if not factors.optical_noise_scale >= 1.0:
            raise ValueError("optical_noise_scale must be >= 1 (it adds noise)")
        super().__init__(cfg, difficulty=difficulty)
        self.scenario_spec, self.split, self.factors = scenario, split, factors
        self.episode_scenario: EpisodeScenario | None = None
        self.noise_rng: np.random.Generator | None = None

    def reset(self, seed):
        super().reset(seed)
        drawn = draw(self.scenario_spec, seed, self.split)
        radius = drawn.offset_radius_m
        if self.difficulty < 1.0:
            # LocalBackend.reset's easy-rung shrink and camera-footprint limit, on
            # the new offset (its own offset draw is discarded).
            radius *= 0.15 + 0.85 * self.difficulty
            depth = max(self.position[2] - 0.16, 0.01)
            radius = min(radius, 0.4 * depth * min(np.tan(np.asarray(self.cfg.fov) / 2)))
        self.position[:2] = self.pad[:2] - radius * np.array(
            [math.cos(drawn.bearing_rad), math.sin(drawn.bearing_rad)])
        self.direction = np.array([math.cos(drawn.heading_rad), math.sin(drawn.heading_rad), 0.0])
        speed = pad_speed(self.scenario_spec, seed, self.v0, self.difficulty)
        if speed is not None:
            self.v0 = speed
        self.pad_velocity = self.direction * self.v0
        if self.factors.pad_acceleration_m_s2 is not None:
            self.a2 = pad_acceleration(seed, self.factors)
        self.noise_rng = np.random.default_rng([int(seed), _STREAM, 0x40])
        self.episode_scenario = drawn
        # Re-prime the handover frame from the new geometry (LocalBackend.reset
        # primed it from the discarded offset).
        self.camera_time = -math.inf
        self.camera_sequence = 0
        self.camera_pending = []
        self.optical_position = None
        self.optical_confidence = 0.0
        return self.measure(), self.truth()

    def _receive_camera(self, frame):
        k = self.factors.optical_noise_scale
        if k != 1.0 and frame[2] is not None and self.noise_rng is not None:
            extra = self.noise_rng.normal(0.0, NOMINAL_OPTICAL_NOISE_M * math.sqrt(k * k - 1.0), 3)
            frame = (frame[0], frame[1], np.asarray(frame[2]) + extra, *frame[3:])
        return super()._receive_camera(frame)
