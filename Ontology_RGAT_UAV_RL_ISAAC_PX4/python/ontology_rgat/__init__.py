# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Ontology-structured reinforcement learning (ORL) for UAV landing on a moving platform.

Isaac Sim owns the physics, PX4 owns the estimator and the attitude loop, the
ROS 2 gateway owns the wire format, and this package owns the experiment: the
local training plant (``spatial``), the landing-task contract (``landing``), the
policies with their training and evaluation (``current_semantics``), and the
owned Isaac/PX4 stack they fly on (``stack``, ``live_stack``, ``bridge``).

The only entry point is the repository's ``run.sh current-semantics-3d``, which
runs ``python/run_current_semantics_3d.py``.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"
__author__ = "Jun Suk Kim (김준석)"
__maintainer__ = "Jun Suk Kim (김준석)"
__email__ = "junsuk55944587@gmail.com"
__license__ = "Apache-2.0"
