# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "ontology_rgat_px4"))
# The learner package. Added after the gateway path so a name that exists in
# both resolves to the gateway's, which is what the ROS node actually imports.
sys.path.insert(1, str(ROOT / "python"))
