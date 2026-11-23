# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""ORL: ontology-structured reinforcement learning (paper Sec. 3).

The landing ontology fixes the computational structure of the PPO actor:

1. Concept-wise inputs, no raw-observation bypass.  The actor reads the 44 semantic features
   phi(O_t) only through the 19 ontology nodes (``NODES``).  Each concept type encodes only
   its own feature slots (Table 2, eq. (4)), and information reaches an action only along
   the declared relations (Table 3, ``RELATIONS``), by one relational attention layer
   applied ``HOPS`` = 3 times (eqs. (5)-(8)).
2. Plane weight sharing (Sec. 3.5).  The x and y horizontal landing planes are two instances
   of the same concepts: they share the node encoders, the relation weights and the lateral
   readout.  The y plane's inputs are aligned to the x plane by declared signs
   (``PLANE_ROLES``): camera y is -body y, so the image coordinate along y is -eta, and a_x
   follows +sin(pitch) while a_y follows -sin(roll).
3. Mirror equivariance (eq. (9)).  Every concept declares its parity under the two mirror
   reflections of the horizontal axes: plane concepts flip under their plane's mirror, the
   view margins of a plane swap, the heading flips under both, everything else is invariant.
   The actor mean is the average over the four reflections, so a mirrored state gets exactly
   the mirrored command.
4. Decision readouts (eqs. (10)-(11)).  The lateral commands are read from the two
   LateralDecision nodes by one shared MLP; the vertical command is
   ``hold(Z) + sigmoid(gate(E)) * descend(Z)`` with Z = VerticalDecision and
   E = DescentEligibility, which gathers the contact condition, view geometry and
   observation quality.

The actor has 2,747 parameters; with the common raw-observation critic and log_std the
model has 4,016.  A checkpoint binds to ``ORL_HASH`` through its arm name ``ORL_ARM``.
"""
from __future__ import annotations

import itertools
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import nn

from .contracts import ACTION_DIMENSION, OBSERVATION_DIMENSION, CurrentSemanticsConfig
from .features import FEATURE_NAMES
from .graph import SELF, GraphSpec, reachable_within
from .models import (MODEL_ARCHITECTURE, CurrentActorCritic, RelationalAttentionLayer,
                     _check_inputs, _linear, _mlp, component_generator)

ORL_VERSIONS = ("orl/1",)
CURRENT_VERSION = "orl/1"
ARM_PREFIX = "orl_ppo"
X, Y = "x", "y"
PLANES = (X, Y)

# Concept -> features of the x plane and of the y plane, each with the sign that aligns the
# y plane with the x plane's convention.  Every plane entry is odd under its plane's mirror.
PLANE_ROLES = MappingProxyType({
    "RelativeGeometry": ((("relative_x", 1), ("relative_y", 1)),),
    "MotionState": ((("relative_vx", 1), ("relative_vy", 1)),
                    (("ugv_vx", 1), ("ugv_vy", 1)),
                    (("own_vx", 1), ("own_vy", 1))),
    "AttitudeState": ((("drone_sinTheta", 1), ("drone_sinPhi", -1)),
                      (("drone_pitchRate", 1), ("drone_rollRate", -1))),
    "ApproachSituation": ((("approach_ex", 1), ("approach_ey", 1)),
                          (("approach_vx", 1), ("approach_vy", 1))),
    "ViewGeometry": ((("view_xi", 1), ("view_eta", -1)),
                     (("view_xi_rate", 1), ("view_eta_rate", -1))),
    "ContactCondition": ((("contact_ex", 1), ("contact_ey", 1)),
                         (("contact_vx", 1), ("contact_vy", 1))),
})
# The image-boundary margins (B(1 + v), B(1 - v)) of the aligned image coordinate v of each
# plane: they swap under that plane's mirror.
MARGINS = MappingProxyType({X: ("view_margin_left", "view_margin_right"),
                            Y: ("view_margin_bottom", "view_margin_top")})
# Shared concepts: (feature, parity), "even" (invariant under both mirrors) or "odd" (flips
# under each mirror: the heading).
SHARED_ROLES = MappingProxyType({
    "ObservationQuality": (("ugv_visionUpdated", "even"), ("ugv_visionAge", "even"),
                           ("drone_navigationValid", "even"), ("drone_navigationAge", "even"),
                           ("view_front_valid", "even")),
    "VerticalState": (("relative_height", "even"), ("drone_vz", "even"),
                      ("contact_vz", "even"), ("contact_tilt", "even"),
                      ("contact_rate", "even"), ("drone_cosPhi", "even"),
                      ("drone_cosTheta", "even"), ("approach_closure", "even")),
    "HeadingState": (("drone_sinPsi", "odd"), ("drone_cosPsi", "even"),
                     ("drone_yawRate", "odd")),
})
DECISIONS = ("LateralDecision", "DescentEligibility", "VerticalDecision")


def _node(concept: str, plane: str | None = None) -> str:
    return concept if plane is None else f"{concept}_{plane}"


def input_dimensions() -> dict[str, int]:
    """Feature slots of each concept type (Table 2)."""
    dims = {c: len(r) for c, r in PLANE_ROLES.items()}
    dims["ViewGeometry"] += 2  # the margin pair
    dims.update({c: len(r) for c, r in SHARED_ROLES.items()})
    return dims


NODES = tuple([_node(c) for c in SHARED_ROLES]
              + [_node(c, p) for p in PLANES for c in (*PLANE_ROLES, "LateralDecision")]
              + ["DescentEligibility", "VerticalDecision"])
RELATIONS = ("qualifiesEstimate", "informsApproach", "affectsVisibility", "conditionsContact",
             "informsLanding", "gatesDescent", "informsVertical", SELF)


def _edges() -> tuple[tuple[str, str, str], ...]:
    """The 42 relation edges of Table 3, then one self-loop per node (61 in all)."""
    edges = []
    for p in PLANES:
        g, m, a = (_node(c, p) for c in ("RelativeGeometry", "MotionState", "AttitudeState"))
        ap, o, c, lat = (_node(n, p) for n in ("ApproachSituation", "ViewGeometry",
                                               "ContactCondition", "LateralDecision"))
        edges += [("ObservationQuality", t, "qualifiesEstimate") for t in (ap, o, c)]
        edges += [(g, ap, "informsApproach"), (m, ap, "informsApproach")]
        edges += [(s, o, "affectsVisibility") for s in (g, m, a, "VerticalState", "HeadingState")]
        edges += [(s, c, "conditionsContact") for s in (g, m, a, "VerticalState")]
        edges += [(s, lat, "informsLanding") for s in (ap, o, c)]
        edges += [(s, "DescentEligibility", "gatesDescent") for s in (c, o)]
    edges += [("ObservationQuality", "DescentEligibility", "gatesDescent")]
    edges += [(s, "VerticalDecision", "informsVertical")
              for s in ("VerticalState", "ObservationQuality", "DescentEligibility")]
    return tuple(edges) + tuple((n, n, SELF) for n in NODES)


GRAPHS = MappingProxyType({
    CURRENT_VERSION: GraphSpec(version=CURRENT_VERSION, nodes=NODES, relations=RELATIONS,
                               edges=_edges(), readout_node="VerticalDecision")})
ORL_GRAPH = GRAPHS[CURRENT_VERSION]
HOPS = 3            # every input concept reaches the vertical command and its LateralDecision
EMBEDDING = 14      # node state width
LATERAL_HIDDEN = 31  # lateral readout width; with EMBEDDING 14 the actor has 2,747 parameters


def _flat_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(FEATURE_NAMES)}


def _check_roles() -> None:
    """Every phi44 feature has exactly one role; each LateralDecision sees its own plane's
    concepts and the shared ones within ``HOPS`` and nothing of the other plane; the vertical
    command sees every concept except the approach (lateral only)."""
    graph = ORL_GRAPH
    used = [f for roles in PLANE_ROLES.values() for pair in roles for f, _ in pair]
    used += [f for pair in MARGINS.values() for f in pair]
    used += [f for roles in SHARED_ROLES.values() for f, _ in roles]
    if sorted(used) != sorted(FEATURE_NAMES):
        raise AssertionError("ORL roles must use every phi44 feature exactly once")
    shared = set(SHARED_ROLES)
    for plane, other in ((X, Y), (Y, X)):
        reach = reachable_within(graph, _node("LateralDecision", plane), HOPS)
        if not ({_node(c, plane) for c in PLANE_ROLES} | shared) <= reach:
            raise AssertionError(f"LateralDecision_{plane} misses a concept within {HOPS} hops")
        if any(n.endswith(f"_{other}") for n in reachable_within(
                graph, _node("LateralDecision", plane), len(NODES))):
            raise AssertionError(f"LateralDecision_{plane} must not read the {other} plane")
    needed = ({_node(c, p) for c in PLANE_ROLES if c != "ApproachSituation" for p in PLANES}
              | shared)
    if not needed <= reachable_within(graph, "VerticalDecision", HOPS):
        raise AssertionError(f"VerticalDecision misses a concept within {HOPS} hops")


_check_roles()


def group_elements() -> tuple[tuple[int, int], ...]:
    """The four reflections (s_x, s_y): identity, mirror x, mirror y, both (G = Z2 x Z2)."""
    return tuple(itertools.product((1, -1), (1, -1)))


def group_matrix(sx: int, sy: int) -> np.ndarray:
    """M_g of eq. (9): the signed permutation of phi44 under the reflection (sx, sy), from the
    declared parities; ``flat @ M.T`` is the reflected decision's phi44."""
    index, m = _flat_index(), np.zeros((len(FEATURE_NAMES), len(FEATURE_NAMES)))
    sign = {X: sx, Y: sy}
    for roles in PLANE_ROLES.values():
        for pair in roles:
            for plane, (name, _) in zip(PLANES, pair):
                m[index[name], index[name]] = sign[plane]
    for plane, (plus, minus) in MARGINS.items():
        if sign[plane] == 1:
            m[index[plus], index[plus]] = m[index[minus], index[minus]] = 1
        else:
            m[index[plus], index[minus]] = m[index[minus], index[plus]] = 1
    for roles in SHARED_ROLES.values():
        for name, parity in roles:
            m[index[name], index[name]] = sx * sy if parity == "odd" else 1
    return m


def spec_dict(version: str = CURRENT_VERSION) -> dict[str, Any]:
    """Everything that defines the network; its digest is ``ORL_HASH``."""
    if version not in ORL_VERSIONS:
        raise ValueError(f"unknown ORL version {version!r}; known {ORL_VERSIONS}")
    return {"version": version, "plane_roles": {k: [list(map(list, p)) for p in v]
                                                for k, v in PLANE_ROLES.items()},
            "margins": dict(MARGINS),
            "shared_roles": {k: [list(r) for r in v] for k, v in SHARED_ROLES.items()},
            "graph": GRAPHS[version].to_dict(), "hops": HOPS, "embedding": EMBEDDING,
            "lateral_hidden": LATERAL_HIDDEN, "group": "Z2 x Z2 mirror average",
            "vertical": "hold + sigmoid(eligibility) * descend", "lateral_bias": False,
            "critic": "common raw21 critic"}


def _digest(version: str) -> str:
    from ..direct_policy.contracts import digest
    return digest(spec_dict(version))


HASHES = MappingProxyType({version: _digest(version) for version in ORL_VERSIONS})
ARMS = MappingProxyType({version: f"{ARM_PREFIX}:{HASHES[version][:16]}"
                         for version in ORL_VERSIONS})
ORL_VERSION = CURRENT_VERSION
ORL_HASH = HASHES[CURRENT_VERSION]
ORL_ARM = ARMS[CURRENT_VERSION]


def version_of(arm: str) -> str:
    """The ORL version an arm name binds to; a stale or foreign hash is refused."""
    for version, name in ARMS.items():
        if arm == name:
            return version
    raise ValueError(f"unknown arm {arm!r}; the ORL arm of this code is {ORL_ARM}")


# ---------------------------------------------------------------------- model

class ORLEncoder(nn.Module):
    """Concept-wise node encoders shared across planes (eq. (4)), the learned initial states
    of the decision nodes, and one relational attention layer applied ``HOPS`` times."""

    def __init__(self, *, seed: int):
        super().__init__()
        dims = input_dimensions()
        self.encoders = nn.ModuleDict({
            concept: _linear(dims[concept], EMBEDDING,
                             component_generator(seed, f"orl.encoder.{concept}"))
            for concept in dims})
        self.decision = nn.Parameter(torch.zeros(len(DECISIONS), EMBEDDING))
        self.layer = RelationalAttentionLayer(EMBEDDING, len(RELATIONS),
                                              generator=component_generator(seed, "orl.layer"))
        for name, values in zip(("source", "target", "relation"), ORL_GRAPH.edge_index()):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.long),
                                 persistent=False)
        index = _flat_index()
        for concept, roles in PLANE_ROLES.items():
            for plane_i, plane in enumerate(PLANES):
                names = [pair[plane_i][0] for pair in roles]
                signs = [float(pair[plane_i][1]) for pair in roles]
                if concept == "ViewGeometry":
                    names += list(MARGINS[plane]); signs += [1.0, 1.0]
                self.register_buffer(f"idx_{concept}_{plane}",
                                     torch.tensor([index[n] for n in names]), persistent=False)
                self.register_buffer(f"sgn_{concept}_{plane}", torch.tensor(signs),
                                     persistent=False)
        for concept, roles in SHARED_ROLES.items():
            self.register_buffer(f"idx_{concept}", torch.tensor([index[n] for n, _ in roles]),
                                 persistent=False)

    def node_states(self, flat44: torch.Tensor) -> torch.Tensor:
        """(B, len(NODES), EMBEDDING) after ``HOPS`` rounds of message passing."""
        batch = flat44.shape[0]
        states = {}
        for concept in SHARED_ROLES:
            states[concept] = torch.tanh(self.encoders[concept](
                flat44[:, getattr(self, f"idx_{concept}")]))
        for concept in PLANE_ROLES:
            for plane in PLANES:
                inputs = flat44[:, getattr(self, f"idx_{concept}_{plane}")] \
                    * getattr(self, f"sgn_{concept}_{plane}")
                states[_node(concept, plane)] = torch.tanh(self.encoders[concept](inputs))
        decision = torch.tanh(self.decision)
        for plane in PLANES:
            states[_node("LateralDecision", plane)] = decision[0].expand(batch, -1)
        states["DescentEligibility"] = decision[1].expand(batch, -1)
        states["VerticalDecision"] = decision[2].expand(batch, -1)
        h = torch.stack([states[n] for n in NODES], dim=1)
        for _ in range(HOPS):
            h = self.layer(h, self.source, self.target, self.relation)
        return h


class ORLReadout(nn.Module):
    """Decision readouts (eqs. (10)-(11)).  Lateral: one bias-free MLP shared by both
    LateralDecision nodes (the mirror average cancels any lateral output bias, so it would be
    a dead parameter).  Vertical: ``hold(Z) + sigmoid(gate(E)) * descend(Z)``."""

    def __init__(self, *, seed: int, output_scale: float):
        super().__init__()
        self.lateral_hidden = _linear(EMBEDDING, LATERAL_HIDDEN,
                                      component_generator(seed, "orl.lateral.hidden"))
        self.lateral_out = _linear(LATERAL_HIDDEN, 1,
                                   component_generator(seed, "orl.lateral.out"), bias=False)
        self.hold = _linear(EMBEDDING, 1, component_generator(seed, "orl.vertical.hold"))
        self.descend = _linear(EMBEDDING, 1, component_generator(seed, "orl.vertical.descend"))
        self.gate = _linear(EMBEDDING, 1, component_generator(seed, "orl.vertical.gate"))
        with torch.no_grad():  # near-zero initial mean, gate half open (MODEL_ARCHITECTURE)
            for layer in (self.lateral_out, self.hold, self.descend):
                layer.weight.mul_(output_scale)
            for layer in (self.hold, self.descend, self.gate):
                layer.bias.zero_()
        self.lateral_index = [NODES.index(_node("LateralDecision", p)) for p in PLANES]
        self.eligibility_index = NODES.index("DescentEligibility")
        self.vertical_index = NODES.index("VerticalDecision")

    def vertical_terms(self, h: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(hold, gate, gated term) of the vertical output, each (B, 1); the output is
        ``hold + gated``."""
        z, e = h[:, self.vertical_index], h[:, self.eligibility_index]
        gate = torch.sigmoid(self.gate(e))
        return self.hold(z), gate, gate * self.descend(z)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        lateral = self.lateral_out(torch.tanh(self.lateral_hidden(h[:, self.lateral_index])))
        hold, _, gated = self.vertical_terms(h)
        return torch.cat((lateral.squeeze(-1), hold + gated), dim=-1)


class ORLActorCritic(CurrentActorCritic):
    """The ORL actor (phi44 through the ontology only, mirror-averaged) with the common raw21
    critic and global log_std of every compared policy."""

    def __init__(self, arm: str, config: CurrentSemanticsConfig, *, seed: int):
        nn.Module.__init__(self)
        self.version = version_of(arm)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative int")
        design = config.model
        self.arm, self.config, self.seed = arm, config, seed
        self.actor_input = "flat44"
        self.encoder = ORLEncoder(seed=seed)
        self.actor_head = ORLReadout(
            seed=seed, output_scale=float(MODEL_ARCHITECTURE["actor_output_init_scale"]))
        self.critic = _mlp(OBSERVATION_DIMENSION, design.critic_hidden, 1,
                           component_generator(seed, "critic"))
        self.log_std = nn.Parameter(torch.full((ACTION_DIMENSION,),
                                               float(design.initial_log_std)))
        self.register_buffer("action_maximum",
                             torch.tensor(config.action.maximum_m_s2, dtype=torch.float32),
                             persistent=False)
        group = group_elements()
        self.register_buffer("group", torch.tensor(
            np.stack([group_matrix(*g) for g in group]), dtype=torch.float32),
            persistent=False)
        # S_g of eq. (9): the horizontal action signs of each reflection.
        self.register_buffer("action_signs", torch.tensor(
            [[sx, sy, 1] for sx, sy in group], dtype=torch.float32), persistent=False)

    def graph_readout(self, nodes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ORL reads phi44; use encoder.node_states(flat44)")

    def unaveraged(self, flat44: torch.Tensor) -> torch.Tensor:
        """f_theta of eq. (9): the ontology network before the mirror average."""
        return self.actor_head(self.encoder.node_states(flat44))

    def _reflected(self, flat44: torch.Tensor) -> torch.Tensor:
        """(members * B, 44): every mirror image of every row, member-major."""
        members, batch = self.group.shape[0], flat44.shape[0]
        return torch.einsum("gij,bj->gbi", self.group, flat44).reshape(members * batch, -1)

    def forward(self, raw21: torch.Tensor, flat44: torch.Tensor,
                nodes: torch.Tensor) -> torch.Tensor:
        """The policy mean mu_t of eq. (9)."""
        _check_inputs(raw21, flat44, nodes)
        batch, members = flat44.shape[0], self.group.shape[0]
        out = self.unaveraged(self._reflected(flat44)).reshape(members, batch, ACTION_DIMENSION)
        return (out * self.action_signs[:, None, :]).mean(0)

    def vertical_terms(self, flat44: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mirror-averaged (B,) hold, gate and gated term of the vertical mean (before tanh),
        as the actor computes them; the vertical mean is ``hold + gated``."""
        batch, members = flat44.shape[0], self.group.shape[0]
        terms = self.actor_head.vertical_terms(self.encoder.node_states(self._reflected(flat44)))
        return {name: term.reshape(members, batch).mean(0)
                for name, term in zip(("hold", "gate", "gated"), terms)}


def orl_model(arm: str, config: CurrentSemanticsConfig, *, seed: int) -> ORLActorCritic:
    return ORLActorCritic(arm, config, seed=seed)
