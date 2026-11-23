#!/usr/bin/env python3
# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""``./run.sh current-semantics-3d``: the three comparisons of the paper.

- learning speed (Sec. 5.1): ``register-speed``, ``speed``, ``speed-report``
- landing success at equal training budgets (Sec. 5.2): ``register-budget``, ``budget``,
  ``budget-report``
- closed-loop Isaac Sim-PX4 evaluation (Sec. 5.3): ``register-isaac-budget``, ``isaac-budget``

A ``register-*`` stage writes the analysis plan's hash before anything trains or flies; the
later stages refuse a plan or contract that differs from it.  ``--dry-run`` prints the
resolved plan and creates nothing.  ``isaac-budget`` starts its own Isaac Sim + PX4 stack and
runs only with ``--allow-isaac``; it never adopts an Isaac that is already running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ontology_rgat.current_semantics.contracts import DEFAULT_CONFIG_PATH, load_config
from ontology_rgat.current_semantics.experiment import (owned, resolved_plan, run_root_for,
                                                       write_json)

SPEED_STAGES = ("register-speed", "speed", "speed-report")
BUDGET_STAGES = ("register-budget", "budget", "budget-report")
ISAAC_BUDGET_STAGES = ("register-isaac-budget", "isaac-budget")
STAGES = (*SPEED_STAGES, *BUDGET_STAGES, *ISAAC_BUDGET_STAGES)
REFUSED_EXIT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="./run.sh current-semantics-3d",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-root", type=Path, default=None,
                        help="results folder (default: results/current_semantics_3d/"
                             "spatial_current_semantics_v1); relative paths resolve against "
                             "the launch directory")
    parser.add_argument("--plan", type=Path, default=None,
                        help="the analysis plan (default: config/current_semantics/"
                             "learning_speed_v1.yaml, equal_budget_v1.yaml or "
                             "isaac_equal_budget_v1.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved plan; create nothing")
    parser.add_argument("--workers", type=int, default=None,
                        help="speed/budget only: rollout processes of this process (results "
                             "do not depend on it; default: the plan's)")
    parser.add_argument("--allow-isaac", action="store_true",
                        help="isaac-budget only: start an owned Isaac Sim + PX4 stack")
    parser.add_argument("--headless", action="store_true", help="isaac-budget only")
    parser.add_argument("--reset-recoveries", type=int, default=None,
                        help="isaac-budget only: owned stack restarts allowed for failures "
                             "before the policy takes over (default 2)")
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.workers is not None and args.stage not in ("speed", "budget"):
        parser.error("--workers applies to --stage speed/budget only")
    if args.stage != "isaac-budget" and (args.allow_isaac or args.headless
                                         or args.reset_recoveries is not None):
        parser.error("--allow-isaac/--headless/--reset-recoveries apply to --stage "
                     "isaac-budget only")


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def _refused(stage: str, reason: str) -> int:
    _print({"stage": stage, "status": "REFUSED", "reason": reason})
    return REFUSED_EXIT


def _report(stage: str, module, out: Path, config_path: Path, plan_path: Path, root: Path,
            keys: tuple[str, ...]) -> int:
    """Write the verdict once every model is complete; another process may hold it."""
    try:
        with owned(out / "report"):
            payload = module.report(config_path, plan_path, root)
            write_json(out / module.RESULT, payload)
    except SystemExit:
        _print({"stage": stage, "status": "REPORT_HELD_ELSEWHERE"})
        return 0
    except ValueError as exc:
        return _refused(stage, str(exc))
    _print({k: payload[k] for k in keys})
    return 0


def _speed(args, config_path: Path, root: Path) -> int:
    """Learning speed (Sec. 5.1): the first update trained at difficulty 1.0 of ORL, MLP PPO
    and untyped-graph PPO on fresh seeds.  Any number of ``speed`` processes share the models;
    whichever finds them all complete writes the verdict."""
    from ontology_rgat.current_semantics import speed
    plan_path = (args.plan or speed.DEFAULT_PATH).expanduser().resolve()
    config = load_config(config_path)
    try:
        plan = speed.load_plan(plan_path, config)
    except FileNotFoundError as exc:
        return _refused(args.stage, f"analysis plan not found ({exc.filename}); the plan "
                        "files are kept by the authors and are not in the public repository")
    except ValueError as exc:
        return _refused(args.stage, str(exc))
    out = speed.out_dir(root)
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, extra={
            "plan_file": str(plan_path), **speed.resolved(plan, config),
            "will_write": [str(out)]}))
        return 0
    if args.stage == "register-speed":
        with owned(out / "registration"):
            status, payload = speed.register(config_path, plan_path, root)
        _print(payload)
        return status
    if args.stage == "speed":
        status, payload = speed.run(config_path, plan_path, root,
                                    log=lambda line: print(line, flush=True),
                                    workers=args.workers)
        if status:
            _print(payload)
            return status
    return _report(args.stage, speed, out, config_path, plan_path, root,
                   ("protocol", "complete", "pending", "arms", "verdict"))


def _budget(args, config_path: Path, root: Path) -> int:
    """Equal training budgets (Sec. 5.2): ten-cell landing success of the same three policies
    after 100, 200 and 600 updates.  Any number of ``budget`` processes share the models;
    whichever finds them all complete writes the verdict."""
    from ontology_rgat.current_semantics import budget
    plan_path = (args.plan or budget.DEFAULT_PATH).expanduser().resolve()
    config = load_config(config_path)
    try:
        plan = budget.load_plan(plan_path, config)
    except FileNotFoundError as exc:
        return _refused(args.stage, f"analysis plan not found ({exc.filename}); the plan "
                        "files are kept by the authors and are not in the public repository")
    except ValueError as exc:
        return _refused(args.stage, str(exc))
    out = budget.out_dir(root)
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, extra={
            "plan_file": str(plan_path), **budget.resolved(plan, config),
            "will_write": [str(out)]}))
        return 0
    if args.stage == "register-budget":
        with owned(out / "registration"):
            status, payload = budget.register(config_path, plan_path, root)
        _print(payload)
        return status
    if args.stage == "budget":
        status, payload = budget.run(config_path, plan_path, root,
                                     log=lambda line: print(line, flush=True),
                                     workers=args.workers)
        if status:
            _print(payload)
            return status
    return _report(args.stage, budget, out, config_path, plan_path, root,
                   ("protocol", "complete", "pending", "verdict"))


def _isaac_budget(args, config_path: Path, root: Path) -> int:
    """Closed loop (Sec. 5.3): every 600-update model of the equal-budget comparison on the
    same Isaac Sim-PX4 cases inside an owned stack, with the identical cases also flown on the
    local plant."""
    from dataclasses import asdict
    from ontology_rgat.current_semantics import budget
    from ontology_rgat.current_semantics import cells
    config = load_config(config_path)
    iplan_path = (args.plan or budget.ISAAC_DEFAULT_PATH).expanduser().resolve()
    try:
        plan = budget.load_plan(budget.DEFAULT_PATH, config)
        iplan = budget.load_isaac_plan(iplan_path, config, plan)
    except FileNotFoundError as exc:
        return _refused(args.stage, f"analysis plan not found ({exc.filename}); the plan "
                        "files are kept by the authors and are not in the public repository")
    except ValueError as exc:
        return _refused(args.stage, str(exc))
    out = budget.isaac_dir(root)
    summary = {"plan_file": str(iplan_path), "plan": iplan.to_dict(),
               "models": len(budget.isaac_keys(plan)),
               "episodes": len(iplan.seeds) * len(budget.isaac_keys(plan))}
    if args.stage == "register-isaac-budget":
        if args.dry_run:
            _print(resolved_plan(config_path, args.stage, root, extra={
                **summary, "starts_isaac": False,
                "will_write": [str(out / budget.ISAAC_REGISTRATION)]}))
            return 0
        with owned(out / "registration"):
            status, payload = budget.register_isaac(config_path, budget.DEFAULT_PATH,
                                                    iplan_path, root)
        _print({k: v for k, v in payload.items() if k != "checkpoints"})
        return status
    problem = budget.check_isaac_registration(config, plan, iplan, root)
    if problem:
        return _refused(args.stage, problem)
    previous = out / budget.ISAAC_RESULT
    if previous.exists() and json.loads(previous.read_text(encoding="utf-8")).get(
            "protocol_complete"):
        return _refused(args.stage, f"this registration's Isaac flight is complete ({previous})")
    extra = {**summary, "out_dir": str(out), "resume": (out / "episodes.jsonl").exists(),
             "starts_owned_stack": True, "adopts_running_isaac": False,
             "headless": args.headless,
             "reset_recoveries": 2 if args.reset_recoveries is None else args.reset_recoveries}
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, extra={
            **extra, "will_write": [str(previous), str(out / "episodes.jsonl")]}))
        return 0
    from ontology_rgat.current_semantics.backend import (CurrentSpatialBackend,
                                                         make_isaac_backend)
    from ontology_rgat.live_stack import live_stack
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    with owned(out):
        local_path = out / budget.LOCAL_REFERENCE
        if local_path.exists():
            local = [json.loads(line) for line in
                     local_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:  # deterministic, so its order relative to the flight does not matter
            print("isaac-budget: local reference on the identical cases", flush=True)
            local = budget.isaac_local_reference(config_path, plan, iplan, root)
            local_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in local),
                                  encoding="utf-8")
        profile = Path(deployment_profile("spatial-reference/12")["path"])
        wrappers: list = []

        def make(cell):  # one Isaac flight, one wrapper per condition
            factors, dropout = cells.cell_conditions(config, asdict(cell), plan.dropout_start_s)
            if not wrappers:
                backend = make_isaac_backend(config, allow_isaac=True, split=cell.split,
                                             factors=factors, dropout=dropout)
            else:
                backend = CurrentSpatialBackend(wrappers[0].flight, config, name="isaac_v12",
                                                split=cell.split, factors=factors,
                                                dropout=dropout)
            wrappers.append(backend)
            return backend

        stack_root = out / "isaac_stack"
        stack_root.mkdir(parents=True, exist_ok=True)
        with live_stack(stack_root, adopt=False, headless=extra["headless"],
                        schema="spatial-reference/12",
                        reset_recoveries=extra["reset_recoveries"], profile_path=profile):
            try:
                flown = budget.isaac_run(config, plan, iplan, root, make, out,
                                         log=lambda line: print(line, flush=True),
                                         resume=extra["resume"])
            finally:
                error = None
                for backend in sorted(wrappers, key=lambda w: not w.active):
                    try:
                        backend.close()
                    except Exception as exc:  # keep closing; report the first failure
                        error = error or exc
                if error is not None:
                    raise error
        flight = wrappers[0].flight if wrappers else None
        payload = budget.isaac_report(config, plan, iplan, flown, local,
                                      backend=getattr(flight, "name", None),
                                      profile_sha256=getattr(flight, "profile_sha256", None))
        write_json(previous, payload)
    status = 0 if payload["protocol_complete"] else 1
    print(f"{args.stage}: wrote {previous} (exit {status})")
    return status


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    config_path = args.config.expanduser().resolve()
    requested = None if args.run_root is None else args.run_root.expanduser()
    if requested is not None and not requested.is_absolute():
        requested = Path.cwd() / requested
    try:
        root = run_root_for(load_config(config_path), requested)
    except ValueError as exc:
        parser.error(str(exc))
    if args.stage in SPEED_STAGES:
        return _speed(args, config_path, root)
    if args.stage in BUDGET_STAGES:
        return _budget(args, config_path, root)
    if args.stage == "isaac-budget" and not args.allow_isaac:
        _print({"stage": args.stage, "status": "REFUSED", "allow_isaac": False,
                "reason": "the Isaac evaluation starts an owned Isaac Sim + PX4 stack and runs "
                          "only with --allow-isaac; no stack was contacted"})
        return REFUSED_EXIT
    return _isaac_budget(args, config_path, root)


if __name__ == "__main__":
    sys.exit(main())
