# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""``./run.sh current-semantics-3d``: dry runs create nothing, the Isaac stage is refused
without --allow-isaac, and options are accepted only by the stages they belong to.

Every invocation writes only under tmp_path; no stage here trains, and none can reach a
flight stack.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ontology_rgat.current_semantics import budget, speed
from ontology_rgat.current_semantics.experiment import LOCK_NAME, owned


PROJECT = Path(__file__).resolve().parents[2]
REPOSITORY = PROJECT.parent
RUN_SH = REPOSITORY / "run.sh"
ROUTE = "current-semantics-3d"
# The analysis plan files are kept by the authors and are not in the public repository.
needs_plans = pytest.mark.skipif(
    not all(p.exists() for p in (speed.DEFAULT_PATH, budget.DEFAULT_PATH,
                                 budget.ISAAC_DEFAULT_PATH)),
    reason="analysis plan files are kept by the authors")


def run(*args, timeout=180, cwd=PROJECT):
    return subprocess.run(["bash", str(RUN_SH), ROUTE, *args], capture_output=True,
                          text=True, timeout=timeout, cwd=cwd)


def payload(done):
    start = done.stdout.index("{")
    return json.loads(done.stdout[start:done.stdout.rindex("}") + 1])


def test_help_lists_the_stages_and_a_bare_call_starts_nothing():
    done = subprocess.run(["bash", str(RUN_SH), "--help"], capture_output=True, text=True,
                          timeout=30)
    assert done.returncode == 0
    for text in (ROUTE, "register-speed", "budget-report", "isaac-budget", "--allow-isaac",
                 "starts nothing"):
        assert text in done.stdout
    bare = subprocess.run(["bash", str(RUN_SH)], capture_output=True, text=True, timeout=30)
    assert bare.returncode == 0 and "Usage" in bare.stdout
    unknown = subprocess.run(["bash", str(RUN_SH), "system"], capture_output=True, text=True,
                             timeout=30)
    assert unknown.returncode == 2


@needs_plans
@pytest.mark.parametrize("stage", ["register-speed", "speed-report", "register-budget",
                                   "budget-report", "register-isaac-budget"])
def test_dry_run_prints_the_plan_and_creates_nothing(tmp_path, stage):
    root = tmp_path / "x"
    done = run("--stage", stage, "--dry-run", "--run-root", str(root))
    assert done.returncode == 0, done.stderr
    plan = payload(done)
    assert plan["schema_id"] == "spatial_current_semantics_v1" and plan["dry_run"]
    assert plan["manifest"]["calibration_hash"] and plan["manifest"]["algorithm_hash"]
    assert not root.exists()


@pytest.mark.parametrize("dry", [False, True])
def test_isaac_budget_without_allow_isaac_is_refused_and_contacts_nothing(tmp_path, dry):
    root = tmp_path / "x"
    done = run("--stage", "isaac-budget", "--run-root", str(root),
               *(["--dry-run"] if dry else []))
    assert done.returncode == 3
    assert "--allow-isaac" in payload(done)["reason"]
    assert not root.exists()


@pytest.mark.parametrize("args", [("--stage", "register-speed", "--workers", "2"),
                                  ("--stage", "budget", "--allow-isaac"),
                                  ("--stage", "speed", "--headless"),
                                  ("--stage", "register-isaac-budget", "--reset-recoveries", "1"),
                                  ("--stage", "develop")])
def test_stage_specific_options_and_unknown_stages_are_rejected(tmp_path, args):
    done = run(*args, "--run-root", str(tmp_path / "x"))
    assert done.returncode == 2
    assert not (tmp_path / "x").exists()


@needs_plans
def test_relative_run_root_resolves_against_the_launch_directory(tmp_path):
    done = run("--stage", "register-speed", "--dry-run", "--run-root", "rel/x", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert payload(done)["run_root"] == str(tmp_path / "rel" / "x")
    assert not (tmp_path / "rel").exists()


def test_lock_refuses_a_live_owner_and_clears_a_dead_one(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    lock = root / LOCK_NAME
    lock.write_text(f"{os.getpid()} now\n")
    with pytest.raises(SystemExit):
        with owned(root):
            pass
    assert lock.read_text().startswith(str(os.getpid()))

    child = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                           capture_output=True, text=True, check=True)
    lock.write_text(f"{child.stdout.strip()} then\n")
    with owned(root):
        assert lock.read_text().split()[0] == str(os.getpid())
    assert not lock.exists()

    lock.write_text("garbage\n")
    with owned(root):
        pass
    assert not lock.exists()
