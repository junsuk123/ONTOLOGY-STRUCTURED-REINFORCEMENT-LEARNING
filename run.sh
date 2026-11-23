#!/usr/bin/env bash
# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# Entry point of the experiments in the paper (Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md).
# A bare ./run.sh only prints this usage; it starts nothing.
set -Eeuo pipefail
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
export PYTHONUNBUFFERED=1

usage() {
  echo 'Usage:'
  echo '  ./run.sh current-semantics-3d --stage STAGE [options]'
  echo '      register-speed | speed | speed-report          learning speed (paper Sec. 5.1)'
  echo '      register-budget | budget | budget-report       equal training budgets (Sec. 5.2)'
  echo '      register-isaac-budget | isaac-budget           Isaac Sim-PX4 closed loop (Sec. 5.3);'
  echo '                                                     isaac-budget only with --allow-isaac'
  echo '  ./run.sh current-semantics-3d --help               every stage and option'
  echo
  echo 'A bare ./run.sh starts nothing.'
}

case "${1:-}" in
  ""|--help|-h)
    usage
    ;;
  current-semantics-3d)
    shift
    # Training is bit-identical only single-threaded.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    exec python3 "$project_root/python/run_current_semantics_3d.py" "$@"
    ;;
  *)
    echo "unknown route '$1'; the only route is current-semantics-3d" >&2
    usage >&2
    exit 2
    ;;
esac
