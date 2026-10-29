#!/usr/bin/env bash
# Unified entry point for the correctness tests.
#
# Usage:
#   bash scripts/run_tests.sh aic-aiv            # AIC/AIV ablation equivalence
#   bash scripts/run_tests.sh aiv-bitwise        # bit-wise AIV comparison
#   bash scripts/run_tests.sh task-mapping       # task-mapping equivalence
#
# Any extra arguments are forwarded to the selected test script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
RESULTS="${ROOT}/results"
mkdir -p "${RESULTS}"

if [[ -f "${ROOT}/scripts/env.sh" ]]; then
  source "${ROOT}/scripts/env.sh"
fi

PYTHON_BIN="${ASCEND_SPMM_PYTHON:-python3}"

case "${MODE}" in
  aic-aiv)
    "${PYTHON_BIN}" "${ROOT}/tests/aic_aiv_ablation/correctness.py" "${@:2}"
    ;;
  aiv-bitwise)
    "${PYTHON_BIN}" "${ROOT}/tests/aic_aiv_ablation/bitwise_correctness.py" "${@:2}"
    ;;
  task-mapping)
    "${PYTHON_BIN}" "${ROOT}/tests/task_mapping/correctness.py" "${@:2}"
    ;;
  *)
    echo "Usage: bash scripts/run_tests.sh {aic-aiv|aiv-bitwise|task-mapping} [arguments]" >&2
    echo "Run bash scripts/run_tests.sh --help for the full option list." >&2
    exit 2
    ;;
esac
