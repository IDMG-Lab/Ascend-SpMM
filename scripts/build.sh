#!/usr/bin/env bash
# Build the AscendC custom operators and install the spmm_npu Python extension.
#
# Requirements: CANN toolkit (8.5.0 or later), CMake >= 3.19, GNU Make,
# PyTorch with a matching torch-npu build, setuptools, wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/kernels/build_out"
INSTALL_ROOT="${ASCEND_SPMM_OPP_ROOT:-${ROOT}/.artifact_opp}"
cd "${ROOT}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  if [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    ASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME}"
  elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  else
    echo "ERROR: source the CANN set_env.sh file or set ASCEND_HOME_PATH." >&2
    exit 2
  fi
fi
export ASCEND_HOME_PATH

for command_name in cmake python3; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${command_name}" >&2
    exit 2
  fi
done

python3 - <<'PY'
import importlib.util
missing = [
    name
    for name in ("torch", "torch_npu", "setuptools", "wheel")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
PY

JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${INSTALL_ROOT}" "${ROOT}/results"

cmake -S "${ROOT}/kernels" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SOURCE_PACKAGE=TRUE \
  -DENABLE_BINARY_PACKAGE=TRUE \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DENABLE_TEST=FALSE \
  -Dvendor_name=customize \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
  -DASCEND_PYTHON_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
  -DENABLE_CROSS_COMPILE=FALSE \
  -DASCEND_PACK_SHARED_LIBRARY=FALSE

cmake --build "${BUILD_DIR}" --target binary -j"${JOBS}"
cmake --build "${BUILD_DIR}" --target package -j"${JOBS}"

RUN_PACKAGE="$(find "${BUILD_DIR}" -type f -name 'custom_opp_*.run' -print | sort | head -n 1)"
if [[ -z "${RUN_PACKAGE}" ]]; then
  echo "ERROR: custom operator package was not generated." >&2
  exit 3
fi

bash "${RUN_PACKAGE}" --quiet --install-path="${INSTALL_ROOT}"
CUSTOM_OPP="${INSTALL_ROOT}/vendors/customize"
if [[ ! -d "${CUSTOM_OPP}" ]]; then
  echo "ERROR: custom operator installation is missing: ${CUSTOM_OPP}" >&2
  exit 3
fi

export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${CUSTOM_OPP}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

rm -rf "${ROOT}/build" "${ROOT}/dist"
python3 "${ROOT}/setup.py" build bdist_wheel
WHEEL="$(find "${ROOT}/dist" -maxdepth 1 -type f -name '*.whl' -print | sort | head -n 1)"
if [[ -z "${WHEEL}" ]]; then
  echo "ERROR: Python wheel was not generated." >&2
  exit 3
fi
python3 -m pip install --force-reinstall --no-deps "${WHEEL}"

cat > "${ROOT}/scripts/env.sh" <<EOF
#!/usr/bin/env bash
export ASCEND_SPMM_ROOT="${ROOT}"
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP}\${ASCEND_CUSTOM_OPP_PATH:+:\${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${CUSTOM_OPP}/op_api/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
chmod +x "${ROOT}/scripts/env.sh"

python3 - <<'PY'
import torch
import torch_npu
import spmm_npu

required = (
    "spmm_ascend_cube_aic_without_pipeline",
    "spmm_ascend_cube_aiv_mb",
    "spmm_csr_aiv_mb_batch",
)
missing = [name for name in required if not hasattr(spmm_npu, name)]
if missing:
    raise RuntimeError("Missing ablation operators: " + ", ".join(missing))
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("spmm_npu loaded from:", spmm_npu.__file__)
print("Ablation operators:", ", ".join(required))
PY

echo "Build complete. Run: source scripts/env.sh"
