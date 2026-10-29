# Ascend-SpMM

AscendC kernels for sparse matrix-dense matrix multiplication (SpMM) on Ascend NPUs, together with a PyTorch NPU extension and numerical correctness tests.

This repository contains the operator implementation and test suite. It does not include the dataset archives, benchmarking/profiling harnesses, or paper plotting scripts.

## Contents

```text
kernels/                      AscendC custom operator package (op_host / op_kernel / cmake)
extension/                    PyTorch NPU extension (C++ bindings, preprocessing helpers)
common/                       Shared ACL / torch-npu helper headers
spmm_npu.py                   Python API exposed by the extension
setup.py                      Wheel build for the extension
scripts/build.sh              Build the operators and install the extension
scripts/run_tests.sh          Unified entry point for the correctness tests
tests/aic_aiv_ablation/       AIC pipeline and AIV variant equivalence tests
tests/task_mapping/           Task-mapping strategy equivalence tests
tests/data/                   SuiteSparse download and CSR conversion helpers
```

## Operators

The AscendC package registers the following operators, all exposed through `spmm_npu`:

| Operator | Description |
| --- | --- |
| `spmm_ascend_cube` | Mixed AIC/AIV SpMM with a hierarchy-aware AIC pipeline (`g=6`) |
| `spmm_ascend_cube_aic_without_pipeline` | AIC-only ablation without the hierarchy-aware pipeline (`g=1`) |
| `spmm_csr` | AIV-only naive CSR SpMM reference |
| `spmm_ascend_cube_aiv_mb` | AIV variants sharing the mixed-kernel scaffold |
| `spmm_csr_aiv_mb_batch` | AIV batched processing over multiple buffers |
| `spmm_cube` | Cube-only SpMM baseline |
| `preprocess` | CSR to packed-format conversion |
| `csr_to_bcsr`, `csr_to_ascend_cube` | Layout conversions used by the kernels |
| `block_level_split` | Block-level AIC/AIV work partitioning |

`spmm_npu.py` additionally provides `pack_node_tensor` / `restore_node_tensor`, which apply the graph permutation `P` and its transpose once during preprocessing rather than inside the SpMM loop.

## Requirements

- Ascend Atlas 800I A2 host with `ascend910b` devices, 20 AICs and 40 AIVs
- CANN toolkit 8.5.0 or later
- Python 3 with a PyTorch build and matching `torch-npu`
- CMake 3.19 or later, GNU Make, `setuptools`, `wheel`
- For the tests only: NumPy, SciPy (`pip install -r tests/requirements.txt`)

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
bash scripts/build.sh
source scripts/env.sh
```

`scripts/build.sh` builds the custom operator package into `kernels/build_out`, installs it under `.artifact_opp/vendors/customize`, builds the `spmm_npu` wheel and installs it into the active Python environment. It writes `scripts/env.sh`, which exports `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`; source it before running any test.

Use a dedicated Python environment for the extension:

```bash
export ASCEND_SPMM_PYTHON=/path/to/torch-npu/bin/python
```

## Data

The tests consume CSR structure files (`.pt`) rather than Matrix Market files. `tests/data/suitesparse_urls.txt` lists the 820 SuiteSparse matrices; download and convert the paper subset with:

```bash
bash tests/data/download_data.sh --output datasets --extract
python3 tests/data/prepare_data.py \
  --input datasets \
  --output datasets_pt \
  --manifest results/dataset_manifest.csv \
  --primary-only \
  --paper-filter
```

`--paper-filter` keeps square matrices with at least 10,000 rows, 10,000 columns and 100,000 nonzeros. The converter coalesces duplicates, drops explicit zeros, sorts CSR columns and records every decision in the manifest.

## Tests

All tests compare the NPU output against a reference and report bit-wise equality, mismatch count and the maximum absolute error. They write a JSON report under `results/` and exit non-zero when any comparison fails.

`tests/task_mapping/correctness.py` compares the proposed task mapping and three alternative strategies (`naive`, `neutronsparse`, `block-split`) against `spmm_csr` on the original CSR:

```bash
bash scripts/run_tests.sh task-mapping --name mip1 --dim 128 --dtype fp16
bash scripts/run_tests.sh task-mapping --name mip1 --dim 128 --dim 256 --dtype fp16 \
  --method ours --method naive
```

`tests/aic_aiv_ablation/correctness.py` checks the AIC pipeline against its `g=1` ablation and every AIV variant against naive CSR:

```bash
bash scripts/run_tests.sh aic-aiv --name mip1 --dim 128 --dtype fp16
```

`tests/aic_aiv_ablation/bitwise_correctness.py` verifies that all AIV feature widths produce bit-identical results:

```bash
bash scripts/run_tests.sh aiv-bitwise --name mip1 \
  --dim 16 --dim 32 --dim 64 --dim 128 --dim 256 --dtype fp16
```

Common options for the Python tests:

| Option | Meaning |
| --- | --- |
| `--dataset-dir` | CSR `.pt` directory, default `./datasets_pt` |
| `--name` | Filter datasets by parent directory or file stem |
| `--index`, `--start`, `--stop`, `--limit` | Select a subset of the discovered datasets |
| `--dim` | Feature width, repeatable; must be a positive multiple of 16 |
| `--dtype` | `fp16` or `fp32`, default `fp16` |
| `--seed` | Base random seed for matrix values and dense operands |
| `--atol`, `--rtol` | Comparison tolerances |
| `--output` | JSON report path |

Tolerances default to `2e-2` for FP16 and `1e-4` for FP32. The AIV bit-wise test uses zero tolerance, so any mismatch is a failure.

## Constraints

- Kernels target `ascend910b` with 20 AICs and 40 AIVs.
- Feature width must be 16-column aligned and at most 1024 for FP16 or 512 for FP32.
- Indices are 32-bit, so a matrix must satisfy `rows + nnz < 2^31`.
- The supplied tests use 16-row windows.
