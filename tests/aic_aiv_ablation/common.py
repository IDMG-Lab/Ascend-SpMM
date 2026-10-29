#!/usr/bin/env python3
"""Shared dataset and invocation helpers for the Figure 10/11 ablations."""

import hashlib
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import spmm_npu


AIC_CORES = 20
AIV_CORES = 40
METHODS = (
    "aic-pipeline",
    "aic-without-pipeline",
    "aiv-naive",
    "aiv-mb",
    "aiv-mb-batch",
    "aiv-round-robin",
)
METHOD_LABELS = {
    "aic-pipeline": "AIC hierarchy-aware pipeline (g=6)",
    "aic-without-pipeline": "AIC without hierarchy-aware pipeline (g=1)",
    "aiv-naive": "AIV Naive CSR",
    "aiv-mb": "AIV Multiple Buffer",
    "aiv-mb-batch": "AIV Multiple Buffer + Batched Process (batch=8, buffers=64)",
    "aiv-round-robin": "AIV Round-Robin Ring Buffer (L=64)",
}
ROOT = Path(__file__).resolve().parents[2]


def load_payload(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {"shape", "crow_indices", "col_indices"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"missing payload fields: {', '.join(missing)}")
    rows, cols = map(int, payload["shape"])
    crow = payload["crow_indices"].to(torch.int64).contiguous()
    col = payload["col_indices"].to(torch.int64).contiguous()
    if rows <= 0 or cols <= 0:
        raise ValueError(f"invalid matrix shape: {(rows, cols)}")
    if crow.ndim != 1 or crow.numel() != rows + 1:
        raise ValueError("crow_indices must contain rows + 1 entries")
    if col.ndim != 1 or int(crow[0]) != 0 or int(crow[-1]) != col.numel():
        raise ValueError("invalid CSR pointer or column array")
    if bool((crow[1:] < crow[:-1]).any().item()):
        raise ValueError("crow_indices must be nondecreasing")
    if not col.numel():
        raise ValueError("empty matrices are not supported")
    if int(col.min()) < 0 or int(col.max()) >= cols:
        raise ValueError("col_indices contains an out-of-range index")
    if max(rows, cols, col.numel()) > torch.iinfo(torch.int32).max:
        raise ValueError("matrix exceeds the 32-bit NPU index contract")
    stem = path.stem
    fallback = stem[:-4] if stem.endswith("_csr") else stem
    name = str(payload.get("name") or fallback)
    return name, rows, cols, crow, col


def select_dataset(dataset_dir: Path, name: str, index: int):
    paths = sorted(
        dataset_dir.rglob("*.pt"),
        key=lambda item: item.relative_to(dataset_dir).as_posix(),
    )
    if name:
        paths = [
            path for path in paths
            if name == path.parent.name or name in path.stem
        ]
        if len(paths) != 1:
            raise ValueError(f"--name selected {len(paths)} datasets; expected exactly one")
        return paths[0]
    if index < 0 or index >= len(paths):
        raise ValueError(
            f"--index {index} is outside the discovered range [0, {len(paths)})"
        )
    return paths[index]


def stable_seed(base_seed: int, relative_path: str) -> int:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def make_start(pointer: torch.Tensor, cores: int) -> torch.Tensor:
    total = int(pointer[-1]) if pointer.numel() else 0
    if total <= 0:
        return torch.zeros(cores, dtype=torch.int32)
    chunk = (total + cores - 1) // cores
    targets = torch.arange(cores, dtype=torch.int64) * chunk
    return (
        torch.searchsorted(pointer.to(torch.int64), targets, right=True) - 1
    ).clamp_min(0).to(torch.int32)


def to_npu(tensor: torch.Tensor, dtype=None):
    return tensor.to(dtype=dtype or tensor.dtype).contiguous().npu()


def make_inputs(col_count: int, cols: int, dim: int, dtype, seed: int):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(col_count, generator=generator).to(dtype).contiguous()
    dense = torch.rand((cols, dim), generator=generator).to(dtype).contiguous()
    return values, dense


def build_invoker(method, rows, cols, crow, col, values, dense):
    if method not in METHODS:
        raise ValueError(f"unknown ablation method: {method}")
    dense_npu = to_npu(dense)

    if method.startswith("aic-"):
        packed = spmm_npu.csr_to_bcsr(
            crow, col, values, rows, cols, 16
        )
        empty_vrow = torch.zeros(1, dtype=torch.int32)
        args = (
            to_npu(packed["crow_window_ptr"], torch.int32),
            to_npu(packed["unique_col_ids"], torch.int32),
            to_npu(packed["value"]),
            to_npu(empty_vrow),
            to_npu(torch.empty(0, dtype=torch.int32)),
            to_npu(torch.empty(0, dtype=values.dtype)),
            dense_npu,
            to_npu(make_start(packed["crow_window_ptr"], AIC_CORES)),
            to_npu(torch.zeros(AIV_CORES, dtype=torch.int32)),
        )
        operator = (
            spmm_npu.spmm_ascend_cube
            if method == "aic-pipeline"
            else spmm_npu.spmm_ascend_cube_aic_without_pipeline
        )
        return lambda: operator(*args), rows

    row_npu = to_npu(crow, torch.int32)
    col_npu = to_npu(col, torch.int32)
    value_npu = to_npu(values)
    vstart_npu = to_npu(make_start(crow, AIV_CORES))
    if method in ("aiv-naive", "aiv-mb-batch"):
        operator = (
            spmm_npu.spmm_csr
            if method == "aiv-naive"
            else spmm_npu.spmm_csr_aiv_mb_batch
        )
        args = (row_npu, col_npu, value_npu, dense_npu, vstart_npu)
        return lambda: operator(*args), rows

    args = (
        to_npu(torch.zeros(1, dtype=torch.int32)),
        to_npu(torch.empty(0, dtype=torch.int32)),
        to_npu(torch.empty(0, dtype=values.dtype)),
        row_npu,
        col_npu,
        value_npu,
        dense_npu,
        to_npu(torch.zeros(AIC_CORES, dtype=torch.int32)),
        vstart_npu,
    )
    operator = (
        spmm_npu.spmm_ascend_cube_aiv_mb
        if method == "aiv-mb"
        else spmm_npu.spmm_ascend_cube
    )
    return lambda: operator(*args), rows


def cpu_reference(rows, cols, crow, col, values, dense):
    matrix = torch.sparse_csr_tensor(
        crow, col, values.float(), size=(rows, cols)
    )
    return torch.sparse.mm(matrix, dense.float())
