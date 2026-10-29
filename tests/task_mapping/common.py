#!/usr/bin/env python3
"""Shared preparation and invocation helpers for the Table 6 methods."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch_npu  # noqa: F401

import spmm_npu


AIC_CORES = 20
AIV_CORES = 40
WINDOW_SIZE = 16
METHODS = ("ours", "naive", "neutronsparse", "block-split")
METHOD_LABELS = {
    "ours": "Ascend-SpMM task mapping",
    "naive": "Naive load balancing",
    "neutronsparse": "NeutronSparse load balancing",
    "block-split": "Block-Level split",
}
DEFAULT_BLOCK_RATIO = (222.0 / 190.0) * 0.5
DEFAULT_NAIVE_FRACTION = 1.0 / 3.0
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
    if "nnz" in payload and int(payload["nnz"]) != col.numel():
        raise ValueError("payload nnz does not match col_indices")
    fallback = path.stem[:-4] if path.stem.endswith("_csr") else path.stem
    name = str(payload.get("name") or fallback)
    return name, rows, cols, crow, col


def discover_datasets(
    dataset_dir: Path,
    name: str = "",
    index: Optional[int] = None,
    start: int = 0,
    stop: Optional[int] = None,
    limit: Optional[int] = None,
):
    paths = sorted(
        dataset_dir.rglob("*.pt"),
        key=lambda item: item.relative_to(dataset_dir).as_posix(),
    )
    if name:
        paths = [
            path for path in paths
            if name == path.parent.name or name in path.stem
        ]
    if index is not None:
        if index < 0 or index >= len(paths):
            raise ValueError(
                f"--index {index} is outside the discovered range [0, {len(paths)})"
            )
        return [paths[index]]
    if start < 0:
        raise ValueError("--start must be non-negative")
    if stop is not None and stop < start:
        raise ValueError("--stop must be greater than or equal to --start")
    paths = paths[start:stop]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no .pt datasets selected under {dataset_dir}")
    return paths


def select_dataset(dataset_dir: Path, name: str, index: int):
    if name:
        paths = discover_datasets(dataset_dir, name=name)
        if len(paths) != 1:
            raise ValueError(
                f"--name selected {len(paths)} datasets; expected exactly one"
            )
        return paths[0]
    return discover_datasets(dataset_dir, index=index)[0]


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


def make_values(nnz: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(nnz, generator=generator) * 2 - 1).to(dtype).contiguous()


def make_dense(cols: int, dim: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand((cols, dim), generator=generator) * 2 - 1
    ).to(dtype).contiguous()


def compute_neutronsparse_threshold(rows: int, cols: int, nnz: int) -> int:
    """Mirror NeutronSparseLite's SplitStrategy.get_smart_threshold."""
    if rows == 0:
        return 32
    average_degree = nnz / rows
    sparsity = nnz / (rows * cols) if rows * cols else 0.0
    if sparsity < 0.0002:
        alpha = 1.8 if rows > 1_000_000 else 1.0
    elif sparsity > 0.01:
        alpha = 0.25
    else:
        alpha = 0.55
    return max(4, int(average_degree * alpha))


def _csr_from_numpy_mask(
    row_counts: np.ndarray,
    columns: np.ndarray,
    mask: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indptr = np.empty(row_counts.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, dtype=np.int64, out=indptr[1:])
    return torch.from_numpy(indptr), torch.from_numpy(columns[mask].copy())


def _neutronsparse_parts(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    rows: int,
    cols: int,
    assume_symmetric: bool,
):
    """Apply NeutronSparseLite's dense-column then dense-row partition."""
    indptr = crow.numpy()
    columns = col.numpy()
    row_degree = np.diff(indptr).astype(np.int64)
    if assume_symmetric and rows == cols:
        column_degree = row_degree.copy()
    else:
        column_degree = np.bincount(columns, minlength=cols).astype(
            np.int64, copy=False
        )
    threshold = compute_neutronsparse_threshold(rows, cols, columns.size)
    dense_column = column_degree > threshold
    edge_dense_column = dense_column[columns]
    row_dense_degree = np.zeros(rows, dtype=np.int64)
    nonempty_rows = row_degree > 0
    row_dense_degree[nonempty_rows] = np.add.reduceat(
        edge_dense_column,
        indptr[:-1][nonempty_rows],
        dtype=np.int64,
    )
    dense_row = row_dense_degree > threshold
    edge_dense_row = np.repeat(dense_row, row_degree)
    np.logical_and(edge_dense_column, edge_dense_row, out=edge_dense_column)
    cube_mask = edge_dense_column
    vector_mask = np.logical_not(cube_mask)
    cube_counts = np.where(dense_row, row_dense_degree, 0)
    vector_counts = row_degree - cube_counts
    cube_crow, cube_col = _csr_from_numpy_mask(
        cube_counts, columns, cube_mask
    )
    vector_crow, vector_col = _csr_from_numpy_mask(
        vector_counts, columns, vector_mask
    )
    cube_mask_tensor = torch.from_numpy(cube_mask)
    vector_mask_tensor = torch.from_numpy(vector_mask)
    metadata = {
        "threshold": threshold,
        "assume_symmetric": bool(assume_symmetric),
        "symmetry_shortcut_used": bool(assume_symmetric and rows == cols),
        "dense_rows": int(dense_row.sum()),
        "dense_columns": int(dense_column.sum()),
        "aic_nnz": int(cube_mask.sum()),
        "aiv_nnz": int(vector_mask.sum()),
    }
    return (
        (cube_crow, cube_col, values[cube_mask_tensor].contiguous()),
        (vector_crow, vector_col, values[vector_mask_tensor].contiguous()),
        metadata,
    )


def _empty_aic(index_dtype: torch.dtype, value_dtype: torch.dtype):
    return {
        "crow_window_ptr": torch.zeros(1, dtype=index_dtype),
        "unique_col_ids": torch.empty(0, dtype=index_dtype),
        "value": torch.empty(0, dtype=value_dtype),
    }


def _pack_cube(crow, col, values, rows: int, cols: int):
    if col.numel() == 0:
        return _empty_aic(crow.dtype, values.dtype)
    counts = crow[1:] - crow[:-1]
    nonempty_rows = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if not nonempty_rows.numel():
        return _empty_aic(crow.dtype, values.dtype)
    packed_rows = int(nonempty_rows[-1]) + 1
    trimmed_crow = crow[:packed_rows + 1].contiguous()
    if int(trimmed_crow[-1]) != col.numel():
        raise ValueError("cube CSR contains edges after its last nonempty row")
    return spmm_npu.csr_to_bcsr(
        trimmed_crow, col.contiguous(), values.contiguous(),
        packed_rows, cols, WINDOW_SIZE,
    )


def _combine_parts(cube, vector, rows: int, cols: int):
    cube_crow, cube_col, cube_values = cube
    vector_crow, vector_col, vector_values = vector
    packed = _pack_cube(cube_crow, cube_col, cube_values, rows, cols)
    return {
        "crow_window_ptr": packed["crow_window_ptr"].contiguous(),
        "unique_col_ids": packed["unique_col_ids"].contiguous(),
        "value": packed["value"].contiguous(),
        "vrow_window_ptr": vector_crow.contiguous(),
        "vcol_indices": vector_col.contiguous(),
        "vvalue": vector_values.contiguous(),
    }


def _naive_parts(crow, col, values, rows: int, fraction: float):
    nnz = col.numel()
    pivot = nnz // 3 if math.isclose(fraction, 1.0 / 3.0) else int(nnz * fraction)
    pivot = max(0, min(nnz, pivot))
    cube_crow = crow.clamp_max(pivot).contiguous()
    vector_crow = (crow - pivot).clamp_min(0).contiguous()
    metadata = {
        "fraction": fraction,
        "pivot_nnz": pivot,
        "aic_nnz": pivot,
        "aiv_nnz": nnz - pivot,
    }
    return (
        (cube_crow, col[:pivot].contiguous(), values[:pivot].contiguous()),
        (vector_crow, col[pivot:].contiguous(), values[pivot:].contiguous()),
        metadata,
    )


def prepare_partition(
    method: str,
    rows: int,
    cols: int,
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    block_ratio: float = DEFAULT_BLOCK_RATIO,
    naive_fraction: float = DEFAULT_NAIVE_FRACTION,
    assume_symmetric: bool = False,
):
    if method not in METHODS:
        raise ValueError(f"unknown partition method: {method}")
    if block_ratio <= 0:
        raise ValueError("block_ratio must be positive")
    if not 0.0 <= naive_fraction <= 1.0:
        raise ValueError("naive_fraction must be in [0, 1]")
    nnz = col.numel()
    if method == "ours":
        mapping = spmm_npu.preprocess(
            crow, col, values, rows, cols, WINDOW_SIZE
        )
        aiv_nnz = int(mapping["vcol_indices"].numel())
        metadata = {
            "aic_nnz": nnz - aiv_nnz,
            "aiv_nnz": aiv_nnz,
            "aic_blocks": int(mapping["unique_col_ids"].numel()),
        }
    elif method == "block-split":
        mapping = spmm_npu.block_level_split(
            crow, col, values, rows, cols, block_ratio, WINDOW_SIZE
        )
        aiv_nnz = int(mapping["vcol_indices"].numel())
        metadata = {
            "ratio": block_ratio,
            "aic_nnz": nnz - aiv_nnz,
            "aiv_nnz": aiv_nnz,
            "aic_blocks": int(mapping["unique_col_ids"].numel()),
        }
    elif method == "neutronsparse":
        cube, vector, metadata = _neutronsparse_parts(
            crow, col, values, rows, cols, assume_symmetric
        )
        mapping = _combine_parts(cube, vector, rows, cols)
    else:
        cube, vector, metadata = _naive_parts(
            crow, col, values, rows, naive_fraction
        )
        mapping = _combine_parts(cube, vector, rows, cols)
    required = {
        "crow_window_ptr", "unique_col_ids", "value",
        "vrow_window_ptr", "vcol_indices", "vvalue",
    }
    missing = sorted(required.difference(mapping))
    if missing:
        raise ValueError(f"partition mapping is missing: {', '.join(missing)}")
    if metadata["aic_nnz"] + metadata["aiv_nnz"] != nnz:
        raise ValueError("partition does not preserve the original NNZ count")
    return mapping, {"method": method, "label": METHOD_LABELS[method], **metadata}


def build_mixed_invoker(mapping: Dict[str, torch.Tensor], dense: torch.Tensor):
    args = (
        to_npu(mapping["crow_window_ptr"], torch.int32),
        to_npu(mapping["unique_col_ids"], torch.int32),
        to_npu(mapping["value"]),
        to_npu(mapping["vrow_window_ptr"], torch.int32),
        to_npu(mapping["vcol_indices"], torch.int32),
        to_npu(mapping["vvalue"]),
        to_npu(dense),
        to_npu(make_start(mapping["crow_window_ptr"], AIC_CORES), torch.int32),
        to_npu(make_start(mapping["vrow_window_ptr"], AIV_CORES), torch.int32),
    )
    return lambda: spmm_npu.spmm_ascend_cube(*args)


def build_reference_invoker(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    dense: torch.Tensor,
):
    args = (
        to_npu(crow, torch.int32),
        to_npu(col, torch.int32),
        to_npu(values),
        to_npu(dense),
        to_npu(make_start(crow, AIV_CORES), torch.int32),
    )
    return lambda: spmm_npu.spmm_csr(*args)
