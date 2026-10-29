"""Public Python API for the Ascend-SpMM NPU extension.

For fixed square graphs, ``graph_layout=True`` returns ``P A P^T`` mappings;
pack node tensors once with ``P`` and restore original ids only when needed.
"""

import torch

from custom_ops_lib import (
    block_level_split,
    csr_to_ascend_cube,
    csr_to_bcsr,
    preprocess,
    spmm_ascend_cube,
    spmm_ascend_cube_aic_without_pipeline,
    spmm_ascend_cube_aiv_mb,
    spmm_csr,
    spmm_csr_aiv_mb_batch,
    spmm_cube,
)


def pack_node_tensor(tensor, inv_permutation, packed_rows=None):
    """Apply ``P`` once to a tensor whose leading dimension is original node id.

    Padding rows, when present, are initialized to zero.  This helper belongs
    in graph preprocessing, not in the per-layer SpMM loop.
    """
    if tensor.ndim < 1:
        raise ValueError("a node tensor must have at least one dimension")
    index = inv_permutation.to(device=tensor.device, dtype=torch.long).reshape(-1)
    if tensor.size(0) != index.numel():
        raise ValueError("inv_permutation must contain one entry per original node")
    if index.numel() and bool((index < 0).any().item()):
        raise ValueError("inv_permutation does not map every original node")
    minimum_rows = int(index.max().item()) + 1 if index.numel() else 0
    if packed_rows is None:
        packed_rows = minimum_rows
    elif torch.is_tensor(packed_rows):
        packed_rows = int(packed_rows.item())
    else:
        packed_rows = int(packed_rows)
    if packed_rows < minimum_rows:
        raise ValueError("packed_rows is smaller than the permutation extent")
    packed = tensor.new_zeros((packed_rows, *tensor.shape[1:]))
    packed.index_copy_(0, index, tensor)
    return packed


def restore_node_tensor(packed_tensor, inv_permutation):
    """Apply ``P^T`` once and return rows in original node-id order."""
    if packed_tensor.ndim < 1:
        raise ValueError("a packed node tensor must have at least one dimension")
    index = inv_permutation.to(
        device=packed_tensor.device, dtype=torch.long
    ).reshape(-1)
    if index.numel() and (
        bool((index < 0).any().item())
        or int(index.max().item()) >= packed_tensor.size(0)
    ):
        raise ValueError("inv_permutation contains an invalid packed node id")
    return packed_tensor.index_select(0, index)


__all__ = [
    "block_level_split",
    "csr_to_ascend_cube",
    "csr_to_bcsr",
    "pack_node_tensor",
    "preprocess",
    "restore_node_tensor",
    "spmm_ascend_cube",
    "spmm_ascend_cube_aic_without_pipeline",
    "spmm_ascend_cube_aiv_mb",
    "spmm_csr",
    "spmm_csr_aiv_mb_batch",
    "spmm_cube",
]
