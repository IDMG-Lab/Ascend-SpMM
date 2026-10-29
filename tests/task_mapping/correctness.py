#!/usr/bin/env python3
"""Validate the Table 6 proposed mapping and three ablations identically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch_npu  # noqa: F401

from common import (
    DEFAULT_BLOCK_RATIO,
    DEFAULT_NAIVE_FRACTION,
    METHOD_LABELS,
    METHODS,
    ROOT,
    build_mixed_invoker,
    build_reference_invoker,
    discover_datasets,
    load_payload,
    make_dense,
    make_values,
    prepare_partition,
    stable_seed,
)


DEFAULT_DIMS = (128,)


def compare(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    if actual_cpu.shape != expected_cpu.shape:
        raise RuntimeError(
            f"shape mismatch: {tuple(actual_cpu.shape)} != {tuple(expected_cpu.shape)}"
        )
    if not bool(torch.isfinite(actual_cpu).all().item()):
        raise RuntimeError("task-mapping output contains NaN or Inf")
    if not bool(torch.isfinite(expected_cpu).all().item()):
        raise RuntimeError("CSR reference contains NaN or Inf")
    difference = (actual_cpu - expected_cpu).abs()
    close = torch.isclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol)
    mismatch_count = int((~close).sum().item())
    result = {
        "bitwise_equal": bool(torch.equal(actual_cpu, expected_cpu)),
        "mismatch_count": mismatch_count,
        "max_abs_error": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs_error": float(difference.mean().item()) if difference.numel() else 0.0,
        "passed": mismatch_count == 0,
    }
    if mismatch_count:
        first = torch.nonzero(~close, as_tuple=False)[0]
        row, column = map(int, first.tolist())
        result.update(
            first_mismatch_row=row,
            first_mismatch_column=column,
            actual_value=float(actual_cpu[row, column].item()),
            reference_value=float(expected_cpu[row, column].item()),
        )
    return result


def parse_args(default_methods: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare the proposed task mapping and the three Table 6 ablations "
            "with the standalone CSR kernel. No timing is collected."
        )
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "datasets_pt",
        help="CSR .pt directory produced by tests/data/prepare_data.py.",
    )
    parser.add_argument("--name", default="", help="Filter by parent or filename.")
    parser.add_argument("--index", type=int, help="Select one filtered dataset index.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive filtered index.")
    parser.add_argument("--stop", type=int, help="Exclusive filtered index.")
    parser.add_argument("--limit", type=int, help="Maximum selected datasets.")
    parser.add_argument(
        "--method", action="append", choices=METHODS,
        help="Table 6 method; repeat to select multiple (default: all).",
    )
    parser.add_argument(
        "--dim", dest="dims", action="append", type=int,
        help="Feature width; repeat to test multiple widths (default: 128).",
    )
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--block-ratio", type=float, default=DEFAULT_BLOCK_RATIO,
        help="Target AIC block/AIV NNZ ratio used by block_level_split.",
    )
    parser.add_argument(
        "--naive-fraction", type=float, default=DEFAULT_NAIVE_FRACTION,
        help="Leading NNZ fraction assigned to AIC by the naive strategy.",
    )
    parser.add_argument(
        "--neutronsparse-assume-symmetric", action="store_true",
        help="Use row degree as column degree on square matrices.",
    )
    parser.add_argument(
        "--output", type=Path,
        help=(
            "JSON output; defaults to table6_correctness_all_<dtype>.json "
            "or a method-specific filename."
        ),
    )
    args = parser.parse_args()
    if args.index is not None and (
        args.start != 0 or args.stop is not None or args.limit is not None
    ):
        parser.error("--index cannot be combined with --start, --stop, or --limit")
    fallback_methods = tuple(default_methods) if default_methods else METHODS
    args.methods = args.method or list(fallback_methods)
    if len(args.methods) != len(set(args.methods)):
        parser.error("Table 6 methods must not contain duplicates")
    args.dims = args.dims or list(DEFAULT_DIMS)
    max_dim = 1024 if args.dtype == "fp16" else 512
    invalid_dims = [
        dim for dim in args.dims
        if dim <= 0 or dim > max_dim or dim % 16
    ]
    if invalid_dims:
        parser.error(
            f"every --dim must be a positive multiple of 16 no greater than "
            f"{max_dim} for {args.dtype}"
        )
    if len(args.dims) != len(set(args.dims)):
        parser.error("feature widths must not contain duplicates")
    if args.block_ratio <= 0:
        parser.error("--block-ratio must be positive")
    if not 0.0 <= args.naive_fraction <= 1.0:
        parser.error("--naive-fraction must be in [0, 1]")
    if args.atol is None:
        args.atol = 2e-2 if args.dtype == "fp16" else 1e-4
    if args.rtol is None:
        args.rtol = 2e-2 if args.dtype == "fp16" else 1e-4
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    return args


def validate_dataset(path, dataset_dir, args, dtype):
    name, rows, cols, crow, col = load_payload(path)
    relative = path.relative_to(dataset_dir).as_posix()
    dataset_seed = stable_seed(args.seed, relative)
    values = make_values(col.numel(), dtype, dataset_seed)

    cases = []
    partition_metadata = {}
    for method in args.methods:
        mapping, metadata = prepare_partition(
            method, rows, cols, crow, col, values,
            block_ratio=args.block_ratio,
            naive_fraction=args.naive_fraction,
            assume_symmetric=args.neutronsparse_assume_symmetric,
        )
        partition_metadata[method] = metadata
        print(
            f"partition={method}, aic_nnz={metadata['aic_nnz']}, "
            f"aiv_nnz={metadata['aiv_nnz']}"
        )

        for dim in args.dims:
            dense = make_dense(cols, dim, dtype, dataset_seed + dim)
            reference_invoke = build_reference_invoker(crow, col, values, dense)
            reference = reference_invoke()[:rows]
            torch.npu.synchronize()
            reference = reference.detach().cpu()
            del reference_invoke
            torch.npu.empty_cache()

            invoke = build_mixed_invoker(mapping, dense)
            output = invoke()[:rows]
            torch.npu.synchronize()
            metrics = compare(output, reference, args.atol, args.rtol)
            record = {
                "dataset": name,
                "path": relative,
                "rows": rows,
                "cols": cols,
                "nnz": col.numel(),
                "method": method,
                "label": METHOD_LABELS[method],
                "reference": "spmm_csr",
                "feature_width": dim,
                "dtype": args.dtype,
                "atol": args.atol,
                "rtol": args.rtol,
                **metadata,
                **metrics,
            }
            cases.append(record)
            print(
                f"{'PASS' if record['passed'] else 'FAIL'}: "
                f"dataset={name}, method={method}, N={dim}, "
                f"mismatches={record['mismatch_count']}, "
                f"max_abs_error={record['max_abs_error']:.8g}"
            )
            del invoke, output, dense, reference
            torch.npu.empty_cache()
        del mapping
        torch.npu.empty_cache()
    del values
    torch.npu.empty_cache()
    return {
        "dataset": name,
        "path": relative,
        "rows": rows,
        "cols": cols,
        "nnz": col.numel(),
        "dataset_seed": dataset_seed,
        "partitions": partition_metadata,
        "cases": cases,
    }


def main(default_methods: Optional[Iterable[str]] = None):
    args = parse_args(default_methods)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("An available Ascend NPU and torch-npu are required")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    try:
        paths = discover_datasets(
            dataset_dir, args.name, args.index, args.start, args.stop, args.limit
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    datasets = []
    cases = []
    for position, path in enumerate(paths, start=1):
        print(f"[{position}/{len(paths)}] validating {path.relative_to(dataset_dir)}")
        result = validate_dataset(path, dataset_dir, args, dtype)
        datasets.append({key: value for key, value in result.items() if key != "cases"})
        cases.extend(result["cases"])

    failures = sum(not case["passed"] for case in cases)
    report = {
        "schema_version": 1,
        "experiment": "Table 6 task-mapping correctness",
        "timing_collected": False,
        "reference": "spmm_csr on the original CSR",
        "dataset_dir": str(dataset_dir),
        "selected_datasets": len(paths),
        "methods": args.methods,
        "dtype": args.dtype,
        "feature_widths": args.dims,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "block_ratio": args.block_ratio,
        "naive_fraction": args.naive_fraction,
        "neutronsparse_assume_symmetric": args.neutronsparse_assume_symmetric,
        "comparisons": len(cases),
        "passed": failures == 0,
        "failed": failures,
        "datasets": datasets,
        "cases": cases,
    }
    if args.output:
        output = args.output
    else:
        method_suffix = args.methods[0] if len(args.methods) == 1 else "all"
        output = (
            ROOT / "results"
            / f"table6_correctness_{method_suffix}_{args.dtype}.json"
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {output}")
    if failures:
        raise SystemExit(f"FAIL: {failures}/{len(cases)} comparisons failed")
    print(f"PASS: all {len(cases)} Table 6 comparisons passed; no timing collected")


if __name__ == "__main__":
    main()
