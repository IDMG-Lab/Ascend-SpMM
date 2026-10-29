#!/usr/bin/env python3
"""Correctness checks for the Figure 10/11 AIC and AIV ablation kernels."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from common import (
    METHOD_LABELS,
    ROOT,
    build_invoker,
    load_payload,
    make_inputs,
    select_dataset,
    stable_seed,
)


AIC_METHODS = ("aic-pipeline", "aic-without-pipeline")
AIV_METHODS = ("aiv-naive", "aiv-mb", "aiv-mb-batch", "aiv-round-robin")
DEFAULT_DIMS = (128,)


def compare(actual, expected, atol, rtol):
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    if actual_cpu.shape != expected_cpu.shape:
        raise RuntimeError(
            f"shape mismatch: {tuple(actual_cpu.shape)} != {tuple(expected_cpu.shape)}"
        )
    if not bool(torch.isfinite(actual_cpu).all().item()):
        raise RuntimeError("ablation output contains NaN or Inf")
    difference = (actual_cpu - expected_cpu).abs()
    close = torch.isclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol)
    mismatch_count = int((~close).sum().item())
    return {
        "bitwise_equal": bool(torch.equal(actual_cpu, expected_cpu)),
        "mismatch_count": mismatch_count,
        "max_abs_error": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs_error": float(difference.mean().item()) if difference.numel() else 0.0,
        "passed": mismatch_count == 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Figure 10/11 AIC and AIV ablation kernels."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "datasets_pt",
        help="CSR .pt directory produced by tests/data/prepare_data.py.",
    )
    parser.add_argument("--name", default="", help="Select one parent or filename match.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index when --name is absent.")
    parser.add_argument(
        "--dim", dest="dims", action="append", type=int,
        help="Feature width; repeat to test multiple widths (default: 128).",
    )
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--aic-atol", type=float)
    parser.add_argument("--aic-rtol", type=float)
    parser.add_argument("--aiv-atol", type=float, default=0.0)
    parser.add_argument("--aiv-rtol", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path,
        help="JSON output (default: results/aic_aiv_correctness_<dtype>.json).",
    )
    args = parser.parse_args()
    args.dims = args.dims or list(DEFAULT_DIMS)
    max_dim = 1024 if args.dtype == "fp16" else 512
    invalid = [dim for dim in args.dims if dim <= 0 or dim > max_dim or dim % 16]
    if invalid:
        parser.error(
            f"every --dim must be a positive multiple of 16 no greater than {max_dim} for {args.dtype}"
        )
    if len(args.dims) != len(set(args.dims)):
        parser.error("feature widths must not contain duplicates")
    tolerances = (args.aic_atol, args.aic_rtol, args.aiv_atol, args.aiv_rtol)
    if any(value is not None and value < 0 for value in tolerances):
        parser.error("all tolerances must be non-negative")
    if args.aic_atol is None:
        args.aic_atol = 2e-2 if args.dtype == "fp16" else 1e-4
    if args.aic_rtol is None:
        args.aic_rtol = 2e-2 if args.dtype == "fp16" else 1e-4
    return args


def main():
    args = parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("An available Ascend NPU and torch-npu are required")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    try:
        path = select_dataset(dataset_dir, args.name, args.index)
        name, rows, cols, crow, col = load_payload(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    relative = path.relative_to(dataset_dir).as_posix()
    dataset_seed = stable_seed(args.seed, relative)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    cases = []
    for dim in args.dims:
        values, dense = make_inputs(col.numel(), cols, dim, dtype, dataset_seed + dim)

        invoke, _ = build_invoker(
            "aic-pipeline", rows, cols, crow, col, values, dense
        )
        aic_reference = invoke()[:rows]
        torch.npu.synchronize()
        aic_reference = aic_reference.detach().cpu()
        del invoke
        torch.npu.empty_cache()

        for method in AIC_METHODS:
            if method == "aic-pipeline":
                output = aic_reference
            else:
                invoke, _ = build_invoker(
                    method, rows, cols, crow, col, values, dense
                )
                output = invoke()[:rows]
                torch.npu.synchronize()
            metrics = compare(
                output, aic_reference, args.aic_atol, args.aic_rtol
            )
            record = {
                "method": method,
                "label": METHOD_LABELS[method],
                "reference": "aic-pipeline",
                "feature_width": dim,
                "atol": args.aic_atol,
                "rtol": args.aic_rtol,
                **metrics,
            }
            cases.append(record)
            print(
                f"{'PASS' if record['passed'] else 'FAIL'}: {method}, N={dim}, "
                f"mismatches={record['mismatch_count']}, "
                f"max_abs_error={record['max_abs_error']:.8g}"
            )
            if method != "aic-pipeline":
                del invoke, output
                torch.npu.empty_cache()

        invoke, _ = build_invoker(
            "aiv-naive", rows, cols, crow, col, values, dense
        )
        aiv_reference = invoke()[:rows]
        torch.npu.synchronize()
        aiv_reference = aiv_reference.detach().cpu()
        del invoke
        torch.npu.empty_cache()

        for method in AIV_METHODS:
            if method == "aiv-naive":
                output = aiv_reference
            else:
                invoke, _ = build_invoker(
                    method, rows, cols, crow, col, values, dense
                )
                output = invoke()[:rows]
                torch.npu.synchronize()
            metrics = compare(
                output, aiv_reference, args.aiv_atol, args.aiv_rtol
            )
            record = {
                "method": method,
                "label": METHOD_LABELS[method],
                "reference": "aiv-naive",
                "feature_width": dim,
                "atol": args.aiv_atol,
                "rtol": args.aiv_rtol,
                **metrics,
            }
            cases.append(record)
            print(
                f"{'PASS' if record['passed'] else 'FAIL'}: {method}, N={dim}, "
                f"bitwise={record['bitwise_equal']}, "
                f"mismatches={record['mismatch_count']}, "
                f"max_abs_error={record['max_abs_error']:.8g}"
            )
            if method != "aiv-naive":
                del invoke, output
                torch.npu.empty_cache()
        del aic_reference, aiv_reference, values, dense
        torch.npu.empty_cache()

    failed = sum(not case["passed"] for case in cases)
    report = {
        "schema_version": 1,
        "experiment": "Figure 10/11 kernel ablations",
        "timing_collected": False,
        "dataset": name,
        "path": relative,
        "rows": rows,
        "cols": cols,
        "nnz": col.numel(),
        "dtype": args.dtype,
        "feature_widths": args.dims,
        "seed": dataset_seed,
        "aic_reference": "hierarchy-aware pipeline with SPTILE=96 (g=6)",
        "aic_ablation": "without hierarchy-aware pipeline, SPTILE=16 (g=1)",
        "aiv_reference": "Naive CSR",
        "aiv_variants": [
            "Multiple Buffer",
            "Multiple Buffer + Batched Process (batch=8, buffers=64)",
            "Round-Robin Ring Buffer",
        ],
        "comparisons": len(cases),
        "passed": failed == 0,
        "failed": failed,
        "cases": cases,
    }
    output = args.output or (
        ROOT / "results" / f"aic_aiv_correctness_{args.dtype}.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {output}")
    if failed:
        raise SystemExit(f"FAIL: {failed}/{len(cases)} comparisons failed")
    print(f"PASS: all {len(cases)} ablation comparisons passed; no timing collected")


if __name__ == "__main__":
    main()
