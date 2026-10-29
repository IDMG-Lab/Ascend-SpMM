#!/usr/bin/env python3
"""Validate the pure-AIV path against the standalone CSR AIV operator.

This is a numerical correctness workload only. It intentionally records no
operator or wall-clock timing.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import spmm_npu


AIC_CORES = 20
AIV_CORES = 40
DEFAULT_DIMS = (16, 32, 64, 128, 256)
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
    if col.ndim != 1:
        raise ValueError("col_indices must be one-dimensional")
    if int(crow[0].item()) != 0 or int(crow[-1].item()) != col.numel():
        raise ValueError("crow_indices endpoints do not match col_indices")
    if bool((crow[1:] < crow[:-1]).any().item()):
        raise ValueError("crow_indices must be nondecreasing")
    if col.numel() == 0:
        raise ValueError("empty matrices are not supported by the NPU kernels")
    if int(col.min().item()) < 0 or int(col.max().item()) >= cols:
        raise ValueError("col_indices contains an out-of-range column")
    if max(rows, cols, col.numel()) > torch.iinfo(torch.int32).max:
        raise ValueError("matrix exceeds the artifact's 32-bit NPU index contract")
    if "nnz" in payload and int(payload["nnz"]) != col.numel():
        raise ValueError("payload nnz does not match col_indices")

    stem = path.stem
    fallback_name = stem[:-4] if stem.endswith("_csr") else stem
    name = str(payload.get("name") or fallback_name)
    return name, rows, cols, crow, col


def discover_datasets(dataset_dir: Path, name: str, index, start: int, stop, limit):
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
        paths = [paths[index]]
    else:
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


def make_start(pointer: torch.Tensor, cores: int) -> torch.Tensor:
    total = int(pointer[-1].item())
    if total <= 0:
        return torch.zeros(cores, dtype=torch.int32)
    chunk = (total + cores - 1) // cores
    targets = torch.arange(cores, dtype=torch.int64) * chunk
    return (
        torch.searchsorted(pointer, targets, right=True) - 1
    ).clamp_min(0).to(torch.int32)


def to_npu(tensor: torch.Tensor, dtype=None):
    return tensor.to(dtype=dtype or tensor.dtype).contiguous().npu()


def stable_dataset_seed(base_seed: int, relative_path: str) -> int:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return (base_seed + offset) % (2**63 - 1)


def validate_dataset(path, dataset_dir, dims, dtype, seed, atol, rtol):
    name, rows, cols, crow, col = load_payload(path)
    relative_path = path.relative_to(dataset_dir).as_posix()
    dataset_seed = stable_dataset_seed(seed, relative_path)
    generator = torch.Generator().manual_seed(dataset_seed)
    values = (torch.rand(col.numel(), generator=generator) * 2 - 1).to(dtype)

    vrow_npu = to_npu(crow, torch.int32)
    vcol_npu = to_npu(col, torch.int32)
    values_npu = to_npu(values)
    vstart_npu = to_npu(make_start(crow, AIV_CORES), torch.int32)

    empty_crow_npu = to_npu(torch.zeros(1, dtype=torch.int32))
    empty_col_npu = to_npu(torch.empty(0, dtype=torch.int32))
    empty_value_npu = to_npu(torch.empty(0, dtype=dtype))
    cstart_npu = to_npu(torch.zeros(AIC_CORES, dtype=torch.int32))

    cases = []
    for dim in dims:
        dense = (torch.rand((cols, dim), generator=generator) * 2 - 1).to(dtype)
        dense_npu = to_npu(dense)

        aiv_output = spmm_npu.spmm_ascend_cube(
            empty_crow_npu,
            empty_col_npu,
            empty_value_npu,
            vrow_npu,
            vcol_npu,
            values_npu,
            dense_npu,
            cstart_npu,
            vstart_npu,
        )[:rows]
        reference_output = spmm_npu.spmm_csr(
            vrow_npu,
            vcol_npu,
            values_npu,
            dense_npu,
            vstart_npu,
        )
        torch.npu.synchronize()

        actual = aiv_output.cpu()
        expected = reference_output.cpu()
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"shape mismatch for {name}, width={dim}: "
                f"{tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        if not bool(torch.isfinite(actual).all().item()):
            raise RuntimeError(f"pure-AIV output is non-finite for {name}, width={dim}")
        if not bool(torch.isfinite(expected).all().item()):
            raise RuntimeError(f"CSR reference is non-finite for {name}, width={dim}")

        difference = (actual.float() - expected.float()).abs()
        close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
        mismatch_count = int((~close).sum().item())
        max_abs_error = float(difference.max().item()) if difference.numel() else 0.0
        mean_abs_error = float(difference.mean().item()) if difference.numel() else 0.0
        passed = mismatch_count == 0
        record = {
            "dataset": name,
            "path": relative_path,
            "rows": rows,
            "cols": cols,
            "nnz": col.numel(),
            "feature_width": dim,
            "dtype": "fp16" if dtype == torch.float16 else "fp32",
            "dataset_seed": dataset_seed,
            "mismatch_count": mismatch_count,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "passed": passed,
        }
        if not passed:
            first = torch.nonzero(~close, as_tuple=False)[0]
            row, column = map(int, first.tolist())
            record.update(
                first_mismatch_row=row,
                first_mismatch_column=column,
                aiv_value=float(actual[row, column].item()),
                reference_value=float(expected[row, column].item()),
            )
        cases.append(record)
        status = "PASS" if passed else "FAIL"
        print(
            f"{status}: dataset={name}, width={dim}, "
            f"dtype={record['dtype']}, mismatches={mismatch_count}, "
            f"max_abs_error={max_abs_error:.8g}"
        )
        del dense_npu, aiv_output, reference_output, actual, expected

    torch.npu.empty_cache()
    return cases


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the pure-AIV branch of spmm_ascend_cube with spmm_csr. "
            "No timing is collected."
        )
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "datasets_pt",
        help="Directory containing CSR .pt files produced by tests/data/prepare_data.py.",
    )
    parser.add_argument("--name", default="", help="Filter by parent directory or filename.")
    parser.add_argument("--index", type=int, help="Select one index after optional name filtering.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive index after name filtering.")
    parser.add_argument("--stop", type=int, help="Exclusive index after name filtering.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected datasets.")
    parser.add_argument(
        "--dim", dest="dims", action="append", type=int,
        help="Feature width; repeat this option to test multiple widths.",
    )
    parser.add_argument(
        "--dims", dest="dims_many", nargs="+", type=int,
        help="Compatibility form for specifying multiple feature widths at once.",
    )
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path,
        help="JSON output; defaults to results/aiv_bitwise_correctness_<dtype>.json.",
    )
    args = parser.parse_args()
    if args.index is not None and (args.start != 0 or args.stop is not None or args.limit is not None):
        parser.error("--index cannot be combined with --start, --stop, or --limit")
    if args.dims is not None and args.dims_many is not None:
        parser.error("use either repeated --dim or --dims, not both")
    if args.dims_many is not None:
        args.dims = args.dims_many
    elif args.dims is None:
        args.dims = list(DEFAULT_DIMS)
    max_dim = 1024 if args.dtype == "fp16" else 512
    invalid_dims = [dim for dim in args.dims if dim <= 0 or dim > max_dim or dim % 16]
    if invalid_dims:
        parser.error(
            f"every feature width must be a positive multiple of 16 no greater than {max_dim} for {args.dtype}"
        )
    if len(set(args.dims)) != len(args.dims):
        parser.error("feature widths must not contain duplicates")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    return args


def main():
    args = parse_args()
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
    cases = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] validating {path.relative_to(dataset_dir)}")
        cases.extend(
            validate_dataset(
                path, dataset_dir, args.dims, dtype, args.seed, args.atol, args.rtol
            )
        )

    failed = sum(not case["passed"] for case in cases)
    report = {
        "schema_version": 1,
        "check": "pure AIV spmm_ascend_cube versus spmm_csr",
        "timing_collected": False,
        "dataset_dir": str(dataset_dir),
        "dtype": args.dtype,
        "feature_widths": args.dims,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "selected_datasets": len(paths),
        "comparisons": len(cases),
        "passed": len(cases) - failed,
        "failed": failed,
        "cases": cases,
    }
    output = args.output or (
        ROOT / "results" / f"aiv_bitwise_correctness_{args.dtype}.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {output}")
    if failed:
        raise SystemExit(f"FAIL: {failed}/{len(cases)} AIV comparisons did not match")
    print(
        f"PASS: {len(cases)} AIV comparisons matched spmm_csr; "
        "no timing was collected"
    )


if __name__ == "__main__":
    main()