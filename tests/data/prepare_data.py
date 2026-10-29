#!/usr/bin/env python3
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.io import mmread


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_matrix(path):
    return path.stem == path.parent.name


def convert_matrix(source, destination):
    matrix = mmread(source)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(np.asarray(matrix))
    csr = matrix.tocsr(copy=False)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    rows, cols = csr.shape
    payload = {
        "name": source.stem,
        "shape": (int(rows), int(cols)),
        "nnz": int(csr.nnz),
        "crow_indices": torch.from_numpy(csr.indptr.astype(np.int64, copy=True)),
        "col_indices": torch.from_numpy(csr.indices.astype(np.int64, copy=True)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return rows, cols, csr.nnz


def main():
    parser = argparse.ArgumentParser(description="Convert Matrix Market files to non-destructive CSR structure files.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("results/dataset_manifest.csv"))
    parser.add_argument("--primary-only", action="store_true", help="Keep only <directory>/<directory>.mtx files.")
    parser.add_argument("--paper-filter", action="store_true", help="Require square shape, >=10K rows/cols, and >=100K NNZ.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-source", action="store_true", help="Compute SHA-256 of each source; expensive for the full collection.")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    files = sorted(input_root.rglob("*.mtx"), key=lambda item: item.relative_to(input_root).as_posix())
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .mtx files found under {input_root}")

    rows_out = []
    converted = 0
    for source in files:
        relative = source.relative_to(input_root)
        record = {
            "name": source.stem,
            "source": relative.as_posix(),
            "source_bytes": source.stat().st_size,
            "source_sha256": "",
            "rows": "",
            "cols": "",
            "nnz": "",
            "included": False,
            "reason": "",
            "output": "",
        }
        try:
            if args.primary_only and not primary_matrix(source):
                record["reason"] = "not_primary_matrix"
                rows_out.append(record)
                continue
            if args.hash_source:
                record["source_sha256"] = sha256_file(source)

            matrix = mmread(source)
            shape = matrix.shape
            nnz = int(matrix.nnz if sparse.issparse(matrix) else np.count_nonzero(matrix))
            record.update(rows=int(shape[0]), cols=int(shape[1]), nnz=nnz)
            if args.paper_filter:
                if shape[0] != shape[1]:
                    record["reason"] = "not_square"
                    rows_out.append(record)
                    continue
                if shape[0] < 10000 or shape[1] < 10000:
                    record["reason"] = "dimension_below_10000"
                    rows_out.append(record)
                    continue
                if nnz < 100000:
                    record["reason"] = "nnz_below_100000"
                    rows_out.append(record)
                    continue

            destination = output_root / relative.parent / f"{source.stem}_csr.pt"
            if destination.exists() and not args.overwrite:
                record.update(included=True, reason="already_exists", output=destination.relative_to(output_root).as_posix())
                rows_out.append(record)
                converted += 1
                continue

            if sparse.issparse(matrix):
                csr = matrix.tocsr(copy=False)
            else:
                csr = sparse.csr_matrix(np.asarray(matrix))
            csr.sum_duplicates()
            csr.eliminate_zeros()
            csr.sort_indices()
            record["nnz"] = int(csr.nnz)
            payload = {
                "name": source.stem,
                "shape": (int(csr.shape[0]), int(csr.shape[1])),
                "nnz": int(csr.nnz),
                "crow_indices": torch.from_numpy(csr.indptr.astype(np.int64, copy=True)),
                "col_indices": torch.from_numpy(csr.indices.astype(np.int64, copy=True)),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
            record.update(included=True, reason="converted", output=destination.relative_to(output_root).as_posix())
            converted += 1
        except Exception as exc:
            record["reason"] = f"error:{type(exc).__name__}:{exc}"
        rows_out.append(record)
        print(f"[{record['reason']}] {relative}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "source", "source_bytes", "source_sha256", "rows", "cols", "nnz", "included", "reason", "output"]
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Converted or retained: {converted}/{len(files)}")
    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
