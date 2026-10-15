"""Small reproducibility utilities and bounded-memory HDF5 access."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def strings(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # The default memory-efficient attention backward warned about nondeterminism.
    # This small baseline fits the deterministic math backend comfortably.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.set_num_threads(4)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CountsReader:
    """Read contiguous CSR row blocks; never load a full expression matrix."""

    def __init__(self, path: Path):
        self.file = h5py.File(path, "r")
        self.x = self.file["X"]
        if self.x.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("Expected official CSR count matrix")
        self.shape = tuple(int(n) for n in self.x.attrs["shape"])
        self.indptr = self.x["indptr"][:]
        target = self.file["obs/target_gene"]
        self.categories = strings(target["categories"][:])
        self.codes = target["codes"][:].astype(np.int32)
        self.genes = strings(self.file["var"][self.file["var"].attrs["_index"]][:])
        self.cell_ids = strings(self.file["obs"][self.file["obs"].attrs["_index"]][:])
        if np.any(self.codes < 0):
            raise ValueError("Unlabelled cells in official input")

    def block(self, start: int, stop: int) -> csr_matrix:
        a, b = int(self.indptr[start]), int(self.indptr[stop])
        return csr_matrix(
            (
                self.x["data"][a:b].astype(np.float32),
                self.x["indices"][a:b],
                self.indptr[start : stop + 1] - a,
            ),
            shape=(stop - start, self.shape[1]),
        )

    def close(self) -> None:
        self.file.close()

    def rows(self, indices: np.ndarray) -> csr_matrix:
        """Read selected CSR rows in caller order, including repeated samples."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.shape[0]):
            raise ValueError("Row indices must be a valid one-dimensional selection")
        if len(indices) == 0:
            return csr_matrix((0, self.shape[1]), dtype=np.float32)
        # Read each unique row once; preserve duplicates when assembling the set.
        saved = {}
        for index in np.unique(indices):
            start, stop = int(self.indptr[index]), int(self.indptr[index + 1])
            saved[int(index)] = (
                self.x["data"][start:stop].astype(np.float32),
                self.x["indices"][start:stop],
            )
        values, columns, pointers = [], [], [0]
        for index in indices:
            data, genes = saved[int(index)]
            values.append(data)
            columns.append(genes)
            pointers.append(pointers[-1] + len(data))
        return csr_matrix(
            (np.concatenate(values), np.concatenate(columns), np.asarray(pointers)),
            shape=(len(indices), self.shape[1]),
        )


def normalize_log(counts: csr_matrix, target_sum: float) -> csr_matrix:
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("Cannot normalize zero-library cells")
    result = counts.multiply((target_sum / library)[:, None]).tocsr()
    np.log1p(result.data, out=result.data)
    return result
