"""Download only the frozen final-test-v1 assets using anonymous pinned URLs."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import sys
import time
from pathlib import Path

import google_crc32c
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusionvc.final_evaluation import (  # noqa: E402
    DOWNLOAD_RECEIPT,
    PROTOCOL,
    atomic_json,
    ensure_space,
    test_asset_specs,
    validate_selection,
    verified_test_download,
)
from fusionvc.io import read_json  # noqa: E402

CHUNK_SIZE = 32 * 1024 * 1024
WORKERS = 8


def headers():
    # No credentials, cloud account, billing project, or requester-pays header.
    return {"Accept-Encoding": "identity", "User-Agent": "FusionVC-final-test-downloader/1.0"}


def validate_remote(spec: dict) -> dict:
    response = requests.head(spec["url"], headers=headers(), timeout=(20, 45), allow_redirects=True)
    response.raise_for_status()
    if (
        response.status_code != 200
        or response.headers.get("Content-Length") != str(spec["size_bytes"])
        or response.headers.get("x-goog-generation") != spec["generation"]
        or f"crc32c={spec['crc32c']}"
        not in {value.strip() for value in response.headers.get("x-goog-hash", "").split(",")}
        or response.headers.get("Accept-Ranges") != "bytes"
    ):
        raise ValueError(f"Pinned official object metadata changed: {spec['path']}")
    return {
        "status": response.status_code,
        "content_length": response.headers["Content-Length"],
        "generation": response.headers["x-goog-generation"],
        "hash": response.headers["x-goog-hash"],
        "anonymous_access": True,
    }


def checksum(path: Path, spec: dict) -> dict:
    if path.stat().st_size != spec["size_bytes"]:
        raise ValueError(f"Final test download size mismatch: {path}")
    crc, sha = google_crc32c.Checksum(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            crc.update(block)
            sha.update(block)
    actual = base64.b64encode(crc.digest()).decode("ascii")
    if actual != spec["crc32c"]:
        raise ValueError(f"Final test CRC32C differs from the pinned official object: {path}")
    return {"sha256": sha.hexdigest(), "crc32c": actual, "verified_against_source_checksum": True}


def download_object(root: Path, spec: dict, selection_sha: str, workers: int = WORKERS) -> dict:
    remote = validate_remote(spec)
    path = root / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    part, resume = path.with_name(path.name + ".part"), path.with_name(path.name + ".resume.json")
    if not path.exists():
        state = {
            "url": spec["url"],
            "generation": spec["generation"],
            "size_bytes": spec["size_bytes"],
            "chunk_size": CHUNK_SIZE,
            "selection_sha256": selection_sha,
            "complete": [],
        }
        if part.exists() or resume.exists():
            if not part.is_file() or not resume.is_file():
                raise ValueError("Incomplete final test transfer has no matching resume state")
            previous = read_json(resume)
            if (
                any(previous.get(key) != value for key, value in state.items() if key != "complete")
                or part.stat().st_size != spec["size_bytes"]
            ):
                raise ValueError("Final test partial file belongs to a different frozen transfer")
            state = previous
        else:
            ensure_space(root, spec["size_bytes"])
            with part.open("wb") as stream:
                stream.truncate(spec["size_bytes"])
            atomic_json(resume, state)
        count = (spec["size_bytes"] + CHUNK_SIZE - 1) // CHUNK_SIZE
        done = set(state["complete"])
        if any(not isinstance(i, int) or not 0 <= i < count for i in done):
            raise ValueError("Invalid final test transfer chunk state")
        ensure_space(root, 0)

        def transfer(index):
            lo = index * CHUNK_SIZE
            hi = min(spec["size_bytes"], lo + CHUNK_SIZE) - 1
            for attempt in range(4):
                try:
                    with requests.get(
                        spec["url"],
                        headers={**headers(), "Range": f"bytes={lo}-{hi}"},
                        timeout=(20, 60),
                        stream=True,
                    ) as response:
                        response.raise_for_status()
                        if (
                            response.status_code != 206
                            or response.headers.get("Content-Range")
                            != f"bytes {lo}-{hi}/{spec['size_bytes']}"
                            or response.headers.get("x-goog-generation") != spec["generation"]
                        ):
                            raise ValueError("Official server did not honor the pinned byte range")
                        size = 0
                        with part.open("r+b", buffering=0) as stream:
                            stream.seek(lo)
                            for block in response.iter_content(1024 * 1024):
                                size += len(block)
                                if size > hi - lo + 1:
                                    raise ValueError(
                                        "Official range response exceeded its requested size"
                                    )
                                stream.write(block)
                        if size != hi - lo + 1:
                            raise ValueError("Incomplete official range response")
                    return index
                except (requests.RequestException, OSError, ValueError) as error:
                    print(
                        f"FINAL DOWNLOAD retry range={index} attempt={attempt + 1}: {error}",
                        flush=True,
                    )
                    if attempt == 3:
                        raise
                    time.sleep(min(2**attempt, 8))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(transfer, i) for i in range(count) if i not in done]
            last_report = 0.0
            try:
                for future in concurrent.futures.as_completed(futures):
                    done.add(future.result())
                    state["complete"] = sorted(done)
                    atomic_json(resume, state)
                    if time.monotonic() - last_report > 15 or len(done) == count:
                        print(
                            f"FINAL DOWNLOAD {path.name}: {len(done)}/{count} verified ranges",
                            flush=True,
                        )
                        last_report = time.monotonic()
            except BaseException:
                # Cancel queued ranges instead of waiting for the entire matrix
                # after one unrecoverable transfer error. Completed ranges stay.
                for future in futures:
                    future.cancel()
                raise
        # A checksum failure retains .part and its state for explicit diagnosis.
        result = checksum(part, spec)
        part.replace(path)
        resume.unlink()
    else:
        # Adoption/reuse verifies the complete original bytes, including CRC32C.
        result = checksum(path, spec)
    return {**spec, **result, "mtime_ns": path.stat().st_mtime_ns, "http_metadata": remote}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, choices=[1, 2, 4, 8, 16, 32], default=WORKERS)
    args = parser.parse_args()
    root = args.root.resolve()
    selection_path = (args.selection or root / "docs/final-model-selection.json").resolve()
    _, digest = validate_selection(root, selection_path)
    status_path = root / DOWNLOAD_RECEIPT
    if status_path.exists() and read_json(status_path).get("selection_sha256") != digest:
        raise ValueError("Test assets were previously accessed under a different frozen selection")
    if status_path.exists() and read_json(status_path).get("status") == "complete":
        verified_test_download(root, digest)
        print("FINAL test download already complete; original immutable receipt reused", flush=True)
        return
    specs = test_asset_specs()
    # Each new object reserves its full allocation. A validated preallocated
    # partial already owns those bytes and needs only the maintained reserve.
    receipt = {
        "protocol": PROTOCOL,
        "selection_sha256": digest,
        "status": "incomplete",
        "files": [],
        "test_used": False,
        "scope": "bytes/checksums only; no test expressions loaded",
    }
    atomic_json(status_path, receipt)
    # Small public metadata first, then the matrix; neither enters model selection.
    for spec in sorted(specs, key=lambda item: item["size_bytes"]):
        receipt["files"].append(download_object(root, spec, digest, args.workers))
        atomic_json(status_path, receipt)
    receipt["status"] = "complete"
    atomic_json(status_path, receipt)
    print("FINAL test assets verified; expression data have not been opened", flush=True)


if __name__ == "__main__":
    main()
