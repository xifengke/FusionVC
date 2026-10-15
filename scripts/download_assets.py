"""Download the initial FusionVC assets with resumable, verified transfers.

Run using the project's .venv-download Python. Test matrices and pretraining
corpora are intentionally outside this initial download manifest.
"""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import gdown
import google_crc32c
import requests

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "docs/download-metadata.json").read_text("utf-8"))
STATUS_PATH = ROOT / "docs/download-status.json"
CHUNK_SIZE = 32 * 1024 * 1024
THREADS = 6
STATUS = {"phase": "initial", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": []}
LOCAL = threading.local()


def emit(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    temporary.replace(path)


def session() -> requests.Session:
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers.update({"User-Agent": "FusionVC-asset-downloader/1.0", "Accept-Encoding": "identity"})
    return LOCAL.session


def verify(path: Path, spec: dict) -> dict:
    expected = spec.get("size_bytes")
    size = path.stat().st_size
    if expected is not None and size != expected:
        raise ValueError(f"Size mismatch: {path.name}: {size} != {expected}")
    sha = hashlib.sha256()
    crc = google_crc32c.Checksum()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
            crc.update(block)
            md5.update(block)
    crc_text = base64.b64encode(crc.digest()).decode("ascii")
    if spec.get("crc32c") and crc_text != spec["crc32c"]:
        raise ValueError(f"CRC32C mismatch: {path.name}")
    if spec.get("md5Hash") and base64.b64encode(md5.digest()).decode("ascii") != spec["md5Hash"]:
        raise ValueError(f"MD5 mismatch: {path.name}")
    if spec.get("sha256") and sha.hexdigest() != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    return {"size_bytes": size, "sha256": sha.hexdigest(), "crc32c": crc_text,
            "verified_against_source_checksum": bool(spec.get("crc32c") or spec.get("md5Hash") or spec.get("sha256"))}


def ranged_download(spec: dict, path: Path) -> None:
    size = spec["size_bytes"]
    part = path.with_name(path.name + ".part")
    state_path = path.with_name(path.name + ".resume.json")
    chunks = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
    state = {"url": spec["url"], "size_bytes": size, "chunk_size": CHUNK_SIZE, "complete": []}
    if part.exists() and state_path.exists():
        previous = json.loads(state_path.read_text("utf-8"))
        if all(previous.get(k) == state[k] for k in ("url", "size_bytes", "chunk_size")) and part.stat().st_size == size:
            state = previous
        else:
            raise ValueError(f"Incompatible partial file: {part}")
    elif part.exists():
        raise ValueError(f"Partial file has no resume metadata: {part}")
    else:
        with part.open("wb") as stream:
            stream.truncate(size)
        save_json(state_path, state)
    done = set(state["complete"])
    pending = [i for i in range(chunks) if i not in done]
    initial = sum(min(CHUNK_SIZE, size - i * CHUNK_SIZE) for i in done)
    start_time = time.monotonic()
    last_report = 0.0
    emit(f"TRANSFER {path.name}: {initial / 1e9:.2f}/{size / 1e9:.2f} GB, {THREADS} streams")

    def transfer(index: int) -> int:
        lo = index * CHUNK_SIZE
        hi = min(size, lo + CHUNK_SIZE) - 1
        error = None
        for attempt in range(6):
            try:
                with session().get(spec["url"], headers={"Range": f"bytes={lo}-{hi}"}, stream=True, timeout=(20, 45)) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError(f"Server did not honor Range: HTTP {response.status_code}")
                    expected_range = f"bytes {lo}-{hi}/{size}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Unexpected Content-Range")
                    written = 0
                    with part.open("r+b", buffering=0) as stream:
                        stream.seek(lo)
                        for block in response.iter_content(1024 * 1024):
                            if written + len(block) > hi - lo + 1:
                                raise ValueError("Range response exceeded expected length")
                            stream.write(block)
                            written += len(block)
                    if written != hi - lo + 1:
                        raise ValueError("Incomplete range response")
                return index
            except Exception as exc:
                error = exc
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"Chunk {index} failed: {error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(transfer, index) for index in pending]
        for future in concurrent.futures.as_completed(futures):
            index = future.result()
            done.add(index)
            state["complete"] = sorted(done)
            save_json(state_path, state)
            now = time.monotonic()
            if now - last_report >= 15 or len(done) == chunks:
                transferred = sum(min(CHUNK_SIZE, size - i * CHUNK_SIZE) for i in done)
                rate = (transferred - initial) / max(now - start_time, 0.01)
                eta = (size - transferred) / max(rate, 1)
                emit(f"PROGRESS {path.name}: {transferred / 1e9:.2f}/{size / 1e9:.2f} GB, {rate / 1e6:.1f} MB/s, ETA {eta / 60:.1f} min")
                last_report = now
    emit(f"CHECKSUM {path.name}")
    result = verify(part, spec)
    part.replace(path)
    state_path.unlink()
    spec.update(result)


def simple_download(spec: dict, path: Path) -> None:
    part = path.with_name(path.name + ".part")
    for attempt in range(4):
        try:
            if spec.get("drive_id"):
                result = gdown.download(id=spec["drive_id"], output=str(part), quiet=True, use_cookies=False)
                if not result:
                    raise RuntimeError("Google Drive did not provide the file")
            else:
                with session().get(spec["url"], stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    with part.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
            if path.suffix == ".json":
                json.loads(part.read_text("utf-8"))
            if path.suffix == ".pdf":
                with part.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise ValueError("Downloaded response is not a PDF")
            result = verify(part, spec)
            part.replace(path)
            spec.update(result)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))


def run_spec(spec: dict) -> None:
    path = ROOT / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(spec)
    STATUS["files"].append(record)
    save_json(STATUS_PATH, STATUS)
    try:
        if path.exists():
            emit(f"VERIFY EXISTING {spec['path']}")
            record.update(verify(path, spec))
        elif spec.get("size_bytes", 0) > 1024 ** 3:
            ranged_download(record, path)
        else:
            emit(f"DOWNLOAD {spec['path']}")
            simple_download(record, path)
        record["status"] = "verified"
        emit(f"READY {spec['path']}: {path.stat().st_size / 1e6:.2f} MB")
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        emit(f"FAILED {spec['path']}: {exc}")
    save_json(STATUS_PATH, STATUS)


def main() -> None:
    assets = []
    for item in META["vcc2025_files"]:
        if "/test/" in item["name"]:
            continue
        relative = item["name"].split("/2025/", 1)[1]
        assets.append({**item, "size_bytes": int(item["size"]),
                       "url": "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/" + item["name"] + "?generation=" + item["generation"],
                       "path": "data/raw/vcc2025/" + relative})
    for item in META["scgpt_files"]:
        assets.append({**item, "drive_id": item["file_id"],
                       "url": "https://drive.google.com/uc?id=" + item["file_id"],
                       "path": "models/scgpt/whole-human/" + item["name"]})
    # Keep future retries on the same checkpoint and tokenizer revision.
    revision = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
    tree_response = session().get(f"https://huggingface.co/api/models/ctheodoris/Geneformer/tree/{revision}?recursive=true&limit=1000", timeout=30)
    tree_response.raise_for_status()
    tree = {item["path"]: item for item in tree_response.json() if item["type"] == "file"}
    for item in META["geneformer_v1_files"]:
        remote = ("geneformer/gene_dictionaries_30m/" if item["name"].endswith(".pkl") else "Geneformer-V1-10M/") + item["name"]
        assets.append({**item, "revision": revision, "sha256": tree[remote].get("lfs", {}).get("oid"),
                       "url": f"https://huggingface.co/ctheodoris/Geneformer/resolve/{revision}/{remote}",
                       "path": "models/geneformer-v1/" + ("dictionaries/" if item["name"].endswith(".pkl") else "") + item["name"]})
    needed = sum(spec["size_bytes"] for spec in assets if not (ROOT / spec["path"]).exists())
    free = shutil.disk_usage(ROOT).free
    emit(f"INITIAL ASSETS {needed / 1e9:.2f} GB; FREE {free / 1e9:.2f} GB")
    if free < needed + 15 * 10 ** 9:
        raise RuntimeError("Insufficient disk space with 15 GB working reserve")
    save_json(ROOT / "docs/initial-download-manifest.json", {"files": assets, "excluded": ["test matrices", "FASTQ", "pretraining corpora"], "total_bytes": sum(spec["size_bytes"] for spec in assets)})
    # Small files and model weights first; then the large matrices.
    for spec in sorted(assets, key=lambda item: item["size_bytes"]):
        run_spec(spec)
    STATUS["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    STATUS["status"] = "complete" if all(x.get("status") == "verified" for x in STATUS["files"]) else "incomplete"
    save_json(STATUS_PATH, STATUS)
    emit(f"INITIAL DOWNLOAD {STATUS['status'].upper()}")
    if STATUS["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
