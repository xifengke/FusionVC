"""Fetch versioned official source files without cloning large model repos."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "scGPT": ("bowang-lab/scGPT", "cebd6fae655b9c585a4807daa3ac31bb764f06b4"),
    "cell-eval": ("ArcInstitute/cell-eval", "6928cf8bd7a706040ccfd13119e4085726dee64a"),
    "cell-eval-vcc2025": ("ArcInstitute/cell-eval", "27c18c4f215a0345c1428329a41fdf7b24c24b50"),
    "scGPT-paper2024": ("bowang-lab/scGPT", "74603ca5d8a2adf2a7420936c73784a1fef01597"),
}
GF_REVISION = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"


def fetch(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                sha = hashlib.sha256()
                part = path.with_name(path.name + ".part")
                with part.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
                        sha.update(block)
                part.replace(path)
            return {"path": path.relative_to(ROOT).as_posix(), "url": url, "bytes": path.stat().st_size, "sha256": sha.hexdigest()}
        except Exception:
            if attempt == 2:
                raise


def main() -> None:
    results = []
    for name, (repo, revision) in SOURCES.items():
        archive = ROOT / "vendor" / f"{name}-{revision[:12]}.zip"
        result = fetch(f"https://codeload.github.com/{repo}/zip/{revision}", archive)
        result["revision"] = revision
        results.append(result)
        destination = ROOT / "vendor" / name
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                relative = Path(*Path(member.filename).parts[1:])
                if not relative.parts or member.is_dir():
                    continue
                target = (destination / relative).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Archive path outside destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
        print(f"READY official {name} source: {revision}", flush=True)

    response = requests.get(f"https://huggingface.co/api/models/ctheodoris/Geneformer/tree/{GF_REVISION}?recursive=true&limit=1000", timeout=30)
    response.raise_for_status()
    selected = [item for item in response.json() if item["type"] == "file" and (
        item["path"].startswith("geneformer/") and item["path"].endswith((".py", ".pkl"))
        or item["path"] in ["README.md", "LICENSE", "LICENSE.txt", "setup.py", "requirements.txt"])]
    def fetch_gf(item: dict) -> dict:
        return fetch(f"https://huggingface.co/ctheodoris/Geneformer/resolve/{GF_REVISION}/{item['path']}", ROOT / "vendor" / "Geneformer" / item["path"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results.extend(pool.map(fetch_gf, selected))
    print(f"READY official Geneformer source: {GF_REVISION} ({len(selected)} files)", flush=True)
    results.append(fetch("https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv", ROOT / "references" / "gene_info.csv"))
    reference = ROOT / "references" / "vcc2025.pdf"
    original_pdf = Path("E:/各种文献/AIVC/PIIS0092867425006750____test.pdf")
    if original_pdf.exists():
        shutil.copyfile(original_pdf, reference)
        results.append({"path": reference.relative_to(ROOT).as_posix(), "source": "User-provided original PDF", "bytes": reference.stat().st_size})
    (ROOT / "docs/source-provenance.json").write_text(json.dumps({"sources": results, "geneformer_revision": GF_REVISION, "evaluator_note": "Use cell-eval-vcc2025 (0.6.6) and the recorded baseline evaluation protocol, not the current evaluator by default."}, ensure_ascii=False, indent=2), "utf-8")
    print("Source preparation complete", flush=True)


if __name__ == "__main__":
    main()
