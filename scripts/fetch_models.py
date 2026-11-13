"""Download the GGUF models listed in models/manifest.json from Hugging Face.

usage: python fetch_models.py [--models-dir DIR] [--max-gib N] [--only a,b] [--with-mmproj] [--all] [--list]
Fetches the models that have artifacts in the package. --max-gib skips files larger than N GiB, --only
keeps files whose name or repo contains one of the substrings. Interrupted downloads resume.
Uses curl (built into Windows 10 and later).
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wanted(item, a):
    name = f"{item['repo']}/{item['path']}".lower()
    if "mmproj" in name:
        return a.with_mmproj
    if not a.all and not (ROOT / "artifacts" / Path(item["path"]).stem).is_dir():
        return False  # nothing in the package to test it with (e.g. the bge-m3 encoder)
    if a.max_gib and item["size"] / 2**30 > a.max_gib:
        return False
    return not a.only or any(s.lower() in name for s in a.only.split(","))


def download(item, models_dir):
    dst = models_dir / item["repo"].replace("/", "__") / Path(item["path"]).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == item["size"]:
        return f"skip {dst.name}"
    url = f"https://huggingface.co/{item['repo']}/resolve/main/{item['path']}"
    for _ in range(8):  # transfers drop now and then; -C - resumes
        subprocess.run(["curl", "-s", "-L", "--ssl-no-revoke", "--fail", "--retry", "3", "-C", "-", "-o", str(dst), url])
        if dst.exists() and dst.stat().st_size >= item["size"]:
            break
    ok = dst.exists() and dst.stat().st_size == item["size"]
    return f"{'ok  ' if ok else 'FAIL'} {dst.name} ({item['size'] / 2**30:.2f} GiB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--max-gib", type=float)
    ap.add_argument("--only")
    ap.add_argument("--with-mmproj", action="store_true", help="also fetch MiniCPM-V's vision projector")
    ap.add_argument("--all", action="store_true", help="also fetch models the package has no artifacts for")
    ap.add_argument("--list", action="store_true", help="print what would be downloaded and stop")
    a = ap.parse_args()
    models_dir = Path(a.models_dir).resolve()
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    items = [i for i in manifest if wanted(i, a)]
    print(f"{len(items)} files, {sum(i['size'] for i in items) / 2**30:.1f} GiB -> {models_dir}")
    if a.list:
        for i in items:
            print(f"  {i['size'] / 2**30:6.2f} GiB  {i['repo']}  {i['path']}")
        return 0
    models_dir.mkdir(parents=True, exist_ok=True)
    if models_dir != ROOT / "models":
        (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    failed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in pool.map(lambda i: download(i, models_dir), items):
            print(line, flush=True)
            failed += line.startswith("FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
