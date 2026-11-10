"""Download, md5-verify and extract the Zenodo records into $SPB_DATA_DIR/<record>/<archive stem>/, recording the record revision."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from smartpix_bench.common.io import data_dir

RECORDS = {"filter": "10783560", "regression": "18472791"}
API = "https://zenodo.org/api/records/{}"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(archive: Path, dest: Path) -> None:
    """Extract, skipping macOS `._*` AppleDouble and `__MACOSX` entries."""
    def junk(name):
        return Path(name).name.startswith("._") or "__MACOSX" in Path(name).parts
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest, [m for m in z.namelist() if not junk(m)])
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest, [m for m in t.getmembers() if not junk(m.name)], filter="data")


def fetch(record: str, root: Path) -> None:
    with urllib.request.urlopen(API.format(record)) as r:
        meta = json.load(r)
    dest = root / record
    dest.mkdir(parents=True, exist_ok=True)
    for f in meta["files"]:
        algo, want = f["checksum"].split(":")
        assert algo == "md5", f["checksum"]
        out = dest / f["key"]
        if not (out.exists() and md5(out) == want):
            print(f"downloading {f['key']} ({f['size'] / 1e9:.2f} GB)", flush=True)
            urllib.request.urlretrieve(f["links"]["self"], out)
        if md5(out) != want:
            raise SystemExit(f"md5 mismatch for {out}")
        target = dest / f["key"].removesuffix(".zip").removesuffix(".tar.gz")
        if target != out and not target.exists():
            partial = target.with_name(target.name + ".partial")  # renamed only once complete
            shutil.rmtree(partial, ignore_errors=True)
            extract(out, partial)
            partial.rename(target)
    rev = {k: meta.get(k) for k in ("id", "doi", "revision", "updated")}
    rev["version"] = meta.get("metadata", {}).get("version")
    rev["files"] = {f["key"]: f["checksum"] for f in meta["files"]}
    (dest / "REVISION.json").write_text(json.dumps(rev, indent=2) + "\n")
    print(json.dumps(rev, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=[*RECORDS, "all"], default="all")
    p.add_argument("--data-dir", type=Path, default=None, help="default: $SPB_DATA_DIR or data/zenodo")
    a = p.parse_args()
    for task, record in RECORDS.items():
        if a.task in (task, "all"):
            fetch(record, a.data_dir or data_dir())


if __name__ == "__main__":
    main()
