"""Fetch the reference checkpoints listed in weights/MANIFEST.json and verify their sha256.

    python scripts/fetch_weights.py --out $SPB_WEIGHTS_DIR [--from-local DIR] [--manifest weights/MANIFEST.json]

Each entry is `<task>/<stem>.<ext>` -> {role, url, sha256, bytes, model}; an entry without a url is a small
artifact versioned next to the manifest (scaler, meta) and is copied. Exits 1 on an empty manifest, a missing
file or any checksum mismatch. Run on a node with internet; jobs never download.
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "weights" / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, info: dict, out: Path, manifest_dir: Path, from_local: Path | None) -> None:
    dest = out / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if from_local is not None:
        shutil.copy(from_local / name, dest)
    elif not info.get("url"):
        shutil.copy(manifest_dir / name, dest)
    else:
        tmp = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(info["url"]) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.replace(dest)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--from-local", type=Path, help="copy checkpoints from this directory instead of downloading")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    a = p.parse_args(argv)
    files = json.loads(a.manifest.read_text())["files"]
    if not files:
        print(f"{a.manifest} lists no weights yet", file=sys.stderr)
        return 1
    ok = True
    for name, info in files.items():
        dest = a.out / name
        if not (dest.exists() and sha256(dest) == info["sha256"]):
            try:
                fetch(name, info, a.out, a.manifest.parent, a.from_local)
            except (OSError, ValueError) as e:
                print(f"{name}: {e}", file=sys.stderr)
                ok = False
                continue
        good = sha256(dest) == info["sha256"]
        ok &= good
        print(f"{name} [{info.get('role', 'reference')}]: {'ok' if good else 'CHECKSUM MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
