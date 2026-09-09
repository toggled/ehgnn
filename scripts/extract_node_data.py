#!/usr/bin/env python3
"""Extract only the node-classification files required by the paper."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_data import NODE_FILES  # noqa: E402


ARCHIVE_SHA256 = "092271df6841226eed6da31dc84ac463b64379ed5a40ff4b11885b44cc448f50"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Downloaded data.zip file")
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT,
        help="Package root in which the data/ directory will be created",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip verification when using an independently obtained archive",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if not args.skip_hash_check:
        observed = sha256(args.archive)
        if observed != ARCHIVE_SHA256:
            raise RuntimeError(
                f"Archive SHA-256 mismatch: observed {observed}, expected {ARCHIVE_SHA256}"
            )

    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        missing = [name for name in NODE_FILES if name not in names]
        if missing:
            raise RuntimeError("Archive is missing required files: " + ", ".join(missing))
        for relative in NODE_FILES:
            destination = args.destination / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(relative) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    total_bytes = sum((args.destination / path).stat().st_size for path in NODE_FILES)
    print(
        f"Extracted {len(NODE_FILES)} files ({total_bytes / 1024**2:.1f} MiB) "
        f"under {args.destination / 'data'}."
    )


if __name__ == "__main__":
    main()
