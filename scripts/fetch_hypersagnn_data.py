#!/usr/bin/env python3
"""Fetch the exact Hyper-SAGNN data files used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_data import HYPEREDGE_SHA256  # noqa: E402


UPSTREAM_REPOSITORY = "https://github.com/ma-compbio/Hyper-SAGNN.git"
UPSTREAM_COMMIT = "69f2fbe21c455aca084497fb2d26a8207a95decd"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/ma-compbio/Hyper-SAGNN/"
    f"{UPSTREAM_COMMIT}"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_relative(package_relative: str) -> Path:
    prefix = "Hyper-SAGNN-master/"
    if not package_relative.startswith(prefix):
        raise ValueError(f"Unexpected destination path: {package_relative}")
    return Path(package_relative[len(prefix) :])


def copy_or_download(
    package_relative: str,
    destination: Path,
    source_root: Optional[Path],
) -> None:
    upstream_path = upstream_relative(package_relative)
    if source_root is not None:
        source = source_root / upstream_path
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("rb") as input_handle, destination.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
        return

    url = f"{RAW_BASE_URL}/{upstream_path.as_posix()}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "EdgeMask-HGNN-reproducibility-downloader/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output_handle:
            shutil.copyfileobj(response, output_handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT,
        help="Package root under which Hyper-SAGNN-master/data/ will be created",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Copy from a checkout of the pinned upstream commit instead of downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file whose checksum is incorrect",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve() if args.source_root is not None else None
    package_root = args.destination.resolve()
    fetched = 0
    reused = 0

    for relative, expected_hash in HYPEREDGE_SHA256.items():
        destination = package_root / relative
        if destination.is_file():
            observed = sha256(destination)
            if observed == expected_hash:
                print(f"[verified] {relative}")
                reused += 1
                continue
            if not args.force:
                raise RuntimeError(
                    f"Existing file has the wrong SHA-256: {destination}\n"
                    f"observed {observed}\nexpected {expected_hash}\n"
                    "Pass --force to replace it."
                )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            copy_or_download(relative, temporary_path, source_root)
            observed = sha256(temporary_path)
            if observed != expected_hash:
                raise RuntimeError(
                    f"SHA-256 mismatch for {relative}: "
                    f"observed {observed}, expected {expected_hash}"
                )
            os.replace(temporary_path, destination)
            temporary_path = None
            print(f"[fetched] {relative}")
            fetched += 1
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    total_bytes = sum((package_root / path).stat().st_size for path in HYPEREDGE_SHA256)
    print(
        f"Hyper-SAGNN data ready: {fetched} fetched, {reused} reused, "
        f"{total_bytes / 1024**2:.1f} MiB verified."
    )
    print(f"Pinned source: {UPSTREAM_REPOSITORY} at {UPSTREAM_COMMIT}")


if __name__ == "__main__":
    main()
