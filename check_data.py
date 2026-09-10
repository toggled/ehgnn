#!/usr/bin/env python3
"""Check that the datasets required by the paper are in the expected paths."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent

NODE_FILES = (
    *(f"data/hetero/{dataset}/{name}.pickle" for dataset in ("actor", "twitch", "pokec") for name in ("features", "hypergraph", "labels")),
    "data/AllSet_all_raw_data/yelp/yelp_restaurant_business_stars.csv",
    "data/AllSet_all_raw_data/yelp/yelp_restaurant_incidence_H.csv",
    "data/AllSet_all_raw_data/yelp/yelp_restaurant_latlong.csv",
    "data/AllSet_all_raw_data/yelp/yelp_restaurant_locations.csv",
    "data/AllSet_all_raw_data/yelp/yelp_restaurant_name.csv",
    *(f"data/AllSet_all_raw_data/coauthorship/dblp/{name}.pickle" for name in ("features", "hypergraph", "labels")),
    "data/AllSet_all_raw_data/walmart-trips/hyperedges-walmart-trips.txt",
    "data/AllSet_all_raw_data/walmart-trips/node-labels-walmart-trips.txt",
    "data/AllSet_all_raw_data/walmart-trips/label-names-walmart-trips.txt",
    *(f"data/AllSet_all_raw_data/cocitation/cora/{name}.pickle" for name in ("features", "hypergraph", "labels")),
)

HYPEREDGE_FILES = tuple(
    f"Hyper-SAGNN-master/data/{dataset}/{split}_data.npz"
    for dataset in ("wordnet", "drug", "MovieLens")
    for split in ("train", "test")
)

HYPEREDGE_SHA256 = {
    "Hyper-SAGNN-master/data/wordnet/train_data.npz": "dc6575c71c5744d12824119d722cdddcff93af0e4c202fd38c5ffc0512921399",
    "Hyper-SAGNN-master/data/wordnet/test_data.npz": "49fa119ca83cf1f2660e1ac1bcbabcebfc74d086e3a11b64d0945d6afca10f87",
    "Hyper-SAGNN-master/data/drug/train_data.npz": "304577d37203e72ac10954ef2e9ab75e2113645af786b10d5a82db21f2db7acc",
    "Hyper-SAGNN-master/data/drug/test_data.npz": "77892d6f299740829c955a69fa21cb560424941439c2e29909fe53e116c5808f",
    "Hyper-SAGNN-master/data/MovieLens/train_data.npz": "50246a9bde10b72968f7e857db741a03ec9b7c2d58908ff6b6cb6d0a0b06e5f4",
    "Hyper-SAGNN-master/data/MovieLens/test_data.npz": "9b3b199fb9d2ee20eac0ac0d108a2a307849a2c28a95689ade342d62bc3537bc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("node", "hyperedge", "all"), default="all")
    args = parser.parse_args()
    expected = []
    if args.scope in {"node", "all"}:
        expected.extend(NODE_FILES)
    if args.scope in {"hyperedge", "all"}:
        expected.extend(HYPEREDGE_FILES)
    missing = [relative for relative in expected if not (ROOT / relative).is_file()]
    if missing:
        print("Missing required dataset files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)
    mismatched = []
    if args.scope in {"hyperedge", "all"}:
        for relative, expected_hash in HYPEREDGE_SHA256.items():
            observed_hash = sha256(ROOT / relative)
            if observed_hash != expected_hash:
                mismatched.append((relative, observed_hash, expected_hash))
    if mismatched:
        print("Hyper-SAGNN dataset checksum mismatches:")
        for relative, observed, expected_hash in mismatched:
            print(f"  {relative}: observed {observed}, expected {expected_hash}")
        raise SystemExit(1)

    verified = len(HYPEREDGE_SHA256) if args.scope in {"hyperedge", "all"} else 0
    suffix = f"; {verified} Hyper-SAGNN checksums verified" if verified else ""
    print(f"Data check passed: {len(expected)} required files found{suffix}.")


if __name__ == "__main__":
    main()
