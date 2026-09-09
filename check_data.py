#!/usr/bin/env python3
"""Check that the datasets required by the paper are in the expected paths."""

from __future__ import annotations

import argparse
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
    print(f"Data check passed: {len(expected)} required files found.")


if __name__ == "__main__":
    main()
