"""Fetch a HANA table and compare TabPFN with baseline classifiers."""

import argparse
import json
from pathlib import Path

from comparison import compare
from config import optional
from hana_client import load_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=optional("HANA_TABLE"), help="Table name in HANA_SCHEMA")
    parser.add_argument("--target", required=True, help="Classification target column")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", default="comparison.json")
    args = parser.parse_args()
    if not args.table:
        parser.error("Set HANA_TABLE in .env or pass --table.")

    result = compare(load_table(args.table), args.target, args.test_size)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))
    print(f"Saved predictions and scores to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
