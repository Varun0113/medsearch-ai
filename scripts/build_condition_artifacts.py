import argparse
from pathlib import Path

from app.assistant.artifacts import build_condition_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build condition grouping artifacts from the medicines CSV.")
    parser.add_argument("--csv", default="app/data/medicines_final.csv", help="Path to medicines CSV")
    parser.add_argument("--rules", default="data/condition_group_rules.json", help="Path to condition grouping rules JSON")
    parser.add_argument("--out", default="indexes/condition_artifacts.json", help="Output artifacts JSON path")
    parser.add_argument("--uses-column", default="uses", help="CSV column containing uses/indications text")
    parser.add_argument("--generic-column", default="generic_name", help="CSV column containing generic name")
    parser.add_argument("--chunksize", type=int, default=20000, help="CSV read chunksize")
    parser.add_argument("--max-variants", type=int, default=200, help="Max variants per group")
    parser.add_argument("--max-generics", type=int, default=100, help="Max generics per group")
    args = parser.parse_args()

    build_condition_artifacts(
        csv_path=Path(args.csv),
        rules_path=Path(args.rules),
        out_path=Path(args.out),
        uses_column=args.uses_column,
        generic_column=args.generic_column,
        chunksize=args.chunksize,
        max_variants_per_group=args.max_variants,
        max_generics_per_group=args.max_generics,
    )


if __name__ == "__main__":
    main()
