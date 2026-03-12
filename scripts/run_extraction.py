"""CLI entry point for the daily data extraction pipeline.

Usage:
    PYTHONPATH=src uv run python scripts/run_extraction.py
    PYTHONPATH=src uv run python scripts/run_extraction.py --start 2021-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grid_risk.config import DEFAULT_START, DEFAULT_END
from grid_risk.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and merge daily grid + weather + calendar data",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help=f"Start date (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=DEFAULT_END,
        help=f"End date (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for parquet file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\nSpain Energy Grid — Daily Extraction Pipeline")
    print(f"  Range: {args.start} to {args.end}")
    print(f"  Output: {args.output_dir}\n")

    output_path = run_pipeline(
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )

    print(f"\nDone. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
