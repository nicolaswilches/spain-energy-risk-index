"""CLI entry point for the data extraction pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Add src/ to path so grid_risk is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grid_risk.config import DEFAULT_START, DEFAULT_END
from grid_risk.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Spain electricity grid data from REE/ESIOS APIs"
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
        "--no-esios",
        action="store_true",
        help="Skip ESIOS API (use 24h-lagged fallback for forecasts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for parquet file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load .env for ESIOS token
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    esios_token = None
    if not args.no_esios:
        esios_token = os.getenv("ESIOS_TOKEN")
        if not esios_token:
            logging.getLogger(__name__).warning(
                "No ESIOS_TOKEN found in .env — falling back to 24h-lagged actuals"
            )

    output_path = run_pipeline(
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        esios_token=esios_token,
    )

    print(f"\nDone! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
