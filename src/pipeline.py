"""Command-line interface for running the AQI prediction pipeline stages."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.cleaning import clean_data
from src.config_loader import load_config
from src.features import (
    add_cyclical_features,
    create_lag_features,
    create_pollutant_interactions,
    create_rolling_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("AQI Bangladesh.csv")
DEFAULT_OUTPUT_DIR = Path("notebooks/data")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run AQI prediction pipeline stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to raw CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for intermediate pipeline outputs",
    )
    parser.add_argument(
        "--stage",
        choices=["clean", "features", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    return parser.parse_args(argv)


def run_clean(
    df_raw: pd.DataFrame, config: Dict[str, Any], output_dir: Path
) -> pd.DataFrame:
    logger.info("Stage: cleaning...")
    df_clean = clean_data(df_raw, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_dir / "df_clean.csv", index=False)
    logger.info(
        "Cleaned data saved → %s/df_clean.csv (%d rows)", output_dir, len(df_clean)
    )
    return df_clean


def run_features(
    df_clean: pd.DataFrame, config: Dict[str, Any], output_dir: Path
) -> pd.DataFrame:
    logger.info("Stage: feature engineering...")
    pollutant_cols = config["data"]["pollutant_cols"]
    aqi_col = config["data"]["aqi_col"]
    lag_cols = pollutant_cols + [aqi_col]

    df_feat = add_cyclical_features(df_clean)
    df_feat = create_lag_features(df_feat, lag_cols=lag_cols, config=config)
    df_feat = create_rolling_features(df_feat, cols=pollutant_cols, config=config)
    df_feat = create_pollutant_interactions(df_feat)

    # Drop rows with lag-NaN values (the first 24 hours per city) for consistency with notebook 03
    lag24_cols = [c for c in df_feat.columns if c.endswith("_lag24")]
    df_feat = df_feat.dropna(subset=lag24_cols).reset_index(drop=True)

    df_feat.to_csv(output_dir / "df_feat.csv", index=False)
    logger.info(
        "Feature data saved → %s/df_feat.csv (%d rows, %d cols)",
        output_dir,
        len(df_feat),
        len(df_feat.columns),
    )
    return df_feat


def main(argv=None):
    args = parse_args(argv)
    config = load_config()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    logger.info("Loading raw data from %s...", args.input)
    df_raw = pd.read_csv(args.input)
    logger.info("Loaded %d rows, %d columns", len(df_raw), len(df_raw.columns))

    df_clean = None
    if args.stage in ("clean", "all"):
        df_clean = run_clean(df_raw, config, args.output_dir)

    if args.stage in ("features", "all"):
        if df_clean is None:
            clean_path = args.output_dir / "df_clean.csv"
            if not clean_path.exists():
                logger.error(
                    "Clean data not found at %s — run --stage clean first", clean_path
                )
                sys.exit(1)
            df_clean = pd.read_csv(clean_path)
            datetime_col = config["data"]["datetime_col"]
            df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col])
        run_features(df_clean, config, args.output_dir)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
