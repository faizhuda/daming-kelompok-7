from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import run_clean, run_features


@pytest.fixture
def sample_clean_df(sample_raw_data, sample_config, tmp_path):
    """Returns a cleaned DataFrame produced by run_clean, for use in feature tests."""
    output_dir = tmp_path / "data"
    return run_clean(sample_raw_data, sample_config, output_dir)


# ── run_clean ─────────────────────────────────────────────────────────────────


def test_run_clean_returns_dataframe(sample_raw_data, sample_config, tmp_path):
    """Verifies run_clean returns a non-empty DataFrame."""
    output_dir = tmp_path / "data"
    result = run_clean(sample_raw_data, sample_config, output_dir)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_run_clean_saves_csv(sample_raw_data, sample_config, tmp_path):
    """Verifies run_clean writes df_clean.csv to the output directory."""
    output_dir = tmp_path / "data"
    run_clean(sample_raw_data, sample_config, output_dir)
    assert (output_dir / "df_clean.csv").exists()


def test_run_clean_drops_carbon_dioxide(sample_raw_data, sample_config, tmp_path):
    """Verifies carbon_dioxide is dropped during cleaning."""
    output_dir = tmp_path / "data"
    result = run_clean(sample_raw_data, sample_config, output_dir)
    assert "carbon_dioxide" not in result.columns


def test_run_clean_no_missing_pollutants(sample_raw_data, sample_config, tmp_path):
    """Verifies no NaN values remain in pollutant columns after cleaning."""
    output_dir = tmp_path / "data"
    result = run_clean(sample_raw_data, sample_config, output_dir)
    for col in sample_config["data"]["pollutant_cols"]:
        assert not result[col].isna().any(), f"NaN found in {col} after clean"


# ── run_features ──────────────────────────────────────────────────────────────


def test_run_features_returns_dataframe(sample_clean_df, sample_config, tmp_path):
    """Verifies run_features returns a non-empty DataFrame."""
    output_dir = tmp_path / "data"
    result = run_features(sample_clean_df, sample_config, output_dir)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_run_features_saves_csv(sample_clean_df, sample_config, tmp_path):
    """Verifies run_features writes df_feat.csv to the output directory."""
    output_dir = tmp_path / "data"
    run_features(sample_clean_df, sample_config, output_dir)
    assert (output_dir / "df_feat.csv").exists()


def test_run_features_adds_more_columns(sample_clean_df, sample_config, tmp_path):
    """Verifies feature engineering expands the column count."""
    output_dir = tmp_path / "data"
    result = run_features(sample_clean_df, sample_config, output_dir)
    assert len(result.columns) > len(sample_clean_df.columns)


def test_run_features_adds_cyclical_columns(sample_clean_df, sample_config, tmp_path):
    """Verifies cyclical encoding columns are present in the output."""
    output_dir = tmp_path / "data"
    result = run_features(sample_clean_df, sample_config, output_dir)
    for col in ("hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"):
        assert col in result.columns, f"Missing cyclical column: {col}"


def test_run_features_adds_lag_columns(sample_clean_df, sample_config, tmp_path):
    """Verifies lag features are created for AQI."""
    output_dir = tmp_path / "data"
    result = run_features(sample_clean_df, sample_config, output_dir)
    assert "aqi_lag1" in result.columns
    assert "aqi_lag3" in result.columns


def test_run_features_adds_rolling_columns(sample_clean_df, sample_config, tmp_path):
    """Verifies rolling mean features are created for pollutants."""
    output_dir = tmp_path / "data"
    result = run_features(sample_clean_df, sample_config, output_dir)
    assert "pm10_roll3m" in result.columns
    assert "pm10_roll3std" in result.columns


# ── main (CLI) ────────────────────────────────────────────────────────────────


def test_main_stage_all_produces_both_outputs(sample_raw_data, tmp_path):
    """Smoke test: verifies main(--stage all) creates both output CSVs."""
    from src.pipeline import main

    csv_path = tmp_path / "AQI Bangladesh.csv"
    sample_raw_data.to_csv(csv_path, index=False)
    output_dir = tmp_path / "data"

    main(
        [
            "--stage",
            "all",
            "--input",
            str(csv_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "df_clean.csv").exists()
    assert (output_dir / "df_feat.csv").exists()


def test_main_stage_clean_only_skips_features(sample_raw_data, tmp_path):
    """Verifies --stage clean only writes df_clean.csv, not df_feat.csv."""
    from src.pipeline import main

    csv_path = tmp_path / "AQI Bangladesh.csv"
    sample_raw_data.to_csv(csv_path, index=False)
    output_dir = tmp_path / "data"

    main(
        [
            "--stage",
            "clean",
            "--input",
            str(csv_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "df_clean.csv").exists()
    assert not (output_dir / "df_feat.csv").exists()
