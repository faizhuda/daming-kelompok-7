import numpy as np
import pandas as pd
from src.cleaning import clean_data, winsorize_city, impute_missing_linear


def test_clean_data(sample_raw_data, sample_config):
    """Verifies that duplicate records, negative pollutant readings, missing target labels,
    and unnecessary columns are processed correctly without raising errors.
    """
    original_length = len(sample_raw_data)
    assert original_length == 49
    assert sample_raw_data["pm10"].iloc[5] == -10.0
    assert pd.isna(sample_raw_data["aqi"].iloc[15])

    cleaned_df = clean_data(sample_raw_data, config=sample_config)

    # 1 duplicate + 1 missing AQI dropped = 2 rows removed
    assert len(cleaned_df) == original_length - 2

    # 2. Negative values should be converted to NaN and then linear-interpolated
    # Hence, index 5 should now contain a valid positive number
    assert cleaned_df["pm10"].iloc[5] >= 0.0
    assert not cleaned_df["pm10"].isna().any()

    # 3. Carbon Dioxide should be dropped as configured
    assert "carbon_dioxide" not in cleaned_df.columns

    # 4. No NaNs should remain in the designated pollutant columns
    for col in sample_config["data"]["pollutant_cols"]:
        assert not cleaned_df[col].isna().any()


def test_winsorize_city(sample_raw_data, sample_config):
    """Verifies that winsorization clips extreme values per city without changing data size."""
    col = "pm10"
    # Artificially inject an extreme value
    sample_raw_data.loc[0, col] = 5000.0

    # Clean up duplicate first for cleaner testing
    df_unique = sample_raw_data.drop_duplicates(subset=["city_id", "datetime"])

    winsorized = winsorize_city(
        df_unique.copy(), col, limits=sample_config["cleaning"]["winsorize_limits"]
    )

    # Length shouldn't change
    assert len(winsorized) == len(df_unique)
    # The extreme value 5000 should be clipped to the 99th percentile limit
    assert winsorized[col].iloc[0] < 5000.0


def test_winsorize_city_does_not_modify_input(sample_raw_data, sample_config):
    """Verifies that winsorize_city returns a new DataFrame without mutating the input."""
    col = "pm10"
    sample_raw_data.loc[0, col] = 5000.0
    original_value = sample_raw_data[col].iloc[0]

    winsorize_city(
        sample_raw_data, col, limits=sample_config["cleaning"]["winsorize_limits"]
    )

    # Input should be unchanged
    assert sample_raw_data[col].iloc[0] == original_value


def test_impute_missing_linear(sample_raw_data, sample_config):
    """Verifies that linear interpolation fills NaN values without leaving gaps."""
    df = (
        sample_raw_data.sort_values(["city_id", "datetime"])
        .reset_index(drop=True)
        .copy()
    )
    # Inject a NaN in the middle of the series
    df.loc[10, "carbon_monoxide"] = np.nan

    result = impute_missing_linear(df, ["carbon_monoxide"])

    # No NaNs should remain
    assert not result["carbon_monoxide"].isna().any()
    # Interpolated value at row 10 should be positive (neighboring values are positive)
    assert result["carbon_monoxide"].iloc[10] > 0


def test_impute_missing_linear_does_not_modify_input(sample_raw_data):
    """Verifies that impute_missing_linear returns a new DataFrame without mutating the input."""
    df = sample_raw_data.copy()
    df.loc[10, "carbon_monoxide"] = np.nan

    impute_missing_linear(df, ["carbon_monoxide"])

    # Input should still have the NaN
    assert pd.isna(df.loc[10, "carbon_monoxide"])
