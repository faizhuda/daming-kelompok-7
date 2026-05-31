import numpy as np
import pandas as pd
from src.features import (
    add_cyclical_features,
    create_lag_features,
    create_rolling_features,
    create_pollutant_interactions,
)


def test_add_cyclical_features(sample_raw_data):
    """Verifies that sin/cos cyclical features are properly generated within boundary scales [-1, 1]."""
    df_feat = add_cyclical_features(sample_raw_data)

    cyclical_cols = [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ]
    for col in cyclical_cols:
        assert col in df_feat.columns
        # Sine and cosine bounds
        assert df_feat[col].min() >= -1.01
        assert df_feat[col].max() <= 1.01


def test_create_lag_features(sample_raw_data, sample_config):
    """Ensures lag shifts align correctly with previous timestamps and return NaNs for lookback boundaries."""
    df_sorted = sample_raw_data.sort_values(["city_id", "datetime"]).reset_index(
        drop=True
    )
    lag_cols = ["pm2_5"]

    df_lag = create_lag_features(df_sorted, lag_cols, config=sample_config)

    # Config defines lags as [1, 3]
    assert "pm2_5_lag1" in df_lag.columns
    assert "pm2_5_lag3" in df_lag.columns

    # Row 1 lag 1 should match Row 0 pm2_5
    assert df_lag["pm2_5_lag1"].iloc[1] == df_sorted["pm2_5"].iloc[0]
    # Row 0 lag 1 should be NaN (no lookback available)
    assert pd.isna(df_lag["pm2_5_lag1"].iloc[0])


def test_create_rolling_features(sample_raw_data, sample_config):
    """Verifies rolling averages use shift(1) so only past values are included (no lookahead)."""
    df_sorted = sample_raw_data.sort_values(["city_id", "datetime"]).reset_index(
        drop=True
    )
    cols = ["pm10"]

    df_roll = create_rolling_features(df_sorted, cols, config=sample_config)

    # Config defines rolling window as [3]
    assert "pm10_roll3m" in df_roll.columns
    assert "pm10_roll3std" in df_roll.columns

    # At index 2: rolling window sees shift(1)[0..2] = [NaN, pm10[0], pm10[1]]
    # With min_periods=1, mean = mean(pm10[0], pm10[1]) — only past values, no current
    expected_mean = df_sorted["pm10"].iloc[0:2].mean()
    assert np.isclose(df_roll["pm10_roll3m"].iloc[2], expected_mean)


def test_create_pollutant_interactions(sample_raw_data, sample_config):
    """Tests if chemical/physical interaction features are computed accurately based on pollutant interactions."""
    # Interaction features are derived from lag1 columns to prevent target leakage.
    # We must create lag features first so the required *_lag1 columns exist.
    df_sorted = sample_raw_data.sort_values(["city_id", "datetime"]).reset_index(
        drop=True
    )
    lag_cols = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]
    df_lagged = create_lag_features(df_sorted, lag_cols, config=sample_config)
    df_feat = create_pollutant_interactions(df_lagged)

    assert "pm_ratio_lag1" in df_feat.columns
    assert "pm_total_lag1" in df_feat.columns
    assert "oxidant_load_lag1" in df_feat.columns
    assert "combustion_idx_lag1" in df_feat.columns

    # Row 1: pm2_5_lag1 == pm2_5 from row 0, pm10_lag1 == pm10 from row 0
    eps = 1e-8
    expected_ratio = df_sorted["pm2_5"].iloc[0] / (df_sorted["pm10"].iloc[0] + eps)
    assert np.isclose(df_feat["pm_ratio_lag1"].iloc[1], expected_ratio)
