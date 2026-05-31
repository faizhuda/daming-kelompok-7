import matplotlib

matplotlib.use("Agg")  # non-interactive backend — required for tests without a display

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Generates a small dummy DataFrame mimicking raw multi-city pollutant sensor data."""
    # Ensure reproducibility
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01 00:00:00", periods=48, freq="h")

    # City 1 data
    df1 = pd.DataFrame(
        {
            "city_id": [1] * 48,
            "city_name": ["Dhaka"] * 48,
            "datetime": dates,
            "pm10": np.random.uniform(50, 150, 48),
            "pm2_5": np.random.uniform(30, 90, 48),
            "carbon_monoxide": np.random.uniform(0.5, 2.5, 48),
            "nitrogen_dioxide": np.random.uniform(10, 40, 48),
            "sulphur_dioxide": np.random.uniform(5, 20, 48),
            "ozone": np.random.uniform(15, 50, 48),
            "carbon_dioxide": np.random.uniform(350, 450, 48),
            "aqi": np.random.uniform(60, 180, 48),
        }
    )

    # Add a duplicate row
    dup = df1.iloc[[0]].copy()
    df1 = pd.concat([df1, dup], ignore_index=True)

    # Introduce negative and missing values for cleaning verification
    df1.loc[5, "pm10"] = -10.0
    df1.loc[10, "pm2_5"] = np.nan
    df1.loc[15, "aqi"] = np.nan  # Missing target (should be dropped)

    return df1


@pytest.fixture
def sample_config() -> dict:
    """Fixture providing simple config dictionary for tests."""
    return {
        "data": {
            "pollutant_cols": [
                "pm10",
                "pm2_5",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ],
            "carbon_dioxide_col": "carbon_dioxide",
            "aqi_col": "aqi",
            "city_id_col": "city_id",
            "datetime_col": "datetime",
        },
        "cleaning": {"winsorize_limits": [0.01, 0.99]},
        "features": {"lags": [1, 3], "rolling_windows": [3]},
        "models": {
            "xgboost": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 5,
            },
        },
        "visualization": {
            "plot_sample_size": 100,
        },
    }
