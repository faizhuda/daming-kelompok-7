from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.utils import validate_columns


def add_cyclical_features(
    df: pd.DataFrame, datetime_col: str = "datetime"
) -> pd.DataFrame:
    """Adds sine and cosine cyclical features for hourly, monthly, and weekly temporal components.

    Args:
        df (pd.DataFrame): The input dataframe.
        datetime_col (str, optional): The name of the datetime column. Defaults to "datetime".

    Returns:
        pd.DataFrame: The dataframe with added cyclical features.
    """
    df_feat = df.copy()
    hour = df_feat[datetime_col].dt.hour
    month = df_feat[datetime_col].dt.month
    dayofweek = df_feat[datetime_col].dt.dayofweek

    df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    df_feat["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    df_feat["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df_feat["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    return df_feat


def create_lag_features(
    df: pd.DataFrame, lag_cols: List[str], config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Creates lagged features for specified columns grouped by city to prevent lookahead bias.

    Args:
        df (pd.DataFrame): The input dataframe.
        lag_cols (List[str]): Columns to create lag features for.
        config (Optional[Dict[str, Any]], optional): Config dictionary.
            If None, loads config from configs/config.yaml. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with added lag features.
    """
    if config is None:
        config = load_config()

    lags = config["features"]["lags"]
    city_id_col = config["data"]["city_id_col"]

    validate_columns(df, [city_id_col], "create_lag_features")
    df_feat = df.copy()
    for col in lag_cols:
        if col in df_feat.columns:
            for lag in lags:
                df_feat[f"{col}_lag{lag}"] = df_feat.groupby(city_id_col)[col].shift(
                    lag
                )
    return df_feat


def create_rolling_features(
    df: pd.DataFrame, cols: List[str], config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Creates rolling mean and standard deviation features for specified columns per city.

    Args:
        df (pd.DataFrame): The input dataframe.
        cols (List[str]): Columns to calculate rolling statistics for.
        config (Optional[Dict[str, Any]], optional): Config dictionary.
            If None, loads config from configs/config.yaml. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with rolling mean and std features.
    """
    if config is None:
        config = load_config()

    windows = config["features"]["rolling_windows"]
    city_id_col = config["data"]["city_id_col"]

    validate_columns(df, [city_id_col], "create_rolling_features")
    df_feat = df.copy()
    for col in cols:
        if col in df_feat.columns:
            for w in windows:
                df_feat[f"{col}_roll{w}m"] = df_feat.groupby(city_id_col)[
                    col
                ].transform(lambda x: x.rolling(w, min_periods=1).mean())
                df_feat[f"{col}_roll{w}std"] = df_feat.groupby(city_id_col)[
                    col
                ].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    return df_feat


def create_pollutant_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Creates physical and chemical interaction features between air pollutants based on previous hour's data (lag1) to prevent target leakage.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with pollutant interaction ratio features.
    """
    df_feat = df.copy()
    eps = 1e-8

    if "pm2_5_lag1" in df_feat.columns and "pm10_lag1" in df_feat.columns:
        df_feat["pm_ratio_lag1"] = df_feat["pm2_5_lag1"] / (df_feat["pm10_lag1"] + eps)
        df_feat["pm_total_lag1"] = df_feat["pm10_lag1"] + df_feat["pm2_5_lag1"]

    if "nitrogen_dioxide_lag1" in df_feat.columns and "ozone_lag1" in df_feat.columns:
        df_feat["oxidant_load_lag1"] = df_feat["nitrogen_dioxide_lag1"] + df_feat["ozone_lag1"]

    if "carbon_monoxide_lag1" in df_feat.columns and "nitrogen_dioxide_lag1" in df_feat.columns:
        df_feat["combustion_idx_lag1"] = df_feat["carbon_monoxide_lag1"] / (
            df_feat["nitrogen_dioxide_lag1"] + eps
        )

    return df_feat
