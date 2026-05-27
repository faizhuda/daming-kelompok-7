from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.utils import validate_columns


def winsorize_city(
    df: pd.DataFrame, col: str, limits: List[float], city_id_col: str = "city_id"
) -> pd.DataFrame:
    """Winsorizes specific column per city to handle outliers by clipping extreme quantiles.

    Args:
        df (pd.DataFrame): The input dataframe containing spatial data.
        col (str): The column name to winsorize.
        limits (List[float]): A list of two floats representing the lower and upper quantile bounds.
        city_id_col (str, optional): The column name identifying cities. Defaults to "city_id".

    Returns:
        pd.DataFrame: The dataframe with the specified column winsorized.
    """

    def winsorize_series(s: pd.Series) -> pd.Series:
        lower = s.quantile(limits[0])
        upper = s.quantile(limits[1])
        return s.clip(lower, upper)

    df[col] = df.groupby(city_id_col)[col].transform(winsorize_series)
    return df


def impute_missing_linear(
    df: pd.DataFrame, cols: List[str], city_id_col: str = "city_id"
) -> pd.DataFrame:
    """Linearly interpolates missing values per city to maintain local trends.

    Args:
        df (pd.DataFrame): The input dataframe.
        cols (List[str]): Columns to perform linear interpolation on.
        city_id_col (str, optional): The column name identifying cities. Defaults to "city_id".

    Returns:
        pd.DataFrame: The dataframe with imputed values.
    """
    for col in cols:
        df[col] = df.groupby(city_id_col)[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
    return df


def clean_data(
    df: pd.DataFrame, config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Main pipeline to clean raw AQI data using configuration boundaries.

    Args:
        df (pd.DataFrame): The raw input dataframe.
        config (Optional[Dict[str, Any]], optional): Config dictionary containing parameters.
            If None, loads config from configs/config.yaml. Defaults to None.

    Returns:
        pd.DataFrame: The cleaned and imputed dataframe.
    """
    if config is None:
        config = load_config()

    # Extract configs
    data_cfg = config["data"]
    clean_cfg = config["cleaning"]

    pollutant_cols = data_cfg["pollutant_cols"]
    carbon_dioxide_col = data_cfg["carbon_dioxide_col"]
    aqi_col = data_cfg["aqi_col"]
    city_id_col = data_cfg["city_id_col"]
    datetime_col = data_cfg["datetime_col"]
    winsorize_limits = clean_cfg["winsorize_limits"]

    validate_columns(df, [datetime_col, city_id_col], "clean_data")
    df_clean = df.copy()

    # 1. Parse datetime
    df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col])
    df_clean = df_clean.sort_values([city_id_col, datetime_col])

    # 2. Drop duplicates
    df_clean = df_clean.drop_duplicates(subset=[city_id_col, datetime_col])

    # 3. Handle negative values (Physical Bounds)
    for col in pollutant_cols:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] < 0, col] = np.nan

    # 4. Winsorizing per city
    for col in pollutant_cols:
        if col in df_clean.columns:
            df_clean = winsorize_city(df_clean, col, winsorize_limits, city_id_col)

    # 5. Drop carbon dioxide (>74% missing)
    if carbon_dioxide_col in df_clean.columns:
        df_clean = df_clean.drop(columns=[carbon_dioxide_col])

    # 6. Drop missing AQI
    if aqi_col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[aqi_col])

    # 7. Interpolation
    df_clean = impute_missing_linear(df_clean, pollutant_cols, city_id_col)

    return df_clean
