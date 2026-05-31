import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)  # noqa: F401 (Any used by clean_data)

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.utils import validate_columns

logger = logging.getLogger(__name__)


def winsorize_city(
    df: pd.DataFrame,
    col: Union[str, List[str]],
    limits: List[float],
    city_id_col: str = "city_id",
) -> pd.DataFrame:
    """Winsorizes specific column(s) per city to handle outliers by clipping extreme quantiles.

    Args:
        df (pd.DataFrame): The input dataframe containing spatial data.
        col (Union[str, List[str]]): Column name or list of column names to winsorize.
        limits (List[float]): A list of two floats representing the lower and upper quantile bounds.
        city_id_col (str, optional): The column name identifying cities. Defaults to "city_id".

    Returns:
        pd.DataFrame: A new dataframe with the specified column(s) winsorized.
    """
    df = df.copy()
    cols = [col] if isinstance(col, str) else list(col)

    def winsorize_series(s: pd.Series) -> pd.Series:
        lower = s.quantile(limits[0])
        upper = s.quantile(limits[1])
        return s.clip(lower, upper)

    for c in cols:
        if c in df.columns:
            df[c] = df.groupby(city_id_col)[c].transform(winsorize_series)
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
        pd.DataFrame: A new dataframe with imputed values.
    """
    df = df.copy()
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
    n_input = len(df)
    df_clean = df.copy()

    # 1. Parse datetime
    df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col])
    df_clean = df_clean.sort_values([city_id_col, datetime_col])

    # 2. Drop duplicates
    df_clean = df_clean.drop_duplicates(subset=[city_id_col, datetime_col])

    # 3. Handle negative values (Physical Bounds) including the target aqi column
    cols_to_check = [c for c in pollutant_cols if c in df_clean.columns]
    if aqi_col in df_clean.columns:
        cols_to_check.append(aqi_col)
    for col in cols_to_check:
        df_clean.loc[df_clean[col] < 0, col] = np.nan

    # 4. Winsorizing per city — cap both pollutants and AQI target to suppress sensor spikes.
    #    Including aqi_col prevents corrupt labels (e.g. AQI=5000) from entering training.
    cols_to_winsorize = [c for c in pollutant_cols if c in df_clean.columns]
    if aqi_col in df_clean.columns:
        cols_to_winsorize.append(aqi_col)
    df_clean = winsorize_city(
        df_clean, cols_to_winsorize, winsorize_limits, city_id_col
    )

    # 5. Drop carbon dioxide (>74% missing)
    if carbon_dioxide_col in df_clean.columns:
        df_clean = df_clean.drop(columns=[carbon_dioxide_col])

    # 6. Drop missing AQI
    if aqi_col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[aqi_col])

    # 7. Interpolation (using the impute_missing_linear function)
    cols_to_impute = [c for c in pollutant_cols if c in df_clean.columns]
    df_clean = impute_missing_linear(df_clean, cols_to_impute, city_id_col)

    logger.info(
        "clean_data: %d → %d rows (dropped %d)",
        n_input,
        len(df_clean),
        n_input - len(df_clean),
    )
    return df_clean
