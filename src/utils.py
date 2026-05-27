import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config_loader import load_config


def validate_columns(
    df: pd.DataFrame, required_cols: List[str], func_name: str
) -> None:
    """Validates that required columns exist in DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_cols (List[str]): Column names that must be present.
        func_name (str): Caller function name, used in the error message.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{func_name}: kolom berikut tidak ditemukan di DataFrame: {sorted(missing)}"
        )


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Compute regression metrics and return them as a labelled dict.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Model predictions.
        model_name (str): Label used in the returned dict and for logging.

    Returns:
        Dict[str, Any]: Keys — 'Model', 'MAE', 'RMSE', 'R2', 'MAPE'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def plot_predictions(
    dates,
    y_true,
    y_pred,
    model_name: str,
    save_dir: str = "results/plots",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Plot predictions against true values and scatter plot.

    Args:
        dates: Array-like of datetime values for the x-axis.
        y_true: Array-like of ground-truth target values.
        y_pred: Array-like of model predictions.
        model_name (str): Label used in the plot title and filename.
        save_dir (str): Directory where the PNG is saved.
        config (Optional[Dict[str, Any]]): Config dict. Loads from YAML if None.

    Returns:
        str: File path of the saved plot image.
    """
    if config is None:
        config = load_config()
    plot_sample_size: int = config.get("visualization", {}).get("plot_sample_size", 500)

    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    dates = np.array(dates)
    n = min(plot_sample_size, len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    axes[0].plot(dates[:n], y_true[:n], label="Aktual", alpha=0.7)
    axes[0].plot(dates[:n], y_pred[:n], label="Prediksi", alpha=0.7)
    axes[0].set_title(f"{model_name} - Prediksi vs Aktual (Sample {n:,} Data)")
    axes[0].legend()

    axes[1].scatter(y_true, y_pred, alpha=0.3, s=5)
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[1].set_xlabel("Aktual")
    axes[1].set_ylabel("Prediksi")
    axes[1].set_title(f"{model_name} - Scatter Aktual vs Prediksi")

    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(save_dir, f"{fname}_prediction.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def log_experiment(
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    log_path: str = "results/experiment_log.csv",
) -> None:
    """Appends a training run record to a CSV experiment log.

    Args:
        model_name (str): Name of the model being logged.
        params (Dict[str, Any]): Hyperparameters used for this run.
        metrics (Dict[str, float]): Evaluation metrics dict (from evaluate_model).
        log_path (str, optional): Path to CSV log file. Defaults to 'results/experiment_log.csv'.
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "params": str(params),
        "mae": metrics.get("MAE"),
        "rmse": metrics.get("RMSE"),
        "r2": metrics.get("R2"),
        "mape": metrics.get("MAPE"),
    }

    file_exists = log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
