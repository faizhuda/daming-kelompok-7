import logging
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np

from src.config_loader import load_config

logger = logging.getLogger(__name__)


def create_sequences(
    X: np.ndarray, y: np.ndarray, look_back: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D dataset into sequential 3-D inputs for sequence models.

    Args:
        X (np.ndarray): Scaled multi-feature 2D input array (n_samples, n_features).
        y (np.ndarray): Target labels 1D array (n_samples,).
        look_back (int): Window size — number of past time steps per sample.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Xs, ys) where Xs has shape
            (n_samples - look_back, look_back, n_features).

    Raises:
        ValueError: If look_back is greater than or equal to len(X).
    """
    if look_back >= len(X):
        raise ValueError(
            f"look_back ({look_back}) must be less than len(X) ({len(X)}). "
            "Provide more data or reduce the look_back window."
        )
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i : i + look_back])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMRegressor:
    """Train a LightGBM Regressor with early stopping on a validation set.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation target vector.
        params (Optional[Dict[str, Any]]): LightGBM hyperparameters. If None,
            reads defaults from config. Defaults to None.
        config (Optional[Dict[str, Any]]): Config dict. Loads from YAML if None.

    Returns:
        lgb.LGBMRegressor: The trained LightGBM regressor model.
    """
    if params is None:
        if config is None:
            config = load_config()
        lgb_cfg = config["models"]["lightgbm"]
        early_stopping_rounds = lgb_cfg.get("early_stopping_rounds", 20)
        params = {
            "n_estimators": lgb_cfg["n_estimators"],
            "num_leaves": lgb_cfg.get("num_leaves", 63),
            "learning_rate": lgb_cfg["learning_rate"],
            "subsample": lgb_cfg.get("subsample", 0.8),
            "colsample_bytree": lgb_cfg.get("colsample_bytree", 0.8),
            "min_child_samples": lgb_cfg.get("min_child_samples", 20),
            "n_jobs": lgb_cfg.get("n_jobs", -1),
            "random_state": 42,
            "verbose": -1,
        }
    else:
        early_stopping_rounds = params.pop("early_stopping_rounds", 20)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = getattr(model, "best_iteration_", None)
    logger.info("LightGBM trained — best_iteration=%s", best_iter)
    return model
