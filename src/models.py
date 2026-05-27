from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np
import xgboost as xgb

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
    """
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i : i + look_back])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train an XGBoost Regressor with early stopping on a validation set.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation target vector.
        params (Optional[Dict[str, Any]]): XGBoost hyperparameters. If None,
            reads defaults from config. Defaults to None.
        config (Optional[Dict[str, Any]]): Config dict. Loads from YAML if None.

    Returns:
        Any: The trained XGBRegressor model.
    """
    if params is None:
        if config is None:
            config = load_config()
        xgb_cfg = config["models"]["xgboost"]
        params = {
            "n_estimators": xgb_cfg["n_estimators"],
            "max_depth": xgb_cfg["max_depth"],
            "learning_rate": xgb_cfg["learning_rate"],
            "subsample": xgb_cfg.get("subsample", 0.8),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
            "random_state": 42,
            "n_jobs": -1,
        }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = getattr(model, "best_iteration", None)
    logger.info("XGBoost trained — best_iteration=%s", best_iter)
    return model
