import numpy as np
import pytest


def test_train_lightgbm_returns_fitted_model(sample_config):
    pytest.importorskip("lightgbm")
    from src.models import train_lightgbm

    np.random.seed(42)
    X_train = np.random.rand(80, 10)
    y_train = np.random.rand(80)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20)

    model = train_lightgbm(X_train, y_train, X_val, y_val, config=sample_config)
    preds = model.predict(X_val)
    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))
