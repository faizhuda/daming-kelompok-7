import numpy as np
import pytest
from src.models import create_sequences


def test_create_sequences_output_shape():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    Xs, ys = create_sequences(X, y, look_back=10)
    assert Xs.shape == (90, 10, 5)
    assert ys.shape == (90,)


def test_create_sequences_target_alignment():
    X = np.arange(50).reshape(50, 1).astype(float)
    y = np.arange(50, dtype=float)
    look_back = 5
    Xs, ys = create_sequences(X, y, look_back)
    assert ys[0] == y[look_back]
    np.testing.assert_array_equal(Xs[0, :, 0], X[:look_back, 0])


def test_create_sequences_look_back_one():
    X = np.ones((10, 3))
    y = np.arange(10, dtype=float)
    Xs, ys = create_sequences(X, y, look_back=1)
    assert Xs.shape == (9, 1, 3)
    assert ys.shape == (9,)


def test_create_sequences_raises_when_look_back_too_large():
    """Verifies a clear ValueError is raised when look_back >= len(X)."""
    X = np.random.rand(5, 3)
    y = np.random.rand(5)
    with pytest.raises(ValueError, match="look_back"):
        create_sequences(X, y, look_back=5)


def test_create_sequences_raises_when_look_back_exceeds_length():
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    with pytest.raises(ValueError, match="look_back"):
        create_sequences(X, y, look_back=20)


def test_train_xgboost_returns_fitted_model(sample_config):
    pytest.importorskip("xgboost")
    from src.models import train_xgboost

    np.random.seed(42)
    X_train = np.random.rand(80, 10)
    y_train = np.random.rand(80)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20)

    model = train_xgboost(X_train, y_train, X_val, y_val, config=sample_config)
    preds = model.predict(X_val)
    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))
