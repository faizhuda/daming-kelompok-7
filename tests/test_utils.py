import csv
import os

import numpy as np
import pandas as pd
import pytest

from src.utils import evaluate_model, log_experiment, plot_predictions, validate_columns


def test_evaluate_model_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = evaluate_model(y_true, y_pred, "PerfectModel")
    assert result["Model"] == "PerfectModel"
    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)
    assert result["R2"] == pytest.approx(1.0)
    assert result["MAPE"] == pytest.approx(0.0)


def test_evaluate_model_known_constant_error():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 21.0, 31.0, 41.0])
    result = evaluate_model(y_true, y_pred, "ConstantOffModel")
    assert result["MAE"] == pytest.approx(1.0)
    assert result["RMSE"] == pytest.approx(1.0)
    assert result["R2"] < 1.0


def test_evaluate_model_returns_all_required_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    result = evaluate_model(y_true, y_pred, "AnyModel")
    assert set(result.keys()) == {"Model", "MAE", "RMSE", "R2", "MAPE"}


def test_evaluate_model_mape_avoids_division_by_zero():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.1, 1.1, 2.1])
    result = evaluate_model(y_true, y_pred, "ZeroTrueModel")
    assert np.isfinite(result["MAPE"])


def test_validate_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    validate_columns(df, ["a", "b"], "test_func")  # should not raise


def test_validate_columns_raises_when_column_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="kolom berikut tidak ditemukan"):
        validate_columns(df, ["a", "missing_col"], "test_func")


def test_validate_columns_error_message_contains_func_name():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="my_function"):
        validate_columns(df, ["x", "y"], "my_function")


def test_log_experiment_creates_csv_on_first_call(tmp_path):
    log_path = str(tmp_path / "test_log.csv")
    metrics = {"MAE": 1.5, "RMSE": 2.0, "R2": 0.85, "MAPE": 5.0}
    log_experiment("TestModel", {"lr": 0.01, "epochs": 10}, metrics, log_path=log_path)

    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["model"] == "TestModel"
    assert float(rows[0]["mae"]) == pytest.approx(1.5)
    assert float(rows[0]["rmse"]) == pytest.approx(2.0)
    assert float(rows[0]["r2"]) == pytest.approx(0.85)


def test_log_experiment_appends_on_subsequent_calls(tmp_path):
    log_path = str(tmp_path / "test_log.csv")
    metrics = {"MAE": 1.0, "RMSE": 1.5, "R2": 0.9, "MAPE": 3.0}
    log_experiment("Model1", {}, metrics, log_path=log_path)
    log_experiment("Model2", {}, metrics, log_path=log_path)

    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["model"] == "Model1"
    assert rows[1]["model"] == "Model2"


def test_plot_predictions_creates_png_file(tmp_path):
    """Smoke test: verifies plot_predictions saves a PNG without crashing."""
    dates = pd.date_range("2023-01-01", periods=20, freq="h")
    y_true = np.random.rand(20) * 100
    y_pred = np.random.rand(20) * 100

    path = plot_predictions(dates, y_true, y_pred, "TestModel", save_dir=str(tmp_path))

    assert os.path.exists(path)
    assert path.endswith(".png")


def test_plot_predictions_respects_sample_size(tmp_path):
    """Verifies plot_predictions uses plot_sample_size from config to limit plotted points."""
    dates = pd.date_range("2023-01-01", periods=50, freq="h")
    y_true = np.random.rand(50) * 100
    y_pred = np.random.rand(50) * 100
    config = {"visualization": {"plot_sample_size": 10}}

    path = plot_predictions(
        dates, y_true, y_pred, "SampleModel", save_dir=str(tmp_path), config=config
    )

    assert os.path.exists(path)
