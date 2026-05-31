import pytest
from src.config_loader import load_config


def test_load_config_returns_dict():
    """Verifies config loads successfully and returns a dict."""
    config = load_config()
    assert isinstance(config, dict)


def test_load_config_has_required_sections():
    """Verifies all expected top-level sections exist."""
    config = load_config()
    for section in ("data", "cleaning", "features", "models", "visualization"):
        assert section in config, f"Missing section: {section}"


def test_load_config_data_section_has_required_keys():
    config = load_config()
    data = config["data"]
    for key in (
        "pollutant_cols",
        "aqi_col",
        "city_id_col",
        "datetime_col",
        "carbon_dioxide_col",
    ):
        assert key in data, f"Missing data key: {key}"


def test_load_config_pollutant_cols_is_nonempty_list():
    config = load_config()
    pollutants = config["data"]["pollutant_cols"]
    assert isinstance(pollutants, list)
    assert len(pollutants) > 0


def test_load_config_xgboost_section_has_required_keys():
    config = load_config()
    xgb = config["models"]["xgboost"]
    for key in ("n_estimators", "max_depth", "learning_rate", "early_stopping_rounds"):
        assert key in xgb, f"Missing xgboost key: {key}"


def test_load_config_raises_on_nonexistent_path():
    """Verifies FileNotFoundError is raised for a bad config path."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_winsorize_limits_are_valid():
    config = load_config()
    limits = config["cleaning"]["winsorize_limits"]
    assert len(limits) == 2
    assert 0.0 <= limits[0] < limits[1] <= 1.0
