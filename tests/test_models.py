"""
Unit tests for supervised learning models: Linear Regression, Logistic Regression,
Decision Trees, and Ensemble Classifiers.
"""

import pytest
import numpy as np

from app.models.linear_regression import train_and_evaluate_linear, predict_linear_severity
from app.models.logistic_regression import train_and_evaluate_logistic, predict_logistic_severity
from app.models.decision_tree import train_and_evaluate_tree, predict_tree_severity
from app.services.model_manager import model_manager


def test_linear_regression_training():
    res = train_and_evaluate_linear(reg_type="none", save_model=False)
    assert res["reg_type"] == "none"
    assert "metrics" in res
    assert "test_r2" in res["metrics"]
    assert "rmse" in res["metrics"]
    assert len(res["coefficients"]) > 0


def test_linear_regression_ridge_lasso():
    res_ridge = train_and_evaluate_linear(reg_type="ridge", save_model=False)
    res_lasso = train_and_evaluate_linear(reg_type="lasso", save_model=False)
    assert res_ridge["reg_type"] == "ridge"
    assert res_lasso["reg_type"] == "lasso"


def test_logistic_regression_training():
    res = train_and_evaluate_logistic(reg_type="none", save_model=False)
    assert res["reg_type"] == "none"
    assert "metrics" in res
    assert "test_accuracy" in res["metrics"]
    assert "precision" in res["metrics"]
    assert "recall" in res["metrics"]
    assert "coefficients" in res


def test_decision_tree_and_rf_training():
    res_dt = train_and_evaluate_tree(algo_key="dt", save_model=False)
    assert res_dt["algo_key"] == "dt"
    assert "metrics" in res_dt
    assert "feature_importances" in res_dt
    assert len(res_dt["feature_importances"]) > 0

    res_rf = train_and_evaluate_tree(algo_key="rf", save_model=False)
    assert res_rf["algo_key"] == "rf"
    assert "metrics" in res_rf


def test_live_predictions():
    sample_input = {
        "Temperature(F)": 72.0,
        "Humidity(%)": 65.0,
        "Pressure(in)": 29.92,
        "Visibility(mi)": 10.0,
        "Wind_Speed(mph)": 8.0,
        "Crossing": 0,
        "Junction": 1,
        "Traffic_Signal": 0,
        "Sunrise_Sunset_Day": 1
    }
    res_linear = train_and_evaluate_linear(reg_type="none", student_input=sample_input, save_model=False)
    assert res_linear["prediction"] is not None
    assert res_linear["prediction"] >= 0.0

    res_logistic = train_and_evaluate_logistic(reg_type="none", student_input=sample_input, save_model=False)
    assert res_logistic["prediction"] is not None
    assert "class" in res_logistic["prediction"]
    assert "probability" in res_logistic["prediction"]
