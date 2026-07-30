import numpy as np

from scripts.compute_metrics import (
    cri_regression_metrics,
    high_risk_metrics,
)


def test_cri_metrics():
    true = np.asarray([0.0, 1.0])
    pred = np.asarray([0.2, 0.8])
    metrics = cri_regression_metrics(true, pred)
    assert abs(metrics["MAE"] - 0.2) < 1.0e-12
    assert abs(metrics["MSE"] - 0.04) < 1.0e-12


def test_confusion_counts():
    true = np.asarray([0.1, 0.9, 0.95, 0.2])
    pred = np.asarray([0.2, 0.8, 0.92, 0.91])
    metrics = high_risk_metrics(true, pred, 0.87)
    assert metrics["TP"] == 1
    assert metrics["FN"] == 1
    assert metrics["FP"] == 1
    assert metrics["TN"] == 1
