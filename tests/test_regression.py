import pandas as pd
from src.models import fit_and_evaluate

def test_linear_regression_returns_required_metrics():
    df = pd.DataFrame({"x": range(100), "z": [v * 2 for v in range(100)]})
    df["target"] = df["x"] * 3 + 2
    result = fit_and_evaluate(df, "target", "Linear Regression", test_size=0.2, random_state=42)
    assert set(result["metrics"]) == {"MAE", "MSE", "RMSE", "R²"}
    assert result["metrics"]["R²"] > 0.99
