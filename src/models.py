import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_OPTIONS = {
    "Linear Regression": "Ordinary least squares baseline.",
    "Ridge Regression": "L2 regularization to shrink coefficients.",
    "Lasso Regression": "L1 regularization; can shrink some coefficients to zero.",
    "Elastic Net": "Combined L1 and L2 regularization.",
    "Polynomial Regression": "Linear regression over polynomial feature expansions.",
    "Decision Tree Regression": "Nonlinear, rule-based regression model.",
    "Random Forest Regression": "Ensemble of decision trees.",
    "K-Nearest Neighbors": "Predict from nearby training observations.",
}

def _make_model(name, params):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge Regression":
        return Ridge(alpha=float(params.get("alpha", 1.0)))
    if name == "Lasso Regression":
        return Lasso(alpha=float(params.get("alpha", 0.1)), max_iter=20000)
    if name == "Elastic Net":
        return ElasticNet(alpha=float(params.get("alpha", 0.1)), l1_ratio=float(params.get("l1_ratio", 0.5)), max_iter=20000)
    if name == "Polynomial Regression":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=int(params.get("degree", 2)), include_bias=False)),
            ("regressor", LinearRegression())
        ])
    if name == "Decision Tree Regression":
        return DecisionTreeRegressor(max_depth=params.get("max_depth", 6), random_state=42)
    if name == "Random Forest Regression":
        depth = params.get("max_depth", 0)
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=None if not depth else int(depth),
            random_state=42, n_jobs=-1
        )
    if name == "K-Nearest Neighbors":
        return KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 5)))
    raise ValueError(f"Unknown model: {name}")

def fit_and_evaluate(df, target, model_name, params=None, test_size=0.2, random_state=42):
    params = params or {}
    clean = df.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=[target])
    X = clean.drop(columns=[target])
    y = pd.to_numeric(clean[target], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", _make_model(model_name, params))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R²": float(r2_score(y_test, pred)),
    }
    predictions = pd.DataFrame({
        "actual": np.asarray(y_test),
        "predicted": np.asarray(pred),
        "residual": np.asarray(y_test) - np.asarray(pred),
    }, index=y_test.index)
    return {
        "model": pipeline, "metrics": metrics, "predictions": predictions,
        "X_test": X_test, "y_test": y_test, "y_pred": pred,
        "feature_names": list(X.columns), "target": target, "model_name": model_name
    }
