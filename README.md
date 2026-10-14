# Regression Explorer

An interactive regression-analysis application for comparing regression techniques across multiple datasets. Built with Python, Streamlit, scikit-learn, Pandas, and Plotly.

## Features
- Five datasets: California Housing, Diabetes, Medical Insurance, Advertising, and Student Performance.
- Eight regression methods: Linear, Ridge, Lasso, Elastic Net, Polynomial, Decision Tree, Random Forest, and K-Nearest Neighbors.
- Train/test evaluation using MAE, MSE, RMSE, and R².
- Actual-vs-predicted, residual, residual-distribution, feature relationship, correlation, coefficient, and model-comparison plots.
- Interactive controls, CSV exports, and automated observations.

## Run locally in VS Code

Recommended: Python 3.10–3.12.

### Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed in the terminal (normally `http://localhost:8501`).

## Methodology
- Each dataset gets its own train/test split.
- Numeric features are median-imputed and standardized; categorical features are imputed and one-hot encoded.
- Preprocessing is fitted only on the training partition using an sklearn Pipeline, preventing preprocessing leakage.
- All metrics are calculated on the held-out test partition.
- For model comparison, the same split is used across candidate models for the selected dataset.

## Dataset notes
- California Housing: target `MedHouseVal`, expressed in units of $100,000.
- Diabetes: target `target`, a quantitative disease-progression measure.
- Medical Insurance: target `charges`.
- Advertising: target `Sales`.
- Student Performance (mathematics): target `G3`. `G1` and `G2` are included as predictors; this is useful for prediction but should be mentioned in the report because they are earlier grades closely related to the final grade.

The first selection of some datasets requires internet access. Respect the original dataset terms and cite the source in your submitted report.

## Important interpretation notes
- Do not compare raw MAE/RMSE values across different datasets as though they were on the same scale.
- A high R² is not by itself proof of a good model. Inspect residuals and all metrics.
- Polynomial expansion can become computationally expensive on high-dimensional data; the app caps the degree at 4.
- This is an educational tool, not a production decision system.

## Vercel / hosting
This first version uses Streamlit because it provides a strong Python-first interactive experience. Streamlit expects a persistent web server, so use a Streamlit-compatible host for the easiest deployment. If Vercel becomes a firm requirement, retain `src/` as the model layer and replace the Streamlit UI with a frontend plus a compatible Python API deployment.
