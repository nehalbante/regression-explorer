import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PURPLE = "#DB2777"
BLUE = "#F472B6"
PINK = "#E11D48"
GRID = "#F2D7E2"
PLOT_TEXT = "#493342"
PLOT_BG = "rgba(255,247,250,0.72)"

def set_plot_theme(theme):
    """Set readable Plotly colors to match the selected app theme."""
    global GRID, PLOT_TEXT, PLOT_BG
    if theme == "Dark":
        GRID = "#543447"
        PLOT_TEXT = "#FCE7F3"
        PLOT_BG = "rgba(35,19,32,0.96)"
    else:
        GRID = "#F2D7E2"
        PLOT_TEXT = "#493342"
        PLOT_BG = "rgba(255,247,250,0.72)"

def _layout(fig, height=430):
    fig.update_layout(
        height=height, template="plotly_white", margin=dict(l=18,r=18,t=58,b=20),
        font=dict(family="DM Sans, sans-serif", color=PLOT_TEXT),
        title_font=dict(size=17, color="#252B45"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(color=PLOT_TEXT), title_font=dict(color=PLOT_TEXT))
    fig.update_xaxes(tickfont=dict(color=PLOT_TEXT), title_font=dict(color=PLOT_TEXT))
    fig.update_layout(legend=dict(font=dict(color=PLOT_TEXT)),
                      title_font=dict(size=17, color=PLOT_TEXT))
    return fig

def metric_cards(metrics):
    import streamlit as st
    cols = st.columns(4)
    descriptions = {"MAE":"Mean absolute error ↓", "MSE":"Mean squared error ↓",
                    "RMSE":"Root mean squared error ↓", "R²":"Explained variance ↑"}
    for col, key in zip(cols, ["MAE","MSE","RMSE","R²"]):
        col.metric(key, f"{metrics[key]:,.4f}", help=descriptions[key])

def actual_vs_predicted(result):
    p = result["predictions"]
    lo = float(min(p.actual.min(), p.predicted.min()))
    hi = float(max(p.actual.max(), p.predicted.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.actual, y=p.predicted, mode="markers",
        marker=dict(size=7, color=BLUE, opacity=.68, line=dict(width=.5,color="#FFFFFF")),
        name="Test predictions", hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
        line=dict(color=PINK, width=2, dash="dash"), name="Perfect prediction"))
    fig.update_layout(title="Actual vs. predicted", xaxis_title="Actual value", yaxis_title="Predicted value")
    return _layout(fig)

def residual_plot(result):
    p = result["predictions"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.predicted, y=p.residual, mode="markers",
        marker=dict(size=7,color=PURPLE,opacity=.65), name="Residuals",
        hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color=PINK, line_width=2)
    fig.update_layout(title="Residuals vs. predicted", xaxis_title="Predicted value", yaxis_title="Residual (actual − predicted)")
    return _layout(fig)

def residual_distribution(result):
    residuals = result["predictions"]["residual"]
    fig = px.histogram(x=residuals, nbins=35, opacity=.82, color_discrete_sequence=[BLUE])
    fig.update_layout(title="Residual distribution", xaxis_title="Residual", yaxis_title="Count", showlegend=False)
    fig.add_vline(x=0, line_dash="dash", line_color=PINK, line_width=2)
    return _layout(fig)

def feature_relationship(df, feature, target):
    plotdf = df[[feature,target]].dropna()
    fig = px.scatter(plotdf, x=feature, y=target, opacity=.62, trendline="ols",
                     color_discrete_sequence=[BLUE], title=f"{feature} vs. {target}")
    fig.update_traces(marker=dict(size=7))
    return _layout(fig)

def model_comparison_chart(df, metric):
    ordered = df.sort_values(metric, ascending=(metric != "R²"))
    fig = px.bar(ordered, x="Model", y=metric, color="Model", text_auto=".4f",
                 title=f"Model comparison · {metric}")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=metric)
    return _layout(fig, 440)

def correlation_heatmap(df):
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale=[[0,"#BE123C"],[0.5,"#FFF1F5"],[1,"#DB2777"]],
        text=np.round(corr.values,2), texttemplate="%{text:.2f}", textfont=dict(color=PLOT_TEXT),
        hovertemplate="%{x} × %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(title="Numeric feature correlation matrix", xaxis_tickangle=-40)
    return _layout(fig, max(420, min(700, 45*len(corr.columns))))

def coefficient_chart(result):
    try:
        prep = result["model"].named_steps["preprocessor"]
        model = result["model"].named_steps["model"]
        names = prep.get_feature_names_out()
        estimator = model
        if hasattr(model, "named_steps"):
            estimator = model.steps[-1][1]
            names = model[:-1].get_feature_names_out(names)
        if not hasattr(estimator, "coef_"):
            import streamlit as st
            st.info("This model does not expose linear coefficients. Use feature relationships and the comparison tab for interpretation.")
            return _layout(go.Figure().update_layout(title="Coefficients unavailable for this model"))
        coef = np.asarray(estimator.coef_).ravel()
        frame = pd.DataFrame({"Feature":names[:len(coef)],"Coefficient":coef})
        frame = frame.reindex(frame.Coefficient.abs().sort_values(ascending=False).index).head(20)
        fig = px.bar(frame.sort_values("Coefficient"), x="Coefficient", y="Feature", orientation="h",
                     color="Coefficient", color_continuous_scale=["#E11D48","#FFF1F5","#DB2777"],
                     title="Largest model coefficients (top 20 by absolute magnitude)")
        return _layout(fig, max(430, 25*len(frame)+100))
    except Exception as exc:
        fig = go.Figure()
        fig.add_annotation(text=f"Coefficient view unavailable: {exc}", showarrow=False)
        return _layout(fig)
