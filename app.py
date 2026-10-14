import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.datasets import load_dataset, DATASET_OPTIONS
from src.models import MODEL_OPTIONS, fit_and_evaluate
from src.visualizations import (
    set_plot_theme, metric_cards, actual_vs_predicted, residual_plot, residual_distribution,
    feature_relationship, model_comparison_chart, correlation_heatmap,
    coefficient_chart
)

st.set_page_config(
    page_title="Regression Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Theme selector is intentionally available before the page styles are applied.
theme = st.sidebar.radio(
    "🎨 Appearance",
    ["Dark" , "Light"],
    horizontal=True,
    index=0,
    help="Switch between light and dark themes. Your selected theme applies to the dashboard."
)

if theme == "Light":
    page_bg = "#fff7fa"
    panel_bg = "rgba(255,255,255,.92)"
    text_color = "#D71199"
    muted_color = "#7c6874"
    plot_bg = "rgba(255,250,252,.95)"
    grid_color = "#f2dfe7"
    sidebar_bg = "#fff0f5"
    accent = "#e84a75"
    accent2 = "#f4729b"
    accent3 = "#b83262"
    metric_bg = "#ffffff"
else:
    page_bg = "#170d16"
    panel_bg = "rgba(39,22,36,.94)"
    text_color = "#fff0f6"
    muted_color = "#d1b6c5"
    plot_bg = "rgba(32,19,31,.96)"
    grid_color = "#503247"
    sidebar_bg = "#21121f"
    accent = "#ff5c8a"
    accent2 = "#f472b6"
    accent3 = "#c02668"
    metric_bg = "#281725"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
:root {{
  --re-bg: {page_bg};
  --re-panel: {panel_bg};
  --re-text: {text_color};
  --re-muted: {muted_color};
  --re-grid: {grid_color};
  --re-accent: {accent};
  --re-accent2: {accent2};
  --re-accent3: {accent3};
}}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
  font-family: 'DM Sans', sans-serif;
}}
.stApp {{
  background:
    radial-gradient(circle at 8% 3%, {accent}22, transparent 28%),
    radial-gradient(circle at 92% 12%, {accent2}24, transparent 25%),
    {page_bg};
  color: {text_color};
}}
[data-testid="stSidebar"] {{
  background: {sidebar_bg};
  border-right: 1px solid {grid_color};
}}
[data-testid="stSidebar"] * {{
  color: {text_color};
}}
/* Keep labels and input values readable in both themes. */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
  color: {text_color} !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
  background-color: {metric_bg} !important;
  border-color: {grid_color} !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stMultiSelect"] [data-baseweb="select"] *,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
  color: {text_color} !important;
  -webkit-text-fill-color: {text_color} !important;
}}
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="select"] [role="listbox"] {{
  background-color: {metric_bg} !important;
  color: {text_color} !important;
}}
[data-baseweb="menu"] *,
[data-baseweb="popover"] * {{
  color: {text_color} !important;
}}
[data-testid="stHeader"] {{
  background: transparent !important;
}}
[data-testid="stToolbar"] {{
  color: {text_color} !important;
}}
.block-container {{ padding-top: 1.5rem; max-width: 1600px; }}
.hero {{
  padding: 1.8rem 2rem; border-radius: 24px; color: #fff !important;
  background: linear-gradient(115deg, {accent} 0%, #db2777 48%, {accent3} 100%);
  box-shadow: 0 16px 38px {accent}30; margin-bottom: 1.2rem;
  position: relative; isolation: isolate; overflow: hidden;
  animation: heroEnter .75s cubic-bezier(.2,.75,.25,1) both;
  
.cube-scene {{  position: absolute; inset: 0; transform-style: preserve-3d; animation: cubeFloat 7.5s ease-in-out infinite; }}
.cube {{ position: absolute; width: 106px; height: 106px; left: 50%; top: 50%; transform: translate(-50%, -50%) rotateX(-18deg) rotateY(32deg); transform-style: preserve-3d; animation: cubeSpin 15s linear infinite; }}
.cube-face {{ position: absolute; inset: 0; border: 1px solid rgba(255,255,255,.74); background: linear-gradient(135deg, rgba(255,255,255,.24), rgba(255,255,255,.035)); box-shadow: inset 0 0 28px rgba(255,255,255,.10); backface-visibility: visible; }}
.cube-front {{ transform: translateZ(53px); }} .cube-back {{ transform: rotateY(180deg) translateZ(53px); }}
.cube-right {{ transform: rotateY(90deg) translateZ(53px); }} .cube-left {{ transform: rotateY(-90deg) translateZ(53px); }}
.cube-top {{ transform: rotateX(90deg) translateZ(53px); }} .cube-bottom {{ transform: rotateX(-90deg) translateZ(53px); }}
.cube-core {{ position: absolute; inset: 36px; transform: translateZ(1px); border-radius: 50%; background: {accent2}; box-shadow: 0 0 26px {accent2}, 0 0 50px rgba(255,255,255,.5); }}
.orbit {{ position: absolute; border: 1px solid rgba(255,255,255,.52); border-radius: 50%; left: 50%; top: 50%; transform-style: preserve-3d; }}
.orbit-one {{ width: 202px; height: 76px; margin: -38px -101px; transform: rotateX(68deg) rotateZ(-18deg); animation: ringOrbit 8s linear infinite; }}
.orbit-two {{ width: 164px; height: 164px; margin: -82px; transform: rotateY(67deg) rotateZ(30deg); animation: ringOrbit 11s linear infinite reverse; }}
.orbit-dot {{ position: absolute; width: 9px; height: 9px; border-radius: 50%; background: #fff; box-shadow: 0 0 13px rgba(255,255,255,.9); top: -5px; left: 31%; }}
# .hero-visual{{width:50%}}
}}

.hero:before {{
  content: ""; position: absolute; z-index: -1; inset: -30% auto -30% -45%; width: 36%;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.16), transparent);
  transform: skewX(-18deg); animation: heroSheen 1.35s .25s ease-out both;
}}
.hero:after {{
  content: ""; position: absolute; z-index: -1; width: 260px; height: 260px; right: -75px; top: -130px;
  border-radius: 50%; border: 1px solid rgba(255,255,255,.3);
  box-shadow: 0 0 0 28px rgba(255,255,255,.07), 0 0 0 58px rgba(255,255,255,.045), 0 0 0 90px rgba(255,255,255,.025);
  animation: orbitFloat 9s ease-in-out infinite alternate;
}}
.hero-content {{ position: relative; z-index: 2; max-width: 760px; }}
.hero h1 {{ font-family: 'Manrope', sans-serif; font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; margin: 0; color: #fff !important; }}
.hero p {{ margin: .55rem 0 0; opacity: .95; font-size: 1.05rem; color: #fff !important; }}
.eyebrow {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .16em; font-weight: 700; opacity: .9; color: #fff !important; }}
h1, h2, h3, h4, h5, h6, p, label, li, span {{ color: inherit; }}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {{
  color: {text_color};
}}
[data-testid="stCaptionContainer"] {{ color: {muted_color}; }}
.panel {{
  border: 1px solid {grid_color}; border-radius: 18px;
  padding: 1rem 1.2rem; background: {panel_bg};
  box-shadow: 0 7px 24px {accent}0d; backdrop-filter: blur(12px);
  transition: border-color .22s ease, box-shadow .22s ease, transform .22s ease;
}}
.panel:hover {{ border-color: {accent}88; box-shadow: 0 12px 30px {accent}18; transform: translateY(-2px); }}
div[data-testid="stMetric"] {{
  position: relative; overflow: hidden;
  background: linear-gradient(145deg, {metric_bg}, {panel_bg}); border: 1px solid {grid_color};
  padding: 1rem 1.1rem; border-radius: 18px;
  box-shadow: 0 7px 24px {accent}12;
  animation: cardEnter .55s cubic-bezier(.2,.75,.25,1) both;
  transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
}}
div[data-testid="stMetric"]:hover {{ transform: translateY(-4px); border-color: {accent}a8; box-shadow: 0 14px 32px {accent}22; }}
div[data-testid="stMetric"]:after {{
  content: ""; position: absolute; left: 0; top: 0; height: 2px; width: 100%;
  background: linear-gradient(90deg, transparent, {accent}, {accent2}, transparent); opacity: .85;
}}
div[data-testid="stHorizontalBlock"] div[data-testid="stMetric"]:nth-child(1) {{ animation-delay: .04s; }}
div[data-testid="stHorizontalBlock"] div[data-testid="stMetric"]:nth-child(2) {{ animation-delay: .10s; }}
div[data-testid="stHorizontalBlock"] div[data-testid="stMetric"]:nth-child(3) {{ animation-delay: .16s; }}
div[data-testid="stHorizontalBlock"] div[data-testid="stMetric"]:nth-child(4) {{ animation-delay: .22s; }}
div[data-testid="stMetricLabel"] {{ color: {muted_color} !important; }}
div[data-testid="stMetricValue"] {{ color: {accent} !important; font-weight: 800; }}
div[data-testid="stMetricDelta"] {{ color: {accent2} !important; }}
.stButton > button {{
  border-radius: 12px; font-weight: 650; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(110deg,{accent},{accent3}); color: white !important; border: 0;
  min-height: 3rem; box-shadow: 0 7px 18px {accent}35; background-size: 150% 100%;
}}
.stButton > button[kind="primary"]:hover {{
  transform: translateY(-2px); box-shadow: 0 12px 26px {accent}45; background-position: 100% 0;
}}
.stButton > button:focus-visible {{ outline: 2px solid {accent}; outline-offset: 3px; }}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {{ transition: border-color .18s ease, box-shadow .18s ease; }}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
[data-testid="stNumberInput"] input:focus {{ border-color: {accent} !important; box-shadow: 0 0 0 3px {accent}20; }}
div[data-testid="stTabs"] {{ border-bottom: 1px solid {grid_color}; }}
div[data-testid="stTabs"] button {{
  position: relative; color: {muted_color} !important; font-weight: 650;
  transition: color .2s ease, background-color .2s ease; border-radius: 10px 10px 0 0;
}}
div[data-testid="stTabs"] button:hover {{ color: {accent} !important; background: {accent}0d; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{
  color: {accent} !important; border-bottom-color: {accent} !important;
}}
div[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid {grid_color}; border-radius: 12px; overflow: hidden;
}}
input, textarea {{ color: {text_color} !important; }}
@keyframes heroEnter {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes cardEnter {{
  from {{ opacity: 0; transform: translateY(12px) scale(.99); }}
  to {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
@keyframes heroSheen {{
  from {{ left: -45%; opacity: 0; }}
  20% {{ opacity: 1; }}
  to {{ left: 130%; opacity: 0; }}
}}
@keyframes orbitFloat {{
  from {{ transform: translate3d(0, 0, 0) rotate(0deg); }}
  to {{ transform: translate3d(-16px, 14px, 0) rotate(8deg); }}
}}
@keyframes cubeFloat {{
  0%, 100% {{ transform: translate3d(0, 0, 0); }}
  50% {{ transform: translate3d(0, -10px, 20px); }}
}}
@keyframes cubeSpin {{
  from {{ transform: translate(-50%, -50%) rotateX(-18deg) rotateY(0deg); }}
  to {{ transform: translate(-50%, -50%) rotateX(342deg) rotateY(360deg); }}
}}
@keyframes ringOrbit {{ to {{ rotate: 360deg; }} }}
/* Keep the visual language clean on small screens. */
@media (max-width: 760px) {{
  .block-container {{ padding-top: .8rem; }}
  .hero {{ padding: 1.35rem 1.2rem; border-radius: 18px; }}
  .hero p {{ font-size: .95rem; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: .01ms !important; transition-duration: .01ms !important; }}
}}

</style>
""", unsafe_allow_html=True)


set_plot_theme(theme)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Interactive machine learning laboratory</div>
  <h1>📈 Regression Explorer</h1>
  <p>Explore datasets, fit regression models, diagnose errors, and compare performance — all in one workspace.</p>
 <div class="hero-visual" style="padding-left: 105rem; top: -60px; position: relative;" aria-hidden="true">
    <div class="cube-scene">
      <div class="orbit orbit-one"><span class="orbit-dot"></span></div>
      <div class="orbit orbit-two"><span class="orbit-dot"></span></div>
      <div class="cube"><span class="cube-face cube-front"></span><span class="cube-face cube-back"></span><span class="cube-face cube-right"></span><span class="cube-face cube-left"></span><span class="cube-face cube-top"></span><span class="cube-face cube-bottom"></span><span class="cube-core"></span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎛️ Experiment controls")
    dataset_name = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()))
    st.caption(DATASET_OPTIONS[dataset_name])
    st.divider()
    model_name = st.selectbox("Regression method", list(MODEL_OPTIONS.keys()))
    st.caption(MODEL_OPTIONS[model_name])
    test_size = st.slider("Test set (%)", min_value=10, max_value=40, value=20, step=5)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    st.markdown("### Model parameters")
    params = {}
    if model_name in ("Ridge Regression", "Lasso Regression", "Elastic Net"):
        params["alpha"] = st.slider("Regularization strength (α)", 0.0001, 10.0, 1.0, step=0.1)
    if model_name == "Elastic Net":
        params["l1_ratio"] = st.slider("L1 ratio", 0.0, 1.0, 0.5, step=0.05)
    if model_name == "Polynomial Regression":
        params["degree"] = st.slider("Polynomial degree", 2, 4, 2)
    if model_name == "Decision Tree Regression":
        params["max_depth"] = st.slider("Maximum tree depth", 2, 20, 6)
    if model_name == "Random Forest Regression":
        params["n_estimators"] = st.slider("Number of trees", 50, 300, 100, step=50)
        params["max_depth"] = st.slider("Maximum tree depth (0 = unlimited)", 0, 30, 0, step=2)
    if model_name == "K-Nearest Neighbors":
        params["n_neighbors"] = st.slider("Neighbors (k)", 2, 25, 5)
    run = st.button("🚀 Fit selected model", type="primary", use_container_width=True)
    st.caption("The selected model is evaluated on a held-out test set.")

try:
    with st.spinner(f"Loading {dataset_name}…"):
        df, target, dataset_note = load_dataset(dataset_name)
except Exception as exc:
    st.error(f"Could not load this dataset: {exc}")
    st.info("Check your internet connection for datasets that need a first-time download.")
    st.stop()

if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}

cache_key = (dataset_name, model_name, int(test_size), int(random_state), tuple(sorted(params.items())))
if run or cache_key not in st.session_state.results_cache:
    with st.spinner("Preprocessing data and training model…"):
        try:
            result = fit_and_evaluate(df, target, model_name, params, test_size / 100, int(random_state))
            st.session_state.results_cache[cache_key] = result
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()
result = st.session_state.results_cache[cache_key]

st.markdown(f"### 🧪 {dataset_name}")
st.caption(dataset_note)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(df):,}")
c2.metric("Input features", str(df.drop(columns=[target]).shape[1]))
c3.metric("Target variable", target)
c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

tabs = st.tabs(["📊 Overview", "⚖️ Model comparison", "🔎 Relationships", "🧭 Residuals", "🎓 Advanced analysis", "🗂️ Dataset"])

with tabs[0]:
    st.markdown("#### Test-set performance")
    metric_cards(result["metrics"])
    left, right = st.columns([1.65, 1])
    with left:
        st.plotly_chart(actual_vs_predicted(result), use_container_width=True, key="actual_pred")
    with right:
        st.markdown("##### Prediction snapshot")
        preview = result["predictions"].copy()
        preview.columns = ["Actual", "Predicted", "Residual"]
        st.dataframe(preview.head(12).style.format("{:.3f}"), use_container_width=True, height=420)
    st.info("Metrics are computed on held-out test observations, not on the training data.")

with tabs[1]:
    st.markdown("#### Compare all regression methods on this dataset")
    with st.spinner("Training comparison models on the same split…"):
        comparison_key = ("compare", dataset_name, int(test_size), int(random_state))
        if comparison_key not in st.session_state.results_cache:
            comparison = {}
            for candidate in MODEL_OPTIONS:
                try:
                    comparison[candidate] = fit_and_evaluate(
                        df, target, candidate, {}, test_size / 100, int(random_state)
                    )
                except Exception as exc:
                    comparison[candidate] = {"error": str(exc)}
            st.session_state.results_cache[comparison_key] = comparison
        comparison = st.session_state.results_cache[comparison_key]
    rows = []
    for name, item in comparison.items():
        if "metrics" in item:
            rows.append({"Model": name, **item["metrics"]})
    comparison_df = pd.DataFrame(rows)
    if not comparison_df.empty:
        display_df = comparison_df.sort_values("RMSE").reset_index(drop=True)
        st.dataframe(display_df.style.format({
            "MAE":"{:.4f}", "MSE":"{:.4f}", "RMSE":"{:.4f}", "R²":"{:.4f}"
        }), use_container_width=True, hide_index=True)
        metric_choice = st.selectbox("Metric for comparison chart", ["RMSE", "MAE", "MSE", "R²"], key="compare_metric")
        st.plotly_chart(model_comparison_chart(comparison_df, metric_choice), use_container_width=True)
        best = display_df.iloc[0]
        st.success(f"Lowest test RMSE in this run: **{best['Model']}** (RMSE = {best['RMSE']:.4f}). Consider MAE and R² as well before drawing a final conclusion.")
    else:
        st.warning("No comparison models completed successfully.")

with tabs[2]:
    st.markdown("#### Explore relationships between variables")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target]
    selected_feature = st.selectbox("Feature to inspect", feature_cols)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(feature_relationship(df, selected_feature, target), use_container_width=True)
    with col2:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

with tabs[3]:
    st.markdown("#### Diagnose prediction errors")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(residual_plot(result), use_container_width=True)
    with col2:
        st.plotly_chart(residual_distribution(result), use_container_width=True)
    residuals = result["predictions"]["residual"]
    st.markdown("##### Residual summary")
    st.dataframe(residuals.describe().to_frame("Residual").T.style.format("{:.4f}"), use_container_width=True)
    st.caption("A good residual plot has no obvious pattern around zero. Residual diagnostics are visual checks, not proof that every regression assumption holds.")

with tabs[4]:
    st.markdown("#### Model interpretation")
    st.plotly_chart(coefficient_chart(result), use_container_width=True)
    st.markdown("##### Automated observations")
    m = result["metrics"]
    st.write(f"- **R² = {m['R²']:.4f}:** the model explains approximately {max(0, m['R²'])*100:.1f}% of the test-target variance when R² is positive.")
    st.write(f"- **RMSE = {m['RMSE']:.4f}** and **MAE = {m['MAE']:.4f}**. RMSE penalizes large errors more heavily than MAE.")
    st.write("- Use the comparison tab to select a model using multiple metrics and visual diagnostics; a single score should not be treated as the whole conclusion.")
    st.download_button(
        "⬇️ Download prediction results (CSV)",
        result["predictions"].rename(columns={"actual":"Actual", "predicted":"Predicted", "residual":"Residual"}).to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset_name.lower().replace(' ','_')}_{model_name.lower().replace(' ','_')}_predictions.csv",
        mime="text/csv",
    )

with tabs[5]:
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(100), use_container_width=True, height=420)
    st.caption(f"Showing up to 100 of {len(df):,} rows.")
    st.markdown("##### Numeric summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    st.download_button("⬇️ Download loaded dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{dataset_name.lower().replace(' ','_')}.csv", mime="text/csv")

st.markdown("---")
st.caption("Regression Explorer • Educational analysis tool • Model quality depends on dataset, split, features, and assumptions.")
