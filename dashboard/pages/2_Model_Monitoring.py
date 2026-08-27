"""Streamlit page for model registry, drift, and policy robustness monitoring."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_FILE = PROJECT_ROOT / "models" / "registry" / "champion_models.csv"
MONITORING_FILE = PROJECT_ROOT / "data" / "processed" / "modeling" / "model_monitoring.csv"
MONTE_CARLO_FILE = PROJECT_ROOT / "data" / "processed" / "modeling" / "monte_carlo_inventory_risk.csv"

PALETTE = ["#003f5c", "#006572", "#008b56", "#78a50a"]
SURFACE = "#0b3441"


st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy:#061d29; --panel:#0b3441; --panel-2:#082a35; --line:#155665;
        --ink:#f3fbf9; --muted:#a9c1bf; --blue:#003f5c; --teal:#006572;
        --green:#008b56; --lime:#78a50a; --red:#ff5c72;
    }

    html, body, .stApp { overflow-x:hidden !important; }
    .stApp {
        color:var(--ink);
        background:
            radial-gradient(circle at 92% 8%,rgba(0,139,86,.11),transparent 26%),
            linear-gradient(145deg,#061d29 0%,#082631 55%,#061d29 100%);
    }
    header,[data-testid="stHeader"] { background:transparent !important; }
    [data-testid="stMain"] { flex:1 1 auto !important; min-width:0 !important; }
    .block-container,[data-testid="stMainBlockContainer"] {
        padding:1.25rem clamp(.8rem,2.4vw,2.4rem) 3rem !important;
        max-width:none !important; width:100% !important; margin:0 !important;
    }

    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#003f5c 0%,#063943 52%,#072c32 100%) !important;
        border-right:1px solid var(--teal) !important;
    }
    [data-testid="stSidebar"][aria-expanded="true"] {
        width:min(300px,86vw) !important; min-width:min(300px,86vw) !important;
        max-width:min(300px,86vw) !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"]:not(:has([data-testid="stSidebarCollapseButton"])) {
        width:0 !important; min-width:0 !important; max-width:0 !important;
        flex-basis:0 !important; border-right:0 !important;
    }
    [data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"] {
        display:flex !important; visibility:visible !important; position:fixed !important;
        top:.65rem !important; left:.7rem !important; z-index:1000000 !important;
        opacity:1 !important; transform:none !important;
    }
    [data-testid="collapsedControl"] button,[data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button {
        color:#fff !important; background:#003f5c !important;
        border:1px solid var(--lime) !important; border-radius:9px !important;
        box-shadow:0 6px 18px rgba(0,25,32,.32) !important;
    }
    .stApp span.material-symbols-rounded,.stApp span.material-symbols-outlined,
    .stApp [data-testid="stIconMaterial"],.stApp [data-testid="stIconMaterial"] span {
        font-family:"Material Symbols Rounded","Material Symbols Outlined" !important;
        font-weight:normal !important; font-style:normal !important;
        letter-spacing:normal !important; text-transform:none !important;
        white-space:nowrap !important; font-feature-settings:"liga" !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding:1rem .9rem 1.4rem !important; }
    [data-testid="stSidebarNav"] a {
        margin:.18rem 0 !important; padding:.62rem .72rem !important;
        border:1px solid transparent; border-radius:8px !important;
        color:#dcebea !important; text-transform:capitalize !important;
    }
    [data-testid="stSidebarNav"] a:hover {
        background:rgba(0,101,114,.34) !important; border-color:rgba(0,139,86,.55);
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        color:#fff !important;
        background:linear-gradient(90deg,#006572,rgba(0,139,86,.76)) !important;
        border-color:#78a50a; font-weight:700 !important;
    }

    [data-testid="stMetric"] {
        min-height:112px; padding:16px 18px; border-radius:9px;
        background:linear-gradient(150deg,rgba(0,101,114,.23),rgba(0,139,86,.08)),var(--panel);
        border:1px solid var(--line); border-left:3px solid var(--teal);
        box-shadow:0 8px 24px rgba(0,22,29,.2);
    }
    [data-testid="stMetricLabel"] {
        color:var(--muted) !important; text-transform:uppercase; letter-spacing:.06em;
    }
    [data-testid="stMetricValue"] { color:var(--ink) !important; }
    [data-testid="stColumn"]:has(.danger-kpi-marker) [data-testid="stMetric"] {
        background:linear-gradient(145deg,rgba(175,38,63,.32),rgba(92,19,36,.30)),#301722;
        border-color:#9f354c; border-left-color:var(--red);
    }
    [data-testid="stColumn"]:has(.danger-kpi-marker) [data-testid="stMetricValue"] {
        color:#ffb5c0 !important;
    }
    .danger-kpi-marker { display:none; }
    [data-testid="stElementContainer"]:has(.danger-kpi-marker),
    .element-container:has(.danger-kpi-marker) {
        display:none !important; height:0 !important; min-height:0 !important;
        margin:0 !important; padding:0 !important;
    }

    .page-eyebrow {
        color:var(--lime); font-size:.75rem; font-weight:750;
        letter-spacing:.17em; text-transform:uppercase;
    }
    .page-title {
        color:var(--ink); font-size:clamp(1.8rem,4vw,2.5rem);
        font-weight:780; margin:.2rem 0; letter-spacing:-.035em;
    }
    .page-subtitle { color:var(--muted); max-width:850px; line-height:1.6; margin-bottom:1.4rem; }
    .section-heading {
        color:var(--ink); margin:1.8rem 0 .8rem; padding:.75rem 0 .1rem .8rem;
        border-top:1px solid var(--line); border-left:3px solid var(--lime);
        font-size:1.1rem; font-weight:700;
    }
    [data-testid="stDataFrame"],[data-testid="stVegaLiteChart"] {
        background:var(--panel); border:1px solid var(--line); border-radius:9px;
        box-shadow:0 7px 22px rgba(0,22,29,.18); overflow:hidden;
    }
    [data-baseweb="select"] > div {
        background:#082a35 !important; border-color:#26717b !important; border-radius:8px !important;
    }
    [data-baseweb="tag"] {
        background:#006572 !important; border:1px solid #78a50a !important;
        border-radius:5px !important;
    }
    [data-baseweb="tag"],[data-baseweb="tag"] * {
        color:#fff !important; -webkit-text-fill-color:#fff !important;
        font-weight:700 !important; opacity:1 !important;
    }
    [role="option"][aria-selected="true"],[role="option"][aria-selected="true"] * {
        background:#006572 !important; color:#fff !important;
    }

    @media (max-width:900px) {
        [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:.8rem !important; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex:1 1 min(100%,320px) !important; width:100% !important; min-width:0 !important;
        }
    }
    @media (max-width:600px) {
        .block-container,[data-testid="stMainBlockContainer"] {
            padding-left:.7rem !important; padding-right:.7rem !important;
        }
        [data-testid="stMetric"] { min-height:96px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def danger_metric(label: str, value: str) -> None:
    st.markdown('<span class="danger-kpi-marker" aria-hidden="true"></span>', unsafe_allow_html=True)
    st.metric(label, value)


def section_heading(title: str) -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)


def themed_hbar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    *,
    percentage: bool = False,
    severity: bool = False,
) -> alt.Chart:
    chart_data = frame[[category, value]].dropna().copy()
    chart_data[category] = chart_data[category].astype(str)
    chart_data[value] = pd.to_numeric(chart_data[value], errors="coerce").fillna(0)
    height = max(190, min(430, 40 * len(chart_data) + 42))

    if severity:
        color = alt.Color(
            f"{category}:N",
            scale=alt.Scale(
                domain=["Critical", "Alert", "Drift", "Fragile", "Watch", "Acceptable", "Robust", "Stable"],
                range=["#d94a70", "#ff5c72", "#ff6f91", "#d94a70", "#78a50a", "#006572", "#008b56", "#008b56"],
            ),
            legend=None,
        )
    else:
        color = alt.Color(f"{value}:Q", scale=alt.Scale(range=PALETTE), legend=None)

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=4, height={"band": .64})
        .encode(
            x=alt.X(f"{value}:Q", title=None, axis=alt.Axis(grid=True, labelFlush=False)),
            y=alt.Y(
                f"{category}:N", title=None,
                sort=alt.EncodingSortField(field=value, order="descending"),
                axis=alt.Axis(labelLimit=240, labelPadding=8),
            ),
            color=color,
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category.replace("_", " ").title()),
                alt.Tooltip(
                    f"{value}:Q",
                    title=value.replace("_", " ").title(),
                    format=".1%" if percentage else ",.0f",
                ),
            ],
        )
        .properties(height=height, padding={"left":6, "right":18, "top":8, "bottom":8})
        .configure_view(fill=SURFACE, stroke=None)
        .configure_axis(
            labelColor="#c5d9d7", titleColor="#a9c1bf", gridColor="#174b58",
            domainColor="#26717b", tickColor="#26717b", labelFontSize=11,
        )
    )


def materialized(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


missing = [path for path in (REGISTRY_FILE, MONITORING_FILE, MONTE_CARLO_FILE) if not materialized(path)]
if missing:
    st.error("Model lifecycle outputs have not been generated yet.")
    st.code(r"python python\data_engineering\run_pipeline.py --from register_models --force")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(REGISTRY_FILE),
        pd.read_csv(MONITORING_FILE),
        pd.read_csv(MONTE_CARLO_FILE),
    )


registry, monitoring, monte_carlo = load_data()
retraining_mask = (
    monitoring["retraining_recommended"]
    .fillna(False)
    .astype(str)
    .str.strip()
    .str.lower()
    .isin({"true", "1", "1.0", "yes"})
)

st.markdown('<div class="page-eyebrow">Model lifecycle intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Model Operations Center</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Champion-model coverage, forecast drift, retraining alerts, '
    'and inventory-policy robustness.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registered champions", f"{len(registry):,}")
c2.metric("Mean validation WAPE", f"{registry['validation_wape'].mean():.1%}")
with c3:
    danger_metric("Retraining alerts", f"{int(retraining_mask.sum()):,}")
with c4:
    danger_metric(
        "Policies above 5% risk",
        f"{monte_carlo['stockout_probability'].gt(0.05).sum():,}",
    )

section_heading("Model and drift distribution")
left, right = st.columns(2)

with left:
    st.markdown("#### Champion model distribution")
    model_distribution = (
        registry["selected_model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="products")
    )
    st.altair_chart(themed_hbar(model_distribution, "model", "products"), width="stretch")

with right:
    st.markdown("#### Drift status")
    drift_distribution = (
        monitoring["drift_status"]
        .value_counts()
        .rename_axis("status")
        .reset_index(name="products")
    )
    st.altair_chart(
        themed_hbar(drift_distribution, "status", "products", severity=True),
        width="stretch",
    )

section_heading("Highest-priority retraining alerts")
alert_columns = [
    "product_id", "selected_model", "model_version", "drift_status",
    "monitoring_score", "mean_shift_z", "volatility_ratio",
    "latest_window_wape", "performance_change_pct", "retraining_recommended",
]
available_alert_columns = [column for column in alert_columns if column in monitoring]
alerts = monitoring.loc[retraining_mask]
st.dataframe(alerts[available_alert_columns].head(100), width="stretch", hide_index=True)

section_heading("Monte Carlo inventory-policy risk")
robustness_options = ["Fragile", "Watch", "Acceptable", "Robust"]
robustness = st.multiselect(
    "Policy robustness",
    options=robustness_options,
    default=["Fragile", "Watch"],
)
policy_view = monte_carlo[monte_carlo["policy_robustness"].isin(robustness)]

risk_distribution = (
    policy_view["policy_robustness"]
    .value_counts()
    .rename_axis("robustness")
    .reset_index(name="products")
)
if not risk_distribution.empty:
    st.altair_chart(
        themed_hbar(risk_distribution, "robustness", "products", severity=True),
        width="stretch",
    )

policy_columns = [
    "product_id", "selected_model", "demand_class", "optimized_reorder_point",
    "p95_lead_time_demand", "stockout_probability", "expected_shortage_units",
    "policy_robustness",
]
available_policy_columns = [column for column in policy_columns if column in policy_view]
st.dataframe(policy_view[available_policy_columns].head(200), width="stretch", hide_index=True)

st.caption(
    "Drift is a monitoring signal, not automatic proof that a model is invalid. "
    "Critical alerts should trigger investigation and controlled retraining."
)
