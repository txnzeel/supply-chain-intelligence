"""Five-layer executive control tower for supply-chain decision intelligence."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

FILES = {
    "optimization": PROCESSED / "inventory_optimization.csv",
    "decisions": PROCESSED / "optimization_decisions.csv",
    "scenario": PROCESSED / "scenario_results.csv",
    "scenario_decisions": PROCESSED / "scenario_decisions.csv",
    "actions": PROCESSED / "executive_action_plan.csv",
    "priorities": PROCESSED / "executive_priorities.csv",
    "brief": PROCESSED / "executive_management_brief.csv",
    "risk": PROCESSED / "risk" / "supply_chain_risk.csv",
    "registry": PROJECT_ROOT / "models" / "registry" / "champion_models.csv",
    "monitoring": PROCESSED / "modeling" / "model_monitoring.csv",
    "monte_carlo": PROCESSED / "modeling" / "monte_carlo_inventory_risk.csv",
}

LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"

LAYERS = {
    "Executive Summary": [
        "Executive Overview", "Portfolio Health", "Executive Action Status",
        "Executive Takeaways",
    ],
    "Management Decisions": [
        "Executive Action Center", "Critical Management Actions",
        "Top Executive Priorities", "Executive Management Brief",
        "Management Action Center",
    ],
    "Performance & Economics": [
        "Cost vs Service Performance", "Risk Class Performance",
        "Category Performance", "Decision Impact", "Business Impact",
    ],
    "Risk & Scenario Intelligence": [
        "Priority Distribution", "Supplier Risk Exposure",
        "Critical Inventory Exposure", "Scenario Analysis",
        "Monte Carlo Policy Robustness",
    ],
    "Portfolio Exploration": [
        "Top Optimization Opportunities", "Product-Level Investigation",
        "Forecast & Model Health", "Data Pipeline Health", "Export",
    ],
}


st.set_page_config(
    page_title="Supply Chain Decision Intelligence",
    page_icon="◈",
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
      background:radial-gradient(circle at 92% 8%,rgba(0,139,86,.11),transparent 26%),
                 linear-gradient(145deg,#061d29 0%,#082631 55%,#061d29 100%);
      color:var(--ink);
    }
    header,[data-testid="stHeader"] { background:transparent !important; }
    [data-testid="stMain"] { flex:1 1 auto !important; min-width:0 !important; }
    .block-container,[data-testid="stMainBlockContainer"] {
      padding:1.25rem clamp(.8rem,2.4vw,2.4rem) 3rem !important;
      max-width:none !important; width:100% !important; margin:0 !important;
    }

    /* Sidebar and the five decision-layer navigation states. */
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
      color:#fff !important; background:#003f5c !important; border:1px solid var(--lime) !important;
      border-radius:9px !important; box-shadow:0 6px 18px rgba(0,25,32,.32) !important;
    }
    .stApp span.material-symbols-rounded,.stApp span.material-symbols-outlined,
    .stApp [data-testid="stIconMaterial"],.stApp [data-testid="stIconMaterial"] span {
      font-family:"Material Symbols Rounded","Material Symbols Outlined" !important;
      font-weight:normal !important; font-style:normal !important; letter-spacing:normal !important;
      text-transform:none !important; white-space:nowrap !important; font-feature-settings:"liga" !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding:1rem .9rem 1.4rem !important; }
    [data-testid="stSidebar"] h3 { color:#fff !important; letter-spacing:.2px; }
    [data-testid="stSidebar"] hr { border-color:rgba(120,165,10,.34) !important; }
    [data-testid="stSidebar"] label { color:#e3efed !important; font-weight:650 !important; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:.35rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
      padding:.62rem .7rem !important; border:1px solid rgba(38,113,123,.65);
      border-radius:8px; background:rgba(6,29,41,.30); transition:.18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
      background:rgba(0,101,114,.38); border-color:var(--green);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
      background:linear-gradient(90deg,var(--teal),rgba(0,139,86,.82));
      border-color:var(--lime); color:#fff !important;
      box-shadow:0 6px 16px rgba(0,29,36,.25);
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
      background:#082a35 !important; border-color:#26717b !important; border-radius:8px !important;
      max-height:118px; overflow-y:auto;
    }
    [data-baseweb="tag"] { background:#006572 !important; border:1px solid #78a50a !important;
      border-radius:5px !important; box-shadow:0 2px 6px rgba(0,25,32,.28); }
    [data-baseweb="tag"],[data-baseweb="tag"] * { color:#fff !important;
      -webkit-text-fill-color:#fff !important; font-weight:700 !important; opacity:1 !important; }
    [role="option"][aria-selected="true"],[role="option"][aria-selected="true"] * {
      background:#006572 !important; color:#fff !important;
    }

    /* Shared cards, headings, tables and charts inherited by every layout. */
    [data-testid="stMetric"] {
      background:linear-gradient(150deg,rgba(0,101,114,.23),rgba(0,139,86,.08)),var(--panel);
      border:1px solid var(--line); border-left:3px solid var(--teal);
      padding:16px 18px; border-radius:9px; min-height:112px;
      box-shadow:0 8px 24px rgba(0,22,29,.2);
    }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; text-transform:uppercase; letter-spacing:.06em; }
    [data-testid="stMetricValue"] { color:var(--ink) !important; }
    [data-testid="stColumn"]:has(.danger-kpi-marker) [data-testid="stMetric"] {
      background:linear-gradient(145deg,rgba(175,38,63,.32),rgba(92,19,36,.30)),#301722;
      border-color:#9f354c; border-left-color:var(--red);
    }
    [data-testid="stColumn"]:has(.danger-kpi-marker) [data-testid="stMetricValue"] { color:#ffb5c0 !important; }
    .danger-kpi-marker { display:none; }
    [data-testid="stElementContainer"]:has(.danger-kpi-marker),
    .element-container:has(.danger-kpi-marker) {
      display:none !important; height:0 !important; min-height:0 !important;
      margin:0 !important; padding:0 !important;
    }
    .eyebrow { color:#78a50a; font-size:.75rem; font-weight:750; letter-spacing:.17em; text-transform:uppercase; }
    .hero { color:var(--ink); font-size:clamp(1.75rem,4vw,2.45rem); font-weight:780;
      margin:.25rem 0; letter-spacing:-.035em; }
    .subhero { color:var(--muted); max-width:900px; margin-bottom:1.4rem; line-height:1.6; }
    .section-title { margin-top:1.8rem; padding:1rem 0 0 .85rem; border-top:1px solid var(--line);
      border-left:3px solid var(--lime); }
    .section-title h3 { color:var(--ink); margin:0; font-size:1.12rem; }
    .section-title p { color:var(--muted); margin:.25rem 0 .8rem; font-size:.88rem; }
    .status { display:inline-block; padding:5px 10px; border-radius:20px; font-size:.75rem;
      background:rgba(0,139,86,.20); color:#b9f2d4; border:1px solid var(--green); }
    [data-testid="stDataFrame"],[data-testid="stVegaLiteChart"] {
      background:var(--panel); border:1px solid var(--line); border-radius:9px;
      box-shadow:0 7px 22px rgba(0,22,29,.18); overflow:hidden;
    }
    .stButton button,.stDownloadButton button { background:var(--teal); color:#fff;
      border:1px solid var(--green); border-radius:7px; }
    .stButton button:hover,.stDownloadButton button:hover { border-color:var(--lime); background:var(--green); }

    @media (max-width:900px) {
      [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:.8rem !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex:1 1 min(100%,320px) !important; width:100% !important; min-width:0 !important;
      }
    }
    @media (max-width:600px) {
      .block-container,[data-testid="stMainBlockContainer"] { padding-left:.7rem !important; padding-right:.7rem !important; }
      [data-testid="stMetric"] { min-height:96px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def materialized(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_SIGNATURE)) != LFS_SIGNATURE


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for name, path in FILES.items():
        if materialized(path):
            try:
                data[name] = pd.read_csv(path)
            except Exception:
                data[name] = pd.DataFrame()
        else:
            data[name] = pd.DataFrame()
    return data


def number(frame: pd.DataFrame, column: str, operation: str = "sum") -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    if operation == "mean":
        return float(values.mean()) if values.notna().any() else 0.0
    return float(values.sum()) if values.notna().any() else 0.0


def money(value: float) -> str:
    """Format modeled financial values in Indian rupees."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 10_000_000:
        return f"{sign}₹{value / 10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"{sign}₹{value / 100_000:.2f} L"
    if value >= 1_000:
        return f"{sign}₹{value / 1_000:.1f}K"
    return f"{sign}₹{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


PALETTE = ["#003f5c", "#006572", "#008b56", "#78a50a"]
CHART_SURFACE = "#0b3441"


def danger_metric(label: str, value: str, **kwargs) -> None:
    """Use the red card treatment for adverse or urgent indicators."""
    st.markdown('<span class="danger-kpi-marker" aria-hidden="true"></span>', unsafe_allow_html=True)
    st.metric(label, value, **kwargs)


def themed_hbar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    *,
    value_title: str = "",
    percentage: bool = False,
    severity: bool = False,
) -> alt.Chart:
    """Responsive horizontal chart shared by all five decision layouts."""
    chart_data = frame[[category, value]].dropna().copy()
    chart_data[category] = chart_data[category].astype(str)
    chart_data[value] = pd.to_numeric(chart_data[value], errors="coerce").fillna(0)
    height = max(170, min(440, 38 * len(chart_data) + 45))
    value_format = ".1%" if percentage else ",.0f"

    if severity:
        color = alt.Color(
            f"{category}:N",
            scale=alt.Scale(
                domain=["Critical", "High", "Medium", "Low", "Fragile", "Robust"],
                range=["#d94a70", "#ff6f91", "#78a50a", "#008b56", "#d94a70", "#008b56"],
            ),
            legend=None,
        )
    else:
        color = alt.Color(
            f"{value}:Q",
            scale=alt.Scale(range=PALETTE),
            legend=None,
        )

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.64})
        .encode(
            x=alt.X(f"{value}:Q", title=value_title or None, axis=alt.Axis(grid=True, labelFlush=False)),
            y=alt.Y(
                f"{category}:N",
                title=None,
                sort=alt.EncodingSortField(field=value, order="descending"),
                axis=alt.Axis(labelLimit=260, labelPadding=8),
            ),
            color=color,
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category.replace("_", " ").title()),
                alt.Tooltip(f"{value}:Q", title=value_title or value.replace("_", " ").title(), format=value_format),
            ],
        )
        .properties(height=height, padding={"left": 6, "right": 18, "top": 8, "bottom": 8})
        .configure_view(fill=CHART_SURFACE, stroke=None)
        .configure_axis(
            labelColor="#c5d9d7", titleColor="#a9c1bf", gridColor="#174b58",
            domainColor="#26717b", tickColor="#26717b", labelFontSize=11,
        )
    )
    return chart


def section(title: str, purpose: str) -> None:
    st.markdown(
        f'<div class="section-title"><h3>{title}</h3><p>{purpose}</p></div>',
        unsafe_allow_html=True,
    )


def first_available(*frames: pd.DataFrame) -> pd.DataFrame:
    return next((frame for frame in frames if not frame.empty), pd.DataFrame())


def available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    available = []
    for column in columns:
        if column in frame.columns and column not in available:
            available.append(column)
    return available


data = load_data()
optimization = data["optimization"]
decisions = data["decisions"]
risk = data["risk"]
actions = data["actions"]
priorities = data["priorities"]
monitoring = data["monitoring"]
monte_carlo = data["monte_carlo"]
registry = data["registry"]

with st.sidebar:
    st.markdown("### ◈ Decision Intelligence")
    st.caption("Designed & developed by Tanzeel Aftab")
    st.caption("Supply-chain control tower")
    active_layer = st.radio("Decision layer", list(LAYERS), label_visibility="collapsed")
    st.divider()
    st.markdown("**Portfolio filters**")

    filter_source = first_available(decisions, optimization, risk)
    categories = []
    if "category" in filter_source:
        categories = sorted(filter_source["category"].dropna().astype(str).unique())
    selected_categories = st.multiselect("Category", categories, default=categories)

    risk_column = next((c for c in ["risk_class", "overall_risk_class", "risk_level"] if c in filter_source), None)
    risk_levels = sorted(filter_source[risk_column].dropna().astype(str).unique()) if risk_column else []
    selected_risk = st.multiselect("Risk class", risk_levels, default=risk_levels)
    st.divider()
    ready_count = sum(not frame.empty for frame in data.values())
    st.markdown(f'<span class="status">{ready_count}/{len(data)} assets ready</span>', unsafe_allow_html=True)


def filtered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not result.empty and selected_categories and "category" in result:
        result = result[result["category"].astype(str).isin(selected_categories)]
    candidate = next((c for c in ["risk_class", "overall_risk_class", "risk_level"] if c in result), None)
    if candidate and selected_risk:
        result = result[result[candidate].astype(str).isin(selected_risk)]
    return result


optimization = filtered(optimization)
decisions = filtered(decisions)
risk = filtered(risk)

st.markdown('<div class="eyebrow">Executive control tower</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero">{active_layer}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subhero">A decision-first view of portfolio health, management actions, '
    'economic performance, uncertainty, and model reliability.</div>',
    unsafe_allow_html=True,
)


def render_executive_summary() -> None:
    base = first_available(decisions, optimization, risk)
    products = base["product_id"].nunique() if "product_id" in base else len(base)
    savings = number(base, "cost_savings")
    fill_change = number(base, "fill_rate_change", "mean")
    critical = 0
    risk_col = next((c for c in ["risk_class", "overall_risk_class", "risk_level"] if c in base), None)
    if risk_col:
        critical = base[risk_col].astype(str).str.contains("critical", case=False).sum()

    section("Executive Overview", "The four numbers that frame the current portfolio decision.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products in scope", f"{products:,}")
    c2.metric("Modeled cost savings", money(savings))
    c3.metric("Average fill-rate change", pct(fill_change))
    with c4:
        danger_metric("Critical exposures", f"{critical:,}")

    section("Portfolio Health", "Risk concentration and intervention coverage across the selected portfolio.")
    left, right = st.columns(2)
    with left:
        if risk_col:
            risk_counts = base[risk_col].value_counts().rename_axis("risk").reset_index(name="products")
            st.altair_chart(themed_hbar(risk_counts, "risk", "products", severity=True), width="stretch")
        else:
            st.info("Risk classifications are unavailable.")
    with right:
        action_col = next((c for c in ["recommended_action", "management_decision"] if c in base), None)
        if action_col:
            action_counts = base[action_col].value_counts().head(8).rename_axis("action").reset_index(name="products")
            st.altair_chart(themed_hbar(action_counts, "action", "products"), width="stretch")
        else:
            st.info("Management actions are unavailable.")

    section("Executive Action Status", "Whether generated actions are concentrated, actionable, and ready for review.")
    action_base = first_available(actions, decisions)
    cols = available_columns(action_base, ["product_id", "priority", "management_decision", "recommended_action", "cost_savings"])
    st.dataframe(action_base[cols].head(12), width="stretch", hide_index=True)

    section("Executive Takeaways", "Automatically derived implications for management attention.")
    fragile = int(monte_carlo["policy_robustness"].eq("Fragile").sum()) if "policy_robustness" in monte_carlo else 0
    alerts = int(monitoring["retraining_recommended"].sum()) if "retraining_recommended" in monitoring else 0
    st.markdown(
        f"- **{critical:,}** products currently carry critical risk classifications.\n"
        f"- **{fragile:,}** inventory policies are fragile under probabilistic stress testing.\n"
        f"- **{alerts:,}** champion models currently require investigation or retraining.\n"
        f"- Modeled portfolio savings total **{money(savings)}** for the selected scope."
    )


def render_management_decisions() -> None:
    action_base = first_available(actions, decisions)
    section("Executive Action Center", "A single operating queue for generated management interventions.")
    columns = available_columns(action_base, ["product_id", "category", "priority", "management_decision", "recommended_action", "financial_exposure", "cost_savings"])
    st.dataframe(action_base[columns].head(30), width="stretch", hide_index=True)

    section("Critical Management Actions", "Interventions with the highest urgency and modeled business exposure.")
    critical = action_base
    if "priority" in critical:
        critical = critical[critical["priority"].astype(str).str.contains("P1|Critical", case=False, regex=True)]
    sort_col = next((c for c in ["urgency_score", "decision_impact_score", "cost_savings"] if c in critical), None)
    if sort_col:
        critical = critical.sort_values(sort_col, ascending=False)
    st.dataframe(critical[columns].head(15), width="stretch", hide_index=True)

    section("Top Executive Priorities", "Ranked actions for leadership review and ownership assignment.")
    priority_base = first_available(priorities, critical)
    pcols = available_columns(priority_base, ["priority_rank", "product_id", "category", "priority", "executive_action", "management_decision", "financial_exposure"])
    st.dataframe(priority_base[pcols].head(15), width="stretch", hide_index=True)

    section("Executive Management Brief", "The compact narrative output produced by the decision-support layer.")
    brief = data["brief"]
    if brief.empty:
        st.info("The management brief output is unavailable.")
    else:
        st.dataframe(brief, width="stretch", hide_index=True)

    section("Management Action Center", "Distribution of action types and the portfolio workload they create.")
    action_col = next((c for c in ["management_decision", "recommended_action", "executive_action"] if c in action_base), None)
    if action_col:
        counts = action_base[action_col].value_counts().rename_axis("action").reset_index(name="products")
        st.altair_chart(themed_hbar(counts, "action", "products"), width="stretch")


def render_performance() -> None:
    base = first_available(decisions, optimization)
    section("Cost vs Service Performance", "Whether optimization improves economics without sacrificing customer service.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline cost", money(number(base, "baseline_total_cost")))
    c2.metric("Optimized cost", money(number(base, "optimized_total_cost")))
    c3.metric("Baseline fill rate", pct(number(base, "baseline_fill_rate", "mean")))
    c4.metric("Optimized fill rate", pct(number(base, "optimized_fill_rate", "mean")))

    section("Risk Class Performance", "Economic and service outcomes segmented by inventory risk.")
    risk_col = next((c for c in ["risk_class", "overall_risk_class", "risk_level"] if c in base), None)
    if risk_col:
        aggregations = {c: "mean" for c in ["cost_savings", "fill_rate_change", "inventory_days_change"] if c in base}
        st.dataframe(base.groupby(risk_col).agg(aggregations).round(4), width="stretch")

    section("Category Performance", "Categories producing the largest opportunity or implementation burden.")
    if "category" in base:
        aggregations = {c: "sum" for c in ["cost_savings", "lost_sales_reduction"] if c in base}
        category = base.groupby("category").agg(aggregations).sort_values(next(iter(aggregations)), ascending=False)
        category_chart = category.reset_index()
        chart_value = next(iter(aggregations))
        st.altair_chart(themed_hbar(category_chart, "category", chart_value), width="stretch")

    section("Decision Impact", "Products ranked by combined financial, service, and urgency impact.")
    score = next((c for c in ["decision_impact_score", "urgency_score", "cost_savings"] if c in base), None)
    cols = available_columns(base, ["product_id", "category", "priority", score, "cost_savings", "fill_rate_change"])
    ranked = base.sort_values(score, ascending=False) if score else base
    st.dataframe(ranked[cols].head(25), width="stretch", hide_index=True)

    section("Business Impact", "How recommended interventions distribute across economic and service outcomes.")
    if "business_impact" in base:
        impact_counts = base["business_impact"].value_counts().rename_axis("impact").reset_index(name="products")
        st.altair_chart(themed_hbar(impact_counts, "impact", "products"), width="stretch")


def render_risk_scenarios() -> None:
    base = first_available(decisions, risk, optimization)
    section("Priority Distribution", "The urgency mix of the action portfolio.")
    if "priority" in base:
        priority_counts = base["priority"].value_counts().rename_axis("priority").reset_index(name="products")
        st.altair_chart(themed_hbar(priority_counts, "priority", "products", severity=True), width="stretch")

    section("Supplier Risk Exposure", "Products where supplier reliability can undermine inventory policy.")
    supplier_score = next((c for c in ["supplier_risk_score", "supplier_risk", "supplier_delay_rate"] if c in base), None)
    cols = available_columns(base, ["product_id", "category", "priority", supplier_score, "optimized_average_supplier_delay", "cost_savings"])
    supplier = base.sort_values(supplier_score, ascending=False) if supplier_score else base
    st.dataframe(supplier[cols].head(20), width="stretch", hide_index=True)

    section("Critical Inventory Exposure", "Products most exposed to stockouts, lost sales, or fragile policies.")
    exposure = base
    score = next((c for c in ["risk_score", "overall_risk_score", "optimized_lost_sales"] if c in exposure), None)
    ecols = available_columns(exposure, ["product_id", "category", "risk_class", score, "optimized_lost_sales", "optimized_stockout_rate"])
    if score:
        exposure = exposure.sort_values(score, ascending=False)
    st.dataframe(exposure[ecols].head(20), width="stretch", hide_index=True)

    section("Scenario Analysis", "Sensitivity of cost and service outcomes under alternative operating conditions.")
    scenario = data["scenario"]
    if scenario.empty:
        st.info("Scenario results are unavailable.")
    else:
        scols = available_columns(scenario, ["scenario", "scenario_name", "product_id", "total_cost", "fill_rate", "lost_sales", "inventory_days"])
        st.dataframe(scenario[scols].head(30), width="stretch", hide_index=True)

    section("Monte Carlo Policy Robustness", "Stockout probability under forecast-error and lead-time uncertainty.")
    if monte_carlo.empty:
        st.info("Run the Monte Carlo pipeline stage to populate this view.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Scenarios evaluated", f"{int(monte_carlo['simulations'].sum()):,}")
        with c2:
            danger_metric("Mean stockout probability", pct(number(monte_carlo, "stockout_probability", "mean")))
        with c3:
            danger_metric("Fragile policies", f"{monte_carlo['policy_robustness'].eq('Fragile').sum():,}")
        robustness = monte_carlo["policy_robustness"].value_counts().rename_axis("robustness").reset_index(name="products")
        st.altair_chart(themed_hbar(robustness, "robustness", "products", severity=True), width="stretch")


def render_exploration() -> None:
    base = first_available(decisions, optimization, risk)
    section("Top Optimization Opportunities", "Products with the greatest modeled economic opportunity.")
    sort_col = next((c for c in ["cost_savings", "decision_impact_score"] if c in base), None)
    ranked = base.sort_values(sort_col, ascending=False) if sort_col else base
    cols = available_columns(base, ["product_id", "category", "risk_class", "cost_savings", "fill_rate_change", "recommended_action"])
    st.dataframe(ranked[cols].head(30), width="stretch", hide_index=True)

    section("Product-Level Investigation", "Trace one product from model choice to inventory decision and risk.")
    products = sorted(base["product_id"].dropna().astype(str).unique()) if "product_id" in base else []
    if products:
        product = st.selectbox("Product", products)
        blocks = []
        for name, frame in data.items():
            if not frame.empty and "product_id" in frame:
                match = frame[frame["product_id"].astype(str).eq(product)]
                if not match.empty:
                    match = match.copy()
                    match.insert(0, "source", name)
                    blocks.append(match)
        for block in blocks:
            st.dataframe(block, width="stretch", hide_index=True)

    section("Forecast & Model Health", "Champion coverage, forecast accuracy, and retraining status.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Champion models", f"{len(registry):,}")
    c2.metric("Mean validation WAPE", pct(number(registry, "validation_wape", "mean")))
    with c3:
        danger_metric("Retraining alerts", f"{int(number(monitoring, 'retraining_recommended')):,}")

    if not monitoring.empty:
        chart_left, chart_right = st.columns(2)

        with chart_left:
            drift_col = next((c for c in ["drift_status", "monitoring_status", "status"] if c in monitoring), None)
            if drift_col:
                st.markdown("#### Drift status")
                drift_counts = (
                    monitoring[drift_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .rename_axis("status")
                    .reset_index(name="models")
                )
                st.altair_chart(
                    themed_hbar(drift_counts, "status", "models", severity=True),
                    width="stretch",
                )
            else:
                st.info("Drift status is unavailable in the monitoring output.")

        with chart_right:
            model_col = next((c for c in ["selected_model", "model_name", "model_type"] if c in monitoring), None)
            wape_col = next((c for c in ["latest_window_wape", "validation_wape", "wape"] if c in monitoring), None)
            if model_col and wape_col:
                st.markdown("#### Latest error by model")
                error_by_model = (
                    monitoring.assign(**{wape_col: pd.to_numeric(monitoring[wape_col], errors="coerce")})
                    .groupby(model_col, dropna=False)[wape_col]
                    .mean()
                    .sort_values(ascending=False)
                    .head(12)
                    .rename_axis("model")
                    .reset_index(name="wape")
                )
                st.altair_chart(
                    themed_hbar(
                        error_by_model,
                        "model",
                        "wape",
                        value_title="Mean latest-window WAPE",
                        percentage=True,
                    ),
                    width="stretch",
                )
            elif model_col:
                st.markdown("#### Champion model distribution")
                model_counts = (
                    monitoring[model_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .head(12)
                    .rename_axis("model")
                    .reset_index(name="products")
                )
                st.altair_chart(themed_hbar(model_counts, "model", "products"), width="stretch")
            else:
                st.info("Model identifiers are unavailable in the monitoring output.")

        st.markdown("#### Monitoring queue")
        mcols = available_columns(monitoring, ["product_id", "selected_model", "drift_status", "monitoring_score", "latest_window_wape", "retraining_recommended"])
        st.dataframe(monitoring[mcols].head(25), width="stretch", hide_index=True)
    else:
        st.info("Run the model-monitoring pipeline stage to populate drift and retraining visuals.")

    section("Data Pipeline Health", "Availability and size of governed decision assets.")
    asset_rows = []
    for name, path in FILES.items():
        frame = data[name]
        asset_rows.append({"asset": name, "status": "Ready" if not frame.empty else "Missing", "rows": len(frame), "path": str(path.relative_to(PROJECT_ROOT))})
    st.dataframe(pd.DataFrame(asset_rows), width="stretch", hide_index=True)

    section("Export", "Download the selected portfolio for further analysis or audit.")
    export = base.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered portfolio", export, "filtered_supply_chain_portfolio.csv", "text/csv")


renderers = {
    "Executive Summary": render_executive_summary,
    "Management Decisions": render_management_decisions,
    "Performance & Economics": render_performance,
    "Risk & Scenario Intelligence": render_risk_scenarios,
    "Portfolio Exploration": render_exploration,
}
renderers[active_layer]()

st.divider()
st.caption(
    "Supply Chain Decision Intelligence Platform · Developed by Tanzeel Aftab · "
    "Financial values are modeled in INR using synthetic data."
)
