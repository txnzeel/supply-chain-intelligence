import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

OPTIMIZATION_FILE = DATA_DIR / "inventory_optimization.csv"
DECISIONS_FILE = DATA_DIR / "optimization_decisions.csv"
OPTIMIZATION_SUMMARY_FILE = DATA_DIR / "optimization_summary.csv"
SCENARIO_RESULTS_FILE = DATA_DIR / "scenario_results.csv"
SCENARIO_SUMMARY_FILE = DATA_DIR / "scenario_summary.csv"
EXECUTIVE_ACTION_PLAN_FILE = DATA_DIR / "executive_action_plan.csv"
EXECUTIVE_ACTION_SUMMARY_FILE = DATA_DIR / "executive_action_summary.csv"
EXECUTIVE_PRIORITIES_FILE = DATA_DIR / "executive_priorities.csv"
EXECUTIVE_MANAGEMENT_BRIEF_FILE = DATA_DIR / "executive_management_brief.csv"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# RESPONSIVE LAYOUT, COLLAPSIBLE SIDEBAR & NO HORIZONTAL SCROLL
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       TYPEFACES — Control Tower identity
       Saira (signage) · Inter (body) · IBM Plex Mono (ledger)
       ===================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Saira:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');


    /* =====================================================
       HEADER / FOOTER
       ===================================================== */

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
    }

    /* Keep Streamlit's header available for the sidebar toggle, but make it
       visually blend into the dashboard. */
    header,
    [data-testid="stHeader"] {
        display: block !important;
        visibility: visible !important;
        height: 3rem !important;
        min-height: 3rem !important;
        background: transparent !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        width: min(300px, 86vw) !important;
        min-width: min(300px, 86vw) !important;
        max-width: min(300px, 86vw) !important;

        background: linear-gradient(
            180deg,
            #003f5c 0%,
            #063943 54%,
            #072c32 100%
        ) !important;

        border-right: 1px solid #006572 !important;

        box-sizing: border-box !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding: 1.15rem 1rem 1.5rem !important;
    }


    /* Make both sidebar controls clear and keyboard/mouse friendly. */

    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
    }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    button[kind="header"] {
        color: #f3fbf9 !important;
        background: rgba(0, 101, 114, 0.22) !important;
        border: 1px solid rgba(120, 165, 10, 0.62) !important;
        border-radius: 10px !important;
    }


    /* =====================================================
       PREVENT HORIZONTAL OVERFLOW
       ===================================================== */

    html,
    body {
        overflow-x: hidden !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    .stApp {
        overflow-x: hidden !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    [data-testid="stMainViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    [data-testid="stMain"] {
        overflow-x: hidden !important;
        max-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }


    /* =====================================================
       MAIN CONTENT CONTAINER
       ===================================================== */

    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
        width: 100% !important;

        padding: 1.25rem clamp(0.85rem, 2.4vw, 2.4rem) 3rem !important;
        margin: 0 !important;

        box-sizing: border-box !important;

        overflow-x: hidden !important;
    }


    /* =====================================================
       SIDEBAR CONTENT STYLING
       ===================================================== */

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small {
        color: #c5d9d7;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f3fbf9 !important;
        font-family: var(--font-signage) !important;
        letter-spacing: 0.2px;
        margin-bottom: 0.35rem !important;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.45rem !important;
    }

    [data-testid="stSidebar"] h3 {
        margin-top: 0.35rem !important;
        padding: 0.55rem 0.65rem !important;
        border-left: 3px solid #78a50a;
        border-radius: 0 6px 6px 0;
        background: rgba(0, 101, 114, 0.24);
        text-transform: uppercase;
        font-size: 0.76rem !important;
        letter-spacing: 1.1px;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(120, 165, 10, 0.34) !important;
        margin: 1rem 0 !important;
    }

    [data-testid="stSidebar"] label {
        color: #e3efed !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.25px;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        min-height: 42px;
        background: rgba(6, 29, 41, 0.72) !important;
        border: 1px solid #26717b !important;
        border-radius: 8px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
        border-color: #78a50a !important;
        box-shadow: 0 0 0 2px rgba(120, 165, 10, 0.18);
    }

    [data-testid="stSidebar"] .stMultiSelect {
        margin-bottom: 0.7rem;
    }

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
        max-height: 118px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #008b56 transparent;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        padding: 0 0.2rem;
        line-height: 1.55;
        color: #a9c1bf !important;
    }

    /* Multipage navigation, including Model Monitoring. */
    [data-testid="stSidebarNav"] {
        padding: 0.25rem 0 0.8rem !important;
    }

    [data-testid="stSidebarNav"] a {
        margin: 0.2rem 0 !important;
        padding: 0.65rem 0.75rem !important;
        border: 1px solid transparent;
        border-radius: 8px !important;
        color: #dcebea !important;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: rgba(0, 101, 114, 0.34) !important;
        border-color: rgba(0, 139, 86, 0.55);
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(90deg, #006572, rgba(0, 139, 86, 0.72)) !important;
        border-color: #78a50a;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 7px 18px rgba(0, 38, 48, 0.28);
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] * {
        color: #ffffff !important;
    }


    /* =====================================================
       DARK THEME STYLES
       ===================================================== */

    :root {
        --bg: #061d29;
        --bg-secondary: #092d38;
        --card: #0b3441;
        --card-2: #082a35;
        --border: #155665;
        --border-soft: #104754;
        --text: #f3fbf9;
        --text-secondary: #c5d9d7;
        --muted: #89aaa8;
        --blue: #003f5c;
        --blue-soft: #006572;
        --aqua: #008b56;
        --amber: #78a50a;
        --red: #ff6f91;
        --grid: #174b58;

        --font-signage: 'Saira', system-ui, sans-serif;
        --font-body: 'Inter', system-ui, sans-serif;
        --font-mono: 'IBM Plex Mono', ui-monospace, monospace;
    }

    html {
        scroll-behavior: smooth;
    }

    body,
    .stApp {
        background:
            radial-gradient(
                ellipse 90% 50% at 50% -10%,
                rgba(0, 101, 114, 0.24),
                transparent 60%
            ),
            radial-gradient(circle at 90% 20%, rgba(0, 139, 86, 0.12), transparent 30%),
            #061d29;

        color: var(--text);
        font-family: var(--font-body);
    }

    .stApp,
    .stApp p,
    .stApp span,
    .stApp div,
    .stApp label {
        font-family: var(--font-body);
    }

    .main {
        padding-top: 0 !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        animation: pageFade 0.6s ease-out;
    }

    @keyframes pageFade {
        from {
            opacity: 0;
            transform: translateY(8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Numbers everywhere read as a ledger */
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    .stDataFrame,
    code {
        font-family: var(--font-mono) !important;
        font-variant-numeric: tabular-nums;
    }

    h1 {
        color: var(--text) !important;
        font-family: var(--font-signage) !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2,
    h3 {
        color: #eef3fa !important;
        font-family: var(--font-signage) !important;
        font-weight: 650 !important;
        letter-spacing: 0.1px;
    }

    /* Streamlit subheaders → quiet signage labels above charts */
    [data-testid="stHeadingWithActionElements"] h3,
    .stApp h3 {
        font-size: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 1.4px !important;
        text-transform: uppercase !important;
        color: #8ea0bd !important;
        margin-bottom: 2px !important;
    }

    p {
        color: var(--text-secondary);
    }


    /* =====================================================
       SECTION TITLES — signage with an amber tick
       ===================================================== */

    .section-title {
        color: var(--text);
        font-family: var(--font-signage);
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.2px;

        margin-top: 38px;
        margin-bottom: 4px;
        padding-left: 14px;

        position: relative;
    }

    .section-title::before {
        content: "";
        position: absolute;
        left: 0;
        top: 3px;
        bottom: 3px;
        width: 3px;
        border-radius: 2px;
        background: var(--amber);
        box-shadow: 0 0 10px rgba(245, 165, 36, 0.45);
    }

    .subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 18px;
        padding-left: 14px;
    }


    /* =====================================================
       MASTHEAD — Control Tower instrument panel (signature)
       ===================================================== */

    .masthead {
        position: relative;
        overflow: hidden;

        background:
            radial-gradient(
                120% 140% at 100% 0%,
                rgba(0, 101, 114, 0.34),
                transparent 55%
            ),
            linear-gradient(145deg, #003f5c 0%, #075060 55%, #063c3d 100%);

        border: 1px solid #26717b;
        border-top: 2px solid var(--amber);
        border-radius: 8px;

        padding: 30px 34px 26px;
        margin-bottom: 20px;

        box-shadow:
            0 18px 48px rgba(0, 0, 0, 0.40),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);

        animation: headerEnter 0.7s ease-out;
    }

    /* faint blueprint grid — the ops-panel texture */
    .masthead-grid {
        position: absolute;
        inset: 0;
        z-index: 1;
        pointer-events: none;

        background-image:
            linear-gradient(rgba(96, 165, 250, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(96, 165, 250, 0.05) 1px, transparent 1px);
        background-size: 44px 44px;

        -webkit-mask-image:
            radial-gradient(120% 120% at 100% 0%, #000 20%, transparent 70%);
        mask-image:
            radial-gradient(120% 120% at 100% 0%, #000 20%, transparent 70%);
    }

    @keyframes headerEnter {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .live-pip {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.6);
        animation: livePulse 1.9s ease-in-out infinite;
    }

    @keyframes livePulse {
        0% {
            box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55);
        }

        70% {
            box-shadow: 0 0 0 7px rgba(52, 211, 153, 0);
        }

        100% {
            box-shadow: 0 0 0 0 rgba(52, 211, 153, 0);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .masthead,
        .block-container,
        [data-testid="stMetric"],
        [data-testid="stAlert"] {
            animation: none !important;
        }

        .live-pip {
            animation: none !important;
        }
    }


    /* =====================================================
       KPI METRICS
       ===================================================== */

    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(145deg, rgba(0, 101, 114, 0.22), rgba(0, 139, 86, 0.10)),
            #0b3441;

        border: 1px solid var(--border);
        border-left: 3px solid var(--blue);
        border-radius: 4px;

        padding: 16px 18px 15px;
        min-height: 104px;

        box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.03) inset,
            0 6px 22px rgba(0, 0, 0, 0.30);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;

        animation: cardEnter 0.5s ease-out both;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-left-color: var(--amber);

        box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.04) inset,
            0 12px 30px rgba(0, 0, 0, 0.38);
    }

    [data-testid="stMetricLabel"] {
        color: #8ea0bd !important;
        font-family: var(--font-signage) !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricLabel"] p {
        font-family: var(--font-signage) !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: #f3fbf9 !important;
        font-family: var(--font-mono) !important;
        font-size: 27px !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }

    [data-testid="stMetricDelta"] {
        font-family: var(--font-mono) !important;
        font-size: 12px !important;
    }

    @keyframes cardEnter {
        from {
            opacity: 0;
            transform: translateY(10px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    /* =====================================================
       BADGES
       ===================================================== */

    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 3px;
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        white-space: nowrap;
    }

    .badge-critical {
        background: rgba(160, 32, 32, 0.18);
        border: 1px solid #a02020;
        color: #f3a6a6;
    }

    .badge-high {
        background: rgba(217, 89, 38, 0.16);
        border: 1px solid #d95926;
        color: #f6b593;
    }

    .badge-medium {
        background: rgba(201, 133, 0, 0.16);
        border: 1px solid #c98500;
        color: #f2cf8a;
    }

    .badge-low {
        background: rgba(25, 158, 112, 0.16);
        border: 1px solid #008b56;
        color: #8fd9bd;
    }

    .badge-info {
        background: rgba(57, 135, 229, 0.14);
        border: 1px solid #2f5f9e;
        color: #9cc2f0;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 5px !important;
        animation: alertEnter 0.4s ease-out;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.14);
    }

    @keyframes alertEnter {
        from {
            opacity: 0;
            transform: translateY(5px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    /* =====================================================
       CHARTS / TABLES
       ===================================================== */

    [data-testid="stVegaLiteChart"] {
        background: #0b3441;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 12px 8px;

        box-shadow:
            0 4px 18px rgba(0, 0, 0, 0.22);

        max-width: 100% !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;

        box-shadow:
            0 4px 18px rgba(0, 0, 0, 0.18);

        max-width: 100% !important;
        box-sizing: border-box !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        background: #003f5c;
        color: #f3fbf9;
        border: 1px solid #006572;
        border-radius: 4px;
        font-family: var(--font-signage);
        font-weight: 600;
        letter-spacing: 0.4px;

        transition:
            background 0.2s ease,
            border-color 0.2s ease,
            transform 0.2s ease;
    }

    .stButton > button:hover {
        background: #006572;
        border-color: var(--amber);
        color: #ffffff;
        transform: translateY(-1px);
    }


    /* =====================================================
       INPUTS
       ===================================================== */

    div[data-baseweb="select"] > div {
        background-color: #082a35;
        border-color: #006572;
        border-radius: 4px;
    }

    div[data-baseweb="select"] span {
        color: #dbe4f0;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #78a50a;
    }

    /* High-contrast selected filters. BaseWeb otherwise uses a yellow fill. */
    [data-baseweb="tag"] {
        background: #006572 !important;
        border: 1px solid #78a50a !important;
        border-radius: 5px !important;
        font-family: var(--font-mono) !important;
        font-size: 11px !important;
        box-shadow: 0 2px 6px rgba(0, 25, 32, 0.28);
    }

    [data-baseweb="tag"],
    [data-baseweb="tag"] *,
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    [data-baseweb="tag"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    [role="option"][aria-selected="true"] {
        background: #006572 !important;
        color: #ffffff !important;
    }

    [role="option"][aria-selected="true"] * {
        color: #ffffff !important;
    }

    hr {
        border-color: #1a2740 !important;
        opacity: 0.8;
    }

    [data-testid="stCaptionContainer"] {
        color: #6b7a93 !important;
        font-family: var(--font-mono) !important;
        font-size: 11.5px;
    }


    /* =====================================================
       SUMMARY CARD
       ===================================================== */

    .summary-card {
        position: relative;
        overflow: hidden;

        background: linear-gradient(
            180deg,
            #0b3441,
            #082a35
        );

        border: 1px solid var(--border);
        border-radius: 6px;

        padding: 20px 22px 20px 26px;
        margin-top: 10px;

        box-shadow:
            0 4px 18px rgba(0, 0, 0, 0.20);
    }

    .summary-card::before {
        content: "";
        position: absolute;

        left: 0;
        top: 0;
        bottom: 0;

        width: 3px;

        background: var(--amber);
    }

    .summary-title {
        color: #f3fbf9;
        font-family: var(--font-signage);
        font-size: 13px;
        font-weight: 650;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .summary-text {
        color: #c5d9d7;
        font-size: 13.5px;
        line-height: 1.7;
    }


    /* =====================================================
       ACTION CARDS
       ===================================================== */

    .action-card {
        background: linear-gradient(
            180deg,
            #0b3441,
            #082a35
        );

        border: 1px solid var(--border);
        border-radius: 6px;

        padding: 16px;
        margin-bottom: 10px;
    }

    .action-title {
        color: #f3fbf9;
        font-family: var(--font-signage);
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 6px;
    }

    .action-value {
        color: #c5d9d7;
        font-family: var(--font-mono);
        font-size: 12px;
        line-height: 1.6;
    }

    /* A marker placed by danger_metric() turns only the relevant KPI card red. */
    [data-testid="stColumn"]:has(.danger-kpi-marker) [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(175, 38, 63, 0.32), rgba(92, 19, 36, 0.30)), #301722;
        border-color: #9f354c;
        border-left-color: #ff5c72;
        box-shadow: 0 8px 24px rgba(120, 18, 43, 0.24);
    }

    [data-testid="stColumn"]:has(.danger-kpi-marker) [data-testid="stMetricValue"] {
        color: #ffb5c0 !important;
    }

    [data-testid="stColumn"]:has(.danger-kpi-marker) [data-testid="stMetricLabel"] p {
        color: #ff8fa2 !important;
    }

    .danger-kpi-marker {
        display: none;
    }

    /* Responsive Streamlit columns and legible charts/tables. */
    [data-testid="stVegaLiteChart"] canvas,
    [data-testid="stVegaLiteChart"] svg {
        max-width: 100% !important;
    }

    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }

    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.8rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 min(100%, 320px) !important;
            width: 100% !important;
            min-width: 0 !important;
        }

        .masthead {
            padding: 24px 20px 22px;
        }

        .masthead [style*="font-size:40px"] {
            font-size: clamp(28px, 8vw, 40px) !important;
        }
    }

    @media (max-width: 600px) {
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
        }

        .section-title {
            font-size: 19px;
            margin-top: 28px;
        }

        [data-testid="stMetric"] {
            min-height: 92px;
        }

        [data-testid="stVegaLiteChart"] {
            padding: 10px 6px 6px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


def risk_badge(risk):
    mapping = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low"
    }

    css_class = mapping.get(
        str(risk),
        "badge-info"
    )

    return f'<span class="badge {css_class}">{risk}</span>'


def priority_badge(priority):
    if pd.isna(priority):
        return ""

    priority = str(priority)

    if "Critical" in priority:
        css_class = "badge-critical"
    elif "High" in priority:
        css_class = "badge-high"
    elif "Medium" in priority:
        css_class = "badge-medium"
    elif "Low" in priority:
        css_class = "badge-low"
    else:
        css_class = "badge-info"

    return f'<span class="badge {css_class}">{priority}</span>'


def safe_mean(series):
    return series.mean() if len(series) > 0 else 0


def money(value):
    return f"${value:,.0f}"


def danger_metric(label, value, **kwargs):
    """Render a metric card with a red treatment for urgent or adverse KPIs."""
    st.markdown(
        '<span class="danger-kpi-marker" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )
    st.metric(label, value, **kwargs)


def pct(value):
    return f"{value:+.2%}"


def safe_numeric(df, column, default=0):
    if column in df.columns:
        return pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(default)

    return pd.Series(
        default,
        index=df.index,
        dtype=float
    )


# =========================================================
# CHART ENGINE — validated palette, themed Altair
#   Palette validated (dataviz skill) on surface #101a2e:
#   categorical / diverging / sequential all PASS.
# =========================================================

CHART_SURFACE = "#0b3441"
INK_PRIMARY = "#f3fbf9"
INK_SECONDARY = "#c5d9d7"
INK_MUTED = "#89aaa8"
GRID = "#174b58"
AXIS = "#26717b"
MONO = "IBM Plex Mono, ui-monospace, monospace"
SIGNAGE = "Saira, system-ui, sans-serif"

BLUE = "#003f5c"
AQUA = "#008b56"
AMBER = "#78a50a"
RED = "#ff6f91"

# Severity is ORDERED magnitude → one warm ordinal ramp (Low→Critical),
# not a 4-hue traffic light (that fails the normal-vision ΔE floor).
SEVERITY_COLORS = {
    "Critical": "#d94a70",
    "High": "#ff6f91",
    "Medium": "#ffb86b",
    "Low": "#008b56",
}
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]

# Executive action priority (P1 most urgent → P3)
PRIORITY_COLORS = {
    "P1 - Immediate": "#d94a70",
    "P2 - High": "#ffb86b",
    "P3 - Planned": "#006572",
}

BLUE_SEQ = ["#003f5c", "#006572", "#008b56", "#78a50a"]


def _base(chart_df, height):
    """Shared config: dark surface, hairline grid, mono ticks."""
    return (
        alt.Chart(chart_df, height=height)
        .configure_view(
            fill=CHART_SURFACE,
            stroke=None,
        )
        .configure_axis(
            labelFont=MONO,
            labelColor=INK_SECONDARY,
            labelFontSize=11,
            titleFont=SIGNAGE,
            titleColor=INK_MUTED,
            titleFontSize=11,
            titleFontWeight=600,
            gridColor=GRID,
            gridWidth=1,
            domainColor=AXIS,
            tickColor=AXIS,
        )
        .configure_axisX(labelFontSize=11)
        .configure_legend(
            labelFont=MONO,
            labelColor=INK_SECONDARY,
            titleFont=SIGNAGE,
            titleColor=INK_MUTED,
            symbolType="square",
        )
    )


def _severity_scale(field):
    domain = [s for s in SEVERITY_ORDER if s in field.unique().tolist()] \
        if hasattr(field, "unique") else SEVERITY_ORDER
    return alt.Scale(
        domain=domain,
        range=[SEVERITY_COLORS[d] for d in domain],
    )


def hbar(
    df,
    cat,
    val,
    color_mode="blue",
    cat_title="",
    val_title="",
    money_fmt=False,
    pct_fmt=False,
    height=None,
    sort_desc=True,
    color_field=None,
    color_domain=None,
    color_range=None,
):
    """
    Horizontal bar with mono value labels + hover tooltip.
    color_mode: 'blue' | 'severity' | 'diverging' | 'sequential' | 'amber' | 'custom'
    """
    df = df.copy()
    df[cat] = df[cat].astype(str)
    df[val] = pd.to_numeric(df[val], errors="coerce").fillna(0)

    n = len(df)
    if height is None:
        height = max(150, min(460, 34 * n + 46))

    order = "-x" if sort_desc else "x"

    y = alt.Y(
        f"{cat}:N",
        sort=alt.EncodingSortField(field=val, order="descending" if sort_desc else "ascending"),
        title=cat_title or None,
        axis=alt.Axis(labelLimit=280, labelPadding=8),
    )
    x = alt.X(
        f"{val}:Q",
        title=val_title or None,
        axis=alt.Axis(grid=True, labelFlush=False),
    )

    # ---- color encoding by mode ----
    if color_mode == "severity":
        color = alt.Color(
            f"{cat}:N",
            scale=_severity_scale(df[cat]),
            legend=None,
        )
    elif color_mode == "custom" and color_field:
        color = alt.Color(
            f"{color_field}:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=None,
        )
    elif color_mode == "diverging":
        df["_pos"] = df[val] >= 0
        color = alt.Color(
            "_pos:N",
            scale=alt.Scale(domain=[True, False], range=[BLUE, RED]),
            legend=None,
        )
    elif color_mode == "sequential":
        color = alt.Color(
            f"{val}:Q",
            scale=alt.Scale(range=BLUE_SEQ),
            legend=None,
        )
    elif color_mode == "amber":
        color = alt.value(AMBER)
    else:
        color = alt.value(BLUE)

    tip_fmt = "$,.0f" if money_fmt else (".2%" if pct_fmt else ",.0f")
    tooltip = [
        alt.Tooltip(f"{cat}:N", title=cat_title or cat.replace("_", " ").title()),
        alt.Tooltip(f"{val}:Q", title=val_title or "Value", format=tip_fmt),
    ]

    bars = _mark_bar_base(df, height).mark_bar(
        cornerRadiusEnd=3,
        height={"band": 0.66},
    ).encode(x=x, y=y, color=color, tooltip=tooltip)

    # ---- adaptive direct labels: inside long bars, outside short bars ----
    lbl_expr = _label_expr(val, money_fmt, pct_fmt)
    max_abs = float(df[val].abs().max()) if not df.empty else 0.0
    inside_threshold = max_abs * 0.18
    long_bar = f"abs(datum['{val}']) >= {inside_threshold}"
    short_bar = f"abs(datum['{val}']) < {inside_threshold}"

    label_base = alt.Chart(df).encode(
        x=x,
        y=y,
        text=alt.Text("_lbl:N"),
    ).transform_calculate(_lbl=lbl_expr)

    labels_inside_positive = label_base.mark_text(
        align="right",
        baseline="middle",
        dx=-7,
        font=MONO,
        fontSize=11,
        color=INK_PRIMARY,
    ).transform_filter(f"{long_bar} && datum['{val}'] >= 0")

    labels_inside_negative = label_base.mark_text(
        align="left",
        baseline="middle",
        dx=7,
        font=MONO,
        fontSize=11,
        color=INK_PRIMARY,
    ).transform_filter(f"{long_bar} && datum['{val}'] < 0")

    labels_outside_positive = label_base.mark_text(
        align="left",
        baseline="middle",
        dx=7,
        font=MONO,
        fontSize=11,
        color=INK_SECONDARY,
    ).transform_filter(f"{short_bar} && datum['{val}'] >= 0")

    labels_outside_negative = label_base.mark_text(
        align="right",
        baseline="middle",
        dx=-7,
        font=MONO,
        fontSize=11,
        color=INK_SECONDARY,
    ).transform_filter(f"{short_bar} && datum['{val}'] < 0")

    layered = (
        bars
        + labels_inside_positive
        + labels_inside_negative
        + labels_outside_positive
        + labels_outside_negative
    ).properties(
        height=height,
        padding={"left": 8, "right": 58, "top": 8, "bottom": 8},
    )
    return _apply_config(layered)


def vbar(
    df,
    cat,
    val,
    color_mode="blue",
    cat_title="",
    val_title="",
    money_fmt=False,
    pct_fmt=False,
    height=300,
    color_field=None,
    color_domain=None,
    color_range=None,
    cat_sort=None,
):
    """Vertical bar (few categories, e.g. Baseline vs Optimized) with labels."""
    df = df.copy()
    df[cat] = df[cat].astype(str)
    df[val] = pd.to_numeric(df[val], errors="coerce").fillna(0)

    x = alt.X(
        f"{cat}:N",
        title=cat_title or None,
        sort=cat_sort,
        axis=alt.Axis(labelAngle=0, labelLimit=200),
    )
    y = alt.Y(f"{val}:Q", title=val_title or None, axis=alt.Axis(grid=True))

    if color_mode == "severity":
        color = alt.Color(f"{cat}:N", scale=_severity_scale(df[cat]), legend=None)
    elif color_mode == "custom" and color_field:
        color = alt.Color(
            f"{color_field}:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=None,
        )
    elif color_mode == "diverging":
        df["_pos"] = df[val] >= 0
        color = alt.Color(
            "_pos:N",
            scale=alt.Scale(domain=[True, False], range=[BLUE, RED]),
            legend=None,
        )
    elif color_mode == "amber":
        color = alt.value(AMBER)
    else:
        color = alt.value(BLUE)

    tip_fmt = "$,.0f" if money_fmt else (".2%" if pct_fmt else ",.0f")
    tooltip = [
        alt.Tooltip(f"{cat}:N", title=cat_title or cat.replace("_", " ").title()),
        alt.Tooltip(f"{val}:Q", title=val_title or "Value", format=tip_fmt),
    ]

    bars = alt.Chart(df).mark_bar(
        cornerRadiusEnd=3,
        width={"band": 0.6},
    ).encode(x=x, y=y, color=color, tooltip=tooltip)

    lbl_expr = _label_expr(val, money_fmt, pct_fmt)
    labels = alt.Chart(df).mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        font=MONO,
        fontSize=11,
        color=INK_SECONDARY,
    ).encode(x=x, y=y, text=alt.Text("_lbl:N")).transform_calculate(_lbl=lbl_expr)

    layered = (bars + labels).properties(
        height=height,
        padding={"left": 8, "right": 8, "top": 28, "bottom": 8},
    )
    return _apply_config(layered)


def _label_expr(val, money_fmt, pct_fmt):
    if money_fmt:
        return (
            f"abs(datum['{val}']) >= 1000000 "
            f"? (datum['{val}'] < 0 ? '-$' : '$') + format(abs(datum['{val}'])/1000000, '.1f') + 'M' "
            f": (abs(datum['{val}']) >= 1000 "
            f"? (datum['{val}'] < 0 ? '-$' : '$') + format(abs(datum['{val}'])/1000, '.0f') + 'K' "
            f": (datum['{val}'] < 0 ? '-$' : '$') + format(abs(datum['{val}']), ',.0f'))"
        )
    if pct_fmt:
        return f"format(datum['{val}'], '.1%')"
    return (
        f"abs(datum['{val}']) >= 1000000 ? format(datum['{val}']/1000000, ',.1f') + 'M' "
        f": (abs(datum['{val}']) >= 10000 ? format(datum['{val}']/1000, ',.0f') + 'K' "
        f": format(datum['{val}'], ',.0f'))"
    )


def _mark_bar_base(df, height):
    return alt.Chart(df, height=height)


def _apply_config(chart):
    """Apply shared theme config to a (possibly layered) chart."""
    return (
        chart
        .configure_view(fill=CHART_SURFACE, stroke=None)
        .configure_axis(
            labelFont=MONO,
            labelColor=INK_SECONDARY,
            labelFontSize=11,
            titleFont=SIGNAGE,
            titleColor=INK_MUTED,
            titleFontSize=11,
            titleFontWeight=600,
            gridColor=GRID,
            gridWidth=1,
            domainColor=AXIS,
            tickColor=AXIS,
        )
        .configure_legend(
            labelFont=MONO,
            labelColor=INK_SECONDARY,
            titleFont=SIGNAGE,
            titleColor=INK_MUTED,
            symbolType="square",
        )
    )


def grouped_bar(
    df,
    cat,
    series_col,
    val,
    series_domain,
    series_range,
    val_title="",
    money_fmt=False,
    pct_fmt=False,
    height=300,
):
    """Grouped vertical bars for 2-series compare (e.g. Baseline vs Optimized)."""
    df = df.copy()
    df[val] = pd.to_numeric(df[val], errors="coerce").fillna(0)

    tip_fmt = "$,.0f" if money_fmt else (".2%" if pct_fmt else ",.0f")
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f"{series_col}:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(f"{val}:Q", title=val_title or None, axis=alt.Axis(grid=True)),
        color=alt.Color(
            f"{series_col}:N",
            scale=alt.Scale(domain=series_domain, range=series_range),
            legend=alt.Legend(title=None, orient="top"),
        ),
        column=alt.Column(f"{cat}:N", title=None, header=alt.Header(
            labelFont=MONO, labelColor=INK_SECONDARY, labelFontSize=11)),
        tooltip=[
            alt.Tooltip(f"{cat}:N"),
            alt.Tooltip(f"{series_col}:N", title="Policy"),
            alt.Tooltip(f"{val}:Q", title=val_title or "Value", format=tip_fmt),
        ],
    ).properties(height=height)
    return _apply_config(chart)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    optimization = pd.read_csv(
        OPTIMIZATION_FILE
    )

    decisions = pd.read_csv(
        DECISIONS_FILE
    )

    optimization_summary = pd.read_csv(
        OPTIMIZATION_SUMMARY_FILE
    )

    scenario_results = pd.read_csv(
        SCENARIO_RESULTS_FILE
    )

    scenario_summary = pd.read_csv(
        SCENARIO_SUMMARY_FILE
    )

    executive_action_plan = pd.read_csv(
        EXECUTIVE_ACTION_PLAN_FILE
    )

    executive_action_summary = pd.read_csv(
        EXECUTIVE_ACTION_SUMMARY_FILE
    )

    executive_priorities = pd.read_csv(
        EXECUTIVE_PRIORITIES_FILE
    )

    executive_management_brief = pd.read_csv(
        EXECUTIVE_MANAGEMENT_BRIEF_FILE
    )

    return (
        optimization,
        decisions,
        optimization_summary,
        scenario_results,
        scenario_summary,
        executive_action_plan,
        executive_action_summary,
        executive_priorities,
        executive_management_brief,
    )


# =========================================================
# FILE VALIDATION
# =========================================================

required_files = [
    OPTIMIZATION_FILE,
    DECISIONS_FILE,
    OPTIMIZATION_SUMMARY_FILE,
    SCENARIO_RESULTS_FILE,
    SCENARIO_SUMMARY_FILE,
    EXECUTIVE_ACTION_PLAN_FILE,
    EXECUTIVE_ACTION_SUMMARY_FILE,
    EXECUTIVE_PRIORITIES_FILE,
    EXECUTIVE_MANAGEMENT_BRIEF_FILE,
]

missing_files = [
    f for f in required_files
    if not f.exists()
]

if missing_files:

    st.error(
        "Required dashboard data files are missing."
    )

    for file in missing_files:
        st.code(str(file))

    st.stop()


(
    optimization,
    decisions,
    optimization_summary,
    scenario_results,
    scenario_summary,
    executive_action_plan,
    executive_action_summary,
    executive_priorities,
    executive_management_brief,
) = load_data()


# =========================================================
# DATA PREPARATION
# =========================================================

numeric_columns = [
    "baseline_service_level",
    "baseline_order_coverage_days",
    "baseline_safety_stock",
    "baseline_reorder_point",
    "baseline_order_quantity",
    "baseline_fill_rate",
    "baseline_stockout_rate",
    "baseline_lost_sales",
    "baseline_inventory_days",
    "baseline_total_cost",
    "optimized_service_level",
    "optimized_safety_stock",
    "optimized_reorder_point",
    "optimized_order_quantity",
    "optimized_order_coverage_days",
    "optimized_fill_rate",
    "optimized_stockout_rate",
    "optimized_lost_sales",
    "optimized_inventory_days",
    "optimized_total_cost",
    "optimized_purchase_orders",
    "optimized_purchase_quantity",
    "optimized_supplier_delay_rate",
    "optimized_average_supplier_delay",
    "lost_sales_reduction",
    "cost_change",
    "cost_savings",
    "inventory_days_change",
    "fill_rate_change",
    "safety_stock_change",
]

for column in numeric_columns:

    if column in optimization.columns:

        optimization[column] = pd.to_numeric(
            optimization[column],
            errors="coerce"
        )


if "cost_savings" not in optimization.columns:

    optimization["cost_savings"] = (
        optimization["baseline_total_cost"]
        - optimization["optimized_total_cost"]
    )


if "lost_sales_reduction" not in optimization.columns:

    optimization["lost_sales_reduction"] = (
        optimization["baseline_lost_sales"]
        - optimization["optimized_lost_sales"]
    )


if "inventory_days_change" not in optimization.columns:

    optimization["inventory_days_change"] = (
        optimization["optimized_inventory_days"]
        - optimization["baseline_inventory_days"]
    )


if "fill_rate_change" not in optimization.columns:

    optimization["fill_rate_change"] = (
        optimization["optimized_fill_rate"]
        - optimization["baseline_fill_rate"]
    )


# =========================================================
# MERGE DECISION INTELLIGENCE
# =========================================================

decision_fields = [
    "product_id",
    "service_level_improvement",
    "lost_sales_reduction_pct",
    "inventory_days_change_pct",
    "order_quantity_change",
    "order_quantity_change_pct",
    "safety_stock_change_pct",
    "decision_impact_score",
    "priority",
    "management_decision",
    "business_impact",
    "policy_change_flag",
    "priority_rank",
]

available_decision_fields = [
    c for c in decision_fields
    if c in decisions.columns
]

if (
    "product_id" in available_decision_fields
    and "product_id" in optimization.columns
):

    decision_subset = decisions[
        available_decision_fields
    ].copy()

    duplicate_columns = [
        c
        for c in decision_subset.columns
        if c != "product_id"
        and c in optimization.columns
    ]

    decision_subset = decision_subset.drop(
        columns=duplicate_columns,
        errors="ignore"
    )

    optimization = optimization.merge(
        decision_subset,
        on="product_id",
        how="left"
    )


# =========================================================
# MERGE EXECUTIVE ACTION INTELLIGENCE
# =========================================================

if (
    "product_id" in executive_action_plan.columns
    and "product_id" in optimization.columns
):

    executive_fields = [
        "product_id",
        "scenario",
        "contingency_severity",
        "action_priority",
        "executive_action_score",
        "action_type",
        "action_owner",
        "action_horizon",
        "additional_safety_stock_required",
        "lost_sales_exposure",
        "contingency_cost_exposure",
        "executive_action",
        "executive_confidence",
    ]

    available_executive_fields = [
        c
        for c in executive_fields
        if c in executive_action_plan.columns
    ]

    executive_subset = executive_action_plan[
        available_executive_fields
    ].copy()

    duplicate_columns = [
        c
        for c in executive_subset.columns
        if c != "product_id"
        and c in optimization.columns
    ]

    executive_subset = executive_subset.drop(
        columns=duplicate_columns,
        errors="ignore"
    )

    optimization = optimization.merge(
        executive_subset,
        on="product_id",
        how="left"
    )


# =========================================================
# DASHBOARD HEADER
# =========================================================

# ---- Live masthead readouts (portfolio-wide, pre-filter) ----

hero_products = int(optimization["product_id"].nunique())

hero_p1 = 0

if "action_priority" in executive_action_plan.columns:

    hero_p1 = int(
        executive_action_plan["action_priority"]
        .astype(str)
        .str.contains("P1", na=False)
        .sum()
    )


def _brief_value(name):

    try:

        rows = executive_management_brief[
            executive_management_brief["metric"]
            .astype(str)
            .str.strip()
            .str.lower()
            == name.lower()
        ]

        if not rows.empty:

            return float(
                pd.to_numeric(
                    rows["value"].iloc[0],
                    errors="coerce"
                )
            )

    except Exception:

        pass

    return None


def _compact_money(value):

    if value is None or pd.isna(value):
        return "--"

    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"{sign}${value / 1_000:.0f}K"

    return f"{sign}${value:,.0f}"


hero_exposure = _brief_value("Total contingency cost exposure")

if hero_exposure is None:

    hero_exposure = _brief_value("Potential lost-sales exposure")


hero_exposure_display = _compact_money(hero_exposure)


def _readout(label, value, accent="#f3fbf9"):

    return (
        '<div style="display:flex;flex-direction:column;gap:3px;padding:0 22px;">'
        f'<span style="font-family:var(--font-signage);font-size:10px;font-weight:600;'
        f'letter-spacing:1.6px;text-transform:uppercase;color:#6b7a93;">{label}</span>'
        f'<span style="font-family:var(--font-mono);font-size:19px;font-weight:600;'
        f'letter-spacing:-0.3px;color:{accent};">{value}</span>'
        '</div>'
    )


ledger_strip = (
    '<div style="display:flex;flex-wrap:wrap;align-items:center;'
    'margin-top:20px;padding-top:18px;border-top:1px solid rgba(245,165,36,0.30);">'
    + _readout("Portfolio", f"{hero_products:,} SKU")
    + '<div style="width:1px;height:34px;background:#1e2b40;"></div>'
    + _readout(
        "P1 Immediate",
        f"{hero_p1:02d}",
        accent="#e66767" if hero_p1 > 0 else "#8fd9bd"
    )
    + '<div style="width:1px;height:34px;background:#1e2b40;"></div>'
    + _readout("Modeled Exposure", hero_exposure_display, accent="#78a50a")
    + '<div style="width:1px;height:34px;background:#1e2b40;"></div>'
    + '<div style="display:flex;flex-direction:column;gap:3px;padding:0 22px;">'
      '<span style="font-family:var(--font-signage);font-size:10px;font-weight:600;'
      'letter-spacing:1.6px;text-transform:uppercase;color:#6b7a93;">Engine</span>'
      '<span style="display:inline-flex;align-items:center;gap:7px;font-family:var(--font-mono);'
      'font-size:14px;font-weight:600;color:#8fd9bd;">'
      '<span class="live-pip"></span>ONLINE</span>'
    '</div>'
    '</div>'
)


st.markdown(
    '<div class="masthead">'
    '<div class="masthead-grid"></div>'
    '<div style="position:relative;z-index:2;">'
    '<div style="font-family:var(--font-mono);font-size:11px;font-weight:500;'
    'letter-spacing:3px;text-transform:uppercase;color:#78a50a;margin-bottom:12px;">'
    '&#9698; Supply-Chain Control Tower &nbsp;&middot;&nbsp; Sector OPS-01</div>'
    '<div style="font-family:var(--font-signage);color:#f3fbf9;font-size:40px;'
    'font-weight:800;letter-spacing:-1px;line-height:1.02;">Supply Chain Intelligence</div>'
    '<div style="color:#c5d9d7;font-size:14px;margin-top:8px;max-width:640px;">'
    'Executive inventory optimization &amp; decision support &mdash; baseline versus '
    'optimized policy, scenario stress, and contingency exposure across the portfolio.</div>'
    + ledger_strip +
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="padding:0.15rem 0.2rem 0.4rem;">
        <div style="display:flex;align-items:center;gap:0.65rem;">
            <span style="display:grid;place-items:center;width:34px;height:34px;border-radius:9px;background:#006572;border:1px solid #78a50a;color:#fff;font-size:17px;">&#9881;</span>
            <div>
                <div style="font-family:var(--font-signage);font-size:1.1rem;font-weight:700;color:#f3fbf9;line-height:1.1;">Dashboard Controls</div>
                <div style="font-family:var(--font-mono);font-size:0.62rem;letter-spacing:1.35px;color:#9fc3bf;margin-top:4px;">PORTFOLIO FILTERS</div>
            </div>
        </div>
        <p style="font-size:0.78rem;line-height:1.55;margin:0.8rem 0 0;color:#c5d9d7;">
            Narrow inventory risks, priorities, and management ownership.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

risk_classes = sorted(
    optimization["risk_class"]
    .dropna()
    .unique()
)

selected_risk_classes = st.sidebar.multiselect(
    "Risk Class",
    options=risk_classes,
    default=risk_classes
)


if "priority" in optimization.columns:

    priorities = sorted(
        optimization["priority"]
        .dropna()
        .unique()
    )

    selected_priorities = st.sidebar.multiselect(
        "Priority",
        options=priorities,
        default=priorities
    )

else:

    selected_priorities = []


drivers = sorted(
    optimization["primary_risk_driver"]
    .dropna()
    .unique()
)

selected_drivers = st.sidebar.multiselect(
    "Primary Risk Driver",
    options=drivers,
    default=drivers
)


if "category" in optimization.columns:

    categories = sorted(
        optimization["category"]
        .dropna()
        .unique()
    )

    selected_categories = st.sidebar.multiselect(
        "Category",
        options=categories,
        default=categories
    )

else:

    selected_categories = []


st.sidebar.divider()

st.sidebar.subheader("Executive Controls")


if "action_priority" in optimization.columns:

    executive_priorities_list = sorted(
        optimization["action_priority"]
        .dropna()
        .unique()
    )

    selected_action_priorities = st.sidebar.multiselect(
        "Executive Priority",
        options=executive_priorities_list,
        default=executive_priorities_list
    )

else:

    selected_action_priorities = []


if "action_owner" in optimization.columns:

    action_owners = sorted(
        optimization["action_owner"]
        .dropna()
        .unique()
    )

    selected_action_owners = st.sidebar.multiselect(
        "Action Owner",
        options=action_owners,
        default=action_owners
    )

else:

    selected_action_owners = []


if "contingency_severity" in optimization.columns:

    contingency_severities = sorted(
        optimization["contingency_severity"]
        .dropna()
        .unique()
    )

    selected_contingency_severities = st.sidebar.multiselect(
        "Contingency Severity",
        options=contingency_severities,
        default=contingency_severities
    )

else:

    selected_contingency_severities = []


st.sidebar.divider()

st.sidebar.caption(
    f"Portfolio: "
    f"{optimization['product_id'].nunique():,} products"
)


# =========================================================
# FILTER DATA
# =========================================================

filtered = optimization[
    optimization["risk_class"].isin(
        selected_risk_classes
    )
    &
    optimization["primary_risk_driver"].isin(
        selected_drivers
    )
].copy()


if "priority" in filtered.columns:

    filtered = filtered[
        filtered["priority"].isin(
            selected_priorities
        )
    ]


if "category" in filtered.columns:

    filtered = filtered[
        filtered["category"].isin(
            selected_categories
        )
    ]


if "action_priority" in filtered.columns:

    filtered = filtered[
        filtered["action_priority"].isin(
            selected_action_priorities
        )
    ]


if "action_owner" in filtered.columns:

    filtered = filtered[
        filtered["action_owner"].isin(
            selected_action_owners
        )
    ]


if "contingency_severity" in filtered.columns:

    filtered = filtered[
        filtered["contingency_severity"].isin(
            selected_contingency_severities
        )
    ]


if filtered.empty:

    st.warning(
        "No products match the selected filters."
    )

    st.stop()


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

section(
    "Executive Overview",
    "Current optimization impact across the selected product portfolio."
)

products_analyzed = filtered["product_id"].nunique()

baseline_cost = filtered[
    "baseline_total_cost"
].sum()

optimized_cost = filtered[
    "optimized_total_cost"
].sum()

cost_savings = (
    baseline_cost
    - optimized_cost
)

baseline_lost_sales = filtered[
    "baseline_lost_sales"
].sum()

optimized_lost_sales = filtered[
    "optimized_lost_sales"
].sum()

lost_sales_reduction = (
    baseline_lost_sales
    - optimized_lost_sales
)

baseline_fill_rate = safe_mean(
    filtered["baseline_fill_rate"]
)

optimized_fill_rate = safe_mean(
    filtered["optimized_fill_rate"]
)

fill_rate_improvement = (
    optimized_fill_rate
    - baseline_fill_rate
)

avg_inventory_days_change = safe_mean(
    filtered["inventory_days_change"]
)

critical_count = filtered[
    "risk_class"
].eq("Critical").sum()

high_risk_count = filtered[
    "risk_class"
].eq("High").sum()


policy_change_count = 0

if "policy_change_flag" in filtered.columns:

    policy_change_count = (
        filtered["policy_change_flag"]
        .astype(str)
        .str.lower()
        .isin([
            "true",
            "yes",
            "1",
            "1.0"
        ])
        .sum()
    )


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Products",
        f"{products_analyzed:,}"
    )

with col2:
    st.metric(
        "Cost Savings",
        money(cost_savings)
    )

with col3:
    st.metric(
        "Lost Sales Reduction",
        f"{lost_sales_reduction:,.0f}"
    )

with col4:
    st.metric(
        "Fill Rate Improvement",
        f"{fill_rate_improvement:+.2%}"
    )

with col5:
    st.metric(
        "Inventory Days",
        f"{avg_inventory_days_change:+.2f}"
    )


# =========================================================
# PORTFOLIO HEALTH
# =========================================================

section(
    "Portfolio Health",
    "A high-level view of inventory risk and policy intervention requirements."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    danger_metric(
        "Critical Exposure",
        f"{critical_count:,}",
        help="Products classified as Critical risk."
    )

with col2:
    danger_metric(
        "High-Risk Products",
        f"{high_risk_count:,}",
        help="Products classified as High risk."
    )

with col3:
    st.metric(
        "Policy Changes",
        f"{policy_change_count:,}",
        help="Products flagged for a policy change."
    )

with col4:

    savings_rate = (
        cost_savings / baseline_cost
        if baseline_cost != 0
        else 0
    )

    st.metric(
        "Portfolio Cost Change",
        f"{savings_rate:+.2%}"
    )


# =========================================================
# EXECUTIVE ACTION STATUS
# =========================================================

if "action_priority" in filtered.columns:

    section(
        "Executive Action Status",
        "Management interventions generated by the executive action planning engine."
    )

    p1_count = filtered[
        "action_priority"
    ].eq("P1 - Immediate").sum()

    p2_count = filtered[
        "action_priority"
    ].eq("P2 - High").sum()

    critical_exposure_count = 0

    if "contingency_severity" in filtered.columns:

        critical_exposure_count = filtered[
            "contingency_severity"
        ].eq("Critical").sum()

    additional_safety_stock = 0

    if "additional_safety_stock_required" in filtered.columns:

        additional_safety_stock = pd.to_numeric(
            filtered[
                "additional_safety_stock_required"
            ],
            errors="coerce"
        ).fillna(0).sum()

    lost_sales_exposure = 0

    if "lost_sales_exposure" in filtered.columns:

        lost_sales_exposure = pd.to_numeric(
            filtered[
                "lost_sales_exposure"
            ],
            errors="coerce"
        ).fillna(0).sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        danger_metric(
            "Immediate Actions",
            f"{p1_count:,}"
        )

    with col2:
        danger_metric(
            "High-Priority Actions",
            f"{p2_count:,}"
        )

    with col3:
        danger_metric(
            "Critical Exposures",
            f"{critical_exposure_count:,}"
        )

    with col4:
        st.metric(
            "Additional Safety Stock",
            f"{additional_safety_stock:,.0f}"
        )

    if lost_sales_exposure > 0:

        st.warning(
            f"Current filtered portfolio carries approximately "
            f"{lost_sales_exposure:,.0f} units of modeled "
            f"lost-sales exposure."
        )


# =========================================================
# EXECUTIVE ACTION CENTER
# =========================================================

section(
    "Executive Action Center",
    "Management actions generated by the executive action-plan engine across optimization, scenario stress, and contingency exposure."
)

action_plan = executive_action_plan.copy()


def numeric_column(df, column, default=0):

    if column in df.columns:

        return pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(default)

    return pd.Series(
        default,
        index=df.index,
        dtype=float
    )


if "additional_safety_stock_required" in action_plan.columns:

    action_plan[
        "additional_safety_stock_required"
    ] = numeric_column(
        action_plan,
        "additional_safety_stock_required"
    )


if "lost_sales_exposure" in action_plan.columns:

    action_plan[
        "lost_sales_exposure"
    ] = numeric_column(
        action_plan,
        "lost_sales_exposure"
    )


if "contingency_cost_exposure" in action_plan.columns:

    action_plan[
        "contingency_cost_exposure"
    ] = numeric_column(
        action_plan,
        "contingency_cost_exposure"
    )


if "executive_action_score" in action_plan.columns:

    action_plan[
        "executive_action_score"
    ] = numeric_column(
        action_plan,
        "executive_action_score"
    )


p1_count = 0
p2_count = 0
critical_exposure_count = 0
high_exposure_count = 0
additional_safety_stock = 0
lost_sales_exposure = 0
contingency_cost_exposure = 0


if "action_priority" in action_plan.columns:

    p1_count = (
        action_plan["action_priority"]
        .astype(str)
        .str.contains(
            "P1",
            na=False
        )
        .sum()
    )

    p2_count = (
        action_plan["action_priority"]
        .astype(str)
        .str.contains(
            "P2",
            na=False
        )
        .sum()
    )


if "contingency_severity" in action_plan.columns:

    critical_exposure_count = (
        action_plan[
            "contingency_severity"
        ]
        .astype(str)
        .str.lower()
        .eq("critical")
        .sum()
    )

    high_exposure_count = (
        action_plan[
            "contingency_severity"
        ]
        .astype(str)
        .str.lower()
        .eq("high")
        .sum()
    )


if "additional_safety_stock_required" in action_plan.columns:

    additional_safety_stock = action_plan[
        "additional_safety_stock_required"
    ].sum()


if "lost_sales_exposure" in action_plan.columns:

    lost_sales_exposure = action_plan[
        "lost_sales_exposure"
    ].sum()


if "contingency_cost_exposure" in action_plan.columns:

    contingency_cost_exposure = action_plan[
        "contingency_cost_exposure"
    ].sum()


col1, col2, col3, col4 = st.columns(4)

with col1:
    danger_metric(
        "P1 Immediate Actions",
        f"{p1_count:,}",
        help="Products requiring immediate executive intervention."
    )

with col2:
    danger_metric(
        "P2 High-Priority Actions",
        f"{p2_count:,}",
        help="Products requiring high-priority management intervention."
    )

with col3:
    danger_metric(
        "Critical Exposures",
        f"{critical_exposure_count:,}",
        help="Products with critical contingency exposure."
    )

with col4:
    danger_metric(
        "High Exposures",
        f"{high_exposure_count:,}",
        help="Products with high contingency exposure."
    )


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Additional Safety Stock",
        f"{additional_safety_stock:,.0f}"
    )

with col2:
    danger_metric(
        "Potential Lost-Sales Exposure",
        f"{lost_sales_exposure:,.0f}"
    )

with col3:
    danger_metric(
        "Contingency Cost Exposure",
        money(contingency_cost_exposure)
    )


if p1_count > 0:

    st.error(
        f"**Executive attention required:** "
        f"{p1_count:,} products are classified as P1 Immediate "
        f"and require action within the near-term operating horizon."
    )

elif p2_count > 0:

    st.warning(
        f"{p2_count:,} products require high-priority management attention."
    )

else:

    st.success(
        "No immediate executive interventions are currently required."
    )


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Executive Action Priority"
    )

    if "action_priority" in action_plan.columns:

        action_priority_counts = (
            action_plan["action_priority"]
            .value_counts()
            .rename_axis("action_priority")
            .reset_index(name="products")
        )

        _ap_dom = [p for p in PRIORITY_COLORS if p in action_priority_counts["action_priority"].tolist()]
        st.altair_chart(
            hbar(
                action_priority_counts,
                "action_priority",
                "products",
                color_mode="custom",
                color_field="action_priority",
                color_domain=_ap_dom,
                color_range=[PRIORITY_COLORS[p] for p in _ap_dom],
                val_title="Products",
            ),
            use_container_width=True,
        )


with col2:

    st.subheader(
        "Action Ownership"
    )

    if "action_owner" in action_plan.columns:

        owner_counts = (
            action_plan["action_owner"]
            .value_counts()
            .rename_axis("action_owner")
            .reset_index(name="products")
        )

        st.altair_chart(
            hbar(
                owner_counts,
                "action_owner",
                "products",
                color_mode="blue",
                val_title="Products",
            ),
            use_container_width=True,
        )


if "action_type" in action_plan.columns:

    st.subheader(
        "Executive Action Type"
    )

    action_type_counts = (
        action_plan["action_type"]
        .value_counts()
        .rename_axis("action_type")
        .reset_index(name="products")
    )

    st.altair_chart(
        hbar(
            action_type_counts,
            "action_type",
            "products",
            color_mode="blue",
            val_title="Products",
        ),
        use_container_width=True,
    )


# =========================================================
# CRITICAL MANAGEMENT ACTIONS
# =========================================================

section(
    "Critical Management Actions",
    "Immediate actions requiring executive attention within the near-term operating horizon."
)

if "action_priority" in action_plan.columns:

    critical_actions = action_plan[
        action_plan["action_priority"]
        .astype(str)
        .str.contains(
            "P1",
            na=False
        )
    ].copy()

    if not critical_actions.empty:

        st.warning(
            f"{len(critical_actions):,} products require immediate intervention."
        )

        critical_action_columns = [
            c for c in [
                "product_id",
                "risk_class",
                "primary_risk_driver",
                "scenario",
                "contingency_severity",
                "action_owner",
                "action_horizon",
                "additional_safety_stock_required",
                "lost_sales_exposure",
                "executive_action",
                "executive_confidence"
            ]
            if c in critical_actions.columns
        ]

        if "executive_action_score" in critical_actions.columns:

            critical_actions = critical_actions.sort_values(
                "executive_action_score",
                ascending=False
            )

        st.dataframe(
            critical_actions[
                critical_action_columns
            ].style.format({
                "additional_safety_stock_required": "{:,.0f}",
                "lost_sales_exposure": "{:,.0f}",
                "executive_action_score": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No P1 immediate management actions identified."
        )


# =========================================================
# TOP EXECUTIVE PRIORITIES
# =========================================================

section(
    "Top Executive Priorities",
    "Highest-ranked management interventions based on contingency severity, operational exposure, and executive action score."
)

priority_data = executive_priorities.copy()


if isinstance(priority_data, list):

    priority_data = pd.DataFrame(
        priority_data
    )


if not priority_data.empty:

    if "executive_action_score" in priority_data.columns:

        priority_data[
            "executive_action_score"
        ] = pd.to_numeric(
            priority_data[
                "executive_action_score"
            ],
            errors="coerce"
        )

        priority_data = priority_data.sort_values(
            "executive_action_score",
            ascending=False
        ).head(20)


    priority_columns = [
        c for c in [
            "product_id",
            "risk_class",
            "primary_risk_driver",
            "scenario",
            "contingency_severity",
            "action_priority",
            "executive_action_score",
            "action_owner",
            "executive_action",
            "executive_confidence"
        ]
        if c in priority_data.columns
    ]


    display_priority = priority_data[
        priority_columns
    ].copy()


    if "risk_class" in display_priority.columns:

        display_priority[
            "risk_class"
        ] = display_priority[
            "risk_class"
        ].apply(risk_badge)


    if "action_priority" in display_priority.columns:

        display_priority[
            "action_priority"
        ] = display_priority[
            "action_priority"
        ].apply(priority_badge)


    st.markdown(
        display_priority.style.format({
            "executive_action_score": "{:.3f}"
        }).to_html(
            escape=False,
            index=False
        ),
        unsafe_allow_html=True,
    )

else:

    st.info(
        "No executive priority data available."
    )


# =========================================================
# EXECUTIVE MANAGEMENT BRIEF
# =========================================================

section(
    "Executive Management Brief",
    "Summary of the most important actions and exposures identified by the decision-support engine."
)

if not executive_management_brief.empty:

    brief = executive_management_brief.copy()

    for _, row in brief.iterrows():

        row_data = row.dropna()

        if row_data.empty:
            continue


        title = None

        for candidate in [
            "title",
            "section",
            "metric",
            "category",
            "brief_type"
        ]:

            if candidate in row.index:

                value = row[candidate]

                if pd.notna(value):

                    title = str(value)
                    break


        narrative = None

        for candidate in [
            "management_brief",
            "executive_summary",
            "summary",
            "message",
            "recommendation",
            "description"
        ]:

            if candidate in row.index:

                value = row[candidate]

                if pd.notna(value):

                    narrative = str(value)
                    break


        if title and narrative:

            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-title">
                        {title}
                    </div>

                    <div class="summary-text">
                        {narrative}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif narrative:

            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-text">
                        {narrative}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:

    st.info(
        "No executive management brief available."
    )


# =========================================================
# EXECUTIVE STATUS
# =========================================================

if (
    cost_savings > 0
    and lost_sales_reduction > 0
):

    st.success(
        f"Optimization is producing a positive modeled outcome: "
        f"{money(cost_savings)} in cost savings and "
        f"{lost_sales_reduction:,.0f} fewer lost-sales units."
    )

elif (
    cost_savings < 0
    and lost_sales_reduction > 0
):

    st.warning(
        f"Optimization improves service but increases modeled cost by "
        f"{money(abs(cost_savings))}. Management should evaluate "
        f"whether the service improvement justifies the additional "
        f"inventory investment."
    )

elif cost_savings < 0:

    st.error(
        f"Selected optimization policies increase modeled cost by "
        f"{money(abs(cost_savings))} without sufficient economic benefit."
    )

else:

    st.info(
        "Selected portfolio shows limited modeled optimization benefit."
    )


# =========================================================
# COST VS SERVICE
# =========================================================

section(
    "Cost vs Service Performance",
    "Comparison between the current baseline policy and the optimized inventory policy."
)

col1, col2 = st.columns(2)


with col1:

    st.subheader("Total Cost")

    cost_data = pd.DataFrame({
        "Scenario": [
            "Baseline",
            "Optimized"
        ],
        "Total Cost": [
            baseline_cost,
            optimized_cost
        ]
    })

    st.altair_chart(
        vbar(
            cost_data,
            "Scenario",
            "Total Cost",
            color_mode="custom",
            color_field="Scenario",
            color_domain=["Baseline", "Optimized"],
            color_range=["#5b6b85", BLUE],
            cat_sort=["Baseline", "Optimized"],
            money_fmt=True,
            val_title="Total Cost",
            height=280,
        ),
        use_container_width=True,
    )


with col2:

    st.subheader("Fill Rate")

    service_data = pd.DataFrame({
        "Scenario": [
            "Baseline",
            "Optimized"
        ],
        "Fill Rate": [
            baseline_fill_rate,
            optimized_fill_rate
        ]
    })

    st.altair_chart(
        vbar(
            service_data,
            "Scenario",
            "Fill Rate",
            color_mode="custom",
            color_field="Scenario",
            color_domain=["Baseline", "Optimized"],
            color_range=["#5b6b85", AQUA],
            cat_sort=["Baseline", "Optimized"],
            pct_fmt=True,
            val_title="Fill Rate",
            height=280,
        ),
        use_container_width=True,
    )


# =========================================================
# RISK CLASS PERFORMANCE
# =========================================================

section(
    "Risk Class Performance",
    "Economic and service impact segmented by inventory risk classification."
)

risk_analysis = (
    filtered
    .groupby("risk_class")
    .agg(
        products=("product_id", "nunique"),
        baseline_lost_sales=("baseline_lost_sales", "sum"),
        optimized_lost_sales=("optimized_lost_sales", "sum"),
        cost_savings=("cost_savings", "sum"),
        avg_fill_rate_change=("fill_rate_change", "mean"),
        avg_inventory_days_change=("inventory_days_change", "mean"),
    )
    .reset_index()
)

risk_analysis["lost_sales_reduction"] = (
    risk_analysis["baseline_lost_sales"]
    - risk_analysis["optimized_lost_sales"]
)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Cost Savings by Risk Class"
    )

    st.altair_chart(
        hbar(
            risk_analysis,
            "risk_class",
            "cost_savings",
            color_mode="diverging",
            money_fmt=True,
            val_title="Cost Savings",
        ),
        use_container_width=True,
    )


with col2:

    st.subheader(
        "Lost-Sales Reduction by Risk Class"
    )

    st.altair_chart(
        hbar(
            risk_analysis,
            "risk_class",
            "lost_sales_reduction",
            color_mode="severity",
            val_title="Lost-Sales Reduction",
        ),
        use_container_width=True,
    )


display_risk = risk_analysis.copy()

display_risk[
    "risk_class"
] = display_risk[
    "risk_class"
].apply(risk_badge)


st.markdown(
    display_risk[
        [
            "risk_class",
            "products",
            "lost_sales_reduction",
            "cost_savings",
            "avg_fill_rate_change",
            "avg_inventory_days_change"
        ]
    ]
    .sort_values(
        "cost_savings",
        ascending=False
    )
    .style.format({
        "lost_sales_reduction": "{:,.0f}",
        "cost_savings": "${:,.2f}",
        "avg_fill_rate_change": "{:+.2%}",
        "avg_inventory_days_change": "{:+.2f}",
    })
    .to_html(
        escape=False,
        index=False
    ),
    unsafe_allow_html=True,
)


# =========================================================
# CATEGORY PERFORMANCE
# =========================================================

if "category" in filtered.columns:

    section(
        "Category Performance",
        "Identify which product categories generate the largest optimization opportunities."
    )

    category_analysis = (
        filtered
        .groupby("category")
        .agg(
            products=("product_id", "nunique"),
            cost_savings=("cost_savings", "sum"),
            lost_sales_reduction=("lost_sales_reduction", "sum"),
            avg_fill_rate_change=("fill_rate_change", "mean"),
            avg_inventory_days_change=("inventory_days_change", "mean"),
        )
        .reset_index()
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Cost Savings by Category"
        )

        st.altair_chart(
            hbar(
                category_analysis,
                "category",
                "cost_savings",
                color_mode="diverging",
                money_fmt=True,
                val_title="Cost Savings",
            ),
            use_container_width=True,
        )


    with col2:

        st.subheader(
            "Lost-Sales Reduction by Category"
        )

        st.altair_chart(
            hbar(
                category_analysis,
                "category",
                "lost_sales_reduction",
                color_mode="blue",
                val_title="Lost-Sales Reduction",
            ),
            use_container_width=True,
        )


    st.dataframe(
        category_analysis
        .sort_values(
            "cost_savings",
            ascending=False
        )
        .style.format({
            "cost_savings": "${:,.2f}",
            "lost_sales_reduction": "{:,.0f}",
            "avg_fill_rate_change": "{:+.2%}",
            "avg_inventory_days_change": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# MANAGEMENT ACTION CENTER
# =========================================================

section(
    "Management Action Center",
    "Prioritized interventions generated from the optimization decision engine."
)

if "management_decision" in filtered.columns:

    decision_counts = (
        filtered["management_decision"]
        .fillna("Not specified")
        .value_counts()
        .rename_axis("management_decision")
        .reset_index(name="products")
    )


    col1, col2 = st.columns([
        1.5,
        1
    ])


    with col1:

        st.subheader(
            "Recommended Management Actions"
        )

        st.altair_chart(
            hbar(
                decision_counts,
                "management_decision",
                "products",
                color_mode="blue",
                val_title="Products",
            ),
            use_container_width=True,
        )


    with col2:

        st.subheader(
            "Action Distribution"
        )

        st.dataframe(
            decision_counts,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PRIORITY DISTRIBUTION
# =========================================================

if "priority" in filtered.columns:

    section(
        "Priority Distribution",
        "Portfolio segmentation based on urgency and business impact."
    )

    priority_counts = (
        filtered["priority"]
        .fillna("Not specified")
        .value_counts()
        .rename_axis("priority")
        .reset_index(name="products")
    )

    def _prio_color(label):
        s = str(label)
        if "Critical" in s:
            return SEVERITY_COLORS["Critical"]
        if "High" in s:
            return SEVERITY_COLORS["High"]
        if "Medium" in s:
            return SEVERITY_COLORS["Medium"]
        if "Low" in s:
            return SEVERITY_COLORS["Low"]
        return BLUE

    _pd_dom = priority_counts["priority"].tolist()
    st.altair_chart(
        hbar(
            priority_counts,
            "priority",
            "products",
            color_mode="custom",
            color_field="priority",
            color_domain=_pd_dom,
            color_range=[_prio_color(p) for p in _pd_dom],
            val_title="Products",
        ),
        use_container_width=True,
    )


# =========================================================
# DECISION IMPACT
# =========================================================

if "decision_impact_score" in filtered.columns:

    section(
        "Decision Impact",
        "Products ranked by the modeled economic and operational importance of intervention."
    )

    impact_df = filtered.copy()

    impact_df[
        "decision_impact_score"
    ] = pd.to_numeric(
        impact_df[
            "decision_impact_score"
        ],
        errors="coerce"
    )

    impact_df = impact_df.sort_values(
        "decision_impact_score",
        ascending=False
    ).head(20)


    impact_columns = [
        "product_id",
        "category",
        "risk_class",
        "decision_impact_score",
        "cost_savings",
        "lost_sales_reduction"
    ]


    for column in [
        "priority",
        "management_decision",
        "business_impact",
        "recommended_action"
    ]:

        if column in impact_df.columns:

            impact_columns.append(
                column
            )


    display_impact = impact_df[
        impact_columns
    ].copy()


    if "risk_class" in display_impact.columns:

        display_impact[
            "risk_class"
        ] = display_impact[
            "risk_class"
        ].apply(risk_badge)


    if "priority" in display_impact.columns:

        display_impact[
            "priority"
        ] = display_impact[
            "priority"
        ].apply(priority_badge)


    st.markdown(
        display_impact.style.format({
            "decision_impact_score": "{:,.2f}",
            "cost_savings": "${:,.2f}",
            "lost_sales_reduction": "{:,.0f}",
        }).to_html(
            escape=False,
            index=False
        ),
        unsafe_allow_html=True,
    )


# =========================================================
# BUSINESS IMPACT
# =========================================================

if "business_impact" in filtered.columns:

    section(
        "Business Impact",
        "Classification of the modeled economic and service outcomes."
    )

    impact_counts = (
        filtered["business_impact"]
        .fillna("Not specified")
        .value_counts()
        .rename_axis("business_impact")
        .reset_index(name="products")
    )

    st.altair_chart(
        hbar(
            impact_counts,
            "business_impact",
            "products",
            color_mode="blue",
            val_title="Products",
        ),
        use_container_width=True,
    )


# =========================================================
# SUPPLIER RISK
# =========================================================

if "optimized_supplier_delay_rate" in filtered.columns:

    section(
        "Supplier Risk Exposure",
        "Products where supplier reliability may affect the optimized inventory policy."
    )

    supplier_rate = pd.to_numeric(
        filtered[
            "optimized_supplier_delay_rate"
        ],
        errors="coerce"
    )

    supplier_delay = pd.to_numeric(
        filtered[
            "optimized_average_supplier_delay"
        ],
        errors="coerce"
    )


    supplier_products = filtered.copy()

    supplier_products[
        "optimized_supplier_delay_rate"
    ] = supplier_rate

    supplier_products[
        "optimized_average_supplier_delay"
    ] = supplier_delay

    supplier_products[
        "supplier_risk_score"
    ] = (
        supplier_rate.fillna(0)
        *
        supplier_delay.fillna(0)
    )


    supplier_products = supplier_products.sort_values(
        "supplier_risk_score",
        ascending=False
    ).head(20)


    supplier_columns = [
        "product_id",
        "category",
        "risk_class",
        "optimized_supplier_delay_rate",
        "optimized_average_supplier_delay",
        "lost_sales_reduction",
        "cost_savings"
    ]


    if "management_decision" in supplier_products.columns:

        supplier_columns.append(
            "management_decision"
        )


    st.dataframe(
        supplier_products[
            supplier_columns
        ].style.format({
            "optimized_supplier_delay_rate": "{:.2%}",
            "optimized_average_supplier_delay": "{:.2f}",
            "lost_sales_reduction": "{:,.0f}",
            "cost_savings": "${:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TOP OPTIMIZATION OPPORTUNITIES
# =========================================================

section(
    "Top Optimization Opportunities",
    "Products with the largest modeled economic and service opportunity."
)

top_columns = [
    "product_id",
    "category",
    "risk_class",
    "primary_risk_driver"
]

optional_columns = [
    "priority",
    "decision_impact_score",
    "lost_sales_reduction",
    "cost_savings",
    "fill_rate_change",
    "inventory_days_change",
    "recommended_action",
    "management_decision",
    "action_priority",
    "executive_action_score",
    "action_owner",
    "executive_action",
    "executive_confidence",
    "action_horizon",
    "contingency_severity"
]


for column in optional_columns:

    if column in filtered.columns:

        top_columns.append(
            column
        )


sort_columns = []


if "executive_action_score" in filtered.columns:

    sort_columns.append(
        "executive_action_score"
    )


if "decision_impact_score" in filtered.columns:

    sort_columns.append(
        "decision_impact_score"
    )


if "cost_savings" in filtered.columns:

    sort_columns.append(
        "cost_savings"
    )


if "lost_sales_reduction" in filtered.columns:

    sort_columns.append(
        "lost_sales_reduction"
    )


if sort_columns:

    top_products = filtered.sort_values(
        sort_columns,
        ascending=False
    ).head(20).copy()

else:

    top_products = filtered.head(
        20
    ).copy()


if "risk_class" in top_products.columns:

    top_products[
        "risk_class"
    ] = top_products[
        "risk_class"
    ].apply(risk_badge)


if "priority" in top_products.columns:

    top_products[
        "priority"
    ] = top_products[
        "priority"
    ].apply(priority_badge)


st.markdown(
    top_products[
        top_columns
    ].style.format({
        "decision_impact_score": "{:,.2f}",
        "lost_sales_reduction": "{:,.0f}",
        "cost_savings": "${:,.2f}",
        "fill_rate_change": "{:+.2%}",
        "inventory_days_change": "{:+.2f}",
    }).to_html(
        escape=False,
        index=False
    ),
    unsafe_allow_html=True,
)


# =========================================================
# CRITICAL INVENTORY EXPOSURE
# =========================================================

critical_products = filtered[
    filtered["risk_class"] == "Critical"
].copy()


if len(critical_products) > 0:

    section(
        "Critical Inventory Exposure",
        "Highest-risk products requiring immediate management attention."
    )

    critical_lost_sales = critical_products[
        "baseline_lost_sales"
    ].sum()

    critical_reduction = critical_products[
        "lost_sales_reduction"
    ].sum()

    critical_savings = critical_products[
        "cost_savings"
    ].sum()


    col1, col2, col3 = st.columns(3)

    with col1:

        danger_metric(
            "Critical Products",
            f"{len(critical_products):,}"
        )

    with col2:

        danger_metric(
            "Baseline Lost Sales",
            f"{critical_lost_sales:,.0f}"
        )

    with col3:

        st.metric(
            "Optimization Savings",
            money(critical_savings)
        )


    if critical_reduction > 0:

        st.success(
            f"Optimization is expected to reduce "
            f"critical-product lost sales by "
            f"{critical_reduction:,.0f} units."
        )

    else:

        st.error(
            "Critical inventory exposure remains unresolved "
            "under the selected optimization policies."
        )


    critical_display_columns = [
        "product_id",
        "category",
        "primary_risk_driver",
        "baseline_lost_sales",
        "optimized_lost_sales",
        "lost_sales_reduction",
        "cost_savings",
        "fill_rate_change",
        "inventory_days_change"
    ]


    if "management_decision" in critical_products.columns:

        critical_display_columns.append(
            "management_decision"
        )


    critical_display = critical_products[
        critical_display_columns
    ].sort_values(
        "lost_sales_reduction",
        ascending=False
    ).head(20).copy()


    st.dataframe(
        critical_display.style.format({
            "baseline_lost_sales": "{:,.0f}",
            "optimized_lost_sales": "{:,.0f}",
            "lost_sales_reduction": "{:,.0f}",
            "cost_savings": "${:,.2f}",
            "fill_rate_change": "{:+.2%}",
            "inventory_days_change": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# SCENARIO ANALYSIS
# =========================================================

section(
    "Scenario Analysis",
    "Stress-test inventory performance under alternative operating conditions."
)


if (
    not scenario_summary.empty
    and "scenario" in scenario_summary.columns
):

    scenario_display = scenario_summary.copy()


    scenario_columns = [
        c for c in [
            "scenario",
            "products",
            "avg_fill_rate",
            "avg_stockout_rate",
            "total_lost_sales",
            "avg_inventory_days",
            "total_purchase_orders",
            "total_purchase_quantity",
            "avg_supplier_delay_rate",
            "avg_actual_lead_time",
            "avg_supplier_delay",
            "fill_rate_change",
            "stockout_rate_change",
            "lost_sales_change",
            "inventory_days_change"
        ]
        if c in scenario_display.columns
    ]


    st.dataframe(
        scenario_display[
            scenario_columns
        ].style.format({
            "avg_fill_rate": "{:.2%}",
            "avg_stockout_rate": "{:.2%}",
            "total_lost_sales": "{:,.0f}",
            "avg_inventory_days": "{:.2f}",
            "total_purchase_orders": "{:,.0f}",
            "total_purchase_quantity": "{:,.0f}",
            "avg_supplier_delay_rate": "{:.2%}",
            "avg_actual_lead_time": "{:.2f}",
            "avg_supplier_delay": "{:.2f}",
            "fill_rate_change": "{:+.2%}",
            "stockout_rate_change": "{:+.2%}",
            "lost_sales_change": "{:+,.0f}",
            "inventory_days_change": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Scenario Fill Rate"
        )

        scenario_chart = (
            scenario_summary[
                [
                    "scenario",
                    "avg_fill_rate"
                ]
            ]
            .dropna()
        )

        st.altair_chart(
            hbar(
                scenario_chart,
                "scenario",
                "avg_fill_rate",
                color_mode="blue",
                pct_fmt=True,
                val_title="Avg Fill Rate",
            ),
            use_container_width=True,
        )


    with col2:

        st.subheader(
            "Scenario Lost Sales"
        )

        lost_sales_chart = (
            scenario_summary[
                [
                    "scenario",
                    "total_lost_sales"
                ]
            ]
            .dropna()
        )

        st.altair_chart(
            hbar(
                lost_sales_chart,
                "scenario",
                "total_lost_sales",
                color_mode="sequential",
                val_title="Total Lost Sales",
            ),
            use_container_width=True,
        )


    if (
        "avg_fill_rate" in scenario_summary.columns
        and
        "total_lost_sales" in scenario_summary.columns
    ):

        scenario_eval = scenario_summary.dropna(
            subset=[
                "avg_fill_rate",
                "total_lost_sales"
            ]
        ).copy()


        if not scenario_eval.empty:

            best_service = scenario_eval.loc[
                scenario_eval[
                    "avg_fill_rate"
                ].idxmax()
            ]

            best_lost_sales = scenario_eval.loc[
                scenario_eval[
                    "total_lost_sales"
                ].idxmin()
            ]


            col1, col2 = st.columns(2)


            with col1:

                st.success(
                    f"Best service scenario: "
                    f"**{best_service['scenario']}** "
                    f"with "
                    f"{best_service['avg_fill_rate']:.2%} "
                    f"average fill rate."
                )


            with col2:

                st.info(
                    f"Lowest lost-sales scenario: "
                    f"**{best_lost_sales['scenario']}** "
                    f"with "
                    f"{best_lost_sales['total_lost_sales']:,.0f} "
                    f"lost-sales units."
                )

else:

    if not scenario_results.empty:

        st.dataframe(
            scenario_results.head(50),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No scenario analysis data available."
        )


# =========================================================
# PRODUCT INVESTIGATION
# =========================================================

section(
    "Product-Level Investigation",
    "Investigate the modeled inventory policy for an individual product."
)

product_ids = sorted(
    filtered["product_id"]
    .dropna()
    .unique()
)


if len(product_ids) > 0:

    selected_product = st.selectbox(
        "Select Product",
        product_ids
    )

    product_row = filtered[
        filtered["product_id"] == selected_product
    ].iloc[0]


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Risk Class",
            product_row["risk_class"]
        )


    with col2:

        st.metric(
            "Lost-Sales Reduction",
            f"{product_row['lost_sales_reduction']:,.0f}"
        )


    with col3:

        st.metric(
            "Cost Savings",
            money(
                product_row["cost_savings"]
            )
        )


    with col4:

        st.metric(
            "Fill Rate Change",
            f"{product_row['fill_rate_change']:+.2%}"
        )


    if "priority" in product_row.index:

        priority_text = str(
            product_row["priority"]
        )

        if "Critical" in priority_text:

            st.error(
                f"**PRIORITY: {priority_text}**"
            )

        elif "High" in priority_text:

            st.warning(
                f"**PRIORITY: {priority_text}**"
            )

        elif "Medium" in priority_text:

            st.info(
                f"**PRIORITY: {priority_text}**"
            )

        else:

            st.success(
                f"**PRIORITY: {priority_text}**"
            )


    product_comparison = pd.DataFrame({

        "Metric": [
            "Service Level",
            "Safety Stock",
            "Reorder Point",
            "Order Quantity",
            "Order Coverage Days",
            "Fill Rate",
            "Lost Sales",
            "Inventory Days",
            "Total Cost"
        ],

        "Baseline": [
            product_row["baseline_service_level"],
            product_row["baseline_safety_stock"],
            product_row["baseline_reorder_point"],
            product_row["baseline_order_quantity"],
            product_row["baseline_order_coverage_days"],
            product_row["baseline_fill_rate"],
            product_row["baseline_lost_sales"],
            product_row["baseline_inventory_days"],
            product_row["baseline_total_cost"],
        ],

        "Optimized": [
            product_row["optimized_service_level"],
            product_row["optimized_safety_stock"],
            product_row["optimized_reorder_point"],
            product_row["optimized_order_quantity"],
            product_row["optimized_order_coverage_days"],
            product_row["optimized_fill_rate"],
            product_row["optimized_lost_sales"],
            product_row["optimized_inventory_days"],
            product_row["optimized_total_cost"],
        ],
    })


    st.dataframe(
        product_comparison.style.format({
            "Baseline": "{:,.2f}",
            "Optimized": "{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True,
    )


    product_change_data = []


    if "safety_stock_change" in product_row.index:

        product_change_data.append({
            "Metric": "Safety Stock Change",
            "Value": product_row[
                "safety_stock_change"
            ]
        })


    if "order_quantity_change" in product_row.index:

        product_change_data.append({
            "Metric": "Order Quantity Change",
            "Value": product_row[
                "order_quantity_change"
            ]
        })


    if "inventory_days_change" in product_row.index:

        product_change_data.append({
            "Metric": "Inventory Days Change",
            "Value": product_row[
                "inventory_days_change"
            ]
        })


    if product_change_data:

        st.subheader(
            "Policy Changes"
        )

        st.dataframe(
            pd.DataFrame(
                product_change_data
            ).style.format({
                "Value": "{:+,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# EXECUTIVE TAKEAWAYS
# =========================================================

section(
    "Executive Takeaways",
    "High-level conclusions from the selected portfolio."
)

positive_savings = filtered[
    filtered["cost_savings"] > 0
]

negative_savings = filtered[
    filtered["cost_savings"] < 0
]


largest_risk_class = (
    None
    if risk_analysis.empty
    else risk_analysis
    .sort_values(
        "lost_sales_reduction",
        ascending=False
    )
    .iloc[0]["risk_class"]
)


if lost_sales_reduction > 0:

    takeaway_1 = (
        f"✅ **{lost_sales_reduction:,.0f}** "
        f"units reduction in lost sales"
    )

else:

    takeaway_1 = (
        f"⚠️ Lost sales change: "
        f"**{lost_sales_reduction:,.0f}** units"
    )


if cost_savings > 0:

    takeaway_2 = (
        f"💰 Cost savings: "
        f"**${cost_savings:,.0f}**"
    )

elif cost_savings < 0:

    takeaway_2 = (
        f"⚠️ Cost increase: "
        f"**${abs(cost_savings):,.0f}**"
    )

else:

    takeaway_2 = (
        "Cost remains unchanged"
    )


takeaway_3 = (
    f"📦 **{len(positive_savings):,}** "
    f"products show positive savings, "
    f"**{len(negative_savings):,}** "
    f"show increased cost"
)


if largest_risk_class is not None:

    takeaway_4 = (
        f"🎯 **{largest_risk_class}** risk products "
        f"drive the largest improvement"
    )

else:

    takeaway_4 = (
        "No dominant risk class identified"
    )


if "decision_impact_score" in filtered.columns:

    highest_impact_product = (
        filtered
        .sort_values(
            "decision_impact_score",
            ascending=False
        )
        .iloc[0]
    )

    takeaway_5 = (
        f"🔑 Priority: "
        f"**{highest_impact_product['product_id']}** "
        f"(Impact Score: "
        f"{highest_impact_product['decision_impact_score']:.2f})"
    )

else:

    takeaway_5 = (
        "Decision impact scoring not available"
    )


st.success(
    f"**{takeaway_1}**"
)

st.info(
    f"**{takeaway_2}**"
)

st.warning(
    f"**{takeaway_3}**"
)

st.info(
    f"**{takeaway_4}**"
)


if "decision_impact_score" in filtered.columns:

    st.success(
        f"**{takeaway_5}**"
    )


# =========================================================
# DOWNLOAD FILTERED PORTFOLIO
# =========================================================

section(
    "Export",
    "Download the currently filtered optimization portfolio for further analysis."
)

export_columns = [
    c for c in [
        "product_id",
        "category",
        "demand_class",
        "risk_class",
        "primary_risk_driver",
        "cost_savings",
        "lost_sales_reduction",
        "fill_rate_change",
        "inventory_days_change",
        "recommended_action",
        "decision_impact_score",
        "priority",
        "management_decision",
        "business_impact",
        "policy_change_flag",
        "priority_rank"
    ]
    if c in filtered.columns
]


export_data = filtered[
    export_columns
].copy()


csv_data = export_data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Filtered Portfolio",
    data=csv_data,
    file_name="filtered_inventory_optimization.csv",
    mime="text/csv",
)


# =========================================================
# END OF DASHBOARD
# =========================================================
