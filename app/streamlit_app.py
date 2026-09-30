from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================
# PROJECT BOOTSTRAP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeromaintain.agents.maintenance_agent import MaintenanceAgent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AeroMaintain AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

CSS = """
<style>

:root {
    --bg: #07111d;
    --surface: #0c1928;
    --surface-2: #102033;
    --surface-3: #14263b;

    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(103, 232, 249, 0.24);

    --text: #f4f8fc;
    --text-soft: #d6e0eb;
    --muted: #91a3b8;
    --muted-2: #66788e;

    --cyan: #67e8f9;
    --cyan-strong: #22d3ee;
    --teal: #5eead4;
    --green: #86efac;
    --amber: #fbbf24;

    --radius: 12px;
    --shadow: 0 12px 34px rgba(0, 0, 0, 0.18);
}


/* ============================================================
   GLOBAL
============================================================ */

html {
    font-size: 16px;
}

html,
body,
[class*="css"] {
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    color: var(--text);

    background:
        radial-gradient(
            circle at 84% -10%,
            rgba(34, 211, 238, 0.07),
            transparent 28%
        ),
        radial-gradient(
            circle at 28% -10%,
            rgba(45, 212, 191, 0.035),
            transparent 22%
        ),
        var(--bg);
}

.block-container {
    width: 100%;
    max-width: 1560px;

    padding-top: 0.85rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

#MainMenu,
footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

div[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0;
}


/* ============================================================
   SIDEBAR
============================================================ */

section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);

    background:
        linear-gradient(
            180deg,
            #091725 0%,
            #07111d 100%
        );
}

section[data-testid="stSidebar"] > div {
    padding-top: 0.85rem;
}

.sidebar-brand {
    padding: 0.05rem 0.05rem 0.85rem;
    margin-bottom: 0.85rem;

    border-bottom: 1px solid var(--border);
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.brand-mark {
    width: 36px;
    height: 36px;
    min-width: 36px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 9px;

    color: #04121c;

    font-size: 0.9rem;
    font-weight: 900;

    background:
        linear-gradient(
            135deg,
            var(--cyan),
            var(--teal)
        );

    box-shadow:
        0 0 22px rgba(94, 234, 212, 0.14);
}

.brand-name {
    color: var(--text);

    font-size: 0.84rem;
    font-weight: 850;

    letter-spacing: 0.08em;
}

.brand-subtitle {
    margin-top: 0.08rem;

    color: var(--muted-2);

    font-size: 0.56rem;
    font-weight: 650;

    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.sidebar-label {
    margin:
        0.9rem
        0
        0.42rem
        0.1rem;

    color: var(--muted-2);

    font-size: 0.57rem;
    font-weight: 850;

    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.layer-active,
.layer-item {
    display: flex;
    align-items: center;

    gap: 0.55rem;

    padding: 0.52rem 0.58rem;
    margin-bottom: 0.16rem;

    border-radius: 8px;

    font-size: 0.68rem;
    line-height: 1.25;
}

.layer-active {
    color: var(--text);

    font-weight: 760;

    border:
        1px solid
        rgba(103, 232, 249, 0.17);

    background:
        linear-gradient(
            90deg,
            rgba(34, 211, 238, 0.11),
            rgba(34, 211, 238, 0.025)
        );
}

.layer-item {
    color: var(--muted);
}

.layer-index {
    min-width: 20px;

    color: var(--cyan);

    font-size: 0.52rem;
    font-weight: 900;

    letter-spacing: 0.06em;
}

.sidebar-system {
    margin-top: 0.15rem;
    padding: 0.58rem;

    border: 1px solid var(--border);
    border-radius: 10px;

    background:
        rgba(15, 28, 45, 0.52);
}

.sidebar-system-row {
    display: grid;
    grid-template-columns: 1fr auto;

    align-items: center;

    gap: 0.4rem;

    padding: 0.27rem 0;

    color: var(--muted);

    font-size: 0.59rem;
}

.sidebar-ready {
    display: inline-flex;
    align-items: center;

    color: var(--green);

    font-size: 0.52rem;
    font-weight: 850;

    letter-spacing: 0.035em;
}

.status-dot {
    display: inline-block;

    width: 6px;
    height: 6px;

    margin-right: 5px;

    border-radius: 50%;

    background: var(--green);

    box-shadow:
        0 0 8px rgba(134, 239, 172, 0.65);
}

.sidebar-environment {
    color: var(--muted);

    font-size: 0.59rem;
    line-height: 1.55;
}


/* ============================================================
   PRODUCT BAR
============================================================ */

.product-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 1rem;

    margin-bottom: 0.55rem;
    padding-bottom: 0.5rem;

    border-bottom: 1px solid var(--border);
}

.product-path {
    display: flex;
    align-items: center;

    gap: 0.42rem;

    color: var(--muted-2);

    font-size: 0.58rem;
    font-weight: 750;

    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.product-path-active {
    color: var(--text-soft);
}

.product-separator {
    color: #334155;
}

.live-badge {
    flex-shrink: 0;

    padding: 0.31rem 0.52rem;

    border:
        1px solid
        rgba(134, 239, 172, 0.17);

    border-radius: 999px;

    background:
        rgba(134, 239, 172, 0.045);

    color: var(--green);

    font-size: 0.54rem;
    font-weight: 850;

    letter-spacing: 0.07em;
    text-transform: uppercase;
}


/* ============================================================
   HERO
============================================================ */

.hero {
    margin-bottom: 0.65rem;
}

.eyebrow {
    margin-bottom: 0.2rem;

    color: var(--cyan);

    font-size: 0.56rem;
    font-weight: 850;

    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.page-title {
    margin: 0;

    color: var(--text);

    font-size: clamp(
        1.65rem,
        2.35vw,
        2.15rem
    );

    font-weight: 790;

    letter-spacing: -0.04em;
    line-height: 1.04;
}

.page-subtitle {
    max-width: 900px;

    margin-top: 0.28rem;

    color: var(--muted);

    font-size: 0.69rem;
    line-height: 1.42;
}


/* ============================================================
   SECTION HEADINGS
============================================================ */

.section-kicker {
    margin-bottom: 0.16rem;

    color: var(--muted-2);

    font-size: 0.53rem;
    font-weight: 850;

    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.section-heading {
    color: var(--text);

    font-size: 0.84rem;
    font-weight: 760;
}

.section-description {
    margin-top: 0.1rem;
    margin-bottom: 0.42rem;

    color: var(--muted);

    font-size: 0.61rem;
    line-height: 1.4;
}


/* ============================================================
   INPUT
============================================================ */

div[data-testid="stTextInput"] input {
    min-height: 39px;

    padding-left: 0.8rem;

    border: 1px solid var(--border);
    border-radius: 9px;

    background:
        rgba(15, 28, 45, 0.82);

    color: var(--text);

    font-size: 0.72rem;
}

div[data-testid="stTextInput"] input:focus {
    border-color:
        rgba(103, 232, 249, 0.46);

    box-shadow:
        0 0 0 1px rgba(103, 232, 249, 0.10);
}

div.stButton > button {
    min-height: 39px;

    border:
        1px solid
        rgba(103, 232, 249, 0.25);

    border-radius: 9px;

    background:
        linear-gradient(
            135deg,
            #22d3ee,
            #2dd4bf
        );

    color: #04121c;

    font-size: 0.63rem;
    font-weight: 900;

    letter-spacing: 0.045em;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

div.stButton > button:hover {
    color: #04121c;

    transform: translateY(-1px);

    box-shadow:
        0 8px 22px rgba(34, 211, 238, 0.14);
}

.sample-hint {
    margin-top: -0.28rem;

    color: var(--muted-2);

    font-size: 0.53rem;
}


/* ============================================================
   COMPONENT HEADER
============================================================ */

.component-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 1rem;

    margin-top: 0.5rem;
    margin-bottom: 0.42rem;
}

.component-main {
    display: flex;
    align-items: baseline;

    gap: 0.6rem;

    flex-wrap: wrap;
}

.component-id {
    color: var(--text);

    font-size: 1.05rem;
    font-weight: 790;

    letter-spacing: -0.025em;
}

.component-caption {
    color: var(--muted-2);

    font-size: 0.55rem;
}

.component-type {
    padding: 0.3rem 0.48rem;

    border:
        1px solid
        rgba(103, 232, 249, 0.17);

    border-radius: 7px;

    background:
        rgba(103, 232, 249, 0.05);

    color: var(--cyan);

    font-size: 0.53rem;
    font-weight: 850;

    letter-spacing: 0.07em;
}


/* ============================================================
   METRIC CARDS
============================================================ */

.metric-card {
    min-height: 83px;

    position: relative;
    overflow: hidden;

    padding: 0.65rem 0.75rem;

    border: 1px solid var(--border);
    border-radius: var(--radius);

    background:
        linear-gradient(
            145deg,
            rgba(15, 28, 45, 0.95),
            rgba(10, 22, 37, 0.95)
        );

    box-shadow: var(--shadow);
}

.metric-card::after {
    content: "";

    position: absolute;

    width: 58px;
    height: 58px;

    top: -34px;
    right: -27px;

    border-radius: 50%;

    background:
        rgba(103, 232, 249, 0.04);
}

.metric-label {
    color: var(--muted-2);

    font-size: 0.48rem;
    font-weight: 850;

    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 0.3rem;

    color: var(--text);

    font-size: 1.05rem;
    font-weight: 790;

    letter-spacing: -0.03em;
}

.metric-meta {
    margin-top: 0.17rem;

    color: var(--muted);

    font-size: 0.51rem;
}

.metric-accent {
    color: var(--cyan);
}

.metric-safe {
    color: var(--green);
}

.metric-alert {
    color: var(--amber);
}


/* ============================================================
   PANEL
============================================================ */

.panel {
    margin-top: 0.28rem;

    padding: 0.75rem;

    border: 1px solid var(--border);
    border-radius: var(--radius);

    background:
        linear-gradient(
            145deg,
            rgba(15, 28, 45, 0.90),
            rgba(9, 20, 34, 0.94)
        );

    box-shadow: var(--shadow);
}

.panel-title {
    color: var(--text);

    font-size: 0.64rem;
    font-weight: 780;
}

.panel-subtitle {
    margin-top: 0.1rem;
    margin-bottom: 0.45rem;

    color: var(--muted-2);

    font-size: 0.53rem;
}


/* ============================================================
   RISK
============================================================ */

.risk-layout {
    display: grid;
    grid-template-columns: 145px 1fr;

    align-items: center;

    gap: 1rem;
}

.risk-number {
    margin-top: 0.12rem;

    color: var(--text);

    font-size: 1.62rem;
    font-weight: 800;

    letter-spacing: -0.05em;
}

.risk-state-safe,
.risk-state-alert {
    margin-top: 0.08rem;

    font-size: 0.51rem;
    font-weight: 850;

    letter-spacing: 0.055em;
    text-transform: uppercase;
}

.risk-state-safe {
    color: var(--green);
}

.risk-state-alert {
    color: var(--amber);
}

.risk-track {
    position: relative;

    height: 7px;

    margin-top: 0.45rem;
    margin-bottom: 1.3rem;

    border-radius: 999px;

    background:
        rgba(148, 163, 184, 0.10);
}

.risk-fill {
    position: absolute;

    left: 0;
    top: 0;

    height: 7px;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #22d3ee,
            #2dd4bf
        );

    box-shadow:
        0 0 12px rgba(45, 212, 191, 0.18);
}

.risk-threshold {
    position: absolute;

    top: -5px;

    width: 2px;
    height: 17px;

    background: var(--amber);
}

.risk-threshold-label {
    position: absolute;

    top: 15px;

    transform: translateX(-50%);

    color: var(--muted);

    font-size: 0.48rem;

    white-space: nowrap;
}

.risk-scale {
    display: flex;
    justify-content: space-between;

    color: var(--muted-2);

    font-size: 0.47rem;
}

.model-meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);

    gap: 0.5rem;

    margin-top: 0.45rem;
    padding-top: 0.45rem;

    border-top: 1px solid var(--border);
}

.model-meta-label {
    color: var(--muted-2);

    font-size: 0.45rem;
    font-weight: 850;

    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.model-meta-value {
    margin-top: 0.08rem;

    color: var(--text-soft);

    font-size: 0.53rem;
}


/* ============================================================
   HISTORY
============================================================ */

.timeline-card {
    position: relative;

    margin: 0.52rem 0;
    padding:
        0.03rem
        0
        0.03rem
        0.75rem;

    border-left:
        2px solid
        rgba(103, 232, 249, 0.30);
}

.timeline-card::before {
    content: "";

    position: absolute;

    left: -5px;
    top: 4px;

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: var(--cyan);

    box-shadow:
        0 0 8px rgba(103, 232, 249, 0.28);
}

.timeline-date {
    color: var(--muted-2);

    font-size: 0.48rem;
    font-weight: 800;

    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.timeline-action {
    margin-top: 0.12rem;

    color: var(--text);

    font-size: 0.63rem;
    font-weight: 780;
}

.timeline-meta {
    margin-top: 0.12rem;

    color: var(--cyan);

    font-size: 0.51rem;
}

.timeline-note {
    margin-top: 0.2rem;

    color: var(--muted);

    font-size: 0.53rem;
    line-height: 1.4;
}

.timeline-action-detail {
    margin-top: 0.2rem;

    color: var(--text-soft);

    font-size: 0.51rem;
    font-weight: 700;
}


/* ============================================================
   EVIDENCE
============================================================ */

.evidence-card {
    margin-bottom: 0.42rem;
    padding: 0.58rem;

    border: 1px solid var(--border);
    border-radius: 8px;

    background:
        rgba(19, 34, 54, 0.43);
}

.evidence-top {
    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 0.5rem;
}

.evidence-index {
    color: var(--cyan);

    font-size: 0.47rem;
    font-weight: 900;

    letter-spacing: 0.08em;
}

.evidence-score {
    color: var(--muted);

    font-size: 0.47rem;
}

.evidence-section {
    margin-top: 0.22rem;

    color: var(--text);

    font-size: 0.61rem;
    font-weight: 780;
}

.evidence-source {
    margin-top: 0.1rem;

    color: var(--muted-2);

    font-size: 0.48rem;
}

.evidence-content {
    margin-top: 0.26rem;

    color: var(--muted);

    font-size: 0.52rem;
    line-height: 1.4;
}


/* ============================================================
   BRIEF
============================================================ */

.brief {
    margin-top: 0.28rem;

    padding: 0.7rem 0.8rem;

    border:
        1px solid
        rgba(94, 234, 212, 0.19);

    border-radius: var(--radius);

    background:
        linear-gradient(
            135deg,
            rgba(34, 211, 238, 0.05),
            rgba(45, 212, 191, 0.022)
        );
}

.brief-label {
    margin-bottom: 0.28rem;

    color: var(--cyan);

    font-size: 0.49rem;
    font-weight: 900;

    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.brief-text {
    color: var(--text-soft);

    font-size: 0.57rem;
    line-height: 1.5;
}


/* ============================================================
   LANDING CAPABILITIES
============================================================ */

.capability-card {
    min-height: 104px;

    padding: 0.72rem;

    border: 1px solid var(--border);
    border-radius: var(--radius);

    background:
        linear-gradient(
            145deg,
            rgba(15, 28, 45, 0.91),
            rgba(10, 21, 35, 0.91)
        );

    box-shadow: var(--shadow);

    transition:
        transform 0.16s ease,
        border-color 0.16s ease;
}

.capability-card:hover {
    transform: translateY(-2px);

    border-color: var(--border-strong);
}

.capability-top {
    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 0.5rem;

    margin-bottom: 0.5rem;
}

.capability-index {
    color: var(--cyan);

    font-size: 0.52rem;
    font-weight: 900;

    letter-spacing: 0.11em;
}

.capability-status {
    color: var(--green);

    font-size: 0.47rem;
    font-weight: 850;

    letter-spacing: 0.06em;
}

.capability-title {
    color: var(--text);

    font-size: 0.66rem;
    font-weight: 780;
}

.capability-text {
    margin-top: 0.22rem;

    color: var(--muted);

    font-size: 0.56rem;
    line-height: 1.4;
}

.capability-tag {
    margin-top: 0.42rem;

    color: var(--cyan);

    font-size: 0.47rem;
    font-weight: 850;

    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.architecture-panel {
    display: flex;
    align-items: center;
    justify-content: center;

    gap: 0.5rem;

    margin-top: 0.6rem;
    padding: 0.55rem 0.7rem;

    border: 1px solid var(--border);
    border-radius: 9px;

    background:
        rgba(12, 25, 40, 0.60);
}

.arch-node {
    color: var(--text-soft);

    font-size: 0.52rem;
    font-weight: 750;

    white-space: nowrap;
}

.arch-arrow {
    color: var(--cyan);

    font-size: 0.58rem;
}


/* ============================================================
   FOOTER
============================================================ */

.product-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 1rem;

    margin-top: 0.9rem;
    padding-top: 0.55rem;

    border-top: 1px solid var(--border);

    color: var(--muted-2);

    font-size: 0.48rem;
}

.footer-pills {
    display: flex;
    gap: 0.28rem;
    flex-wrap: wrap;
}

.footer-pill {
    padding: 0.17rem 0.32rem;

    border: 1px solid var(--border);
    border-radius: 999px;

    background:
        rgba(15, 28, 45, 0.42);

    color: var(--muted);
}


/* ============================================================
   STREAMLIT
============================================================ */

div[data-testid="stAlert"] {
    border: 1px solid var(--border);
    border-radius: 9px;

    background:
        rgba(15, 28, 45, 0.75);
}


/* ============================================================
   RESPONSIVE
============================================================ */

@media (max-width: 1200px) {
    .block-container {
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    .sidebar-system-row {
        grid-template-columns: 1fr;
        gap: 0.12rem;
    }

    .architecture-panel {
        flex-wrap: wrap;
    }
}

@media (max-width: 900px) {
    .risk-layout {
        grid-template-columns: 1fr;
        gap: 0.35rem;
    }

    .model-meta-grid {
        grid-template-columns: 1fr;
    }

    .product-footer {
        flex-direction: column;
        align-items: flex-start;
    }
}

@media (max-width: 700px) {
    .block-container {
        padding-top: 0.7rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .page-title {
        font-size: 1.6rem;
    }

    .component-header {
        flex-direction: column;
        align-items: flex-start;
    }

    .product-path {
        flex-wrap: wrap;
    }
}

</style>
"""

st.markdown(
    CSS,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================


def render_html(markup: str) -> None:
    cleaned = "\n".join(
        line.strip()
        for line in markup.strip().splitlines()
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True,
    )


def esc(value: Any) -> str:
    if value is None:
        return ""

    return html.escape(
        str(value)
    )


def readable_component_type(
    value: Any,
) -> str:
    return str(
        value or "UNKNOWN"
    ).replace(
        "_",
        " ",
    )


def readable_date(
    value: Any,
) -> str:
    text = str(
        value or "Unknown date"
    )

    if "T" in text:
        return text.split("T")[0]

    if " " in text:
        return text.split(" ")[0]

    return text


@st.cache_resource
def get_agent() -> MaintenanceAgent:
    return MaintenanceAgent()


# ============================================================
# UI BUILDERS
# ============================================================


def metric_card(
    label: str,
    value: str,
    meta: str,
    value_class: str = "",
) -> str:
    return f"""
    <div class="metric-card">

        <div class="metric-label">
            {esc(label)}
        </div>

        <div class="metric-value {value_class}">
            {esc(value)}
        </div>

        <div class="metric-meta">
            {esc(meta)}
        </div>

    </div>
    """


def capability_card(
    index: str,
    title: str,
    text: str,
    tag: str,
) -> str:
    return f"""
    <div class="capability-card">

        <div class="capability-top">

            <div class="capability-index">
                {esc(index)}
            </div>

            <div class="capability-status">
                ● READY
            </div>

        </div>

        <div class="capability-title">
            {esc(title)}
        </div>

        <div class="capability-text">
            {esc(text)}
        </div>

        <div class="capability-tag">
            {esc(tag)}
        </div>

    </div>
    """


def risk_panel(
    risk: dict[str, Any],
) -> str:
    score = float(
        risk.get(
            "risk_score",
            0.0,
        )
    )

    threshold = float(
        risk.get(
            "threshold",
            0.0,
        )
    )

    above = bool(
        risk.get(
            "above_threshold",
            False,
        )
    )

    score_pct = max(
        0.0,
        min(
            score * 100.0,
            100.0,
        ),
    )

    threshold_pct = max(
        0.0,
        min(
            threshold * 100.0,
            100.0,
        ),
    )

    if above:
        state = "Review threshold exceeded"
        state_class = "risk-state-alert"
    else:
        state = "Below review threshold"
        state_class = "risk-state-safe"

    model_type = risk.get(
        "model_type",
        "Unknown",
    )

    horizon = risk.get(
        "prediction_horizon_days",
        "—",
    )

    return f"""
    <div class="panel">

        <div class="risk-layout">

            <div>

                <div class="panel-title">
                    Predictive Risk Signal
                </div>

                <div class="risk-number">
                    {score:.4f}
                </div>

                <div class="{state_class}">
                    {esc(state)}
                </div>

            </div>

            <div>

                <div class="panel-subtitle">
                    Model risk score compared with the
                    validation-selected review threshold
                </div>

                <div class="risk-track">

                    <div
                        class="risk-fill"
                        style="width:{score_pct:.2f}%">
                    </div>

                    <div
                        class="risk-threshold"
                        style="left:{threshold_pct:.2f}%">
                    </div>

                    <div
                        class="risk-threshold-label"
                        style="left:{threshold_pct:.2f}%">
                        threshold {threshold:.4f}
                    </div>

                </div>

                <div class="risk-scale">
                    <span>0.0</span>
                    <span>Risk score</span>
                    <span>1.0</span>
                </div>

                <div class="model-meta-grid">

                    <div>
                        <div class="model-meta-label">
                            Model
                        </div>

                        <div class="model-meta-value">
                            {esc(model_type)}
                        </div>
                    </div>

                    <div>
                        <div class="model-meta-label">
                            Horizon
                        </div>

                        <div class="model-meta-value">
                            {esc(horizon)} days
                        </div>
                    </div>

                    <div>
                        <div class="model-meta-label">
                            Output Contract
                        </div>

                        <div class="model-meta-value">
                            Uncalibrated risk score
                        </div>
                    </div>

                </div>

            </div>

        </div>

    </div>
    """


def history_markup(
    history: list[dict[str, Any]],
) -> str:
    if not history:
        return """
        <div class="timeline-note">
            No maintenance events were returned for this component.
        </div>
        """

    cards: list[str] = []

    for event in history:
        date = readable_date(
            event.get("timestamp")
            or event.get("event_date")
            or event.get("maintenance_date")
        )

        event_type = (
            event.get("event_type")
            or "Maintenance event"
        )

        work_order = (
            event.get("work_order_id")
            or "No work order"
        )

        fault_code = (
            event.get("fault_code")
            or "No fault code"
        )

        note = (
            event.get("technician_note")
            or ""
        )

        corrective_action = (
            event.get("corrective_action")
            or ""
        )

        cards.append(
            f"""
            <div class="timeline-card">

                <div class="timeline-date">
                    {esc(date)}
                </div>

                <div class="timeline-action">
                    {esc(event_type)}
                </div>

                <div class="timeline-meta">
                    {esc(work_order)}
                    ·
                    {esc(fault_code)}
                </div>

                <div class="timeline-note">
                    {esc(note)}
                </div>

                <div class="timeline-action-detail">
                    Corrective action:
                    {esc(corrective_action)}
                </div>

            </div>
            """
        )

    return "".join(
        cards
    )


def evidence_markup(
    evidence: list[dict[str, Any]],
) -> str:
    if not evidence:
        return """
        <div class="evidence-content">
            No maintenance evidence was retrieved.
        </div>
        """

    cards: list[str] = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        score = float(
            item.get(
                "score",
                0.0,
            )
        )

        section = item.get(
            "section",
            "Evidence",
        )

        title = item.get(
            "title",
            "",
        )

        document_id = item.get(
            "document_id",
            "Synthetic knowledge base",
        )

        content = item.get(
            "content",
            "",
        )

        content_lines = [
            line.strip()
            for line in str(content).splitlines()
            if line.strip()
        ]

        filtered_lines = [
            line
            for line in content_lines
            if not (
                line.startswith("Title:")
                or line.startswith("Component Type:")
                or line.startswith("Section:")
            )
        ]

        display_content = " ".join(
            filtered_lines
        )

        cards.append(
            f"""
            <div class="evidence-card">

                <div class="evidence-top">

                    <div class="evidence-index">
                        EVIDENCE {index:02d}
                    </div>

                    <div class="evidence-score">
                        relevance {score:.3f}
                    </div>

                </div>

                <div class="evidence-section">
                    {esc(section)}
                </div>

                <div class="evidence-source">
                    {esc(document_id)}
                    ·
                    {esc(title)}
                </div>

                <div class="evidence-content">
                    {esc(display_content)}
                </div>

            </div>
            """
        )

    return "".join(
        cards
    )


def investigation_brief(
    result: dict[str, Any],
) -> str:
    risk = result.get(
        "risk",
        {},
    )

    summary = result.get(
        "summary",
        {},
    )

    component_id = result.get(
        "component_id",
        "Unknown",
    )

    component_type = readable_component_type(
        result.get(
            "component_type"
        )
    )

    score = float(
        risk.get(
            "risk_score",
            0.0,
        )
    )

    threshold = float(
        risk.get(
            "threshold",
            0.0,
        )
    )

    above = bool(
        risk.get(
            "above_threshold",
            False,
        )
    )

    history_count = summary.get(
        "maintenance_event_count_returned",
        len(
            result.get(
                "maintenance_history",
                [],
            )
        ),
    )

    evidence_count = summary.get(
        "evidence_count",
        len(
            result.get(
                "evidence",
                [],
            )
        ),
    )

    evidence_sections = summary.get(
        "evidence_sections",
        [],
    )

    state_text = (
        "exceeds the configured review threshold"
        if above
        else "remains below the configured review threshold"
    )

    sections_text = (
        ", ".join(
            str(section)
            for section in evidence_sections
        )
        if evidence_sections
        else "retrieved maintenance evidence"
    )

    return (
        f"{component_id} ({component_type}) produced a predictive "
        f"risk score of {score:.4f} against a review threshold of "
        f"{threshold:.4f} and {state_text}. "
        f"The investigation returned {history_count} maintenance "
        f"event(s) and {evidence_count} evidence chunk(s), including "
        f"{sections_text}. The result is synthetic decision-support "
        f"output and is not an autonomous maintenance decision."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">

            <div class="brand-row">

                <div class="brand-mark">
                    A
                </div>

                <div>

                    <div class="brand-name">
                        AEROMAINTAIN
                    </div>

                    <div class="brand-subtitle">
                        Intelligence Platform
                    </div>

                </div>

            </div>

        </div>

        <div class="sidebar-label">
            Intelligence Layers
        </div>

        <div class="layer-item">
            <span class="layer-index">01</span>
            Predictive Risk
        </div>

        <div class="layer-item">
            <span class="layer-index">02</span>
            Maintenance History
        </div>

        <div class="layer-item">
            <span class="layer-index">03</span>
            Evidence Retrieval
        </div>

        <div class="layer-active">
            <span class="layer-index">04</span>
            Agent Investigation
        </div>

        <div class="sidebar-label">
            Platform Status
        </div>

        <div class="sidebar-system">

            <div class="sidebar-system-row">

                <span>
                    Predictive ML
                </span>

                <span class="sidebar-ready">
                    <span class="status-dot"></span>
                    READY
                </span>

            </div>

            <div class="sidebar-system-row">

                <span>
                    LSA Retrieval
                </span>

                <span class="sidebar-ready">
                    <span class="status-dot"></span>
                    READY
                </span>

            </div>

            <div class="sidebar-system-row">

                <span>
                    Maintenance Agent
                </span>

                <span class="sidebar-ready">
                    <span class="status-dot"></span>
                    READY
                </span>

            </div>

        </div>

        <div class="sidebar-label">
            Environment
        </div>

        <div class="sidebar-environment">
            Synthetic fleet<br>
            Local model inference<br>
            Local semantic retrieval<br>
            Evidence-driven workflow
        </div>
        """
    )


# ============================================================
# PRODUCT BAR
# ============================================================

render_html(
    """
    <div class="product-bar">

        <div class="product-path">

            <span>
                AEROMAINTAIN
            </span>

            <span class="product-separator">
                /
            </span>

            <span class="product-path-active">
                COMPONENT INTELLIGENCE
            </span>

        </div>

        <div class="live-badge">
            <span class="status-dot"></span>
            System Operational
        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="eyebrow">
            Aerospace Reliability Intelligence
        </div>

        <h1 class="page-title">
            Maintenance Intelligence
        </h1>

        <div class="page-subtitle">
            From component telemetry to a traceable maintenance
            investigation — combining predictive risk, event history,
            and retrieved engineering evidence.
        </div>

    </div>
    """
)


# ============================================================
# INVESTIGATION CONSOLE
# ============================================================

render_html(
    """
    <div class="section-kicker">
        Investigation Console
    </div>

    <div class="section-heading">
        Analyze a component
    </div>

    <div class="section-description">
        Run the end-to-end AeroMaintain intelligence workflow.
    </div>
    """
)

input_col, button_col, spacer_col = st.columns(
    [4.3, 1.45, 0.65],
    vertical_alignment="bottom",
)

with input_col:
    component_id = st.text_input(
        "Component ID",
        value="CMP-00196",
        placeholder="e.g. CMP-00196",
        label_visibility="collapsed",
    )

with button_col:
    analyze_clicked = st.button(
        "RUN ANALYSIS →",
        use_container_width=True,
    )

render_html(
    """
    <div class="sample-hint">
        Demo component · CMP-00196
    </div>
    """
)


# ============================================================
# STATE
# ============================================================

if "investigation" not in st.session_state:
    st.session_state.investigation = None


# ============================================================
# EXECUTION
# ============================================================

if analyze_clicked:
    clean_component_id = component_id.strip()

    if not clean_component_id:
        st.warning(
            "Enter a component ID before running analysis."
        )

    else:
        try:
            with st.spinner(
                "Running predictive risk, maintenance history, "
                "and evidence retrieval..."
            ):
                agent = get_agent()

                st.session_state.investigation = (
                    agent.investigate_component(
                        component_id=clean_component_id,
                        top_k_documents=3,
                        history_limit=5,
                    )
                )

        except ValueError as exc:
            st.session_state.investigation = None

            st.error(
                str(exc)
            )

        except FileNotFoundError as exc:
            st.session_state.investigation = None

            st.error(
                "Predictive model artifact unavailable. "
                f"{exc}"
            )

        except Exception as exc:
            st.session_state.investigation = None

            st.error(
                "Investigation could not be completed. "
                f"{exc}"
            )


# ============================================================
# MAIN EXPERIENCE
# ============================================================

result = st.session_state.investigation


if result:
    risk = result.get(
        "risk",
        {},
    )

    history = result.get(
        "maintenance_history",
        [],
    )

    evidence = result.get(
        "evidence",
        [],
    )

    result_component_id = result.get(
        "component_id",
        component_id,
    )

    component_type = readable_component_type(
        result.get(
            "component_type"
        )
    )

    score = float(
        risk.get(
            "risk_score",
            0.0,
        )
    )

    threshold = float(
        risk.get(
            "threshold",
            0.0,
        )
    )

    horizon = risk.get(
        "prediction_horizon_days",
        "—",
    )

    above_threshold = bool(
        risk.get(
            "above_threshold",
            False,
        )
    )

    if above_threshold:
        state_label = "REVIEW SIGNAL"
        state_class = "metric-alert"
    else:
        state_label = "BELOW THRESHOLD"
        state_class = "metric-safe"

    render_html(
        f"""
        <div class="component-header">

            <div class="component-main">

                <div class="component-id">
                    {esc(result_component_id)}
                </div>

                <div class="component-caption">
                    Component investigation
                </div>

            </div>

            <div class="component-type">
                {esc(component_type)}
            </div>

        </div>
        """
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(
        4,
        gap="small",
    )

    with metric_1:
        render_html(
            metric_card(
                "Risk Score",
                f"{score:.4f}",
                "Predictive ML signal",
                "metric-accent",
            )
        )

    with metric_2:
        render_html(
            metric_card(
                "Review Threshold",
                f"{threshold:.4f}",
                state_label,
                state_class,
            )
        )

    with metric_3:
        render_html(
            metric_card(
                "Prediction Horizon",
                f"{horizon} DAYS",
                "Forward-looking window",
            )
        )

    with metric_4:
        render_html(
            metric_card(
                "Maintenance Events",
                str(len(history)),
                "Returned event history",
            )
        )

    render_html(
        """
        <div style="height:0.45rem;"></div>

        <div class="section-kicker">
            Predictive Intelligence
        </div>

        <div class="section-heading">
            Risk profile
        </div>
        """
    )

    render_html(
        risk_panel(
            risk
        )
    )

    render_html(
        """
        <div style="height:0.55rem;"></div>
        """
    )

    history_col, evidence_col = st.columns(
        [0.92, 1.08],
        gap="large",
    )

    with history_col:
        render_html(
            """
            <div class="section-kicker">
                Operational Context
            </div>

            <div class="section-heading">
                Maintenance history
            </div>
            """
        )

        render_html(
            f"""
            <div class="panel">

                <div class="panel-title">
                    Component Event Timeline
                </div>

                <div class="panel-subtitle">
                    Historical maintenance context returned by the agent
                </div>

                {history_markup(history)}

            </div>
            """
        )

    with evidence_col:
        render_html(
            """
            <div class="section-kicker">
                Retrieval Intelligence
            </div>

            <div class="section-heading">
                Engineering evidence
            </div>
            """
        )

        render_html(
            f"""
            <div class="panel">

                <div class="panel-title">
                    Retrieved Maintenance Evidence
                </div>

                <div class="panel-subtitle">
                    LSA-ranked synthetic knowledge grounding
                    this investigation
                </div>

                {evidence_markup(evidence)}

            </div>
            """
        )

    render_html(
        """
        <div style="height:0.55rem;"></div>

        <div class="section-kicker">
            Agent Output
        </div>

        <div class="section-heading">
            Investigation brief
        </div>
        """
    )

    render_html(
        f"""
        <div class="brief">

            <div class="brief-label">
                ◈ Evidence-Driven Assessment
            </div>

            <div class="brief-text">
                {esc(investigation_brief(result))}
            </div>

        </div>
        """
    )


else:
    render_html(
        """
        <div style="height:0.55rem;"></div>

        <div class="section-kicker">
            Intelligence Stack
        </div>

        <div class="section-heading">
            One investigation. Four intelligence layers.
        </div>

        <div class="section-description">
            The workflow combines predictive modeling,
            operational context, semantic retrieval,
            and deterministic agent orchestration.
        </div>
        """
    )

    cap_1, cap_2, cap_3, cap_4 = st.columns(
        4,
        gap="small",
    )

    with cap_1:
        render_html(
            capability_card(
                "01",
                "Predictive Risk",
                (
                    "Leakage-aware temporal modeling produces "
                    "a 30-day component risk signal."
                ),
                "Predictive ML",
            )
        )

    with cap_2:
        render_html(
            capability_card(
                "02",
                "Maintenance Context",
                (
                    "Historical work orders and corrective "
                    "actions provide operational context."
                ),
                "Event History",
            )
        )

    with cap_3:
        render_html(
            capability_card(
                "03",
                "Evidence Retrieval",
                (
                    "Local latent-semantic retrieval ranks "
                    "synthetic maintenance knowledge."
                ),
                "LSA Retrieval",
            )
        )

    with cap_4:
        render_html(
            capability_card(
                "04",
                "Agent Investigation",
                (
                    "Risk, history, and evidence are assembled "
                    "into one traceable assessment."
                ),
                "Orchestration",
            )
        )

    render_html(
        """
        <div class="architecture-panel">

            <span class="arch-node">
                TELEMETRY
            </span>

            <span class="arch-arrow">
                →
            </span>

            <span class="arch-node">
                FEATURES
            </span>

            <span class="arch-arrow">
                →
            </span>

            <span class="arch-node">
                PREDICTIVE ML
            </span>

            <span class="arch-arrow">
                +
            </span>

            <span class="arch-node">
                HISTORY
            </span>

            <span class="arch-arrow">
                +
            </span>

            <span class="arch-node">
                LSA RETRIEVAL
            </span>

            <span class="arch-arrow">
                →
            </span>

            <span class="arch-node">
                AGENT
            </span>

            <span class="arch-arrow">
                →
            </span>

            <span class="arch-node">
                INVESTIGATION
            </span>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="product-footer">

        <div>
            Synthetic portfolio environment ·
            Decision-support only ·
            Not OEM-approved maintenance guidance
        </div>

        <div class="footer-pills">

            <span class="footer-pill">
                Predictive ML
            </span>

            <span class="footer-pill">
                LSA
            </span>

            <span class="footer-pill">
                RAG
            </span>

            <span class="footer-pill">
                Agent
            </span>

            <span class="footer-pill">
                FastAPI
            </span>

        </div>

    </div>
    """
)