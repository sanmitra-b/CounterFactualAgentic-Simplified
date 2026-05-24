"""
Layer 6 — CFASimplified Risk Intelligence Dashboard
Streamlit app that reads all pipeline outputs and presents them
in a clean, dark-themed analytical UI.

"""

import json
import ast
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Risk Intelligence · CFASimplified",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")

SEVERITY_COLOR = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MEDIUM":   "#eab308",
    "LOW":      "#22c55e",
}
SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

FRED_LABELS = {
    "UNRATE": "Unemployment Rate (%)",
    "ICSA":   "Initial Jobless Claims (000s)",
    "PAYEMS": "Total Nonfarm Payrolls (000s)",
    "JTSJOL": "Job Openings (000s)",
}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark analytical theme, IBM Plex Mono + Syne
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}
.stApp { background-color: #0a0a0f; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    color: #60a5fa !important;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Page header ── */
.page-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.page-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    color: #f1f5f9;
}
.metric-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* ── Risk cards ── */
.risk-card {
    background: #111827;
    border-left: 4px solid #334155;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
}
.risk-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.4rem;
}
.risk-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #64748b;
    margin-bottom: 0.6rem;
}
.risk-evidence {
    font-size: 0.8rem;
    color: #94a3b8;
    font-style: italic;
    border-left: 2px solid #1e293b;
    padding-left: 0.6rem;
    margin-top: 0.5rem;
}
.confidence-bar-outer {
    background: #1e293b;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 0.5rem;
}
.confidence-bar-inner {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}

/* ── Solution cards ── */
.solution-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.solution-action {
    font-size: 0.82rem;
    color: #34d399;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.3rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
    color: #e2e8f0;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Pill badges ── */
.badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-right: 0.3rem;
}
.badge-critical { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-high     { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.badge-medium   { background: rgba(234,179,8,0.15);  color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
.badge-low      { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-neutral  { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }
.badge-blue     { background: rgba(96,165,250,0.12); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }

/* ── Source table ── */
.stDataFrame { background: #111827 !important; }

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    color: #64748b !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
}

/* ── Hide streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
</style>
"""

if hasattr(st, "html"):
    st.html(CUSTOM_CSS)
else:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_json(path: Path):
    """Load JSON, return None if missing."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_input_bundle(path: Path):
    """Load layer1 bundle and parse stringified lists."""
    raw = load_json(path)
    if raw is None:
        return None
    def _parse(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return []
        return val or []
    return {
        "domain":             raw.get("domain", "ai_job_risk"),
        "fetched_at":         raw.get("fetched_at", ""),
        "completeness_score": float(raw.get("completeness_score", 0)),
        "news":               _parse(raw.get("news", [])),
        "social":             _parse(raw.get("social", [])),
        # Some pipeline runs still serialize jobs under `weather`.
        "jobs":               _parse(raw.get("jobs", raw.get("weather", []))),
        "stocks":             _parse(raw.get("stocks", [])),
        "weather":            _parse(raw.get("weather", [])),
        "commodities":        _parse(raw.get("commodities", [])),
        "errors":             _parse(raw.get("errors", [])),
    }


@st.cache_data(show_spinner=False)
def load_risk_report(path: Path):
    raw = load_json(path)
    return raw or {}


@st.cache_data(show_spinner=False)
def load_counterfactuals(path: Path):
    raw = load_json(path)
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return raw.get("risks", raw.get("results", []))


@st.cache_data(show_spinner=False)
def load_solution_bundle(path: Path):
    return load_json(path) or {}


@st.cache_data(show_spinner=False)
def load_enriched_bundle(path: Path):
    raw = load_json(path)
    if raw is None:
        return {"news": [], "social": []}
    return {
        "news": raw.get("news", []) or [],
        "social": raw.get("social", []) or [],
    }


def build_unified_sources_df(bundle_data: dict | None, enriched_data: dict | None) -> pd.DataFrame:
    """Build a single table across all source types with sentiment labels where available."""
    rows = []
    bundle_data = bundle_data or {}
    enriched_data = enriched_data or {"news": [], "social": []}

    for item in enriched_data.get("news", []):
        rows.append(
            {
                "source_type": "news",
                "source": item.get("source", "news"),
                "timestamp": item.get("published_at", ""),
                "title": item.get("title", ""),
                "text": item.get("body", ""),
                "sentiment": (item.get("sentiment", {}) or {}).get("label", "unknown"),
                "url": item.get("url", ""),
            }
        )

    for item in enriched_data.get("social", []):
        rows.append(
            {
                "source_type": "social",
                "source": item.get("platform", "social"),
                "timestamp": item.get("created_at", ""),
                "title": "",
                "text": item.get("text", ""),
                "sentiment": (item.get("sentiment", {}) or {}).get("label", "unknown"),
                "url": "",
            }
        )

    # Include other Layer 1 sources (financial/weather/etc.) with N/A sentiment.
    for item in (bundle_data.get("commodities", []) or []):
        rows.append(
            {
                "source_type": "financial",
                "source": item.get("commodity", "financial"),
                "timestamp": item.get("fetched_at", ""),
                "title": item.get("commodity", ""),
                "text": f"price={item.get('price', 'NA')} change_pct={item.get('change_pct', 'NA')}",
                "sentiment": "n/a",
                "url": "",
            }
        )

    for item in (bundle_data.get("stocks", []) or []):
        rows.append(
            {
                "source_type": "financial",
                "source": item.get("ticker", "stock"),
                "timestamp": item.get("fetched_at", ""),
                "title": item.get("ticker", ""),
                "text": f"price={item.get('price', 'NA')} change_pct={item.get('change_pct', 'NA')}",
                "sentiment": "n/a",
                "url": "",
            }
        )

    for item in (bundle_data.get("weather", []) or []):
        rows.append(
            {
                "source_type": "weather",
                "source": item.get("location", "weather"),
                "timestamp": item.get("fetched_at", ""),
                "title": item.get("location", ""),
                "text": str(item),
                "sentiment": "n/a",
                "url": "",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["source_type", "source", "timestamp", "title", "text", "sentiment", "url"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp", ascending=False)


def normalize_counterfactual_results(raw_results: list, risk_rows: list) -> list:
    """Normalize Layer 4 output into a flat list suitable for charts/cards."""
    severity_by_title = {r.get("title", ""): r.get("severity", "LOW") for r in risk_rows}
    normalized = []
    for row in raw_results or []:
        base = row.get("best_intervention") if isinstance(row, dict) and row.get("best_intervention") else row
        intervention = (base.get("intervention", {}) or {}) if isinstance(base, dict) else {}
        risk_title = base.get("risk_title", row.get("risk_title", row.get("title", "Risk")))
        normalized.append(
            {
                "risk_title": risk_title,
                "severity": base.get("severity", severity_by_title.get(risk_title, "LOW")),
                "causal_var": base.get("causal_var", intervention.get("variable", "—")),
                "ite_mean": float(base.get("ite_mean", 0.0) or 0.0),
                "p_improve": float(base.get("p_improve", base.get("p_improvement", base.get("probability_of_improvement", 0.0))) or 0.0),
            }
        )
    return normalized


def parse_solution_mappings(solution_bundle: dict) -> list:
    """Map Layer 5 `mappings` output into the dashboard's card format."""
    mappings = (solution_bundle or {}).get("mappings", [])
    risks = []
    for m in mappings:
        mitigations = []
        for s in m.get("top_mitigations", []):
            action_steps = s.get("action_steps", [])
            action = " | ".join(action_steps[:2]) if action_steps else s.get("description", "")
            mitigations.append(
                {
                    "type": s.get("intervention_type", "mitigation"),
                    "action": action,
                    "confidence": float(s.get("confidence_score", 0.0) or 0.0),
                    "cosine_sim": float(s.get("cosine_similarity", 0.0) or 0.0),
                    "category": s.get("risk_category", ""),
                }
            )

        risks.append(
            {
                "title": m.get("risk_title", ""),
                "severity": m.get("severity_label", "LOW"),
                "category": m.get("risk_category", ""),
                "solutions": mitigations,
            }
        )
    return risks


# ─────────────────────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def render_badge(text: str, style: str = "neutral"):
    return f'<span class="badge badge-{style}">{text}</span>'


def severity_style(sev: str) -> str:
    return sev.lower() if sev.lower() in ("critical","high","medium","low") else "neutral"


def conf_bar(conf: float, color: str = "#3b82f6") -> str:
    pct = int(conf * 100)
    return (
        f'<div class="confidence-bar-outer">'
        f'<div class="confidence-bar-inner" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
    )


def fred_description(code: str) -> str:
    return FRED_LABELS.get(code, code)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">⚠ CFASimplified</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#475569;margin-bottom:1rem;">AI RISK INTELLIGENCE DASHBOARD</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Sources", "Risk Report", "Counterfactuals", "Solutions", "Labor Indicators"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#334155;">DATA FILES</div>', unsafe_allow_html=True)

    file_status = {
        "Layer 1 · Input Bundle":    (DATA_DIR / "risk_input_bundle.json").exists(),
        "Layer 2 · Enriched Bundle": (DATA_DIR / "enriched_risk_bundle.json").exists(),
        "Layer 3 · Risk Report":     (DATA_DIR / "risk_report.json").exists(),
        "Layer 4 · Counterfactuals": (DATA_DIR / "counterfactual_bundle.json").exists(),
        "Layer 5 · Solutions":       (DATA_DIR / "solution_mapping_report.json").exists(),
    }
    for label, exists in file_status.items():
        dot = "🟢" if exists else "🔴"
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#475569;margin:0.2rem 0;">'
            f'{dot} {label}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:0.58rem;color:#334155;">'
        'Layer 6 · No external API calls<br>All data from pipeline outputs</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

bundle      = load_input_bundle(DATA_DIR / "risk_input_bundle.json")
enriched    = load_enriched_bundle(DATA_DIR / "enriched_risk_bundle.json")
risk_report = load_risk_report(DATA_DIR / "risk_report.json")
cf_results  = load_counterfactuals(DATA_DIR / "counterfactual_bundle.json")
sol_bundle  = load_solution_bundle(DATA_DIR / "solution_mapping_report.json")

# Derived counts (safe fallbacks)
news_items    = bundle["news"]       if bundle else []
social_items  = bundle["social"]     if bundle else []
jobs_items    = bundle["jobs"]       if bundle else []
commodities   = bundle["commodities"] if bundle else []
completeness  = bundle["completeness_score"] if bundle else risk_report.get("layer1_completeness", 0.75)
total_signals = len(news_items) + len(social_items) + len(commodities) + len(jobs_items)
top_risks     = risk_report.get("top_risks", [])
soft_risks    = risk_report.get("soft_risks", [])
agg_sentiment = risk_report.get("aggregate_sentiment", "neutral")
avg_rel       = float(risk_report.get("avg_reliability", 0.80))
sources_df    = build_unified_sources_df(bundle, enriched)
cf_results_norm = normalize_counterfactual_results(cf_results, top_risks)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown('<div class="page-header">AI Job Risk Intelligence</div>', unsafe_allow_html=True)
    fetched = bundle["fetched_at"] if bundle else risk_report.get("analysed_at", "")
    st.markdown(f'<div class="page-sub">DOMAIN: AI_JOB_RISK · LAST RUN: {fetched[:10] if fetched else "—"} · MODEL: {risk_report.get("model_used","llama-3.3-70b-versatile")}</div>', unsafe_allow_html=True)

    # ── KPI row ──
    kpi_cols = st.columns(5)
    kpis = [
        ("Total Signals",    f"{total_signals:,}", f"{len(news_items)} news · {len(social_items)} social · {len(commodities)} financial · {len(jobs_items)} jobs"),
        ("Layer 1 Complete", f"{int(completeness*100)}%",   "data pipeline coverage"),
        ("Avg Reliability",  f"{avg_rel:.2f}",              "source trust score"),
        ("Risks Identified", str(len(top_risks)),           f"{len(soft_risks)} emerging"),
        ("Sentiment",        agg_sentiment.upper(),         "aggregate signal tone"),
    ]
    for col, (label, value, delta) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-delta">{delta}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Risk severity overview ──
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown('<div class="section-header">Top Risks — Severity Overview</div>', unsafe_allow_html=True)
        show_p30 = any(float(r.get("probability_next_30d", r.get("probability_30d", 0)) or 0) > 0 for r in top_risks)
        for r in top_risks:
            sev   = r.get("severity", "LOW")
            conf  = r.get("confidence", 0)
            p30   = r.get("probability_next_30d", r.get("probability_30d", 0))
            color = SEVERITY_COLOR.get(sev, "#64748b")
            p30_html = (
                f'<div><span style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#64748b;">P(30d)</span><br>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#94a3b8;">{int(p30*100)}%</span></div>'
                if show_p30 else ""
            )
            st.markdown(
                f'<div class="risk-card" style="border-left-color:{color};">'
                f'{render_badge(sev, severity_style(sev))} '
                f'<span class="risk-title">{SEVERITY_EMOJI.get(sev,"")} {r.get("title","")}</span><br>'
                f'<span class="risk-meta">Category: {r.get("category","")}</span>'
                f'<div style="display:flex;gap:1.5rem;margin-top:0.5rem;">'
                f'<div><span style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#64748b;">CONFIDENCE</span><br>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{color};">{int(conf*100)}%</span></div>'
                f'{p30_html}'
                f'</div>'
                f'{conf_bar(conf, color)}'
                f'<div class="risk-evidence">Evidence: {r.get("evidence","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown('<div class="section-header">Confidence Distribution</div>', unsafe_allow_html=True)
        if top_risks:
            fig = go.Figure(go.Bar(
                x=[r.get("confidence",0)*100 for r in top_risks],
                y=[r.get("title","")[:30]+"…" if len(r.get("title",""))>30 else r.get("title","") for r in top_risks],
                orientation="h",
                marker=dict(
                    color=[SEVERITY_COLOR.get(r.get("severity","LOW"), "#64748b") for r in top_risks],
                    line=dict(width=0),
                ),
                text=[f'{int(r.get("confidence",0)*100)}%' for r in top_risks],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10, color="#94a3b8"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0,120]),
                yaxis=dict(showgrid=False, tickfont=dict(family="IBM Plex Mono", size=10, color="#94a3b8")),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Signal Sources</div>', unsafe_allow_html=True)
        src_counts = {
            "News":       len(news_items),
            "Social":     len(social_items),
            "Financial":  len(commodities),
        }
        fig2 = go.Figure(go.Pie(
            labels=list(src_counts.keys()),
            values=list(src_counts.values()),
            hole=0.65,
            marker=dict(colors=["#3b82f6", "#8b5cf6", "#34d399"]),
            textinfo="none",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=180,
            showlegend=True,
            legend=dict(
                font=dict(family="IBM Plex Mono", size=9, color="#94a3b8"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Geo Heatmap</div>', unsafe_allow_html=True)
        geo_coords = {
            "US": (37.0902, -95.7129),
            "USA": (37.0902, -95.7129),
            "India": (20.5937, 78.9629),
            "UK": (55.3781, -3.4360),
            "Germany": (51.1657, 10.4515),
            "Global": (0.0, 0.0),
            "New York": (40.7128, -74.0060),
        }
        geo_counts = {}
        for risk in top_risks:
            for place in risk.get("geo", []) if isinstance(risk.get("geo", []), list) else [risk.get("geo", "")]:
                if place:
                    geo_counts[place] = geo_counts.get(place, 0) + 1

        for item in (enriched.get("news", []) or []) + (enriched.get("social", []) or []):
            for tag in item.get("geo_tags", []) or []:
                geo_counts[tag] = geo_counts.get(tag, 0) + 1

        heat_rows = []
        for place, count in geo_counts.items():
            if place in geo_coords:
                lat, lon = geo_coords[place]
                heat_rows.append({"place": place, "count": count, "lat": lat, "lon": lon})

        if heat_rows:
            df_geo = pd.DataFrame(heat_rows)
            fig_geo = px.scatter_geo(
                df_geo,
                lat="lat",
                lon="lon",
                size="count",
                color="count",
                hover_name="place",
                color_continuous_scale="Blues",
                projection="natural earth",
            )
            fig_geo.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                geo=dict(bgcolor="rgba(0,0,0,0)", showcountries=True, countrycolor="#334155"),
                height=220,
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.markdown('<div style="font-size:0.75rem;color:#64748b;">No geo tags available in current outputs.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SOURCES
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Sources":
    st.markdown('<div class="page-header">Signal Sources</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">ALL RECORDS · FILTERABLE BY SOURCE TYPE · SENTIMENT LABELS FROM LAYER 2</div>', unsafe_allow_html=True)

    if sources_df.empty:
        st.info("No source records found. Run Layers 1 and 2 first.")
    else:
        all_types = sorted(sources_df["source_type"].dropna().unique().tolist())
        all_sentiments = sorted(sources_df["sentiment"].dropna().unique().tolist())

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_types = st.multiselect("Filter by source type", all_types, default=all_types)
        with col_f2:
            selected_sentiments = st.multiselect("Filter by sentiment", all_sentiments, default=all_sentiments)

        df_view = sources_df[
            sources_df["source_type"].isin(selected_types) & sources_df["sentiment"].isin(selected_sentiments)
        ].copy()

        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#64748b;margin-bottom:0.5rem;">'
            f'{len(df_view):,} records shown · {len(sources_df):,} total records</div>',
            unsafe_allow_html=True,
        )

        df_view["timestamp"] = df_view["timestamp"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            df_view[["source_type", "source", "sentiment", "timestamp", "title", "text"]],
            use_container_width=True,
            hide_index=True,
            height=520,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RISK REPORT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Risk Report":
    st.markdown('<div class="page-header">Risk Report</div>', unsafe_allow_html=True)
    analysed = risk_report.get("analysed_at","")[:10]
    model    = risk_report.get("model_used","llama-3.3-70b-versatile")
    st.markdown(f'<div class="page-sub">ANALYSED: {analysed} · MODEL: {model} · LAYER 3 OUTPUT</div>', unsafe_allow_html=True)

    # ── Top risks ──
    st.markdown('<div class="section-header">Top Identified Risks</div>', unsafe_allow_html=True)

    for r in top_risks:
        sev     = r.get("severity","LOW")
        conf    = r.get("confidence",0)
        p30     = r.get("probability_next_30d", r.get("probability_30d", 0))
        color   = SEVERITY_COLOR.get(sev,"#64748b")
        geo     = r.get("geo", [])
        geo_str = ", ".join(geo) if isinstance(geo, list) else str(geo)
        cause   = r.get("cause_effect", r.get("causal_chain","—"))
        action  = r.get("action","—")
        evidence= r.get("evidence","—")

        with st.expander(f'{SEVERITY_EMOJI.get(sev,"")}  #{r.get("rank","")}  {r.get("title","")}  —  {int(conf*100)}% confidence', expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            for col, label, val in [
                (c1, "SEVERITY",    f'<span style="color:{color};font-family:Syne,sans-serif;font-weight:700;">{sev}</span>'),
                (c2, "CONFIDENCE",  f'<span style="color:{color};font-family:Syne,sans-serif;font-weight:700;">{int(conf*100)}%</span>'),
                (c3, "P(30d)",      f'<span style="color:#94a3b8;font-family:Syne,sans-serif;font-weight:700;">{int(p30*100)}%</span>'),
                (c4, "GEO",         f'<span style="color:#60a5fa;font-size:0.8rem;">{geo_str}</span>'),
            ]:
                col.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#475569;margin-bottom:0.2rem;">{label}</div>'
                    f'{val}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f'{conf_bar(conf, color)}', unsafe_allow_html=True)

            st.markdown(f"""
<div style="margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
  <div>
    <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#475569;margin-bottom:0.3rem;">CATEGORY</div>
    <div style="font-size:0.82rem;">{r.get("category","—")}</div>
  </div>
  <div>
    <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#475569;margin-bottom:0.3rem;">CAUSE → EFFECT CHAIN</div>
    <div style="font-size:0.82rem;color:#a78bfa;">{cause}</div>
  </div>
  <div>
    <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#475569;margin-bottom:0.3rem;">RECOMMENDED ACTION</div>
    <div style="font-size:0.82rem;color:#34d399;">{action}</div>
  </div>
  <div>
    <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#475569;margin-bottom:0.3rem;">EVIDENCE</div>
    <div style="font-size:0.8rem;color:#94a3b8;font-style:italic;">{evidence}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Soft / emerging risks ──
    if soft_risks:
        st.markdown('<div class="section-header">Soft / Emerging Risks</div>', unsafe_allow_html=True)
        for sr in soft_risks:
            st.markdown(
                f'<div class="solution-card">'
                f'{render_badge(sr.get("category","—"), "neutral")}'
                f'<div style="font-size:0.82rem;margin-top:0.4rem;">{sr.get("description","—")}</div>'
                f'<div style="font-size:0.75rem;color:#64748b;font-style:italic;margin-top:0.3rem;">{sr.get("evidence","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COUNTERFACTUALS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Counterfactuals":
    st.markdown('<div class="page-header">Counterfactual Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">LAYER 4 · INDIVIDUAL TREATMENT EFFECTS · CAUSAL INFERENCE</div>', unsafe_allow_html=True)

    if not cf_results_norm:
        st.info("No counterfactual results found. Run Layer 4 first.")
        st.stop()

    # Load bundle to get mitigation efficiency metric
    cf_bundle = load_counterfactuals(DATA_DIR / "counterfactual_bundle.json")
    if isinstance(cf_bundle, dict) and "mitigation_efficiency" in cf_bundle:
        mitigation_eff = cf_bundle.get("mitigation_efficiency", 0)
        avg_delta = cf_bundle.get("avg_delta", 0)
        total_scenarios = cf_bundle.get("total_scenarios", 0)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">η_mitigation (Macro Efficiency)</div>'
                f'<div class="metric-value" style="color:#34d399;">{mitigation_eff*100:.2f}%</div>'
                f'<div class="metric-delta">avg % risk reduction</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Avg Delta P(30d)</div>'
                f'<div class="metric-value" style="color:#60a5fa;">{abs(avg_delta)*100:.2f}%</div>'
                f'<div class="metric-delta">risk reduction per scenario</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Total Scenarios</div>'
                f'<div class="metric-value" style="color:#a78bfa;">{total_scenarios}</div>'
                f'<div class="metric-delta">generated interventions</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        
        st.markdown("---")

    # ITE bar chart
    titles  = [r.get("risk_title", r.get("title",""))[:35] for r in cf_results_norm]
    ite_vals= [r.get("ite_mean", 0) for r in cf_results_norm]
    sevs    = [r.get("severity","LOW") for r in cf_results_norm]

    fig = go.Figure(go.Bar(
        x=ite_vals,
        y=titles,
        orientation="h",
        marker=dict(
            color=[SEVERITY_COLOR.get(s,"#64748b") for s in sevs],
            line=dict(width=0),
        ),
        text=[f'{v:.4f}' for v in ite_vals],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#94a3b8"),
    ))
    fig.update_layout(
        title=dict(text="Individual Treatment Effect (ITE) per Risk", font=dict(family="Syne",size=13,color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=60,t=40,b=10), height=220,
        xaxis=dict(tickfont=dict(family="IBM Plex Mono",size=9,color="#64748b"), showgrid=False, zeroline=True, zerolinecolor="#1e293b"),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono",size=9,color="#94a3b8"), showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">P(improve) Gauge</div>', unsafe_allow_html=True)
    selected_risk = st.selectbox(
        "Select risk",
        options=[r.get("risk_title", "Risk") for r in cf_results_norm],
    )
    gauge_row = next((r for r in cf_results_norm if r.get("risk_title") == selected_risk), None)
    if gauge_row:
        gauge_val = float(gauge_row.get("p_improve", 0.0) or 0.0) * 100
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                number={"suffix": "%", "font": {"color": "#34d399", "family": "Syne", "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                    "bar": {"color": "#34d399"},
                    "steps": [
                        {"range": [0, 50], "color": "#1f2937"},
                        {"range": [50, 75], "color": "#334155"},
                        {"range": [75, 100], "color": "#065f46"},
                    ],
                },
                title={"text": "Probability of Improvement", "font": {"color": "#94a3b8", "family": "IBM Plex Mono"}},
            )
        )
        gauge_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown('<div class="section-header">Intervention Traces</div>', unsafe_allow_html=True)

    for r in cf_results_norm:
        title      = r.get("risk_title", r.get("title","Risk"))
        sev        = r.get("severity","LOW")
        causal_var = r.get("causal_var","—")
        ite_mean   = r.get("ite_mean", 0)
        p_improve  = r.get("p_improve", r.get("p_improvement", 1.0))
        color      = SEVERITY_COLOR.get(sev,"#64748b")

        # P(improve) gauge
        p_pct = int(float(p_improve)*100)

        st.markdown(
            f'<div class="risk-card" style="border-left-color:{color};">'
            f'{render_badge(sev, severity_style(sev))}'
            f'<span class="risk-title" style="margin-left:0.5rem;">{title}</span>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:0.8rem;">'
            f'<div><div class="metric-label">CAUSAL VARIABLE</div><div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#a78bfa;">{causal_var}</div></div>'
            f'<div><div class="metric-label">ITE MEAN</div><div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:{color};">{ite_mean:.4f}</div></div>'
            f'<div><div class="metric-label">P(IMPROVE)</div><div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#34d399;">{p_pct}%</div></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Solutions":
    st.markdown('<div class="page-header">Risk Solutions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">LAYER 5 · RAG MITIGATION MAPPING · CHROMADB · all-MiniLM-L6-v2</div>', unsafe_allow_html=True)

    sol_risks = parse_solution_mappings(sol_bundle)

    if not sol_risks:
        sol_risks = sol_bundle.get("risks", [])

    if not sol_risks:
        st.info("No solutions found. Run Layer 5 first.")
        st.stop()

    # ── Portfolio top-5 chart ──
    all_solutions = []
    for risk in sol_risks:
        for s in risk.get("solutions", []):
            all_solutions.append({
                "risk":       risk.get("title","")[:30],
                "type":       s.get("type",""),
                "confidence": s.get("confidence", s.get("conf", 0.6)),
                "cosine":     s.get("cosine_sim", s.get("cos", 0.5)),
                "action":     s.get("action",""),
            })

    if all_solutions:
        df_sol = pd.DataFrame(all_solutions).sort_values("confidence", ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=df_sol["confidence"],
            y=(df_sol["type"] + " · " + df_sol["risk"]).str[:45],
            orientation="h",
            marker=dict(color="#3b82f6", line=dict(width=0)),
            text=[f'{v:.2f}' for v in df_sol["confidence"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=9, color="#94a3b8"),
        ))
        fig.update_layout(
            title=dict(text="Top Mitigations by Confidence Score", font=dict(family="Syne",size=13,color="#e2e8f0")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=60,t=40,b=10), height=280,
            xaxis=dict(range=[0,1.1], showgrid=False, tickfont=dict(family="IBM Plex Mono",size=9,color="#475569")),
            yaxis=dict(showgrid=False, tickfont=dict(family="IBM Plex Mono",size=9,color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-risk solution cards ──
    st.markdown('<div class="section-header">Solutions by Risk</div>', unsafe_allow_html=True)

    for risk in sol_risks:
        sev   = risk.get("severity","LOW")
        color = SEVERITY_COLOR.get(sev,"#64748b")
        title = risk.get("title","")

        st.markdown(
            f'<div style="margin:1.2rem 0 0.5rem 0;">'
            f'{render_badge(sev, severity_style(sev))}'
            f'<span style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;margin-left:0.5rem;">{title}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        solutions = risk.get("solutions",[])
        if not solutions:
            st.markdown('<div style="font-size:0.78rem;color:#475569;padding-left:1rem;">No solutions mapped.</div>', unsafe_allow_html=True)
            continue

        for idx, s in enumerate(solutions, 1):
            conf    = s.get("confidence", s.get("conf", 0.6))
            cos     = s.get("cosine_sim", s.get("cos", 0.5))
            stype   = s.get("type","—")
            action  = s.get("action","—")
            cat     = s.get("category","")
            category_html = (
                f'<div style="font-size:0.72rem;color:#475569;margin-top:0.2rem;">{cat}</div>' if cat else ""
            )

            st.markdown(
                f'<div class="solution-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div>{render_badge(f"#{idx}", "neutral")} {render_badge(stype, "blue")}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#64748b;">'
                f'conf={conf:.3f} &nbsp;|&nbsp; cos={cos:.3f}</div>'
                f'</div>'
                f'<div class="solution-action">→ {action}</div>'
                f'{category_html}'
                f'{conf_bar(conf)}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LABOR INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Labor Indicators":
    st.markdown('<div class="page-header">Labor Market Indicators</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">FRED DATA · 2014 – 2026 · LAYER 1 FINANCIAL SIGNALS</div>', unsafe_allow_html=True)

    if not commodities:
        st.info("No financial data found. Run Layer 1 first.")
    else:
        df_fin = pd.DataFrame(commodities)
        df_fin["fetched_at"] = pd.to_datetime(df_fin["fetched_at"], errors="coerce")
        df_fin = df_fin.dropna(subset=["fetched_at"]).sort_values("fetched_at")

        col1, col2 = st.columns(2)
        plots = [
            ("UNRATE", col1, "#ef4444"),
            ("ICSA",   col2, "#f97316"),
            ("PAYEMS", col1, "#3b82f6"),
            ("JTSJOL", col2, "#34d399"),
        ]

        for code, col, clr in plots:
            df_c = df_fin[df_fin["commodity"] == code]
            if df_c.empty:
                continue
            latest_val = df_c["price"].iloc[-1]
            latest_dt  = df_c["fetched_at"].iloc[-1].strftime("%b %Y")
            with col:
                st.markdown(
                    f'<div class="metric-card" style="text-align:left;margin-bottom:0.5rem;">'
                    f'<div class="metric-label">{fred_description(code)}</div>'
                    f'<div class="metric-value" style="color:{clr};">{latest_val:,.1f}</div>'
                    f'<div class="metric-delta">as of {latest_dt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                fig = go.Figure(go.Scatter(
                    x=df_c["fetched_at"], y=df_c["price"],
                    mode="lines",
                    line=dict(color=clr, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.06)",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10,r=10,t=5,b=10), height=160,
                    xaxis=dict(tickfont=dict(family="IBM Plex Mono",size=8,color="#475569"), showgrid=False, zeroline=False),
                    yaxis=dict(tickfont=dict(family="IBM Plex Mono",size=8,color="#475569"), gridcolor="#1e293b", zeroline=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Correlation note
        st.markdown('<div class="section-header">Reading These Together</div>', unsafe_allow_html=True)
        notes = {
            "UNRATE": "Unemployment Rate rising above 5% signals macro labor stress — consistent with AI displacement narrative in top risks.",
            "ICSA":   "Initial Jobless Claims (weekly) spike above 300K historically signals recession. Watch for sustained increases.",
            "PAYEMS": "Total Nonfarm Payrolls — absolute number of employed. Slowdown in monthly gains = hiring freeze evidence.",
            "JTSJOL": "Job Openings — divergence from UNRATE (high openings + high unemployment) = structural skills mismatch signal.",
        }
        for code, note in notes.items():
            if not df_fin[df_fin["commodity"]==code].empty:
                st.markdown(
                    f'<div class="solution-card">'
                    f'{render_badge(code, "blue")}'
                    f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{note}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
