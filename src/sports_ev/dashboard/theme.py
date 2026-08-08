"""SportsPredictor visual theme for the unified Streamlit dashboard."""

from __future__ import annotations

APP_TITLE = "SportsPredictor"
APP_SUBTITLE = "EV Research Control Room"
APP_TAGLINE = (
    "Refresh odds, flag +EV paper bets, and track CLV. NFL and MLB from one control room. "
    "Research only. No real money is placed."
)

# Design tokens ported from sportsPredictor/web/styles.css
THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --sp-bg: #050b17;
        --sp-bg-alt: #0c1426;
        --sp-fg: #f8fbff;
        --sp-muted: #9aa7c7;
        --sp-accent: #5c7cfa;
        --sp-accent-soft: rgba(92, 124, 250, 0.16);
        --sp-highlight: #f58bff;
        --sp-border: rgba(148, 163, 184, 0.14);
        --sp-card: rgba(15, 22, 38, 0.7);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background: radial-gradient(140% 140% at 50% 0%, rgba(92, 124, 250, 0.18), transparent 50%) var(--sp-bg) !important;
        color: var(--sp-fg);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            linear-gradient(rgba(92, 124, 250, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(92, 124, 250, 0.05) 1px, transparent 1px);
        background-size: 60px 60px;
        opacity: 0.35;
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
        min-height: 0;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    .stDeployButton {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    div[data-testid="stSpinner"] {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1001;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        color: var(--sp-fg) !important;
    }

    [data-testid="stCaptionContainer"] p,
    .stMarkdown p {
        color: var(--sp-muted);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050b17 0%, #0c1426 100%);
        border-right: 1px solid var(--sp-border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1rem !important;
        color: var(--sp-fg) !important;
    }

    [data-testid="stMetric"] {
        background: var(--sp-card);
        border: 1px solid var(--sp-border);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(10, 16, 28, 0.35);
        backdrop-filter: blur(8px);
    }

    [data-testid="stMetricLabel"] {
        color: var(--sp-muted) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    [data-testid="stMetricValue"] {
        color: var(--sp-fg) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
        color: var(--sp-fg);
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(92, 124, 250, 0.45);
        background: rgba(92, 124, 250, 0.15);
        color: #ffffff;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(92, 124, 250, 0.95), rgba(245, 139, 255, 0.85));
        border: none;
        color: #0b1020;
        font-weight: 700;
        box-shadow: 0 16px 32px rgba(92, 124, 250, 0.28);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 20px 40px rgba(92, 124, 250, 0.38);
        color: #0b1020;
    }

    [data-testid="stExpander"] {
        background: var(--sp-card);
        border: 1px solid var(--sp-border);
        border-radius: 16px;
        backdrop-filter: blur(8px);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--sp-border);
        border-radius: 16px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--sp-border);
    }

    hr {
        border-color: var(--sp-border) !important;
        opacity: 0.7;
    }

    .sp-brand-row {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 0.75rem;
        flex-wrap: wrap;
    }

    .sp-brand-mark {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 52px;
        height: 52px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(92, 124, 250, 0.85), rgba(245, 139, 255, 0.55));
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        color: #0b1020;
        box-shadow: 0 10px 24px rgba(92, 124, 250, 0.35);
        flex-shrink: 0;
    }

    .sp-brand-copy {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .sp-brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: var(--sp-fg);
        margin: 0;
        line-height: 1.2;
    }

    .sp-brand-subtitle {
        font-size: 0.85rem;
        color: var(--sp-muted);
        margin: 0;
    }

    .sp-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 999px;
        background: rgba(92, 124, 250, 0.18);
        color: var(--sp-accent);
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.72rem;
        margin-bottom: 0.75rem;
    }

    .sp-hero-lead {
        font-size: 1.05rem;
        color: var(--sp-muted);
        line-height: 1.65;
        max-width: 720px;
        margin: 0 0 1rem 0;
    }

    .sp-highlights {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 0 0 1.5rem 0;
        padding: 0;
        list-style: none;
    }

    .sp-highlights li {
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 12px;
        background: rgba(92, 124, 250, 0.12);
        border: 1px solid rgba(92, 124, 250, 0.18);
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.82);
    }

    .sp-section-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(92, 124, 250, 0.14);
        color: var(--sp-accent);
        border: 1px solid rgba(92, 124, 250, 0.25);
        margin-bottom: 0.35rem;
    }

    .sp-status-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.05);
        color: var(--sp-muted);
        border: 1px solid var(--sp-border);
    }

    .sp-status-pill.ready {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border-color: rgba(46, 160, 67, 0.35);
    }

    .sp-status-pill.warn {
        background: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border-color: rgba(210, 153, 34, 0.35);
    }

    .sp-disclaimer {
        padding: 0.85rem 1rem;
        border-radius: 14px;
        background: rgba(245, 139, 255, 0.08);
        border: 1px solid rgba(245, 139, 255, 0.2);
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
    }

    .sp-hero {
        margin-bottom: 1.5rem;
        padding: 1.25rem 1.35rem 1.35rem;
        border: 1px solid var(--sp-border);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15, 22, 38, 0.92), rgba(12, 20, 38, 0.78));
        backdrop-filter: blur(8px);
    }

    .sp-hero-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 0.85rem;
    }

    .sp-compact-header {
        margin-bottom: 1rem;
        padding: 0.85rem 1rem;
        border: 1px solid var(--sp-border);
        border-radius: 14px;
        background: var(--sp-card);
    }

    [data-testid="stSidebar"] hr {
        margin: 0.75rem 0;
        border-color: var(--sp-border);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.1rem;
        border-radius: 10px 10px 0 0;
    }

    /* Keep main content scrollable when alerts/toasts stack */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section.main,
    .main .block-container {
        overflow-y: auto !important;
        overflow-x: hidden;
    }

    [data-testid="stToast"] {
        max-width: 420px;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        overflow-y: auto;
    }
</style>
"""


def compact_header_html(*, sport_label: str) -> str:
    return f"""
<div class="sp-compact-header">
    <div class="sp-brand-row">
        <span class="sp-brand-mark">SP</span>
        <div class="sp-brand-copy">
            <p class="sp-brand-title">{APP_TITLE}</p>
            <p class="sp-brand-subtitle">{sport_label} · paper bets · research only</p>
        </div>
    </div>
</div>
"""


def hero_html(*, sport_label: str) -> str:
    """Render the SportsPredictor hero header."""
    return f"""
<div class="sp-hero">
    <div class="sp-hero-top">
        <div class="sp-brand-row">
            <span class="sp-brand-mark">SP</span>
            <div class="sp-brand-copy">
                <p class="sp-brand-title">{APP_TITLE}</p>
                <p class="sp-brand-subtitle">{APP_SUBTITLE} · {sport_label}</p>
            </div>
        </div>
    </div>
    <span class="sp-pill">Research control room</span>
    <p class="sp-hero-lead">{APP_TAGLINE}</p>
    <ul class="sp-highlights">
        <li>NFL spread + MLB moneyline models</li>
        <li>Pinnacle, DraftKings, FanDuel scrapers</li>
        <li>Paper bet tracking with CLV trends</li>
    </ul>
    <div class="sp-disclaimer">
        <strong>Research only.</strong> This tool flags paper bets and tracks hypothetical performance.
        No wagers are placed automatically.
    </div>
</div>
"""
