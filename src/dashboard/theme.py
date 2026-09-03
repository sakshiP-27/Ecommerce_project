"""CartIQ dashboard theme  -  restrained navy business look + Plotly styling."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Brand board: darker navy + warm orange (no pale midtones in blends)
ABYSS = "#121820"  # darker than #1B2632
BLUE_FANTASTIC = "#1E2A38"  # darker navy
FLAME = "#E8954A"  # clearer orange accent (not washed out, not neon)
TRUFFLE = "#A35139"  # deep terracotta bridge (stays dark/warm)
PALLADIAN = "#EEE9DF"
OATMEAL = "#C9C1B1"
EMBER = "#7A4032"  # dark rust mid-stop (never light/white)

COLORWAY = [FLAME, TRUFFLE, OATMEAL, BLUE_FANTASTIC, PALLADIAN]
CONTINUOUS = [
    [0.0, ABYSS],
    [0.35, BLUE_FANTASTIC],
    [0.65, EMBER],
    [0.85, TRUFFLE],
    [1.0, FLAME],
]

_ASSETS = Path(__file__).resolve().parent / "assets"
_LOGO = _ASSETS / "cartiq_logo.png"
_LOGO_SVG = _ASSETS / "cartiq_mark.svg"


def apply_cartiq_theme() -> None:
    """Inject global CSS on every page (needed for Streamlit multipage)."""
    st.markdown(
        f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {{
  --cartiq-abyss: {ABYSS};
  --cartiq-navy: {BLUE_FANTASTIC};
  --cartiq-ember: {EMBER};
  --cartiq-truffle: {TRUFFLE};
  --cartiq-flame: {FLAME};
  --cartiq-cream: {PALLADIAN};
  --cartiq-oat: {OATMEAL};
}}

/* Do NOT force fonts on Material icon glyphs  -  that caused "keyboard_double_" text */
html, body, .stApp, .main, [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span:not(.material-symbols-rounded):not([class*="material"]),
.stMarkdown, label, input, textarea, button {{
  font-family: "Source Sans 3", sans-serif;
}}

/* Preserve Streamlit / Material icon fonts */
.material-symbols-rounded,
.material-symbols-outlined,
[class*="material-symbols"],
span[data-testid="stIconMaterial"],
[data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stHeader"] span[class*="material"] {{
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-style: normal !important;
  font-weight: normal !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  speak: never;
  -webkit-font-smoothing: antialiased;
}}

/* App background: dark navy into warm orange, dark-only mid stops */
.stApp {{
  background-color: {ABYSS} !important;
  background-image:
    linear-gradient(
      180deg,
      {ABYSS} 0%,
      {ABYSS} 42%,
      {BLUE_FANTASTIC} 62%,
      {EMBER} 78%,
      {TRUFFLE} 88%,
      {FLAME} 100%
    ) !important;
  background-attachment: fixed !important;
  color: {PALLADIAN} !important;
}}

.stApp::before {{
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.07;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  mix-blend-mode: soft-light;
}}

[data-testid="stAppViewContainer"] > .main {{
  background: transparent !important;
}}

.block-container {{
  position: relative;
  z-index: 1;
  padding-top: 2.75rem !important;
  padding-bottom: 2.75rem !important;
  max-width: 1120px !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {ABYSS} 0%, #1a2530 100%) !important;
  border-right: 1px solid rgba(238, 233, 223, 0.08);
}}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  background: transparent !important;
}}

section[data-testid="stSidebar"] a {{
  border-radius: 10px !important;
}}

section[data-testid="stSidebar"] a:hover {{
  background: rgba(216, 154, 106, 0.1) !important;
}}

/* Typography  -  headings only */
h1, h2, h3, h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {{
  font-family: "Space Grotesk", sans-serif !important;
  letter-spacing: -0.03em !important;
  color: {PALLADIAN} !important;
  font-weight: 600 !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
  background: rgba(36, 48, 62, 0.78) !important;
  border: 1px solid rgba(238, 233, 223, 0.1) !important;
  border-radius: 14px !important;
  padding: 0.95rem 1rem !important;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
}}

[data-testid="stMetricValue"] {{
  font-family: "Space Grotesk", sans-serif !important;
  color: {FLAME} !important;
  font-weight: 700 !important;
  font-size: 1.75rem !important;
}}

[data-testid="stMetricLabel"] {{
  color: {OATMEAL} !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.75rem !important;
}}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {{
  background: linear-gradient(125deg, {FLAME} 0%, {TRUFFLE} 100%) !important;
  color: {ABYSS} !important;
  border: none !important;
  border-radius: 999px !important;
  font-family: "Space Grotesk", sans-serif !important;
  font-weight: 600 !important;
  padding: 0.55rem 1.3rem !important;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.22);
}}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {{
  filter: brightness(1.06);
  color: {ABYSS} !important;
}}

[data-testid="stPageLink-NavLink"] {{
  background: rgba(216, 154, 106, 0.92) !important;
  color: {ABYSS} !important;
  border-radius: 999px !important;
  padding: 0.5rem 1.15rem !important;
  font-family: "Space Grotesk", sans-serif !important;
  font-weight: 600 !important;
  border: 1px solid rgba(238, 233, 223, 0.12) !important;
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.2);
}}

[data-testid="stPageLink-NavLink"] p,
[data-testid="stPageLink-NavLink"] span {{
  color: {ABYSS} !important;
  font-family: "Space Grotesk", sans-serif !important;
}}

[data-testid="stDataFrame"] {{
  border: 1px solid rgba(238, 233, 223, 0.1);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(27, 38, 50, 0.45);
}}

.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: rgba(27, 38, 50, 0.7) !important;
  color: {PALLADIAN} !important;
  border-color: rgba(238, 233, 223, 0.14) !important;
  border-radius: 10px !important;
}}

button[data-baseweb="tab"] {{
  font-family: "Space Grotesk", sans-serif !important;
  color: {OATMEAL} !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
  color: {FLAME} !important;
}}

div[data-testid="stAlert"] {{
  border-radius: 12px !important;
  background: rgba(27, 38, 50, 0.62) !important;
  border: 1px solid rgba(238, 233, 223, 0.12) !important;
}}

/* Compact, filled hero  -  not a tall empty gradient slab */
.cartiq-hero {{
  position: relative;
  margin: 0.75rem 0 1.35rem;
  padding: 2rem 1.75rem 1.65rem;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(238, 233, 223, 0.1);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
  animation: cartiqRise 650ms ease both;
}}

.cartiq-hero-bg {{
  position: absolute;
  inset: 0;
  z-index: 0;
  background:
    linear-gradient(
      155deg,
      {ABYSS} 0%,
      {BLUE_FANTASTIC} 38%,
      {EMBER} 68%,
      {TRUFFLE} 84%,
      {FLAME} 100%
    );
}}

.cartiq-hero-inner {{
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 1.2rem 1.4rem;
  align-items: center;
}}

.cartiq-mark {{
  width: 76px;
  height: 76px;
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(232, 149, 74, 0.28), rgba(18, 24, 32, 0.92));
  border: 1px solid rgba(232, 149, 74, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: "Space Grotesk", sans-serif;
  font-weight: 700;
  font-size: 1.35rem;
  color: {FLAME};
  letter-spacing: -0.04em;
}}

.cartiq-hero-copy {{
  min-width: 0;
}}

.cartiq-brand {{
  font-family: "Space Grotesk", sans-serif !important;
  font-size: 72px !important;
  font-weight: 700 !important;
  line-height: 0.9 !important;
  letter-spacing: -0.05em !important;
  color: #FFFFFF !important;
  margin: 0 0 0.65rem !important;
  text-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
}}

.cartiq-tagline {{
  font-family: "Space Grotesk", sans-serif;
  font-size: clamp(1.05rem, 2vw, 1.35rem);
  font-weight: 500;
  color: {FLAME};
  margin: 0 0 0.5rem;
  text-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
}}

.cartiq-support {{
  font-family: "Source Sans 3", sans-serif;
  font-size: 1.05rem;
  line-height: 1.55;
  color: rgba(238, 233, 223, 0.92);
  max-width: 40rem;
  margin: 0;
}}

.cartiq-hero-meta {{
  grid-column: 1 / -1;
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 0.35rem;
}}

.cartiq-chip {{
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: {PALLADIAN};
  background: rgba(27, 38, 50, 0.55);
  border: 1px solid rgba(238, 233, 223, 0.12);
  border-radius: 999px;
  padding: 0.35rem 0.75rem;
}}

.cartiq-panel {{
  background: rgba(36, 48, 62, 0.72);
  border: 1px solid rgba(238, 233, 223, 0.1);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  margin: 0.75rem 0 1.1rem;
}}

.cartiq-kicker {{
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: {FLAME};
  margin: 0.25rem 0 0.5rem;
}}

.cartiq-section {{
  background: rgba(27, 38, 50, 0.48);
  border: 1px solid rgba(238, 233, 223, 0.08);
  border-radius: 16px;
  padding: 1.1rem 1.2rem 1.25rem;
  margin-bottom: 1.1rem;
}}

@keyframes cartiqRise {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 768px) {{
  .cartiq-hero-inner {{
    grid-template-columns: 1fr;
  }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def page_setup(title: str) -> None:
    st.set_page_config(
        page_title=f"CartIQ · {title}",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Logo above sidebar navigation
    if _LOGO.exists():
        st.logo(str(_LOGO), size="large")
    elif _LOGO_SVG.exists():
        st.logo(str(_LOGO_SVG), size="large")
    apply_cartiq_theme()


def style_figure(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(27, 38, 50, 0.45)",
        font=dict(family="Source Sans 3, sans-serif", color=PALLADIAN, size=13),
        title=dict(
            font=dict(family="Space Grotesk, sans-serif", size=18, color=PALLADIAN)
        ),
        colorway=COLORWAY,
        legend=dict(
            bgcolor="rgba(27, 38, 50, 0.75)",
            bordercolor="rgba(201, 193, 177, 0.16)",
            borderwidth=1,
            font=dict(color=PALLADIAN),
        ),
        margin=dict(l=40, r=24, t=56, b=40),
    )
    fig.update_xaxes(
        gridcolor="rgba(201, 193, 177, 0.12)",
        zerolinecolor="rgba(201, 193, 177, 0.2)",
        color=OATMEAL,
        title_font=dict(color=OATMEAL),
    )
    fig.update_yaxes(
        gridcolor="rgba(201, 193, 177, 0.12)",
        zerolinecolor="rgba(201, 193, 177, 0.2)",
        color=OATMEAL,
        title_font=dict(color=OATMEAL),
    )
    return fig


def bar_colors(n: int) -> list[str]:
    return [COLORWAY[i % len(COLORWAY)] for i in range(n)]
