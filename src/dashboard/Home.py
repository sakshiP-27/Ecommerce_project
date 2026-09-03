"""CartIQ landing page (Streamlit entry point)."""

import base64
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.dashboard.theme import page_setup

page_setup("Landing")

_ASSETS = Path(__file__).resolve().parent / "assets"


def _local_img(name: str) -> str:
    """Encode a local asset as a data URI for reliable Docker loads."""
    path = _ASSETS / name
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# Unique Unsplash images per section (no repeats)
IMG_OVERVIEW = (
    "https://images.unsplash.com/photo-1460925895917-afdab827c52f"
    "?auto=format&fit=crop&w=1400&q=80"
)
IMG_STEP_INGEST = (
    "https://images.unsplash.com/photo-1472851294608-062f824d29cc"
    "?auto=format&fit=crop&w=1400&q=80"
)
IMG_STEP_SCORE = (
    "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d"
    "?auto=format&fit=crop&w=1400&q=80"
)
IMG_STEP_EXPLAIN = (
    "https://images.unsplash.com/photo-1552664730-d307ca884978"
    "?auto=format&fit=crop&w=1400&q=80"
)
IMG_FEAT_LIVE = (
    "https://images.unsplash.com/photo-1563013544-824ae1b704d3"
    "?auto=format&fit=crop&w=1400&q=80"
)
IMG_FEAT_FUNNEL = (
    "https://images.unsplash.com/photo-1441986300917-64674bd600d8"
    "?auto=format&fit=crop&w=1400&q=80"
)
IMG_FEAT_METRICS = (
    "https://images.unsplash.com/photo-1551288049-bebda4e38f71"
    "?auto=format&fit=crop&w=1400&q=80"
)
# Local asset — Unsplash remote URL was failing in the browser/Docker
IMG_FEAT_EXPLAIN = _local_img("feat_explain.jpg")

# --- Hero ---
st.markdown(
    f"""
<div class="cartiq-hero cartiq-landing-hero">
  <div class="cartiq-hero-bg"></div>
  <div class="cartiq-hero-inner">
    <div class="cartiq-mark">IQ</div>
    <div class="cartiq-hero-copy">
      <p class="cartiq-brand" style="font-size:72px !important;font-weight:700 !important;line-height:0.9 !important;color:#FFFFFF !important;margin:0 0 0.65rem 0 !important;font-family:'Space Grotesk',sans-serif !important;">CartIQ</p>
      <p class="cartiq-tagline">Intelligence Behind Every Cart.</p>
      <p class="cartiq-support">
        Predict purchase intent from ecommerce sessions, explain every score,
        and help teams act before shoppers bounce.
      </p>
    </div>
    <div class="cartiq-hero-meta">
      <span class="cartiq-chip">Purchase intent</span>
      <span class="cartiq-chip">Explainable AI</span>
      <span class="cartiq-chip">Live scoring</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.2, 1.2, 2])
with c1:
    st.page_link("pages/1_Live_Scoring.py", label="Open dashboard")
with c2:
    st.page_link("pages/0_Dashboard.py", label="View overview")

# --- Overview ---
st.markdown('<div class="cartiq-kicker">Overview</div>', unsafe_allow_html=True)
st.markdown('<h2 class="cartiq-section-title">Built for teams who care about conversion</h2>', unsafe_allow_html=True)

ov_left, ov_right = st.columns([1.05, 0.95], gap="large")
with ov_left:
    st.markdown(
        """
<div class="cartiq-panel cartiq-fade">
  <p class="cartiq-body">
    CartIQ turns raw browsing behaviour into a clear purchase-intent signal.
    It scores each session, surfaces the reasons behind the prediction, and
    gives product and growth teams a practical way to prioritise outreach.
  </p>
  <p class="cartiq-body" style="margin-bottom:0;">
    From funnel analytics to model performance and calibration, everything sits
    in one place so decisions stay grounded in evidence, not guesswork.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
with ov_right:
    st.markdown(
        f"""
<div class="cartiq-media cartiq-fade">
  <img src="{IMG_OVERVIEW}" alt="Analytics workspace on a laptop" />
</div>
""",
        unsafe_allow_html=True,
    )

# --- How it works ---
st.markdown('<div class="cartiq-kicker">How it works</div>', unsafe_allow_html=True)
st.markdown('<h2 class="cartiq-section-title">Three steps from session to action</h2>', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3, gap="medium")
steps = [
    (
        "01",
        "Ingest the session",
        "Capture browsing signals such as pages viewed, time on site, bounce and exit rates.",
        IMG_STEP_INGEST,
    ),
    (
        "02",
        "Score purchase intent",
        "The production CatBoost model returns a probability and a clear buy / no-buy decision.",
        IMG_STEP_SCORE,
    ),
    (
        "03",
        "Explain and act",
        "SHAP highlights the drivers behind each score so teams know what to do next.",
        IMG_STEP_EXPLAIN,
    ),
]
for col, (num, title, body, img) in zip((s1, s2, s3), steps):
    with col:
        st.markdown(
            f"""
<div class="cartiq-step cartiq-fade">
  <div class="cartiq-media cartiq-media-sm">
    <img src="{img}" alt="{title}" />
  </div>
  <p class="cartiq-step-num">{num}</p>
  <h3 class="cartiq-step-title">{title}</h3>
  <p class="cartiq-body">{body}</p>
</div>
""",
            unsafe_allow_html=True,
        )

# --- Features ---
st.markdown('<div class="cartiq-kicker">Features</div>', unsafe_allow_html=True)
st.markdown('<h2 class="cartiq-section-title">Everything you need in one workspace</h2>', unsafe_allow_html=True)

f1, f2 = st.columns(2, gap="large")
with f1:
    st.markdown(
        f"""
<div class="cartiq-feature cartiq-fade">
  <div class="cartiq-media">
    <img src="{IMG_FEAT_LIVE}" alt="Online payment and live scoring" />
  </div>
  <h3 class="cartiq-step-title">Live session scoring</h3>
  <p class="cartiq-body">
    Score any session in seconds and see purchase probability against your
    operating threshold, with SHAP explanations beside the result.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="cartiq-feature cartiq-fade">
  <div class="cartiq-media">
    <img src="{IMG_FEAT_FUNNEL}" alt="Retail shopping floor" />
  </div>
  <h3 class="cartiq-step-title">Funnel analytics</h3>
  <p class="cartiq-body">
    Explore conversion by month, visitor type, traffic source, and weekend
    versus weekday to spot where intent rises or drops.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        f"""
<div class="cartiq-feature cartiq-fade">
  <div class="cartiq-media">
    <img src="{IMG_FEAT_METRICS}" alt="Model performance charts" />
  </div>
  <h3 class="cartiq-step-title">Model comparison and metrics</h3>
  <p class="cartiq-body">
    Compare candidates on PR-AUC, precision, and recall, then inspect profit
    curves, calibration, and confusion matrices for the production model.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="cartiq-feature cartiq-fade">
  <div class="cartiq-media">
    <img src="{IMG_FEAT_EXPLAIN}" alt="Data screens used for model explanations" />
  </div>
  <h3 class="cartiq-step-title">Explainability built in</h3>
  <p class="cartiq-body">
    Global and local explanations show which features push a session toward
    purchase or away from it, so stakeholders can trust the score.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

# --- Bottom CTA ---
st.markdown(
    """
<div class="cartiq-cta-band cartiq-fade">
  <p class="cartiq-cta-title">Ready to score a session?</p>
  <p class="cartiq-body" style="margin:0;">
    Jump into the dashboard and try live scoring with real ecommerce features.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
b1, b2, _ = st.columns([1.2, 1.2, 2])
with b1:
    st.page_link("pages/1_Live_Scoring.py", label="Score a session")
with b2:
    st.page_link("pages/0_Dashboard.py", label="Go to dashboard")

# --- Footer ---
st.markdown(
    """
<div class="cartiq-footer">
  <p class="cartiq-footer-brand">CartIQ</p>
  <p class="cartiq-footer-name">Sakshi Paygude</p>
  <p class="cartiq-footer-note">Purchase-intent intelligence for ecommerce teams</p>
</div>
""",
    unsafe_allow_html=True,
)
