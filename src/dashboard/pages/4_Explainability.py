import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="Explainability", layout="wide")
st.title("Explainability")
st.markdown(
    "Global and local explanations from Week 3 (SHAP), plus a short SHAP vs LIME comparison."
)

reports = _ROOT / "reports"
summary_path = reports / "shap_summary_plot.png"
bar_path = reports / "shap_bar_plot.png"

st.subheader("Global SHAP explanations")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Summary (beeswarm) plot**")
    if summary_path.exists():
        st.image(str(summary_path), use_container_width=True)
    else:
        st.warning(f"Missing `{summary_path.name}` — re-save it from the Week 3 notebook.")
    st.caption(
        "Each dot is one test session. Position = SHAP impact; color = feature value. "
        "Features at the top matter most overall."
    )
with c2:
    st.markdown("**Mean |SHAP| bar plot**")
    if bar_path.exists():
        st.image(str(bar_path), use_container_width=True)
    else:
        st.warning(f"Missing `{bar_path.name}` — re-save it from the Week 3 notebook.")
    st.caption("Simple ranking of average importance — good for non-technical readers.")

st.subheader("What drives predictions?")
st.markdown(
    """
Across the test set, the strongest drivers are typically:

1. **PageValues** — high values push toward purchase; zero/low values push against it  
2. **ProductRelated / TotalPages** — more engagement supports purchase intent  
3. **ExitRates / BounceRates** — higher rates push toward no purchase  

Use **Live Session Scoring** for interactive local waterfall explanations on any session.
"""
)

st.subheader("SHAP vs LIME (same 3 example sessions)")
st.markdown(
    """
**Clear buyer:** Both methods agree **PageValues** is the main positive driver, with
**TotalPages** / **ProductRelated** also supporting purchase.

**Clear non-buyer:** Both point to **low/zero PageValues** as the strongest negative
signal, with **ExitRates** and thin browsing also hurting.

**Uncertain (near threshold):** Both say **PageValues** pushes up while **ExitRates**
(and weaker product activity) pull down — which is why the score sits near the boundary.

**Overall:** SHAP and LIME largely agree on the top features (especially PageValues),
which makes the explanations more trustworthy.
"""
)

st.info(
    "SHAP charts here are static exports from Week 3 (matplotlib). "
    "Funnel / comparison pages use interactive Plotly charts."
)
