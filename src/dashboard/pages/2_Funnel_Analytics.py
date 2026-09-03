import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.loaders import load_data
from src.dashboard.theme import FLAME, TRUFFLE, apply_cartiq_theme, page_setup, style_figure

page_setup("Funnel Analytics")
apply_cartiq_theme()

st.markdown('<div class="cartiq-kicker">Cohorts</div>', unsafe_allow_html=True)
st.title("Funnel analytics")
st.markdown(
    "Interactive conversion views across months, visitor types, traffic sources, "
    "and weekend vs weekday."
)

df = load_data()


def conversion_by(col: str) -> pd.DataFrame:
    out = (
        df.groupby(col, dropna=False)["Converted"]
        .agg(sessions="count", purchases="sum", conversion_rate="mean")
        .reset_index()
        .sort_values("conversion_rate", ascending=False)
    )
    out["conversion_rate_pct"] = (out["conversion_rate"] * 100).round(2)
    return out


tab1, tab2, tab3, tab4 = st.tabs(
    ["By Month", "By Visitor Type", "By Traffic Type", "By Weekend"]
)

with tab1:
    data = conversion_by("Month")
    fig = px.bar(
        data,
        x="Month",
        y="conversion_rate_pct",
        hover_data=["sessions", "purchases"],
        title="Conversion rate by month",
        labels={"conversion_rate_pct": "Conversion rate (%)"},
        color_discrete_sequence=[FLAME],
    )
    st.plotly_chart(style_figure(fig), use_container_width=True)
    st.dataframe(data, use_container_width=True, hide_index=True)

with tab2:
    data = conversion_by("VisitorType")
    fig = px.bar(
        data,
        x="VisitorType",
        y="conversion_rate_pct",
        hover_data=["sessions", "purchases"],
        title="Conversion rate by visitor type",
        labels={"conversion_rate_pct": "Conversion rate (%)"},
        color="VisitorType",
        color_discrete_sequence=[FLAME, TRUFFLE, "#C9C1B1"],
    )
    st.plotly_chart(style_figure(fig), use_container_width=True)
    st.dataframe(data, use_container_width=True, hide_index=True)

with tab3:
    data = conversion_by("TrafficType")
    data["TrafficType"] = data["TrafficType"].astype(str)
    fig = px.bar(
        data,
        x="TrafficType",
        y="conversion_rate_pct",
        hover_data=["sessions", "purchases"],
        title="Conversion rate by traffic type",
        labels={"conversion_rate_pct": "Conversion rate (%)"},
        color_discrete_sequence=[FLAME],
    )
    st.plotly_chart(style_figure(fig), use_container_width=True)
    st.dataframe(data, use_container_width=True, hide_index=True)

with tab4:
    data = conversion_by("Weekend")
    data["Weekend"] = data["Weekend"].map({True: "Weekend", False: "Weekday"})
    fig = px.bar(
        data,
        x="Weekend",
        y="conversion_rate_pct",
        hover_data=["sessions", "purchases"],
        title="Conversion rate: weekend vs weekday",
        labels={"conversion_rate_pct": "Conversion rate (%)", "Weekend": "Day type"},
        color="Weekend",
        color_discrete_sequence=[FLAME, TRUFFLE],
    )
    st.plotly_chart(style_figure(fig), use_container_width=True)
    st.dataframe(data, use_container_width=True, hide_index=True)
