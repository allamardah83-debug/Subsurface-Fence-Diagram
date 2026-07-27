from __future__ import annotations

import pandas as pd
import streamlit as st

from pathlib import Path

from fence_diagram import fig_to_png_bytes, intervals_to_elevation, plot_fence_diagram, read_table
SAMPLE_FILE = Path(__file__).with_name("sample_data.xlsx")

st.set_page_config(page_title="Subsurface Fence Diagram", layout="wide")
st.title("Subsurface Fence Diagram Generator")
st.caption("Upload boring logs in CSV or Excel and generate a fence-style subsurface profile.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload boring log table", type=["csv", "xlsx", "xls"])
    use_sample = st.checkbox("Use sample data", value=(uploaded is None))
    title = st.text_input("Plot title", value="Subsurface Fence Diagram")
    interval_width = st.slider("Log width", min_value=0.3, max_value=2.0, value=0.75, step=0.05)
    gap = st.slider("Spacing between borings", min_value=0.6, max_value=4.0, value=1.25, step=0.05)
    annotate_n = st.checkbox("Show N-values", value=True)
    show_legend = st.checkbox("Show legend", value=True)

if uploaded is not None:
    try:
        df = read_table(uploaded)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()
elif use_sample:
    df = read_table(SAMPLE_FILE)
else:
    st.info("Upload a CSV or Excel file, or check 'Use sample data'.")
    st.stop()

st.subheader("Input table preview")
st.dataframe(df.head(60), use_container_width=True)

required_cols = ["borehole_id"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

try:
    plot_df = intervals_to_elevation(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Expected columns", expanded=False):
    st.markdown(
        """
        Use one of these patterns:

        - `borehole_id, ground_elev, top_depth, bottom_depth, unit`
        - `borehole_id, top_elev, bottom_elev, unit`

        Optional columns: `x`, `y`, `water_elev`, `water_depth`, `n_value`, `recovery`, `rqd`, `uscs`, `description`, `formation`, `rock_type`.
        """
    )

left, right = st.columns([2.2, 1])
with left:
    try:
        fig = plot_fence_diagram(
            plot_df,
            interval_width=interval_width,
            gap=gap,
            title=title,
            show_legend=show_legend,
            annotate_n=annotate_n,
        )
        st.pyplot(fig, use_container_width=True)
        png_bytes = fig_to_png_bytes(fig)
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name="subsurface_fence_diagram.png",
            mime="image/png",
        )
    except Exception as exc:
        st.error(f"Could not render plot: {exc}")

with right:
    st.subheader("Field checks")
    checks = []
    if "ground_elev" in df.columns:
        checks.append(f"Ground elevations present for {df['ground_elev'].notna().sum()} rows.")
    if {"top_depth", "bottom_depth"} <= set(df.columns):
        checks.append("Depth-based intervals detected.")
    if {"top_elev", "bottom_elev"} <= set(df.columns):
        checks.append("Elevation-based intervals detected.")
    if "water_elev" in df.columns or "water_depth" in df.columns:
        checks.append("Water level data detected.")
    if "n_value" in df.columns:
        checks.append("SPT N-values detected.")
    if "recovery" in df.columns:
        checks.append("Core recovery detected.")
    if "rqd" in df.columns:
        checks.append("RQD detected.")
    if "x" in df.columns or "station" in df.columns:
        checks.append("Coordinate/station data detected.")
    if checks:
        for c in checks:
            st.write(f"- {c}")
    else:
        st.write("No optional fields detected.")

st.divider()
st.markdown("Built for Streamlit + GitHub deployment.")
