from __future__ import annotations

import math
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

STYLE_LIBRARY: Dict[str, Dict[str, str | None]] = {
    "asphalt": {"facecolor": "#111111", "hatch": None, "edgecolor": "black", "label": "Asphalt"},
    "fill": {"facecolor": "#d9d9d9", "hatch": "xx", "edgecolor": "#777777", "label": "Fill"},
    "topsoil": {"facecolor": "#c2b280", "hatch": "...", "edgecolor": "#6b5b3e", "label": "Topsoil"},
    "water": {"facecolor": "#ffffff", "hatch": None, "edgecolor": "#1f77b4", "label": "Water"},
    "gp-gc": {"facecolor": "#f0e6c8", "hatch": "oo", "edgecolor": "#8a7f2b", "label": "GP-GC"},
    "gc": {"facecolor": "#f0e6c8", "hatch": "oo", "edgecolor": "#8f6f52", "label": "GC"},
    "gp": {"facecolor": "#f6e7c1", "hatch": "oo", "edgecolor": "#9a8351", "label": "GP"},
    "sp-sc": {"facecolor": "#f9e6c0", "hatch": "///", "edgecolor": "#888888", "label": "SP-SC"},
    "sc": {"facecolor": "#f2f2f2", "hatch": "///", "edgecolor": "#888888", "label": "SC"},
    "sp": {"facecolor": "#f9f9c9", "hatch": "...", "edgecolor": "#888888", "label": "SP"},
    "cl-ml": {"facecolor": "#ceb1d5", "hatch": "|||/", "edgecolor": "#7d5d7d", "label": "CL-ML"},
    "cl": {"facecolor": "#d9c2a3", "hatch": "///", "edgecolor": "#7d674d", "label": "CL"},
    "ch": {"facecolor": "#c8a47e", "hatch": "///", "edgecolor": "#7d5d3a", "label": "CH"},
    "ol": {"facecolor": "#b4a89a", "hatch": "..", "edgecolor": "#6f6259", "label": "OL"},
    "limestone": {"facecolor": "#d0d0d0", "hatch": "--", "edgecolor": "#6a6a6a", "label": "Limestone"},
    "rock": {"facecolor": "#bdbdbd", "hatch": "--", "edgecolor": "#6a6a6a", "label": "Rock"},
    "void": {"facecolor": "#ffffff", "hatch": None, "edgecolor": "black", "label": "Void"},
}

ALIASES = {
    "bh": "borehole_id",
    "boring": "borehole_id",
    "hole": "borehole_id",
    "id": "borehole_id",
    "bh_id": "borehole_id",
    "boring_id": "borehole_id",
    "ground": "ground_elev",
    "groundlevel": "ground_elev",
    "ground_elevation": "ground_elev",
    "surface_elev": "ground_elev",
    "surface_elevation": "ground_elev",
    "elevation": "ground_elev",
    "top": "top_depth",
    "from": "top_depth",
    "start_depth": "top_depth",
    "bottom": "bottom_depth",
    "to": "bottom_depth",
    "end_depth": "bottom_depth",
    "uscs_classification": "uscs",
    "uscs_group": "uscs",
    "symbol": "uscs",
    "soil": "uscs",
    "material": "unit",
    "lithology": "unit",
    "lith": "unit",
    "desc": "description",
    "blows": "n_value",
    "n": "n_value",
    "nvalue": "n_value",
    "blow_count": "n_value",
    "blowcounts": "n_value",
    "water": "water_elev",
    "waterlevel": "water_elev",
    "water_level": "water_elev",
    "wl": "water_elev",
    "rqd_percent": "rqd",
    "core_recovery": "recovery",
    "recovery_percent": "recovery",
    "rock": "rock_type",
    "rocktype": "rock_type",
    "remarks_note": "remarks",
    "chainage": "x",
    "station": "x",
}


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        normalized = _normalize_text(col).replace("-", "_")
        rename_map[col] = ALIASES.get(normalized, normalized)
    return df.rename(columns=rename_map)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "x",
        "y",
        "ground_elev",
        "top_depth",
        "bottom_depth",
        "top_elev",
        "bottom_elev",
        "n_value",
        "water_elev",
        "water_depth",
        "rqd",
        "recovery",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _first_nonempty(row: pd.Series, keys: List[str]) -> str:
    for key in keys:
        if key in row.index and not pd.isna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def intervals_to_elevation(df: pd.DataFrame) -> pd.DataFrame:
    df = _canonicalize_columns(df.copy())
    df = _coerce_numeric_columns(df)

    for col in [
        "borehole_id",
        "x",
        "y",
        "ground_elev",
        "top_depth",
        "bottom_depth",
        "top_elev",
        "bottom_elev",
        "uscs",
        "unit",
        "description",
        "n_value",
        "water_elev",
        "water_depth",
        "rqd",
        "recovery",
        "rock_type",
        "formation",
        "remarks",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Use unit as a fallback classification if uscs is blank.
    df["uscs"] = df["uscs"].astype("object")
    mask = df["uscs"].isna() | (df["uscs"].astype(str).str.strip() == "")
    df.loc[mask, "uscs"] = df.loc[mask, "unit"].astype("object")

    # Derive elevations from depth when possible.
    mask = df["ground_elev"].notna() & df["top_depth"].notna() & df["top_elev"].isna()
    df.loc[mask, "top_elev"] = df.loc[mask, "ground_elev"] - df.loc[mask, "top_depth"]

    mask = df["ground_elev"].notna() & df["bottom_depth"].notna() & df["bottom_elev"].isna()
    df.loc[mask, "bottom_elev"] = df.loc[mask, "ground_elev"] - df.loc[mask, "bottom_depth"]

    # Derive depths from elevations when needed.
    mask = df["ground_elev"].notna() & df["top_elev"].notna() & df["top_depth"].isna()
    df.loc[mask, "top_depth"] = df.loc[mask, "ground_elev"] - df.loc[mask, "top_elev"]

    mask = df["ground_elev"].notna() & df["bottom_elev"].notna() & df["bottom_depth"].isna()
    df.loc[mask, "bottom_depth"] = df.loc[mask, "ground_elev"] - df.loc[mask, "bottom_elev"]

    # Fill ground elevation if only elevations are available.
    mask = df["ground_elev"].isna() & df["top_elev"].notna() & df["top_depth"].notna()
    df.loc[mask, "ground_elev"] = df.loc[mask, "top_elev"] + df.loc[mask, "top_depth"]

    mask = df["ground_elev"].isna() & df["bottom_elev"].notna() & df["bottom_depth"].notna()
    df.loc[mask, "ground_elev"] = df.loc[mask, "bottom_elev"] + df.loc[mask, "bottom_depth"]

    # Water level.
    mask = df["water_elev"].isna() & df["water_depth"].notna() & df["ground_elev"].notna()
    df.loc[mask, "water_elev"] = df.loc[mask, "ground_elev"] - df.loc[mask, "water_depth"]

    return df


def read_table(uploaded) -> pd.DataFrame:
    if uploaded is None:
        raise ValueError("No file uploaded.")

    name = getattr(uploaded, "name", "") or ""
    lower = name.lower()

    if lower.endswith(".csv"):
        df = pd.read_csv(uploaded)

    elif lower.endswith(".xls") or lower.endswith(".xlsx"):
        try:
            df = pd.read_excel(
                uploaded,
                sheet_name="Input_Data",
                header=3,
            )
        except Exception:
            df = pd.read_excel(uploaded)

    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            df = pd.read_excel(uploaded)

    df = intervals_to_elevation(df)

    if "borehole_id" not in df.columns or df["borehole_id"].isna().all():
        raise ValueError("The uploaded file must contain a 'borehole_id' column.")

    return df


def _style_key_from_row(row: pd.Series) -> str:
    candidates = [
        _first_nonempty(row, ["uscs"]),
        _first_nonempty(row, ["unit"]),
        _first_nonempty(row, ["rock_type"]),
        _first_nonempty(row, ["formation"]),
        _first_nonempty(row, ["description"]),
        _first_nonempty(row, ["remarks"]),
    ]
    text = _normalize_text(" ".join(candidates))
    if not text:
        return "void"

    checks = [
        ("asphalt", ["asphalt"]),
        ("topsoil", ["topsoil"]),
        ("fill", ["fill", "engineered-fill"]),
        ("water", ["water", "wl"]),
        ("limestone", ["limestone", "bedrock", "lms"]),
        ("rock", ["rock", "rc", "core"]),
        ("gp-gc", ["gp-gc", "gc-gp"]),
        ("sp-sc", ["sp-sc", "sc-sp"]),
        ("cl-ml", ["cl-ml", "ml-cl"]),
        ("ch", ["ch"]),
        ("cl", ["cl"]),
        ("ol", ["ol"]),
        ("sc", ["sc"]),
        ("sp", ["sp"]),
        ("gc", ["gc"]),
        ("gp", ["gp"]),
    ]
    for key, needles in checks:
        if any(needle in text for needle in needles):
            return key
    return "void"


def _style_for_row(row: pd.Series) -> Tuple[str, Dict[str, str | None]]:
    key = _style_key_from_row(row)
    style = STYLE_LIBRARY.get(key, STYLE_LIBRARY["void"]).copy()
    return key, style


def _material_label(row: pd.Series) -> str:
    label = _first_nonempty(row, ["uscs", "unit", "rock_type", "formation"])
    if not label:
        label = _first_nonempty(row, ["description"])
    return label


def _first_valid(series: pd.Series):
    for value in series.tolist():
        if pd.notna(value):
            return value
    return np.nan


def _prepare_borehole_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("borehole_id", as_index=False).agg(
        {
            "x": _first_valid,
            "y": _first_valid,
            "ground_elev": _first_valid,
            "water_elev": _first_valid,
        }
    )
    return summary


def _compute_positions(summary: pd.DataFrame, gap: float) -> Tuple[pd.DataFrame, np.ndarray]:
    summary = summary.copy()
    if summary["x"].notna().any():
        summary = summary.sort_values(["x", "borehole_id"], kind="mergesort").reset_index(drop=True)
        xvals = pd.to_numeric(summary["x"], errors="coerce").to_numpy(dtype=float)
        valid = xvals[np.isfinite(xvals)]
        if len(valid) > 1:
            diffs = np.diff(np.sort(np.unique(valid)))
            diffs = diffs[diffs > 0]
            scale = float(np.median(diffs)) if len(diffs) else 1.0
        else:
            scale = 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        positions = (xvals - np.nanmin(xvals)) / scale * gap
        return summary, positions
    summary = summary.sort_values("borehole_id").reset_index(drop=True)
    positions = np.arange(len(summary), dtype=float) * gap
    return summary, positions


def _interval_top_bottom(row: pd.Series) -> Optional[Tuple[float, float]]:
    top = pd.to_numeric(row.get("top_elev"), errors="coerce")
    bottom = pd.to_numeric(row.get("bottom_elev"), errors="coerce")
    if pd.notna(top) and pd.notna(bottom):
        if bottom > top:
            top, bottom = bottom, top
        return float(top), float(bottom)
    return None


def plot_fence_diagram(
    df: pd.DataFrame,
    interval_width: float = 0.75,
    gap: float = 1.25,
    annotate_n: bool = True,
    show_legend: bool = True,
    title: str = "Subsurface Fence Diagram",
):
    df = intervals_to_elevation(df)
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data available.", ha="center", va="center", fontsize=14)
        return fig

    summary = _prepare_borehole_summary(df)
    summary, positions = _compute_positions(summary, gap)
    borehole_ids = summary["borehole_id"].astype(str).tolist()

    all_elevations = pd.concat(
        [
            pd.to_numeric(df.get("ground_elev"), errors="coerce"),
            pd.to_numeric(df.get("top_elev"), errors="coerce"),
            pd.to_numeric(df.get("bottom_elev"), errors="coerce"),
            pd.to_numeric(df.get("water_elev"), errors="coerce"),
        ],
        axis=0,
        ignore_index=True,
    ).dropna()
    y_max = float(all_elevations.max()) if len(all_elevations) else 100.0
    y_min = float(all_elevations.min()) if len(all_elevations) else 0.0
    y_pad = max(5.0, (y_max - y_min) * 0.1 if y_max > y_min else 5.0)

    fig = plt.figure(figsize=(max(16.0, 1.6 * len(summary) + 6.0), 10.5))
    gs = GridSpec(1, 2, width_ratios=[1.1, 7.9], wspace=0.02, figure=fig)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    ax_leg.set_axis_off()
    ax_leg.text(0.0, 0.95, "Legend Key", fontsize=12, fontweight="bold", ha="left", va="top")

    legend_items = []
    seen_styles: List[str] = []
    for bh_id in borehole_ids:
        bh_rows = df[df["borehole_id"].astype(str) == bh_id]
        for _, row in bh_rows.iterrows():
            key, style = _style_for_row(row)
            if key not in seen_styles and key in STYLE_LIBRARY:
                seen_styles.append(key)
                legend_items.append((key, style))

    if "water_elev" in df.columns and df["water_elev"].notna().any():
        legend_items.append(("water_line", {"label": "Water", "facecolor": "#ffffff", "edgecolor": "#1f77b4", "hatch": None}))

    y_cursor = 0.88
    for key, style in legend_items:
        if key == "water_line":
            ax_leg.plot([0.0, 0.35], [y_cursor - 0.03, y_cursor - 0.03], color="#1f77b4", linestyle="--", linewidth=1.6)
            ax_leg.text(0.42, y_cursor - 0.03, style["label"], fontsize=9, ha="left", va="center")
        else:
            rect = Rectangle((0.0, y_cursor - 0.06), 0.35, 0.04, facecolor=style["facecolor"], edgecolor=style["edgecolor"], hatch=style["hatch"], linewidth=0.8)
            ax_leg.add_patch(rect)
            ax_leg.text(0.42, y_cursor - 0.04, style["label"], fontsize=9, ha="left", va="center")
        y_cursor -= 0.065
        if y_cursor < 0.05:
            break
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    ground_points: List[Tuple[float, float]] = []
    water_points: List[Tuple[float, float]] = []

    for idx, bh in summary.iterrows():
        bh_id = str(bh["borehole_id"])
        x_center = float(positions[idx])
                # Track positions
        lith_left = x_center - interval_width / 2.0
        lith_right = x_center + interval_width / 2.0

        n_x = lith_right + 0.08
        recovery_x = lith_right + 0.22
        rqd_x = lith_right + 0.36
        bh_rows = df[df["borehole_id"].astype(str) == bh_id].copy()
        bh_rows = bh_rows.sort_values(["top_elev", "bottom_elev"], ascending=[False, False])

        ground_elev = pd.to_numeric(bh.get("ground_elev"), errors="coerce")
        water_elev = pd.to_numeric(bh.get("water_elev"), errors="coerce")
        if pd.notna(ground_elev):
            ground_points.append((x_center, float(ground_elev)))
        if pd.notna(water_elev):
            water_points.append((x_center, float(water_elev)))

        ax.text(x_center, y_max + y_pad * 0.3, bh_id, ha="center", va="bottom", rotation=90, fontsize=10, fontweight="bold")
       

        if pd.notna(ground_elev):
           ax.hlines(
    float(ground_elev),
    lith_left,
    lith_right,
    colors="black",
    linewidth=1.1,
    zorder=4,
)

        for _, row in bh_rows.iterrows():
            interval = _interval_top_bottom(row)
            if interval is None:
                continue
            top, bottom = interval
            thickness = top - bottom
            if thickness <= 0:
                continue
            key, style = _style_for_row(row)
            patch = Rectangle(
    (lith_left, bottom),
    lith_right - lith_left,
                thickness,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=0.8,
                zorder=2,
            )
            ax.add_patch(patch)

            label = _material_label(row)
            mid_y = (top + bottom) / 2.0
            if label:
                ax.text(lith_left + 0.03, mid_y, label, ha="left", va="center", fontsize=7.25, zorder=3)

                   # Fixed value tracks to the right of the lithology box.
        n_x = x_center + interval_width / 2.0 + 0.06
        recovery_x = x_center + interval_width / 2.0 + 0.22
        rqd_x = x_center + interval_width / 2.0 + 0.38

        if annotate_n:
            n_value = row.get("n_value")
            if pd.notna(n_value):
                try:
                    n_text = str(int(round(float(n_value))))
                except Exception:
                    n_text = str(n_value)
                ax.text(n_x, mid_y, n_text, ha="left", va="center", fontsize=7.0, zorder=3)

            recovery = row.get("recovery")
            if pd.notna(recovery):
                try:
                    recovery_text = f"{float(recovery):g}"
                except Exception:
                    recovery_text = str(recovery)
                ax.text(recovery_x, mid_y, recovery_text, ha="left", va="center", fontsize=7.0, zorder=3)

            rqd = row.get("rqd")
            if pd.notna(rqd):
                try:
                    rqd_text = f"{float(rqd):g}"
                except Exception:
                    rqd_text = str(rqd)
                ax.text(rqd_x, mid_y, rqd_text, ha="left", va="center", fontsize=7.0, zorder=3)

            # Light depth/elevation ticks on the left of each boring.
            ax.text(x_center - interval_width / 2.0 - 0.03, top, f"{top:.1f}", ha="right", va="center", fontsize=7.5)

        if pd.notna(water_elev):
            ax.hlines(float(water_elev), x_center - interval_width / 2.0, x_center + interval_width / 2.0, colors="#1f77b4", linestyles="--", linewidth=1.2, zorder=4)
            ax.text(x_center + interval_width / 2.0 + 0.04, float(water_elev), "WL", color="#1f77b4", fontsize=7, va="center", ha="left", zorder=4)

    if len(ground_points) >= 2:
        ground_points = sorted(ground_points, key=lambda t: t[0])
        ax.plot([p[0] for p in ground_points], [p[1] for p in ground_points], color="black", linewidth=1.4, marker="o", markersize=2.5, zorder=5)

    if len(water_points) >= 2:
        water_points = sorted(water_points, key=lambda t: t[0])
        ax.plot([p[0] for p in water_points], [p[1] for p in water_points], color="#1f77b4", linewidth=1.2, linestyle="--", marker="o", markersize=2.5, zorder=5)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_ylabel("Elevation", fontsize=11)
    ax.set_xlim(float(positions.min()) - interval_width * 1.8, float(positions.max()) + interval_width * 2.2)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    elev_range = y_max - y_min
    major = 10 if elev_range >= 60 else 5
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(major / 2.0))
    ax.tick_params(axis="y", which="both", right=True, labelright=True)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # The left legend axis holds the legend, matching the style of the reference figure.
    if show_legend and not legend_items:
        pass

    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.06)
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


SAMPLE_DATA = intervals_to_elevation(
    pd.DataFrame(
        [
            {"borehole_id": "B-1", "x": 100, "y": 200, "ground_elev": 888.9, "top_depth": 0.0, "bottom_depth": 1.2, "unit": "Asphalt", "n_value": 5, "recovery": 100, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-1", "x": 100, "y": 200, "ground_elev": 888.9, "top_depth": 1.2, "bottom_depth": 5.0, "unit": "Fill", "n_value": 8, "recovery": 95, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-1", "x": 100, "y": 200, "ground_elev": 888.9, "top_depth": 5.0, "bottom_depth": 13.0, "unit": "SP-SC", "n_value": 25, "recovery": 90, "rqd": 11, "water_elev": 883.2},
            {"borehole_id": "B-1", "x": 100, "y": 200, "ground_elev": 888.9, "top_depth": 13.0, "bottom_depth": 25.0, "unit": "SP", "n_value": 38, "recovery": 92, "rqd": 14, "water_elev": np.nan},
            {"borehole_id": "B-1", "x": 100, "y": 200, "ground_elev": 888.9, "top_depth": 25.0, "bottom_depth": 43.0, "unit": "GC", "n_value": 47, "recovery": 88, "rqd": 11, "water_elev": np.nan},
            {"borehole_id": "B-1", "x": 100, "y": 200, "ground_elev": 888.9, "top_depth": 43.0, "bottom_depth": 58.0, "unit": "GP-GC", "n_value": 65, "recovery": 85, "rqd": 10, "water_elev": np.nan},
            {"borehole_id": "B-1", "x": 100, "y": 200, "ground_elev": 888.9, "top_depth": 58.0, "bottom_depth": 95.0, "unit": "Limestone", "n_value": 95, "recovery": 98, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-2", "x": 220, "y": 200, "ground_elev": 889.1, "top_depth": 0.0, "bottom_depth": 0.8, "unit": "Asphalt", "n_value": 10, "recovery": 100, "rqd": 10, "water_elev": np.nan},
            {"borehole_id": "B-2", "x": 220, "y": 200, "ground_elev": 889.1, "top_depth": 0.8, "bottom_depth": 10.0, "unit": "Fill", "n_value": 8, "recovery": 95, "rqd": 7, "water_elev": np.nan},
            {"borehole_id": "B-2", "x": 220, "y": 200, "ground_elev": 889.1, "top_depth": 10.0, "bottom_depth": 28.0, "unit": "SP-SC", "n_value": 50, "recovery": 90, "rqd": 24, "water_elev": 884.8},
            {"borehole_id": "B-2", "x": 220, "y": 200, "ground_elev": 889.1, "top_depth": 28.0, "bottom_depth": 38.0, "unit": "GC", "n_value": 41, "recovery": 88, "rqd": 13, "water_elev": np.nan},
            {"borehole_id": "B-2", "x": 220, "y": 200, "ground_elev": 889.1, "top_depth": 38.0, "bottom_depth": 65.0, "unit": "Limestone", "n_value": 24, "recovery": 97, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-3", "x": 340, "y": 200, "ground_elev": 887.8, "top_depth": 0.0, "bottom_depth": 1.0, "unit": "Fill", "n_value": 7, "recovery": 100, "rqd": 7, "water_elev": np.nan},
            {"borehole_id": "B-3", "x": 340, "y": 200, "ground_elev": 887.8, "top_depth": 1.0, "bottom_depth": 8.0, "unit": "SP", "n_value": 9, "recovery": 93, "rqd": 9, "water_elev": np.nan},
            {"borehole_id": "B-3", "x": 340, "y": 200, "ground_elev": 887.8, "top_depth": 8.0, "bottom_depth": 12.0, "unit": "SC", "n_value": 5, "recovery": 91, "rqd": 5, "water_elev": np.nan},
            {"borehole_id": "B-3", "x": 340, "y": 200, "ground_elev": 887.8, "top_depth": 12.0, "bottom_depth": 32.0, "unit": "GC", "n_value": 13, "recovery": 85, "rqd": 13, "water_elev": np.nan},
            {"borehole_id": "B-4", "x": 460, "y": 200, "ground_elev": 889.4, "top_depth": 0.0, "bottom_depth": 22.0, "unit": "Void", "n_value": 40, "recovery": np.nan, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-4", "x": 460, "y": 200, "ground_elev": 889.4, "top_depth": 22.0, "bottom_depth": 28.5, "unit": "GP-GC", "n_value": 40, "recovery": 88, "rqd": 40, "water_elev": 886.8},
            {"borehole_id": "B-4", "x": 460, "y": 200, "ground_elev": 889.4, "top_depth": 28.5, "bottom_depth": 65.5, "unit": "Limestone", "n_value": 40, "recovery": 100, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-5", "x": 580, "y": 200, "ground_elev": 889.6, "top_depth": 0.0, "bottom_depth": 0.8, "unit": "Asphalt", "n_value": np.nan, "recovery": np.nan, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-5", "x": 580, "y": 200, "ground_elev": 889.6, "top_depth": 0.8, "bottom_depth": 22.0, "unit": "Void", "n_value": np.nan, "recovery": np.nan, "rqd": np.nan, "water_elev": 888.1},
            {"borehole_id": "B-5", "x": 580, "y": 200, "ground_elev": 889.6, "top_depth": 22.0, "bottom_depth": 61.0, "unit": "RC", "rock_type": "RC", "n_value": np.nan, "recovery": 100, "rqd": np.nan, "water_elev": np.nan},
        ]
    )
)

__all__ = ["SAMPLE_DATA", "fig_to_png_bytes", "intervals_to_elevation", "plot_fence_diagram", "read_table"]
