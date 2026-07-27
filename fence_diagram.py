import math
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SOIL_STYLES: Dict[str, Dict[str, str]] = {
    "asphalt": {"facecolor": "black", "hatch": None, "edgecolor": "black"},
    "fill": {"facecolor": "#d9d9d9", "hatch": "xx", "edgecolor": "#777777"},
    "topsoil": {"facecolor": "#c2b280", "hatch": "...", "edgecolor": "#6b5b3e"},
    "water": {"facecolor": "none", "hatch": None, "edgecolor": "#1f77b4"},
    "gp-gc": {"facecolor": "#f0e68c", "hatch": "oo", "edgecolor": "#8a7f2b"},
    "gc": {"facecolor": "#e7d7c9", "hatch": "oo", "edgecolor": "#8f6f52"},
    "gp": {"facecolor": "#f6e7c1", "hatch": "oo", "edgecolor": "#9a8351"},
    "sp-sc": {"facecolor": "#e6e6e6", "hatch": "///", "edgecolor": "#888888"},
    "sc": {"facecolor": "#f2f2f2", "hatch": "///", "edgecolor": "#888888"},
    "sp": {"facecolor": "#f9f9f9", "hatch": "...", "edgecolor": "#888888"},
    "cl": {"facecolor": "#d9c2a3", "hatch": "///", "edgecolor": "#7d674d"},
    "ch": {"facecolor": "#c8a47e", "hatch": "///", "edgecolor": "#7d5d3a"},
    "limestone": {"facecolor": "#d0d0d0", "hatch": "--", "edgecolor": "#6a6a6a"},
    "rock": {"facecolor": "#bdbdbd", "hatch": "--", "edgecolor": "#6a6a6a"},
    "void": {"facecolor": "white", "hatch": None, "edgecolor": "black"},
}


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            key = n.lower().strip().replace(" ", "_").replace("-", "_")
            if key in cols:
                return cols[key]
        return None

    renames = {}
    mapping = {
        "borehole_id": ["borehole_id", "boring", "borehole", "bh", "hole_id", "id"],
        "ground_elev": ["ground_elev", "surface_elev", "surface_elevation", "elev_ground", "g_elev", "datum"],
        "x": ["x", "east", "easting", "x_coord", "station_x"],
        "y": ["y", "north", "northing", "y_coord", "station_y"],
        "station": ["station", "chainage", "ch", "dist", "distance"],
        "order": ["order", "borehole_order", "plot_order", "sequence"],
        "top_depth": ["top_depth", "from_depth", "depth_from", "depth_top", "top"],
        "bottom_depth": ["bottom_depth", "to_depth", "depth_to", "depth_bottom", "bottom"],
        "top_elev": ["top_elev", "top_elevation", "from_elev", "elev_from"],
        "bottom_elev": ["bottom_elev", "bottom_elevation", "to_elev", "elev_to"],
        "unit": ["unit", "lithology", "soil", "material", "strata", "stratum", "description"],
        "water_elev": ["water_elev", "gw_elev", "water_level_elev", "water_table", "wt_elev"],
        "water_depth": ["water_depth", "gw_depth", "water_level_depth", "wt_depth"],
        "n_value": ["n_value", "n", "blow_count", "spt_n"],
        "rqd": ["rqd", "rock_quality_designation"],
        "casing_depth": ["casing_depth", "case_depth"],
    }
    for canonical, names in mapping.items():
        src = pick(*names)
        if src is not None and src != canonical:
            renames[src] = canonical

    df = df.rename(columns=renames)
    return df


def read_table(file_obj) -> pd.DataFrame:
    name = getattr(file_obj, "name", "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file_obj)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file_obj)
    else:
        raise ValueError("Unsupported file type. Upload CSV or Excel.")
    return normalize_columns(df)


def style_for_unit(unit: str) -> Dict[str, str]:
    u = _normalize_text(unit).lower()
    for key, style in SOIL_STYLES.items():
        if key in u:
            return style
    # broad heuristics
    if any(k in u for k in ["asphalt", "pavement"]):
        return SOIL_STYLES["asphalt"]
    if any(k in u for k in ["fill", "backfill", "ff"]):
        return SOIL_STYLES["fill"]
    if any(k in u for k in ["topsoil", "organic"]):
        return SOIL_STYLES["topsoil"]
    if any(k in u for k in ["limestone", "rock", "bedrock", "limerock"]):
        return SOIL_STYLES["limestone"]
    if any(k in u for k in ["clay", "cl"]):
        return SOIL_STYLES["cl"]
    if any(k in u for k in ["silty clay", "ch", "lean clay"]):
        return SOIL_STYLES["ch"]
    if any(k in u for k in ["sand", "sp"]):
        return SOIL_STYLES["sp"]
    return {"facecolor": "#f5f5f5", "hatch": None, "edgecolor": "#999999"}


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def intervals_to_elevation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "top_elev" not in df.columns or "bottom_elev" not in df.columns:
        if "ground_elev" not in df.columns:
            raise ValueError("Need either top_elev/bottom_elev or ground_elev with depth columns.")
        if "top_depth" not in df.columns or "bottom_depth" not in df.columns:
            raise ValueError("Need depth columns top_depth and bottom_depth when elevations are not provided.")
        df["top_elev"] = safe_numeric(df["ground_elev"]) - safe_numeric(df["top_depth"])
        df["bottom_elev"] = safe_numeric(df["ground_elev"]) - safe_numeric(df["bottom_depth"])
    df["top_elev"] = safe_numeric(df["top_elev"])
    df["bottom_elev"] = safe_numeric(df["bottom_elev"])
    return df


def borehole_positions(df_bh: pd.DataFrame, section_mode: str = "auto") -> pd.DataFrame:
    df = df_bh.copy()
    if "station" in df.columns and df["station"].notna().any():
        df["x_plot"] = safe_numeric(df["station"])
    elif "order" in df.columns and df["order"].notna().any():
        df["x_plot"] = safe_numeric(df["order"])
    elif "x" in df.columns and df["x"].notna().any():
        df = df.sort_values([c for c in ["x", "y"] if c in df.columns])
        df["x_plot"] = np.arange(len(df))
    else:
        df["x_plot"] = np.arange(len(df))
    return df


def build_borehole_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["borehole_id"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "ground_elev" not in df.columns and not ({"top_elev", "bottom_elev"} <= set(df.columns)):
        raise ValueError("Need ground_elev or interval elevations.")

    df = df.copy()
    for c in ["ground_elev", "top_depth", "bottom_depth", "top_elev", "bottom_elev", "water_elev", "water_depth", "n_value", "rqd", "order", "station", "x", "y"]:
        if c in df.columns:
            df[c] = safe_numeric(df[c])
    return df


def plot_fence_diagram(
    data: pd.DataFrame,
    interval_width: float = 0.75,
    gap: float = 1.25,
    title: str = "Subsurface Fence Diagram",
    show_legend: bool = True,
    annotate_n: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    data = build_borehole_table(data)
    data = intervals_to_elevation(data)

    boreholes = (
        data[[c for c in ["borehole_id", "ground_elev", "water_elev", "water_depth", "x", "y", "station", "order"] if c in data.columns]]
        .drop_duplicates(subset=["borehole_id"])
        .copy()
    )
    boreholes = boreholes.sort_values(by=[c for c in ["order", "station", "x_plot", "x"] if c in boreholes.columns], na_position="last")
    if "x_plot" not in boreholes.columns:
        boreholes = borehole_positions(boreholes)
    else:
        boreholes = boreholes.sort_values("x_plot")

    # Build x positions with equal gaps but retain order.
    boreholes = boreholes.reset_index(drop=True)
    boreholes["x_plot"] = np.arange(len(boreholes)) * gap
    xmap = dict(zip(boreholes["borehole_id"], boreholes["x_plot"]))

    fig, ax = plt.subplots(figsize=(max(12, len(boreholes) * 1.7), 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ymin = float(np.nanmin(data["bottom_elev"]))
    ymax_candidates = [float(np.nanmax(data["top_elev"]))]
    if "ground_elev" in data.columns and data["ground_elev"].notna().any():
        ymax_candidates.append(float(np.nanmax(data["ground_elev"])))
    if "water_elev" in data.columns and data["water_elev"].notna().any():
        ymax_candidates.append(float(np.nanmax(data["water_elev"])))
    ymax = max(ymax_candidates)
    pad = max(4.0, 0.05 * (ymax - ymin if ymax > ymin else 10))

    # Plot each borehole intervals.
    unit_seen = []
    for bh in boreholes["borehole_id"]:
        bx = xmap[bh]
        intervals = data[data["borehole_id"] == bh].sort_values(["top_elev", "bottom_elev"], ascending=[False, False])
        if intervals.empty:
            continue
        bh_meta = boreholes[boreholes["borehole_id"] == bh].iloc[0]

        # Ground line / casing cap.
        ground_elev = intervals["top_elev"].max() if "ground_elev" not in intervals.columns or intervals["ground_elev"].isna().all() else float(bh_meta.get("ground_elev", np.nan))
        if math.isnan(ground_elev):
            ground_elev = float(intervals["top_elev"].max())

        # Borehole panel outline.
        ax.add_patch(Rectangle((bx - interval_width / 2, ymin - pad), interval_width, (ymax + pad) - (ymin - pad), fill=False, linewidth=0.8, edgecolor="#c7c7c7", zorder=0))

        for _, row in intervals.iterrows():
            top = float(row["top_elev"])
            bot = float(row["bottom_elev"])
            unit = _normalize_text(row.get("unit", ""))
            style = style_for_unit(unit)
            if unit and unit.lower() not in unit_seen:
                unit_seen.append(unit.lower())
            rect = Rectangle(
                (bx - interval_width / 2, bot),
                interval_width,
                top - bot,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=0.8,
                zorder=2,
            )
            ax.add_patch(rect)

            # Interval boundary lines.
            ax.plot([bx - interval_width / 2, bx + interval_width / 2], [top, top], color="#666666", linewidth=0.5, zorder=3)
            ax.plot([bx - interval_width / 2, bx + interval_width / 2], [bot, bot], color="#666666", linewidth=0.5, zorder=3)

            if annotate_n and "n_value" in row and pd.notna(row.get("n_value")):
                ax.text(bx + interval_width * 0.62, (top + bot) / 2, f"N={int(row['n_value']) if float(row['n_value']).is_integer() else row['n_value']}", va="center", ha="left", fontsize=8, color="#2c2c2c")

        # Ground line at top of borehole intervals.
        top_y = float(intervals["top_elev"].max())
        ax.plot([bx - interval_width / 2, bx + interval_width / 2], [top_y, top_y], color="black", linewidth=2.0, zorder=4)

        # Water level.
        water_y = np.nan
        if "water_elev" in bh_meta and pd.notna(bh_meta["water_elev"]):
            water_y = float(bh_meta["water_elev"])
        elif "water_depth" in bh_meta and pd.notna(bh_meta["water_depth"]) and pd.notna(bh_meta.get("ground_elev", np.nan)):
            water_y = float(bh_meta["ground_elev"] - bh_meta["water_depth"])
        if not math.isnan(water_y):
            ax.hlines(water_y, bx - interval_width / 2, bx + interval_width / 2, colors="#1f77b4", linestyles="--", linewidth=1.1, zorder=5)
            ax.text(bx, water_y + 0.6, "WL", ha="center", va="bottom", fontsize=8, color="#1f77b4")

        # Borehole label.
        ax.text(bx, ymax + pad * 0.35, str(bh), ha="center", va="bottom", fontsize=10, fontweight="bold", rotation=90)
        if pd.notna(bh_meta.get("x", np.nan)) and pd.notna(bh_meta.get("y", np.nan)):
            ax.text(bx, ymax + pad * 0.12, f"({bh_meta['x']:.1f}, {bh_meta['y']:.1f})", ha="center", va="bottom", fontsize=7, color="#555555")

        # Elevation ticks on left of each borehole.
        for ytick in np.arange(np.floor((top_y - 15) / 10) * 10, np.ceil((top_y + 5) / 10) * 10 + 1, 10):
            if ymin - pad <= ytick <= ymax + pad:
                ax.text(bx - interval_width * 0.68, ytick, f"{ytick:.0f}", ha="right", va="center", fontsize=7, color="#666666")

    # Connect ground line between boreholes.
    ground_pts_x = []
    ground_pts_y = []
    for bh in boreholes["borehole_id"]:
        bx = xmap[bh]
        top_y = float(data.loc[data["borehole_id"] == bh, "top_elev"].max())
        ground_pts_x.append(bx)
        ground_pts_y.append(top_y)
    ax.plot(ground_pts_x, ground_pts_y, color="#444444", linewidth=1.4, linestyle="-", zorder=4)

    # Axes formatting.
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    ax.set_xlim(-gap * 0.7, (len(boreholes) - 1) * gap + gap * 0.7)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.invert_yaxis()  # higher elevations at top
    ax.set_xticks([])
    ax.set_ylabel("Elevation")
    ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.6)

    # Legend from visible units.
    if show_legend:
        unique_units = []
        for u in data["unit"].fillna(""):
            u = str(u).strip()
            if not u:
                continue
            if u.lower() not in [x.lower() for x in unique_units]:
                unique_units.append(u)
        handles = []
        labels = []
        for unit in unique_units[:12]:
            style = style_for_unit(unit)
            handles.append(Rectangle((0, 0), 1, 1, facecolor=style["facecolor"], edgecolor=style["edgecolor"], hatch=style["hatch"], linewidth=0.8))
            labels.append(unit)
        if handles:
            ax.legend(handles, labels, title="Units", loc="lower right", frameon=True, fontsize=8, title_fontsize=9)

    return fig, ax


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 200) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


SAMPLE_DATA = pd.DataFrame(
    [
        ["B-1", 888.9, 100, 200, 0, 1.2, 5.0, "Asphalt", np.nan, 12],
        ["B-1", 888.9, 100, 200, 1.2, 5.0, 8.0, "Fill", np.nan, 8],
        ["B-1", 888.9, 100, 200, 5.0, 13.0, 25.0, "SP-SC", 883.2, 11],
        ["B-1", 888.9, 100, 200, 13.0, 25.0, 38.0, "SP", np.nan, 14],
        ["B-1", 888.9, 100, 200, 25.0, 43.0, 47.0, "GC", np.nan, 11],
        ["B-1", 888.9, 100, 200, 43.0, 58.0, 65.0, "GP-GC", np.nan, 10],
        ["B-1", 888.9, 100, 200, 58.0, 95.0, 95.0, "Limestone", np.nan, np.nan],
        ["B-2", 889.1, 220, 200, 0, 0.8, 10.0, "Asphalt", np.nan, 10],
        ["B-2", 889.1, 220, 200, 0.8, 10.0, 8.0, "Fill", np.nan, 7],
        ["B-2", 889.1, 220, 200, 10.0, 28.0, 50.0, "SP-SC", 884.8, 24],
        ["B-2", 889.1, 220, 200, 28.0, 38.0, 41.0, "GC", np.nan, 13],
        ["B-2", 889.1, 220, 200, 38.0, 65.0, 24.0, "Limestone", np.nan, np.nan],
        ["B-3", 887.8, 340, 200, 0, 1.0, 7.0, "Fill", np.nan, 7],
        ["B-3", 887.8, 340, 200, 1.0, 8.0, 9.0, "SP", np.nan, 9],
        ["B-3", 887.8, 340, 200, 8.0, 12.0, 5.0, "SC", np.nan, 5],
        ["B-3", 887.8, 340, 200, 12.0, 32.0, 13.0, "GC", np.nan, 13],
        ["B-3", 887.8, 340, 200, 32.0, 40.0, 13.0, "Fill", np.nan, 13],
        ["B-3", 887.8, 340, 200, 40.0, 70.0, 20.0, "RC", np.nan, np.nan],
        ["B-4", 889.4, 460, 200, 0, 22.0, 40.0, "Void", np.nan, np.nan],
        ["B-4", 889.4, 460, 200, 22.0, 28.5, 40.0, "GP-GC", 866.8, 40],
        ["B-4", 889.4, 460, 200, 28.5, 65.5, 40.0, "Limestone", np.nan, np.nan],
    ],
    columns=["borehole_id", "ground_elev", "x", "y", "top_depth", "bottom_depth", "n_value", "unit", "water_elev", "rqd"],
)
SAMPLE_DATA["top_elev"] = SAMPLE_DATA["ground_elev"] - SAMPLE_DATA["top_depth"]
SAMPLE_DATA["bottom_elev"] = SAMPLE_DATA["ground_elev"] - SAMPLE_DATA["bottom_depth"]
