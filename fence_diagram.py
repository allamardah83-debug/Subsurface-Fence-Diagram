from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


STYLE_LIBRARY: Dict[str, Dict[str, str | None]] = {
    "asphalt": {"facecolor": "#111111", "hatch": None, "edgecolor": "black", "label": "Asphalt"},
    "fill": {"facecolor": "#d9d9d9", "hatch": "xx", "edgecolor": "#777777", "label": "Fill"},
    "topsoil": {"facecolor": "#c2b280", "hatch": "...", "edgecolor": "#6b5b3e", "label": "Topsoil"},
    "water": {"facecolor": "#eaf4ff", "hatch": None, "edgecolor": "#1f77b4", "label": "Water"},
    "gp-gc": {"facecolor": "#f0e6c8", "hatch": "oo", "edgecolor": "#8a7f2b", "label": "GP-GC"},
    "gc": {"facecolor": "#f0e6c8", "hatch": "oo", "edgecolor": "#8f6f52", "label": "GC"},
    "gp": {"facecolor": "#f6e7c1", "hatch": "oo", "edgecolor": "#9a8351", "label": "GP"},
    "sp-sc": {"facecolor": "#f9e6c0", "hatch": "///", "edgecolor": "#888888", "label": "SP-SC"},
    "sc": {"facecolor": "#f2f2f2", "hatch": "///", "edgecolor": "#888888", "label": "SC"},
    "sp": {"facecolor": "#f9f9c9", "hatch": "...", "edgecolor": "#888888", "label": "SP"},
    "cl-ml": {"facecolor": "#c39bd3", "hatch": "|||/", "edgecolor": "#7d5d7d", "label": "CL-ML"},
    "cl": {"facecolor": "#d9c2a3", "hatch": "///", "edgecolor": "#7d674d", "label": "CL"},
    "ch": {"facecolor": "#c8a47e", "hatch": "///", "edgecolor": "#7d5d3a", "label": "CH"},
    "ol": {"facecolor": "#b8a99a", "hatch": "..", "edgecolor": "#6f6259", "label": "OL"},
    "limestone": {"facecolor": "#d0d0d0", "hatch": "--", "edgecolor": "#6a6a6a", "label": "Limestone"},
    "rock": {"facecolor": "#bdbdbd", "hatch": "--", "edgecolor": "#6a6a6a", "label": "Rock"},
    "void": {"facecolor": "white", "hatch": None, "edgecolor": "black", "label": "Void"},
}


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _first_nonempty(row: pd.Series, keys: List[str]) -> str:
    for key in keys:
        if key in row.index and not pd.isna(row[key]) and str(row[key]).strip() != "":
            return str(row[key]).strip()
    return ""


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        normalized = _normalize_text(col).replace("-", "_")
        rename_map[col] = normalized

    df = df.rename(columns=rename_map)

    aliases = {
        "bh": "borehole_id",
        "boring": "borehole_id",
        "hole": "borehole_id",
        "id": "borehole_id",
        "bh_id": "borehole_id",
        "boring_id": "borehole_id",
        "station": "x",
        "chainage": "x",
        "stationing": "x",
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
        "description": "description",
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
    }

    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    return df


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
        "rqd",
        "recovery",
        "water_depth",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
        "rqd",
        "recovery",
        "rock_type",
        "formation",
        "remarks",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    if "ground_elev" in df.columns and "top_depth" in df.columns and "top_elev" not in df.columns:
        df["top_elev"] = df["ground_elev"] - df["top_depth"]

    if "ground_elev" in df.columns and "bottom_depth" in df.columns and "bottom_elev" not in df.columns:
        df["bottom_elev"] = df["ground_elev"] - df["bottom_depth"]

    if "top_elev" in df.columns and "top_depth" in df.columns and df["ground_elev"].isna().any():
        mask = df["ground_elev"].isna() & df["top_elev"].notna() & df["top_depth"].notna()
        df.loc[mask, "ground_elev"] = df.loc[mask, "top_elev"] + df.loc[mask, "top_depth"]

    if "bottom_elev" in df.columns and "bottom_depth" in df.columns and df["ground_elev"].isna().any():
        mask = df["ground_elev"].isna() & df["bottom_elev"].notna() & df["bottom_depth"].notna()
        df.loc[mask, "ground_elev"] = df.loc[mask, "bottom_elev"] + df.loc[mask, "bottom_depth"]

    if "top_depth" not in df.columns and "ground_elev" in df.columns and "top_elev" in df.columns:
        df["top_depth"] = df["ground_elev"] - df["top_elev"]

    if "bottom_depth" not in df.columns and "ground_elev" in df.columns and "bottom_elev" in df.columns:
        df["bottom_depth"] = df["ground_elev"] - df["bottom_elev"]

    if "water_elev" not in df.columns:
        df["water_elev"] = np.nan
    if "water_depth" in df.columns:
        mask = df["water_elev"].isna() & df["water_depth"].notna() & df["ground_elev"].notna()
        df.loc[mask, "water_elev"] = df.loc[mask, "ground_elev"] - df.loc[mask, "water_depth"]

    return df


def read_table(uploaded) -> pd.DataFrame:
    if uploaded is None:
        raise ValueError("No file uploaded.")

    filename = getattr(uploaded, "name", "") or ""
    lower = filename.lower()

    if lower.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif lower.endswith(".xls") or lower.endswith(".xlsx"):
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
        ("sp-sc", ["sp-sc", "sc-sp"]),
        ("cl-ml", ["cl-ml", "ml-cl"]),
        ("gp-gc", ["gp-gc", "gc-gp"]),
        ("asphalt", ["asphalt"]),
        ("topsoil", ["topsoil"]),
        ("fill", ["fill", "engineered-fill"]),
        ("water", ["water", "wl"]),
        ("limestone", ["limestone", "lms", "bedrock"]),
        ("rock", ["rock", "ref", "core"]),
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
    cols = [c for c in ["borehole_id", "x", "y", "ground_elev", "water_elev"] if c in df.columns]
    summary = (
        df.groupby("borehole_id", as_index=False)[cols[1:]]
        .agg(_first_valid)
        if len(cols) > 1
        else df[["borehole_id"]].drop_duplicates().copy()
    )

    if "x" not in summary.columns:
        summary["x"] = np.nan
    if "y" not in summary.columns:
        summary["y"] = np.nan
    if "ground_elev" not in summary.columns:
        summary["ground_elev"] = np.nan
    if "water_elev" not in summary.columns:
        summary["water_elev"] = np.nan

    return summary


def _compute_positions(summary: pd.DataFrame, gap: float) -> Tuple[pd.DataFrame, np.ndarray]:
    summary = summary.copy()

    if "x" in summary.columns and summary["x"].notna().any():
        summary = summary.sort_values(["x", "borehole_id"], kind="mergesort").reset_index(drop=True)
        xs = pd.to_numeric(summary["x"], errors="coerce").to_numpy(dtype=float)

        valid = xs[np.isfinite(xs)]
        if len(valid) > 1:
            unique_sorted = np.sort(np.unique(valid))
            diffs = np.diff(unique_sorted)
            diffs = diffs[diffs > 0]
            ref = float(np.median(diffs)) if len(diffs) else 1.0
        else:
            ref = 1.0

        if not np.isfinite(ref) or ref <= 0:
            ref = 1.0

        positions = (xs - np.nanmin(xs)) / ref * gap
        return summary, positions

    summary = summary.sort_values("borehole_id").reset_index(drop=True)
    positions = np.arange(len(summary), dtype=float) * gap
    return summary, positions


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
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data available.", ha="center", va="center", fontsize=14)
        return fig

    if "borehole_id" not in df.columns:
        raise ValueError("Input table must contain a 'borehole_id' column.")

    summary = _prepare_borehole_summary(df)
    summary, positions = _compute_positions(summary, gap=gap)

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

    if len(all_elevations):
        y_max = float(all_elevations.max())
        y_min = float(all_elevations.min())
    else:
        y_max = 100.0
        y_min = 0.0

    y_pad = max(5.0, (y_max - y_min) * 0.10 if y_max > y_min else 5.0)

    fig_width = max(12.0, 1.4 * len(summary) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 8.5))

    used_style_keys: List[str] = []
    ground_points: List[Tuple[float, float]] = []
    water_points: List[Tuple[float, float]] = []

    for idx, bh in summary.iterrows():
        bh_id = str(bh["borehole_id"])
        x_center = float(positions[idx])

        bh_rows = df[df["borehole_id"].astype(str) == bh_id].copy()
        bh_rows = bh_rows.sort_values(["top_elev", "bottom_elev"], ascending=[False, False])

        ground_elev = pd.to_numeric(bh.get("ground_elev"), errors="coerce")
        water_elev = pd.to_numeric(bh.get("water_elev"), errors="coerce")

        if pd.notna(ground_elev):
            ground_points.append((x_center, float(ground_elev)))

        if pd.notna(water_elev):
            water_points.append((x_center, float(water_elev)))

        # Borehole label
        ax.text(
            x_center,
            y_max + y_pad * 0.25,
            bh_id,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9,
            fontweight="bold",
        )

        # Depth/elevation intervals
        for _, row in bh_rows.iterrows():
            top = pd.to_numeric(row.get("top_elev"), errors="coerce")
            bottom = pd.to_numeric(row.get("bottom_elev"), errors="coerce")

            if pd.isna(top) or pd.isna(bottom):
                continue

            if bottom > top:
                top, bottom = bottom, top

            thickness = float(top - bottom)
            if thickness <= 0:
                continue

            key, style = _style_for_row(row)
            used_style_keys.append(key)

            patch = Rectangle(
                (x_center - interval_width / 2.0, bottom),
                interval_width,
                thickness,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=0.8,
                zorder=2,
            )
            ax.add_patch(patch)

            mid_y = (top + bottom) / 2.0
            label = _material_label(row)
            if label:
                ax.text(
                    x_center - interval_width / 2.0 + 0.03,
                    mid_y,
                    label,
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    zorder=3,
                )

            notes: List[str] = []
            if annotate_n:
                n_value = row.get("n_value")
                if pd.notna(n_value):
                    try:
                        notes.append(f"N={int(round(float(n_value)))}")
                    except Exception:
                        notes.append(f"N={n_value}")

                recovery = row.get("recovery")
                if pd.notna(recovery):
                    try:
                        notes.append(f"Rec={float(recovery):g}%")
                    except Exception:
                        notes.append(f"Rec={recovery}%")

                rqd = row.get("rqd")
                if pd.notna(rqd):
                    try:
                        notes.append(f"RQD={float(rqd):g}%")
                    except Exception:
                        notes.append(f"RQD={rqd}%")

            if notes:
                ax.text(
                    x_center + interval_width / 2.0 + 0.07,
                    mid_y,
                    "\n".join(notes),
                    ha="left",
                    va="center",
                    fontsize=7.25,
                    zorder=3,
                )

        # Borehole water line
        if pd.notna(water_elev):
            ax.hlines(
                float(water_elev),
                x_center - interval_width / 2.0,
                x_center + interval_width / 2.0,
                colors="#1f77b4",
                linestyles="--",
                linewidth=1.2,
                zorder=4,
            )
            ax.text(
                x_center + interval_width / 2.0 + 0.04,
                float(water_elev),
                "WL",
                color="#1f77b4",
                fontsize=7,
                va="center",
                ha="left",
                zorder=4,
            )

        # Ground line above each borehole
        if pd.notna(ground_elev):
            ax.hlines(
                float(ground_elev),
                x_center - interval_width / 2.0,
                x_center + interval_width / 2.0,
                colors="black",
                linewidth=1.2,
                zorder=4,
            )

    # Connect ground surface and water table across boreholes
    if len(ground_points) >= 2:
        ground_points = sorted(ground_points, key=lambda t: t[0])
        ax.plot(
            [p[0] for p in ground_points],
            [p[1] for p in ground_points],
            color="black",
            linewidth=1.6,
            marker="o",
            markersize=2.5,
            zorder=5,
        )

    if len(water_points) >= 2:
        water_points = sorted(water_points, key=lambda t: t[0])
        ax.plot(
            [p[0] for p in water_points],
            [p[1] for p in water_points],
            color="#1f77b4",
            linewidth=1.2,
            linestyle="--",
            marker="o",
            markersize=2.5,
            zorder=5,
        )

    # Axes, labels, and layout
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_ylabel("Elevation", fontsize=11)

    ax.set_xlim(float(positions.min()) - interval_width * 1.6, float(positions.max()) + interval_width * 1.8)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xticks(positions)
    ax.set_xticklabels(borehole_ids, rotation=0, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if show_legend and used_style_keys:
        ordered_keys: List[str] = []
        for key in used_style_keys:
            if key not in ordered_keys and key in STYLE_LIBRARY:
                ordered_keys.append(key)

        handles: List[Any] = []
        for key in ordered_keys:
            style = STYLE_LIBRARY[key]
            handles.append(
                Patch(
                    facecolor=style["facecolor"],
                    edgecolor=style["edgecolor"],
                    hatch=style["hatch"],
                    label=style["label"],
                )
            )

        if len(water_points) >= 1:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#1f77b4",
                    linestyle="--",
                    linewidth=1.2,
                    label="Water level",
                )
            )

        if handles:
            ax.legend(
                handles=handles,
                title="Legend",
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=True,
            )

    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


SAMPLE_DATA = intervals_to_elevation(
    pd.DataFrame(
        [
            {"borehole_id": "B-1", "ground_elev": 888.9, "x": 100, "y": 200, "top_depth": 0.0, "bottom_depth": 1.2, "n_value": 5, "unit": "Asphalt", "recovery": 100, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-1", "ground_elev": 888.9, "x": 100, "y": 200, "top_depth": 1.2, "bottom_depth": 5.0, "n_value": 8, "unit": "Fill", "recovery": 95, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-1", "ground_elev": 888.9, "x": 100, "y": 200, "top_depth": 5.0, "bottom_depth": 13.0, "n_value": 25, "unit": "SP-SC", "recovery": 90, "rqd": 11, "water_elev": 883.2},
            {"borehole_id": "B-1", "ground_elev": 888.9, "x": 100, "y": 200, "top_depth": 13.0, "bottom_depth": 25.0, "n_value": 38, "unit": "SP", "recovery": 92, "rqd": 14, "water_elev": np.nan},
            {"borehole_id": "B-1", "ground_elev": 888.9, "x": 100, "y": 200, "top_depth": 25.0, "bottom_depth": 43.0, "n_value": 47, "unit": "GC", "recovery": 88, "rqd": 11, "water_elev": np.nan},
            {"borehole_id": "B-1", "ground_elev": 888.9, "x": 100, "y": 200, "top_depth": 43.0, "bottom_depth": 58.0, "n_value": 65, "unit": "GP-GC", "recovery": 85, "rqd": 10, "water_elev": np.nan},
            {"borehole_id": "B-1", "ground_elev": 888.9, "x": 100, "y": 200, "top_depth": 58.0, "bottom_depth": 95.0, "n_value": 95, "unit": "Limestone", "recovery": 98, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-2", "ground_elev": 889.1, "x": 220, "y": 200, "top_depth": 0.0, "bottom_depth": 0.8, "n_value": 10, "unit": "Asphalt", "recovery": 100, "rqd": 10, "water_elev": np.nan},
            {"borehole_id": "B-2", "ground_elev": 889.1, "x": 220, "y": 200, "top_depth": 0.8, "bottom_depth": 10.0, "n_value": 8, "unit": "Fill", "recovery": 95, "rqd": 7, "water_elev": np.nan},
            {"borehole_id": "B-2", "ground_elev": 889.1, "x": 220, "y": 200, "top_depth": 10.0, "bottom_depth": 28.0, "n_value": 50, "unit": "SP-SC", "recovery": 90, "rqd": 24, "water_elev": 884.8},
            {"borehole_id": "B-2", "ground_elev": 889.1, "x": 220, "y": 200, "top_depth": 28.0, "bottom_depth": 38.0, "n_value": 41, "unit": "GC", "recovery": 88, "rqd": 13, "water_elev": np.nan},
            {"borehole_id": "B-2", "ground_elev": 889.1, "x": 220, "y": 200, "top_depth": 38.0, "bottom_depth": 65.0, "n_value": 24, "unit": "Limestone", "recovery": 97, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-3", "ground_elev": 887.8, "x": 340, "y": 200, "top_depth": 0.0, "bottom_depth": 1.0, "n_value": 7, "unit": "Fill", "recovery": 100, "rqd": 7, "water_elev": np.nan},
            {"borehole_id": "B-3", "ground_elev": 887.8, "x": 340, "y": 200, "top_depth": 1.0, "bottom_depth": 8.0, "n_value": 9, "unit": "SP", "recovery": 93, "rqd": 9, "water_elev": np.nan},
            {"borehole_id": "B-3", "ground_elev": 887.8, "x": 340, "y": 200, "top_depth": 8.0, "bottom_depth": 12.0, "n_value": 5, "unit": "SC", "recovery": 91, "rqd": 5, "water_elev": np.nan},
            {"borehole_id": "B-3", "ground_elev": 887.8, "x": 340, "y": 200, "top_depth": 12.0, "bottom_depth": 32.0, "n_value": 13, "unit": "GC", "recovery": 85, "rqd": 13, "water_elev": np.nan},
            {"borehole_id": "B-4", "ground_elev": 889.4, "x": 460, "y": 200, "top_depth": 0.0, "bottom_depth": 22.0, "n_value": 40, "unit": "Void", "recovery": np.nan, "rqd": np.nan, "water_elev": np.nan},
            {"borehole_id": "B-4", "ground_elev": 889.4, "x": 460, "y": 200, "top_depth": 22.0, "bottom_depth": 28.5, "n_value": 40, "unit": "GP-GC", "recovery": 88, "rqd": 40, "water_elev": 886.8},
            {"borehole_id": "B-4", "ground_elev": 889.4, "x": 460, "y": 200, "top_depth": 28.5, "bottom_depth": 65.5, "n_value": 40, "unit": "Limestone", "recovery": 100, "rqd": np.nan, "water_elev": np.nan},
        ]
    )
)


__all__ = [
    "SAMPLE_DATA",
    "fig_to_png_bytes",
    "intervals_to_elevation",
    "plot_fence_diagram",
    "read_table",
]
