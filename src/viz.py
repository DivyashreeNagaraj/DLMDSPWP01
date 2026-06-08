"""
viz.py
------
Visualization of training data, selected ideal functions, and mapped
test points using Bokeh.

The plot is saved as an HTML file for interactive exploration.
"""

import pandas as pd
from typing import List

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.palettes import Category10

from modeling import SelectedFunction
from exceptions import VisualizationError

# Colour palette for up to 10 series
PALETTE = Category10[10]


def build_plot(
    train_df: pd.DataFrame,
    ideal_df: pd.DataFrame,
    selections: List[SelectedFunction],
    mapped_df: pd.DataFrame,
    output_path: str = "regression_plot.html",
) -> None:
    """
    Create and save an interactive Bokeh plot showing:

    - Training data points (circles, one colour per training column)
    - Selected ideal functions (solid lines)
    - Mapped test points (triangles, one colour per ideal function)

    Parameters
    ----------
    train_df     : pd.DataFrame – training data (x + y1..y4)
    ideal_df     : pd.DataFrame – ideal functions (x + y1..y50)
    selections   : List[SelectedFunction]
    mapped_df    : pd.DataFrame – columns: x, y, delta_y, ideal_func
    output_path  : str          – path for the HTML output file
    """
    try:
        output_file(output_path, title="DLMDSPWP01 – Regression Analysis")

        p = figure(
            title="DLMDSPWP01 – Ideal Function Mapping",
            x_axis_label="x",
            y_axis_label="y",
            width=1000,
            height=550,
            toolbar_location="above",
        )

        legend_items: List[LegendItem] = []
        colour_idx = 0

        # ── Training data points ───────────────────────────────────────
        train_y_cols = [c for c in train_df.columns if c != "x"]
        for t_col in train_y_cols:
            colour = PALETTE[colour_idx % len(PALETTE)]
            src = ColumnDataSource({"x": train_df["x"], "y": train_df[t_col]})
            rend = p.circle("x", "y", source=src, size=5, color=colour,
                            alpha=0.6, legend_label=f"train: {t_col}")
            colour_idx += 1

        # ── Selected ideal functions (lines) ──────────────────────────
        for sel in selections:
            colour = PALETTE[colour_idx % len(PALETTE)]
            src = ColumnDataSource({"x": ideal_df["x"], "y": ideal_df[sel.ideal_col]})
            p.line("x", "y", source=src, line_width=2, color=colour,
                   alpha=0.85, legend_label=f"ideal: {sel.ideal_col}")
            colour_idx += 1

        # ── Mapped test points ─────────────────────────────────────────
        if not mapped_df.empty:
            src = ColumnDataSource(
                {"x": mapped_df["x"], "y": mapped_df["y"],
                 "delta": mapped_df["delta_y"], "func": mapped_df["ideal_func"]}
            )
            p.triangle("x", "y", source=src, size=10, color="black",
                       alpha=0.7, legend_label="mapped test")

        # ── Legend styling ─────────────────────────────────────────────
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "10pt"

        save(p)
        print(f"\nPlot saved to: {output_path}")

    except Exception as exc:
        raise VisualizationError(f"Failed to create Bokeh plot: {exc}")
