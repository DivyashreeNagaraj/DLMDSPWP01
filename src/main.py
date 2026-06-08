"""
main.py
-------
Entry point for the DLMDSPWP01 regression analysis pipeline.

Execution order
---------------
1. Load training, ideal-function, and test CSV files.
2. Persist raw datasets to SQLite.
3. Select the four best-fitting ideal functions via least squares.
4. Map test data to selected ideal functions using the sqrt(2) rule.
5. Persist the mapping results to SQLite.
6. Generate an interactive Bokeh visualisation.

Usage
-----
    python main.py

Adjust DATA_DIR and DB_PATH below if your files live elsewhere.
"""

import sys
import os

# Allow imports from the src/ folder when running main.py directly
sys.path.insert(0, os.path.dirname(__file__))

from datasets import TrainingDataset, IdealFunctionDataset, TestDataset
from modeling import IdealFunctionSelector, DataMapper
from db import DatabaseManager
from viz import build_plot
from exceptions import DataLoadError, DatabaseError, MappingError, VisualizationError

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IDEAL_CSV = os.path.join(DATA_DIR, "ideal.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
DB_PATH   = os.path.join(DATA_DIR, "regression_results.db")
PLOT_PATH = os.path.join(DATA_DIR, "regression_plot.html")


def main() -> None:
    """Run the full regression analysis pipeline."""

    # ── Step 1: Load datasets ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading datasets")
    print("=" * 60)

    try:
        train_ds = TrainingDataset(TRAIN_CSV)
        train_ds.load()
        print(f"  Training data:  {train_ds.data.shape}")

        ideal_ds = IdealFunctionDataset(IDEAL_CSV)
        ideal_ds.load()
        print(f"  Ideal functions: {ideal_ds.data.shape}")

        test_ds = TestDataset(TEST_CSV)
        test_ds.load()
        print(f"  Test data:      {test_ds.data.shape}")

    except DataLoadError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # ── Step 2: Persist raw data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Saving raw datasets to SQLite")
    print("=" * 60)

    try:
        db = DatabaseManager(DB_PATH)
        db.save_dataframe(train_ds.data, "training_data")
        db.save_dataframe(ideal_ds.data, "ideal_functions")
        db.save_dataframe(test_ds.data,  "test_data_raw")
    except DatabaseError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # ── Step 3: Select ideal functions ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Selecting ideal functions (least-squares)")
    print("=" * 60)

    try:
        selector = IdealFunctionSelector(train_ds.data, ideal_ds.data)
        selections = selector.select()
    except MappingError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("\nSelection summary:")
    for s in selections:
        print(f"  {s.train_col:4s} → {s.ideal_col:5s}  "
              f"SSE={s.sse:>12.3f}  max_dev={s.max_dev:.6f}")

    # ── Step 4: Map test data ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Mapping test data (sqrt(2) rule)")
    print("=" * 60)

    try:
        mapper = DataMapper(test_ds.data, ideal_ds.data, selections)
        mapper.map_all()
        mapped_df = mapper.to_dataframe()
    except MappingError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not mapped_df.empty:
        print(f"  Max observed deviation: {mapped_df['delta_y'].max():.6f}")

    # ── Step 5: Persist mapping results ──────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Saving mapping results to SQLite")
    print("=" * 60)

    try:
        db.save_dataframe(mapped_df, "test_mapping")
        print(f"  Tables in database: {db.list_tables()}")
    except DatabaseError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # ── Step 6: Visualise ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Building Bokeh visualisation")
    print("=" * 60)

    try:
        build_plot(
            train_df=train_ds.data,
            ideal_df=ideal_ds.data,
            selections=selections,
            mapped_df=mapped_df,
            output_path=PLOT_PATH,
        )
    except VisualizationError as exc:
        print(f"[WARNING] Visualisation failed: {exc}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
