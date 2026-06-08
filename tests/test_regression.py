"""
test_regression.py
------------------
Unit tests for the DLMDSPWP01 regression pipeline.

Tests cover:
- Dataset loading and validation (DataLoadError on bad inputs)
- SSE computation correctness
- Ideal function selection (exactly 4 results, non-negative SSE)
- Test data mapping (threshold rule respected, correct output columns)
- Database persistence (save and reload round-trip)

Run with:  pytest tests/test_regression.py -v
"""

import math
import pytest
import numpy as np
import pandas as pd


from datasets import TrainingDataset, IdealFunctionDataset
from datasets import TestDataset as CsvTestDataset   # aliased to avoid pytest clash
from modeling import IdealFunctionSelector, DataMapper as TestDataMapper, SelectedFunction
from db import DatabaseManager
from exceptions import DataLoadError


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_train_df() -> pd.DataFrame:
    """4 training functions over x = -2..2 (20 points)."""
    x = np.linspace(-2, 2, 20)
    return pd.DataFrame({
        "x":  x,
        "y1": x,
        "y2": x ** 2,
        "y3": np.sin(x),
        "y4": np.cos(x),
    })


def make_ideal_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    50 ideal functions; columns y1..y4 are exact copies of the training
    columns so the selector has clear winners.  y5..y50 are noise.
    """
    x = train_df["x"].values
    data = {"x": x}
    # Exact matches (column names match training column names intentionally)
    data["y1"]  = train_df["y1"].values
    data["y2"]  = train_df["y2"].values
    data["y3"]  = train_df["y3"].values
    data["y4"]  = train_df["y4"].values
    rng = np.random.default_rng(0)
    for i in range(5, 51):
        data[f"y{i}"] = rng.standard_normal(len(x)) * 10
    return pd.DataFrame(data)


def make_test_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Test points at training x-values with tiny noise around y1."""
    rng = np.random.default_rng(42)
    subset = train_df.sample(10, random_state=0)
    return pd.DataFrame({
        "x": subset["x"].values,
        "y": subset["y1"].values + rng.normal(0, 0.01, 10),
    })


def make_selections(train_df, ideal_df) -> list:
    """Run the selector and return results."""
    sel = IdealFunctionSelector(train_df, ideal_df)
    return sel.select()


# ── Dataset loading tests ────────────────────────────────────────────────────

class TestDatasetLoading:

    def test_training_dataset_loads_csv(self, tmp_path):
        """TrainingDataset should load a valid CSV without error."""
        csv = tmp_path / "train.csv"
        make_train_df().to_csv(csv, index=False)
        ds = TrainingDataset(str(csv))
        ds.load()
        assert not ds.data.empty
        assert "x" in ds.data.columns

    def test_missing_file_raises_data_load_error(self, tmp_path):
        """A non-existent path should raise DataLoadError."""
        ds = TrainingDataset(str(tmp_path / "nope.csv"))
        with pytest.raises(DataLoadError):
            ds.load()

    def test_empty_csv_raises_data_load_error(self, tmp_path):
        """An empty CSV file should raise DataLoadError."""
        csv = tmp_path / "empty.csv"
        csv.write_text("")
        ds = TrainingDataset(str(csv))
        with pytest.raises(DataLoadError):
            ds.load()

    def test_test_dataset_requires_y_column(self, tmp_path):
        """TestDataset without a 'y' column should raise DataLoadError."""
        csv = tmp_path / "test_bad.csv"
        pd.DataFrame({"x": [1, 2], "z": [3, 4]}).to_csv(csv, index=False)
        ds = CsvTestDataset(str(csv))
        with pytest.raises(DataLoadError):
            ds.load()


# ── Ideal function selector tests ────────────────────────────────────────────

class TestIdealFunctionSelector:

    def setup_method(self):
        self.train_df = make_train_df()
        self.ideal_df = make_ideal_df(self.train_df)

    def test_select_returns_four_functions(self):
        """Selector must return exactly four SelectedFunction objects."""
        results = make_selections(self.train_df, self.ideal_df)
        assert len(results) == 4

    def test_sse_is_non_negative(self):
        """SSE must be >= 0 for all selected functions."""
        for r in make_selections(self.train_df, self.ideal_df):
            assert r.sse >= 0.0, f"Negative SSE: {r}"

    def test_max_dev_is_non_negative(self):
        """max_dev must be >= 0 for all selected functions."""
        for r in make_selections(self.train_df, self.ideal_df):
            assert r.max_dev >= 0.0, f"Negative max_dev: {r}"

    def test_perfect_match_has_near_zero_sse(self):
        """
        When ideal set contains exact copies of training columns,
        the SSE should be effectively zero (< 1e-10).
        """
        results = make_selections(self.train_df, self.ideal_df)
        for r in results:
            assert r.sse < 1e-10, (
                f"Expected near-zero SSE, got SSE={r.sse} "
                f"for {r.train_col} → {r.ideal_col}"
            )

    def test_selected_ideal_col_exists_in_ideal_df(self):
        """Every chosen ideal_col must be a column in the ideal DataFrame."""
        results = make_selections(self.train_df, self.ideal_df)
        for r in results:
            assert r.ideal_col in self.ideal_df.columns, (
                f"Selected column '{r.ideal_col}' not found in ideal_df"
            )


# ── Mapper tests ─────────────────────────────────────────────────────────────

class TestTestDataMapper:

    def setup_method(self):
        self.train_df  = make_train_df()
        self.ideal_df  = make_ideal_df(self.train_df)
        self.test_df   = make_test_df(self.train_df)
        self.selections = make_selections(self.train_df, self.ideal_df)
        self.mapper = TestDataMapper(self.test_df, self.ideal_df, self.selections)

    def test_mapping_runs_without_error(self):
        """map_all() should return a list without raising."""
        results = self.mapper.map_all()
        assert isinstance(results, list)

    def test_mapping_respects_threshold(self):
        """
        Every mapped point must satisfy delta_y <= max_dev * sqrt(2).
        Verified by checking each MappingResult's stored threshold value.
        """
        results = self.mapper.map_all()
        if not results:
            pytest.skip("No points mapped in this fixture.")
        tol = 1e-9
        for r in results:
            assert r.delta_y <= r.threshold + tol, (
                f"Threshold violated: delta={r.delta_y} > threshold={r.threshold}"
            )

    def test_to_dataframe_has_correct_columns(self):
        """Output DataFrame must have exactly: x, y, delta_y, ideal_func."""
        self.mapper.map_all()
        df = self.mapper.to_dataframe()
        assert set(df.columns) == {"x", "y", "delta_y", "ideal_func"}

    def test_mapped_points_come_from_test_data(self):
        """Every mapped x value must appear in the original test data."""
        self.mapper.map_all()
        df = self.mapper.to_dataframe()
        test_x_set = set(np.round(self.test_df["x"].values, 6))
        for x_val in df["x"].values:
            assert round(x_val, 6) in test_x_set


# ── Database tests ────────────────────────────────────────────────────────────

class TestDatabaseManager:

    def test_save_and_reload(self, tmp_path):
        """A saved DataFrame should be retrievable and equal to the original."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        original = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        db.save_dataframe(original, "my_table")
        reloaded = db.load_table("my_table")
        pd.testing.assert_frame_equal(original, reloaded)

    def test_list_tables(self, tmp_path):
        """list_tables() should return names of all tables saved."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        db.save_dataframe(pd.DataFrame({"x": [1]}), "alpha")
        db.save_dataframe(pd.DataFrame({"x": [2]}), "beta")
        tables = db.list_tables()
        assert "alpha" in tables
        assert "beta" in tables

    def test_overwrite_table(self, tmp_path):
        """Saving with if_exists='replace' should overwrite old data."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        db.save_dataframe(pd.DataFrame({"v": [1, 2, 3]}), "tbl")
        db.save_dataframe(pd.DataFrame({"v": [99]}), "tbl", if_exists="replace")
        df = db.load_table("tbl")
        assert len(df) == 1
        assert df["v"].iloc[0] == 99
