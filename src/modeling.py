"""
modeling.py
-----------
Core analytical logic for the DLMDSPWP01 assignment.

Classes
-------
IdealFunctionSelector
    Selects the four best-fitting ideal functions from 50 candidates
    using the least-squares (minimum SSE) criterion.

DataMapper
    Maps each test x-y pair to one of the four chosen ideal functions
    if the absolute deviation does not exceed max_training_deviation * sqrt(2).
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

from exceptions import MappingError


@dataclass
class SelectedFunction:
    """
    Metadata for one ideal function chosen for a training dataset.

    Attributes
    ----------
    train_col : str   – e.g. 'y1'
    ideal_col : str   – e.g. 'y23'
    sse       : float – sum of squared errors vs. training data
    max_dev   : float – maximum absolute deviation between train and ideal
    """
    train_col: str
    ideal_col: str
    sse: float
    max_dev: float


class IdealFunctionSelector:
    """
    Selects the best-matching ideal function for each training column.

    For each of the four training y-columns, every one of the 50 ideal
    y-columns is evaluated.  The ideal column with the lowest SSE is chosen.

    Parameters
    ----------
    train_df : pd.DataFrame  – 'x' + y-columns from TrainingDataset
    ideal_df : pd.DataFrame  – 'x' + y-columns from IdealFunctionDataset
    """

    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        self.train_df = train_df.copy()
        self.ideal_df = ideal_df.copy()
        self.results: List[SelectedFunction] = []

        # Build an x-indexed lookup for the ideal dataframe once
        self._ideal_indexed = self.ideal_df.set_index("x")
        self._train_indexed = self.train_df.set_index("x")

    # ------------------------------------------------------------------
    def _compute_sse(self, t_col: str, i_col: str) -> Tuple[float, float]:
        """
        Compute SSE and max absolute deviation using index alignment.

        Parameters
        ----------
        t_col : str – training column name
        i_col : str – ideal column name

        Returns
        -------
        (sse, max_abs_deviation)
        """
        t_series = self._train_indexed[t_col]
        i_series = self._ideal_indexed[i_col]

        # Align on x-index
        t_aligned, i_aligned = t_series.align(i_series, join="inner")
        diff = t_aligned.values - i_aligned.values

        sse = float(np.sum(diff ** 2))
        max_dev = float(np.max(np.abs(diff)))
        return sse, max_dev

    # ------------------------------------------------------------------
    def select(self) -> List[SelectedFunction]:
        """
        Run selection for all four training columns.

        Returns
        -------
        List[SelectedFunction] – four entries, one per training column.
        """
        train_y_cols = [c for c in self.train_df.columns if c != "x"]
        ideal_y_cols = [c for c in self.ideal_df.columns if c != "x"]

        self.results = []

        for t_col in train_y_cols:
            best_ideal: Optional[str] = None
            best_sse = math.inf
            best_max_dev = math.inf

            for i_col in ideal_y_cols:
                try:
                    sse, max_dev = self._compute_sse(t_col, i_col)
                except Exception as exc:
                    print(f"Warning: could not compare {t_col} vs {i_col}: {exc}")
                    continue

                if sse < best_sse:
                    best_sse = sse
                    best_ideal = i_col
                    best_max_dev = max_dev

            if best_ideal is None:
                raise MappingError(
                    f"Could not find any ideal function for training column '{t_col}'."
                )

            self.results.append(
                SelectedFunction(
                    train_col=t_col,
                    ideal_col=best_ideal,
                    sse=round(best_sse, 4),
                    max_dev=round(best_max_dev, 6),
                )
            )
            print(
                f"  {t_col} → {best_ideal}  "
                f"(SSE={best_sse:.3f}, max_dev={best_max_dev:.6f})"
            )

        return self.results


@dataclass
class MappingResult:
    """
    Single row of the test-mapping output table.

    Attributes
    ----------
    x         : float
    y         : float
    delta_y   : float – |y_test – y_ideal|
    ideal_col : str   – matched ideal function name
    threshold : float – max_dev * sqrt(2) used for this match
    """
    x: float
    y: float
    delta_y: float
    ideal_col: str
    threshold: float


class DataMapper:
    """
    Maps each test data point to one of the four selected ideal functions.

    A point is accepted if:
        |y_test – y_ideal(x)| <= max_training_deviation * sqrt(2)

    Parameters
    ----------
    test_df    : pd.DataFrame           – from TestDataset
    ideal_df   : pd.DataFrame           – from IdealFunctionDataset
    selections : List[SelectedFunction] – from IdealFunctionSelector.select()
    """

    SQRT2 = math.sqrt(2)

    def __init__(
        self,
        test_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        selections: List[SelectedFunction],
    ):
        self.test_df = test_df.copy()
        self.ideal_df = ideal_df.copy()
        self.selections = selections
        self.mapped: List[MappingResult] = []
        self.unmatched: List[Tuple[float, float]] = []
        self._ideal_indexed = self.ideal_df.set_index("x")

    # ------------------------------------------------------------------
    def _lookup_ideal_y(self, ideal_col: str, x_val: float) -> Optional[float]:
        """
        Return the y-value of ideal_col at x_val, or None if x not present.
        """
        if ideal_col not in self._ideal_indexed.columns:
            return None
        try:
            # Find closest x using index
            idx = self._ideal_indexed.index
            mask = np.isclose(idx.values.astype(float), x_val, atol=1e-9)
            if not mask.any():
                return None
            return float(self._ideal_indexed.loc[idx[mask][0], ideal_col])
        except Exception:
            return None

    # ------------------------------------------------------------------
    def map_all(self) -> List[MappingResult]:
        """
        Iterate over every test point and assign to the best ideal function.

        Returns
        -------
        List[MappingResult]
        """
        self.mapped = []
        self.unmatched = []

        for _, row in self.test_df.iterrows():
            x_val = float(row["x"])
            y_val = float(row["y"])
            best: Optional[MappingResult] = None

            for sel in self.selections:
                y_ideal = self._lookup_ideal_y(sel.ideal_col, x_val)
                if y_ideal is None:
                    continue

                delta = abs(y_val - y_ideal)
                threshold = sel.max_dev * self.SQRT2

                if delta <= threshold:
                    if best is None or delta < best.delta_y:
                        best = MappingResult(
                            x=x_val,
                            y=y_val,
                            delta_y=round(delta, 6),
                            ideal_col=sel.ideal_col,
                            threshold=round(threshold, 6),
                        )

            if best is not None:
                self.mapped.append(best)
            else:
                self.unmatched.append((x_val, y_val))

        print(
            f"\nMapping complete: {len(self.mapped)} assigned, "
            f"{len(self.unmatched)} unassigned."
        )
        return self.mapped

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert mapping results to the required 4-column output table."""
        if not self.mapped:
            return pd.DataFrame(columns=["x", "y", "delta_y", "ideal_func"])

        return pd.DataFrame(
            {
                "x":          [r.x         for r in self.mapped],
                "y":          [r.y         for r in self.mapped],
                "delta_y":    [r.delta_y   for r in self.mapped],
                "ideal_func": [r.ideal_col for r in self.mapped],
            }
        )
