"""
datasets.py
-----------
Dataset classes for loading and validating CSV input files.

Hierarchy
---------
BaseDataset (abstract base)
    ├── TrainingDataset
    ├── IdealFunctionDataset
    └── TestDataset

Each class loads its CSV file into a Pandas DataFrame and performs
basic validation (column presence, numeric types, no empty files).
"""

import pandas as pd
from abc import ABC, abstractmethod
from exceptions import DataLoadError


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDataset(ABC):
    """
    Abstract base class for all dataset types.

    Parameters
    ----------
    filepath : str
        Path to the CSV file to load.

    Attributes
    ----------
    filepath : str
    data     : pd.DataFrame  (populated by load())
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def load(self) -> None:
        """Load the CSV file into self.data.  Must be implemented by subclasses."""
        pass

    def _read_csv(self) -> pd.DataFrame:
        """
        Internal helper: read CSV and raise DataLoadError on failure.

        Returns
        -------
        pd.DataFrame
        """
        try:
            df = pd.read_csv(self.filepath)

            if len(df.columns) == 1:
                df = pd.read_csv(self.filepath, sep=r"\s+")

        except Exception as exc:
         raise DataLoadError(f"Could not parse CSV '{self.filepath}': {exc}")

        if df.empty:
            raise DataLoadError(f"CSV file is empty: {self.filepath}")

        return df

    def _validate_numeric(self, df: pd.DataFrame) -> None:
        """
        Ensure every column in df is numeric, raising DataLoadError if not.

        Parameters
        ----------
        df : pd.DataFrame
        """
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise DataLoadError(
                f"Non-numeric columns found in '{self.filepath}': {non_numeric}"
            )

    def describe(self) -> pd.DataFrame:
        """Return a statistical summary of the loaded data."""
        return self.data.describe()


# ---------------------------------------------------------------------------
# Training dataset  (inherits BaseDataset)
# ---------------------------------------------------------------------------

class TrainingDataset(BaseDataset):
    """
    Loads the training CSV file.

    Expected columns: x, y1, y2, y3, y4  (any capitalisation for 'x').
    The x column is renamed to lowercase 'x'.

    Parameters
    ----------
    filepath : str
        Path to train.csv.
    """

    def load(self) -> None:
        """Load and validate training data."""
        df = self._read_csv()

        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        if "x" not in df.columns:
            raise DataLoadError(f"Training CSV must contain an 'x' column: {self.filepath}")

        self._validate_numeric(df)
        self.data = df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ideal-function dataset  (inherits BaseDataset)
# ---------------------------------------------------------------------------

class IdealFunctionDataset(BaseDataset):
    """
    Loads the ideal-functions CSV file (50 candidate functions + x column).

    Parameters
    ----------
    filepath : str
        Path to ideal.csv.
    """

    def load(self) -> None:
        """Load and validate ideal-function data."""
        df = self._read_csv()
        df.columns = [c.strip().lower() for c in df.columns]

        if "x" not in df.columns:
            raise DataLoadError(f"Ideal-function CSV must contain an 'x' column: {self.filepath}")

        self._validate_numeric(df)
        self.data = df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test dataset  (inherits BaseDataset)
# ---------------------------------------------------------------------------

class TestDataset(BaseDataset):
    """
    Loads the test CSV file (x-y pairs for mapping).

    Parameters
    ----------
    filepath : str
        Path to test.csv.
    """

    def load(self) -> None:
        """Load and validate test data."""
        df = self._read_csv()
        df.columns = [c.strip().lower() for c in df.columns]

        if "x" not in df.columns or "y" not in df.columns:
            raise DataLoadError(
                f"Test CSV must contain 'x' and 'y' columns: {self.filepath}"
            )

        self._validate_numeric(df)
        self.data = df.reset_index(drop=True)
