"""
db.py
-----
Database persistence layer using SQLAlchemy + SQLite.

All datasets and results are stored in a local SQLite file so that
the analytical workflow is reproducible and queryable after execution.

Tables created
--------------
training_data      – x + four training y-columns
ideal_functions    – x + fifty ideal y-columns
test_mapping       – x, y, delta_y, ideal_func  (mapped test points)
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from exceptions import DatabaseError


class DatabaseManager:
    """
    Manages SQLite storage for all project datasets.

    Parameters
    ----------
    db_path : str
        File path for the SQLite database, e.g. 'results.db'.
        Use ':memory:' for in-memory testing.
    """

    def __init__(self, db_path: str = "regression_results.db"):
        self.db_path = db_path
        try:
            self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Could not create database engine: {exc}")

    # ------------------------------------------------------------------
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
        """
        Write a DataFrame to the SQLite database.

        Parameters
        ----------
        df         : pd.DataFrame
        table_name : str
        if_exists  : str  – 'replace' (default) | 'append' | 'fail'
        """
        try:
            df.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
            print(f"  Saved {len(df)} rows to table '{table_name}'.")
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to write table '{table_name}': {exc}")

    # ------------------------------------------------------------------
    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Read an entire table back into a DataFrame.

        Parameters
        ----------
        table_name : str

        Returns
        -------
        pd.DataFrame
        """
        try:
            return pd.read_sql_table(table_name, con=self.engine)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to read table '{table_name}': {exc}")

    # ------------------------------------------------------------------
    def list_tables(self):
        """Return a list of all table names in the database."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            return [row[0] for row in result]
