"""
exceptions.py
-------------
Custom exception classes for the DLMDSPWP01 regression project.
Provides domain-specific error types for clearer error handling.
"""


class DataLoadError(Exception):
    """Raised when a CSV file cannot be loaded or is malformed."""
    pass


class DatabaseError(Exception):
    """Raised when a database operation fails."""
    pass


class MappingError(Exception):
    """Raised when test-data mapping encounters an unexpected condition."""
    pass


class VisualizationError(Exception):
    """Raised when Bokeh plot generation fails."""
    pass
