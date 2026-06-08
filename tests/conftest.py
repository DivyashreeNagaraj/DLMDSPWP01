"""
conftest.py
-----------
Pytest configuration: ensures the src/ directory is on sys.path
before any test module is imported, so all imports resolve to the
same module objects regardless of how pytest discovers them.
"""
import sys
import os

# Add src/ to path FIRST so all modules resolve from there
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, os.path.abspath(src_path))
