"""
Pytest configuration for IaCBench test suite.
"""
import sys
from pathlib import Path

# Add project root to PYTHONPATH so tests can import iachench and infraagent
sys.path.insert(0, str(Path(__file__).parent.parent))
