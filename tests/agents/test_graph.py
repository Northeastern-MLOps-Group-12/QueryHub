import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from agents.load_data_to_vector.graph import build_graph_to_load

def test_build_graph_returns_compiled_graph():
    """Ensure build_graph returns a compiled StateGraph-like object."""
    graph = build_graph_to_load()
    assert graph is not None
    # should have a method to run or traverse
    assert hasattr(graph, "invoke") or hasattr(graph, "run")
