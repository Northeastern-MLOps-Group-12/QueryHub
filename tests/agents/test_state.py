import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from agents.load_data_to_vector.state import AgentState

def test_agent_state_defaults():
    """Check default values and field structure of AgentState."""
    state = AgentState()
    assert isinstance(state.engine, str)
    assert isinstance(state.creds, dict)
    assert state.engine == ""
    assert state.creds == {}
