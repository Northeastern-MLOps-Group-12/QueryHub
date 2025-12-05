import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
from fastapi import FastAPI

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.connectors_api import router  # ✅ Router imported instead of app
from backend.models.connector_request import ConnectorRequest

# Create a FastAPI test instance
app = FastAPI()
app.include_router(router)

client = TestClient(app)


@patch("backend.connectors_api.build_graph_to_load")
def test_connect_add_connection_success(mock_build_graph, sample_request_payload):
    """Test POST /connect/addConnection returns success."""

    # Mock the graph and its invoke method
    mock_graph_instance = MagicMock()
    mock_build_graph.return_value = mock_graph_instance
    mock_graph_instance.invoke.return_value = MagicMock()  # final_state
    
    response = client.post("/connect/addConnection", json=sample_request_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "connector created" in data["message"]

    mock_build_graph.assert_called_once()
    mock_graph_instance.invoke.assert_called_once()


@patch("backend.connectors_api.build_graph_to_load")
def test_connect_add_connection_failure(mock_build_graph, sample_request_payload):
    """Test POST /connect/addConnection handles exceptions correctly."""

    # Force the graph to raise an exception
    mock_graph_instance = MagicMock()
    mock_build_graph.return_value = mock_graph_instance
    mock_graph_instance.invoke.side_effect = Exception("Something went wrong")

    response = client.post("/connect/addConnection", json=sample_request_payload)

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Something went wrong"


def test_connect_add_connection_invalid_payload():
    """Test invalid request payload raises 422 validation error."""
    
    invalid_payload = {
        "engine": "postgres"
        # missing provider + config
    }

    response = client.post("/connect/addConnection", json=invalid_payload)

    assert response.status_code == 422
    assert "detail" in response.json()
