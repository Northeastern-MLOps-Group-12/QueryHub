import os
import pytest
from unittest.mock import MagicMock, patch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from agents.load_data_to_vector import load_creds_to_vectordb


def test_save_creds_to_gcp_calls_connector_methods(sample_state):
    """Ensure save_creds_to_gcp connects and saves credentials properly."""
    with patch("agents.load_data_to_vector.load_creds_to_vectordb.Connector") as mock_connector:
        mock_instance = MagicMock()
        mock_connector.get_connector.return_value = mock_instance

        updated_state = load_creds_to_vectordb.save_creds_to_gcp(sample_state)

        mock_connector.get_connector.assert_called_once_with(engine="postgres", config=sample_state.creds)
        mock_instance.connect.assert_called_once()
        mock_instance.analyze_and_save.assert_called_once()
        assert updated_state.engine == "postgres"


def test_build_vector_store_calls_correct_methods(sample_state):
    """Ensure build_vector_store constructs ChromaVectorStore and builds it."""
    with patch("agents.load_data_to_vector.load_creds_to_vectordb.Connector") as mock_connector, \
         patch("agents.load_data_to_vector.load_creds_to_vectordb.ChromaVectorStore") as mock_store:
        
        mock_connector.get_connector.return_value = MagicMock()
        mock_instance = MagicMock()
        mock_store.return_value = mock_instance

        state = load_creds_to_vectordb.build_vector_store(sample_state)

        mock_store.assert_called_once()
        mock_instance.build.assert_called_once()
        assert state.engine == "postgres"
