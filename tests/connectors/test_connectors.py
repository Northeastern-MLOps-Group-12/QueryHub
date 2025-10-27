import pytest
from unittest.mock import MagicMock, patch
import sys
import os
# ensure project root is on sys.path so 'connectors' package can be imported during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from connectors.engines.postgres.postgres_connector import PostgresConnector

def test_connect_success(sample_config, mocker):
    connector = PostgresConnector(sample_config)

    # Mock SQLAlchemy create_engine and connection
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    mocker.patch("connectors.engines.postgres.postgres_connector.create_engine", return_value=mock_engine)

    conn = connector.connect()
    assert conn == mock_conn
    mock_engine.connect.assert_called_once()

def test_execute_query_select(sample_config, mocker):
    connector = PostgresConnector(sample_config)
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn
    connector.engine = mock_engine

    # Mock result of execute
    mock_result = MagicMock()
    mock_result.returns_rows = True
    mock_conn.execute.return_value = mock_result

    result = connector.execute_query("SELECT * FROM users;")
    assert result == mock_result

def test_query_returns_dict(sample_config, mocker):
    connector = PostgresConnector(sample_config)
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    connector.engine = mock_engine

    mock_row = {"id": 1, "name": "Ashwin"}
    mock_result = [mock_row]
    # Mock result.mappings()
    mock_conn.execute.return_value.mappings.return_value = [mock_row]

    result = connector.query("SELECT * FROM users;")
    assert result == [mock_row]

def test_analyze_and_save(sample_config, mocker):
    connector = PostgresConnector(sample_config)
    connector.engine = MagicMock()
    
    # Mock get_db and create_record
    mock_db = MagicMock()
    mocker.patch("connectors.engines.postgres.postgres_connector.get_db", return_value=iter([mock_db]))
    mock_create_record = mocker.patch("connectors.engines.postgres.postgres_connector.create_record", return_value=True)

    connector.analyze_and_save()
    # connector may not expose a Credentials attribute in tests; accept any value for that arg
    mock_create_record.assert_called_once_with(mock_db, mocker.ANY, mocker.ANY)
