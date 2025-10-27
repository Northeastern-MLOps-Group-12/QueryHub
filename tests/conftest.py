# tests/conftest.py
import pytest

@pytest.fixture
def sample_config():
    """Return a sample PostgreSQL config dictionary."""
    return {
        "user_id": "123",
        "connection_name": "test_conn",
        "db_type": "postgres",
        "db_user": "test_user",
        "db_password": "test_pass",
        "db_name": "test_db",
        "db_host": "localhost",
        "db_port": 5432,
    }

@pytest.fixture
def sample_state():
    class DummyState:
        def __init__(self):
            self.creds = {
                "db_user": "user",
                "db_password": "pass",
                "db_host": "localhost",
                "db_name": "testdb",
                "db_type": "postgres",
                "connection_name": "test_conn",
                "user_id": 1,
            }
            self.engine = "postgres"
    return DummyState()


@pytest.fixture
def sample_request_payload():
    return {
        "engine": "postgres",
        "provider": "gcp",
        "config": {
            "user_id": "123",
            "connection_name": "test_conn",
            "db_type": "postgres",
            "db_user": "test_user",
            "db_password": "test_pass",
            "db_name": "test_db",
            "db_host": "localhost",
            "db_port": "5432"
        }
    }

