# connectors/factory.py

from .base_connector import BaseConnector
from .engines.postgres.postgres_connector import PostgresConnector
# from .engines.mysql.mysql_connector import MySQLConnector


def get_connector(engine: str) -> BaseConnector:
    """
    Factory that returns an instance of the correct connector 
    based on the database engine (e.g., postgres, mysql).

    Args:
        engine (str): The database engine.

    Returns:
        BaseConnector subclass instance.
    """
    engine = engine.lower()

    if engine == "postgres":
        return PostgresConnector()
    # elif engine == "mysql":
    #     return MySQLConnector()
    else:
        raise ValueError(f"Unsupported engine type: {engine}")
