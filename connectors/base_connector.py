# connectors/base_connector.py

from abc import ABC, abstractmethod

class BaseConnector(ABC):
    """
    Abstract base class that defines the standard interface for all database connectors.
    """

    @abstractmethod
    def connect(self, config: dict):
        """Establish a connection to the database."""
        pass

    @abstractmethod
    def get_schema(self) -> dict:
        """Fetch the database schema as a structured dictionary."""
        pass

    @abstractmethod
    def execute_query(self, query: str) -> list:
        """Execute a SQL query and return the results as a list of dictionaries."""
        pass