# connectors/factory.py

from .base_connector import BaseConnector

# Import the specific, final connector classes from their provider files
# from .engines.postgres.providers.aws_rds import AwsRdsPostgresConnector
from .engines.postgres.providers.gcp_cloudsql import GcpCloudSqlPostgresConnector
# from .engines.mysql.providers.aws_rds import AwsRdsMySQLConnector
# from .engines.mysql.providers.gcp_cloudsql import GcpCloudSqlMySQLConnector

def get_connector(provider: str, engine: str) -> BaseConnector:
    """
    Factory that returns an instance of the correct connector 
    based on the cloud provider and database engine.

    Args:
        provider: The cloud provider (e.g., "gcp", "aws").
        engine: The database engine (e.g., "postgres", "mysql").

    Returns:
        An instance of a class that inherits from BaseConnector.
    """

    # Sanitize inputs to be lowercase
    provider = provider.lower()
    engine = engine.lower()

    if provider == "gcp" and engine == "postgres":
        return GcpCloudSqlPostgresConnector()
    # elif provider == "aws" and engine == "postgres":
    #     return AwsRdsPostgresConnector()
    # elif provider == "gcp" and engine == "mysql":
    #     return GcpCloudSqlMySQLConnector()
    # elif provider == "aws" and engine == "mysql":
    #     return AwsRdsMySQLConnector()
    else:
        # If no match is found, raise an error.
        raise ValueError(f"Unsupported provider/engine combination: {provider}/{engine}")