from .engines.postgres.postgres_connector import PostgresConnector
# from .engines.mysql.mysql_connector import MySQLConnector

class Connector:
    @staticmethod
    def get_connector(engine: str, config: dict):
        engine = engine.lower()

        if engine == "postgres":
            return PostgresConnector(config=config)
        # elif engine == "mysql":
        #     return MySQLConnector(config=config)
        else:
            raise ValueError(f"Unsupported engine type: {engine}")
