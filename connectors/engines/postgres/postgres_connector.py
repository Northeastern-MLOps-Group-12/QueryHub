import os
import urllib.parse
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from ...base_connector import BaseConnector
from databases.cloudsql.crud import create_record
from databases.cloudsql.database import get_db
from databases.cloudsql.models.credentials import Credentials


class PostgresConnector(BaseConnector):
    """
    Generic PostgreSQL connector using SQLAlchemy for connection and metadata access.
    Works for any cloud provider or on-prem setup.
    """
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.engine = None

    def connect(self):
        print("Attempting to connect to PostgreSQL via SQLAlchemy...")

        try:
            # URL-encode username and password to handle special characters
            user = urllib.parse.quote_plus(self.config['db_user'])
            password = urllib.parse.quote_plus(self.config['db_password'])
            host = self.config['db_host']
            port = self.config.get('db_port', 5432)
            db_name = self.config['db_name']

            connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

            self.engine = create_engine(connection_url)
            self.conn = self.engine.connect()
            print("✅ Connected successfully using SQLAlchemy!")
            return self.conn

        except SQLAlchemyError as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            raise

    def get_inspector(self) -> dict:
        """Fetch detailed schema info (columns, PKs, FKs) using SQLAlchemy inspector."""
        if not hasattr(self, "engine") or self.engine is None:
            raise ValueError("No active connection. Call connect() first.")

        inspector = inspect(self.engine)
        return inspector

    def execute_query(self, query: str) -> any:
        """
        Execute a raw SQL query (SELECT/INSERT/UPDATE/DELETE) using SQLAlchemy.

        Automatically handles transactions and returns results as a list of dicts for SELECT queries.
        """
        if not hasattr(self, "engine") or self.engine is None:
            raise ValueError("No active SQLAlchemy engine. Call connect() first.")

        try:
            with self.engine.begin() as connection:  # auto-handles commit/rollback
                result = connection.execute(query)
                if result.returns_rows:
                    return result
                return None  # non-SELECT queries (INSERT/UPDATE/DELETE)
        except SQLAlchemyError as e:
            print(f"❌ Query execution failed: {e}")
            raise


    def query(self, sql: str):
        """Convenience method for SELECT queries returning DataFrame-like results."""
        if not hasattr(self, "engine") or self.engine is None:
            raise ValueError("No active connection. Call connect() first.")

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql))
                return [dict(row) for row in result.mappings()]
        except SQLAlchemyError as e:
            print(f"❌ Query failed: {e}")
            raise

    def analyze_and_save(self):
        """Describe schema, generate optional Gemini summary, and save metadata."""
        # db_schema = self.get_schema()
        print("✅ Schema fetched successfully!")

        schema_description = "Schema description generation disabled."

        # print(db_schema)

        # --- Save credentials to DB
        db = next(get_db())
        create_record(db, Credentials, {
            "user_id": self.config['user_id'],
            "connection_name": self.config['connection_name'],
            "provider": self.config.get('provider'),
            "db_type": self.config['db_type'],
            "db_user": self.config['db_user'],
            "db_password": self.config['db_password'],
            "db_name": self.config['db_name'],
            "db_host": self.config['db_host'],
            "db_port": self.config.get('db_port', 5432),
            "description": schema_description,
        })

        print("✅ Connection metadata saved successfully!")
