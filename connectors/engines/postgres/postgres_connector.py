import os
import pg8000.dbapi
from ...base_connector import BaseConnector
from databases.cloudsql.crud import create_record
from databases.cloudsql.database import get_db
from databases.cloudsql.models.credentials import Credentials
from utils.gemini_llm import GeminiClient
from utils.prompt_builder import PromptBuilder


class PostgresConnector(BaseConnector):
    """
    Generic PostgreSQL connector that works for any cloud provider or on-prem,
    using host/port-based connections.
    """

    def connect(self, config: dict):
        print("Attempting to connect to PostgreSQL...")

        try:
            self.conn = pg8000.dbapi.connect(
                host=config['db_host'],
                port=config.get('db_port', 5432),
                user=config['db_user'],
                password=config['db_password'],
                database=config['db_name']
            )
            print("✅ Connected successfully!")
            self.config = config  # store config for later use
            return self.conn

        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            raise

    def get_schema(self) -> dict:
        """Fetch the schema of the connected PostgreSQL database."""
        if not hasattr(self, "conn") or self.conn is None:
            raise ValueError("No active connection. Call connect() first.")

        cursor = self.conn.cursor()
        schema_query = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
        """
        cursor.execute(schema_query)
        schema_data = cursor.fetchall()
        cursor.close()

        db_schema = {}
        for table_name, column_name, data_type, is_nullable in schema_data:
            db_schema.setdefault(table_name, []).append({
                'column_name': column_name,
                'data_type': data_type,
                'is_nullable': is_nullable
            })
        return db_schema

    def execute_query(self, query: str) -> list:
        """Execute any SQL query and return results."""
        if not hasattr(self, "conn") or self.conn is None:
            raise ValueError("No active connection. Call connect() first.")

        cursor = self.conn.cursor()
        cursor.execute(query)
        try:
            results = cursor.fetchall()
        except pg8000.dbapi.ProgrammingError:
            results = []  # e.g., for INSERT/UPDATE
        cursor.close()
        return results

    def analyze_and_save(self):
        """Main logic: describe schema, call Gemini, and save connection metadata."""
        db_schema = self.get_schema()
        print("✅ Schema fetched successfully!")

        # --- Gemini (optional)
        prompt = PromptBuilder.schema_description_prompt(db_schema)
        gemini_client = GeminiClient(api_key=os.environ.get("GEMINI_API_KEY"))
        schema_description = gemini_client.generate(prompt)
        # schema_description = "Schema description generation disabled."

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
