# import os
# import pg8000.dbapi
# from ...base_connector import BaseConnector
# from databases.cloudsql.crud import create_record
# from databases.cloudsql.database import get_db
# from databases.cloudsql.models.credentials import Credentials
# from utils.gemini_llm import GeminiClient
# from utils.prompt_builder import PromptBuilder


# class PostgresConnector(BaseConnector):
#     """
#     Generic PostgreSQL connector that works for any cloud provider or on-prem,
#     using host/port-based connections.
#     """

#     def connect(self, config: dict):
#         print("Attempting to connect to PostgreSQL...")

#         try:
#             self.conn = pg8000.dbapi.connect(
#                 host=config['db_host'],
#                 port=config.get('db_port', 5432),
#                 user=config['db_user'],
#                 password=config['db_password'],
#                 database=config['db_name']
#             )
#             print("✅ Connected successfully!")
#             self.config = config  # store config for later use
#             return self.conn

#         except Exception as e:
#             print(f"❌ Failed to connect to PostgreSQL: {e}")
#             raise

#     # def get_schema(self) -> dict:
#     #     """Fetch the schema of the connected PostgreSQL database."""
#     #     if not hasattr(self, "conn") or self.conn is None:
#     #         raise ValueError("No active connection. Call connect() first.")

#     #     cursor = self.conn.cursor()
#     #     schema_query = """
#     #     SELECT table_name, column_name, data_type, is_nullable
#     #     FROM information_schema.columns
#     #     WHERE table_schema = 'public'
#     #     ORDER BY table_name, ordinal_position;
#     #     """
#     #     cursor.execute(schema_query)
#     #     schema_data = cursor.fetchall()
#     #     cursor.close()

#     #     db_schema = {}
#     #     for table_name, column_name, data_type, is_nullable in schema_data:
#     #         db_schema.setdefault(table_name, []).append({
#     #             'column_name': column_name,
#     #             'data_type': data_type,
#     #             'is_nullable': is_nullable
#     #         })
#     #     return db_schema


#     def get_schema(self) -> dict:
#         """Fetch the schema (columns, PK, FK) of the connected PostgreSQL database."""
#         if not hasattr(self, "conn") or self.conn is None:
#             raise ValueError("No active connection. Call connect() first.")

#         cursor = self.conn.cursor()

#         # Fetch columns
#         schema_query = """
#         SELECT 
#             c.table_name, 
#             c.column_name, 
#             c.data_type, 
#             c.is_nullable
#         FROM information_schema.columns c
#         WHERE c.table_schema = 'public'
#         ORDER BY c.table_name, c.ordinal_position;
#         """
#         cursor.execute(schema_query)
#         columns = cursor.fetchall()

#         # Fetch primary keys
#         pk_query = """
#         SELECT 
#             tc.table_name, 
#             kcu.column_name
#         FROM information_schema.table_constraints tc
#         JOIN information_schema.key_column_usage kcu
#         ON tc.constraint_name = kcu.constraint_name
#         AND tc.table_schema = kcu.table_schema
#         WHERE tc.constraint_type = 'PRIMARY KEY'
#         AND tc.table_schema = 'public';
#         """
#         cursor.execute(pk_query)
#         pks = cursor.fetchall()
#         pk_map = {}
#         for table_name, column_name in pks:
#             pk_map.setdefault(table_name, set()).add(column_name)

#         # Fetch foreign keys
#         fk_query = """
#         SELECT
#             tc.table_name AS table_name,
#             kcu.column_name AS column_name,
#             ccu.table_name AS foreign_table,
#             ccu.column_name AS foreign_column
#         FROM information_schema.table_constraints AS tc
#         JOIN information_schema.key_column_usage AS kcu
#         ON tc.constraint_name = kcu.constraint_name
#         AND tc.table_schema = kcu.table_schema
#         JOIN information_schema.constraint_column_usage AS ccu
#         ON ccu.constraint_name = tc.constraint_name
#         AND ccu.table_schema = tc.table_schema
#         WHERE tc.constraint_type = 'FOREIGN KEY'
#         AND tc.table_schema = 'public';
#         """
#         cursor.execute(fk_query)
#         fks = cursor.fetchall()
#         fk_map = {}
#         for table_name, column_name, foreign_table, foreign_column in fks:
#             fk_map.setdefault(table_name, {})[column_name] = {
#                 "references_table": foreign_table,
#                 "references_column": foreign_column
#             }

#         cursor.close()

#         # Build schema dictionary
#         db_schema = {}
#         for table_name, column_name, data_type, is_nullable in columns:
#             db_schema.setdefault(table_name, []).append({
#                 "column_name": column_name,
#                 "data_type": data_type,
#                 "is_nullable": is_nullable,
#                 "is_primary_key": column_name in pk_map.get(table_name, set()),
#                 "is_foreign_key": column_name in fk_map.get(table_name, {}),
#                 "references": fk_map.get(table_name, {}).get(column_name)
#             })

#         return db_schema

#     def execute_query(self, query: str) -> list:
#         """Execute any SQL query and return results."""
#         if not hasattr(self, "conn") or self.conn is None:
#             raise ValueError("No active connection. Call connect() first.")

#         cursor = self.conn.cursor()
#         cursor.execute(query)
#         try:
#             results = cursor.fetchall()
#         except pg8000.dbapi.ProgrammingError:
#             results = []  # e.g., for INSERT/UPDATE
#         cursor.close()
#         return results

#     def analyze_and_save(self):
#         """Main logic: describe schema, call Gemini, and save connection metadata."""
#         db_schema = self.get_schema()
#         print("✅ Schema fetched successfully!")

#         # --- Gemini (optional)
#         # prompt = PromptBuilder.schema_description_prompt(db_schema)
#         # gemini_client = GeminiClient(api_key=os.environ.get("GEMINI_API_KEY"))
#         # schema_description = gemini_client.generate(prompt)
#         schema_description = "Schema description generation disabled."
#         print(db_schema)
#         # --- Save credentials to DB
#         db = next(get_db())
#         create_record(db, Credentials, {
#             "user_id": self.config['user_id'],
#             "connection_name": self.config['connection_name'],
#             "provider": self.config.get('provider'),
#             "db_type": self.config['db_type'],
#             "db_user": self.config['db_user'],
#             "db_password": self.config['db_password'],
#             "db_name": self.config['db_name'],
#             "db_host": self.config['db_host'],
#             "db_port": self.config.get('db_port', 5432),
#             "description": schema_description,
#         })

#         print("✅ Connection metadata saved successfully!")


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
            # connection_url = "postgresql+psycopg2://postgres:QueryHub%40123@34.28.5.75:5432/queryhub"

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
        # for table_name in inspector.get_table_names(schema='public'):
        #     columns = inspector.get_columns(table_name, schema='public')
        #     pk_columns = set(inspector.get_pk_constraint(table_name, schema='public').get("constrained_columns", []))
        #     fk_constraints = inspector.get_foreign_keys(table_name, schema='public')

        #     # Build foreign key map
        #     fk_map = {}
        #     for fk in fk_constraints:
        #         for col, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
        #             fk_map[col] = {
        #                 "references_table": fk['referred_table'],
        #                 "references_column": ref_col
        #             }

        #     db_schema[table_name] = []
        #     for col in columns:
        #         db_schema[table_name].append({
        #             "column_name": col['name'],
        #             "data_type": str(col['type']),
        #             "is_nullable": col.get('nullable', True),
        #             "is_primary_key": col['name'] in pk_columns,
        #             "is_foreign_key": col['name'] in fk_map,
        #             "references": fk_map.get(col['name'])
        #         })

        # return db_schema

    def execute_query(self, query: str) -> any:
        """
        Execute a raw SQL query (SELECT/INSERT/UPDATE/DELETE) using SQLAlchemy.

        Automatically handles transactions and returns results as a list of dicts for SELECT queries.
        """
        if not hasattr(self, "engine") or self.engine is None:
            raise ValueError("No active SQLAlchemy engine. Call connect() first.")

        try:
            with self.engine.begin() as connection:  # auto-handles commit/rollback
                result = connection.execute(text(query))
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

        # --- Gemini (optional)
        # prompt = PromptBuilder.schema_description_prompt(db_schema)
        # gemini_client = GeminiClient(api_key=os.environ.get("GEMINI_API_KEY"))
        # schema_description = gemini_client.generate(prompt)
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
