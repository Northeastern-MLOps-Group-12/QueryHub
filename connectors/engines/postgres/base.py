import psycopg2
from abc import abstractmethod
# from ..base_connector import BaseConnector1

from ...base_connector import BaseConnector

class BasePostgresConnector(BaseConnector):
    """
    Generic parent class for all PostgreSQL connectors.
    It implements the common logic for fetching schema and executing queries.
    """
    def __init__(self):
        self.conn = None

    @abstractmethod
    def connect(self, config: dict):
        """
        This method is left abstract. The specific child classes
        (e.g., for AWS or GCP) will provide their unique implementation.
        """
        pass

    def get_schema(self) -> dict:
        """
        Fetches the database schema by querying the information_schema.
        This code is the same for any PostgreSQL database.
        """
        if not self.conn:
            raise Exception("Connection not established. Call connect() first.")
            
        schema_info = {}
        try:
            # Use a cursor to interact with the database
            with self.conn.cursor() as cur:
                query = """
                SELECT 
                    c.table_name, c.column_name, c.data_type
                FROM 
                    information_schema.columns c
                WHERE 
                    c.table_schema = 'public'
                ORDER BY 
                    c.table_name, c.ordinal_position;
                """
                cur.execute(query)
                
                for row in cur.fetchall():
                    table_name, column_name, data_type = row
                    if table_name not in schema_info:
                        schema_info[table_name] = []
                    
                    schema_info[table_name].append(f"{column_name} ({data_type})")
            
            return schema_info

        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error fetching schema: {error}")
            return {}

    def execute_query(self, query: str) -> list:
        """

        Executes a SQL query and returns the results as a list of dictionaries.
        This code is the same for any PostgreSQL database.
        """
        if not self.conn:
            raise Exception("Connection not established. Call connect() first.")
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query)
                # Fetch all rows and convert them to a list of dictionaries
                results = [dict(row) for row in cur.fetchall()]
                return results

        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error executing query: {error}")
            raise