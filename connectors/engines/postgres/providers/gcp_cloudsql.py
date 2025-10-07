from google.cloud.sql.connector import Connector
import pg8000.dbapi
from ...postgres.base import BasePostgresConnector
from databases.cloudsql.crud import create_record
from databases.cloudsql.database import get_db
from databases.cloudsql.models.credentials import Credentials




class GcpCloudSqlPostgresConnector(BasePostgresConnector):
    """
    Specific connector for a PostgreSQL database hosted on GCP Cloud SQL.
    """
    def connect(self, config: dict):
        """
        Establishes a secure connection to a Cloud SQL instance using the
        Cloud SQL Python Connector.

        Args:
            config (dict): A dictionary containing connection details.
                           Expected keys: 'instance_connection_name', 'user',
                           'db_name', and 'password'.
        """
        print("Attempting to connect to GCP Cloud SQL PostgreSQL...")
        try:
            # password = get_secret_value(config['password_secret_ref']) 
            connector = Connector()
            self.conn = connector.connect(
                config['instance'], # e.g., "project:region:instance"
                "pg8000",
                user=config['user'],
                password=config['db_password'],
                db=config['db_name']
            )

            db = next(get_db())
            create_record(db, Credentials, {
                "user_id": config['user_id'],
                "name": config['name'],
                "instance": config['instance'],
                "provider": config['provider'],
                "db_type": config['db_type'],
                "db_user": config['db_user'],
                "db_password": config['db_password'],
                "db_name": config['db_name'],
                "user": config['user'],
            })
            print("✅ Successfully connected to GCP Cloud SQL PostgreSQL!")

        except Exception as e:
            print(f"❌ Failed to connect to GCP Cloud SQL: {e}")
            raise