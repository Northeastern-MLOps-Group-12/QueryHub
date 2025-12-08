# Add these imports at the top
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from connectors.connector import Connector
from .state import AgentState
from .utils.serialization_helpers import sanitize_for_serialization
from databases.cloudsql.crud import get_records_by_user_and_db
from databases.cloudsql.database import get_db
from urllib.parse import quote_plus

# ✅ MONITORING IMPORTS (NEW)
from backend.monitoring import (
    track_sql_error,
    track_query_result_size,
    time_block,
    sql_execution_duration
)


# Conditional edge function for retry logic
def should_retry_sql_generation(state: AgentState) -> str:
    """
    Decides whether to retry SQL generation or end the workflow.
    Returns "retry" if error and retries available, "end" otherwise.
    """
    if state.execution_success:
        return "end"
    elif len(state.prev_errors) < state.max_retries:
        return "retry"
    else:
        return "end"
    
def get_conn_str(user_id: int, db_name: str) -> str:
        db = next(get_db())
        print(f"Connecting to DB: {db_name} for User ID: {user_id}")
        creds = get_records_by_user_and_db(db, user_id, db_name)[0]
        print(f"Retrieved Credentials: {creds}")
        user = creds.db_user
        password = creds.db_password
        host = creds.db_host
        port = creds.db_port
        db_type = creds.db_type


        # -------------------------------
        # POSTGRES
        # -------------------------------
        if db_type == "postgres":
            port = port or 5432
            password_enc = quote_plus(password)
            return f"postgresql+psycopg2://{user}:{password_enc}@{host}:{port}/{db_name}"

        # -------------------------------
        # MYSQL
        # -------------------------------
        if db_type == "mysql":
            port = port or 3306
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"

        # -------------------------------
        # CLOUD SQL SERVER
        # -------------------------------
        if db_type == "sqlserver":
            port = port or 1433
            return (
                f"mssql+pyodbc://{user}:{password}@{host}:{port}/{db}"
                "?driver=ODBC+Driver+18+for+SQL+Server"
                "&TrustServerCertificate=Yes"
            )

        # -------------------------------
        # LOCAL SQL SERVER (Windows Auth)
        # -------------------------------
        if db_type == "mssql_local":
            return (
                f"mssql+pyodbc://@{host}/{db}?"
                "trusted_connection=yes&"
                "driver=ODBC+Driver+18+for+SQL+Server&"
                "TrustServerCertificate=Yes&"
                "MARS_Connection=Yes"
            )
        return {
            "error": True,
            "error_message": f"Unsupported database type: {db_type}"
        }


def execute_sql_query(state: AgentState):
    print("_____________________execute_sql_query______________________")
    """
    Execute the generated SQL query against the database.
    Handles successful execution and error cases.
    """
    
    # Extract state variables - direct access, no .get()
    generated_sql = state.generated_sql
    prev_sqls = state.prev_sqls
    prev_errors = state.prev_errors
    
    # ✅ GET DB TYPE FOR MONITORING
    db_type = "unknown"
    try:
        db = next(get_db())
        creds = get_records_by_user_and_db(db, state.user_id, state.db_name)
        if creds:
            db_type = creds[0].db_type
    except:
        pass
    
    try:
        # Get database connection string
        conn_str = get_conn_str(state.user_id, state.db_name)
        print(f"Connection String: {conn_str}")
        
        # Create database engine
        engine = create_engine(conn_str)
        
        # ✅ TRACK SQL EXECUTION TIME
        with time_block(sql_execution_duration, db_name=state.db_name, db_type=db_type):
            # Execute query and fetch results
            with engine.connect() as connection:
                print(f"Executing SQL:\n{generated_sql}")
                result = connection.execute(text(generated_sql))
                print("SQL executed successfully.")
                
                # Fetch all results as a list of dictionaries
                columns = result.keys()
                rows = result.fetchall()
                
                # Convert to list of dicts for easier handling
                query_results = [dict(zip(columns, row)) for row in rows]
                
                # CRITICAL: Sanitize to convert Decimal/numpy types to native Python types
                query_results = sanitize_for_serialization(query_results)
        
        # ✅ TRACK RESULT SIZE
        row_count = len(query_results)
        track_query_result_size(row_count)
        print(f"📊 Returned {row_count} rows")
        
        # Return only updated fields as dict
        return {
            "error": False,
            "error_message": "",
            "query_results": query_results,
            "execution_success": True,
            "execution_error": None
        }
    
    except Exception as e:
        # Capture the error
        print(f"Error executing SQL: {e}")
        error_message = str(e)
        
        # ✅ GET COMPLEXITY TYPE FROM STATE (NEW - CRITICAL FIX)
        complexity_type = 'unknown'
        if state.sql_complexity and isinstance(state.sql_complexity, dict):
            complexity_type = state.sql_complexity.get('primary_complexity', 'unknown')
        
        print(f"📊 Error occurred in {complexity_type} query")
        
        # ✅ TRACK SQL ERROR WITH COMPLEXITY TYPE (UPDATED)
        track_sql_error(
            db_name=state.db_name,
            db_type=db_type,
            error_message=error_message,
            complexity_type=complexity_type  # ✅ NOW INCLUDES COMPLEXITY TYPE
        )
        
        # Append to error history
        updated_prev_sqls = prev_sqls + [generated_sql]
        updated_prev_errors = prev_errors + [error_message]
        
        # Return only updated fields as dict
        return {
            "error": True,
            "error_message": "SQL execution failed. Please try again.",
            "query_results": [],
            "execution_success": False,
            "execution_error": error_message,
            "prev_sqls": updated_prev_sqls,
            "prev_errors": updated_prev_errors
        }


# # Add these imports at the top
# import pandas as pd
# import sqlalchemy
# from sqlalchemy import create_engine, text
# from connectors.connector import Connector
# from .state import AgentState
# from .utils.serialization_helpers import sanitize_for_serialization
# from databases.cloudsql.crud import get_records_by_user_and_db
# from databases.cloudsql.database import get_db
# from urllib.parse import quote_plus

# # ✅ MONITORING IMPORTS (NEW)
# from backend.monitoring import (
#     track_sql_error,
#     track_query_result_size,
#     time_block,
#     sql_execution_duration
# )


# # Conditional edge function for retry logic
# def should_retry_sql_generation(state: AgentState) -> str:
#     """
#     Decides whether to retry SQL generation or end the workflow.
#     Returns "retry" if error and retries available, "end" otherwise.
#     """
#     if state.execution_success:
#         return "end"
#     elif len(state.prev_errors) < state.max_retries:
#         return "retry"
#     else:
#         return "end"
    
# def get_conn_str(user_id: int, db_name: str) -> str:
#         db = next(get_db())
#         print(f"Connecting to DB: {db_name} for User ID: {user_id}")
#         creds = get_records_by_user_and_db(db, user_id, db_name)[0]
#         print(f"Retrieved Credentials: {creds}")
#         user = creds.db_user
#         password = creds.db_password
#         host = creds.db_host
#         port = creds.db_port
#         db_type = creds.db_type


#         # -------------------------------
#         # POSTGRES
#         # -------------------------------
#         if db_type == "postgres":
#             port = port or 5432
#             password_enc = quote_plus(password)
#             return f"postgresql+psycopg2://{user}:{password_enc}@{host}:{port}/{db_name}"

#         # -------------------------------
#         # MYSQL
#         # -------------------------------
#         if db_type == "mysql":
#             port = port or 3306
#             return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"

#         # -------------------------------
#         # CLOUD SQL SERVER
#         # -------------------------------
#         if db_type == "sqlserver":
#             port = port or 1433
#             return (
#                 f"mssql+pyodbc://{user}:{password}@{host}:{port}/{db}"
#                 "?driver=ODBC+Driver+18+for+SQL+Server"
#                 "&TrustServerCertificate=Yes"
#             )

#         # -------------------------------
#         # LOCAL SQL SERVER (Windows Auth)
#         # -------------------------------
#         if db_type == "mssql_local":
#             return (
#                 f"mssql+pyodbc://@{host}/{db}?"
#                 "trusted_connection=yes&"
#                 "driver=ODBC+Driver+18+for+SQL+Server&"
#                 "TrustServerCertificate=Yes&"
#                 "MARS_Connection=Yes"
#             )
#         return {
#             "error": True,
#             "error_message": f"Unsupported database type: {db_type}"
#         }


# def execute_sql_query(state: AgentState):
#     print("_____________________execute_sql_query______________________")
#     """
#     Execute the generated SQL query against the database.
#     Handles successful execution and error cases.
#     """
    
#     # Extract state variables - direct access, no .get()
#     generated_sql = state.generated_sql
#     prev_sqls = state.prev_sqls
#     prev_errors = state.prev_errors
    
#     # ✅ GET DB TYPE FOR MONITORING (NEW)
#     db_type = "unknown"
#     try:
#         db = next(get_db())
#         creds = get_records_by_user_and_db(db, state.user_id, state.db_name)
#         if creds:
#             db_type = creds[0].db_type
#     except:
#         pass
    
#     try:
#         # Get database connection string
#         conn_str = get_conn_str(state.user_id, state.db_name)
#         print(f"Connection String: {conn_str}")
        
#         # Create database engine
#         engine = create_engine(conn_str)
        
#         # ✅ TRACK SQL EXECUTION TIME (NEW)
#         with time_block(sql_execution_duration, db_name=state.db_name, db_type=db_type):
#             # Execute query and fetch results
#             with engine.connect() as connection:
#                 print(f"Executing SQL:\n{generated_sql}")
#                 result = connection.execute(text(generated_sql))
#                 print("SQL executed successfully.")
                
#                 # Fetch all results as a list of dictionaries
#                 columns = result.keys()
#                 rows = result.fetchall()
                
#                 # Convert to list of dicts for easier handling
#                 query_results = [dict(zip(columns, row)) for row in rows]
                
#                 # CRITICAL: Sanitize to convert Decimal/numpy types to native Python types
#                 query_results = sanitize_for_serialization(query_results)
        
#         # ✅ TRACK RESULT SIZE (NEW)
#         row_count = len(query_results)
#         track_query_result_size(row_count)
#         print(f"📊 Returned {row_count} rows")
        
#         # Return only updated fields as dict
#         return {
#             "error": False,
#             "error_message": "",
#             "query_results": query_results,
#             "execution_success": True,
#             "execution_error": None
#         }
    
#     except Exception as e:
#         # Capture the error
#         print(f"Error executing SQL: {e}")
#         error_message = str(e)
        
#         # ✅ TRACK SQL ERROR (NEW)
#         track_sql_error(
#             db_name=state.db_name,
#             db_type=db_type,
#             error_message=error_message
#         )
        
#         # Append to error history
#         updated_prev_sqls = prev_sqls + [generated_sql]
#         updated_prev_errors = prev_errors + [error_message]
        
#         # Return only updated fields as dict
#         return {
#             "error": True,
#             "error_message": "SQL execution failed. Please try again.",
#             "query_results": [],
#             "execution_success": False,
#             "execution_error": error_message,
#             "prev_sqls": updated_prev_sqls,
#             "prev_errors": updated_prev_errors
#         }
