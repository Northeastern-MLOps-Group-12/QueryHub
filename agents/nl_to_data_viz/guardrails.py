"""
SQL Guardrails Module

Simple security checks for generated SQL queries.
Blocks dangerous commands and functions only.
"""

import re
from typing import Dict
from .state import AgentState

# ✅ MONITORING IMPORTS (NEW)
from backend.monitoring import (
    track_validation_failure,
    time_block,
    sql_validation_duration
)


# List of dangerous SQL commands that modify data or schema
DANGEROUS_COMMANDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", 
    "TRUNCATE", "REPLACE", "MERGE", "GRANT", "REVOKE",
    "EXEC", "EXECUTE", "CALL", "RENAME", "COMMENT"
]

# List of dangerous SQL functions that should be blocked
DANGEROUS_FUNCTIONS = [
    "xp_cmdshell",  # SQL Server command execution
    "pg_sleep",     # PostgreSQL sleep (potential DoS)
    "benchmark",    # MySQL benchmark (potential DoS)
    "load_file",    # MySQL file reading
    "into outfile", # MySQL file writing
    "into dumpfile" # MySQL file writing
]


def validate_sql_query(state: AgentState) -> Dict:
    """
    Validate the generated SQL query for dangerous commands and functions.
    
    Args:
        state: AgentState containing the generated_sql to validate
    
    Returns:
        Dict with error flag and message if validation fails
    """
    sql = state.generated_sql.strip()
    
    # Convert to uppercase for case-insensitive checking
    sql_upper = sql.upper()
    
    # ✅ TRACK VALIDATION TIME (NEW)
    with time_block(sql_validation_duration):
        # Check for dangerous commands
        for command in DANGEROUS_COMMANDS:
            pattern = r'\b' + re.escape(command) + r'\b'
            if re.search(pattern, sql_upper):
                # ✅ TRACK VALIDATION FAILURE (NEW)
                track_validation_failure(f"dangerous_command_{command.lower()}")
                
                print(f"❌ Validation failed: {command} not allowed")
                return {
                    "error": True,
                    "error_message": "Operation not allowed"
                }
        
        # Check for dangerous functions
        for func in DANGEROUS_FUNCTIONS:
            if func.upper() in sql_upper:
                # ✅ TRACK VALIDATION FAILURE (NEW)
                track_validation_failure(f"dangerous_function_{func.lower()}")
                
                print(f"❌ Validation failed: {func} not allowed")
                return {
                    "error": True,
                    "error_message": "Operation not allowed"
                }
    
    # All checks passed
    print(f"✓ SQL query passed security validation")
    return {
        "error": False,
        "error_message": ""
    }