"""
Test file for SQL guardrails validation
Run this to test various SQL queries against the security rules
"""

from agents.nl_to_data_viz.state import AgentState
from agents.nl_to_data_viz.guardrails import validate_sql_query


def test_query(description: str, sql: str):
    """Helper function to test a SQL query"""
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"{'='*70}")
    print(f"SQL: {sql}")
    print(f"-" * 70)
    
    state = AgentState(generated_sql=sql)
    result = validate_sql_query(state)
    
    if result["error"]:
        print(f"❌ BLOCKED: {result['error_message']}")
    else:
        print(f"✅ ALLOWED: Query passed validation")
    
    return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SQL GUARDRAILS VALIDATION TESTS")
    print("="*70)
    
    # Test 1: Valid SELECT query
    test_query(
        "Valid SELECT query",
        "SELECT * FROM invoices WHERE amount > 1000"
    )
    
    # Test 2: SELECT with JOINs
    test_query(
        "Valid SELECT with JOIN",
        "SELECT i.*, c.name FROM invoices i JOIN customers c ON i.customer_id = c.id"
    )
    
    # Test 3: DELETE query (should be blocked)
    test_query(
        "DELETE query (should be BLOCKED)",
        "DELETE FROM invoices WHERE id = 1"
    )
    
    # Test 4: UPDATE query (should be blocked)
    test_query(
        "UPDATE query (should be BLOCKED)",
        "UPDATE invoices SET amount = 0 WHERE id = 1"
    )
    
    # Test 5: INSERT query (should be blocked)
    test_query(
        "INSERT query (should be BLOCKED)",
        "INSERT INTO invoices (amount) VALUES (1000)"
    )
    
    # Test 6: DROP TABLE (should be blocked)
    test_query(
        "DROP TABLE (should be BLOCKED)",
        "DROP TABLE invoices"
    )
    
    # Test 7: CREATE TABLE (should be blocked)
    test_query(
        "CREATE TABLE (should be BLOCKED)",
        "CREATE TABLE test (id INT)"
    )
    
    # Test 8: ALTER TABLE (should be blocked)
    test_query(
        "ALTER TABLE (should be BLOCKED)",
        "ALTER TABLE invoices ADD COLUMN test VARCHAR(100)"
    )
    
    # Test 9: TRUNCATE (should be blocked)
    test_query(
        "TRUNCATE (should be BLOCKED)",
        "TRUNCATE TABLE invoices"
    )
    
    # Test 10: EXEC command (should be blocked)
    test_query(
        "EXEC command (should be BLOCKED)",
        "EXEC sp_executesql N'SELECT * FROM invoices'"
    )
    
    # Test 11: xp_cmdshell function (should be blocked)
    test_query(
        "xp_cmdshell function (should be BLOCKED)",
        "SELECT * FROM invoices; EXEC xp_cmdshell 'dir'"
    )
    
    # Test 12: Complex valid SELECT
    test_query(
        "Complex valid SELECT",
        """
        SELECT 
            c.name,
            COUNT(i.id) as invoice_count,
            SUM(i.amount) as total_amount
        FROM customers c
        LEFT JOIN invoices i ON c.id = i.customer_id
        WHERE i.date >= '2024-01-01'
        GROUP BY c.name
        HAVING COUNT(i.id) > 5
        ORDER BY total_amount DESC
        LIMIT 10
        """
    )
    
    # Test 13: SELECT with subquery (should be allowed)
    test_query(
        "SELECT with subquery",
        "SELECT * FROM invoices WHERE amount > (SELECT AVG(amount) FROM invoices)"
    )
    
    # Test 14: GRANT (should be blocked)
    test_query(
        "GRANT (should be BLOCKED)",
        "GRANT SELECT ON invoices TO user1"
    )
    
    # Test 15: Lowercase dangerous command
    test_query(
        "Lowercase delete (should be BLOCKED)",
        "delete from invoices where id = 1"
    )
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("Expected results:")
    print("  ✅ Tests 1, 2, 12, 13: Should be ALLOWED")
    print("  ❌ Tests 3-11, 14, 15: Should be BLOCKED")
    print("="*70 + "\n")