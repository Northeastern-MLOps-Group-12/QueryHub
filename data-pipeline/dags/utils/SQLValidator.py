import sqlglot

def _validate_single_sql(args):
    """
    Validate a single SQL query - must be at module level for pickling
    
    Args:
        args: tuple of (idx, sql, source)
        
    Returns:
        tuple: (idx, is_valid, error_msg, error_type)
    """
    idx, sql, source = args
    try:
        import sqlglot
        parsed = sqlglot.parse_one(sql)
        if parsed is not None:
            return (idx, True, None, None)
        else:
            return (idx, False, "Parsing returned None", "ParseError")
    except Exception as e:
        error_msg = str(e)[:200]
        error_type = type(e).__name__
        return (idx, False, error_msg, error_type)

