def get_conn_str(db_name,db_type = None):
    db_conn_str = db_conn_str = (
        f"mssql+pyodbc://@JAY/{db_name}?"
        "trusted_connection=yes&"
        "driver=ODBC+Driver+18+for+SQL+Server&"
        "TrustServerCertificate=Yes&"
        "MARS_Connection=Yes"
    )
    return db_conn_str