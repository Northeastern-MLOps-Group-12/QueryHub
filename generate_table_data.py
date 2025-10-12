from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv
import json
import warnings
warnings.filterwarnings("ignore")
from table_summarizer import generate_description
from dataset_summarizer import generate_dataset_description
load_dotenv()

def generate_data(conn_str):
    
    engine = create_engine(conn_str)
    inspector = inspect(engine)

    tables_metadata = []

    try:
        for table in inspector.get_table_names():
            print(table)

            table_info = {"Table": table}

            # Columns
            columns_info = [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                    "default": col.get("default"),
                    "autoincrement": col.get("autoincrement")
                }
                for col in inspector.get_columns(table)
            ]
            table_info["Columns"] = columns_info

            # Primary Keys
            pk = inspector.get_pk_constraint(table)
            table_info["PrimaryKey"] = pk.get("constrained_columns")

            # Foreign Keys
            fks = inspector.get_foreign_keys(table)
            table_info["ForeignKeys"] = [
                {
                    "columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"]
                }
                for fk in fks
            ]

            # Indexes
            indexes = inspector.get_indexes(table)
            table_info["Indexes"] = [
                {
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx["unique"]
                }
                for idx in indexes
            ]

            # Row count
            with engine.connect() as conn:
                count_result = conn.execute(text(f'SELECT COUNT(*) AS total_rows FROM "{table}"'))
                row_count = count_result.fetchone()[0]
            table_info["TotalRows"] = row_count

            tables_metadata.append(table_info)

    except Exception as e:
        print("Error:", e)


    table_descriptions = []
    for table_meta in tables_metadata:
        table_name = table_meta["Table"]
        description = generate_description(table_meta)
        table_descriptions.append({"Table": table_name, "Description": description})

    with open("table_descriptions.json","w+") as f:
        json.dump(table_descriptions,f)
    f.close()

    dataset_desc = generate_dataset_description(tables_metadata)

    return tables_metadata , table_descriptions , dataset_desc