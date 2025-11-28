import json

def structure_vector_store_data(raw_data: dict) -> dict:
    """
    Transform raw ChromaDB data into a clean, structured format.
    """
    structured = {
        "tables": {},
        "dataset_summary": None
    }
    
    for collection_name, collection_data in raw_data.items():
        ids = collection_data.get("ids", [])
        documents = collection_data.get("documents", [])
        metadatas = collection_data.get("metadatas", [])
        
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            # Check if this is the dataset summary
            if "Dataset Summary" in metadata:
                structured["dataset_summary"] = metadata["Dataset Summary"]
                continue
            
            # Parse table data
            table_name = metadata.get("table_name", doc_id)
            
            # Parse JSON strings in metadata
            columns = json.loads(metadata.get("columns", "[]"))
            primary_key = json.loads(metadata.get("primary_key", "[]"))
            foreign_keys = json.loads(metadata.get("foreign_keys", "[]"))
            indexes = json.loads(metadata.get("indexes", "[]"))
            total_rows = metadata.get("total_rows", 0)
            
            # Extract description from document (format: "TableName\n\nDescription")
            description = ""
            if document:
                parts = document.split("\n\n", 1)
                if len(parts) > 1:
                    description = parts[1]
            
            structured["tables"][table_name] = {
                "description": description,
                "total_rows": total_rows,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "columns": [
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "nullable": col["nullable"],
                        "default": col.get("default"),
                        "autoincrement": col.get("autoincrement", False)
                    }
                    for col in columns
                ]
            }
    
    return structured
