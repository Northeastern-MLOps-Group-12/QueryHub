from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter
from fastapi.responses import JSONResponse
from connectors.connector import Connector
from databases.cloudsql.crud import get_records_by_user_id
from databases.cloudsql.database import get_db
from .models.connector_request import ConnectorRequest 
from agents.load_data_to_vector.graph import build_graph_to_load
from agents.update_data_in_vector.graph import build_graph_to_update
from agents.load_data_to_vector.state import AgentState
from fastapi.middleware.cors import CORSMiddleware
from vectorstore.chroma_vector_store import ChromaVectorStore
from backend.utils.connectors_api_utils import structure_vector_store_data
import os
import json
from connectors.engines.postgres.postgres_connector import PostgresConnector
from backend.utils.vectorstore_gcs import delete_vectorstore_from_gcs
from .utils.vectorstore_gcs import vectorstore_exists_in_gcs, download_vectorstore_from_gcs

# Connector API service entrypoint: exposes endpoints to create/test connectors.
# - ConnectorRequest: Pydantic model describing the incoming payload (engine, provider, config).
# - AgentState / build_graph: orchestrate the steps to validate and persist a connector.
# Keep this module lightweight; heavy work is delegated to agents and connector classes.

# Security & deployment notes:
# - FRONTEND_ORIGIN is read from the environment below; ensure it's set to the allowed origin(s).
# - For production, avoid allow_origins=["*"] and prefer explicit origins or a validated list.
# - Consider validating FRONTEND_ORIGIN here (e.g., ensure it's a proper URL) before using it in CORS.

# Error handling:
# - Routes should raise HTTPException for client errors; unexpected exceptions are caught in the route.
# - If more fine-grained error handling is needed, add custom exception classes in connectors/agents.

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MODEL = os.getenv("MODEL")

router = APIRouter()

EMBEDDING_MODEL = os.getenv('EMBD_MODEL_PROVIDER', 'gpt')
MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')

@router.post("/connect/addConnection")
def connect(request: ConnectorRequest):
    """
    Call the factory function to get a connector instance and test connection.
    """
    try:
        initial_state = AgentState(engine=request.engine, creds=request.config)
        graph = build_graph_to_load()

        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": 1}}
        )

        return {"success": True, "message": f"{request.engine}-{request.provider} connector created!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.put("/connect/updateConnection")
def connect(request: ConnectorRequest):
    """
    Call the factory function to get a connector instance and test connection.
    """
    try:
        initial_state = AgentState(engine=request.engine, creds=request.config)
        graph = build_graph_to_update()

        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": 1}}
        )

        return {"success": True, "message": f"{request.engine}-{request.provider} connector updated!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/connect/getAllConnections/{user_id}")
def get_all_connections(user_id: str):
    """
    Get all vector store collections for a specific user with structured data.
    Fetches from GCS if not available locally (Cloud Run scenario).
    """
    try:
        vector_stores_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectorstore", "VectorStores")
        
        # Ensure directory exists
        os.makedirs(vector_stores_dir, exist_ok=True)
        
        all_connections = {}
        
        # Get all database connections for this user from the database
        db = next(get_db())
        creds = get_records_by_user_id(db, int(user_id))
        db.close()
        
        # Process each database connection
        for cred in creds:
            db_name = cred.db_name
            
            # Check if vector store exists locally
            local_vectorstore_path = os.path.join(vector_stores_dir, f"chroma_{db_name}_{user_id}")
            local_exists = os.path.exists(local_vectorstore_path)
            
            # If not local, check GCS and download if available
            if not local_exists:
                print(f"📥 Vector store not found locally for {db_name}, checking GCS...")
                if vectorstore_exists_in_gcs(user_id=user_id, db_name=db_name):
                    print(f"✅ Found in GCS, downloading for {db_name}...")
                    download_success = download_vectorstore_from_gcs(
                        user_id=user_id,
                        db_name=db_name,
                        local_vectorstore_path=local_vectorstore_path
                    )
                    if not download_success:
                        print(f"⚠️ Failed to download vector store for {db_name}, skipping...")
                        continue
                else:
                    print(f"⚠️ Vector store not found in GCS for {db_name}, skipping...")
                    continue
            
            # Now read the vector store (either local or just downloaded)
            try:
                vector_store = ChromaVectorStore(
                    user_id=user_id, 
                    db_name=db_name, 
                    embedding_model=EMBEDDING_MODEL, 
                    model=MODEL
                )
                raw_data = vector_store.get_all_vector_stores()
                
                # Structure the data
                all_connections[db_name] = structure_vector_store_data(raw_data)
            except Exception as e:
                print(f"⚠️ Warning: Failed to read vector store for {db_name}: {e}")
                continue
        
        return {"success": True, "user_id": user_id, "connections": all_connections}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/connect/deleteConnection/{user_id}/{db_name}")
def delete_connection(user_id: int, db_name: str):
    """
    Delete a specific vector store for a user.
    """
    try:
        vector_store = ChromaVectorStore(
            user_id=user_id, 
            db_name=db_name, 
            embedding_model=EMBEDDING_MODEL, 
            model=MODEL
        )
        
        if not vector_store.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Connection '{db_name}' not found for user '{user_id}'"
            )
        
        is_vectore_deleted = vector_store.delete()

        if is_vectore_deleted:
            # Also delete credentials from the database
            connector = PostgresConnector(config={
                "user_id": user_id,
                "db_name": db_name
            })
            connector.delete_creds()

            try:
                delete_vectorstore_from_gcs(user_id=str(user_id), db_name=db_name)
            except Exception as e:
                print(f"⚠️ Warning: Failed to delete vector store from GCS: {e}")
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to delete connection '{db_name}' for user '{user_id}'"
            )

        
        return {
            "success": True, 
            "message": f"Connection '{db_name}' deleted successfully for user '{user_id}'"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))