from fastapi import APIRouter, HTTPException
from ..models.chat_request import ChatRequest
from pydantic import BaseModel
from typing import Optional
import os
import json
from pathlib import Path

# Router for chat/query endpoints
router = APIRouter()

# Global references (will be set by main.py on startup)
AGENT = None
MEMORY = None
GLOBAL_SESSION_ID = None


def set_global_agent(agent, memory, session_id):
    """
    Called by main.py on startup to set global agent instances.
    """
    global AGENT, MEMORY, GLOBAL_SESSION_ID
    AGENT = agent
    MEMORY = memory
    GLOBAL_SESSION_ID = session_id

class QueryResponse(BaseModel):
    session_id: str
    user_id: str
    db_name: str
    generated_sql: str
    result_url: Optional[str] = None
    result_gcs_path: Optional[str] = None
    dashboard_url: Optional[str] = None
    cloud_viz_files: Optional[list[dict]] = None
    execution_success: bool
    selected_db_similarity: float
    error: bool
    error_message: str
    message: str

# @router.post("/chats/messages")
def query(request: ChatRequest):
    """
    Process a user query through the LangGraph workflow.
    
    Endpoint: POST /api/chats/chats/messages
    
    Request contains:
    - text: The user's query
    - user_id: The user identifier
    
    Uses global agent, memory, and session_id initialized at startup.
    """
    try:
        # Ensure agent is initialized
        if AGENT is None or GLOBAL_SESSION_ID is None:
            raise HTTPException(
                status_code=503,
                detail="Agent not initialized. Server may still be starting up."
            )
        
        # Determine max retries based on MODE environment variable
        MODE = os.getenv("MODE", "API")
        max_retries = 3 if MODE == "API" else 1
        
        print(f"\n{'='*60}")
        print(f"Processing Query")
        print(f"{'='*60}")
        print(f"User ID: {request.user_id}")
        print(f"Global Session ID: {GLOBAL_SESSION_ID}")
        print(f"Query: {request.text}")
        print(f"Max Retries: {max_retries}")
        
        # Create state from request parameters
        state = {
            "user_id": request.user_id,
            "session_id": GLOBAL_SESSION_ID,
            "user_query": request.text,
            "db_name": "",
            "db_config": {},
            "dataset_description": "",
            "initial_necessary_table_details": [],
            "all_table_details": [],
            "final_necessary_table_details": [],
            "rephrased_query": "",
            "top_k": 4,
            "initial_top_k": 10,
            "generated_sql": "",
            "prev_sqls": [],
            "prev_errors": [],
            "max_retries": max_retries,
            "local_viz_path": os.path.join(os.getcwd(), "query_visualizations"),
            "database_metadata": {},
            "available_databases": [],
            "selected_db_similarity": 0.0,
            "database_selection_ranking": [],
            "execution_success": False,
            "error": False,
            "error_message": ""
        }
        
        # Configure agent with global session ID
        config = {"configurable": {"thread_id": GLOBAL_SESSION_ID}}
        
        # Run the workflow with global agent
        result = AGENT.invoke(input=state, config=config)

        output_file = Path("result1.json")
        
        # Save dictionary as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Saved dictionary to {output_file}")
        
        print(f"\n{'='*60}")
        print(f"Query Results")
        print(f"{'='*60}")
        print(f"User ID: {result.get('user_id', 'N/A')}")
        print(f"Error: {result.get('error', False)}")
        if result.get('error'):
            print(f"Error Message: {result.get('error_message', 'Unknown error')}")
        else:
            print(f"Selected DB: {result.get('db_name', 'N/A')}")
            print(f"Similarity: {result.get('selected_db_similarity', 0):.3f}")
            print(f"Execution Success: {result.get('execution_success', False)}")
            print(f"Generated SQL:\n{result.get('generated_sql', 'N/A')}")
        
        # Build response
        response = QueryResponse(
            session_id=GLOBAL_SESSION_ID,
            user_id=result.get("user_id", request.user_id),
            db_name=result.get("db_name", ""),
            generated_sql=result.get("generated_sql", ""),
            result_url=result.get("result_url", ""),
            result_gcs_path=result.get("result_gcs_path", ""),
            dashboard_url=result.get("visualization_metadata", {}).get("dashboard_url"),
            cloud_viz_files = result.get("cloud_viz_files", {}),
            execution_success=result.get("execution_success", False),
            selected_db_similarity=result.get("selected_db_similarity", 0.0),
            error=result.get("error", False),
            error_message=result.get("error_message", ""),
            message=(
                result.get("error_message", "An error occurred") 
                if result.get("error") 
                else "Query processed successfully" if result.get("execution_success") 
                else "Query execution failed"
            )
        )
        
        return response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def build_visualization(user_query, user_id):
    """
    Process a user query through the LangGraph workflow.
    
    Endpoint: POST /api/chats/chats/messages
    
    Request contains:
    - text: The user's query
    - user_id: The user identifier
    
    Uses global agent, memory, and session_id initialized at startup.
    """
    try:
        # Ensure agent is initialized
        if AGENT is None or GLOBAL_SESSION_ID is None:
            raise HTTPException(
                status_code=503,
                detail="Agent not initialized. Server may still be starting up."
            )
        
        # Determine max retries based on MODE environment variable
        MODE = os.getenv("MODE", "API")
        max_retries = 3 if MODE == "API" else 1
        
        print(f"\n{'='*60}")
        print(f"Processing Query")
        print(f"{'='*60}")
        print(f"User ID: {user_id}")
        print(f"Global Session ID: {GLOBAL_SESSION_ID}")
        print(f"Query: {user_query}")
        print(f"Max Retries: {max_retries}")
        
        # Create state from request parameters
        state = {
            "user_id": user_id,
            "session_id": GLOBAL_SESSION_ID,
            "user_query": user_query,
            "db_name": "",
            "db_config": {},
            "dataset_description": "",
            "initial_necessary_table_details": [],
            "all_table_details": [],
            "final_necessary_table_details": [],
            "rephrased_query": "",
            "top_k": 4,
            "initial_top_k": 10,
            "generated_sql": "",
            "prev_sqls": [],
            "prev_errors": [],
            "max_retries": max_retries,
            "local_viz_path": os.path.join(os.getcwd(), "query_visualizations"),
            "database_metadata": {},
            "available_databases": [],
            "selected_db_similarity": 0.0,
            "database_selection_ranking": [],
            "execution_success": False,
            "error": False,
            "error_message": ""
        }
        
        # Configure agent with global session ID
        config = {"configurable": {"thread_id": GLOBAL_SESSION_ID}}
        
        # Run the workflow with global agent
        result = AGENT.invoke(input=state, config=config)

        output_file = Path("result1.json")
        
        # Save dictionary as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Saved dictionary to {output_file}")
        
        print(f"\n{'='*60}")
        print(f"Query Results")
        print(f"{'='*60}")
        print(f"User ID: {result.get('user_id', 'N/A')}")
        print(f"Result URL: {result.get('result_url', 'N/A')}")
        print(f"Cloud Viz Files: {result.get('cloud_viz_files', 'N/A')}")
        print(f"Error: {result.get('error', False)}")
        if result.get('error'):
            print(f"Error Message: {result.get('error_message', 'Unknown error')}")
        else:
            print(f"Selected DB: {result.get('db_name', 'N/A')}")
            print(f"Similarity: {result.get('selected_db_similarity', 0):.3f}")
            print(f"Execution Success: {result.get('execution_success', False)}")
            print(f"Generated SQL:\n{result.get('generated_sql', 'N/A')}")
        
        # Build response
        response = QueryResponse(
            session_id=GLOBAL_SESSION_ID,
            user_id=result.get("user_id", user_id),
            db_name=result.get("db_name", ""),
            generated_sql=result.get("generated_sql", ""),
            result_url=result.get("result_url", ""),
            result_gcs_path=result.get("result_gcs_path", ""),
            dashboard_url=result.get("visualization_metadata", {}).get("dashboard_url"),
            cloud_viz_files = result.get("cloud_viz_files", []),
            execution_success=result.get("execution_success", False),
            selected_db_similarity=result.get("selected_db_similarity", 0.0),
            error=result.get("error", False),
            error_message=result.get("error_message", ""),
            message=(
                result.get("error_message", "An error occurred") 
                if result.get("error") 
                else "Query processed successfully" if result.get("execution_success") 
                else "Query execution failed"
            )
        )
        
        return response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chats/session/info")
def get_session_info():
    """
    Get information about the global session.
    
    Endpoint: GET /api/chats/chats/session/info
    """
    if GLOBAL_SESSION_ID is None:
        raise HTTPException(status_code=503, detail="Session not initialized")
    
    MODE = os.getenv("MODE", "API")
    return {
        "session_id": GLOBAL_SESSION_ID,
        "mode": MODE,
        "max_retries": 3 if MODE == "API" else 1,
        "status": "active"
    }
