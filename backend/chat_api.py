"""
Chat API - Query Processing Endpoint - COMPLETE
Handles user queries with comprehensive monitoring
"""

from fastapi import APIRouter, HTTPException
from .models.chat_request import ChatRequest
from pydantic import BaseModel
from typing import Optional
import os
import time

from backend.monitoring import (
    track_query_request,
    track_retry_count,
    query_processing_duration,
    active_sessions,
    update_system_metrics,
    record_request
)

router = APIRouter()

AGENT = None
MEMORY = None
GLOBAL_SESSION_ID = None


def set_global_agent(agent, memory, session_id):
    """Called by main.py on startup to set global agent instances"""
    global AGENT, MEMORY, GLOBAL_SESSION_ID
    AGENT = agent
    MEMORY = memory
    GLOBAL_SESSION_ID = session_id


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    session_id: str
    user_id: str
    db_name: str
    generated_sql: str
    result_url: Optional[str] = None
    dashboard_url: Optional[str] = None
    execution_success: bool
    selected_db_similarity: float
    error: bool
    error_message: str
    message: str
    sql_complexity: Optional[dict] = None


@router.post("/chats/messages")
def query(request: ChatRequest):
    """
    Process a user query through the LangGraph workflow - COMPLETE MONITORING
    
    Tracks:
    - Request count and success rate
    - End-to-end latency
    - Requests per second
    - System metrics (CPU, memory)
    - Retry attempts
    """
    
    # Record request for RPS calculation
    current_rps = record_request()
    
    # Track start time
    start_time = time.time()
    success = False
    
    # Update system metrics
    update_system_metrics()
    
    try:
        if AGENT is None or GLOBAL_SESSION_ID is None:
            raise HTTPException(
                status_code=503,
                detail="Agent not initialized. Server may still be starting up."
            )
        
        # Increment active sessions
        active_sessions.inc()
        
        # Determine max retries
        MODE = os.getenv("MODE", "API")
        max_retries = 3 if MODE == "API" else 1
        
        print(f"\n{'='*70}")
        print(f"📥 Processing Query")
        print(f"{'='*70}")
        print(f"User ID: {request.user_id}")
        print(f"Global Session ID: {GLOBAL_SESSION_ID}")
        print(f"Query: {request.text}")
        print(f"Max Retries: {max_retries}")
        print(f"Current RPS: {current_rps:.2f}")
        
        # Create state
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
            "error_message": "",
            "sql_complexity": {}
        }
        
        # Configure agent
        config = {"configurable": {"thread_id": GLOBAL_SESSION_ID}}
        
        # Run workflow
        result = AGENT.invoke(input=state, config=config)
        
        # Determine success
        success = not result.get('error', False) and result.get('execution_success', False)
        
        # Track retry count
        retry_count = len(result.get('prev_errors', []))
        track_retry_count(retry_count)
        
        print(f"\n{'='*70}")
        print(f"📤 Query Results")
        print(f"{'='*70}")
        print(f"User ID: {result.get('user_id', 'N/A')}")
        print(f"Error: {result.get('error', False)}")
        if result.get('error'):
            print(f"Error Message: {result.get('error_message', 'Unknown error')}")
        else:
            print(f"Selected DB: {result.get('db_name', 'N/A')}")
            print(f"Similarity: {result.get('selected_db_similarity', 0):.3f}")
            print(f"Execution Success: {result.get('execution_success', False)}")
            print(f"Retry Attempts: {retry_count}")
            print(f"SQL Complexity: {result.get('sql_complexity', {}).get('primary_complexity', 'N/A')}")
            print(f"Generated SQL:\n{result.get('generated_sql', 'N/A')}")
        
        # Build response
        response = QueryResponse(
            session_id=GLOBAL_SESSION_ID,
            user_id=result.get("user_id", request.user_id),
            db_name=result.get("db_name", ""),
            generated_sql=result.get("generated_sql", ""),
            result_url=result.get("result_url"),
            dashboard_url=result.get("visualization_metadata", {}).get("dashboard_url"),
            execution_success=result.get("execution_success", False),
            selected_db_similarity=result.get("selected_db_similarity", 0.0),
            error=result.get("error", False),
            error_message=result.get("error_message", ""),
            message=(
                result.get("error_message", "An error occurred") 
                if result.get("error") 
                else "Query processed successfully" if result.get("execution_success") 
                else "Query execution failed"
            ),
            sql_complexity=result.get("sql_complexity")
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # ALWAYS track metrics
        duration = time.time() - start_time
        
        # Record query duration
        query_processing_duration.labels(user_id=request.user_id).observe(duration)
        
        # Track request count
        track_query_request(request.user_id, success)
        
        # Decrement active sessions
        active_sessions.dec()
        
        # Print final metrics
        print(f"\n{'='*70}")
        print(f"📊 Request Metrics")
        print(f"{'='*70}")
        print(f"Total Duration: {duration:.3f}s")
        print(f"Success: {success}")
        print(f"Current RPS: {current_rps:.2f}")
        print(f"{'='*70}\n")


@router.get("/chats/session/info")
def get_session_info():
    """Get information about the global session"""
    if GLOBAL_SESSION_ID is None:
        raise HTTPException(status_code=503, detail="Session not initialized")
    
    MODE = os.getenv("MODE", "API")
    return {
        "session_id": GLOBAL_SESSION_ID,
        "mode": MODE,
        "max_retries": 3 if MODE == "API" else 1,
        "status": "active"
    }