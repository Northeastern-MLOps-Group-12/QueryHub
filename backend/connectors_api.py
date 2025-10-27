from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from connectors.connector import Connector
from .models.connector_request import ConnectorRequest 
from agents.load_data_to_vector.graph import build_graph
from agents.load_data_to_vector.state import AgentState
from fastapi.middleware.cors import CORSMiddleware
import os

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

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

app = FastAPI(title="Connector Service API")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],  
    allow_credentials=True,
    allow_methods=["*"],        
    allow_headers=["*"],         
)

@app.post("/connect/addConnection")
def connect(request: ConnectorRequest):
    """
    Call the factory function to get a connector instance and test connection.
    """
    try:
        initial_state = AgentState(engine=request.engine, creds=request.config)
        graph = build_graph()

        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": 1}}
        )

        return {"success": True, "message": f"{request.engine}-{request.provider} connector created!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))