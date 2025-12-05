from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter
from fastapi.responses import JSONResponse
from connectors.connector import Connector
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
from agents.nl_to_data_viz.graph import build_visualization
from .models.chat_request import ChatRequest 

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

router = APIRouter()

@router.post("/chats/messages")
def query(request: ChatRequest):
    build_visualization(request.text)
    
 