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
from google.cloud import firestore
from google.oauth2 import service_account
from .user_api import get_current_user
from databases.cloudsql.models.user import User
from .models.chat_model import (
    CreateChatRequest,
    CreateChatResponse,
    SendMessageRequest,
    SendMessageResponse,
    ChatsListResponse,
    SignedUrlResponse,
    ChatDetail,
)
from backend.utils import chat_utils

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

# Global clients
db = None
bucket = None

@router.post("/chats/messages")
def query(request: ChatRequest):
    build_visualization(request.text, request.user_id)


# ==================== Firestore Initialization ====================

def initialize_firestore():
    """Initialize Firestore with service account"""
    global db, bucket
    
    try:
        print("🔑 Initializing Firestore...")
        cred_path = os.getenv("GCS_CREDENTIALS_PATH")
        
        if not os.path.exists(cred_path):
            print(f"⚠️  Firestore credentials not found at {cred_path}")
            return
        
        credentials = service_account.Credentials.from_service_account_file(cred_path)
        
        project_id = os.getenv("PROJECT_ID")
        database_id = os.getenv("FIREBASE_DATABASE_ID")
        storage_bucket_name = os.getenv("GCS_BUCKET_NAME")
        
        # Init Firestore
        db = firestore.Client(
            project=project_id,
            database=database_id,
            credentials=credentials
        )
        
        # Init Storage
        from google.cloud import storage
        storage_client = storage.Client(credentials=credentials, project=project_id)
        bucket = storage_client.bucket(storage_bucket_name)
        
        print(f"✅ Firestore initialized (Project: {project_id}, DB: {database_id})")
        
    except Exception as e:
        print(f"❌ Firestore init error: {str(e)}")


def check_firestore():
    """Check if Firestore is ready"""
    if db is None:
        raise HTTPException(status_code=503, detail="Firestore not initialized")
    return db


# ==================== API Endpoints ====================

@router.get("/chats", response_model=ChatsListResponse)
async def list_chats(
    # current_user: User = Depends(get_current_user),
    user_id,
    firestore_db = Depends(check_firestore)
):
    """
    GET /api/chats
    Lists all chats for authenticated user
    
    Response:
    {
        "chats": [
            {
                "chat_id": "64n6ktonTRpHZZjy4C7Y",
                "chat_title": "Sales Analysis Q3",
                "created_at": 1765024422022,
                "updated_at": 1765024422022
            }
        ]
    }
    """
    chats = chat_utils.list_user_chats(firestore_db, user_id)
    return ChatsListResponse(chats=chats)


@router.post("/chats", response_model=CreateChatResponse, status_code=201)
async def create_chat(
    request: CreateChatRequest,
    # current_user: User = Depends(get_current_user),
    user_id,
    firestore_db = Depends(check_firestore)
):
    """
    POST /api/chats
    Creates new chat
    
    Request:
    {
        "chat_title": "Sales Analysis Q3"
    }
    
    Response:
    {
        "chat_id": "64n6ktonTRpHZZjy4C7Y",
        "created_at": 1765024422022
    }
    """
    chat_id, created_at, updated_at = chat_utils.create_chat(firestore_db, user_id, request.chat_title)
    return CreateChatResponse(chat_id=chat_id, created_at=created_at, updated_at=updated_at)


@router.get("/chats/{chat_id}", response_model=ChatDetail)
async def get_chat(
    chat_id: str,
    # current_user: User = Depends(get_current_user),
    user_id,
    firestore_db = Depends(check_firestore)
):
    """
    GET /api/chats/{chat_id}
    Returns full chat with history
    
    Response:
    {
        "chat_id": "64n6ktonTRpHZZjy4C7Y",
        "user_id": "user_112",
        "chat_title": "Sales Analysis Q3",
        "created_at": 1765024422022,
        "updated_at": 1765024422022,
        "history": [...]
    }
    """
    return chat_utils.get_chat_detail(firestore_db, chat_id, user_id)


@router.post("/chats/{chat_id}/messages", response_model=SendMessageResponse, status_code=201)
async def send_message(
    chat_id: str,
    request: SendMessageRequest,
    # current_user: User = Depends(get_current_user),
    user_id,
    firestore_db = Depends(check_firestore)
):
    """
    POST /api/chats/{chat_id}/messages
    Sends message and gets bot response
    
    Request:
    {
        "text": "Show me sales data for November"
    }
    
    Response:
    {
        "message_id": "37877976-e1be-4853-8e69-f4f424bb741d",
        "sender": "bot",
        "created_at": 1765024422022,
        "content": {
            "text": "I found 14,500 records...",
            "query": "SELECT * FROM sales...",
            "attachment": {...},
            "visualization": {...}
        }
    }
    """
    # user_id = f"user_{current_user.user_id}"
    bot_message = chat_utils.send_message(firestore_db, chat_id, user_id, request.text)
    if isinstance(bot_message, dict):
        return SendMessageResponse(**bot_message)
    else:
        return SendMessageResponse(**bot_message.dict()) 


@router.get("/messages/{message_id}/attachment", response_model=SignedUrlResponse)
async def get_attachment_url(
    message_id: int,
    # current_user: User = Depends(get_current_user),
    user_id,
    firestore_db = Depends(check_firestore)
):
    """
    GET /api/messages/{message_id}/attachment
    Gets signed URL for attachment (5 min)
    
    Response:
    {
        "url": "https://storage.googleapis.com/...",
        "expires_in_seconds": 300
    }
    """
    # user_id = f"user_{current_user.user_id}"
    url, expires = chat_utils.get_attachment_url(firestore_db, bucket, message_id, user_id)
    return SignedUrlResponse(url=url, expires_in_seconds=expires)


@router.get("/messages/{message_id}/visualization", response_model=SignedUrlResponse)
async def get_visualization_url(
    message_id: int,
    # current_user: User = Depends(get_current_user),
    user_id,
    firestore_db = Depends(check_firestore)
):
    """
    GET /api/messages/{message_id}/visualization
    Gets signed URL for visualization (1 min)
    
    Response:
    {
        "url": "https://storage.googleapis.com/...",
        "expires_in_seconds": 60
    }
    """
    # user_id = f"user_{current_user.user_id}"
    url, expires = chat_utils.get_visualization_url(firestore_db, bucket, message_id, user_id)
    return SignedUrlResponse(url=url, expires_in_seconds=expires)


@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    # current_user: User = Depends(get_current_user),
    user_id,
    firestore_db = Depends(check_firestore)
):
    """
    DELETE /api/chats/{chat_id}
    Deletes chat
    
    Response:
    {
        "message": "Chat deleted successfully",
        "chat_id": "64n6ktonTRpHZZjy4C7Y"
    }
    """
    # user_id = f"user_{current_user.user_id}"
    return chat_utils.delete_chat(firestore_db, chat_id, user_id)
 