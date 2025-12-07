"""
Chat Models and Schemas for QueryHub
Matches exact Firestore structure with Unix timestamps
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ==================== Enums ====================

class MessageSender(str, Enum):
    """Message sender type"""
    USER = "user"
    BOT = "bot"


# ==================== Request Models ====================

class CreateChatRequest(BaseModel):
    """Request model for creating a new chat"""
    chat_title: str = Field(..., min_length=1, max_length=200)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_title": "Sales Analysis Q3 2025"
            }
        }


class SendMessageRequest(BaseModel):
    """Request model for sending a message"""
    text: str = Field(..., min_length=1, max_length=5000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Show me sales data for November"
            }
        }


# ==================== Content Models ====================

class AttachmentInfo(BaseModel):
    """Metadata for file attachments"""
    has_attachment: bool
    gcs_storage_path: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "has_attachment": True,
                "gcs_storage_path": "query_results/111/1764976697/data.parquet"
            }
        }


class VisualizationInfo(BaseModel):
    """Metadata for visualization files"""
    has_visualization: bool
    gcs_storage_path: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "has_visualization": True,
                "gcs_storage_path": "visualizations/111/chinook_d9297a9e_20251205_231813/dashboard.html"
            }
        }


class MessageContent(BaseModel):
    """Content of a message"""
    text: str
    query: Optional[str] = None
    attachment: Optional[AttachmentInfo] = None
    visualization: Optional[VisualizationInfo] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I found 14,500 records matching your query.",
                "query": "SELECT * FROM sales WHERE date >= '2025-11-01'",
                "attachment": {
                    "has_attachment": True,
                    "file_name": "data.parquet",
                    "file_type": "application/octet-stream",
                    "file_size_bytes": 4503200,
                    "gcs_storage_path": "query_results/111/1764976697/data.parquet"
                },
                "visualization": {
                    "has_visualization": True,
                    "gcs_storage_path": "visualizations/111/chinook_d9297a9e/dashboard.html"
                }
            }
        }


class Message(BaseModel):
    """Individual message in a chat conversation"""
    message_id: str
    sender: MessageSender
    created_at: int  # Unix timestamp in milliseconds
    content: MessageContent
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "213c5772-3918-4396-a6fb-c4016242aa2b",
                "sender": "user",
                "created_at": 1765024422022,
                "content": {
                    "text": "Show me sales data for November"
                }
            }
        }


# ==================== Response Models ====================

class CreateChatResponse(BaseModel):
    """Response model after creating a chat"""
    chat_id: str
    created_at: str  # Unix timestamp in milliseconds
    updated_at: str  # Unix timestamp in milliseconds
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "64n6ktonTRpHZZjy4C7Y",
                "created_at": 1765024422022,
                "updated_at": 1765024422022
            }
        }



class SendMessageResponse(BaseModel):
    """Response model after sending a message (bot's reply)"""
    message_id: int
    sender: MessageSender
    created_at: str  
    content: MessageContent
    
    class Config:
        exclude_none = True
        json_schema_extra = {
            "example": {
                "message_id": 1765024422022,
                "sender": "bot",
                "created_at": 1765024422022,
                "content": {
                    "text": "I found 14,500 records matching your query.",
                    "query": "SELECT * FROM sales WHERE date >= '2025-11-01'",
                    "attachment": {
                        "has_attachment": True,
                        "file_name": "data.parquet",
                        "file_type": "application/octet-stream",
                        "file_size_bytes": 4503200,
                        "gcs_storage_path": "query_results/111/1764976697/data.parquet"
                    },
                    "visualization": {
                        "has_visualization": True,
                        "gcs_storage_path": "visualizations/111/chinook_d9297a9e/dashboard.html"
                    }
                }
            }
        }


# ==================== Chat Models ====================

class ChatSummary(BaseModel):
    """Summary information for a chat (used in list view)"""
    chat_id: str
    chat_title: str
    created_at: int  # Unix timestamp in milliseconds
    updated_at: int  # Unix timestamp in milliseconds
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "64n6ktonTRpHZZjy4C7Y",
                "chat_title": "Sales Analysis Q3",
                "created_at": 1765024422022,
                "updated_at": 1765024422022
            }
        }


class ChatDetail(BaseModel):
    """Full chat details including conversation history"""
    chat_id: str
    user_id: str
    chat_title: str
    created_at: int  # Unix timestamp in milliseconds
    updated_at: int  # Unix timestamp in milliseconds
    history: List[Message] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "64n6ktonTRpHZZjy4C7Y",
                "user_id": "user_112",
                "chat_title": "Sales Analysis Q3",
                "created_at": 1765024422022,
                "updated_at": 1765024422022,
                "history": [
                    {
                        "message_id": "213c5772-3918-4396-a6fb-c4016242aa2b",
                        "sender": "user",
                        "created_at": 1765024422022,
                        "content": {
                            "text": "Show me sales data for November"
                        }
                    }
                ]
            }
        }


class ChatsListResponse(BaseModel):
    """Response model for listing all chats"""
    chats: List[ChatSummary] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chats": [
                    {
                        "chat_id": "64n6ktonTRpHZZjy4C7Y",
                        "chat_title": "Sales Analysis Q3",
                        "created_at": 1765024422022,
                        "updated_at": 1765024422022
                    },
                    {
                        "chat_id": "chat_124",
                        "chat_title": "Marketing Trends 2025",
                        "created_at": 1765024422022,
                        "updated_at": 1765024422022
                    }
                ]
            }
        }


# ==================== File Access Models ====================

class SignedUrlResponse(BaseModel):
    """Response model for signed URL generation"""
    url: str = Field(..., description="Signed URL for temporary file access")
    expires_in_seconds: int = Field(..., description="URL validity duration in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://storage.googleapis.com/queryhub-private/query_results/111/data.parquet?Signature=abc123",
                "expires_in_seconds": 300
            }
        }


# ==================== Firestore Document Schema ====================

class ChatDocument(BaseModel):
    """
    Complete chat document structure as stored in Firestore
    Collection: 'chats'
    Database: 'queryhub-chats'
    """
    chat_id: str
    user_id: str
    chat_title: str
    created_at: int  # Unix timestamp in milliseconds
    updated_at: int  # Unix timestamp in milliseconds
    history: List[Message] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "64n6ktonTRpHZZjy4C7Y",
                "user_id": "user_112",
                "chat_title": "Sales Analysis Q3",
                "created_at": 1765024422022,
                "updated_at": 1765024422022,
                "history": [
                    {
                        "message_id": "213c5772-3918-4396-a6fb-c4016242aa2b",
                        "sender": "user",
                        "created_at": 1765024422022,
                        "content": {
                            "text": "Show me sales data for November"
                        }
                    },
                    {
                        "message_id": "37877976-e1be-4853-8e69-f4f424bb741d",
                        "sender": "bot",
                        "created_at": 1765024422022,
                        "content": {
                            "text": "I found 14,500 records matching your query.",
                            "query": "SELECT * FROM sales WHERE date >= '2025-11-01'",
                            "attachment": {
                                "has_attachment": True,
                                "file_name": "data.parquet",
                                "file_type": "application/octet-stream",
                                "file_size_bytes": 4503200,
                                "gcs_storage_path": "query_results/111/1764976697/data.parquet"
                            },
                            "visualization": {
                                "has_visualization": True,
                                "gcs_storage_path": "visualizations/111/chinook_d9297a9e/dashboard.html"
                            }
                        }
                    }
                ]
            }
        }
