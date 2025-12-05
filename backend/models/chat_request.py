from pydantic import BaseModel
from typing import Dict

class ChatRequest(BaseModel):
    """Pydantic model for incoming connector requests."""
    text: str              # e.g., "postgres" or "mysql"
    user_id:str
    