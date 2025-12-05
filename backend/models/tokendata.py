from pydantic import BaseModel
from typing import Optional

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None