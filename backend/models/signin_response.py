from pydantic import BaseModel

class SignInResponse(BaseModel):
    status: int
    data: dict