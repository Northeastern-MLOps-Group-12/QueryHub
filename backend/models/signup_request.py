from pydantic import BaseModel, EmailStr

class SignUpRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str