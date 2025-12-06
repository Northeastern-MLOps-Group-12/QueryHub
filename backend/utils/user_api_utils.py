import os
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from databases.cloudsql.models.user import User
from .user_security import verify_password, create_access_token, get_password_hash
from ..models.signup_request import SignUpRequest
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))

def authenticate_user(db: Session, email: str, password: str) -> User:
    """Authenticate user with email and password"""
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not verify_password(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    return user


def create_user(db: Session, user_data: SignUpRequest) -> User:
    """Create a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create new user
    new_user = User(
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        email=user_data.email,
        password=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


def create_user_token(user: User) -> str:
    """Create JWT token for user"""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    print("Creating access token with expiration:", access_token_expires)
    access_token = create_access_token(
        data={"user_id": user.user_id, "email": user.email},
        expires_delta=access_token_expires
    )
    print("Access token created:", access_token)
    return access_token