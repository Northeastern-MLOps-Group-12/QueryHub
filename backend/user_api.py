from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from databases.cloudsql.database import get_db
from .models.signin_request import SignInRequest
from .models.signin_response import SignInResponse
from .models.signup_request import SignUpRequest
from .models.user_response import UserResponse
from .utils.user_api_utils import authenticate_user, create_user_token, create_user
from .utils.user_security import get_current_user
from databases.cloudsql.models.user import User

router = APIRouter()

# Sign Up Route
@router.post("/signup", response_model=SignInResponse, status_code=201)
async def sign_up(
    user_data: SignUpRequest,
    db: Session = Depends(get_db)
):
    """
    Sign up endpoint - Create new user and return token
    """
    try:
        print("Creating user with email:", user_data.email)
        user = create_user(db, user_data)
        print("User created:", user)
        token = create_user_token(user)
        print("Token generated for user:", token)
        
        return {
            "status": 201,
            "data": {
                "user": {
                    "user_id": user.user_id,  # ← Only user_id
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email
                },
                "token": token
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during sign up" + str(e)
        )


# Sign In Route
@router.post("/signin", response_model=SignInResponse, status_code=200)
async def sign_in(
    credentials: SignInRequest,
    db: Session = Depends(get_db)
):
    """
    Sign in endpoint - Authenticate and return token
    """
    try:
        user = authenticate_user(db, credentials.email, credentials.password)
        token = create_user_token(user)
        
        return {
            "status": 200,
            "data": {
                "user": {
                    "user_id": user.user_id,  # ← Only user_id
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email
                },
                "token": token
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during sign in"
        )


# Profile Route (Protected)
@router.get("/profile", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """
    Get current user profile - Protected route requiring authentication
    """
    return {
        "user_id": current_user.user_id,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "email": current_user.email
    }


# Sign Out Route
@router.post("/signout")
async def sign_out(current_user: User = Depends(get_current_user)):
    """
    Sign out endpoint - Returns success message
    (Token invalidation handled on frontend by removing token)
    """
    return {
        "status": 200,
        "message": f"User {current_user.email} signed out successfully"
    }