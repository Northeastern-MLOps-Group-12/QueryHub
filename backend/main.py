import os
from backend import user_api, connectors_api, chat_api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from databases.cloudsql.database import engine, Base

load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="QueryHub API",
    description="Backend API with JWT Authentication and Connector Management",
    version="1.0.0"
)

# CORS configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    print("🔑 Initializing Firebase for chat functionality...")
    chat_api.initialize_firestore()
    print("✓ Firebase initialized for chat functionality")
except Exception as e:
    print(f"⚠ Firebase initialization skipped: {e}")

# Include routers
app.include_router(user_api.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(connectors_api.router, prefix="/api/connector", tags=["Connections"])
app.include_router(chat_api.router, prefix="/api/chats", tags=["Chats"])

@app.get("/")
async def root():
    return {"message": "Welcome to QueryHub API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
