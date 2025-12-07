import os
from backend import user_api, connectors_api, chat_api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from databases.cloudsql.database import engine, Base
from agents.nl_to_data_viz.graph import initialize_agent
import warnings
warnings.filterwarnings("ignore")


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
MODE = os.getenv("MODE", "API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD

@app.on_event("startup")
async def startup_event():
    """
    Initialize global agent, memory, and session ID on server startup.
    These are shared across ALL requests.
    """
    print("="*60)
    print("Starting QueryHub API Server")
    print("="*60)
    print("Initializing LangGraph agent...")
    
    # Initialize the agent (compiled once, reused forever)
    agent, memory, session_id = initialize_agent()
    
    # Pass the global instances to chat_api
    chat_api.set_global_agent(agent, memory, session_id)
    
    print("✓ Agent compiled successfully!")
    print(f"✓ Memory initialized")
    print(f"✓ Global Session ID: {session_id}")
    print(f"✓ Running in {MODE} mode")
    print(f"✓ Max retries: {3 if MODE == 'API' else 1}")
    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("="*60)
    print("Shutting down QueryHub API Server")
    print("✓ Shutdown complete")
    print("="*60)

=======
try:
    print("🔑 Initializing Firebase for chat functionality...")
    chat_api.initialize_firestore()
    print("✓ Firebase initialized for chat functionality")
except Exception as e:
    print(f"⚠ Firebase initialization skipped: {e}")
>>>>>>> 4ad9e4d796220acdee00d48d7a080978a6820302

# Include routers
app.include_router(user_api.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(connectors_api.router, prefix="/api/connector", tags=["Connections"])
app.include_router(chat_api.router, prefix="/api/chats", tags=["Chats"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to QueryHub API",
        "session_id": chat_api.GLOBAL_SESSION_ID if chat_api.GLOBAL_SESSION_ID else "Not initialized"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": MODE,
        "agent_ready": chat_api.AGENT is not None,
        "session_id": chat_api.GLOBAL_SESSION_ID
    }