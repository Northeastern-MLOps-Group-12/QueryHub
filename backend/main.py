import os
from backend import user_api, connectors_api
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

# Include routers
app.include_router(user_api.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(connectors_api.router, prefix="/api/connector", tags=["Connections"])

@app.get("/")
async def root():
    return {"message": "Welcome to QueryHub API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
