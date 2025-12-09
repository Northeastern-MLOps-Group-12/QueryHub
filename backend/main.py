import os
import time
import asyncio
from backend import user_api, connectors_api, chat_api, utils
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from databases.cloudsql.database import engine, Base
from agents.nl_to_data_viz.graph import initialize_agent
import warnings
warnings.filterwarnings("ignore")


# Prometheus imports
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from backend.monitoring import (
    agent_initialization_status,
    system_uptime_seconds,
    update_system_metrics
)

warnings.filterwarnings("ignore")
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="QueryHub API",
    description="Text-to-SQL API with Comprehensive Monitoring",
    version="1.0.0"
)

# CORS configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
MODE = os.getenv("MODE", "API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PROMETHEUS MONITORING SETUP
# ============================================================================

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=False,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)

instrumentator.instrument(app)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Track system uptime
startup_time = time.time()

# Background task flag
background_task_running = False

# ============================================================================
# BACKGROUND TASK FOR SYSTEM METRICS
# ============================================================================

async def update_system_metrics_task():
    """Background task to update system resource metrics every 5 seconds"""
    global background_task_running
    background_task_running = True
    
    print("🔄 Starting system metrics background task...")
    
    while background_task_running:
        try:
            update_system_metrics()
            current_uptime = time.time() - startup_time
            system_uptime_seconds.inc(5)
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"❌ Error in system metrics task: {e}")
            await asyncio.sleep(5)


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("="*70)
    print("🚀 Starting QueryHub API Server")
    print("="*70)
    
    agent_initialization_status.set(0)
    
    try:
        print("📊 Initializing LangGraph agent...")
        
        agent, memory, session_id = initialize_agent()
        chat_api.set_global_agent(agent, memory, session_id)
        
        agent_initialization_status.set(1)
        
        print("✅ Agent compiled successfully!")
        print(f"✅ Memory initialized")
        print(f"✅ Global Session ID: {session_id}")
        print(f"✅ Running in {MODE} mode")
        print(f"✅ Max retries: {3 if MODE == 'API' else 1}")
        print(f"✅ Prometheus metrics enabled at /metrics")
        
        asyncio.create_task(update_system_metrics_task())
        print("✅ System metrics background task started")
        
        print("="*70)
        print("✨ QueryHub API is ready to accept requests!")
        print("="*70)

        try:
            print("🔑 Initializing Firebase for chat functionality...")
            chat_api.initialize_firestore()
            print("✓ Firebase initialized for chat functionality")
        except Exception as e:
            print(f"⚠ Firebase initialization skipped: {e}")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        agent_initialization_status.set(0)
        raise


# ============================================================================
# SHUTDOWN EVENT
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global background_task_running
    background_task_running = False
    
    agent_initialization_status.set(0)
    
    print("="*70)
    print("🛑 Shutting down QueryHub API Server")
    print("✅ Background tasks stopped")
    print("✅ Shutdown complete")
    print("="*70)


# ============================================================================
# API ROUTERS
# ============================================================================

app.include_router(user_api.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(connectors_api.router, prefix="/api/connector", tags=["Connections"])
app.include_router(chat_api.router, prefix="/api/chats", tags=["Chats"])


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - provides basic system information"""
    uptime_seconds = time.time() - startup_time
    
    return {
        "message": "Welcome to QueryHub API",
        "version": "1.0.0",
        "status": "operational",
        "session_id": chat_api.GLOBAL_SESSION_ID if chat_api.GLOBAL_SESSION_ID else "Not initialized",
        "uptime_seconds": round(uptime_seconds, 2),
        "mode": MODE,
        "monitoring": {
            "metrics_endpoint": "/metrics",
            "health_endpoint": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - provides detailed system health status"""
    uptime_seconds = time.time() - startup_time
    
    import psutil
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "status": "healthy",
        "mode": MODE,
        "agent_ready": chat_api.AGENT is not None,
        "session_id": chat_api.GLOBAL_SESSION_ID,
        "uptime_seconds": round(uptime_seconds, 2),
        "system": {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_mb": round(memory.available / (1024 * 1024), 2),
            "disk_usage_percent": disk.percent
        },
        "monitoring": {
            "metrics_endpoint": "/metrics",
            "background_task_running": background_task_running
        }
    }


@app.get("/metrics-info")
async def metrics_info():
    """Documentation endpoint explaining available metrics"""
    return {
        "metrics_endpoint": "/metrics",
        "format": "Prometheus format",
        "categories": {
            "system_resources": [
                "queryhub_system_cpu_usage_percent",
                "queryhub_system_memory_usage_percent",
                "queryhub_process_cpu_usage_percent",
                "queryhub_process_memory_usage_bytes"
            ],
            "request_metrics": [
                "queryhub_requests_per_second",
                "queryhub_query_requests_total",
                "queryhub_query_processing_duration_seconds"
            ],
            "llm_metrics": [
                "queryhub_llm_time_to_first_token_seconds",
                "queryhub_llm_total_generation_time_seconds",
                "queryhub_llm_tokens_per_second",
                "queryhub_llm_average_time_per_token_seconds"
            ],
            "sql_complexity": [
                "queryhub_sql_complexity_distribution_total",
                "queryhub_sql_complexity_score",
                "queryhub_sql_features_detected_total",
                "queryhub_sql_join_count",
                "queryhub_sql_subquery_nesting_level"
            ],
            "component_timing": [
                "queryhub_database_selection_duration_seconds",
                "queryhub_sql_generation_duration_seconds",
                "queryhub_sql_execution_duration_seconds",
                "queryhub_visualization_generation_duration_seconds"
            ],
            "error_tracking": [
                "queryhub_sql_validation_failures_total",
                "queryhub_sql_execution_errors_total",
                "queryhub_workflow_errors_total"
            ]
        },
        "visualization_tools": {
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3001"
        }
    }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "backend.main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )
