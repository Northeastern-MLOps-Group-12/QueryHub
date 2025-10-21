from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from connectors.connector import Connector
from .models.connector_request import ConnectorRequest 
from agents.load_data_to_vector.graph import build_graph
from agents.load_data_to_vector.state import AgentState
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Connector Service API")

# Enable CORS
origins = [
    "http://localhost:5173",  # your frontend origin
    # you can add production domains here, e.g. "https://myfrontend.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] to allow all (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],         # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],         # allow all headers
)

@app.post("/connect/addConnection")
def connect(request: ConnectorRequest):
    """
    Call the factory function to get a connector instance and test connection.
    """
    try:
        initial_state = AgentState(engine=request.engine, creds=request.config)
        graph = build_graph()

        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": 1}}
        )

        return {"success": True, "message": f"{request.engine}-{request.provider} connector created!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Handle preflight OPTIONS requests globally
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return JSONResponse(status_code=200, content={"message": "OK"})
