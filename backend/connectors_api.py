from fastapi import FastAPI, HTTPException, Depends
from connectors.connector import Connector
from .models.connector_request import ConnectorRequest 
from ..agents.load_data_to_vector.graph import build_graph
from ..agents.load_data_to_vector.state import AgentState



app = FastAPI(title="Connector Service API")

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
