from fastapi import FastAPI, HTTPException, Depends
from connectors.connector import get_connector
from .models.connector import ConnectorRequest  # import model

app = FastAPI(title="Connector Service API")

@app.post("/connect/")
def connect(request: ConnectorRequest):
    """
    Call the factory function to get a connector instance and test connection.
    """
    try:
        connector_instance = get_connector(
            engine=request.engine
        )

        # Optional: test connection if your connector has a connect() method
        connector_instance.connect(request.config)
        connector_instance.analyze_and_save()

        return {"success": True, "message": f"{request.engine}-{request.provider} connector created!"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
