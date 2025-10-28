import os              
import uvicorn          
from backend.connectors_api import app
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # Get the port from environment variable, default to 8080 if not set
    # Useful for cloud platforms like Cloud Run that provide a PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    
    # Run the FastAPI app using uvicorn
    # host="0.0.0.0" makes it accessible from outside the container
    uvicorn.run(app, host="0.0.0.0", port=port)
