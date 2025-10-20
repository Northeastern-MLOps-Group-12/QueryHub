import os
import uvicorn
from backend.connectors_api import app  # adjust import to match your file1

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run provides this
    uvicorn.run(app, host="0.0.0.0", port=port)
