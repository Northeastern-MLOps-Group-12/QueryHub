# connectors/main.py

from api.connectors_api import app  # make sure this import works

# No need for this anymore:
# if __name__ == "__main__":
#     import os
#     import uvicorn
#     port = int(os.environ.get("PORT", 8080))
#     uvicorn.run(app, host="0.0.0.0", port=port)
