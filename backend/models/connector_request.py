from pydantic import BaseModel
from typing import Dict

class ConnectorRequest(BaseModel):
    """Pydantic model for incoming connector requests."""
    engine: str              # e.g., "postgres" or "mysql"
    provider: str            # e.g., "gcp" or "aws"
    config: Dict[str, str]   # dictionary with connection parameters
