from pydantic import BaseModel , Field
from typing import List,Dict

class AgentState(BaseModel):
    engine : str = Field(default="",description="The company DB engine.")
    creds : dict = Field(default={},description="The company DB credentials.")
    
    